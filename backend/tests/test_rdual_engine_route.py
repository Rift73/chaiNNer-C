"""Upscale Image's Use TensorRT Engine for RDUAL (nodes/impl/pytorch/rdual_engine.py)
and TensorRT's Upscale Image on an engine set, on the CPU: the set's build and session
are stand-ins, the PyTorch result is a small RDUAL's with random weights."""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from weakref import WeakKeyDictionary

import numpy as np
import pytest
import torch
from spandrel.architectures.RDUAL import RDUAL, RDUALArch

from api import NodeContext, NodeId
from nodes.impl.pytorch import rdual_engine
from nodes.impl.pytorch.rdual_window import RDUALWindows, release_memory
from nodes.impl.pytorch.utils import is_out_of_memory
from nodes.impl.tensorrt import rdual, rdual_session
from nodes.impl.tensorrt.model import TemporalInfo, TensorRTEngine, TensorRTEngineInfo
from nodes.impl.upscale.basic_upscale import PAD_SIZE, PaddingType
from nodes.impl.upscale.temporal_window import LOOK_AHEAD, WARM_UP, Fit
from nodes.impl.upscale.tile_report import TileReport
from nodes.impl.upscale.tiler import NoTiling

SIZE = (16, 20)
SET_INFO = TensorRTEngineInfo(
    precision="mixed",
    input_channels=3,
    output_channels=3,
    scale=2,
    gpu_architecture="sm_120",
    tensorrt_version="11.4.0",
    has_dynamic_shapes=False,
    min_shape=(1, 3, *SIZE),
    opt_shape=(1, 3, *SIZE),
    max_shape=(1, 3, *SIZE),
    temporal=TemporalInfo("RDUAL", *SIZE, "c64_h8_w10", ("prefix", "flow")),
)


@pytest.fixture(scope="module")
def descriptor() -> Any:
    torch.manual_seed(0)
    model = RDUAL(scale=2, use_checkpoint=False)
    model.get_buffer("flow_loaded").fill_(True)
    return RDUALArch().load(model.state_dict())


class Context:
    """The NodeContext members the route uses, outside a loop."""

    def __init__(self) -> None:
        self.iteration = None
        self.node_id = NodeId("node")
        # The TensorRT package's settings: no Timing Cache folder
        self.settings = SimpleNamespace(
            of_package=lambda _package: SimpleNamespace(
                get_cache_location=lambda _k: None
            )
        )
        self.warnings: list[str] = []
        self.cleanups: list[Any] = []

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def add_cleanup(self, cleanup: Any, after: str) -> None:
        assert after == "chain"
        self.cleanups.append(cleanup)


@pytest.fixture
def route(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, list[Any]]:
    """The machine allows a set; the set's build and session are recorded stand-ins."""
    calls: dict[str, list[Any]] = {"build": [], "run": [], "held": [], "released": []}
    monkeypatch.setattr(rdual_engine, "hold_engine", calls["held"].append)
    monkeypatch.setattr(rdual_engine, "release_engine", calls["released"].append)
    monkeypatch.setattr(rdual, "unavailable", lambda _model, _device: None)
    monkeypatch.setattr(rdual_engine, "_failed", WeakKeyDictionary())
    monkeypatch.setattr(rdual_engine, "cache_folder", lambda *_: tmp_path)

    def engine_set(_model: Any, height: int, width: int, *args: Any) -> TensorRTEngine:
        assert args[-1] is None  # the Timing Cache folder, off
        calls["build"].append((height, width))
        return TensorRTEngine(b"set", SET_INFO)

    def upscale(_c: Any, img: np.ndarray, engine: Any, *_: Any) -> np.ndarray:
        calls["run"].append(engine)
        return np.full((img.shape[0] * 2, img.shape[1] * 2, 3), 0.5, np.float32)

    monkeypatch.setattr(rdual, "engine_set", engine_set)
    monkeypatch.setattr(rdual_session, "temporal_engine_upscale", upscale)
    return calls


def run(
    descriptor: Any,
    context: Context,
    img: np.ndarray,
    padding: PaddingType = PaddingType.NONE,
) -> np.ndarray:
    fit = Fit(
        NoTiling(), shorten=True, out_of_memory=is_out_of_memory, release=release_memory
    )
    return rdual_engine.temporal_upscale_through_engine(
        cast(NodeContext, context),
        img,
        descriptor,
        torch.device("cpu"),
        lambda: rdual_engine.RDUALWindows(descriptor, torch.device("cpu")),
        0,
        TileReport(),
        padding,
        fit,
    )


def pytorch(descriptor: Any, img: np.ndarray) -> np.ndarray:
    windows = RDUALWindows(descriptor, torch.device("cpu"))
    outputs, _ = windows.run([windows.convert(img)], 1, None, 0)
    return windows.to_image(outputs[0])


@pytest.fixture
def frame() -> np.ndarray:
    return np.random.default_rng(0).random((*SIZE, 3), dtype=np.float32)


def test_the_set_runs_the_frame_and_the_node_holds_it(
    descriptor: Any,
    route: dict[str, list[Any]],
    frame: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
):
    def no_pytorch(*_: Any) -> None:
        raise AssertionError("the model went to PyTorch's executor")

    # The model stays where it is while the set runs.
    monkeypatch.setattr(rdual_engine, "RDUALWindows", no_pytorch)
    context = Context()
    out = run(descriptor, context, frame)

    assert np.all(out == 0.5)
    assert route["build"] == [SIZE]
    assert context.warnings == []
    assert route["held"] == [context.node_id] and route["released"] == []


def test_a_padded_frame_takes_a_set_of_its_padded_size(
    descriptor: Any, route: dict[str, list[Any]], frame: np.ndarray
):
    run(descriptor, Context(), frame, PaddingType.REFLECT_MIRROR)
    assert route["build"] == [(SIZE[0] + 2 * PAD_SIZE, SIZE[1] + 2 * PAD_SIZE)]


def test_a_failed_build_warns_once_and_the_chain_stays_on_pytorch(
    descriptor: Any,
    route: dict[str, list[Any]],
    frame: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
):
    builds: list[int] = []

    def fails(*_: Any) -> TensorRTEngine:
        builds.append(1)
        raise RuntimeError("no build here")

    monkeypatch.setattr(rdual, "engine_set", fails)
    context = Context()
    first = run(descriptor, context, frame)
    second = run(descriptor, context, frame)

    np.testing.assert_array_equal(first, pytorch(descriptor, frame))
    np.testing.assert_array_equal(second, first)
    assert builds == [1]
    assert context.warnings == [
        (
            "Upscale Image runs this RDUAL model on PyTorch, not through a TensorRT"
            " engine: RuntimeError: no build here"
        )
    ]
    # The next chain run tries again.
    for cleanup in context.cleanups:
        cleanup()
    run(descriptor, context, frame)
    assert builds == [1, 1]


def test_a_failed_frame_goes_on_with_pytorch(
    descriptor: Any,
    route: dict[str, list[Any]],
    frame: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
):
    def fails(*_: Any) -> np.ndarray:
        raise RuntimeError("the set failed")

    monkeypatch.setattr(rdual_session, "temporal_engine_upscale", fails)
    context = Context()
    out = run(descriptor, context, frame)

    np.testing.assert_array_equal(out, pytorch(descriptor, frame))
    assert context.warnings == [
        (
            "Upscale Image runs this RDUAL model on PyTorch, not through a TensorRT"
            " engine: RuntimeError: the set failed"
        )
    ]
    # The set's session is let go before PyTorch runs the frame.
    assert route["held"] == route["released"] == [context.node_id]


def test_a_machine_without_tensorrt_says_nothing(
    descriptor: Any,
    route: dict[str, list[Any]],
    frame: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(rdual, "unavailable", lambda _model, _device: "")
    context = Context()
    out = run(descriptor, context, frame)

    np.testing.assert_array_equal(out, pytorch(descriptor, frame))
    assert route["build"] == [] and context.warnings == []


def tensorrt_upscale() -> Any:
    nodes = importlib.import_module(
        "packages.chaiNNer_tensorrt.tensorrt.processing.upscale_image"
    )
    if not hasattr(nodes, "upscale_image_node"):
        pytest.skip("TensorRT's nodes are not registered here")
    return nodes


def test_tensorrt_upscale_image_runs_a_set_on_the_temporal_path(
    route: dict[str, list[Any]], frame: np.ndarray, monkeypatch: pytest.MonkeyPatch
):
    nodes = tensorrt_upscale()
    monkeypatch.setattr(nodes, "get_settings", lambda _: SimpleNamespace(gpu_index=0))
    engine = TensorRTEngine(b"set", SET_INFO)
    out, report = nodes.engine_upscale(
        cast(NodeContext, Context()), frame, engine, True, True
    )

    assert np.all(out == 0.5) and route["run"] == [engine]
    assert isinstance(report, str)


def test_tensorrt_upscale_image_reads_context_for_a_set_only():
    nodes = tensorrt_upscale()
    data = next(
        d
        for d in nodes.processing_group.nodes
        if d.schema_id == "chainner:tensorrt:upscale_image"
    )
    assert data.temporal_context == (WARM_UP, LOOK_AHEAD)
    gate, applies = data.temporal_if
    assert gate == 0
    image_engine = TensorRTEngine(
        b"engine", dataclasses.replace(SET_INFO, temporal=None)
    )
    assert applies(TensorRTEngine(b"set", SET_INFO))
    assert not applies(image_engine)
    assert applies(None)
