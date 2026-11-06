"""ONNX Upscale Image pads each tile only as much as the model needs (upstream chaiNNer #3106)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from weakref import WeakKeyDictionary

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.ESRGAN import ESRGAN
from spandrel.architectures.RealCUGAN import UpCunet2x
from spandrel.architectures.SwinIR import SwinIR

# The node's modules load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import NodeContext, Progress, SettingsParser
from nodes.impl.migraphx.build_cache import program_key
from nodes.impl.native_onnx_runtime import NativeSession, ort_errors
from nodes.impl.onnx import size_probe
from nodes.impl.onnx.auto_split import onnx_auto_split
from nodes.impl.onnx.load import load_onnx_model
from nodes.impl.onnx.model import SIZE_REQ_METADATA_KEY, OnnxGeneric, SizeReq
from nodes.impl.onnx.session import create_inference_session
from nodes.impl.pytorch.convert_to_onnx_impl import convert_to_onnx_impl
from nodes.impl.tensorrt.build_cache import engine_key
from nodes.impl.upscale.auto_split_tiles import NO_TILING
from nodes.impl.upscale.tiler import NoTiling
from packages.chaiNNer_onnx.onnx.io import load_model as load_model_module
from packages.chaiNNer_onnx.onnx.processing import upscale_image as upscale_image_module
from packages.chaiNNer_onnx.onnx.processing.upscale_image import upscale_image_node


class Context:
    """The NodeContext members ONNX Upscale Image uses, on the CPU provider."""

    def __init__(self, storage_dir: Path | None = None):
        self.settings = SettingsParser({"execution_provider": "CPUExecutionProvider"})
        self.storage_dir = storage_dir
        progress = Progress.noop_progress()
        self.check_aborted = progress.check_aborted
        self.set_progress = progress.set_progress

    def add_cleanup(self, fn: Callable[[], None], after: str = "chain"):
        pass


def exported(make: Callable[[], torch.nn.Module]) -> OnnxGeneric:
    """A randomly initialised spandrel model through Convert To ONNX and Load Model."""
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(make().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    for _, v in model.model.named_parameters():
        v.requires_grad = False
    loaded = load_onnx_model(convert_to_onnx_impl(model.eval(), torch.device("cpu")))
    assert isinstance(loaded, OnnxGeneric)
    return loaded


def stub(blocksize: int, fixed_output: bool = False) -> OnnxGeneric:
    """SpaceToDepth then DepthToSpace: the identity, which runs only at multiples of
    `blocksize` and whose shapes infer at 16x16, like a DRCT export. With
    `fixed_output`, a Reshape to 1x3x16x16 that runs at 16x16 only."""
    image = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])
    if fixed_output:
        output = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 3, 16, 16]
        )
        shape = numpy_helper.from_array(np.array([1, 3, 16, 16], np.int64), "shape")
        nodes = [helper.make_node("Reshape", ["input", "shape"], ["output"])]
        initializers = [shape]
    else:
        output = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 3, "h", "w"]
        )
        nodes = [
            helper.make_node("SpaceToDepth", ["input"], ["s"], blocksize=blocksize),
            helper.make_node("DepthToSpace", ["s"], ["output"], blocksize=blocksize),
        ]
        initializers = []
    graph = helper.make_graph(nodes, "stub", [image], [output], initializers)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    )
    loaded = load_onnx_model(model.SerializeToString())
    assert isinstance(loaded, OnnxGeneric)
    return loaded


def swinir() -> SwinIR:
    return SwinIR(
        img_size=32,
        window_size=8,
        embed_dim=12,
        depths=[1],
        num_heads=[2],
        mlp_ratio=2,
        upscale=2,
        upsampler="pixelshuffle",
    )


def upscale(
    model: OnnxGeneric, image: np.ndarray, storage_dir: Path | None = None
) -> np.ndarray:
    image, _ = upscale_image_node(
        cast(NodeContext, Context(storage_dir)), image, model, NO_TILING, 0, False
    )
    return image


def unpadded(model: OnnxGeneric, image: np.ndarray) -> np.ndarray:
    """ORT on the image as is (RGB, NCHW), clipped like the node's output."""
    session = ort.InferenceSession(model.bytes, providers=["CPUExecutionProvider"])
    rgb = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None])
    (output,) = session.run(None, {"input": rgb})
    return np.clip(np.asarray(output)[0].transpose(1, 2, 0)[:, :, ::-1], 0, 1)


def image(h: int, w: int) -> np.ndarray:
    return np.random.default_rng(0).random((h, w, 3), dtype=np.float32)


def test_a_convolutional_model_is_not_padded():
    model = exported(lambda: ESRGAN(num_filters=8, num_blocks=1, scale=4))
    assert model.info.size_req == SizeReq(multiple_of=16)  # shape inference
    assert size_probe.get_size_req(model) == SizeReq(minimum=16, multiple_of=1)
    img = image(37, 51)
    np.testing.assert_array_equal(upscale(model, img), unpadded(model, img))


def test_a_window_8_model_runs_at_any_size():
    # Spandrel 0.4.2+c2 pads SwinIR per input inside the exported graph, so the probe
    # finds no multiple; a graph that does need 8 (below) is found to need 8.
    model = exported(swinir)
    assert model.info.scale_width is None  # its shapes do not infer
    found = size_probe.get_size_req(model)
    assert found is not None and found.multiple_of == 1
    for h, w in [(37, 51), (3, 3)]:
        assert upscale(model, image(h, w)).shape == (2 * h, 2 * w, 3)
    needs_8 = stub(8)
    found = size_probe.get_size_req(needs_8)
    assert found is not None and found.multiple_of == 8
    assert upscale(needs_8, image(37, 51)).shape == (37, 51, 3)


def test_a_model_that_needs_20px_gets_a_minimum_that_runs():
    # RealCUGAN reflect-pads by 18 inside, so it needs a size of at least 20. Without
    # the minimum Convert To ONNX records (32), as from another exporter, the probe
    # finds the 32 that runs.
    export = exported(lambda: UpCunet2x(in_channels=3, out_channels=3))
    proto = onnx.load_from_string(export.bytes)
    del proto.metadata_props[:]
    model = load_onnx_model(proto.SerializeToString())
    assert isinstance(model, OnnxGeneric)
    assert size_probe.get_size_req(model) == SizeReq(minimum=32, multiple_of=1)
    for h, w in [(37, 51), (3, 3)]:
        assert upscale(model, image(h, w)).shape == (2 * h, 2 * w, 3)


@pytest.mark.usefixtures("fresh_cache")
@pytest.mark.parametrize(
    "entry",
    [
        '{"minimum": 32, "multiple_of"',
        '{"minimum": 32}',
        "[32, 1]",
        '{"minimum": 32, "multiple_of": 0}',
    ],
)
def test_a_malformed_recorded_requirement_is_logged_and_the_probe_applies(
    entry: str, caplog: pytest.LogCaptureFixture
):
    # An entry edited, or written by another tool: Load Model still loads the model,
    # whose requirement is then the probe's alone.
    proto = onnx.load_from_string(stub(8).bytes)
    recorded = proto.metadata_props.add()
    recorded.key = SIZE_REQ_METADATA_KEY
    recorded.value = entry
    with caplog.at_level(logging.WARNING):
        model = load_onnx_model(proto.SerializeToString())
    assert isinstance(model, OnnxGeneric)
    assert size_probe.get_size_req(model) == SizeReq(minimum=16, multiple_of=8)
    (warning,) = [
        r.message for r in caplog.records if SIZE_REQ_METADATA_KEY in r.message
    ]
    assert "'stub'" in warning and repr(entry) in warning


def test_a_model_that_needs_16_is_padded_as_before():
    model = stub(16)
    assert model.info.size_req == SizeReq(multiple_of=16)
    assert size_probe.get_size_req(model) == model.info.size_req
    img = image(37, 51)
    before = onnx_auto_split(
        img,
        create_inference_session(model, 0, "CPUExecutionProvider"),
        change_shape=False,
        tiler=NoTiling(),
        size_req=model.info.size_req,
    )
    np.testing.assert_array_equal(upscale(model, img), np.clip(before, 0, 1))


@pytest.mark.parametrize("blocksize", [2, 8, 12, 24, 32, 48, 64])
def test_the_probe_finds_the_multiple(blocksize: int):
    # 12, 24 and 48: windows of 6, 12 or 24 after a pixel unshuffle of 2, say
    assert size_probe.get_size_req(stub(blocksize)) == SizeReq(
        minimum=16, multiple_of=blocksize
    )


class Recorded:
    """A model's CPU session that records the input sizes it runs at."""

    def __init__(self, model: OnnxGeneric):
        self.session = ort.InferenceSession(
            model.bytes, providers=["CPUExecutionProvider"]
        )
        self.sizes: list[tuple[int, int]] = []

    def get_inputs(self) -> Any:
        return self.session.get_inputs()

    def get_outputs(self) -> Any:
        return self.session.get_outputs()

    def get_providers(self) -> list[str]:
        return list(self.session.get_providers())

    def run(self, names: Any, feed: dict[str, np.ndarray], options: Any = None) -> Any:
        (image,) = feed.values()
        height, width = image.shape[-2:]
        self.sizes.append((height, width))
        return self.session.run(names, feed, options)


@pytest.mark.usefixtures("fresh_cache")
def test_the_probe_runs_no_size_below_16():
    # The 12 stub's shapes give no scale, so shape inference gives no minimum; a
    # converted OmniSR ends the process below 12px (ORT divides by zero).
    model = stub(12)
    assert model.info.scale_width is None
    session = Recorded(model)
    assert size_probe.get_size_req(model, cast(Any, session)) == SizeReq(16, 12)
    assert min(min(size) for size in session.sizes) >= 16


def test_a_model_shape_inference_gives_no_minimum_is_padded_to_16():
    # A 3x3 convolution reshaped to its own shape: shape inference cannot give the
    # output's size, so neither a scale nor a minimum.
    weight = numpy_helper.from_array(
        np.random.default_rng(0).standard_normal((3, 3, 3, 3)).astype(np.float32),
        "weight",
    )
    dims: list[str | int] = [1, 3, "h", "w"]
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["input", "weight"], ["conv"], pads=[1, 1, 1, 1]),
            helper.make_node("Shape", ["conv"], ["shape"]),
            helper.make_node("Reshape", ["conv", "shape"], ["output"]),
        ],
        "conv",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, dims)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, dims)],
        [weight],
    )
    model = load_onnx_model(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
        ).SerializeToString()
    )
    assert isinstance(model, OnnxGeneric) and model.info.scale_width is None
    img = image(5, 7)
    padded = onnx_auto_split(
        img,
        create_inference_session(model, 0, "CPUExecutionProvider"),
        change_shape=False,
        tiler=NoTiling(),
        size_req=SizeReq(minimum=16),
    )
    np.testing.assert_array_equal(upscale(model, img), np.clip(padded, 0, 1))


def test_a_converted_model_below_its_minimum_is_padded_as_in_pytorch():
    # Spandrel pads SwinIR to 16px, whose export's shapes do not infer.
    torch.manual_seed(0)
    descriptor = ModelLoader().load_from_state_dict(swinir().state_dict())
    assert isinstance(descriptor, ImageModelDescriptor)
    assert descriptor.size_requirements.minimum == 16
    descriptor.model.requires_grad_(False)
    descriptor.eval()
    model = load_onnx_model(convert_to_onnx_impl(descriptor, torch.device("cpu")))
    assert isinstance(model, OnnxGeneric)
    for h, w in [(5, 7), (12, 12)]:
        img = image(h, w)
        with torch.no_grad():
            rgb = torch.from_numpy(img[:, :, ::-1].transpose(2, 0, 1).copy())[None]
            expected = descriptor(rgb).numpy()[0].transpose(1, 2, 0)[:, :, ::-1]
        np.testing.assert_allclose(
            upscale(model, img), np.clip(expected, 0, 1), rtol=0, atol=1e-5
        )


def test_a_model_that_does_not_run_at_the_probe_size_keeps_shape_inference():
    model = stub(1, fixed_output=True)
    assert model.info.size_req == SizeReq(multiple_of=16)
    assert size_probe.get_size_req(model) is model.info.size_req


@pytest.fixture
def fresh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe results of earlier tests (kept by the model's bytes) forgotten."""
    monkeypatch.setattr(size_probe, "_cache", WeakKeyDictionary())
    monkeypatch.setattr(size_probe, "_by_content", {})


def counted(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    before: Callable[[], None] | None = None,
) -> list[object]:
    """The first argument of every call of size_probe's function `name`, which `before`
    runs ahead of."""
    calls: list[object] = []
    original = getattr(size_probe, name)

    def record(first: Any, *args: Any) -> Any:
        calls.append(first)
        if before is not None:
            before()
        return original(first, *args)

    monkeypatch.setattr(size_probe, name, record)
    return calls


@pytest.mark.usefixtures("fresh_cache")
def test_the_probe_runs_once_per_model_on_the_nodes_cpu_session(
    monkeypatch: pytest.MonkeyPatch,
):
    # The node's CPU session runs the probe: no session is made for it.
    measured = counted(monkeypatch, "_measure")
    made = counted(monkeypatch, "_cpu_session")
    model = stub(4)
    for h, w in [(37, 51), (20, 30), (64, 64)]:
        upscale(model, image(h, w))
    assert len(measured) == 1 and made == []
    assert size_probe.get_size_req(model) == SizeReq(minimum=16, multiple_of=4)


@pytest.mark.usefixtures("fresh_cache")
def test_a_model_loaded_again_is_not_probed_again(monkeypatch: pytest.MonkeyPatch):
    measured = counted(monkeypatch, "_measure")
    first, again = stub(8), load_onnx_model(stub(8).bytes)
    assert isinstance(again, OnnxGeneric) and again is not first
    assert size_probe.get_size_req(first) == size_probe.get_size_req(again)
    assert len(measured) == 1


@pytest.mark.usefixtures("fresh_cache")
def test_a_fallback_is_not_kept_for_the_model_loaded_again(
    monkeypatch: pytest.MonkeyPatch,
):
    # No CPU session for the first probe (no memory for it, say): shape inference's
    # requirement holds for that model, while the model loaded again is measured.
    def no_session(model: OnnxGeneric) -> Any:
        raise MemoryError("no memory for a session")

    first, again = stub(8), load_onnx_model(stub(8).bytes)
    assert isinstance(again, OnnxGeneric)
    measured = SizeReq(minimum=16, multiple_of=8)
    with monkeypatch.context() as patch:
        patch.setattr(size_probe, "_cpu_session", no_session)
        assert size_probe.get_size_req(first) == first.info.size_req != measured
    assert size_probe.get_size_req(again) == measured
    assert size_probe.get_size_req(first) == first.info.size_req


@pytest.mark.parametrize(
    "error",
    [
        MemoryError(),
        ort_errors.RuntimeException(
            "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Failed to allocate memory for"
            " requested buffer of size 1048576"
        ),
    ],
    ids=["python", "onnxruntime"],
)
@pytest.mark.usefixtures("fresh_cache")
def test_a_probe_short_of_memory_is_not_kept_for_the_model_loaded_again(
    error: Exception,
):
    # Every run fails for lack of memory, which says nothing of the model: shape
    # inference's requirement holds for that model, and the model loaded again is
    # measured.
    class ShortOfMemory(Recorded):
        def run(self, names: Any, feed: Any, options: Any = None) -> Any:
            raise error

    first, again = stub(8), load_onnx_model(stub(8).bytes)
    assert isinstance(again, OnnxGeneric)
    session = cast(Any, ShortOfMemory(first))
    assert size_probe.get_size_req(first, session) is first.info.size_req
    assert size_probe.get_size_req(again) == SizeReq(minimum=16, multiple_of=8)


class FailsOnce:
    """A session whose first run fails with `message`, as a run does that lacks memory
    for a moment; later runs are the real session's."""

    def __init__(self, session: Any, message: str):
        self.session, self.message, self.failed = session, message, False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.session, name)

    def run(self, *args: Any) -> Any:
        if self.failed:
            return self.session.run(*args)
        self.failed = True
        raise ort_errors.Fail(self.message)


@pytest.mark.parametrize(
    "message",
    [
        "[ONNXRuntimeError] : 1 : FAIL : Native ONNX allocation failed",
        (
            "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned"
            " while running Conv node. Name:'c' Status Message: bad allocation"
        ),
        (
            "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned"
            " while running Conv node. Name:'c' Status Message: std::bad_alloc"
        ),
    ],
    ids=["the native adapter", "a kernel on Windows", "a kernel on Linux"],
)
@pytest.mark.usefixtures("fresh_cache")
def test_a_failure_of_memory_that_does_not_last_keeps_nothing_on_disk(
    message: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # The first multiple fails for lack of memory and the next one runs: the file would
    # hold a multiple of 2 for a model that takes every size.
    model = stub(1)
    with monkeypatch.context() as patch:
        patch.setattr(
            size_probe,
            "_cpu_session",
            lambda m: FailsOnce(NativeSession(m.bytes), message),
        )
        assert size_probe.get_size_req(model, None, tmp_path).multiple_of == 2
    assert not (tmp_path / size_probe.DISK_FOLDER).exists()
    forget_memory(monkeypatch)
    again = size_probe.get_size_req(loaded_again(model), None, tmp_path)
    assert again.multiple_of == 1


@pytest.mark.usefixtures("fresh_cache")
def test_a_model_without_any_requirement_is_padded_to_16(
    monkeypatch: pytest.MonkeyPatch,
):
    # No CPU session for the probe, and shape inference gives no scale: tiles are still
    # padded to 16px (a converted OmniSR ends the process at 8px).
    def no_session(model: OnnxGeneric) -> Any:
        raise MemoryError("no memory for a session")

    monkeypatch.setattr(size_probe, "_cpu_session", no_session)
    model = stub(12)
    assert model.info.scale_width is None
    assert size_probe.get_size_req(model) == SizeReq(minimum=16)


@pytest.mark.usefixtures("fresh_cache")
def test_a_model_that_does_not_run_is_not_probed_again(
    monkeypatch: pytest.MonkeyPatch,
):
    measured = counted(monkeypatch, "_measure")
    first = stub(1, fixed_output=True)
    again = load_onnx_model(first.bytes)
    assert isinstance(again, OnnxGeneric)
    assert size_probe.get_size_req(first) is first.info.size_req
    assert size_probe.get_size_req(again) is again.info.size_req
    assert len(measured) == 1


@pytest.mark.usefixtures("fresh_cache")
def test_a_model_onnx_runtime_refuses_is_not_probed_again(
    monkeypatch: pytest.MonkeyPatch,
):
    # A CPU session refused for the model itself (an op ONNX Runtime does not know,
    # as in a TensorRT plugin model) holds for the model loaded again.
    sessions: list[OnnxGeneric] = []

    def refused(model: OnnxGeneric) -> Any:
        sessions.append(model)
        raise ort_errors.InvalidGraph("No Op registered for Plugin_TRT")

    monkeypatch.setattr(size_probe, "_cpu_session", refused)
    first, again = stub(8), load_onnx_model(stub(8).bytes)
    assert isinstance(again, OnnxGeneric)
    assert size_probe.get_size_req(first) is first.info.size_req
    assert size_probe.get_size_req(again) is again.info.size_req
    assert sessions == [first]


@pytest.mark.usefixtures("fresh_cache")
def test_a_gpu_session_is_not_used_by_the_probe(monkeypatch: pytest.MonkeyPatch):
    class GpuSession:
        def get_providers(self) -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    made = counted(monkeypatch, "_cpu_session")
    model = stub(2)
    gpu = cast(Any, GpuSession())
    assert size_probe.get_size_req(model, gpu) == SizeReq(minimum=16, multiple_of=2)
    assert made == [model]


@pytest.mark.usefixtures("fresh_cache")
def test_models_probe_concurrently_and_each_once(monkeypatch: pytest.MonkeyPatch):
    # A slow probe of one model holds only that model: another model's probe
    # finishes meanwhile, and callers of the slow model wait for its one probe.
    slow, fast = stub(2), stub(4)
    release, started = threading.Event(), threading.Event()
    probes: list[OnnxGeneric] = []

    def probe(
        model: OnnxGeneric, session: object, fallback: SizeReq | None
    ) -> tuple[SizeReq, bool]:
        probes.append(model)
        if model is slow:
            started.set()
            assert release.wait(30)
        return SizeReq(minimum=1, multiple_of=2 if model is slow else 4), True

    monkeypatch.setattr(size_probe, "_probe", probe)
    with ThreadPoolExecutor(4) as pool:
        waiting = [pool.submit(size_probe.get_size_req, slow) for _ in range(3)]
        assert started.wait(30)
        assert size_probe.get_size_req(fast) == SizeReq(minimum=1, multiple_of=4)
        assert not any(future.done() for future in waiting)
        release.set()
        results = [future.result(30) for future in waiting]
    assert results == [SizeReq(minimum=1, multiple_of=2)] * 3
    assert probes.count(slow) == 1 and probes.count(fast) == 1


@pytest.mark.parametrize("model", ["2", "8", "32", "fixed", "realcugan"])
def test_native_and_python_sessions_measure_alike(
    monkeypatch: pytest.MonkeyPatch, model: str
):
    # The native adapter (which releases the GIL) and ORT's Python binding, each as
    # the node's CPU session, find the same requirement.
    if model == "realcugan":
        loaded = exported(lambda: UpCunet2x(in_channels=3, out_channels=3))
    elif model == "fixed":
        loaded = stub(1, fixed_output=True)
    else:
        loaded = stub(int(model))
    found = []
    native = NativeSession(loaded.bytes)
    try:
        for session in (
            ort.InferenceSession(loaded.bytes, providers=["CPUExecutionProvider"]),
            native,
        ):
            monkeypatch.setattr(size_probe, "_cache", WeakKeyDictionary())
            monkeypatch.setattr(size_probe, "_by_content", {})
            found.append(size_probe.get_size_req(loaded, cast(Any, session)))
    finally:
        native.close()
    assert found[0] == found[1]


@pytest.mark.usefixtures("fresh_cache")
def test_rejected_sizes_are_not_logged_by_onnx_runtime(capfd: pytest.CaptureFixture):
    model = stub(8)
    assert size_probe.get_size_req(model) == SizeReq(minimum=16, multiple_of=8)
    # ONNX Runtime writes its log in UTF-16.
    assert "onnxruntime" not in capfd.readouterr().err.replace("\0", "")


def kept_files(storage_dir: Path) -> list[Path]:
    return sorted((storage_dir / size_probe.DISK_FOLDER).iterdir())


def forget_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """What a new start of the backend has: nothing of the probe in memory."""
    monkeypatch.setattr(size_probe, "_cache", WeakKeyDictionary())
    monkeypatch.setattr(size_probe, "_by_content", {})


def loaded_again(model: OnnxGeneric) -> OnnxGeneric:
    again = load_onnx_model(model.bytes)
    assert isinstance(again, OnnxGeneric) and again is not model
    return again


def join_background_probes() -> None:
    for thread in threading.enumerate():
        if thread.name == "onnx-size-probe":
            thread.join(60)


def checksummed(key: str, values: object) -> bytes:
    """A file of `_encode`'s layout whose checksum holds."""
    body = json.dumps({"key": key, "size_req": values})
    digest = hashlib.sha256(body.encode()).hexdigest()
    return json.dumps({"body": body, "sha256": digest}).encode()


@pytest.mark.usefixtures("fresh_cache")
def test_a_new_start_reads_the_requirement_and_does_not_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    model = stub(8)
    found = size_probe.get_size_req(model, None, tmp_path)
    assert found == SizeReq(minimum=16, multiple_of=8)
    assert len(kept_files(tmp_path)) == 1
    forget_memory(monkeypatch)
    measured = counted(monkeypatch, "_measure")
    made = counted(monkeypatch, "_cpu_session")
    assert size_probe.get_size_req(loaded_again(model), None, tmp_path) == found
    assert measured == [] and made == []


@pytest.mark.usefixtures("fresh_cache")
def test_a_model_that_does_not_run_is_measured_again_at_a_new_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # A failure may not last, and on disk it would hold the fallback's padding for good.
    def rejects(session: Any, fallback: SizeReq | None) -> tuple[None, bool]:
        return None, True

    model = stub(8)
    with monkeypatch.context() as patch:
        patch.setattr(size_probe, "_measure", rejects)
        found = size_probe.get_size_req(model, None, tmp_path)
        assert found == model.info.size_req
    assert not (tmp_path / size_probe.DISK_FOLDER).exists()
    forget_memory(monkeypatch)
    found = size_probe.get_size_req(loaded_again(model), None, tmp_path)
    assert found == SizeReq(minimum=16, multiple_of=8)


@pytest.mark.usefixtures("fresh_cache")
def test_a_measurement_that_does_not_hold_for_the_bytes_is_not_kept_on_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def no_session(model: OnnxGeneric) -> Any:
        raise MemoryError("no memory for a session")

    monkeypatch.setattr(size_probe, "_cpu_session", no_session)
    size_probe.get_size_req(stub(8), None, tmp_path)
    assert not (tmp_path / size_probe.DISK_FOLDER).exists()


# The damaged files of `key`, a checksum that holds included
DAMAGED: dict[str, Callable[[str, bytes], bytes]] = {
    "empty": lambda _, good: b"",
    "cut short": lambda _, good: good[: len(good) // 2],
    "not text": lambda _, good: b"\xff\xfe" + good,
    "not a record": lambda _, good: b"[1, 2, 3]",
    "no body": lambda _, good: b'{"sha256": "00"}',
    "body not text": lambda _, good: b'{"body": 7, "sha256": "00"}',
    "a changed digit": lambda _, good: good.replace(b"16", b"32"),
    "another key": lambda key, good: checksummed("0" * len(key), [32, 32]),
    "no requirement field": lambda key, good: checksummed(key, "many"),
    "no values": lambda key, good: checksummed(key, None),
    "too few values": lambda key, good: checksummed(key, [32]),
    "a zero multiple": lambda key, good: checksummed(key, [32, 0]),
    "values not numbers": lambda key, good: checksummed(key, ["32", "1"]),
}


@pytest.mark.usefixtures("fresh_cache")
@pytest.mark.parametrize("damage", DAMAGED)
def test_a_damaged_file_is_probed_again_and_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, damage: str
):
    model = stub(4)
    img = image(37, 51)
    want = upscale(model, img, tmp_path / "clean")
    forget_memory(monkeypatch)
    size_probe.get_size_req(loaded_again(model), None, tmp_path)
    (path,) = kept_files(tmp_path)
    good = path.read_bytes()
    path.write_bytes(DAMAGED[damage](path.name.split(".")[0], good))

    forget_memory(monkeypatch)
    measured = counted(monkeypatch, "_measure")
    again = loaded_again(model)
    assert np.array_equal(upscale(again, img, tmp_path), want)
    assert len(measured) == 1
    # The damaged file is gone: the probe's own result replaced it.
    assert [p.read_bytes() for p in kept_files(tmp_path)] == [good]


@pytest.mark.usefixtures("fresh_cache")
def test_a_file_of_another_model_under_this_models_name_is_not_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    needs_8, needs_4 = stub(8), stub(4)
    size_probe.get_size_req(needs_8, None, tmp_path)
    (path,) = kept_files(tmp_path)
    good_of_8 = path.read_bytes()
    size_probe.get_size_req(needs_4, None, tmp_path)
    (of_4,) = (p for p in kept_files(tmp_path) if p != path)
    of_4.write_bytes(good_of_8)
    forget_memory(monkeypatch)
    found = size_probe.get_size_req(loaded_again(needs_4), None, tmp_path)
    assert found == SizeReq(minimum=16, multiple_of=4)


@pytest.mark.usefixtures("fresh_cache")
@pytest.mark.parametrize("stale", ["ONNX Runtime", "the probe's code", "fallback"])
def test_a_stale_file_is_not_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stale: str
):
    # A requirement that would change the output, kept under what is now out of date
    wrong = SizeReq(minimum=64, multiple_of=64)

    def measures_wrong(session: Any, fallback: SizeReq | None) -> tuple[SizeReq, bool]:
        return wrong, True

    model = stub(4)
    with monkeypatch.context() as patch:
        patch.setattr(size_probe, "_measure", measures_wrong)
        if stale == "ONNX Runtime":
            # size_probe's name only: Linux's native adapter finds its library by the real one.
            patch.setattr(
                size_probe,
                "ort",
                SimpleNamespace(**{**vars(size_probe.ort), "__version__": "0.0.0"}),
            )
        elif stale == "the probe's code":
            patch.setattr(size_probe, "_code_identity", lambda: "other code")
        else:
            patch.setattr(model.info, "size_req", SizeReq(32, 32))
        assert size_probe.get_size_req(model, None, tmp_path) == wrong
    assert len(kept_files(tmp_path)) == 1
    forget_memory(monkeypatch)
    measured = counted(monkeypatch, "_measure")
    found = size_probe.get_size_req(loaded_again(model), None, tmp_path)
    assert found == SizeReq(minimum=16, multiple_of=4)
    assert len(measured) == 1


@pytest.mark.usefixtures("fresh_cache")
def test_the_least_recently_used_files_are_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(size_probe, "DISK_ENTRIES", 2)
    first, second, third = stub(2), stub(4), stub(8)
    size_probe.get_size_req(first, None, tmp_path)
    (first_file,) = kept_files(tmp_path)
    os.utime(first_file, (1, 1))
    size_probe.get_size_req(second, None, tmp_path)
    (second_file,) = (p for p in kept_files(tmp_path) if p != first_file)
    os.utime(second_file, (2, 2))
    # Reading the first marks it used, so the second is the one removed.
    forget_memory(monkeypatch)
    size_probe.get_size_req(loaded_again(first), None, tmp_path)
    size_probe.get_size_req(third, None, tmp_path)
    files = kept_files(tmp_path)
    assert len(files) == 2 and first_file in files and second_file not in files


@pytest.mark.usefixtures("fresh_cache")
def test_the_removal_leaves_another_probes_temporary_file_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # That probe renames it next: removing it would lose its requirement.
    monkeypatch.setattr(size_probe, "DISK_ENTRIES", 1)
    folder = tmp_path / size_probe.DISK_FOLDER
    folder.mkdir()
    other = folder / "another-key.abc.tmp"
    other.write_bytes(b"")
    os.utime(other, (1, 1))
    size_probe.get_size_req(stub(2), None, tmp_path)
    assert other.exists()


@pytest.mark.usefixtures("fresh_cache")
def test_a_file_that_vanishes_during_the_removal_is_no_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    size_probe.get_size_req(stub(2), None, tmp_path)
    (vanishing,) = kept_files(tmp_path)
    size_probe.get_size_req(stub(4), None, tmp_path)
    (oldest,) = (p for p in kept_files(tmp_path) if p != vanishing)
    os.utime(oldest, (1, 1))
    real_stat = Path.stat

    def stat(path: Path, **kwargs: Any) -> os.stat_result:
        if path == vanishing:
            raise FileNotFoundError(path)
        return real_stat(path, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(size_probe, "DISK_ENTRIES", 1)
    with caplog.at_level(logging.WARNING, logger="sanic.root"):
        size_probe.get_size_req(stub(8), None, tmp_path)
    assert "cannot keep" not in caplog.text
    assert not oldest.exists()


@pytest.mark.usefixtures("fresh_cache")
def test_a_storage_that_cannot_be_used_is_logged_and_the_probe_holds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    # A file where the folder would be: nothing can be read or written there.
    blocked = tmp_path / "storage"
    blocked.mkdir()
    (blocked / size_probe.DISK_FOLDER).write_bytes(b"")
    with caplog.at_level(logging.WARNING, logger="sanic.root"):
        found = size_probe.get_size_req(stub(8), None, blocked)
    assert found == SizeReq(minimum=16, multiple_of=8)
    assert "cannot keep" in caplog.text


@pytest.mark.usefixtures("fresh_cache")
def test_the_node_waits_for_the_background_probe_and_does_not_probe_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    class GpuSession:
        def get_providers(self) -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    entered, release = threading.Event(), threading.Event()

    def hold() -> None:
        entered.set()
        assert release.wait(60)

    calls = counted(monkeypatch, "_measure", before=hold)
    made = counted(monkeypatch, "_cpu_session")
    model = stub(8)
    size_probe.probe_in_background(model, tmp_path)
    size_probe.probe_in_background(model, tmp_path)
    assert entered.wait(60)
    with ThreadPoolExecutor(1) as pool:
        waiting = pool.submit(
            size_probe.get_size_req, model, cast(Any, GpuSession()), tmp_path
        )
        assert not waiting.done()
        release.set()
        assert waiting.result(60) == SizeReq(minimum=16, multiple_of=8)
    assert len(calls) == 1 and made == [model]
    assert len(kept_files(tmp_path)) == 1


@pytest.mark.usefixtures("fresh_cache")
def test_a_background_probe_that_does_not_hold_leaves_the_probe_to_the_node(
    monkeypatch: pytest.MonkeyPatch,
):
    def no_session(model: OnnxGeneric) -> Any:
        raise MemoryError("no memory for a session")

    model = stub(8)
    with monkeypatch.context() as patch:
        patch.setattr(size_probe, "_cpu_session", no_session)
        size_probe.probe_in_background(model, None)
        join_background_probes()
    assert size_probe.get_size_req(model) == SizeReq(minimum=16, multiple_of=8)


@pytest.mark.usefixtures("fresh_cache")
def test_a_background_probe_that_raises_is_logged_and_the_node_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    def broken(session: Any, fallback: SizeReq | None) -> Any:
        raise RuntimeError("broken probe")

    monkeypatch.setattr(size_probe, "_measure", broken)
    model = stub(8)
    with caplog.at_level(logging.WARNING, logger="sanic.root"):
        size_probe.probe_in_background(model, None)
        join_background_probes()
    assert "broken probe" in caplog.text
    with pytest.raises(RuntimeError, match="broken probe"):
        size_probe.get_size_req(model)


@pytest.mark.usefixtures("fresh_cache")
def test_a_model_of_a_fixed_input_size_is_not_probed_in_the_background(
    monkeypatch: pytest.MonkeyPatch,
):
    started = counted(monkeypatch, "_background_probe")
    model = stub(8)
    model.info.fixed_input_width = 64
    size_probe.probe_in_background(model, None)
    assert started == []


def node_context(provider: str, storage_dir: Path) -> NodeContext:
    context = Context(storage_dir)
    context.settings = SettingsParser({"execution_provider": provider})
    return cast(NodeContext, context)


@pytest.mark.parametrize(
    ("provider", "started"),
    [("CUDAExecutionProvider", True), ("CPUExecutionProvider", False)],
)
def test_load_model_starts_the_probe_on_a_gpu_provider_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: str, started: bool
):
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        load_model_module,
        "probe_in_background",
        lambda model, storage: calls.append((model, storage)),
    )
    path = tmp_path / "model.onnx"
    path.write_bytes(stub(8).bytes)
    (model, _, name) = load_model_module.load_model_and_probe_node(
        node_context(provider, tmp_path), path
    )
    assert name == "model"
    assert calls == ([(model, tmp_path)] if started else [])


def test_upscale_image_starts_the_probe_on_a_gpu_provider_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        upscale_image_module,
        "probe_in_background",
        lambda model, storage: calls.append((model, storage)),
    )

    class Stop(Exception):
        pass

    def no_session(*args: Any) -> Any:
        raise Stop

    monkeypatch.setattr(upscale_image_module, "get_onnx_session", no_session)
    model = stub(8)
    for provider in ("CPUExecutionProvider", "CUDAExecutionProvider"):
        with pytest.raises(Stop):
            upscale_image_node(
                node_context(provider, tmp_path),
                image(32, 32),
                model,
                NO_TILING,
                0,
                False,
            )
    assert calls == [(model, tmp_path)]


@pytest.mark.usefixtures("fresh_cache")
def test_the_bytes_of_a_model_are_hashed_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    model = stub(8)
    hashed: list[object] = []
    original = hashlib.sha256

    def counting(data: bytes = b"") -> Any:
        if data == model.bytes:
            hashed.append(data)
        return original(data)

    monkeypatch.setattr(hashlib, "sha256", counting)
    size_probe.get_size_req(model, None, tmp_path)
    assert engine_key(model, (), "gpu", (1, 0), "11", None)
    assert program_key(model, (), "gpu", "gfx", "runner", None)
    assert len(hashed) == 1
