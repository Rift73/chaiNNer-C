"""ONNX Upscale Image pads each tile only as much as the model needs (upstream chaiNNer #3106)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast
from weakref import WeakKeyDictionary

import numpy as np
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
from nodes.impl.native_onnx_runtime import NativeSession
from nodes.impl.onnx import size_probe
from nodes.impl.onnx.auto_split import onnx_auto_split
from nodes.impl.onnx.load import load_onnx_model
from nodes.impl.onnx.model import OnnxGeneric, SizeReq
from nodes.impl.onnx.session import create_inference_session
from nodes.impl.pytorch.convert_to_onnx_impl import convert_to_onnx_impl
from nodes.impl.upscale.auto_split_tiles import NO_TILING
from nodes.impl.upscale.tiler import NoTiling
from packages.chaiNNer_onnx.onnx.processing.upscale_image import upscale_image_node


class Context:
    """The NodeContext members ONNX Upscale Image uses, on the CPU provider."""

    def __init__(self):
        self.settings = SettingsParser({"execution_provider": "CPUExecutionProvider"})
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


def upscale(model: OnnxGeneric, image: np.ndarray) -> np.ndarray:
    return upscale_image_node(
        cast(NodeContext, Context()), image, model, NO_TILING, 0, False
    )


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
    model = exported(
        lambda: SwinIR(
            img_size=32,
            window_size=8,
            embed_dim=12,
            depths=[1],
            num_heads=[2],
            mlp_ratio=2,
            upscale=2,
            upsampler="pixelshuffle",
        )
    )
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
    # RealCUGAN reflect-pads by 18 inside, so it needs an even size of at least 20.
    model = exported(lambda: UpCunet2x(in_channels=3, out_channels=3))
    assert size_probe.get_size_req(model) == SizeReq(minimum=32, multiple_of=2)
    for h, w in [(37, 51), (3, 3)]:
        assert upscale(model, image(h, w)).shape == (2 * h, 2 * w, 3)


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


@pytest.mark.parametrize("blocksize", [2, 8, 32])
def test_the_probe_finds_the_multiple(blocksize: int):
    assert size_probe.get_size_req(stub(blocksize)) == SizeReq(
        minimum=16, multiple_of=blocksize
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


def counted(monkeypatch: pytest.MonkeyPatch, name: str) -> list[object]:
    """The first argument of every call of size_probe's function `name`."""
    calls: list[object] = []
    original = getattr(size_probe, name)

    def record(first: Any, *args: Any) -> Any:
        calls.append(first)
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

    def probe(model: OnnxGeneric, session: object, fallback: SizeReq | None) -> SizeReq:
        probes.append(model)
        if model is slow:
            started.set()
            assert release.wait(30)
        return SizeReq(minimum=1, multiple_of=2 if model is slow else 4)

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
