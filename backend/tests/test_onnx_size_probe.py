"""ONNX Upscale Image pads each tile only as much as the model needs (upstream chaiNNer #3106)."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.ESRGAN import ESRGAN
from spandrel.architectures.RealCUGAN import UpCunet2x
from spandrel.architectures.SwinIR import SwinIR

from api import NodeContext, SettingsParser
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


def test_the_probe_runs_once_per_model(monkeypatch: pytest.MonkeyPatch):
    # Counts the CPU sessions the probe creates; the node's own session is separate.
    sessions: list[bytes] = []

    def session(model: bytes, *args: Any, **kwargs: Any) -> ort.InferenceSession:
        sessions.append(model)
        return ort.InferenceSession(model, *args, **kwargs)

    probe_ort = SimpleNamespace(
        SessionOptions=ort.SessionOptions, InferenceSession=session
    )
    monkeypatch.setattr(size_probe, "ort", probe_ort)
    model = stub(4)
    for h, w in [(37, 51), (20, 30), (64, 64)]:
        upscale(model, image(h, w))
    assert sessions == [model.bytes]
