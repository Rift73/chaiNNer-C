"""NCNN has one float type, so Convert To NCNN takes out a Cast between FP16 and FP32: an FP16
ONNX that takes its image in FP32 has them around its first convolution. A Cast from or to
any other type stays refused."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from ncnn import ncnn
from onnx import TensorProto, helper, numpy_helper
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.Compact import Compact

# The converter is native, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from nodes.impl.ncnn.model import NcnnModel
from nodes.impl.onnx.load import load_onnx_model, with_recorded_arch
from nodes.impl.pytorch.convert_to_onnx_impl import convert_to_onnx_impl
from packages.chaiNNer_onnx.onnx.utility.convert_to_ncnn import convert_to_ncnn_node

C = 4


def run_ncnn(model: NcnnModel, image: np.ndarray, directory: Path) -> np.ndarray:
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    model.write_bin(directory / "model.bin")
    assert net.load_param_mem(model.write_param()) == 0
    assert net.load_model(str(directory / "model.bin")) == 0
    extractor = net.create_extractor()
    assert extractor.input(model.layers[0].outputs[0], ncnn.Mat(image)) == 0
    ret, output = extractor.extract(model.layers[-1].outputs[0])
    assert ret == 0
    return np.array(output)


def tiny_compact() -> ImageModelDescriptor:
    torch.manual_seed(0)
    state = Compact(num_feat=8, num_conv=2, upscale=2).state_dict()
    model = ModelLoader().load_from_state_dict(state)
    assert isinstance(model, ImageModelDescriptor)
    return model.eval()


@pytest.mark.parametrize("fp16", [0, 1], ids=["fp32", "fp16"])
def test_an_fp16_compact_export_converts_and_matches_its_onnx(
    fp16: int, tmp_path: Path
):
    model = tiny_compact()
    data = convert_to_onnx_impl(model, torch.device("cpu"), True, opset_version=20)
    data = with_recorded_arch(data, model.architecture.name, "fp16")
    wrapped, _ = convert_to_ncnn_node(load_onnx_model(data), fp16)
    assert "Cast" not in {layer.op_type for layer in wrapped.model.layers}
    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].type == "tensor(float)"
    rng = np.random.default_rng(0)
    gradient = np.linspace(0, 1, 32, dtype=np.float32)[None, None, :]
    for image in (
        rng.random((3, 32, 32), dtype=np.float32),
        gradient.repeat(32, axis=1).repeat(3, axis=0),
    ):
        expected = session.run(None, {"input": image[None]})[0][0].astype(np.float32)
        actual = run_ncnn(wrapped.model, image.copy(), tmp_path)
        # The ONNX's output is FP16: 1/2048 near 1.
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3)


def graph_of(*steps: tuple[str, int]) -> bytes:
    """A 1x1 Conv, then Casts `(output, to)` in a row; the last one's output is the graph's."""
    conv = helper.make_node("Conv", ["x", "w"], ["conv"])
    conv.attribute.append(helper.make_attribute("kernel_shape", [1, 1]))
    nodes = [conv]
    previous = "conv"
    for output, to in steps:
        nodes.append(helper.make_node("Cast", [previous], [output], to=to))
        previous = output
    weight = np.eye(C, dtype=np.float32).reshape(C, C, 1, 1)
    graph = helper.make_graph(
        nodes,
        "casts",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, "h", "w"])],
        [helper.make_tensor_value_info(previous, steps[-1][1], None)],
        [numpy_helper.from_array(weight, "w")],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    ).SerializeToString()


def test_a_cast_that_makes_the_graph_output_hands_it_its_name():
    data = graph_of(("y", TensorProto.FLOAT16))
    model = convert_to_ncnn_node(load_onnx_model(data), 0)[0].model
    assert [layer.op_type for layer in model.layers] == ["Input", "Convolution"]
    assert model.layers[-1].outputs == ["y"]


def test_casts_in_a_row_go_together():
    data = graph_of(("a", TensorProto.FLOAT16), ("y", TensorProto.FLOAT))
    model = convert_to_ncnn_node(load_onnx_model(data), 0)[0].model
    assert [layer.op_type for layer in model.layers] == ["Input", "Convolution"]
    assert model.layers[-1].outputs == ["y"]


@pytest.mark.parametrize("to", [TensorProto.INT32, TensorProto.DOUBLE])
def test_a_cast_to_any_other_type_stays_refused(to: int):
    data = graph_of(("y", to))
    with pytest.raises(ValueError, match="Cast not currently supported"):
        convert_to_ncnn_node(load_onnx_model(data), 0)
