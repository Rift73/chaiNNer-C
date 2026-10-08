"""A converted NCNN Reduction keeps its axes: ncnn's reduce_all (param 1) defaults to
1, so a per-axis reduction must write 1=0, which the schema's old default of 0
omitted, and ncnn reduced over everything."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from ncnn import ncnn
from onnx import ModelProto, TensorProto, helper

from nodes.impl.ncnn.model import NcnnModel
from nodes.impl.onnx.onnx_to_ncnn import Onnx2NcnnConverter

RNG = np.random.default_rng(5)
WEIGHT = RNG.uniform(-1, 1, (3, 3, 1, 1)).astype(np.float32)
BIAS = RNG.uniform(-1, 1, 3).astype(np.float32)
IMAGE = np.random.default_rng(3).uniform(0, 1, (3, 8, 8)).astype(np.float32)


def squared_deviation() -> ModelProto:
    """(c - mean over channels of c)^2 of a 1x1 convolution c, as LayerNorm2d does."""
    nodes = [
        helper.make_node("Conv", ["data", "w", "b"], ["c"], kernel_shape=[1, 1]),
        helper.make_node("ReduceMean", ["c"], ["m"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["c", "m"], ["d"]),
        helper.make_node("Mul", ["d", "d"], ["output"]),
    ]
    initializers = [
        helper.make_tensor("w", TensorProto.FLOAT, WEIGHT.shape, WEIGHT.ravel()),
        helper.make_tensor("b", TensorProto.FLOAT, BIAS.shape, BIAS),
    ]
    shape = [1, 3, 8, 8]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        initializers,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=7
    )


def run_ncnn(model: NcnnModel, directory: Path) -> np.ndarray:
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 1
    model.write_bin(directory / "model.bin")
    assert net.load_param_mem(model.write_param()) == 0
    assert net.load_model(str(directory / "model.bin")) == 0
    ex = net.create_extractor()
    assert ex.input(model.layers[0].outputs[0], ncnn.Mat(IMAGE)) == 0
    ret, out = ex.extract(model.layers[-1].outputs[0])
    assert ret == 0
    return np.array(out)


def test_a_converted_reduction_keeps_its_axes(tmp_path: Path):
    onnx_model = squared_deviation()
    (expected,) = ort.InferenceSession(
        onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"data": IMAGE[None]})
    converted = Onnx2NcnnConverter(onnx_model).convert(False, False)
    (reduction,) = [layer for layer in converted.layers if layer.op_type == "Reduction"]
    assert reduction.params[1].value == 0
    np.testing.assert_allclose(
        run_ncnn(converted, tmp_path), np.asarray(expected)[0], rtol=0, atol=1e-5
    )

    # Save Model, then Load Model: the per-axis reduction survives the round trip.
    param = converted.write_param()
    (tmp_path / "saved.param").write_text(param, encoding="utf-8")
    converted.write_bin(tmp_path / "saved.bin")
    loaded = NcnnModel.load_from_file(
        str(tmp_path / "saved.param"), str(tmp_path / "saved.bin")
    )
    assert loaded.write_param() == param
    np.testing.assert_allclose(
        run_ncnn(loaded, tmp_path), np.asarray(expected)[0], rtol=0, atol=1e-5
    )
