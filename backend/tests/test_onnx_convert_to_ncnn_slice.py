"""Convert To NCNN on Slice: crops convert and give ONNX Runtime's result on ncnn.
Upstream's converter refused every explicit step of 1 (its test was inverted; PyTorch's
export writes steps), took an array's truth value for more than one axis, kept the
batch axis's entry it meant to drop, and converted a step of 2 as if it were 1 (a
silently wrong crop: NCNN's Crop has no step), which is now refused."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from ncnn import ncnn
from onnx import TensorProto, helper, numpy_helper

# The converter is native, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from nodes.impl.onnx.load import load_onnx_model
from packages.chaiNNer_onnx.onnx.utility.convert_to_ncnn import convert_to_ncnn_node


def model(inputs: dict[str, list[int]]) -> bytes:
    """A 1x1 Conv (NCNN models need one), then a Slice with these inputs."""
    weights = [
        numpy_helper.from_array(np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1), "w")
    ]
    names = ["mid"]
    for key in ("starts", "ends", "axes", "steps"):
        if key in inputs:
            weights.append(
                numpy_helper.from_array(np.array(inputs[key], np.int64), key)
            )
            names.append(key)
    conv = helper.make_node("Conv", ["x", "w"], ["mid"])
    conv.attribute.append(helper.make_attribute("kernel_shape", [1, 1]))
    graph = helper.make_graph(
        [conv, helper.make_node("Slice", names, ["y"])],
        "slice",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        weights,
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    ).SerializeToString()


@pytest.mark.parametrize(
    "inputs",
    [
        {"starts": [1, 2], "ends": [7, 9], "axes": [2, 3], "steps": [1, 1]},
        {"starts": [1, 2], "ends": [7, 9], "axes": [2, 3]},
        {"starts": [2], "ends": [9], "axes": [3], "steps": [1]},
        {"starts": [0, 1], "ends": [1, 7], "axes": [0, 2], "steps": [1, 1]},
    ],
)
def test_a_crop_converts_and_runs_as_onnx_runtime(
    tmp_path: Path, inputs: dict[str, list[int]]
):
    data = model(inputs)
    image = np.random.default_rng(0).random((1, 3, 10, 12), dtype=np.float32)
    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    (expected,) = session.run(None, {"x": image})

    wrapper = convert_to_ncnn_node(load_onnx_model(data), 0)[0]
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    wrapper.model.write_bin(tmp_path / "m.bin")
    assert net.load_param_mem(wrapper.model.write_param()) == 0
    assert net.load_model(str(tmp_path / "m.bin")) == 0
    extractor = net.create_extractor()
    extractor.input("x", ncnn.Mat(np.ascontiguousarray(image[0])))
    ret, out = extractor.extract("y")
    assert ret == 0
    np.testing.assert_allclose(np.array(out), np.asarray(expected)[0], atol=1e-6)


def test_a_step_ncnn_cannot_crop_with_is_refused():
    data = model({"starts": [0], "ends": [8], "axes": [2], "steps": [2]})
    with pytest.raises(AssertionError, match=r"Unsupported Slice step \[2\]"):
        convert_to_ncnn_node(load_onnx_model(data), 0)
