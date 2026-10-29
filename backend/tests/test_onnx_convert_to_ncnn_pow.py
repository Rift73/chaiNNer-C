"""Convert To NCNN raises to an integer power with ncnn layers that take a negative
base as PyTorch does: ncnn's BinaryOp pow goes through the base's logarithm in its SIMD
path, so a negative base gave about 2.4e38 where PyTorch and ONNX Runtime give x * x for
x ** 2 (a variance written as (x - mean) ** 2, as NAFNet's LayerNorm has, squares
negative values). A fractional power keeps ncnn's pow (PyTorch gives NaN for a negative
base there)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

# The converter is native, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from ncnn import ncnn

from nodes.impl.onnx.load import load_onnx_model
from packages.chaiNNer_onnx.onnx.utility.convert_to_ncnn import convert_to_ncnn_node

# Eight channels: ncnn packs them (elempack 4 or 8), the path whose pow was wrong.
C, H, W = 8, 6, 10


def model(exponent: float) -> bytes:
    """A 1x1 identity Conv (NCNN models need one), then Pow by `exponent`."""
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["x", "w"], ["mid"], kernel_shape=[1, 1]),
            helper.make_node("Pow", ["mid", "k"], ["y"]),
        ],
        "pow",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, "h", "w"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [
            numpy_helper.from_array(
                np.eye(C, dtype=np.float32).reshape(C, C, 1, 1), "w"
            ),
            numpy_helper.from_array(np.array(exponent, np.float32), "k"),
        ],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    ).SerializeToString()


def run(exponent: float, image: np.ndarray, tmp_path: Path) -> tuple[str, np.ndarray]:
    """The converted model's param and its output on ncnn's CPU in FP32."""
    converted = convert_to_ncnn_node(load_onnx_model(model(exponent)), 0)[0].model
    param = converted.write_param()
    converted.write_bin(tmp_path / "model.bin")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    assert net.load_param_mem(param) == 0
    assert net.load_model(str(tmp_path / "model.bin")) == 0
    extractor = net.create_extractor()
    assert extractor.input("x", ncnn.Mat(np.ascontiguousarray(image))) == 0
    ret, output = extractor.extract("y")
    assert ret == 0
    return param, np.array(output)


def negative_and_positive() -> np.ndarray:
    return np.random.default_rng(0).uniform(-2, 2, (C, H, W)).astype(np.float32)


def test_a_square_of_negative_values_is_their_square(tmp_path: Path):
    image = negative_and_positive()
    param, output = run(2.0, image, tmp_path)
    assert "UnaryOp" in param and "BinaryOp" not in param
    np.testing.assert_array_equal(output, image * image)


@pytest.mark.parametrize("exponent", [3.0, -1.0, 4.0])
def test_other_integer_powers_of_negative_values_match_pytorch(
    exponent: float, tmp_path: Path
):
    image = negative_and_positive()
    param, output = run(exponent, image, tmp_path)
    assert "Power" in param and "BinaryOp" not in param
    expected = torch.from_numpy(image).pow(exponent).numpy()
    np.testing.assert_allclose(output, expected, rtol=1e-6)


def test_a_fractional_power_stays_ncnn_pow(tmp_path: Path):
    image = np.random.default_rng(0).uniform(0.1, 2, (C, H, W)).astype(np.float32)
    param, output = run(0.5, image, tmp_path)
    assert "BinaryOp" in param and "Power" not in param
    np.testing.assert_allclose(output, np.sqrt(image), rtol=1e-6)
