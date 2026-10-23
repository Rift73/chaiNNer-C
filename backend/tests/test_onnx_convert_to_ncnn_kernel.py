"""Convert To NCNN on convolutions: a ConvTranspose converts (upstream's converter
compared its absent output_shape, an empty array, with 2, which NumPy 2 refuses, so
every ConvTranspose failed), and a Conv or ConvTranspose without the optional
kernel_shape gets the one its weights have, as ONNX infers it (the converter read only
the attribute and made a kernel 0 wide, which failed with ZeroDivisionError)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from ncnn import ncnn
from onnx import TensorProto, helper, numpy_helper

# The converter is native, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from nodes.impl.ncnn.model import NcnnModelWrapper
from nodes.impl.onnx.load import load_onnx_model
from packages.chaiNNer_onnx.onnx.utility.convert_to_ncnn import convert_to_ncnn_node

RNG = np.random.default_rng(0)
WEIGHTS = {
    "w": RNG.standard_normal((4, 3, 3, 3)).astype(np.float32) * 0.2,
    "b": RNG.standard_normal(4).astype(np.float32) * 0.1,
    "tw": RNG.standard_normal((4, 3, 2, 2)).astype(np.float32) * 0.2,
}


def model(with_kernel_shape: bool) -> bytes:
    """A 3x3 Conv (3 -> 4 channels, padded), then a 2x2 stride-2 ConvTranspose back to 3."""
    conv = helper.make_node("Conv", ["x", "w", "b"], ["mid"], pads=[1, 1, 1, 1])
    transpose = helper.make_node("ConvTranspose", ["mid", "tw"], ["y"], strides=[2, 2])
    if with_kernel_shape:
        conv.attribute.append(helper.make_attribute("kernel_shape", [3, 3]))
        transpose.attribute.append(helper.make_attribute("kernel_shape", [2, 2]))
    graph = helper.make_graph(
        [conv, transpose],
        "kernel",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, "h2", "w2"])],
        [numpy_helper.from_array(v, k) for k, v in WEIGHTS.items()],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    ).SerializeToString()


def converted(data: bytes) -> NcnnModelWrapper:
    return convert_to_ncnn_node(load_onnx_model(data), 0)[0]


def test_a_conv_transpose_converts_and_runs_as_onnx_runtime(tmp_path: Path):
    wrapper = converted(model(True))
    assert (wrapper.scale, wrapper.in_nc, wrapper.out_nc) == (2, 3, 3)

    image = RNG.random((1, 3, 7, 9), dtype=np.float32)
    session = ort.InferenceSession(model(True), providers=["CPUExecutionProvider"])
    (expected,) = session.run(None, {"x": image})

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    wrapper.model.write_bin(tmp_path / "m.bin")
    assert net.load_param_mem(wrapper.model.write_param()) == 0
    assert net.load_model(str(tmp_path / "m.bin")) == 0
    extractor = net.create_extractor()
    extractor.input("x", ncnn.Mat(np.ascontiguousarray(image[0])))
    ret, out = extractor.extract("y")
    assert ret == 0
    np.testing.assert_allclose(np.array(out), np.asarray(expected)[0], atol=1e-5)


def test_a_conv_without_kernel_shape_converts_as_with_it():
    without, with_it = converted(model(False)), converted(model(True))
    assert without.model.write_param() == with_it.model.write_param()
    assert without.model.serialize_weights() == with_it.model.serialize_weights()
