"""Convert To NCNN refuses a model that needs a constant tensor: NCNN keeps one in a
MemoryData layer, which chaiNNer does not write, and the converter counted the layer
but left it out, so the node returned files ncnn refuses ("find_blob_index_by_name k
failed"). A scalar constant, which the converter folds into its operation, converts."""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

# The converter is native, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from nodes.impl.onnx.load import load_onnx_model
from packages.chaiNNer_onnx.onnx.utility.convert_to_ncnn import convert_to_ncnn_node

C = 4


def model(constant: np.ndarray) -> bytes:
    """A 1x1 Conv (NCNN models need one), then a Mul by this constant."""
    conv = helper.make_node("Conv", ["x", "w"], ["mid"])
    conv.attribute.append(helper.make_attribute("kernel_shape", [1, 1]))
    graph = helper.make_graph(
        [conv, helper.make_node("Mul", ["mid", "k"], ["y"])],
        "constants",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, "h", "w"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [
            numpy_helper.from_array(
                np.eye(C, dtype=np.float32).reshape(C, C, 1, 1), "w"
            ),
            numpy_helper.from_array(constant, "k"),
        ],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    ).SerializeToString()


@pytest.mark.parametrize("shape", [(1, C, 1, 1), (C, 1, 1), (1,)])
def test_a_constant_tensor_is_refused(shape: tuple[int, ...]):
    data = model(np.full(shape, 0.5, np.float32))
    with pytest.raises(ValueError, match=r"needs constant tensors.*\(constants: k\)"):
        convert_to_ncnn_node(load_onnx_model(data), 0)


def test_a_scalar_constant_converts():
    scalar = np.array(0.5, np.float32)
    wrapper = convert_to_ncnn_node(load_onnx_model(model(scalar)), 0)[0]
    assert "BinaryOp" in wrapper.model.write_param()
