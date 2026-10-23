"""ONNX Interpolate Models refuses two models of different precisions with a clear error:
the mix puts the interpolated weights into B's graph, which ONNX Runtime refused with
a bare type error ("Type parameter (T) of Optype (Conv) bound to different types").
At 0 % or 100 % the node still returns A or B as it is."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

# The model loader is native, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import NodeContext
from nodes.impl.onnx.load import load_onnx_model
from nodes.impl.onnx.model import OnnxModel
from packages.chaiNNer_onnx.onnx.utility.interpolate_models import (
    interpolate_models_node,
)


def model(dtype: type) -> OnnxModel:
    """A 3-channel 1x1 Conv in this precision."""
    kind = TensorProto.FLOAT16 if dtype is np.float16 else TensorProto.FLOAT
    conv = helper.make_node("Conv", ["x", "w"], ["y"])
    conv.attribute.append(helper.make_attribute("kernel_shape", [1, 1]))
    graph = helper.make_graph(
        [conv],
        "precision",
        [helper.make_tensor_value_info("x", kind, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("y", kind, [1, 3, "h", "w"])],
        [numpy_helper.from_array(np.eye(3, dtype=dtype).reshape(3, 3, 1, 1), "w")],
    )
    return load_onnx_model(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
        ).SerializeToString()
    )


# The node context is used only by the result's test upscale, after the refusal.
CONTEXT = cast(NodeContext, None)


def test_different_precisions_are_refused():
    fp32, fp16 = model(np.float32), model(np.float16)
    assert (fp32.info.dtype, fp16.info.dtype) == ("fp32", "fp16")
    with pytest.raises(ValueError, match=r"different precisions \(fp32 and fp16\)"):
        interpolate_models_node(CONTEXT, fp32, fp16, 50)


def test_the_ends_still_return_a_model_as_it_is():
    fp32, fp16 = model(np.float32), model(np.float16)
    assert interpolate_models_node(CONTEXT, fp32, fp16, 0)[0] is fp32
    assert interpolate_models_node(CONTEXT, fp32, fp16, 100)[0] is fp16
