"""Convert To NCNN refuses a model that computes, as it runs, an input NCNN needs as a
constant (here a Reshape's shape, read from the input's size as NAFNet's LayerNorm
views its weights), naming the tensor and the node: the converter raised a bare
KeyError naming the tensor."""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

# The converter is native, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from nodes.impl.onnx.load import load_onnx_model
from packages.chaiNNer_onnx.onnx.utility.convert_to_ncnn import convert_to_ncnn_node

C = 4


def model(cast_view: bool = False) -> bytes:
    """A 1x1 Conv (NCNN models need one), then a Mul by a per-channel weight reshaped to
    (1, C, 1, 1), C read from the Conv's output; with `cast_view`, a Cast of the shape,
    unused, comes first among the nodes taking it."""
    one = numpy_helper.from_array(np.array([1], np.int64), "one")
    cast = [helper.make_node("Cast", ["view"], ["as_float"], name="as_float", to=1)]
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["mid"], kernel_shape=[1, 1]),
        helper.make_node("Shape", ["mid"], ["shape"]),
        helper.make_node("Gather", ["shape", "index"], ["channels"], axis=0),
        helper.make_node("Unsqueeze", ["channels", "axes"], ["channels_1d"]),
        helper.make_node(
            "Concat", ["one", "channels_1d", "one", "one"], ["view"], axis=0
        ),
        *(cast if cast_view else []),
        helper.make_node("Reshape", ["k", "view"], ["k_view"], name="weight_view"),
        helper.make_node("Mul", ["mid", "k_view"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "runtime_shapes",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, "h", "w"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [
            numpy_helper.from_array(
                np.eye(C, dtype=np.float32).reshape(C, C, 1, 1), "w"
            ),
            numpy_helper.from_array(np.arange(C, dtype=np.float32), "k"),
            numpy_helper.from_array(np.array(1, np.int64), "index"),
            numpy_helper.from_array(np.array([0], np.int64), "axes"),
            one,
        ],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    ).SerializeToString()


def test_a_shape_computed_as_the_model_runs_is_named():
    with pytest.raises(
        ValueError,
        match=r"computes view as it runs.*\(taken by Reshape node weight_view\)",
    ):
        convert_to_ncnn_node(load_onnx_model(model()), 0)


def test_every_node_taking_the_computed_tensor_is_named():
    # The converter does not say which node needed the constant.
    with pytest.raises(
        ValueError,
        match=r"\(taken by Cast node as_float, Reshape node weight_view\)",
    ):
        convert_to_ncnn_node(load_onnx_model(model(cast_view=True)), 0)
