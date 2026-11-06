from __future__ import annotations

from typing import Literal, Sequence, Tuple, Union

import onnx
from onnx.onnx_pb import ModelProto, ValueInfoProto
from sanic.log import logger

from ..native_graph import graph

OnnxTensorFormat = Literal["BCHW", "BHWC"]
OnnxTensorShape = Tuple[
    Union[int, str],
    Union[int, str],
    Union[int, str],
    Union[int, str],
]
OnnxParsedTensorShape = Tuple[
    OnnxTensorFormat,
    int,
    Union[int, None],
    Union[int, None],
]
"""
The elements are:
- The tensor format (BCHW or BHWC).
- The number of channels.
- The width (optional).
- The height (optional).
"""


def _as_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _or_else(value: int | None, default: int) -> int:
    return value if value is not None else default


def parse_onnx_shape(shape: Sequence[int | str | None]) -> OnnxParsedTensorShape:
    return graph().onnx_parse_shape(shape)


def to_onnx_tensor_shape(
    tensor: onnx.TypeProto.Tensor,
) -> OnnxTensorShape:
    return graph().onnx_tensor_shape(tensor)


def is_tensor_input(input: ValueInfoProto) -> bool:
    return graph().onnx_tensor_input(input)


def is_image_to_image(model: ModelProto) -> bool:
    """
    Returns whether the model is an image to image model (single image input -> single image output).
    """
    return graph().onnx_image_to_image(model)


class ModelShapeInference:
    def __init__(self, model: ModelProto):
        graph().onnx_infer_init(self, model)

    def infer_shape(
        self, input_size: tuple[int, int]
    ) -> tuple[
        tuple[int | None, int | None, int | None],
        tuple[int | None, int | None, int | None],
    ]:
        """
        input_shape: The size of the input tensors as width, height.

        return: The shapes of the input and output tensors in HWC format.

        **This will mutate the model.**
        """
        return graph().onnx_infer_shape(self, input_size)


def get_tensor_fp_datatype(model: ModelProto) -> str:
    return graph().onnx_fp_type(model)


def get_opset(model: onnx.ModelProto) -> int:
    return graph().onnx_opset(model)


def _reshape_copies_from_a_reshape(model_proto: ModelProto) -> bool:
    """Whether a Reshape reads another Reshape's output with a 0 in its constant shape,
    which copies that dimension from the output. PyTorch's pixel_unshuffle exports this
    pair."""
    graph_proto = model_proto.graph
    constants = {t.name: t for t in graph_proto.initializer}
    for node in graph_proto.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    constants[node.output[0]] = attribute.t
    reshaped = {
        node.output[0] for node in graph_proto.node if node.op_type == "Reshape"
    }
    return any(
        node.op_type == "Reshape"
        and node.input[0] in reshaped
        and node.input[1] in constants
        and 0 in onnx.numpy_helper.to_array(constants[node.input[1]])
        for node in graph_proto.node
    )


def safely_optimize_onnx_model(model_proto: ModelProto) -> ModelProto:
    """
    Optimizes the model using onnxoptimizer. If onnxoptimizer is not installed, the model is returned as is.
    So is a model that is not valid ONNX: onnxoptimizer's native code can end the
    process on one (a node without the inputs its op needs, an output no node makes).
    Its eliminate_consecutive_idempotent_ops drops the first of two Reshapes, which
    changes the second's result when that copies a dimension from the first, so a model
    with such a pair goes without that pass.
    """
    # The checker takes a model in memory up to 2 GiB, as onnxoptimizer does.
    if model_proto.ByteSize() <= onnx.checker.MAXIMUM_PROTOBUF:
        try:
            onnx.checker.check_model(model_proto)
        except onnx.checker.ValidationError as e:
            logger.warning(f"Not optimizing the ONNX model, which is not valid: {e}")
            return model_proto
    try:
        import onnxoptimizer

        passes = onnxoptimizer.get_fuse_and_elimination_passes()
        if _reshape_copies_from_a_reshape(model_proto):
            passes.remove("eliminate_consecutive_idempotent_ops")
        model_proto = onnxoptimizer.optimize(model_proto, passes)
    except Exception:
        pass
    return model_proto
