from __future__ import annotations

from typing import Literal, Sequence, Tuple, Union

import onnx
from onnx.onnx_pb import ModelProto, ValueInfoProto

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


def safely_optimize_onnx_model(model_proto: ModelProto) -> ModelProto:
    """
    Optimizes the model using onnxoptimizer. If onnxoptimizer is not installed, the model is returned as is.
    """
    try:
        import onnxoptimizer

        passes = onnxoptimizer.get_fuse_and_elimination_passes()
        model_proto = onnxoptimizer.optimize(model_proto, passes)
    except Exception:
        pass
    return model_proto
