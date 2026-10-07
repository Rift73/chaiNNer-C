from sys import float_info

import numpy as np
from onnx import numpy_helper as onph
from onnx.onnx_pb import AttributeProto, NodeProto, TensorProto
from sanic.log import logger

from ..native_graph import graph

INT64_MIN, INT64_MAX = np.iinfo(np.int64).min, np.iinfo(np.int64).max
FLOAT32_MAX = float_info.max

APT = AttributeProto
TPT = TensorProto


def get_node_attr_ai(node: NodeProto, key: str) -> np.ndarray:
    return graph().onnx_get_node_attr_ai(node, key)


def set_node_attr_ai(node: NodeProto, key: str, value: np.ndarray) -> None:
    graph().onnx_set_node_attr_ai(node, key, value, AttributeProto)


def get_node_attr_af(node: NodeProto, key: str) -> np.ndarray:
    return graph().onnx_get_node_attr_af(node, key)


def get_node_attr_i(node: NodeProto, key: str, default: int = 0) -> int:
    return graph().onnx_get_node_attr_i(node, key, default)


def get_node_attr_f(node: NodeProto, key: str, default: float = 0) -> float:
    return graph().onnx_get_node_attr_f(node, key, default)


def get_node_attr_s(node: NodeProto, key: str, default: str = ""):
    return graph().onnx_get_node_attr_s(node, key, default)


def get_node_attr_tensor(node: NodeProto, key: str) -> TensorProto:
    return graph().onnx_get_node_attr_tensor(node, key, TensorProto)


def get_node_attr_from_input_f(tp: TensorProto) -> float:
    return graph().onnx_get_node_attr_from_input_f(tp, onph.to_array)


def get_node_attr_from_input_ai(tp: TensorProto) -> np.ndarray:
    return graph().onnx_get_node_attr_from_input_ai(tp, onph.to_array, logger)


def get_node_attr_from_input_af(tp: TensorProto) -> np.ndarray:
    return graph().onnx_get_node_attr_from_input_af(tp, onph.to_array, logger)


def get_tensor_proto_data_size(tp: TensorProto, fpmode: int = TPT.FLOAT) -> int:
    return graph().onnx_get_tensor_proto_data_size(tp, fpmode)
