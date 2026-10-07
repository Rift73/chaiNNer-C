from __future__ import annotations

import onnx
import onnx.inliner
from sanic.log import logger

from ..native_graph import graph
from .model import OnnxGeneric, OnnxInfo, OnnxModel, OnnxRemBg, SizeReq
from .utils import (
    ModelShapeInference,
    get_opset,
    get_tensor_fp_datatype,
)


def _detect_size_req(infer: ModelShapeInference):
    return graph().onnx_detect_size_req(infer, SizeReq, logger)


def load_onnx_model(model_or_bytes: onnx.ModelProto | bytes) -> OnnxModel:
    return graph().onnx_load_model(
        model_or_bytes,
        {
            "onnx": onnx,
            "graph": graph,
            "get_opset": get_opset,
            "get_tensor_fp_datatype": get_tensor_fp_datatype,
            "OnnxInfo": OnnxInfo,
            "OnnxRemBg": OnnxRemBg,
            "OnnxGeneric": OnnxGeneric,
            "ModelShapeInference": ModelShapeInference,
            "_detect_size_req": _detect_size_req,
        },
    )
