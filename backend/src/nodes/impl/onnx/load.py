from __future__ import annotations

import onnx
import onnx.inliner
from sanic.log import logger

from ..native_graph import graph
from ..tensorrt.dual import export_manifest
from .model import MIXED, OnnxGeneric, OnnxInfo, OnnxModel, OnnxRemBg, SizeReq
from .utils import (
    ModelShapeInference,
    get_opset,
    get_tensor_fp_datatype,
)


def _detect_size_req(infer: ModelShapeInference):
    return graph().onnx_detect_size_req(infer, SizeReq, logger)


def _fp_datatype(model: onnx.ModelProto) -> str:
    """A DUAL ONNX from Convert To ONNX keeps its own mixed precision (BF16 body, FP32
    input and output); any other model has one."""
    if export_manifest(model) is not None:
        return MIXED
    return get_tensor_fp_datatype(model)


def load_onnx_model(model_or_bytes: onnx.ModelProto | bytes) -> OnnxModel:
    return graph().onnx_load_model(
        model_or_bytes,
        {
            "onnx": onnx,
            "graph": graph,
            "get_opset": get_opset,
            "get_tensor_fp_datatype": _fp_datatype,
            "OnnxInfo": OnnxInfo,
            "OnnxRemBg": OnnxRemBg,
            "OnnxGeneric": OnnxGeneric,
            "ModelShapeInference": ModelShapeInference,
            "_detect_size_req": _detect_size_req,
        },
    )
