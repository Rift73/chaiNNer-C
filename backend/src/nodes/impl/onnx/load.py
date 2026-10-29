from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

import onnx
import onnx.inliner
from google.protobuf.message import DecodeError
from sanic.log import logger

from ..native_graph import graph
from ..tensorrt import drft
from ..tensorrt.dual import export_manifest
from .autocast import mixed_precision
from .model import (
    MIXED,
    SIZE_REQ_METADATA_KEY,
    OnnxGeneric,
    OnnxInfo,
    OnnxModel,
    OnnxRemBg,
    SizeReq,
)
from .utils import (
    ModelShapeInference,
    get_opset,
    get_tensor_fp_datatype,
)


def _detect_size_req(infer: ModelShapeInference):
    return graph().onnx_detect_size_req(infer, SizeReq, logger)


@dataclass(frozen=True)
class ModelFacts:
    """What a session or an engine build needs from a model's parsed graph."""

    # Whether the native CPU session runs it (dense numeric tensor inputs and outputs)
    native_session: bool
    # The exporter's manifest of a DUAL ONNX from Convert To ONNX; None for any other
    dual_manifest: dict[str, Any] | None
    # The exporter's manifest of a legacy DRFT ONNX from Convert To ONNX; None for any
    # other
    drft_manifest: dict[str, Any] | None = None
    # The PyTorch model's size requirement, which Convert To ONNX records; None for any
    # other
    size_req: SizeReq | None = None


# The facts load_onnx_model found while it parsed each model, so that its sessions and
# engine builds do not parse the model's bytes again.
_facts: WeakKeyDictionary[OnnxModel, ModelFacts] = WeakKeyDictionary()


def _facts_of(model: onnx.ModelProto) -> ModelFacts:
    # Imports ONNX Runtime, which the PyTorch package (Convert To ONNX) does not load
    # with this module.
    from ..native_onnx_runtime import supports_graph

    return ModelFacts(
        supports_graph(model.graph),
        export_manifest(model),
        drft.export_manifest(model),
        _recorded_size_req(model),
    )


def _recorded_size_req(model: onnx.ModelProto) -> SizeReq | None:
    """The size requirement Convert To ONNX recorded in `model`'s metadata, if any. An
    entry it cannot have written (edited, or another tool's) is logged and left out,
    and the probe alone gives the requirement (onnx/size_probe.py)."""
    for entry in model.metadata_props:
        if entry.key == SIZE_REQ_METADATA_KEY:
            try:
                req = json.loads(entry.value)
                # Spandrel's minimum of 0 (none) is a SizeReq's 1
                return SizeReq(max(1, req["minimum"]), req["multiple_of"])
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(
                    f"The ONNX model {model.graph.name!r} has a malformed {entry.key}"
                    f" entry {entry.value!r} ({type(e).__name__}: {e}); its size"
                    " requirement is probed instead."
                )
                return None
    return None


def model_facts(model: OnnxModel) -> ModelFacts:
    """`model`'s facts: those load_onnx_model kept, else parsed from its bytes once."""
    facts = _facts.get(model)
    if facts is None:
        try:
            facts = _facts_of(onnx.load_model_from_string(model.bytes))
        except DecodeError:
            # ORT reports these bytes with its own InvalidProtobuf error.
            facts = ModelFacts(native_session=True, dual_manifest=None)
        _facts[model] = facts
    return facts


def load_onnx_model(model_or_bytes: onnx.ModelProto | bytes) -> OnnxModel:
    parsed: list[ModelFacts] = []

    def fp_datatype(model: onnx.ModelProto) -> str:
        """A DUAL or legacy DRFT ONNX from Convert To ONNX keeps its own mixed
        precision (a BF16 or FP16 body, FP32 input and output), as does one Convert To ONNX made
        mixed with ModelOpt ("mixed-fp16" or "mixed-bf16"); any other model has one.
        Called once with the parsed model, it also keeps the model's facts."""
        facts = _facts_of(model)
        parsed.append(facts)
        if facts.dual_manifest is not None or facts.drft_manifest is not None:
            return MIXED
        return mixed_precision(model) or get_tensor_fp_datatype(model)

    def detect_size_req(infer: ModelShapeInference):
        """Shape inference's size requirement, except for a legacy DRFT ONNX from
        Convert To ONNX: ONNX's checker knows none of its TensorRT plugin ops, so every
        size fails alike (no requirement, no scale) and would log its traceback."""
        (facts,) = parsed
        if facts.drft_manifest is not None:
            return SizeReq(), None
        return _detect_size_req(infer)

    loaded = graph().onnx_load_model(
        model_or_bytes,
        {
            "onnx": onnx,
            "graph": graph,
            "get_opset": get_opset,
            "get_tensor_fp_datatype": fp_datatype,
            "OnnxInfo": OnnxInfo,
            "OnnxRemBg": OnnxRemBg,
            "OnnxGeneric": OnnxGeneric,
            "ModelShapeInference": ModelShapeInference,
            "_detect_size_req": detect_size_req,
        },
    )
    (_facts[loaded],) = parsed
    return loaded
