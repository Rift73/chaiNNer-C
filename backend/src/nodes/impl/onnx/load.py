from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

import onnx
import onnx.inliner
from google.protobuf.message import DecodeError
from sanic.log import logger

from api import NodeContext

from ..native_graph import graph
from ..tensorrt import drft
from ..tensorrt.dual import export_manifest
from . import rdual_bundle
from .autocast import mixed_precision
from .frozen_grid import FrozenGrid, find_frozen_grid
from .model import (
    ARCH_METADATA_KEY,
    MIXED,
    NO_RECORD_ARCH,
    PRECISION_METADATA_KEY,
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
    # The architecture and precision Convert To ONNX records; None for any other
    recorded: tuple[str, str] | None = None
    # DySample's sampling grid frozen at one size; None for any other
    frozen_grid: FrozenGrid | None = None


# An RDUAL ONNX bundle's: no graph to parse, and no session runs it (rdual_bundle.refuse)
_BUNDLE_FACTS = ModelFacts(native_session=False, dual_manifest=None)

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
        _recorded_arch(model),
        find_frozen_grid(model),
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


def _recorded_arch(model: onnx.ModelProto) -> tuple[str, str] | None:
    """The architecture and precision Convert To ONNX recorded in `model`'s metadata, if
    any; an architecture alone is a released version's record, which means FP16. An entry
    it cannot have written (edited, or another tool's, or one that kept the metadata and
    changed the precision: an FP16 export has an FP16 input or output, whichever its first
    layer takes, and an FP32 one neither) is logged and left out, and the model is one
    with no record: Build Engine and Build Program show its Precision."""
    entries = {entry.key: entry.value for entry in model.metadata_props}
    arch = entries.get(ARCH_METADATA_KEY)
    if arch is None:
        return None
    precision = entries.get(PRECISION_METADATA_KEY, "fp16")
    has_fp16 = any(
        value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
        for value in (*model.graph.input, *model.graph.output)
    )
    if (
        arch not in ("", "DUAL", NO_RECORD_ARCH)
        and precision in ("fp16", "fp32")
        and has_fp16 == (precision == "fp16")
    ):
        return arch, precision
    logger.warning(
        f"The ONNX model {model.graph.name!r} records architecture {arch!r} and precision"
        f" {precision!r}, which it cannot be; it is treated as one without a record."
    )
    return None


def with_recorded_arch(onnx_bytes: bytes, arch: str, precision: str) -> bytes:
    """The ONNX with its PyTorch model's architecture and its precision in its metadata."""
    model = onnx.load_model_from_string(onnx_bytes)
    model.metadata_props.add(key=ARCH_METADATA_KEY, value=arch)
    model.metadata_props.add(key=PRECISION_METADATA_KEY, value=precision)
    return model.SerializeToString()


def model_facts(model: OnnxModel) -> ModelFacts:
    """`model`'s facts: those load_onnx_model kept, else parsed from its bytes once."""
    facts = _facts.get(model)
    if facts is None and rdual_bundle.is_bundle(model.bytes):
        facts = _BUNDLE_FACTS
        _facts[model] = facts
    if facts is None:
        try:
            facts = _facts_of(onnx.load_model_from_string(model.bytes))
        except DecodeError:
            # ORT reports these bytes with its own InvalidProtobuf error.
            facts = ModelFacts(native_session=True, dual_manifest=None)
        _facts[model] = facts
    return facts


def warn_if_grid_frozen(context: NodeContext, model: OnnxModel, subject: str) -> None:
    """A warning, not a refusal: the model is right at the one size, and its weights may
    be all there is of it."""
    grid = model_facts(model).frozen_grid
    if grid is not None:
        size = "the one it was converted at"
        if grid.right_at is not None:
            size = f"{grid.right_at[0]}x{grid.right_at[1]}"
        context.warn(
            f"{subject} samples wrongly at every size but {size}, with no error: its"
            " conversion froze DySample's sampling grid. It must be converted again from"
            " the PyTorch model (Convert To ONNX)."
        )


def load_onnx_model(model_or_bytes: onnx.ModelProto | bytes) -> OnnxModel:
    """The model of an ONNX, or of an RDUAL ONNX bundle's bytes (rdual_bundle.py)."""
    if isinstance(model_or_bytes, bytes) and rdual_bundle.is_bundle(model_or_bytes):
        bundle = rdual_bundle.load(model_or_bytes)
        _facts[bundle] = _BUNDLE_FACTS
        return bundle
    parsed: list[ModelFacts] = []

    def fp_datatype(model: onnx.ModelProto) -> str:
        """A DUAL or legacy DRFT ONNX from Convert To ONNX keeps its own mixed
        precision (a BF16 or FP16 body, FP32 input and output), as does one an earlier version
        made mixed with ModelOpt ("mixed-fp16" or "mixed-bf16"); any other model has one.
        Called once with the parsed model, it also keeps the model's facts."""
        facts = _facts_of(model)
        parsed.append(facts)
        if facts.dual_manifest is not None or facts.drft_manifest is not None:
            return MIXED
        if facts.recorded is not None:
            return facts.recorded[1]
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
    (facts,) = parsed
    _facts[loaded] = facts
    if facts.recorded is not None:
        loaded.info.arch = facts.recorded[0]
    return loaded
