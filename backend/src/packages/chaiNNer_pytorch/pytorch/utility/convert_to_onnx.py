from __future__ import annotations

import math
import tempfile
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from weakref import WeakKeyDictionary

import numpy as np
import onnx
import torch
from sanic.log import logger
from spandrel import ImageModelDescriptor

from api import Aborted, NodeContext, NodeId
from nodes.groups import Condition, if_group
from nodes.impl.onnx import rdual_bundle
from nodes.impl.onnx.load import load_onnx_model, with_recorded_arch
from nodes.impl.onnx.model import (
    FP16_ARCHS,
    FP16_ARCHS_NAVI,
    MIXED,
    NO_RECORD_ARCH,
    OnnxGeneric,
    converts_in_fp16,
)
from nodes.impl.pytorch.convert_to_onnx_impl import (
    export_copy,
    export_onnx,
    is_onnx_supported,
)
from nodes.impl.tensorrt import drft_export, dual
from nodes.properties.inputs import (
    BoolInput,
    EnumInput,
    NumberInput,
    SrModelInput,
)
from nodes.properties.outputs import OnnxModelOutput, TextOutput

from ...settings import get_settings
from .. import utility_group
from ..io.load_model import full_precision_source


class OnnxDataType(Enum):
    FP32 = 0
    FP16 = 1


DATA_TYPE_LABELS: dict[OnnxDataType, str] = {
    OnnxDataType.FP32: "fp32",
    OnnxDataType.FP16: "fp16",
}


class Opset(Enum):
    OPSET_14 = 14
    OPSET_15 = 15
    OPSET_16 = 16
    OPSET_17 = 17
    OPSET_18 = 18
    OPSET_19 = 19
    OPSET_20 = 20
    OPSET_21 = 21
    OPSET_22 = 22
    OPSET_23 = 23


OPSET_LABELS: dict[Opset, str] = {
    Opset.OPSET_14: "14",
    Opset.OPSET_15: "15",
    Opset.OPSET_16: "16",
    Opset.OPSET_17: "17",
    Opset.OPSET_18: "18",
    Opset.OPSET_19: "19",
    Opset.OPSET_20: "20",
    Opset.OPSET_21: "21",
    Opset.OPSET_22: "22",
    Opset.OPSET_23: "23",
}


class DualShape(Enum):
    DYNAMIC = "dynamic"
    FIXED = "fixed"


DUAL_SHAPE_LABELS: dict[DualShape, str] = {
    DualShape.DYNAMIC: "Dynamic (any size)",
    DualShape.FIXED: "Fixed (one size)",
}


@dataclass
class _Export:
    result: tuple[OnnxGeneric, str, str]
    verified: bool
    # What the conversion warned of, which holds for every use of this file
    warnings: tuple[str, ...] = ()


# Each node's latest export of a model while the model lives, with everything it
# depends on: a re-run returns the same ONNX model (with the sessions kept for it)
# without exporting again. One per node and model: a superseded export's sessions would
# hold GPU memory, and two nodes converting one model with different options keep both.
_exports: WeakKeyDictionary[
    ImageModelDescriptor, dict[NodeId, tuple[Hashable, _Export]]
] = WeakKeyDictionary()
_exports_lock = Lock()


def _kept_export(
    model: ImageModelDescriptor, node: NodeId, key: Hashable
) -> _Export | None:
    """The node's kept export of the model for these options, else another node's (then
    kept for this node too): nodes with the same options share one ONNX model, so its
    sessions and engines exist once."""
    with _exports_lock:
        kept = _exports.get(model, {})
        own = kept.get(node)
        if own is not None and own[0] == key:
            return own[1]
        for other in kept.values():
            if other[0] == key:
                kept[node] = other
                return other[1]
    return None


def _keep_export(
    model: ImageModelDescriptor, node: NodeId, key: Hashable, export: _Export
) -> None:
    with _exports_lock:
        _exports.setdefault(model, {})[node] = key, export


# DUAL and RDUAL convert to TensorRT-only ONNX, of the TensorRT Shape inputs
IS_DUAL = Condition.type(0, 'PyTorchModel { arch: "DUAL" | "RDUAL" }')
# Shown until the model is known to be DUAL or RDUAL
NOT_DUAL = Condition.type(
    0,
    'PyTorchModel { arch: invStrSet("DUAL") & invStrSet("RDUAL") }',
    if_not_connected=True,
)
# Every architecture has one automatic precision, so Data Type is hidden for any model,
# known or not, and a value a saved chain holds is ignored.
DATA_TYPE_APPLIES = Condition.const(False)


FP16_ARCH_NAMES = ", ".join(
    f"{arch} ({', '.join(sizes)})" if sizes else arch
    for arch, sizes in FP16_ARCHS.items()
)


@utility_group.register(
    schema_id="chainner:pytorch:convert_to_onnx",
    name="Convert To ONNX",
    description=[
        f"Convert a PyTorch model to ONNX, in a precision the node chooses. FP16, whatever PyTorch's FP16 setting, for the architectures measured to hold in FP16 on TensorRT: {FP16_ARCH_NAMES}. FP32 for every other architecture or size, which is always right and only slower. FP Mode says which it is.",
        "An FP16 export is checked on the CPU against the model in FP32 on small generated images, a guard against one odd model and no proof of what a GPU does. If the export fails or does not hold in that check, the model converts in FP32 and the node warns why. An FP32 export is compared with the model too, and the node warns if it does not match.",
        "The ONNX records its architecture and precision, so Build Engine and Build Program build it in that precision with no Precision to pick, also after it is saved and loaded again.",
        "With PyTorch's FP16 setting on, Load Model keeps a model in FP16; the conversion reads its file again for its FP32 weights (if the file changed since, it exports the FP16 weights, and the log says so).",
        "A legacy DRFT model (LayerNorm, not i-LN) converts to a TensorRT-only ONNX with chaiNNer-C's DRFT TensorRT plugins, which ONNX Runtime cannot run: its precision is fixed and mixed (FP16 with BF16 attention, FP32 input and output), its opset is 20, it takes every input size from 64 px (twice the window for larger windows), and Opset and Verify do not apply to it.",
        "An RDUAL model (a video model) converts to TensorRT-only ONNX parts for one frame size, with TensorRT Shape Fixed at the video's height and width (any size; Dynamic is refused). The four parts and their manifest are one ONNX model, which Build Engine builds into one engine set and Save Model writes as one file that only chaiNNer-C reads.",
        "It is recommended to save converted models as a separate step, then load the converted models instead of converting them every time you run the chain.",
        "Note: Converted models are not guaranteed to work with other programs that support ONNX models. This is for a variety of reasons and cannot be changed.",
    ],
    icon="ONNX",
    inputs=[
        SrModelInput("PyTorch Model"),
        if_group(NOT_DUAL)(
            if_group(DATA_TYPE_APPLIES)(
                EnumInput(
                    OnnxDataType,
                    label="Data Type",
                    default=OnnxDataType.FP32,
                    option_labels=DATA_TYPE_LABELS,
                ).with_docs(
                    "Hidden for every model: the node chooses the precision, and a value a"
                    " saved chain holds is ignored.",
                    hint=True,
                ),
            ),
            EnumInput(
                Opset,
                label="Opset",
                default=Opset.OPSET_20,
                option_labels=OPSET_LABELS,
            ).with_docs(
                "Opset 17 and newer export LayerNorm as one ONNX operator, and 20 and newer"
                " GELU too, which TensorRT and ONNX Runtime run as single layers. Older"
                " opsets suit older ONNX programs. GFISRv2 and FIGSR need 17 or newer.",
                hint=True,
            ),
            BoolInput("Verify", default=False).with_docs(
                "An FP16 export is always verified, with this on or off, against the model in FP32, kept in FP16 between its layers as a GPU keeps it: no NaN or infinite values; on a smooth image at two sizes, on flat black and white and on a gradient, a root mean square difference of at most 0.005; on a drawn and a textured image, a root mean square difference of at most 0.015. This guards against one odd model on the CPU and is no proof of what a GPU does. PyTorch's own FP16 is no reference: it and ONNX Runtime round at different places.",
                "An export in FP32 is compared on those images too, and with this off the node warns if it does not match the model. With this on, it also runs on noise, must stay within 0.001 of the model, and the node refuses it if not.",
                hint=True,
            ),
        ),
        if_group(IS_DUAL)(
            EnumInput(
                DualShape,
                label="TensorRT Shape",
                default=DualShape.DYNAMIC,
                option_labels=DUAL_SHAPE_LABELS,
            )
            .with_id(6)
            .with_docs(
                "DUAL converts to a TensorRT-only ONNX. Its precision is fixed and mixed"
                " (BF16 body, FP32 input and output) and it uses opset 20 with DUAL's"
                " TensorRT plugins, which ONNX Runtime cannot run, so the node hides Data"
                " Type, Opset and Verify for DUAL.",
                "Dynamic: one ONNX for every input size that is a multiple of 4 px (as"
                " DUAL pads its input), so images are not padded further; Build Engine's"
                " shape inputs choose the engine's sizes (e.g. 64x64 to 1920x1088 for"
                " whole 1080p frames).",
                "Fixed: an ONNX for one input size (a multiple of 4); Upscale Image tiles"
                " at that size, padding smaller images. DUAL pools over its whole input,"
                " so tiles and padding can change its output slightly against one whole"
                " image.",
                "RDUAL takes Fixed only, at the video's frame size (any height and"
                " width): its engines are built for one size.",
                hint=True,
            ),
        ),
        if_group(IS_DUAL & Condition.enum(6, DualShape.FIXED))(
            NumberInput("TensorRT Height", default=512, min=4, step=4, unit="px")
            .with_id(4)
            .with_docs("The fixed input height (for DUAL a multiple of 4)."),
            NumberInput("TensorRT Width", default=512, min=4, step=4, unit="px")
            .with_id(5)
            .with_docs("The fixed input width (for DUAL a multiple of 4)."),
        ),
    ],
    outputs=[
        OnnxModelOutput(
            model_type=(
                "OnnxGenericModel & pytorchToOnnx(Input0)"
                f' & OnnxModel {{ arch: Input0.arch & invStrSet("{NO_RECORD_ARCH}") }}'
            ),
            label="ONNX Model",
        ),
        TextOutput(
            "FP Mode",
            f"""
                match Input0.arch {{
                    "DUAL" => "mixed",
                    "RDUAL" => "mixed",
                    "DRFT" => "mixed" | "fp32",
                    {FP16_ARCHS_NAVI} => "fp16" | "fp32",
                    _ => "fp32",
                }}
            """,
        ),
        TextOutput(
            "Opset",
            """
                let opset = Input2;
                match Input0.arch {
                    "DUAL" => "opset20",
                    "RDUAL" => "opset20",
                    _ => match opset {
                        Opset::Opset14 => "opset14",
                        Opset::Opset15 => "opset15",
                        Opset::Opset16 => "opset16",
                        Opset::Opset17 => "opset17",
                        Opset::Opset18 => "opset18",
                        Opset::Opset19 => "opset19",
                        Opset::Opset20 => "opset20",
                        Opset::Opset21 => "opset21",
                        Opset::Opset22 => "opset22",
                        Opset::Opset23 => "opset23",
                    },
                }
            """,
        ),
    ],
    node_context=True,
)
def convert_to_onnx_node(
    context: NodeContext,
    model: ImageModelDescriptor,
    data_type: OnnxDataType,
    opset: Opset,
    verify: bool,
    dual_shape: DualShape,
    dual_height: int,
    dual_width: int,
) -> tuple[OnnxGeneric, str, str]:
    if model.architecture.id == "DUAL":
        size = None
        if dual_shape == DualShape.FIXED:
            if dual_height % 4 or dual_width % 4:
                raise ValueError(
                    "DUAL's TensorRT height and width must be multiples of 4."
                )
            size = (dual_height, dual_width)
        key = ("DUAL", model.dtype, size)
        kept = _kept_export(model, context.node_id, key)
        if kept is not None:
            return kept.result
        onnx_bytes = dual.export_onnx(
            full_precision_source(model, "Convert To ONNX").model.state_dict(),
            model.tags,
            model.scale,
            size,
            context=context,
        )
        onnx_model = load_onnx_model(onnx_bytes)
        assert onnx_model.sub_type == "Generic"
        result = onnx_model, MIXED, "opset20"
        _keep_export(model, context.node_id, key, _Export(result, verified=False))
        return result

    if model.architecture.id == "RDUAL":
        return _rdual_onnx(context, model, dual_shape, (dual_height, dual_width))

    if drft_export.is_legacy_drft(model):
        key = ("DRFT", model.dtype)
        kept = _kept_export(model, context.node_id, key)
        if kept is not None:
            return kept.result
        onnx_model = load_onnx_model(
            drft_export.export_onnx(
                full_precision_source(model, "Convert To ONNX"), model.device
            )
        )
        assert onnx_model.sub_type == "Generic"
        result = onnx_model, MIXED, "opset20"
        _keep_export(model, context.node_id, key, _Export(result, verified=False))
        return result

    assert is_onnx_supported(model), (
        f"{model.architecture.name} is not supported for ONNX conversion at this time."
    )

    # One automatic precision (FP16_ARCHS), whatever a saved chain's hidden Data Type holds and
    # whatever PyTorch's FP16 setting: export_copy converts a copy of the model, and both
    # precisions start from its FP32 weights, also what the FP16 export is checked against.
    device = get_settings(context).device
    key = (opset, device, model.dtype)
    kept = _kept_export(model, context.node_id, key)
    if kept is not None:
        # A kept export keeps its verdict; only one in FP32 can have gone without a check.
        if verify and not kept.verified:
            source = full_precision_source(model, "Convert To ONNX")
            verify_models(
                source, kept.result[0].bytes, False, device, context.check_aborted
            )
            kept.verified = True
        for warning in kept.warnings:
            context.warn(warning)
        return kept.result

    arch = model.architecture.name
    source = full_precision_source(model, "Convert To ONNX")
    warnings: list[str] = []
    onnx_model_bytes = None
    if converts_in_fp16(arch, model.tags):
        try:
            onnx_model_bytes = export_verified(
                source, device, True, opset.value, True, context.check_aborted
            )
        except Aborted:
            raise
        except Exception as e:
            logger.warning(
                "Convert To ONNX: FP16 does not hold for this %s model",
                arch,
                exc_info=True,
            )
            warnings.append(
                f"This {arch} model converted in FP32, as its FP16 conversion does not"
                f" hold ({type(e).__name__}: {e})."
            )
    fp16 = onnx_model_bytes is not None
    if onnx_model_bytes is None:
        context.check_aborted()
        onnx_model_bytes = export_verified(
            source, device, False, opset.value, verify, context.check_aborted
        )
        if not verify:
            # Verify refuses a wrong file. Without it, the file still must not look fine.
            try:
                check_fp32_export(
                    source, onnx_model_bytes, device, context.check_aborted
                )
            except Aborted:
                raise
            except ExportMismatch as e:
                logger.warning(
                    "Convert To ONNX: the FP32 export of this %s model does not match it",
                    arch,
                    exc_info=True,
                )
                warnings.append(
                    f"The ONNX of this {arch} model does not match it in PyTorch, and may"
                    f" give wrong images ({e})."
                )
            except Exception as e:
                logger.warning(
                    "Convert To ONNX: the FP32 export of this %s model could not be"
                    " compared with it",
                    arch,
                    exc_info=True,
                )
                warnings.append(
                    f"The ONNX of this {arch} model could not be compared with it in"
                    " PyTorch on small generated images, so nothing says it is right"
                    f" ({type(e).__name__}: {e})."
                )
    for warning in warnings:
        context.warn(warning)
    precision = "fp16" if fp16 else "fp32"
    onnx_model = load_onnx_model(with_recorded_arch(onnx_model_bytes, arch, precision))
    assert onnx_model.sub_type == "Generic"
    result = onnx_model, precision, f"opset{opset.value}"
    _keep_export(
        model,
        context.node_id,
        key,
        _Export(result, verified=fp16 or verify, warnings=tuple(warnings)),
    )
    return result


def _rdual_onnx(
    context: NodeContext,
    model: ImageModelDescriptor,
    shape: DualShape,
    size: tuple[int, int],
) -> tuple[OnnxGeneric, str, str]:
    """RDUAL's ONNX bundle (rdual_bundle.py) of rdual_export.py's parts for frames of
    `size` (height, width); the flow part is traced on the settings' GPU if there is one."""
    # The model's spandrel holds RDUAL, which rdual_export imports.
    from nodes.impl.tensorrt import rdual, rdual_export

    if shape != DualShape.FIXED:
        raise ValueError(
            "RDUAL converts to ONNX for one frame size: set TensorRT Shape to Fixed with"
            " the video's height and width."
        )
    key = ("RDUAL", model.dtype, size)
    kept = _kept_export(model, context.node_id, key)
    if kept is not None:
        return kept.result
    settings = get_settings(context)
    gpu_index = settings.gpu_index if settings.device.type == "cuda" else None
    with tempfile.TemporaryDirectory(prefix="chaiNNer-rdual-onnx-") as temporary:
        folder = Path(temporary)
        rdual.export(
            full_precision_source(model, "Convert To ONNX"),
            *size,
            folder,
            gpu_index,
            context,
        )
        onnx_model = load_onnx_model(rdual_bundle.pack(folder, rdual_export.OPSET))
    assert onnx_model.sub_type == "Generic"
    result = onnx_model, MIXED, "opset20"
    _keep_export(model, context.node_id, key, _Export(result, verified=False))
    return result


def no_stop() -> None:
    """A check for Stop where no node runs."""


def export_verified(
    model: ImageModelDescriptor,
    device: torch.device,
    fp16: bool,
    opset: int,
    verify: bool,
    check_aborted: Callable[[], None] = no_stop,
) -> bytes:
    """The ONNX export of a copy of `model`; with `verify`, checked against `model`
    (verify_models: in FP32 for an FP16 export)."""
    export_model = export_copy(model, device, use_half=fp16)
    onnx_model_bytes = export_onnx(export_model, opset_version=opset)
    if verify:
        verify_models(model, onnx_model_bytes, fp16, device, check_aborted)
    return onnx_model_bytes


# The check's limit for an FP16 export, against the FP32 model on a smooth image. FP16
# PyTorch is no reference: it and ONNX Runtime round at different places, some models
# amplify the rounding (trained SPANs differ by 0.2 on noise at full amplitude), and
# OmniSR's FP16 PyTorch gives NaN where its export does not. Against FP32 a correct
# export stays within 0.004 on five trained SPANs and, on five trained OmniSRs over 40
# seeds, within a root mean square of 0.001 with single samples up to 0.017, on a smooth
# image; a trained ESRGAN reads 0.0011 with one sample at 0.0295. A wrong one (a size
# frozen in the trace, a sum that overflows FP16, one tensor halved) has a root mean
# square of 0.016 or more, so the check limits the root mean square only: a deep model
# carries a rounding into a few samples, and the largest difference tells nothing apart.
# One 8-bit step is 0.0039.
FP16_VERIFY_RMS = 0.005
# The check's limit on the images with detail (detail_inputs), where FP16 parts from FP32
# as it does on real images. A model the smooth images pass can fail here: drawn images
# push some architectures past FP16's range. On these images on the CPU, trained weights of
# the FP16 architectures read 0.0014 or less, and those of architectures that do not hold
# read 0.041 or more where the CPU shows it at all. It guards against one odd weight.
FP16_DETAIL_RMS = 0.015
# What an FP32 export's output may differ from the model's on the same images, when Verify
# is off: correct exports read 0.0003 or less, SeemoRe's 0.0104.
FP32_MATCH_RMS = 0.002
# Verify's limits for any other export, against the model on noise. A correct export
# differs by rounding: a root mean square of 0.0003 or less on eleven trained models of
# ten architectures. A trained DITN amplifies rounding in single samples, up to 0.038 over
# 100 seeds, as PyTorch does against itself (FP32 and FP64 differ by up to 0.0087). A
# wrong one, a trained RealPLKSR with DySample's grid frozen in the trace, has a root
# mean square of 0.036 or more; other sizes frozen in a trace give another shape or fail.
VERIFY_RMS = 0.001
VERIFY_MAX = 0.1


def verify_inputs(model: ImageModelDescriptor, fp16: bool) -> list[torch.Tensor]:
    """Verify's inputs, float32 on the model's device: two sizes, since a size requirement
    can pad one to the export's example, the one size where a value the trace froze in
    the graph is right; smooth as images are for an FP16 export (see FP16_VERIFY_RMS),
    with flat black, flat white and a gradient at the smaller size, where trained HAT, DAT
    and Swin2SR models write past FP16's range and go NaN; noise for any other."""
    inputs: list[torch.Tensor] = []
    channels, device = model.input_channels, model.device
    generator = torch.Generator().manual_seed(0)
    for size in (16, 40):
        dummy_size = size + model.size_requirements.get_padding(size, size)[0]
        if fp16:
            side = dummy_size // 8
            coarse = torch.rand(1, channels, side, side, generator=generator).to(device)
            inputs.append(
                torch.nn.functional.interpolate(
                    coarse, size=dummy_size, mode="bilinear", align_corners=False
                )
            )
        else:
            inputs.append(
                torch.rand(1, channels, dummy_size, dummy_size, generator=generator).to(
                    device
                )
            )
    if fp16:
        ramp = torch.linspace(0, 1, inputs[0].shape[-1], device=device)
        inputs += [
            torch.zeros_like(inputs[0]),
            torch.ones_like(inputs[0]),
            ramp.expand_as(inputs[0]).clone(),
        ]
    return inputs


def detail_inputs(model: ImageModelDescriptor) -> list[torch.Tensor]:
    """The FP16 check's images with detail, float32 on the model's device and the same in
    every run: black strokes on near white beside a flat colour, with hard edges, as drawn
    images have them; and shapes with shading and fine texture, as photographs have them."""
    channels, device = model.input_channels, model.device
    size = 64 + model.size_requirements.get_padding(64, 64)[0]
    generator = torch.Generator().manual_seed(0)
    y, x = torch.meshgrid(
        torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij"
    )

    def colour(*values: float) -> torch.Tensor:
        planes = [*values, *[1.0] * channels][:channels]
        return torch.tensor(planes).view(channels, 1, 1)

    drawn = torch.where(
        (x > 0.62) & (y < 0.42),
        colour(0.76, 0.58, 0.12) * (0.7 + 0.3 * x),
        colour(0.98, 0.97, 0.87).expand(channels, size, size),
    )
    thick = (y - 0.55 - 0.22 * torch.sin(9 * x)).abs() < 0.11
    thin = (y - 0.18 - 0.08 * torch.sin(17 * x)).abs() < 0.02
    drawn = torch.where(thick | thin, torch.zeros_like(drawn), drawn)

    def layer(cell: int, planes: int, hard: bool) -> torch.Tensor:
        side = size // cell + 2
        coarse = torch.rand(1, planes, side, side, generator=generator)
        if hard:
            return torch.nn.functional.interpolate(coarse, size=size, mode="nearest")
        return torch.nn.functional.interpolate(
            coarse, size=size, mode="bilinear", align_corners=False
        )

    grey = (
        0.5 * layer(8, 1, True)
        + 0.25 * layer(4, 1, False)
        + 0.15 * layer(2, 1, False)
        + 0.1 * layer(1, 1, True)
    )
    tint = 0.6 * layer(8, channels, True) + 0.4 * layer(3, channels, False)
    shot = 0.8 * grey + 0.2 * tint
    return [drawn[None].to(device), shot.to(device)]


# The operators whose results need no holding (with_fp16_held): those that only move values
# or work on shapes and masks, and activations, which cannot grow what the node before
# them wrote. Each hold costs ONNX Runtime's session time.
NEEDS_NO_HOLD = frozenset(
    {
        "And",
        "Cast",
        "Clip",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "DepthToSpace",
        "Equal",
        "Expand",
        "Flatten",
        "Gather",
        "Gelu",
        "Greater",
        "Identity",
        "LeakyRelu",
        "Less",
        "Not",
        "Or",
        "PRelu",
        "Pad",
        "Range",
        "Relu",
        "Reshape",
        "Shape",
        "Sigmoid",
        "Size",
        "Slice",
        "SpaceToDepth",
        "Split",
        "Squeeze",
        "Tanh",
        "Tile",
        "Transpose",
        "Trilu",
        "Unsqueeze",
        "Where",
    }
)


def with_fp16_held(onnx_model: bytes) -> bytes:
    """An FP16 ONNX that ONNX Runtime's CPU keeps in FP16 between nodes, as a GPU does. Having
    no FP16 kernels for arithmetic, it casts around each such node and drops the casts
    between two of them, so that a chain of them runs in FP32 and an overflow inside it
    does not show. A Squeeze of an Unsqueeze after each node keeps those casts (an
    Identity, Reshape, Transpose or Concat does not)."""
    model = onnx.load_model_from_string(onnx_model)
    graph = model.graph
    axis = "hold_axis"
    graph.initializer.append(
        onnx.numpy_helper.from_array(np.array([0], np.int64), axis)
    )
    outputs = {output.name for output in graph.output}
    nodes = list(graph.node)
    del graph.node[:]
    for node in nodes:
        held = [
            (index, name)
            for index, name in enumerate(node.output)
            if name and name not in outputs and node.op_type not in NEEDS_NO_HOLD
        ]
        for index, name in held:
            node.output[index] = f"{name}_raw"
        graph.node.append(node)
        for _, name in held:
            graph.node.extend(
                [
                    onnx.helper.make_node(
                        "Unsqueeze", [f"{name}_raw", axis], [f"{name}_wide"]
                    ),
                    onnx.helper.make_node("Squeeze", [f"{name}_wide", axis], [name]),
                ]
            )
    return model.SerializeToString()


class ExportMismatch(AssertionError):
    """An export ran and gave what its model does not."""


def fp32_reference(
    model: ImageModelDescriptor, device: torch.device
) -> ImageModelDescriptor:
    """`model` in FP32 on the export's device, as a copy if it is not there already."""
    if model.dtype == torch.float32 and model.device == device:
        return model
    return export_copy(model, device)


def compare_on_images(
    onnx_model: bytes,
    reference: ImageModelDescriptor,
    checks: list[tuple[torch.Tensor, float, float]],
    check_aborted: Callable[[], None] = no_stop,
) -> None:
    """Runs the ONNX on the CPU against `reference` on each image, within its limits on the
    root mean square and the largest difference, without NaN or Inf where `reference` has
    none (a model that fails in FP32 too says nothing about its export). Stop is looked at
    before the session, which can take a minute, and before each image."""
    import onnxruntime as ort

    check_aborted()
    session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_dtype = (
        np.float16 if session.get_inputs()[0].type == "tensor(float16)" else np.float32
    )
    for image, rms_limit, largest_limit in checks:
        check_aborted()
        with torch.no_grad():
            # On a copy: a model can write to its input, as RCAN does.
            torch_out = reference(image.to(reference.device, reference.dtype).clone())
        torch_out = torch_out.detach().cpu().numpy().astype(np.float64)
        onnx_out = np.asarray(
            session.run(
                [output_name],
                {input_name: image.detach().cpu().numpy().astype(onnx_dtype)},
            )[0]
        ).astype(np.float64)
        if torch_out.shape != onnx_out.shape:
            raise ExportMismatch(
                f"The ONNX model gives {onnx_out.shape}, PyTorch {torch_out.shape}."
            )
        if not np.isfinite(torch_out).all():
            continue
        if not np.isfinite(onnx_out).all():
            raise ExportMismatch(
                "The ONNX model gives NaN or infinite values on a generated"
                f" {image.shape[-1]} px image."
            )
        difference = np.abs(onnx_out - torch_out)
        rms = float(np.sqrt(np.mean(difference * difference)))
        largest = float(difference.max())
        if rms > rms_limit or largest > largest_limit:
            limit = "" if math.isinf(largest_limit) else f" (limit {largest_limit})"
            raise ExportMismatch(
                "The ONNX model's output differs from the FP32 model's on a generated"
                f" {image.shape[-1]} px image: root mean square {rms:.4f} (limit"
                f" {rms_limit}), largest {largest:.4f}{limit}."
            )


def fp32_match_checks(
    reference: ImageModelDescriptor,
) -> list[tuple[torch.Tensor, float, float]]:
    """The FP16 check's images for an FP32 export: within FP32_MATCH_RMS, far above what a
    correct export differs by."""
    images = [*verify_inputs(reference, True), *detail_inputs(reference)]
    return [(image, FP32_MATCH_RMS, math.inf) for image in images]


def check_fp32_export(
    model: ImageModelDescriptor,
    onnx_model: bytes,
    device: torch.device,
    check_aborted: Callable[[], None] = no_stop,
) -> None:
    """An FP32 export against `model` on the FP16 check's images."""
    reference = fp32_reference(model, device)
    compare_on_images(
        onnx_model, reference, fp32_match_checks(reference), check_aborted
    )


def verify_models(
    model: ImageModelDescriptor,
    onnx_model: bytes,
    fp16: bool,
    device: torch.device,
    check_aborted: Callable[[], None] = no_stop,
) -> None:
    """Runs the ONNX on the CPU against `model` in FP32 on `device`: an FP16 export, kept in
    FP16 between its nodes, on verify_inputs within FP16_VERIFY_RMS and on detail_inputs
    within FP16_DETAIL_RMS; any other export on verify_inputs within
    VERIFY_RMS and VERIFY_MAX, and on fp32_match_checks; each without NaN or Inf.

    This is a cheap guard on the CPU, not a proof for a GPU: a GPU's FP16 kernels round in
    places of their own."""
    # based on: https://github.com/muslll/neosr/blob/d871e468fa9e82dc874f65b2289ad5726a5ab3b9/convert.py#L61
    reference = fp32_reference(model, device)
    if not fp16:
        compare_on_images(
            onnx_model,
            reference,
            [
                *(
                    (image, VERIFY_RMS, VERIFY_MAX)
                    for image in verify_inputs(reference, False)
                ),
                *fp32_match_checks(reference),
            ],
            check_aborted,
        )
        return
    compare_on_images(
        with_fp16_held(onnx_model),
        reference,
        [
            *(
                (image, FP16_VERIFY_RMS, math.inf)
                for image in verify_inputs(reference, True)
            ),
            *((image, FP16_DETAIL_RMS, math.inf) for image in detail_inputs(reference)),
        ],
        check_aborted,
    )
