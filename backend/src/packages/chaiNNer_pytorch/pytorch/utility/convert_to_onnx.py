from __future__ import annotations

import hashlib
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from weakref import WeakKeyDictionary

import numpy as np
import torch
from spandrel import ImageModelDescriptor

from api import NodeContext
from nodes.groups import Condition, if_group
from nodes.impl.image_utils import as_target_channels
from nodes.impl.onnx.autocast import LowPrecision, autocast
from nodes.impl.onnx.load import load_onnx_model
from nodes.impl.onnx.model import MIXED, OnnxGeneric
from nodes.impl.pytorch.convert_to_onnx_impl import (
    export_copy,
    export_onnx,
    is_onnx_supported,
)
from nodes.impl.tensorrt import drft_export, dual
from nodes.properties.inputs import (
    BoolInput,
    EnumInput,
    ImageInput,
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
    MIXED_FP16 = 2
    MIXED_BF16 = 3


DATA_TYPE_LABELS: dict[OnnxDataType, str] = {
    OnnxDataType.FP32: "fp32",
    OnnxDataType.FP16: "fp16",
    OnnxDataType.MIXED_FP16: "Mixed FP16 (ModelOpt)",
    OnnxDataType.MIXED_BF16: "Mixed BF16 (ModelOpt)",
}
LOW_PRECISION: dict[OnnxDataType, LowPrecision] = {
    OnnxDataType.MIXED_FP16: "fp16",
    OnnxDataType.MIXED_BF16: "bf16",
}
# AutoCast calibrates on at most this many pixels per side (a centre crop).
CALIBRATION_SIZE = 256


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


# A model's latest export while it lives, with everything it depends on: a re-run
# returns the same ONNX model (with the sessions kept for it) without exporting again.
# One per model: a superseded export's sessions would hold GPU memory.
_exports: WeakKeyDictionary[ImageModelDescriptor, tuple[Hashable, _Export]] = (
    WeakKeyDictionary()
)
_exports_lock = Lock()


def _kept_export(model: ImageModelDescriptor, key: Hashable) -> _Export | None:
    with _exports_lock:
        kept = _exports.get(model)
    return kept[1] if kept is not None and kept[0] == key else None


def _keep_export(model: ImageModelDescriptor, key: Hashable, export: _Export) -> None:
    with _exports_lock:
        _exports[model] = key, export


IS_DUAL = Condition.type(0, 'PyTorchModel { arch: "DUAL" }')
# Shown until the model is known to be DUAL
NOT_DUAL = Condition.type(
    0, 'PyTorchModel { arch: invStrSet("DUAL") }', if_not_connected=True
)


@utility_group.register(
    schema_id="chainner:pytorch:convert_to_onnx",
    name="Convert To ONNX",
    description=[
        "Convert a PyTorch model to ONNX. Note: fp16 conversion will only work if PyTorch fp16 mode is turned on.",
        "It is recommended to save converted models as a separate step, then load the converted models instead of converting them every time you run the chain.",
        "Note: Converted models are not guaranteed to work with other programs that support ONNX models. This is for a variety of reasons and cannot be changed.",
    ],
    icon="ONNX",
    inputs=[
        SrModelInput("PyTorch Model"),
        if_group(NOT_DUAL)(
            EnumInput(
                OnnxDataType,
                label="Data Type",
                default=OnnxDataType.FP32,
                option_labels=DATA_TYPE_LABELS,
            ).with_docs(
                "Mixed FP16 and Mixed BF16 export in FP32, then NVIDIA ModelOpt's AutoCast"
                " converts each node to FP16 or BF16 unless its values on the calibration"
                " image, or its weights, would overflow or lose too much there; those"
                " nodes stay FP32, as do the input and output. TensorRT builds these"
                " models in their own precision (Build Engine's Precision does not apply)."
                " Needs the NVIDIA ModelOpt package.",
                "With PyTorch's FP16 setting on, Load Model keeps a model in FP16; FP32 and"
                " the mixed types read its file again for its FP32 weights (if the file"
                " changed since, they export the FP16 weights, and the log says so).",
                "A legacy DRFT model (LayerNorm, not i-LN) converts to a TensorRT-only"
                " ONNX with chaiNNer-C's DRFT TensorRT plugins instead, which ONNX Runtime"
                " cannot run: its precision is fixed and mixed (FP16 inside, FP32 input and"
                " output), its opset is 20, it takes every input size from"
                " 64 px (twice the window for larger windows), and Data Type, Opset and"
                " Verify do not apply to it.",
                hint=True,
            ),
            EnumInput(
                Opset,
                label="Opset",
                default=Opset.OPSET_20,
                option_labels=OPSET_LABELS,
            ).with_docs(
                "Opset 17 and newer export LayerNorm as one ONNX operator, and 20 and newer"
                " GELU too, which TensorRT and ONNX Runtime run as single layers. Older"
                " opsets suit older ONNX programs.",
                hint=True,
            ),
            BoolInput("Verify", default=False).with_docs(
                "Runs the ONNX and PyTorch models to verify that they produce the same output. It's recommended to keep this on to ensure the conversion is correct.",
                "Verification requires ONNX to be installed.",
                "A mixed-precision model is verified as its FP32 export, before AutoCast.",
                hint=True,
            ),
            if_group(
                Condition.enum(1, (OnnxDataType.MIXED_FP16, OnnxDataType.MIXED_BF16))
            )(
                ImageInput("Calibration Image")
                .make_optional()
                .with_id(7)
                .with_docs(
                    "An image like those the model will upscale, for AutoCast to measure"
                    " each node's values on (a centre crop of at most"
                    f" {CALIBRATION_SIZE}x{CALIBRATION_SIZE}). Without one, it uses"
                    " random noise.",
                    hint=True,
                ),
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
                hint=True,
            ),
        ),
        if_group(IS_DUAL & Condition.enum(6, DualShape.FIXED))(
            NumberInput("TensorRT Height", default=512, min=4, step=4, unit="px")
            .with_id(4)
            .with_docs("The fixed input height (a multiple of 4)."),
            NumberInput("TensorRT Width", default=512, min=4, step=4, unit="px")
            .with_id(5)
            .with_docs("The fixed input width (a multiple of 4)."),
        ),
    ],
    outputs=[
        OnnxModelOutput(
            model_type="OnnxGenericModel & pytorchToOnnx(Input0) & OnnxModel { arch: Input0.arch }",
            label="ONNX Model",
        ),
        TextOutput(
            "FP Mode",
            """
                match Input0.arch {
                    "DUAL" => "mixed",
                    _ => match Input1 {
                        OnnxDataType::Fp32 => "fp32",
                        OnnxDataType::Fp16 => "fp16",
                        OnnxDataType::MixedFp16 => "mixed-fp16",
                        OnnxDataType::MixedBf16 => "mixed-bf16",
                    },
                }
            """,
        ),
        TextOutput(
            "Opset",
            """
                let opset = Input2;
                match Input0.arch {
                    "DUAL" => "opset20",
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
    calibration_image: np.ndarray | None,
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
        kept = _kept_export(model, key)
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
        _keep_export(model, key, _Export(result, verified=False))
        return result

    if drft_export.is_legacy_drft(model):
        key = ("DRFT", model.dtype)
        kept = _kept_export(model, key)
        if kept is not None:
            return kept.result
        onnx_model = load_onnx_model(
            drft_export.export_onnx(full_precision_source(model, "Convert To ONNX"))
        )
        assert onnx_model.sub_type == "Generic"
        result = onnx_model, MIXED, "opset20"
        _keep_export(model, key, _Export(result, verified=False))
        return result

    assert is_onnx_supported(model), (
        f"{model.architecture} is not supported for ONNX conversion at this time."
    )

    fp16 = data_type == OnnxDataType.FP16
    exec_options = get_settings(context)
    device = exec_options.device
    if fp16:
        if not exec_options.use_fp16:
            raise ValueError(
                "PyTorch fp16 mode must be supported and turned on in settings to convert model as fp16."
            )
        if not model.supports_half:
            raise ValueError(
                f"This {model.architecture.name} model does not support FP16. Please convert as FP32."
            )

    low = LOW_PRECISION.get(data_type)
    mixed = None if low is None else (low, calibration_input(model, calibration_image))
    calibration_hash = None
    if mixed is not None:
        calibration_hash = mixed[1].shape, hashlib.sha256(mixed[1]).digest()
    key = (data_type, opset, device, model.dtype, calibration_hash)
    kept = _kept_export(model, key)
    if kept is not None and (kept.verified or not verify):
        return kept.result

    source = model if fp16 else full_precision_source(model, "Convert To ONNX")
    onnx_model_bytes = export_verified(source, device, fp16, opset.value, verify)
    onnx_model = load_onnx_model(onnx_model_bytes)
    assert onnx_model.sub_type == "Generic"

    if mixed is None:
        result = onnx_model, "fp16" if fp16 else "fp32", f"opset{opset.value}"
    else:
        low, calibration = mixed
        onnx_model = load_onnx_model(autocast(onnx_model_bytes, low, calibration))
        assert onnx_model.sub_type == "Generic"
        result = onnx_model, f"mixed-{low}", f"opset{opset.value}"
    _keep_export(model, key, _Export(result, verified=verify))
    return result


def export_verified(
    model: ImageModelDescriptor,
    device: torch.device,
    fp16: bool,
    opset: int,
    verify: bool,
) -> bytes:
    """The ONNX export of a copy of `model`; with `verify`, checked against that copy."""
    export_model = export_copy(model, device, use_half=fp16)
    onnx_model_bytes = export_onnx(export_model, opset_version=opset)
    if verify:
        verify_models(export_model, onnx_model_bytes, fp16)
    return onnx_model_bytes


def calibration_input(
    model: ImageModelDescriptor, image: np.ndarray | None
) -> np.ndarray:
    """AutoCast's calibration input (NCHW float32 RGB): a centre crop of `image`, or
    seeded noise, padded to a size the model accepts."""
    if image is None:
        image = np.random.default_rng(0).random((64, 64, 3), dtype=np.float32)
    height, width = image.shape[:2]
    top = max(0, (height - CALIBRATION_SIZE) // 2)
    left = max(0, (width - CALIBRATION_SIZE) // 2)
    image = image[top : top + CALIBRATION_SIZE, left : left + CALIBRATION_SIZE]
    image = as_target_channels(image, model.input_channels, narrowing=True)
    if image.ndim == 3 and image.shape[2] >= 3:
        # chaiNNer's images are BGR(A); the model takes RGB(A).
        image = np.concatenate([image[:, :, 2::-1], image[:, :, 3:]], axis=2)
    if image.ndim == 2:
        image = image[:, :, None]
    pad_w, pad_h = model.size_requirements.get_padding(image.shape[1], image.shape[0])
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)


def verify_models(pytorch_model: ImageModelDescriptor, onnx_model: bytes, fp16: bool):
    # based on: https://github.com/muslll/neosr/blob/d871e468fa9e82dc874f65b2289ad5726a5ab3b9/convert.py#L61
    import onnxruntime as ort

    dummy_size = 16
    dummy_size += pytorch_model.size_requirements.get_padding(dummy_size, dummy_size)[0]
    dummy_input = torch.rand(
        1,
        pytorch_model.input_channels,
        dummy_size,
        dummy_size,
        device=pytorch_model.device,
        dtype=pytorch_model.dtype,
    )

    # pytorch
    torch_out = pytorch_model(dummy_input).detach().cpu().numpy()

    # onnx
    session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_out = session.run(
        [output_name],
        {input_name: dummy_input.detach().cpu().numpy()},
    )[0]

    np.testing.assert_allclose(torch_out, np.asarray(onnx_out), rtol=0.01, atol=0.001)
