from __future__ import annotations

from enum import Enum

import numpy as np
import torch
from spandrel import ImageModelDescriptor

from api import NodeContext
from nodes.groups import Condition, if_group
from nodes.impl.onnx.load import load_onnx_model
from nodes.impl.onnx.model import MIXED, OnnxGeneric
from nodes.impl.pytorch.convert_to_onnx_impl import (
    convert_to_onnx_impl,
    is_onnx_supported,
)
from nodes.impl.tensorrt import dual
from nodes.properties.inputs import (
    BoolInput,
    EnumInput,
    NumberInput,
    OnnxFpDropdown,
    SrModelInput,
)
from nodes.properties.outputs import OnnxModelOutput, TextOutput

from ...settings import get_settings
from .. import utility_group


class Opset(Enum):
    OPSET_14 = 14
    OPSET_15 = 15
    OPSET_16 = 16
    OPSET_17 = 17


OPSET_LABELS: dict[Opset, str] = {
    Opset.OPSET_14: "14",
    Opset.OPSET_15: "15",
    Opset.OPSET_16: "16",
    Opset.OPSET_17: "17",
}


class DualShape(Enum):
    DYNAMIC = "dynamic"
    FIXED = "fixed"


DUAL_SHAPE_LABELS: dict[DualShape, str] = {
    DualShape.DYNAMIC: "Dynamic (any size)",
    DualShape.FIXED: "Fixed (one size)",
}

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
            OnnxFpDropdown(),
            EnumInput(
                Opset,
                label="Opset",
                default=Opset.OPSET_14,
                option_labels=OPSET_LABELS,
            ),
            BoolInput("Verify", default=False).with_docs(
                "Runs the ONNX and PyTorch models to verify that they produce the same output. It's recommended to keep this on to ensure the conversion is correct.",
                "Verification requires ONNX to be installed.",
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
                    _ => FpMode::toString(Input1),
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
    is_fp16: int,
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
        onnx_bytes = dual.export_onnx(
            model.model.state_dict(), model.tags, model.scale, size
        )
        onnx_model = load_onnx_model(onnx_bytes)
        assert onnx_model.sub_type == "Generic"
        return onnx_model, MIXED, "opset20"

    assert is_onnx_supported(model), (
        f"{model.architecture} is not supported for ONNX conversion at this time."
    )

    fp16 = bool(is_fp16)
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

    model.eval().to(device)

    onnx_model_bytes = convert_to_onnx_impl(
        model,
        device,
        use_half=fp16,
        opset_version=opset.value,
    )
    onnx_model = load_onnx_model(onnx_model_bytes)
    assert onnx_model.sub_type == "Generic"

    if verify:
        verify_models(model, onnx_model_bytes, fp16)

    fp_mode = "fp16" if fp16 else "fp32"

    return onnx_model, fp_mode, f"opset{opset.value}"


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
