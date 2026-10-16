from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from api import NodeContext
from nodes.groups import if_enum_group
from nodes.impl.upscale.auto_split_tiles import CUSTOM, TILE_SIZE_256, TileSize
from nodes.impl.vapoursynth.script import VideoClip, clip_var, one_clip
from nodes.properties.inputs import (
    EnumInput,
    FileInput,
    NumberInput,
    TileSizeDropdown,
    VideoClipInput,
)
from nodes.properties.outputs import VideoClipOutput

from . import functions_group

# The script imports chaiNNer's bridge from backend/src (the prologue's sys.path line).
BRIDGE_IMPORT = "from nodes.impl.vapoursynth import chainner_vs"


class AiUpscaleBackend(Enum):
    PYTORCH = "PyTorch"
    ONNX = "ONNX"
    TENSORRT = "TensorRT"


class AiUpscalePrecision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"


@functions_group.register(
    schema_id="chainner:vapoursynth:ai_upscale",
    name="AI Upscale",
    description=[
        (
            "Upscales or restores every frame of a clip with an AI model, inside the"
            " VapourSynth script, using chaiNNer's own PyTorch, ONNX and TensorRT"
            " upscalers: each frame comes out as chaiNNer's Upscale Image node would"
            " make it."
        ),
        (
            "The clip must be 32-bit float RGB or gray (RGBS or GRAYS). The result has"
            " the same format at the model's scale; a gray clip goes through an RGB"
            " model as RGB and comes back as gray."
        ),
        (
            "Each frame is clipped to 0-1 before the model, as chaiNNer's images"
            " always are."
        ),
    ],
    icon="MdAutoFixHigh",
    inputs=[
        VideoClipInput().with_id(0),
        FileInput(
            label="Model",
            file_kind="pth",
            filetypes=[
                ".pth",
                ".pt",
                ".ckpt",
                ".safetensors",
                ".onnx",
                ".engine",
                ".trt",
                ".plan",
            ],
        )
        .with_id(1)
        .with_docs(
            "A PyTorch model (.pth, .pt, .ckpt, .safetensors), an ONNX model (.onnx)"
            " or a TensorRT engine (.engine, .trt, .plan). The backend must match the"
            " file."
        ),
        EnumInput(
            AiUpscaleBackend,
            "Backend",
            default=AiUpscaleBackend.PYTORCH,
            option_labels={
                AiUpscaleBackend.PYTORCH: "PyTorch",
                AiUpscaleBackend.ONNX: "ONNX",
                AiUpscaleBackend.TENSORRT: "TensorRT",
            },
            icons={
                AiUpscaleBackend.PYTORCH: "PyTorch",
                AiUpscaleBackend.ONNX: "ONNX",
                AiUpscaleBackend.TENSORRT: "BsNvidia",
            },
        )
        .with_id(2)
        .with_docs(
            "PyTorch and ONNX run on the first NVIDIA GPU when there is one, else on"
            " the CPU. TensorRT engines need an NVIDIA GPU."
        ),
        if_enum_group(2, AiUpscaleBackend.PYTORCH)(
            EnumInput(
                AiUpscalePrecision,
                "Precision",
                default=AiUpscalePrecision.FP32,
                option_labels={
                    AiUpscalePrecision.FP32: "FP32",
                    AiUpscalePrecision.FP16: "FP16",
                },
            )
            .with_id(3)
            .with_docs(
                "FP16 uses less memory and is faster on most NVIDIA GPUs. On the CPU,"
                " PyTorch runs in FP32 either way. ONNX models and TensorRT engines run"
                " in their own precision."
            ),
        ),
        if_enum_group(2, (AiUpscaleBackend.PYTORCH, AiUpscaleBackend.ONNX))(
            TileSizeDropdown()
            .with_id(4)
            .with_docs(
                "Frames larger than the tile size are upscaled in overlapping tiles"
                " that are blended back together without seams.",
                "Auto estimates the tile size from free memory for PyTorch; ONNX models"
                " use 256 px tiles on Auto. TensorRT engines tile by the engine's own"
                " shape limits.",
            ),
            if_enum_group(4, CUSTOM)(
                NumberInput(
                    "Custom Tile Size",
                    min=1,
                    max=None,
                    default=TILE_SIZE_256,
                    unit="px",
                ).with_id(5),
            ),
        ),
    ],
    outputs=[VideoClipOutput()],
    node_context=True,
)
def ai_upscale_node(
    context: NodeContext,
    clip: VideoClip,
    model: Path,
    backend: AiUpscaleBackend,
    precision: AiUpscalePrecision,
    tile_size: TileSize,
    custom_tile_size: int,
) -> VideoClip:
    var = clip_var(context.node_id)
    tile = custom_tile_size if tile_size == CUSTOM else tile_size
    call = (
        f"chainner_vs.upscale({one_clip(clip, 'Clip')}, {os.path.abspath(model)!r},"
        f" {backend.value!r}, {precision.value!r}, {int(tile)})"
    )
    return VideoClip(
        var=var,
        stmt=f"{var} = {call}",
        deps=(clip,),
        imports=frozenset({BRIDGE_IMPORT}),
        node_id=context.node_id,
    )
