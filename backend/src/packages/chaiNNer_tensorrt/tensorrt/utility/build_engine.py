from __future__ import annotations

from enum import Enum

import tensorrt as trt
from sanic.log import logger

from api import NodeContext
from nodes.groups import if_enum_group
from nodes.impl.onnx.model import OnnxModel
from nodes.impl.tensorrt import dual
from nodes.impl.tensorrt.engine_builder import BuildConfig, build_engine_from_onnx
from nodes.impl.tensorrt.memory import get_cuda_compute_capability
from nodes.impl.tensorrt.model import TensorRTEngine, TensorRTEngineInfo
from nodes.properties.inputs import (
    BoolInput,
    EnumInput,
    NumberInput,
    OnnxModelInput,
)
from nodes.properties.outputs import TensorRTEngineOutput

from ...settings import TensorRTSettings, get_settings
from .. import utility_group


class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


PRECISION_LABELS = {
    Precision.FP32: "FP32 (Higher Precision)",
    Precision.FP16: "FP16 (Faster)",
    Precision.BF16: "BF16 (Balanced)",
}


class ShapeMode(Enum):
    FIXED = "fixed"
    DYNAMIC = "dynamic"


SHAPE_MODE_LABELS = {
    ShapeMode.FIXED: "Fixed (Single Size)",
    ShapeMode.DYNAMIC: "Dynamic (Variable Sizes)",
}


if utility_group is not None:

    @utility_group.register(
        schema_id="chainner:tensorrt:build_engine",
        name="Build Engine",
        description=[
            "Convert an ONNX model to a TensorRT engine.",
            "Building an engine can take several minutes depending on the model size and optimization settings.",
            "The built engine is optimized specifically for your GPU and TensorRT version.",
            "It is recommended to save the built engine for reuse, as building is slow.",
            "A DUAL ONNX from Convert To ONNX builds in its own precision with its TensorRT plugins (Triton kernels compiled for your GPU; no C++ compiler or CUDA Toolkit is needed). A dynamic DUAL ONNX takes this node's shape inputs, multiples of 64 px (e.g. Dynamic, 64x64 to 1920x1088 for whole 1080p frames); a fixed one keeps its own size. Upscale Image replays DUAL engines from a CUDA graph per input size.",
        ],
        icon="BsNvidia",
        inputs=[
            OnnxModelInput("ONNX Model"),
            EnumInput(
                Precision,
                label="Precision",
                default=Precision.FP16,
                option_labels=PRECISION_LABELS,
            ).with_docs(
                "FP16: lower precision but faster and uses less memory, especially on RTX GPUs. FP16 also does not work with certain models.",
                "BF16: same exponent range as FP32 with reduced mantissa. Better numerical stability than FP16 while still being faster than FP32. Good for models that produce NaN/artifacts with FP16.",
                "FP32: higher precision but slower. Use especially if FP16 fails.",
            ),
            EnumInput(
                ShapeMode,
                label="Shape Mode",
                default=ShapeMode.FIXED,
                option_labels=SHAPE_MODE_LABELS,
            ).with_docs(
                "Fixed: Build engine for a single input size. Fastest inference.",
                "Dynamic: Build engine for variable input sizes. More flexible but slightly slower.",
            ),
            if_enum_group(2, ShapeMode.DYNAMIC)(
                NumberInput(
                    "Min Height",
                    default=64,
                    min=16,
                    max=4096,
                    unit="px",
                ).with_docs("Minimum input height for dynamic shapes."),
                NumberInput(
                    "Min Width",
                    default=64,
                    min=16,
                    max=4096,
                    unit="px",
                ).with_docs("Minimum input width for dynamic shapes."),
                NumberInput(
                    "Optimal Height",
                    default=512,
                    min=16,
                    max=4096,
                    unit="px",
                ).with_docs("Optimal input height (used for optimization)."),
                NumberInput(
                    "Optimal Width",
                    default=512,
                    min=16,
                    max=4096,
                    unit="px",
                ).with_docs("Optimal input width (used for optimization)."),
                NumberInput(
                    "Max Height",
                    default=2048,
                    min=16,
                    max=8192,
                    unit="px",
                ).with_docs("Maximum input height for dynamic shapes."),
                NumberInput(
                    "Max Width",
                    default=2048,
                    min=16,
                    max=8192,
                    unit="px",
                ).with_docs("Maximum input width for dynamic shapes."),
            ),
            if_enum_group(2, ShapeMode.FIXED)(
                NumberInput(
                    "Height",
                    default=256,
                    min=16,
                    max=8192,
                    unit="px",
                ).with_docs("Fixed input height."),
                NumberInput(
                    "Width",
                    default=256,
                    min=16,
                    max=8192,
                    unit="px",
                ).with_docs("Fixed input width."),
            ),
            NumberInput(
                "Workspace (GB)",
                default=4.0,
                min=1.0,
                max=32.0,
                precision=1,
                step=0.5,
            ).with_docs(
                "Maximum GPU memory for building. Larger values may allow better optimizations.",
                hint=True,
            ),
            BoolInput("Allow TF32", default=False).with_docs(
                "Lets TensorRT run the convolutions and matrix products of FP32 layers in TF32 on"
                " tensor cores: FP32's range, but inputs rounded to a 10-bit mantissa, about FP16's"
                " precision. Faster on RTX 30-series and newer GPUs.",
                "Off keeps FP32 layers fully FP32. It applies to every FP32 layer, including those"
                " of mixed-precision models.",
            ),
        ],
        outputs=[
            TensorRTEngineOutput(),
        ],
        node_context=True,
    )
    def build_engine_node(
        context: NodeContext,
        onnx_model: OnnxModel,
        precision: Precision,
        shape_mode: ShapeMode,
        min_height: int,
        min_width: int,
        opt_height: int,
        opt_width: int,
        max_height: int,
        max_width: int,
        static_height: int,
        static_width: int,
        workspace: float,
        allow_tf32: bool,
    ) -> TensorRTEngine:
        settings = get_settings(context)
        gpu_index = settings.gpu_index

        use_dynamic = shape_mode == ShapeMode.DYNAMIC
        if not use_dynamic:
            min_height = opt_height = max_height = static_height
            min_width = opt_width = max_width = static_width

        manifest = dual.export_manifest(onnx_model.bytes)
        if manifest is not None:
            profile = (
                (min_height, min_width),
                (opt_height, opt_width),
                (max_height, max_width),
            )
            return build_dual_engine(onnx_model, manifest, settings, profile)

        # Determine timing cache path
        timing_cache_path = None
        if settings.timing_cache_path:
            import hashlib

            # Create a cache key based on the model
            model_hash = hashlib.md5(onnx_model.bytes[:1024]).hexdigest()[:8]
            timing_cache_path = (
                f"{settings.timing_cache_path}/timing_{model_hash}.cache"
            )

        config = BuildConfig(
            precision=precision.value,
            workspace_size_gb=workspace,
            min_shape=(min_height, min_width),
            opt_shape=(opt_height, opt_width),
            max_shape=(max_height, max_width),
            use_dynamic_shapes=use_dynamic,
            allow_tf32=allow_tf32,
        )

        logger.info(
            "Building TensorRT engine: precision=%s, dynamic=%s, workspace=%.1fGB, tf32=%s",
            precision.value,
            use_dynamic,
            workspace,
            allow_tf32,
        )

        engine = build_engine_from_onnx(
            onnx_model.bytes,
            config,
            gpu_index=gpu_index,
            timing_cache_path=timing_cache_path,
        )

        return engine


def build_dual_engine(
    onnx_model: OnnxModel,
    manifest: dict,
    settings: TensorRTSettings,
    profile: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> TensorRTEngine:
    """DUAL's specialized engine, in the ONNX's precision. A dynamic DUAL ONNX takes the
    node's shape inputs as its (height, width) profile; a fixed one has its own size.
    The node's precision, workspace and TF32 inputs do not apply."""
    major, minor = get_cuda_compute_capability(settings.gpu_index)
    spec = manifest["specialization"]
    if spec.get("dynamic", False):
        dual.check_profile(manifest, *profile)
    else:
        size = (spec["height"], spec["width"])
        profile = (size, size, size)
    low, high = profile[0], profile[2]
    logger.info(
        "Building a DUAL engine (%s x%d, %dx%d to %dx%d, plugins %s, SM %d%d); the"
        " node's precision, workspace and TF32 inputs do not apply",
        spec["factory"],
        spec["scale"],
        low[1],
        low[0],
        high[1],
        high[0],
        spec["plugin_key"],
        major,
        minor,
    )
    engine_bytes = dual.build_engine(
        onnx_model.bytes,
        manifest,
        settings.gpu_index,
        profile if spec.get("dynamic", False) else None,
    )
    min_shape, opt_shape, max_shape = ((1, 3, h, w) for h, w in profile)
    info = TensorRTEngineInfo(
        precision="bf16",
        input_channels=3,
        output_channels=3,
        scale=spec["scale"],
        gpu_architecture=f"sm_{major}{minor}",
        tensorrt_version=trt.__version__,
        has_dynamic_shapes=min_shape != max_shape,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
    )
    return TensorRTEngine(engine_bytes, info)
