from __future__ import annotations

import dataclasses
from enum import Enum

import tensorrt as trt
from sanic.log import logger

from api import NodeContext
from nodes.groups import Condition, if_enum_group, if_group
from nodes.impl.onnx.load import model_facts
from nodes.impl.onnx.model import OnnxModel
from nodes.impl.tensorrt import drft, dual
from nodes.impl.tensorrt.build_cache import (
    cached_engine,
    drft_plugin_identity,
    dual_plugin_identity,
    engine_key,
)
from nodes.impl.tensorrt.engine_builder import (
    BuildConfig,
    build_engine_from_onnx,
    timing_cache_name,
)
from nodes.impl.tensorrt.memory import (
    get_cuda_compute_capability,
    get_cuda_device_name,
)
from nodes.impl.tensorrt.model import TensorRTEngine, TensorRTEngineInfo
from nodes.impl.tensorrt.tiling import TILE_OVERLAP
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


class TilingOptimization(Enum):
    NONE = "none"
    FAST = "fast"
    MODERATE = "moderate"
    FULL = "full"


TILING_OPTIMIZATION_LABELS = {
    TilingOptimization.NONE: "None (TensorRT's default)",
    TilingOptimization.FAST: "Fast",
    TilingOptimization.MODERATE: "Moderate",
    TilingOptimization.FULL: "Full (slowest build)",
}

# The smallest engine size Upscale Image tiles a larger image with: neighbouring tiles
# overlap by 3 * TILE_OVERLAP px (tiling.tile_starts). A smaller engine would only
# upscale images up to its own size.
MIN_TILE_SIZE = 3 * TILE_OVERLAP + 1
TILE_SIZE_DOC = (
    f"At least {MIN_TILE_SIZE} px, so that Upscale Image can tile a larger image. Below"
    " 64 px, a large image takes very many tiles and upscales very slowly."
)

# A Workspace of 0 is TensorRT's default (as much as the GPU's memory), or for a DUAL
# ONNX its guide's.
DUAL_WORKSPACE_GIB = 8.0

# Shown until the ONNX is known to be DUAL's, which keeps its own precision
NOT_DUAL = Condition.type(
    0, 'OnnxModel { arch: invStrSet("DUAL") }', if_not_connected=True
)


if utility_group is not None:

    @utility_group.register(
        schema_id="chainner:tensorrt:build_engine",
        name="Build Engine",
        description=[
            "Convert an ONNX model to a TensorRT engine.",
            "Building an engine can take several minutes depending on the model size and optimization settings.",
            "The built engine is optimized specifically for your GPU and TensorRT version.",
            "It is recommended to save the built engine for reuse, as building is slow.",
            "A DUAL ONNX from Convert To ONNX builds in its own mixed precision (BF16 body, FP32 input and output) with its TensorRT plugins (Triton kernels compiled for your GPU; no C++ compiler or CUDA Toolkit is needed). A dynamic DUAL ONNX takes this node's shape inputs, multiples of 4 px (e.g. Dynamic, 64x64 to 1920x1088 for whole 1080p frames); a fixed one keeps its own size. Precision does not apply and is hidden; Workspace, Allow TF32, Sparse Weights, Optimization Level and Tiling Optimization do. Upscale Image replays DUAL engines from a CUDA graph per input size.",
            "A legacy DRFT ONNX from Convert To ONNX also builds in its own mixed precision, with chaiNNer-C's DRFT TensorRT plugins (chaiNNer-C loads them with the engine; another program needs the plugin library too). Its sizes, Fixed or Dynamic, must be multiples of the ONNX's alignment and at least its minimum size on each side (Build Engine says which if not). Precision does not apply; Workspace (0: TensorRT's default), Allow TF32, Sparse Weights, Optimization Level and Tiling Optimization do.",
        ],
        icon="BsNvidia",
        inputs=[
            OnnxModelInput("ONNX Model"),
            if_group(NOT_DUAL)(
                EnumInput(
                    Precision,
                    label="Precision",
                    default=Precision.FP16,
                    option_labels=PRECISION_LABELS,
                ).with_docs(
                    "FP16: lower precision but faster and uses less memory, especially on RTX GPUs. FP16 also does not work with certain models.",
                    "BF16: same exponent range as FP32 with reduced mantissa. Better numerical stability than FP16 while still being faster than FP32. Good for models that produce NaN/artifacts with FP16.",
                    "FP32: higher precision but slower. Use especially if FP16 fails.",
                    "A mixed-precision ONNX (Convert To ONNX's Mixed FP16 or Mixed BF16) keeps its own precision, and this setting does not apply.",
                ),
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
                    min=MIN_TILE_SIZE,
                    max=8192,
                    unit="px",
                ).with_docs(
                    "Maximum input height for dynamic shapes.",
                    TILE_SIZE_DOC,
                ),
                NumberInput(
                    "Max Width",
                    default=2048,
                    min=MIN_TILE_SIZE,
                    max=8192,
                    unit="px",
                ).with_docs(
                    "Maximum input width for dynamic shapes.",
                    TILE_SIZE_DOC,
                ),
            ),
            if_enum_group(2, ShapeMode.FIXED)(
                NumberInput(
                    "Height",
                    default=256,
                    min=MIN_TILE_SIZE,
                    max=8192,
                    unit="px",
                ).with_docs(
                    "Fixed input height.",
                    TILE_SIZE_DOC,
                ),
                NumberInput(
                    "Width",
                    default=256,
                    min=MIN_TILE_SIZE,
                    max=8192,
                    unit="px",
                ).with_docs(
                    "Fixed input width.",
                    TILE_SIZE_DOC,
                ),
            ),
            NumberInput(
                "Workspace (GiB)",
                default=0,
                min=0,
                max=None,
                precision=2,
                step=1,
            ).with_docs(
                "Maximum GPU memory TensorRT may use while building. Larger values may allow"
                " better optimizations; any amount can be typed.",
                "0 is automatic: TensorRT's default, up to all of the GPU's memory, or"
                f" {DUAL_WORKSPACE_GIB:g} GiB for a DUAL ONNX.",
                hint=True,
            ),
            BoolInput("Allow TF32", default=False).with_docs(
                "Lets TensorRT run the convolutions and matrix products of FP32 layers in TF32 on"
                " tensor cores: FP32's range, but inputs rounded to a 10-bit mantissa, about FP16's"
                " precision. Faster on RTX 30-series and newer GPUs.",
                "Off keeps FP32 layers fully FP32. It applies to every FP32 layer, including those"
                " of mixed-precision models such as DUAL's.",
            ),
            BoolInput("Sparse Weights", default=False).with_docs(
                "Lets TensorRT use 2:4 structured-sparse kernels (RTX 30-series and newer) for"
                " layers whose weights are already 2:4 sparse, as in models trained or pruned for"
                " it. A dense model gains nothing and builds slower.",
            ),
            NumberInput(
                "Optimization Level",
                default=3,
                min=0,
                max=5,
            ).with_docs(
                "TensorRT's builder optimization level. Higher levels try more kernels: the build"
                " takes longer and the engine may run faster. 3 is TensorRT's default; 5 can take"
                " much longer.",
            ),
            EnumInput(
                TilingOptimization,
                label="Tiling Optimization",
                default=TilingOptimization.NONE,
                option_labels=TILING_OPTIMIZATION_LABELS,
            ).with_docs(
                "Lets TensorRT fuse chains of layers into kernels that keep their"
                " intermediate results in the GPU's L2 cache instead of writing them to"
                " memory. This is not image tiling: it changes only how the engine runs.",
                "Gains depend on the model. On a 4x HAT-L and DAT2 from Convert To ONNX at"
                " 256x256 (Optimization Level 3), Full ran 0.5% and 1.6% faster than None"
                " and made the build 4 to 5 times longer.",
                "None keeps TensorRT's default (no tiling optimization).",
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
        sparse_weights: bool,
        optimization_level: int,
        tiling_optimization: TilingOptimization,
    ) -> TensorRTEngine:
        settings = get_settings(context)
        gpu_index = settings.gpu_index

        use_dynamic = shape_mode == ShapeMode.DYNAMIC
        if not use_dynamic:
            min_height = opt_height = max_height = static_height
            min_width = opt_width = max_width = static_width

        facts = model_facts(onnx_model)
        profile = (
            (min_height, min_width),
            (opt_height, opt_width),
            (max_height, max_width),
        )
        if facts.drft_manifest is not None:
            return build_drft_engine(
                context,
                onnx_model,
                facts.drft_manifest,
                settings,
                profile,
                workspace_gib=workspace,
                optimization_level=optimization_level,
                tf32=allow_tf32,
                sparse_weights=sparse_weights,
                tiling_optimization=tiling_optimization.value,
            )
        manifest = facts.dual_manifest
        if manifest is not None:
            return build_dual_engine(
                context,
                onnx_model,
                manifest,
                settings,
                profile,
                workspace_gib=workspace if workspace > 0 else DUAL_WORKSPACE_GIB,
                optimization_level=optimization_level,
                tf32=allow_tf32,
                sparse_weights=sparse_weights,
                tiling_optimization=tiling_optimization.value,
            )
        # One timing cache per model, GPU and TensorRT version
        timing_cache_path = None
        if settings.timing_cache_path:
            name = timing_cache_name(
                onnx_model.bytes,
                get_cuda_device_name(gpu_index),
                get_cuda_compute_capability(gpu_index),
                trt.__version__,
            )
            timing_cache_path = f"{settings.timing_cache_path}/{name}"

        # A mixed-precision ONNX (ModelOpt AutoCast) keeps its own types.
        mixed = onnx_model.info.dtype.startswith("mixed-")
        config = BuildConfig(
            precision="mixed" if mixed else precision.value,
            workspace_size_gb=workspace,
            min_shape=(min_height, min_width),
            opt_shape=(opt_height, opt_width),
            max_shape=(max_height, max_width),
            use_dynamic_shapes=use_dynamic,
            allow_tf32=allow_tf32,
            optimization_level=optimization_level,
            sparse_weights=sparse_weights,
            tiling_optimization=tiling_optimization.value,
        )

        def build() -> TensorRTEngine:
            logger.info(
                "Building TensorRT engine: precision=%s, dynamic=%s, workspace=%s,"
                " tf32=%s, sparse=%s, level=%d, tiling=%s",
                config.precision,
                use_dynamic,
                f"{workspace:g}GiB" if workspace > 0 else "automatic",
                allow_tf32,
                sparse_weights,
                optimization_level,
                tiling_optimization.value,
            )
            return build_engine_from_onnx(
                onnx_model.bytes,
                config,
                gpu_index=gpu_index,
                timing_cache_path=timing_cache_path,
            )

        key = engine_key(
            onnx_model,
            ("onnx", *dataclasses.astuple(config)),
            get_cuda_device_name(gpu_index),
            get_cuda_compute_capability(gpu_index),
            trt.__version__,
            None,
        )
        return cached_engine(key, build, settings.engine_cache_path)


def build_dual_engine(
    context: NodeContext,
    onnx_model: OnnxModel,
    manifest: dict,
    settings: TensorRTSettings,
    profile: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    *,
    workspace_gib: float,
    optimization_level: int,
    tf32: bool,
    sparse_weights: bool,
    tiling_optimization: str,
) -> TensorRTEngine:
    """DUAL's specialized engine, in the ONNX's mixed precision. A dynamic DUAL ONNX
    takes the node's shape inputs as its (height, width) profile; a fixed one has its
    own size. The node's precision input does not apply (hidden); its workspace, TF32,
    sparsity, optimization level and tiling optimization do."""
    major, minor = get_cuda_compute_capability(settings.gpu_index)
    spec = manifest["specialization"]
    if spec.get("dynamic", False):
        dual.check_profile(manifest, *profile)
    else:
        size = (spec["height"], spec["width"])
        profile = (size, size, size)
    low, high = profile[0], profile[2]
    build_profile = profile if spec.get("dynamic", False) else None
    min_shape, opt_shape, max_shape = ((1, 3, h, w) for h, w in profile)
    info = TensorRTEngineInfo(
        precision="mixed",
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

    def build() -> TensorRTEngine:
        logger.info(
            "Building a DUAL engine (%s x%d, %dx%d to %dx%d, plugins %s, SM %d%d):"
            " workspace=%gGiB, tf32=%s, sparse=%s, level=%d, tiling=%s",
            spec["factory"],
            spec["scale"],
            low[1],
            low[0],
            high[1],
            high[0],
            spec["plugin_key"],
            major,
            minor,
            workspace_gib,
            tf32,
            sparse_weights,
            optimization_level,
            tiling_optimization,
        )
        engine_bytes = dual.build_engine(
            onnx_model.bytes,
            manifest,
            settings.gpu_index,
            build_profile,
            workspace_gib=workspace_gib,
            optimization_level=optimization_level,
            tf32=tf32,
            sparse_weights=sparse_weights,
            tiling_optimization=tiling_optimization,
            context=context,
        )
        return TensorRTEngine(engine_bytes, info)

    key = engine_key(
        onnx_model,
        (
            "dual",
            build_profile,
            workspace_gib,
            optimization_level,
            tf32,
            sparse_weights,
            tiling_optimization,
        ),
        get_cuda_device_name(settings.gpu_index),
        (major, minor),
        trt.__version__,
        dual_plugin_identity(),
    )
    return cached_engine(key, build, settings.engine_cache_path)


def build_drft_engine(
    context: NodeContext,
    onnx_model: OnnxModel,
    manifest: dict,
    settings: TensorRTSettings,
    profile: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    *,
    workspace_gib: float,
    optimization_level: int,
    tf32: bool,
    sparse_weights: bool,
    tiling_optimization: str,
) -> TensorRTEngine:
    """Legacy DRFT's engine with chaiNNer-C's DRFT plugins (registered at load), in the
    ONNX's mixed precision. The node's shape inputs are its (height, width) profile,
    fixed or dynamic; its precision input does not apply; its workspace (0: TensorRT's
    default), TF32, sparsity, optimization level and tiling optimization do."""
    major, minor = get_cuda_compute_capability(settings.gpu_index)
    drft.check_profile(manifest, *profile)
    low, high = profile[0], profile[2]
    min_shape, opt_shape, max_shape = ((1, 3, h, w) for h, w in profile)
    info = TensorRTEngineInfo(
        precision="mixed",
        input_channels=3,
        output_channels=3,
        scale=manifest["scale"],
        gpu_architecture=f"sm_{major}{minor}",
        tensorrt_version=trt.__version__,
        has_dynamic_shapes=min_shape != max_shape,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
    )

    def build() -> TensorRTEngine:
        logger.info(
            "Building a DRFT engine (x%d, window %d, %dx%d to %dx%d, SM %d%d):"
            " workspace=%s, tf32=%s, sparse=%s, level=%d, tiling=%s",
            manifest["scale"],
            manifest["window_size"],
            low[1],
            low[0],
            high[1],
            high[0],
            major,
            minor,
            f"{workspace_gib:g}GiB" if workspace_gib > 0 else "automatic",
            tf32,
            sparse_weights,
            optimization_level,
            tiling_optimization,
        )
        engine_bytes = drft.build_engine(
            onnx_model.bytes,
            manifest,
            settings.gpu_index,
            profile,
            workspace_gib=workspace_gib,
            optimization_level=optimization_level,
            tf32=tf32,
            sparse_weights=sparse_weights,
            tiling_optimization=tiling_optimization,
            context=context,
        )
        return TensorRTEngine(engine_bytes, info)

    key = engine_key(
        onnx_model,
        (
            "drft",
            profile,
            workspace_gib,
            optimization_level,
            tf32,
            sparse_weights,
            tiling_optimization,
        ),
        get_cuda_device_name(settings.gpu_index),
        (major, minor),
        trt.__version__,
        drft_plugin_identity(),
    )
    return cached_engine(key, build, settings.engine_cache_path)
