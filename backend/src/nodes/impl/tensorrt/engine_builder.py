"""TensorRT engine building utilities."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Literal

import tensorrt as trt
from cuda.bindings import runtime as cudart
from sanic.log import logger

from .engine_info import deserialize_engine, read_engine_info
from .memory import check_cuda, get_cuda_compute_capability, get_cuda_device_name
from .model import TensorRTEngine, TensorRTEngineInfo
from .trt_logger import TRT_LOGGER


@dataclass
class BuildConfig:
    """Configuration for TensorRT engine building."""

    # "mixed": the ONNX's own per-node types (Convert To ONNX's Mixed data types).
    precision: Literal["fp32", "fp16", "bf16", "mixed"]
    # 0: TensorRT's default, as much as the GPU's total memory.
    workspace_size_gb: float
    min_shape: tuple[int, int]  # (height, width)
    opt_shape: tuple[int, int]  # (height, width)
    max_shape: tuple[int, int]  # (height, width)
    use_dynamic_shapes: bool
    allow_tf32: bool
    # TensorRT's builder optimization level: 0 (fastest build) to 5 (most kernels tried).
    optimization_level: int = 3
    # 2:4 structured-sparse kernels for weights that are already 2:4 sparse.
    sparse_weights: bool = False
    # TensorRT's tiling optimization (fused kernels that keep their intermediate tensors
    # in the L2 cache; not image tiling): "none" is TensorRT's default.
    tiling_optimization: Literal["none", "fast", "moderate", "full"] = "none"


# TensorRT skipping a tactic for want of memory (a warning), and the bytes it asked for.
_MEMORY_REQUEST = re.compile(r"insufficient memory on requested size of (\d+)")


def _build_failure(messages: list[tuple[int, str]], config: BuildConfig) -> str:
    """Why a build failed, from the TensorRT messages logged during it."""
    message = "Failed to build TensorRT engine."
    warning = int(trt.ILogger.Severity.WARNING)
    errors = [text for level, text in messages if level < warning]
    if errors:
        message += " TensorRT: " + " ".join(errors[-3:])
    skipped = [int(n) for _, text in messages for n in _MEMORY_REQUEST.findall(text)]
    if skipped:
        needed = f"{max(skipped) / 1024**3:.2f} GiB"
        if config.workspace_size_gb > 0:
            message += (
                f"\nSome layers needed {needed} of scratch memory, more than the"
                f" Workspace (GiB) of {config.workspace_size_gb:g}: raise it, set it to 0"
                " (TensorRT's default, up to all of the GPU's memory), or lower the"
                " maximum size."
            )
        else:
            message += (
                f"\nSome layers needed {needed} of scratch memory, more than the GPU had"
                " free: close other programs using it or lower the maximum size."
            )
    return message


_PRECISION_DTYPES = {
    "fp32": trt.DataType.FLOAT,
    "fp16": trt.DataType.HALF,
    "bf16": trt.DataType.BF16,
}


def _uses_precision_flags() -> bool:
    """TensorRT <= 10 sets precision with builder flags; TensorRT 11 removed them."""
    return hasattr(trt.BuilderFlag, "FP16") and hasattr(trt.BuilderFlag, "BF16")


def _create_network(builder: Any) -> Any:
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if not _uses_precision_flags():
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    return builder.create_network(flags)


def _apply_precision(builder_config: Any, network: Any, precision: str) -> None:
    """Make the engine use `precision`, or fail with a clear reason."""
    if precision == "mixed":
        return
    if _uses_precision_flags():
        if precision == "fp16":
            builder_config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "bf16":
            builder_config.set_flag(trt.BuilderFlag.BF16)
        return
    # Strongly typed network: the ONNX model's tensor types decide the precision.
    onnx_dtype = network.get_input(0).dtype
    if onnx_dtype != _PRECISION_DTYPES[precision]:
        raise ValueError(
            f"TensorRT {trt.__version__} builds strongly typed engines, so the precision"
            f" comes from the ONNX model, whose input is {onnx_dtype}. To build a"
            f" {precision} engine, convert the model to {precision} first (for example"
            " with the FP16 option of PyTorch's Convert To ONNX node), or pick the"
            " model's own precision."
        )


def timing_cache_name(
    onnx_bytes: bytes,
    gpu_name: str,
    compute_capability: tuple[int, int],
    tensorrt_version: str,
) -> str:
    """The timing cache file of a model on a GPU. TensorRT refuses a cache from
    another GPU or TensorRT version, so each gets its own file."""
    model = hashlib.md5(onnx_bytes[:1024]).hexdigest()[:8]
    gpu = re.sub(r"[^A-Za-z0-9.]+", "-", gpu_name).strip("-")
    major, minor = compute_capability
    return f"timing_{model}_{gpu}_sm{major}{minor}_trt{tensorrt_version}.cache"


def parse_onnx_model(
    onnx_bytes: bytes,
    network: Any,
    parser: Any,
) -> None:
    """Parse an ONNX model into a TensorRT network."""
    success = parser.parse(onnx_bytes)
    if not success:
        error_msgs = []
        for i in range(parser.num_errors):
            error = parser.get_error(i)
            error_msgs.append(f"  {error.desc()}")
        errors_str = "\n".join(error_msgs)
        raise RuntimeError(f"Failed to parse ONNX model:\n{errors_str}")


def build_engine_from_onnx(
    onnx_bytes: bytes,
    config: BuildConfig,
    gpu_index: int = 0,
    timing_cache_path: str | None = None,
) -> TensorRTEngine:
    """
    Build a TensorRT engine from an ONNX model.

    Args:
        onnx_bytes: The ONNX model as bytes
        config: Build configuration
        gpu_index: GPU device index to use
        timing_cache_path: Optional path to timing cache for faster rebuilds

    Returns:
        A TensorRTEngine instance
    """
    # Set the CUDA device
    check_cuda(cudart.cudaSetDevice(gpu_index))

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = _create_network(builder)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    logger.info("Parsing ONNX model...")
    parse_onnx_model(onnx_bytes, network, parser)

    # Get input/output info
    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)

    input_shape = input_tensor.shape
    output_shape = output_tensor.shape

    # Detect channels (assuming NCHW format)
    input_channels = input_shape[1] if len(input_shape) >= 4 else 3
    output_channels = output_shape[1] if len(output_shape) >= 4 else 3

    # Calculate scale from input/output dimensions
    scale = None
    if len(input_shape) >= 4 and len(output_shape) >= 4:
        in_h, in_w = input_shape[2], input_shape[3]
        out_h, out_w = output_shape[2], output_shape[3]

        if in_h > 0 and in_w > 0 and out_h > 0 and out_w > 0:
            scale_h = out_h // in_h if out_h % in_h == 0 else None
            scale_w = out_w // in_w if out_w % in_w == 0 else None
            if scale_h == scale_w and scale_h is not None:
                scale = scale_h

    # Configure builder
    builder_config = builder.create_builder_config()
    if config.workspace_size_gb > 0:
        workspace_bytes = int(config.workspace_size_gb * (1024**3))
        builder_config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_bytes
        )

    _apply_precision(builder_config, network, config.precision)
    if not config.allow_tf32:
        # TF32 is a math mode, not a tensor type, so strongly typed networks keep it:
        # on by default, it runs float32 convolutions and matrix products with
        # 10-bit mantissa inputs (about FP16's precision).
        builder_config.clear_flag(trt.BuilderFlag.TF32)
    if config.sparse_weights:
        builder_config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    builder_config.builder_optimization_level = config.optimization_level
    builder_config.tiling_optimization_level = getattr(
        trt.TilingOptimizationLevel, config.tiling_optimization.upper()
    )

    # Configure dynamic shapes if needed
    has_dynamic = any(d == -1 for d in input_shape)
    if has_dynamic or config.use_dynamic_shapes:
        logger.info("Configuring dynamic shapes...")
        profile = builder.create_optimization_profile()

        min_h, min_w = config.min_shape
        opt_h, opt_w = config.opt_shape
        max_h, max_w = config.max_shape

        # Set the optimization profile for the input tensor
        min_dims = (1, input_channels, min_h, min_w)
        opt_dims = (1, input_channels, opt_h, opt_w)
        max_dims = (1, input_channels, max_h, max_w)

        profile.set_shape(input_tensor.name, min_dims, opt_dims, max_dims)
        builder_config.add_optimization_profile(profile)

    # Load the timing cache, or start one: TensorRT saves only a cache that was set
    # (get_timing_cache is None otherwise), and refuses one from another GPU.
    if timing_cache_path:
        try:
            with open(timing_cache_path, "rb") as f:
                cache_data = f.read()
            timing_cache = builder_config.create_timing_cache(cache_data)
            if builder_config.set_timing_cache(timing_cache, ignore_mismatch=False):
                logger.info("Loaded timing cache from %s", timing_cache_path)
            else:
                logger.info("Replacing the timing cache of another GPU or version")
        except FileNotFoundError:
            logger.debug("No timing cache found at %s", timing_cache_path)
        except Exception as e:
            logger.warning("Failed to load timing cache: %s", e)
        if builder_config.get_timing_cache() is None:
            timing_cache = builder_config.create_timing_cache(b"")
            builder_config.set_timing_cache(timing_cache, ignore_mismatch=False)

    # Build the engine
    logger.info("Building TensorRT engine (this may take a while)...")
    with TRT_LOGGER.capture() as messages:
        serialized_engine = builder.build_serialized_network(network, builder_config)

    if serialized_engine is None:
        raise RuntimeError(_build_failure(messages, config))
    engine_bytes = bytes(serialized_engine)

    if scale is None:
        # A dynamic ONNX model's scale needs TensorRT's shape inference, as Load
        # Engine does; Upscale Image cannot run an engine without it.
        loaded = deserialize_engine(engine_bytes, gpu_index)
        try:
            scale = read_engine_info(loaded).scale
        finally:
            loaded.release()

    # Save timing cache
    if timing_cache_path:
        try:
            timing_cache = builder_config.get_timing_cache()
            if timing_cache:
                cache_data = timing_cache.serialize()
                with open(timing_cache_path, "wb") as f:
                    f.write(cache_data)
                logger.info("Saved timing cache to %s", timing_cache_path)
        except Exception as e:
            logger.warning("Failed to save timing cache: %s", e)

    # Get GPU info
    major, minor = get_cuda_compute_capability(gpu_index)
    gpu_arch = f"sm_{major}{minor}"
    gpu_name = get_cuda_device_name(gpu_index)
    logger.info("Built engine for %s (%s)", gpu_name, gpu_arch)

    # Record the accepted input shapes. A fixed engine accepts exactly one shape:
    # the profile shape for a dynamic ONNX model, or the model's own static shape.
    min_shape = (1, input_channels, config.min_shape[0], config.min_shape[1])
    opt_shape = (1, input_channels, config.opt_shape[0], config.opt_shape[1])
    max_shape = (1, input_channels, config.max_shape[0], config.max_shape[1])
    if not config.use_dynamic_shapes:
        if not has_dynamic and len(input_shape) >= 4:
            opt_shape = (1, input_channels, input_shape[2], input_shape[3])
        min_shape = opt_shape
        max_shape = opt_shape

    # Create engine info
    info = TensorRTEngineInfo(
        precision=config.precision,
        input_channels=input_channels,
        output_channels=output_channels,
        scale=scale,
        gpu_architecture=gpu_arch,
        tensorrt_version=trt.__version__,
        has_dynamic_shapes=min_shape != max_shape,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
    )

    return TensorRTEngine(engine_bytes, info)
