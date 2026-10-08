"""Read the metadata of a serialized TensorRT engine."""

from __future__ import annotations

from typing import Literal

import tensorrt as trt
from cuda.bindings import runtime as cudart
from sanic.log import logger

from .dual import input_alignment
from .memory import check_cuda, get_cuda_compute_capability
from .model import DeserializedEngine, TensorRTEngineInfo


def _dims(shape: trt.Dims) -> tuple[int, int, int, int]:
    n, c, h, w = (int(d) for d in shape)
    return n, c, h, w


def deserialize_engine(engine_bytes: bytes, gpu_index: int) -> DeserializedEngine:
    """Deserialize an engine on a GPU, kept together with its runtime."""
    check_cuda(cudart.cudaSetDevice(gpu_index))
    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, "")
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(
            "Failed to deserialize the TensorRT engine. It may have been built with an"
            " incompatible TensorRT version or for a different GPU architecture."
        )
    return DeserializedEngine(runtime, engine, gpu_index)


def read_engine_info(loaded: DeserializedEngine) -> TensorRTEngineInfo:
    """Describe a deserialized engine's I/O, profile and upscale factor."""
    check_cuda(cudart.cudaSetDevice(loaded.gpu_index))
    major, minor = get_cuda_compute_capability(loaded.gpu_index)
    engine = loaded.engine

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    outputs = [n for n in names if n not in inputs]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"Expected one input and one output tensor, got {inputs} and {outputs}."
        )
    input_name, output_name = inputs[0], outputs[0]

    input_shape = engine.get_tensor_shape(input_name)
    if len(input_shape) != 4:
        raise ValueError(f"Expected an NCHW input tensor, got {input_shape}.")
    min_shape, opt_shape, max_shape = (
        _dims(s) for s in engine.get_tensor_profile_shape(input_name, 0)
    )

    # The scale comes from TensorRT's shape inference at the profile's minimum
    # shape; a user-managed context allocates no activation memory for this.
    probe = engine.create_execution_context(
        trt.ExecutionContextAllocationStrategy.USER_MANAGED
    )
    if probe is None or not probe.set_input_shape(input_name, min_shape):
        raise RuntimeError("Failed to infer the TensorRT engine's output shape.")
    output_shape = _dims(probe.get_tensor_shape(output_name))
    del probe
    logger.info("TensorRT engine: %s -> %s", min_shape, output_shape)

    in_h, in_w = min_shape[2], min_shape[3]
    out_h, out_w = output_shape[2], output_shape[3]
    if out_h % in_h or out_w % in_w or out_h // in_h != out_w // in_w:
        raise ValueError(
            f"Input {min_shape} -> output {output_shape} is not a uniform integer upscale."
        )

    input_dtype = engine.get_tensor_dtype(input_name)
    precision: Literal["fp32", "fp16", "bf16", "mixed"] = "fp32"
    if input_alignment(input_name) is not None:  # DUAL's (dual.ALIGNMENT)
        precision = "mixed"
    elif input_dtype == trt.DataType.HALF:
        precision = "fp16"
    elif input_dtype == trt.DataType.BF16:
        precision = "bf16"

    return TensorRTEngineInfo(
        precision=precision,
        input_channels=int(input_shape[1]),
        output_channels=output_shape[1],
        scale=out_h // in_h,
        gpu_architecture=f"sm_{major}{minor}",
        tensorrt_version=trt.__version__,
        has_dynamic_shapes=any(d == -1 for d in input_shape),
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
    )
