"""CUDA runtime helpers for TensorRT (cuda-python)."""

from __future__ import annotations

from threading import Lock
from typing import Any

from cuda.bindings import runtime as cudart

# TensorRT synchronizes the GPU's streams when it deserializes an engine or makes or
# destroys an execution context, which invalidates a CUDA graph capture on any thread:
# these take turns, across engines.
gpu_setup_lock = Lock()


class CudaOutOfMemory(RuntimeError):
    """An allocation found no memory, on the device or pinned on the host."""


def check_cuda(result: tuple[Any, ...]) -> Any:
    """Raise on a CUDA runtime error; return the call's value, if any."""
    err = result[0]
    if err != cudart.cudaError_t.cudaSuccess:
        _, name = cudart.cudaGetErrorName(err)
        _, text = cudart.cudaGetErrorString(err)
        message = f"CUDA error {name.decode()}: {text.decode()}"
        if err == cudart.cudaError_t.cudaErrorMemoryAllocation:
            # Taken, so that this thread's next enqueue does not fail on it: TensorRT's
            # reads CUDA's per-thread last error, and goes on only if there is none.
            cudart.cudaGetLastError()
            raise CudaOutOfMemory(message)
        raise RuntimeError(message)
    if len(result) == 1:
        return None
    return result[1] if len(result) == 2 else result[1:]


def get_cuda_device_name(device_id: int = 0) -> str:
    """Get the name of a CUDA device."""
    name = check_cuda(cudart.cudaGetDeviceProperties(device_id)).name
    return name.decode("utf-8").rstrip("\0") if isinstance(name, bytes) else str(name)


def get_cuda_compute_capability(device_id: int = 0) -> tuple[int, int]:
    """Get the compute capability of a CUDA device."""
    props = check_cuda(cudart.cudaGetDeviceProperties(device_id))
    return props.major, props.minor
