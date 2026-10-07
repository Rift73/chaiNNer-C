"""CUDA runtime helpers for TensorRT (cuda-python)."""

from __future__ import annotations

from typing import Any

from cuda.bindings import runtime as cudart


def check_cuda(result: tuple[Any, ...]) -> Any:
    """Raise on a CUDA runtime error; return the call's value, if any."""
    err = result[0]
    if err != cudart.cudaError_t.cudaSuccess:
        _, name = cudart.cudaGetErrorName(err)
        _, text = cudart.cudaGetErrorString(err)
        raise RuntimeError(f"CUDA error {name.decode()}: {text.decode()}")
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
