"""Owned double kernel buffers and exact normal-map derivative preparation."""

from __future__ import annotations

import ctypes as ct
import math
from functools import lru_cache

import numpy as np

from .native import check, lib


@lru_cache(maxsize=1)
def _api():
    library = lib()
    dp, size = ct.POINTER(ct.c_double), ct.c_size_t
    library.cn_normal_gaussian_kernel.argtypes = [dp, size, dp, size]
    library.cn_normal_kernel_pair.argtypes = [dp, dp, dp, size, size, ct.c_int]
    for name in (
        "cn_normal_gaussian_kernel",
        "cn_normal_kernel_pair",
    ):
        getattr(library, name).restype = ct.c_int
    return library


def _double(array: np.ndarray):
    return array.ctypes.data_as(ct.POINTER(ct.c_double))


def gaussian_kernel(parameters: list[tuple[float, float]]) -> np.ndarray:
    # Scalar size/error preparation retains the public helper's Python errors;
    # all per-coefficient sampling and exponential evaluation is performed in C.
    total = sum(weight for _, weight in parameters)
    radius = 0
    if total != 0:
        radius = 1
        for sigma, weight in parameters:
            if weight > 0:
                radius = max(radius, math.ceil(2 * sigma))
        radius += 1
        for sigma, weight in parameters:
            _ = weight / (math.pi * (2 * sigma * sigma))
    source = np.require(parameters, dtype=np.float64, requirements=["C", "A"])
    if source.size == 0:
        source = source.reshape(0, 2)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("Gaussian parameters must be sigma/weight pairs")
    result = np.empty((radius * 2 + 1, radius * 2 + 1), np.float64)
    check(
        _api().cn_normal_gaussian_kernel(
            _double(source), len(parameters), _double(result), radius
        )
    )
    return result


def kernel_pair(
    kernel: np.ndarray, *, normalize: bool
) -> tuple[np.ndarray, np.ndarray]:
    source = np.require(kernel, dtype=np.float64, requirements=["C", "A"])
    if source.ndim != 2 or not source.size:
        raise ValueError("Normal derivative kernel must be a nonempty matrix")
    x = np.empty(source.shape, np.float64)
    y = np.empty(source.shape[::-1], np.float64)
    check(
        _api().cn_normal_kernel_pair(
            _double(source), _double(x), _double(y), *source.shape, normalize
        )
    )
    return x, y
