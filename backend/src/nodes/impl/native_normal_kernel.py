"""Owned double kernel buffers and exact normal-map derivative preparation."""

from __future__ import annotations

import ctypes as ct
import math
from functools import lru_cache

import numpy as np

from .native import check, lib
from .native_prepared import PreparedCache, float_state
from .native_unsharp import weighted

_gaussian_plans = PreparedCache[np.ndarray](2 * 1024 * 1024, 16)
_pair_plans = PreparedCache[tuple[np.ndarray, np.ndarray]](2 * 1024 * 1024, 32)


@lru_cache(maxsize=1)
def _api():
    library = lib()
    dp, size = ct.POINTER(ct.c_double), ct.c_size_t
    library.cn_normal_gaussian_kernel.argtypes = [dp, size, dp, size]
    library.cn_normal_kernel_pair.argtypes = [dp, dp, dp, size, size, ct.c_int]
    for name in ("cn_normal_gaussian_kernel", "cn_normal_kernel_pair"):
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
    payload = source.tobytes()
    key = (radius, payload, float_state())

    def build():
        result = np.empty((radius * 2 + 1, radius * 2 + 1), np.float64)
        check(
            _api().cn_normal_gaussian_kernel(
                _double(source), len(parameters), _double(result), radius
            )
        )
        result.setflags(write=False)
        return result, result.nbytes + len(payload)

    # Public helpers continue returning independently writable arrays.
    return _gaussian_plans.get(key, build).copy()


def kernel_pair(
    kernel: np.ndarray, *, normalize: bool
) -> tuple[np.ndarray, np.ndarray]:
    source = np.require(kernel, dtype=np.float64, requirements=["C", "A"])
    if source.ndim != 2 or not source.size:
        raise ValueError("Normal derivative kernel must be a nonempty matrix")
    payload = source.tobytes()
    key = (source.shape, bool(normalize), payload, float_state())

    def build():
        x = np.empty(source.shape, np.float64)
        y = np.empty(source.shape[::-1], np.float64)
        check(
            _api().cn_normal_kernel_pair(
                _double(source), _double(x), _double(y), *source.shape, normalize
            )
        )
        x.setflags(write=False)
        y.setflags(write=False)
        return (x, y), x.nbytes + y.nbytes + len(payload)

    x, y = _pair_plans.get(key, build)
    return x.copy(), y.copy()


def sharpen(image: np.ndarray, blurred: np.ndarray) -> np.ndarray:
    # Upstream's cv2.addWeighted(height, 2.0, blurred, -1.0, 0) is Unsharp Mask's
    # weighted sum at amount 1: one mirror of OpenCV's addWeighted.
    return weighted(image, blurred, 1.0)
