"""C luminance and double extrema for contrast-adaptive High Boost filtering."""

from __future__ import annotations

import ctypes as ct

import numpy as np

from .native import check, f32, lib, ptr

_lib = lib()
_dp = ct.POINTER(ct.c_double)
_fp = ct.POINTER(ct.c_float)
_up = ct.POINTER(ct.c_uint8)
_n = ct.c_size_t
_lib.cn_cas_luminance.argtypes = [_fp, _dp, _n, _n, ct.c_int]
_lib.cn_cas_luminance.restype = ct.c_int
_lib.cn_cas_extrema_f64.argtypes = [_dp, _dp, _dp, _n, _n, _up, _n, _n]
_lib.cn_cas_extrema_f64.restype = ct.c_int


def luminance(image: np.ndarray) -> np.ndarray:
    source = f32(image)
    if source.ndim != 3 or source.shape[2] < 3 or not all(source.shape):
        raise ValueError(
            "CAS color luminance requires a nonempty image with at least 3 channels"
        )
    result = np.empty(source.shape[:2], np.float64)
    # The installed NumPy/OpenBLAS three-element double dot uses sequential FMA.
    check(
        _lib.cn_cas_luminance(
            ptr(source), result.ctypes.data_as(_dp), result.size, source.shape[2], 1
        )
    )
    return result


def extrema(image: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != 2 or image.dtype != np.float64 or not all(image.shape):
        raise ValueError("CAS double extrema require a nonempty float64 plane")
    if (
        kernel.ndim != 2
        or kernel.dtype != np.uint8
        or not all(kernel.shape)
        or not np.any(kernel)
    ):
        raise ValueError(
            "CAS morphology requires a nonempty uint8 mask with an active tap"
        )
    source = np.require(image, requirements=["C", "A"])
    mask = np.require(kernel, requirements=["C", "A"])
    low, high = np.empty_like(source), np.empty_like(source)
    check(
        _lib.cn_cas_extrema_f64(
            source.ctypes.data_as(_dp),
            low.ctypes.data_as(_dp),
            high.ctypes.data_as(_dp),
            *source.shape,
            mask.ctypes.data_as(_up),
            *mask.shape,
        )
    )
    return low, high
