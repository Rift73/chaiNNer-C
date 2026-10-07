"""Checked C box-filter kernels with the preserved OpenCV rounding contract."""

from __future__ import annotations

import ctypes as ct
import math
import operator

import cv2
import numpy as np

from .native import check, f32, lib, ptr
from .native_opencv_simd import float32_lanes, target
from .native_versions import CV_CN_MAX

_lib = lib()
_u8 = ct.POINTER(ct.c_uint8)
_n = ct.c_size_t
_lib.cn_box_mean_u8.argtypes = [_u8, _u8, _n, _n, _n]
_lib.cn_box_mean_u8.restype = ct.c_int
_fp = ct.POINTER(ct.c_float)
_lib.cn_box_separable.argtypes = [_fp, _fp, _n, _n, _n, ct.c_double, ct.c_double, _n]
_lib.cn_box_separable.restype = ct.c_int
_lib.cn_box_kernel_2d.argtypes = [_fp, _n, ct.c_double, ct.c_double]
_lib.cn_box_kernel_2d.restype = ct.c_int


def kernel_2d(radius_x: float, radius_y: float) -> np.ndarray:
    if not 0 <= radius_x <= 1000 or not 0 <= radius_y <= 1000:
        raise ValueError("Box kernel radii must be finite values in [0,1000]")
    result = np.empty(
        (math.ceil(radius_y) * 2 + 1, math.ceil(radius_x) * 2 + 1), np.float32
    )
    check(_lib.cn_box_kernel_2d(ptr(result), result.size, radius_x, radius_y))
    return result


def mean_u8(image: np.ndarray, radius: int) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 2 or not all(image.shape):
        raise ValueError("Adaptive mean requires a nonempty uint8 grayscale image")
    radius = operator.index(radius)
    if not 0 <= radius <= (0x7FFFFFFF - 1) // 2:
        raise ValueError("Adaptive mean radius exceeds the supported integer range")
    source = np.require(image, requirements=["C", "A"])
    result = np.empty(source.shape, np.uint8)
    check(
        _lib.cn_box_mean_u8(
            source.ctypes.data_as(_u8),
            result.ctypes.data_as(_u8),
            *source.shape,
            radius,
        )
    )
    return result


def separable_box(image: np.ndarray, radius_x: float, radius_y: float) -> np.ndarray:
    source = f32(image)
    if source.ndim not in (2, 3) or not all(source.shape):
        raise ValueError("Separable Box Blur requires a nonempty image")
    channels = source.shape[2] if source.ndim == 3 else 1
    if channels > CV_CN_MAX:
        raise cv2.error(f"Box Blur supports at most {CV_CN_MAX} image channels.")
    shape = source.shape[:2] if channels == 1 else source.shape
    result = np.empty(shape, np.float32)
    # sepFilter2D's dispatch width (filter.simd.hpp).
    lanes = float32_lanes(target("filter"))
    check(
        _lib.cn_box_separable(
            ptr(source),
            ptr(result),
            *source.shape[:2],
            channels,
            radius_x,
            radius_y,
            lanes,
        )
    )
    return result
