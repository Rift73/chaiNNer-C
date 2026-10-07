"""Checked running-sum blur and compatible OpenCV spatial convolution paths."""

from __future__ import annotations

import ctypes as ct

import cv2
import numpy as np

from .native import check, f32, lib, ptr
from .native_opencv_simd import float32_lanes, target
from .native_spectral_filter import filter2d as _spectral_filter
from .native_versions import CV_CN_MAX, cv2_is_tested

_lib = lib()
_p = ct.POINTER(ct.c_float)
_n = ct.c_size_t
_lib.cn_convolution_box.argtypes = [_p, _p, _n, _n, _n, _n, _n]
_lib.cn_convolution_box.restype = ct.c_int
_lib.cn_convolution_spatial.argtypes = [_p, _p, _n, _n, _n, _p, _n, _n, _n, _n]
_lib.cn_convolution_spatial.restype = ct.c_int


def _image(image: np.ndarray) -> tuple[np.ndarray, int]:
    source = f32(image)
    if source.ndim not in (2, 3) or any(size == 0 for size in source.shape):
        raise ValueError(f"Invalid convolution image shape {source.shape}")
    return source, source.shape[2] if source.ndim == 3 else 1


def box_blur(image: np.ndarray, radius_x: int, radius_y: int) -> np.ndarray:
    source, channels = _image(image)
    if not 0 <= radius_x <= 1000 or not 0 <= radius_y <= 1000:
        raise ValueError("Box blur radii must be integers in [0,1000]")
    if channels > CV_CN_MAX:
        return cv2.blur(
            source,
            (2 * radius_x + 1, 2 * radius_y + 1),
            borderType=cv2.BORDER_REFLECT_101,
        )
    shape = source.shape[:2] if channels == 1 else source.shape
    result = np.empty(shape, np.float32)
    check(
        _lib.cn_convolution_box(
            ptr(source),
            ptr(result),
            source.shape[0],
            source.shape[1],
            channels,
            radius_x,
            radius_y,
        )
    )
    return result


def _spatial_lanes() -> int | None:
    # The preserved application ships this exact OpenCV dispatch contract.
    # Other versions/dispatches retain their own accumulation and DFT choices.
    if not cv2_is_tested(__name__) or cv2.ipp.useIPP_NotExact():
        return None
    mode = target("filter")
    if mode == "BASELINE":
        return 0
    return float32_lanes(mode) if mode == "AVX2" else None


def convolve(image: np.ndarray, kernel: object, padding: object) -> np.ndarray:
    source, channels = _image(image)
    if not isinstance(kernel, np.ndarray) or kernel.ndim != 2 or not kernel.size:
        raise ValueError("Convolution kernel must be a nonempty two-dimensional array")
    if kernel.dtype not in (np.float32, np.float64):
        raise TypeError("Convolution kernel must use float32 or float64 coefficients")
    if not isinstance(padding, int) or padding < 0:
        raise ValueError("Convolution padding must be a nonnegative integer")
    if padding > (np.iinfo(np.int32).max - max(source.shape[:2])) // 2:
        raise OverflowError("Padded image exceeds OpenCV's dimension range")
    if cv2_is_tested(__name__) and channels <= CV_CN_MAX:
        # filter.dispatch.cpp selects DFT at area130 for float32 when SSE3 is
        # active, otherwise at area50. Disabled hardware features have '?' in
        # getCPUFeaturesLine, including when setUseOptimized(False) is active.
        features = cv2.getCPUFeaturesLine().split()
        threshold = 130 if "SSE3" in features or "*SSE3" in features else 50
        if kernel.size >= threshold:
            return _spectral_filter(source, kernel, padding)
    lanes = _spatial_lanes()
    if lanes is None or channels > CV_CN_MAX or kernel.size >= 130:
        padded = cv2.copyMakeBorder(
            source,
            padding,
            padding,
            padding,
            padding,
            borderType=cv2.BORDER_CONSTANT,
            value=(0.0,),
        )
        return cv2.filter2D(padded, -1, kernel)
    # OpenCV silently narrows double coefficients, including overflow to Inf.
    with np.errstate(over="ignore", invalid="ignore"):
        coefficients = np.require(kernel, dtype=np.float32, requirements=["C", "A"])
    height, width = (size + padding * 2 for size in source.shape[:2])
    shape = (height, width) if channels == 1 else (height, width, channels)
    result = np.empty(shape, np.float32)
    check(
        _lib.cn_convolution_spatial(
            ptr(source),
            ptr(result),
            source.shape[0],
            source.shape[1],
            channels,
            ptr(coefficients),
            *coefficients.shape,
            padding,
            lanes,
        )
    )
    return result
