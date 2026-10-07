"""Exact float32 weighted sharpening with checked, owned C buffers."""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import cv2
import numpy as np

from .native import check, f32, lib, ptr
from .native_opencv_simd import float32_lanes, target
from .native_versions import CV_CN_MAX


@lru_cache(maxsize=1)
def _api():
    library = lib()
    fp = ct.POINTER(ct.c_float)
    library.cn_unsharp_weighted.argtypes = [
        fp,
        fp,
        fp,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_double,
        ct.c_int,
    ]
    library.cn_unsharp_weighted.restype = ct.c_int
    return library


def _opencv_contiguous(image: np.ndarray) -> bool:
    # The Python OpenCV adapter copies noninterleaved/reversed arrays. Ordinary
    # row-strided views remain Mat views and retain an independent SIMD row tail.
    channels = 1 if image.ndim == 2 else image.shape[2]
    row = image.shape[1] * channels * image.itemsize
    copied = (
        (image.shape[1] > 1 and image.strides[1] != channels * image.itemsize)
        or (image.ndim == 3 and channels > 1 and image.strides[2] != image.itemsize)
        or (image.shape[0] > 1 and image.strides[0] < row)
    )
    return copied or image.shape[0] == 1 or image.strides[0] == row


def weighted(image: np.ndarray, blurred: np.ndarray, amount: float) -> np.ndarray:
    """Apply the node's original addWeighted coefficients without OpenCV loops."""
    if image.dtype != np.float32 or blurred.dtype != np.float32:
        return cv2.addWeighted(image, amount + 1, blurred, -amount, 0)
    source, low = f32(image), f32(blurred)
    if source.shape != low.shape or source.ndim not in (2, 3) or not source.size:
        raise ValueError("Unsharp Mask requires matching nonempty image shapes")
    channels = source.shape[2] if source.ndim == 3 else 1
    if channels > CV_CN_MAX:
        raise cv2.error(f"Unsharp Mask supports at most {CV_CN_MAX} image channels.")
    row = source.shape[1] * channels
    if _opencv_contiguous(image) and _opencv_contiguous(blurred):
        row = source.size
    lanes = float32_lanes(target("arithm"))
    shape = source.shape[:2] if channels == 1 else source.shape
    result = np.empty(shape, np.float32)
    check(
        _api().cn_unsharp_weighted(
            ptr(source), ptr(low), ptr(result), source.size, row, amount, lanes
        )
    )
    return result
