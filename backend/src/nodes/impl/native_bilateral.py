"""Exact exposed OpenCV bilateral algorithm, with its default IPP primitive retained."""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import cv2
import numpy as np

from .native import check, f32, lib
from .native_channels import concatenate_channels
from .native_opencv_simd import float32_lanes, target
from .native_versions import cv2_is_tested


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    dll.cn_bilateral_f32.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_double,
        ct.c_double,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
    ]
    dll.cn_bilateral_f32.restype = ct.c_int
    return dll


def _row_planes(image: np.ndarray) -> bool:
    """Whether OpenCV's minMaxIdx scans the image row by row.

    cv2's binding (modules/python/src2/cv2_convert.cpp) wraps the array without a
    copy unless its strides forbid it. A wrapped array whose rows are apart is a
    non-continuous Mat, which NAryMatIterator walks one row at a time. The Mat's
    step is unsigned, so a negative row stride (a reversed single column) is apart.
    """
    shape, strides, item = image.shape, image.strides, image.itemsize
    last = image.ndim - 1
    if any(
        size > 1 and (stride != item if axis == last else stride < strides[axis + 1])
        for axis, (size, stride) in enumerate(zip(shape, strides, strict=True))
    ):
        return False
    if image.ndim == 3 and strides[1] != item * shape[2]:
        return False
    row = item * int(np.prod(shape[1:]))
    return shape[0] > 1 and not 0 <= strides[0] <= row


def bilateral(
    image: np.ndarray, radius: int, sigma_color: float, sigma_space: float
) -> np.ndarray:
    if image.dtype != np.float32 or not cv2_is_tested(__name__) or cv2.ipp.useIPP():
        # Installed default IPP is proprietary and differs in saved byte values
        # from the exposed OpenCV algorithm. Do not substitute its approximation.
        return cv2.bilateralFilter(
            image,
            radius * 2 + 1,
            sigma_color,
            sigma_space,
            borderType=cv2.BORDER_REFLECT_101,
        )
    source = f32(image)
    channels = source.shape[2] if source.ndim == 3 else 1
    if source.ndim not in (2, 3) or not all(source.shape) or channels not in (1, 3):
        # Preserve OpenCV's original errors for images outside the node's
        # supported bilateral channel count rather than accepting new layouts.
        return cv2.bilateralFilter(
            image,
            radius * 2 + 1,
            sigma_color,
            sigma_space,
            borderType=cv2.BORDER_REFLECT_101,
        )
    out = np.empty(source.shape[:2] if channels == 1 else source.shape, np.float32)
    status = _api().cn_bilateral_f32(
        source.ctypes.data,
        out.ctypes.data,
        *source.shape[:2],
        channels,
        radius,
        sigma_color,
        sigma_space,
        float32_lanes(target("bilateral_filter")),
        float32_lanes(target("minmax")),
        _row_planes(image),
    )
    if status == 4:
        raise cv2.error("Unknown C++ exception from OpenCV code")
    check(status)
    return out


def surface(
    image: np.ndarray, radius: int, sigma_color: int, sigma_space: int
) -> np.ndarray:
    sigma = sigma_color / 255
    if image.ndim == 3 and image.shape[2] == 4:
        rgb = bilateral(image[:, :, :3], radius, sigma, sigma_space)
        alpha = bilateral(image[:, :, 3], radius, sigma, sigma_space)
        return (
            concatenate_channels(rgb, alpha)
            if image.dtype == np.float32
            else np.dstack((rgb, alpha))
        )
    return bilateral(image, radius, sigma, sigma_space)
