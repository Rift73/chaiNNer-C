"""C Gaussian kernel construction and separable filtering."""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import cv2
import numpy as np

from .native import check, f32, lib
from .native_opencv_simd import float32_lanes, target
from .native_versions import CV_CN_MAX


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    dll.cn_gaussian_kernel.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_double, ct.c_int]
    dll.cn_gaussian_kernel.restype = ct.c_int
    dll.cn_gaussian_f32.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_double,
        ct.c_double,
        ct.c_int,
        ct.c_size_t,
    ]
    dll.cn_gaussian_f32.restype = ct.c_int
    dll.cn_gaussian_mean_u8.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
    ]
    dll.cn_gaussian_mean_u8.restype = ct.c_int
    dll.cn_ssim_window.argtypes = [ct.c_void_p]
    dll.cn_ssim_window.restype = ct.c_int
    return dll


def kernel(size: int, sigma: float, *, double: bool = False) -> np.ndarray:
    out = np.empty(size, np.float64 if double else np.float32)
    check(_api().cn_gaussian_kernel(out.ctypes.data, size, sigma, int(double)))
    return out


def ssim_window() -> np.ndarray:
    out = np.empty((11, 11), np.float64)
    check(_api().cn_ssim_window(out.ctypes.data))
    return out


def mean_u8(image: np.ndarray, radius: int) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 2 or not all(image.shape):
        raise ValueError("Gaussian mean requires a nonempty uint8 grayscale image")
    source = np.require(image, requirements=["C", "A"])
    out = np.empty(source.shape, np.uint8)
    # GaussianBlurFixedPoint's dispatch (smooth.simd.hpp).
    lanes = float32_lanes(target("smooth"))
    check(
        _api().cn_gaussian_mean_u8(
            source.ctypes.data,
            out.ctypes.data,
            *source.shape,
            radius,
            lanes,
        )
    )
    return out


def gaussian(
    image: np.ndarray,
    sigma_x: float,
    sigma_y: float = 0,
    *,
    size: tuple[int, int] = (0, 0),
    border: int = cv2.BORDER_REFLECT,
) -> np.ndarray:
    source = f32(image)
    if source.ndim not in (2, 3) or not all(source.shape):
        raise ValueError("Gaussian filtering requires a nonempty image")
    channels = source.shape[2] if source.ndim == 3 else 1
    if channels > CV_CN_MAX:
        raise cv2.error(f"Gaussian Blur supports at most {CV_CN_MAX} image channels.")
    out = np.empty(source.shape[:2] if channels == 1 else source.shape, np.float32)
    # Float GaussianBlur is sepFilter2D's dispatch (filter.simd.hpp).
    lanes = float32_lanes(target("filter"))
    check(
        _api().cn_gaussian_f32(
            source.ctypes.data,
            out.ctypes.data,
            *source.shape[:2],
            channels,
            *size,
            sigma_x,
            sigma_y,
            border,
            lanes,
        )
    )
    return out
