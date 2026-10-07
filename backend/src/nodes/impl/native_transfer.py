"""Stretch Contrast bounds, stretch and channel assembly in C."""

from __future__ import annotations

import ctypes as ct

import numpy as np

from .native import check, f32, lib, ptr
from .native_adjustments import adjust
from .native_channels import merge_channels
from .native_numpy_reduce import numpy_reduce_contiguous, numpy_sum_block
from .native_numpy_simd import float32_lanes, loop_target

_lib = lib()
_double = ct.POINTER(ct.c_double)
_lib.cn_contrast_bounds_complete.argtypes = [
    ct.POINTER(ct.c_float),
    ct.c_size_t,
    ct.c_int,
    ct.c_double,
    ct.c_int,
    ct.c_size_t,
    ct.c_size_t,
    _double,
]
_lib.cn_contrast_bounds_complete.restype = ct.c_int
_lib.cn_contrast_stretch_wide.argtypes = [
    ct.POINTER(ct.c_float),
    _double,
    ct.c_size_t,
    ct.c_double,
    ct.c_double,
]
_lib.cn_contrast_stretch_wide.restype = ct.c_int
_lib.cn_contrast_merge_wide.argtypes = [
    ct.POINTER(ct.c_void_p),
    ct.POINTER(ct.c_int),
    _double,
    ct.c_size_t,
    ct.c_size_t,
]
_lib.cn_contrast_merge_wide.restype = ct.c_int


def contrast_bounds(
    image: np.ndarray, percentile: float | None = None
) -> tuple[float, float]:
    if percentile is not None and not 0 <= percentile <= 50:
        raise ValueError("Contrast percentile must be in [0, 50]")
    if image.dtype != np.float32 or not image.size:
        # Preserve public dtype and empty-array exceptions outside valid node
        # images. Every nonempty float32 value, including NaN/Inf/-0, uses C.
        if percentile is None:
            return float(np.min(image)), float(np.max(image))
        return float(np.percentile(image, percentile)), float(
            np.percentile(image, 100 - percentile)
        )
    # Min/max follows physical order; percentile makes a C-order working copy.
    source = f32(np.ravel(image, order="K") if percentile is None else image)
    # The C reduction takes one width for np.min and np.max: NumPy's own
    # minimum and maximum loops, which share loops_minmax's dispatch targets.
    # A strided inner stream takes their scalar path (lanes 1).
    lanes = float32_lanes(loop_target("minimum", "fff"))
    if float32_lanes(loop_target("maximum", "fff")) != lanes:
        raise RuntimeError("NumPy dispatches minimum and maximum at different widths")
    if not numpy_reduce_contiguous(image):
        lanes = 1
    # NumPy 2.5 interpolates a Python-number percentile (its weak_q) in float32.
    mode = 0 if percentile is None else 2 if type(percentile) in (int, float) else 1
    result = np.empty(2, dtype=np.float64)
    check(
        _lib.cn_contrast_bounds_complete(
            ptr(source),
            source.size,
            mode,
            0 if percentile is None else percentile,
            lanes,
            *numpy_sum_block(image),
            result.ctypes.data_as(_double),
        )
    )
    return float(result[0]), float(result[1])


def stretch_range(image: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    if minimum > maximum:
        raise ValueError("min must be less than max")
    if minimum == maximum:
        return adjust(image, 1, 0)
    span = maximum - minimum
    if np.result_type(image, span) == np.float32:
        return adjust(image, 7, minimum, span)
    source = f32(image)
    result = np.empty(source.shape, np.float64)
    check(
        _lib.cn_contrast_stretch_wide(
            ptr(source), result.ctypes.data_as(_double), source.size, minimum, span
        )
    )
    return result


def stack_stretched(channels: list[np.ndarray]) -> np.ndarray:
    if all(channel.dtype == np.float32 for channel in channels):
        return merge_channels(*channels)
    if not 1 <= len(channels) <= 4 or any(
        channel.ndim != 2
        or channel.shape != channels[0].shape
        or not channel.size
        or channel.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
        for channel in channels
    ):
        raise ValueError("Contrast channel assembly requires matching float images")
    arrays = [np.require(channel, requirements=["C", "A"]) for channel in channels]
    pointers = (ct.c_void_p * len(arrays))(*(a.ctypes.data for a in arrays))
    kinds = (ct.c_int * len(arrays))(*(a.dtype == np.float64 for a in arrays))
    result = np.empty((*arrays[0].shape, len(arrays)), np.float64)
    check(
        _lib.cn_contrast_merge_wide(
            pointers, kinds, result.ctypes.data_as(_double), arrays[0].size, len(arrays)
        )
    )
    return result
