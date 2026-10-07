"""Validated buffers for native statistics, palettes, and filter arithmetic.

The per-pixel arithmetic, reductions and SSIM kernel construction live in C.
The convolution bridge preserves the pinned native transform dependency where
its numerical behavior is required for exact pixels.
"""

from __future__ import annotations

import ctypes as ct

import numpy as np

from .native import (
    check,
    checked_f32,
    empty_ordered,
    f32,
    lib,
    memory_order,
    numpy_order,
    ptr,
)
from .native_numpy_reduce import numpy_reduce_contiguous, numpy_sum_block
from .native_numpy_simd import float32_lanes, loop_target

_lib = lib()
_f = ct.POINTER(ct.c_float)
_d = ct.POINTER(ct.c_double)
_n = ct.c_size_t
_v = ct.c_void_p
_signatures = {
    "statistics": [_f, _f, _n, ct.c_double, ct.c_int, _n, _n, ct.c_int, _d],
    "mean": [_f, _n, _n, _n, _f],
    "grayscale": [_f, _n, _n, ct.c_float, ct.POINTER(ct.c_int)],
    "binary": [_f, _f, _f, _n, ct.c_int, ct.c_float, ct.c_float],
    "ssim": [_f, _f, _f, _f, _f, _f, _n],
    "palette": [_f, _f, _f, _n, _n, _n],
    "distinct": [_f, _f, _n, _n, ct.POINTER(_n)],
    "cas": [_v, _v, _f, _f, _v, _n, _n, ct.c_double, ct.c_int, ct.c_int],
    "correction": [_f, _f, _f, _f, _f, _f, _n, ct.c_int],
    "channel_stats": [_f, _n, _n, _n, _f],
    "transfer": [_f, _f, _v, _n, _f, _f, _f, ct.c_int, ct.c_int],
}
for _name, _args in _signatures.items():
    _function = getattr(_lib, "cn_analysis_" + _name)
    _function.argtypes = _args
    _function.restype = ct.c_int


def _image(image: np.ndarray) -> tuple[np.ndarray, int]:
    source = f32(image)
    if source.ndim not in (2, 3) or any(size == 0 for size in source.shape):
        raise ValueError(f"Invalid image shape {source.shape}")
    return source, source.shape[2] if source.ndim == 3 else 1


def statistics(image: np.ndarray, percentile: float) -> tuple[float, ...]:
    quantile_source, _ = _image(image)
    # NumPy's scalar reduction follows physical order for transposed dense arrays.
    source = f32(np.ravel(image, order="K"))
    # minimum and maximum share loops_minmax's dispatch; a strided inner stream
    # takes its scalar path (lanes 1).
    lanes = (
        float32_lanes(loop_target("minimum", "fff"))
        if numpy_reduce_contiguous(image)
        else 1
    )
    result = np.empty(4, dtype=np.float64)
    check(
        _lib.cn_analysis_statistics(
            ptr(source),
            ptr(quantile_source),
            source.size,
            percentile,
            lanes,
            *numpy_sum_block(image),
            # np.percentile: weak_q = type(q) in (int, float)
            type(percentile) in (int, float),
            result.ctypes.data_as(_d),
        )
    )
    return tuple(float(value) for value in result)


def mean(image: np.ndarray) -> np.float32:
    source = f32(np.ravel(image, order="K"))
    result = np.empty(1, dtype=np.float32)
    check(
        _lib.cn_analysis_mean(
            ptr(source), source.size, *numpy_sum_block(image), ptr(result)
        )
    )
    return result[0]


def is_grayscale(image: np.ndarray, tolerance: float) -> bool:
    source, channels = _image(image)
    result = ct.c_int()
    check(
        _lib.cn_analysis_grayscale(
            ptr(source), source.size // channels, channels, tolerance, ct.byref(result)
        )
    )
    return bool(result.value)


def binary(
    image: np.ndarray,
    other: np.ndarray,
    operation: int,
    scale: float = 0,
    threshold: float = 0,
) -> np.ndarray:
    for operand in (checked_f32(image), checked_f32(other)):
        if operand.ndim not in (2, 3) or any(size == 0 for size in operand.shape):
            raise ValueError(f"Invalid image shape {operand.shape}")
    if image.shape != other.shape:
        raise ValueError("Native image operands must have the same shape")
    # Upstream's ufuncs give order K over both operands (numpy_order). The kernel
    # runs flat in that memory order; an operand stored in another is copied.
    result = empty_ordered(image.shape, numpy_order(image, other))
    order = memory_order(result)
    source, second = (
        np.require(operand.transpose(order), requirements=["C", "A"])
        for operand in (image, other)
    )
    check(
        _lib.cn_analysis_binary(
            ptr(source),
            ptr(second),
            ptr(result.transpose(order)),
            source.size,
            operation,
            scale,
            threshold,
        )
    )
    return result


def calculate_ssim(image: np.ndarray, other: np.ndarray) -> float:
    source, channels = _image(image)
    second, second_channels = _image(other)
    if channels != 1 or second_channels != 1 or source.shape != second.shape:
        raise ValueError("SSIM requires equally shaped grayscale images")
    from .native_convolution import convolve
    from .native_gaussian import ssim_window

    window = ssim_window()
    maps = [
        f32(convolve(value, window, 0)[5:-5, 5:-5])
        for value in (
            source,
            second,
            binary(source, source, 3),
            binary(second, second, 3),
            binary(source, second, 3),
        )
    ]
    result = np.empty_like(maps[0])
    check(
        _lib.cn_analysis_ssim(*(ptr(value) for value in maps), ptr(result), result.size)
    )
    return float(mean(result))


def apply_palette(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    source, channels = _image(image)
    lookup, palette_channels = _image(palette)
    if channels != 1:
        raise ValueError("Apply Palette requires a grayscale image")
    assert lookup.shape[1] <= 2**24, (
        "Quantizing float32 values with more than 2**24 levels doesn't make sense, because only integers up to 2**24 can be represented exactly using float32."
    )
    # np.take retains a singleton channel on the index image and adds palette
    # dimensions; a 2D grayscale palette adds no trailing dimension.
    shape = source.shape + (() if lookup.ndim == 2 else (palette_channels,))
    result = np.empty(shape, dtype=np.float32)
    status = _lib.cn_analysis_palette(
        ptr(source),
        ptr(lookup),
        ptr(result),
        source.size,
        lookup.shape[1],
        palette_channels,
    )
    if status == 1:
        raise IndexError("Quantized palette index is out of bounds for axis 0")
    check(status)
    return result


def distinct_colors(image: np.ndarray) -> np.ndarray:
    source, channels = _image(image)
    result = np.empty((source.size // channels, channels), dtype=np.float32)
    count = _n()
    check(
        _lib.cn_analysis_distinct(
            ptr(source), ptr(result), result.shape[0], channels, ct.byref(count)
        )
    )
    return result[: count.value].reshape(1, count.value, channels)


def cas_mask(low: np.ndarray, high: np.ndarray, bias: float) -> np.ndarray:
    if (
        low.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
        or high.dtype != low.dtype
    ):
        raise TypeError("CAS extrema must have matching float32 or float64 types")
    if low.ndim != 2 or low.shape != high.shape or not low.size or not bias > 0:
        raise ValueError("Invalid CAS extrema or bias")
    a = np.require(low, requirements=["C", "A"])
    b = np.require(high, requirements=["C", "A"])
    result = np.empty_like(a)
    check(
        _lib.cn_analysis_cas(
            a.ctypes.data,
            b.ctypes.data,
            None,
            None,
            result.ctypes.data,
            a.size,
            1,
            1 / bias,
            a.dtype == np.float64,
            False,
        )
    )
    return result


def cas_mix(image: np.ndarray, sharpened: np.ndarray, mask: np.ndarray) -> np.ndarray:
    source, channels = _image(image)
    second, _ = _image(sharpened)
    if source.shape != second.shape or mask.shape != source.shape[:2]:
        raise ValueError("Invalid CAS image/mask shapes")
    if mask.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("CAS mask must be float32 or float64")
    weights = np.require(mask, requirements=["C", "A"])
    result = np.empty(source.shape, dtype=mask.dtype)
    check(
        _lib.cn_analysis_cas(
            weights.ctypes.data,
            None,
            ptr(source),
            ptr(second),
            result.ctypes.data,
            weights.size,
            channels,
            1,
            weights.dtype == np.float64,
            True,
        )
    )
    return result


def correction(
    image: np.ndarray,
    reference: np.ndarray,
    alpha: np.ndarray | None,
    reference_alpha: np.ndarray | None,
    *,
    add: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    source, channels = _image(image)
    second, second_channels = _image(reference)
    if channels != 3 or second_channels != 3 or source.shape != second.shape:
        raise ValueError("Correction requires equally shaped three-channel images")
    alphas = []
    for value in (alpha, reference_alpha):
        if value is None:
            alphas.append(None)
        else:
            converted, depth = _image(value)
            if depth != 1 or converted.shape[:2] != source.shape[:2]:
                raise ValueError("Correction alpha does not match the image")
            alphas.append(converted)
    result = np.empty_like(source)
    alpha_result = (
        np.empty_like(alphas[0]) if all(a is not None for a in alphas) else None
    )
    check(
        _lib.cn_analysis_correction(
            ptr(source),
            ptr(second),
            *(ptr(a) if a is not None else None for a in alphas),
            ptr(result),
            ptr(alpha_result) if alpha_result is not None else None,
            source.shape[0] * source.shape[1],
            add,
        )
    )
    return result, alpha_result


def channel_stats(image: np.ndarray) -> np.ndarray:
    source = f32(image)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Channel statistics require an (N,3) array")
    result = np.empty(6, dtype=np.float32)
    check(
        _lib.cn_analysis_channel_stats(
            ptr(source),
            source.shape[0],
            # Upstream reduces np.split(image, 3, 1)'s first column, image[:, :1].
            *numpy_sum_block(image[:, :1]),
            ptr(result),
        )
    )
    return result


def color_transfer(
    image: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    reference_valid: np.ndarray,
    limits: tuple[float, ...],
    *,
    reciprocal: bool,
    scale: bool,
) -> np.ndarray:
    source, channels = _image(image)
    second, second_channels = _image(reference)
    if channels != 3 or second_channels != 3:
        raise ValueError("Color transfer requires three-channel images")
    for mask, value in ((valid, source), (reference_valid, second)):
        if mask.dtype != np.bool_ or mask.shape != value.shape[:2]:
            raise ValueError("Color transfer masks must be matching boolean images")
    bounds = f32(np.asarray(limits, dtype=np.float32))
    if bounds.shape != (6,):
        raise ValueError("Color transfer requires six channel bounds")
    mask = np.require(valid, requirements=["C", "A"])
    stats = channel_stats(source[valid])
    reference_stats = channel_stats(second[reference_valid])
    result = np.empty_like(source)
    check(
        _lib.cn_analysis_transfer(
            ptr(source),
            ptr(result),
            mask.ctypes.data,
            mask.size,
            ptr(stats),
            ptr(reference_stats),
            ptr(bounds),
            reciprocal,
            scale,
        )
    )
    return result
