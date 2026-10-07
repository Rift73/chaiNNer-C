"""Owned float32 buffers for exact Gamma and Fill Alpha C kernels."""

from __future__ import annotations

import ctypes as ct
import operator
from functools import lru_cache

import numpy as np

from chainner_ext.chainner_ext import PanicException

from .native import check, f32, lib, ptr


@lru_cache(maxsize=1)
def _api():
    library = lib()
    fp, size = ct.POINTER(ct.c_float), ct.c_size_t
    library.cn_alpha_gamma.argtypes = [fp, fp, size, size, ct.c_float]
    library.cn_alpha_gamma.restype = ct.c_int
    library.cn_alpha_extend.argtypes = [fp, fp, size, size, ct.c_float, size]
    library.cn_alpha_extend.restype = ct.c_int
    library.cn_alpha_fragment.argtypes = [fp, fp, size, size, ct.c_float, size, size]
    library.cn_alpha_fragment.restype = ct.c_int
    library.cn_alpha_nearest.argtypes = [fp, fp, size, size, ct.c_float, size, ct.c_int]
    library.cn_alpha_nearest.restype = ct.c_int
    library.cn_gamma_uses_approximation.argtypes = []
    library.cn_gamma_uses_approximation.restype = ct.c_int
    return library


def _image(image: np.ndarray) -> tuple[np.ndarray, int]:
    source = f32(image)
    if source.ndim not in (2, 3):
        raise ValueError("Expected a two- or three-dimensional image")
    channels = source.shape[2] if source.ndim == 3 else 1
    if not channels:
        raise ValueError("An image must have at least one channel")
    return source, channels


def fast_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    source, channels = _image(image)
    result = np.empty((*source.shape[:2], channels), np.float32)
    check(
        _api().cn_alpha_gamma(
            ptr(source), ptr(result), source.shape[0] * source.shape[1], channels, gamma
        )
    )
    return result


def fill_alpha_extend_color(
    image: np.ndarray, threshold: float, iterations: int
) -> np.ndarray:
    source, channels = _image(image)
    if channels != 4:
        raise ValueError("Alpha extension requires four channels")
    iterations = operator.index(iterations)
    if not 0 <= iterations <= 0xFFFFFFFF:
        raise ValueError(
            "Alpha-extension iterations must fit an unsigned 32-bit integer"
        )
    result = np.empty(source.shape, np.float32)
    check(
        _api().cn_alpha_extend(
            ptr(source), ptr(result), *source.shape[:2], threshold, iterations
        )
    )
    return result


def _unsigned(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"argument '{name}': {error}") from error
    if result < 0:
        raise OverflowError("can't convert negative int to unsigned")
    if result > 0xFFFFFFFF:
        raise OverflowError("out of range integral type conversion attempted")
    return result


def fill_alpha_fragment_blur(
    image: np.ndarray, threshold: float, iterations: int, fragment_count: int
) -> np.ndarray:
    source, channels = _image(image)
    if channels != 4:
        raise ValueError("Alpha extension requires four channels")
    iterations = _unsigned(iterations, "iterations")
    fragment_count = _unsigned(fragment_count, "fragment_count")
    if iterations and not 1 <= fragment_count <= 255:
        # The original Rust kernel asserted these bounds and panicked. Raise
        # chainner_ext's pyo3_runtime.PanicException with its message; no image
        # kernel executes.
        bound = "count >= 1" if fragment_count < 1 else "count <= 255"
        raise PanicException(f"assertion failed: {bound}")
    result = np.empty(source.shape, np.float32)
    check(
        _api().cn_alpha_fragment(
            ptr(source),
            ptr(result),
            *source.shape[:2],
            threshold,
            iterations,
            fragment_count,
        )
    )
    return result


def fill_alpha_nearest_color(
    image: np.ndarray, threshold: float, min_radius: int, anti_aliasing: bool
) -> np.ndarray:
    source, channels = _image(image)
    if channels != 4:
        raise ValueError("Alpha extension requires four channels")
    min_radius = _unsigned(min_radius, "min_radius")
    if type(anti_aliasing) is not bool:
        raise TypeError(
            f"argument 'anti_aliasing': '{type(anti_aliasing).__name__}' object "
            "cannot be converted to 'PyBool'"
        )
    result = np.empty(source.shape, np.float32)
    check(
        _api().cn_alpha_nearest(
            ptr(source),
            ptr(result),
            *source.shape[:2],
            threshold,
            min_radius,
            anti_aliasing,
        )
    )
    return result
