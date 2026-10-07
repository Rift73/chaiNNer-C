"""Owned byte buffers for the complete CPU non-local means implementation."""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache
from operator import index

import numpy as np

from .native import check, lib


@lru_cache(maxsize=1)
def _api():
    library = lib()
    common = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_size_t]
    library.cn_denoise_lab_u8.argtypes = [*common, ct.c_int]
    library.cn_denoise_nlm_u8.argtypes = [*common, ct.c_float, ct.c_int, ct.c_int]
    library.cn_denoise_u8.argtypes = [
        *common,
        ct.c_float,
        ct.c_float,
        ct.c_int,
        ct.c_int,
    ]
    for name in ("cn_denoise_lab_u8", "cn_denoise_nlm_u8", "cn_denoise_u8"):
        getattr(library, name).restype = ct.c_int
    return library


def _image(image: np.ndarray) -> tuple[np.ndarray, int]:
    if image.dtype != np.uint8:
        raise TypeError("Non-local means requires an unsigned-byte image")
    if image.ndim not in (2, 3) or not image.size:
        raise ValueError("Non-local means requires a nonempty image")
    channels = image.shape[2] if image.ndim == 3 else 1
    if channels not in (1, 2, 3, 4):
        raise ValueError("Non-local means requires one to four channels")
    return np.require(image, requirements=["C", "A"]), channels


def _radii(patch: int, search: int) -> tuple[int, int]:
    patch, search = index(patch), index(search)
    if not 1 <= patch <= 30 or not 1 <= search <= 30:
        raise ValueError("Non-local means radii must be between 1 and 30")
    return patch, search


def lab(image: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    source, channels = _image(image)
    result = np.empty_like(source)
    check(
        _api().cn_denoise_lab_u8(
            source.ctypes.data, result.ctypes.data, *source.shape[:2], channels, inverse
        )
    )
    return result


def nlm(image: np.ndarray, strength: float, patch: int, search: int) -> np.ndarray:
    patch, search = _radii(patch, search)
    source, channels = _image(image)
    shape = source.shape[:2] if channels == 1 else source.shape
    result = np.empty(shape, np.uint8)
    check(
        _api().cn_denoise_nlm_u8(
            source.ctypes.data,
            result.ctypes.data,
            *source.shape[:2],
            channels,
            strength,
            patch,
            search,
        )
    )
    return result


def denoise(
    image: np.ndarray, strength: float, color_strength: float, patch: int, search: int
) -> np.ndarray:
    patch, search = _radii(patch, search)
    source, channels = _image(image)
    shape = source.shape[:2] if channels == 1 else source.shape
    result = np.empty(shape, np.uint8)
    check(
        _api().cn_denoise_u8(
            source.ctypes.data,
            result.ctypes.data,
            *source.shape[:2],
            channels,
            strength,
            color_strength,
            patch,
            search,
        )
    )
    return result
