"""The installed eleven float32 pixel-art algorithms, executed by C++ kernels."""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import numpy as np

from .native import check, f32, lib, ptr

_MODES = {
    ("adv_mame", 2): 0,
    ("adv_mame", 3): 1,
    ("adv_mame", 4): 2,
    ("eagle", 2): 3,
    ("eagle", 3): 4,
    ("super_eagle", 2): 5,
    ("sai", 2): 6,
    ("super_sai", 2): 7,
    ("hqx", 2): 8,
    ("hqx", 3): 9,
    ("hqx", 4): 10,
}


@lru_cache(maxsize=1)
def _api():
    api = lib()
    fp = ct.POINTER(ct.c_float)
    api.cn_pixel_art_f32.argtypes = [fp, fp] + [ct.c_size_t] * 3 + [ct.c_int]
    api.cn_pixel_art_f32.restype = ct.c_int
    return api


def pixel_art_upscale(image: np.ndarray, algorithm: str, scale: int) -> np.ndarray:
    if image.dtype != np.float32:
        raise TypeError("Pixel-art images must have dtype float32")
    if image.ndim not in (2, 3):
        raise ValueError("Pixel-art images must have two or three dimensions")
    channels = image.shape[2] if image.ndim == 3 else 1
    if channels not in (1, 3, 4):
        raise ValueError(
            f"Argument 'img' does not have the right shape. Expected 1, 3, or 4 channels but found {channels}."
        )
    height, width = image.shape[:2]
    if not height or not width:
        raise ValueError("Pixel-art images must be nonempty")
    mode = _MODES.get((algorithm, scale))
    if mode is None:
        if algorithm in {key[0] for key in _MODES}:
            raise ValueError(
                f"Scale {scale} is not supported for pixel art upscaling algorithm '{algorithm}'."
            )
        raise ValueError(f"Unknown pixel art upscaling algorithm '{algorithm}'.")
    source = f32(image)
    # Rust's binding returns PyArray3 for grayscale too; node output enforcement
    # performs the established one-channel view normalization.
    result = np.empty((height * scale, width * scale, channels), np.float32)
    check(
        _api().cn_pixel_art_f32(ptr(source), ptr(result), height, width, channels, mode)
    )
    return result
