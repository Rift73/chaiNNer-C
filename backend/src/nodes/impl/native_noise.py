"""Seeded Add Noise and ndarray permutation using NumPy 1.24.4 C streams."""

from __future__ import annotations

import ctypes as ct
import operator
from functools import lru_cache

import numpy as np

from .native import check, f32, lib, ptr


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    fp, up = ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint32)
    dll.cn_add_seeded_noise.argtypes = [
        fp,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_int,
        ct.c_double,
        up,
        ct.c_size_t,
        fp,
    ]
    dll.cn_add_seeded_noise.restype = ct.c_int
    dll.cn_seeded_permutation.argtypes = [
        up,
        ct.c_size_t,
        ct.POINTER(ct.c_int64),
        ct.c_size_t,
    ]
    dll.cn_seeded_permutation.restype = ct.c_int
    return dll


def seed_words(seed: int) -> np.ndarray:
    """Marshal an integer's base-2**32 digits; all seed mixing happens in C."""
    value = operator.index(seed)
    if value < 0:
        raise ValueError("expected non-negative integer")
    count = max(1, (value.bit_length() + 31) // 32)
    return np.frombuffer(value.to_bytes(count * 4, "little"), dtype=np.uint32)


def seeded_permutation(seed: int, count: int) -> np.ndarray:
    count = operator.index(count)
    if count < 0:
        raise ValueError("Permutation length must be nonnegative")
    words = seed_words(seed)
    out = np.empty(count, dtype=np.int64)
    check(
        _api().cn_seeded_permutation(
            words.ctypes.data_as(ct.POINTER(ct.c_uint32)),
            words.size,
            out.ctypes.data_as(ct.POINTER(ct.c_int64)),
            count,
        )
    )
    return out


def add_noise(
    image: np.ndarray, amount: float, channels: int, seed: int, mode: int
) -> np.ndarray:
    words = seed_words(seed)
    image = f32(image)
    if image.ndim not in (2, 3) or 0 in image.shape:
        raise ValueError("Expected a nonempty image")
    source_channels = image.shape[2] if image.ndim == 3 else 1
    assert source_channels != 2, "Noise cannot be added to 2-channel images."
    if channels not in (1, 3) or mode not in range(5):
        raise ValueError("Invalid noise channels or mode")
    if not np.isfinite(amount) or not 0 <= amount <= 1:
        raise ValueError("Noise amount must be finite and between 0 and 1")
    if np.signbit(amount) and mode in (0, 1, 4):
        raise ValueError("high - low < 0" if mode == 1 else "scale < 0")
    out_channels = max(source_channels, channels)
    shape = (*image.shape[:2], out_channels) if out_channels > 1 else image.shape[:2]
    out = np.empty(shape, dtype=np.float32)
    check(
        _api().cn_add_seeded_noise(
            ptr(image),
            image.shape[0] * image.shape[1],
            source_channels,
            channels,
            mode,
            amount,
            words.ctypes.data_as(ct.POINTER(ct.c_uint32)),
            words.size,
            ptr(out),
        )
    )
    return out
