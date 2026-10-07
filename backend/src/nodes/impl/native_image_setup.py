"""Small bounded image preparation predicates implemented in C."""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import numpy as np

from .native import check, f32, lib, ptr


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    dll.cn_image_exceptional.argtypes = [ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_int)]
    dll.cn_image_exceptional.restype = ct.c_int
    dll.cn_image_distinct_exceeds.argtypes = (
        [ct.c_void_p] + [ct.c_size_t] * 3 + [ct.POINTER(ct.c_int)]
    )
    dll.cn_image_distinct_exceeds.restype = ct.c_int
    return dll


def exceptional(image: np.ndarray) -> bool:
    source = f32(image)
    result = ct.c_int()
    check(_api().cn_image_exceptional(ptr(source), source.size, ct.byref(result)))
    return bool(result.value)


def distinct_exceeds(image: np.ndarray, limit: int) -> bool:
    source = f32(image)
    if source.ndim not in (2, 3) or not all(source.shape):
        raise ValueError("Distinct colors require a nonempty image")
    if not 0 <= limit <= 4096:
        raise ValueError("Distinct color limit must be in [0,4096]")
    channels = source.shape[2] if source.ndim == 3 else 1
    result = ct.c_int()
    check(
        _api().cn_image_distinct_exceeds(
            ptr(source),
            source.shape[0] * source.shape[1],
            channels,
            limit,
            ct.byref(result),
        )
    )
    return bool(result.value)
