"""C Create Noise pipeline; Python marshals scalar settings and owns the image."""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import numpy as np

from .native import check, lib


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    dll.cn_create_procedural.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_uint64,
        ct.c_int,
        ct.c_double,
        ct.c_double,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_double,
        ct.c_double,
        ct.c_int,
    ]
    dll.cn_create_procedural.restype = ct.c_int
    return dll


def create(
    width: int,
    height: int,
    seed: int,
    method: int,
    scale: float,
    brightness: float,
    horizontal: bool,
    vertical: bool,
    spherical: bool,
    layers: int,
    scale_ratio: float,
    brightness_ratio: float,
    increment_seed: bool,
) -> np.ndarray:
    # Preserve Python's scalar overflow before entering C; no image arithmetic.
    if layers:
        _ = scale_ratio ** (layers - 1)
        _ = brightness_ratio ** (layers - 1)
    out = np.empty((height, width), dtype=np.float32 if layers else np.float64)
    check(
        _api().cn_create_procedural(
            out.ctypes.data,
            height,
            width,
            seed,
            method,
            scale,
            brightness,
            int(horizontal),
            int(vertical),
            int(spherical),
            layers,
            scale_ratio,
            brightness_ratio,
            int(increment_seed),
        )
    )
    return out
