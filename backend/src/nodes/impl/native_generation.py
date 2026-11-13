"""Checked bridge for the fused image fill kernel (colour, checkerboard, gradient)."""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import numpy as np

from .native import check, f32, lib, ptr


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    fp, size = ct.POINTER(ct.c_float), ct.c_size_t
    dll.cn_image_fill.argtypes = [fp, size, size, size, fp, fp, fp, ct.c_int, size]
    dll.cn_image_fill.restype = ct.c_int
    return dll


def image_fill(
    width: int,
    height: int,
    color: tuple[float, ...] | np.ndarray,
    *,
    second: tuple[float, ...] | np.ndarray | None = None,
    square: int = 0,
    gradient: np.ndarray | None = None,
    keep_channel: bool = False,
) -> np.ndarray:
    if width <= 0 or height <= 0 or len(color) not in (1, 3, 4):
        raise ValueError("Invalid image dimensions or color")
    a = np.require(color, dtype=np.float32, requirements=["C", "A"])
    b = (
        np.require(second, dtype=np.float32, requirements=["C", "A"])
        if second is not None
        else None
    )
    if a.ndim != 1 or (b is not None and b.shape != a.shape):
        raise ValueError("Color channels must match")
    if gradient is not None:
        gradient = f32(gradient)
        if gradient.shape != (height, width) or b is None:
            raise ValueError("Gradient dimensions and colors must match")
        mode = 2
    elif b is not None:
        if square <= 0:
            raise ValueError("Checkerboard square size must be positive")
        mode = 1
    else:
        mode = 0
    shape = (
        (height, width, len(color))
        if len(color) > 1 or keep_channel
        else (height, width)
    )
    out = np.empty(shape, dtype=np.float32)
    check(
        _api().cn_image_fill(
            ptr(out),
            height,
            width,
            len(color),
            ptr(a),
            ptr(b) if b is not None else None,
            ptr(gradient) if gradient is not None else None,
            mode,
            square,
        )
    )
    return out
