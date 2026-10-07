"""Pillow-compatible byte affine sampling implemented in C.

Only the six-scalar transform construction remains Python, preserving Python's
decimal rounding and the same output dimensions used by the node's UI types.
Pixel coordinate evaluation, sampling, alpha conversion and normalization are C.
Transform construction adapts PIL.Image.Image.rotate from Pillow 9.2.0;
the HPND copyright and permission notice is included in native/src/rotate_ops.c.
"""

from __future__ import annotations

import ctypes as ct
import math
from functools import lru_cache

import numpy as np

from .native import check, lib, ptr


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    dll.cn_rotate_affine_u8.argtypes = [
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_float),
        *([ct.c_size_t] * 5),
        ct.POINTER(ct.c_double),
        ct.c_int,
        ct.POINTER(ct.c_uint8),
        ct.c_int,
    ]
    dll.cn_rotate_affine_u8.restype = ct.c_int
    return dll


def rotate(
    image: object,
    angle: float,
    interpolation: int,
    expand: bool,
    fill: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("C affine rotation requires a uint8 image")
    if image.ndim not in (2, 3):
        raise ValueError("Expected a two- or three-dimensional image")
    h, w = image.shape[:2]
    c = image.shape[2] if image.ndim == 3 else 1
    if image.ndim == 3 and c not in (2, 3, 4):
        raise TypeError(f"Cannot handle this data type: (1, 1, {c}), |u1")
    if not h or not w:
        raise ValueError("Expected a nonempty image")
    if max(h, w) > 2**31 - 1:
        raise OverflowError("Image dimensions exceed the affine coordinate range")
    if len(fill) != c or any(not 0 <= v <= 255 for v in fill):
        raise ValueError("Expected one byte fill value per image channel")
    if interpolation not in (0, 2, 3):
        raise ValueError("Expected nearest, linear or cubic rotation interpolation")
    angle %= 360.0
    quarter = -1
    if angle in (0, 180) or (angle in (90, 270) and (expand or w == h)):
        quarter = int(angle // 90)
        out_w, out_h = (h, w) if quarter % 2 else (w, h)
        matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    else:
        angle = -math.radians(angle)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x: float, y: float) -> tuple[float, float]:
            a, b, c, d, e, f = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(-w / 2.0, -h / 2.0)
        matrix[2] += w / 2.0
        matrix[5] += h / 2.0
        out_w, out_h = w, h
        if expand:
            corners = [transform(x, y) for x, y in ((0, 0), (w, 0), (w, h), (0, h))]
            xx, yy = zip(*corners, strict=True)
            out_w = math.ceil(max(xx)) - math.floor(min(xx))
            out_h = math.ceil(max(yy)) - math.floor(min(yy))
            matrix[2], matrix[5] = transform(-(out_w - w) / 2.0, -(out_h - h) / 2.0)
    if not all(math.isfinite(value) for value in matrix):
        raise ValueError("Rotation transform must be finite")
    if max(out_h, out_w) > 2**31 - 1:
        raise OverflowError("Rotation output exceeds the affine coordinate range")
    source = np.require(image, requirements=["C", "A"])
    out = np.empty((out_h, out_w) if c == 1 else (out_h, out_w, c), np.float32)
    coefficients = (ct.c_double * 6)(*matrix)
    color = (ct.c_uint8 * c)(*fill)
    check(
        _api().cn_rotate_affine_u8(
            source.ctypes.data_as(ct.POINTER(ct.c_uint8)),
            ptr(out),
            h,
            w,
            c,
            out_h,
            out_w,
            coefficients,
            interpolation,
            color,
            quarter,
        )
    )
    return out
