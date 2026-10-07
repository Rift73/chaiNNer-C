"""Owned-buffer bridges for transparency and color representation arithmetic.

Float32 is the workflow image format. Callers retain their original algorithms
for other dtypes explicitly; a native load or call failure is always raised.
"""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache
from typing import overload

import numpy as np

from .native import (
    check,
    checked_f32,
    empty_ordered,
    f32,
    lib,
    numpy_order,
    ordered_layout,
    pixel_source,
    pixel_target,
    ptr,
)


@lru_cache(maxsize=1)
def _api():
    library = lib()
    fp, dp, up = (ct.POINTER(t) for t in (ct.c_float, ct.c_double, ct.c_uint8))
    size, integer, real = ct.c_size_t, ct.c_int, ct.c_float
    signatures = {
        "cn_chroma_key": [
            fp,
            fp,
            size,
            size,
            size,
            real,
            real,
            integer,
            fp,
            size,
            size,
            up,
            up,
        ],
        "cn_chroma_trimap": [up, up, dp, size],
        "cn_matting_input": [
            fp,
            size,
            ct.c_void_p,
            integer,
            size,
            ct.c_double,
            ct.c_double,
            integer,
            dp,
            dp,
        ],
        "cn_color_representation": [fp, fp, size, integer, fp, fp, fp, fp],
    }
    for name, signature in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = signature, integer
    return library


def _image(image: np.ndarray, channels: tuple[int, ...]) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] not in channels or 0 in image.shape:
        raise ValueError("Invalid image shape for transparency/color kernel")
    return f32(image)


@overload
def chroma_key(
    image: np.ndarray,
    color: tuple[float, ...] | np.ndarray,
    background: float,
    foreground: None = None,
) -> np.ndarray: ...


@overload
def chroma_key(
    image: np.ndarray,
    color: tuple[float, ...] | np.ndarray,
    background: float,
    foreground: float,
) -> tuple[np.ndarray, np.ndarray]: ...


def chroma_key(
    image: np.ndarray,
    color: tuple[float, ...] | np.ndarray,
    background: float,
    foreground: float | None = None,
):
    if image.ndim != 3 or image.shape[2] != 3 or 0 in image.shape:
        raise ValueError("Invalid image shape for transparency/color kernel")
    source, pixel, channel = pixel_source(checked_f32(image))
    key = np.require(color, dtype=np.float32, requirements=["C", "A"])
    if key.shape != (3,):
        raise ValueError("Chroma key must have three channels")
    h, w, _ = image.shape
    if foreground is None:
        # Upstream's np.dstack([img, alpha]), alpha C-ordered: img's order K.
        out = empty_ordered(
            (h, w, 4), numpy_order(image, ordered_layout((h, w, 1), (0, 1, 2)))
        )
        target, out_pixel, out_channel = pixel_target(out)
        check(
            _api().cn_chroma_key(
                ptr(source),
                ptr(key),
                h * w,
                pixel,
                channel,
                background,
                0,
                0,
                ptr(target),
                out_pixel,
                out_channel,
                None,
                None,
            )
        )
        if target is not out:
            np.copyto(out, target)
        return out
    bg, fg = (np.empty((h, w), np.uint8) for _ in range(2))
    up = ct.POINTER(ct.c_uint8)
    check(
        _api().cn_chroma_key(
            ptr(source),
            ptr(key),
            h * w,
            pixel,
            channel,
            background,
            foreground,
            1,
            None,
            0,
            0,
            bg.ctypes.data_as(up),
            fg.ctypes.data_as(up),
        )
    )
    return bg, fg


def chroma_trimap(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    if (
        background.ndim != 2
        or 0 in background.shape
        or foreground.shape != background.shape
        or background.dtype != np.uint8
        or foreground.dtype != np.uint8
    ):
        raise ValueError("Trimap masks must be matching uint8 planes")
    bg, fg = (np.require(a, requirements=["C", "A"]) for a in (background, foreground))
    out = np.empty(bg.shape, np.float64)
    up, dp = ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_double)
    check(
        _api().cn_chroma_trimap(
            bg.ctypes.data_as(up),
            fg.ctypes.data_as(up),
            out.ctypes.data_as(dp),
            out.size,
        )
    )
    return out


def matting_input(
    image: np.ndarray,
    trimap: np.ndarray,
    foreground: float | None = None,
    background: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    source = _image(image, (3, 4))
    if trimap.shape not in (
        source.shape[:2],
        (*source.shape[:2], 1),
    ) or trimap.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Matting requires a matching float32/float64 trimap")
    trimap = np.require(trimap, requirements=["C", "A"])
    rgb = np.empty((*source.shape[:2], 3), np.float64)
    out = np.empty(trimap.shape, np.float64)
    dp = ct.POINTER(ct.c_double)
    check(
        _api().cn_matting_input(
            ptr(source),
            source.shape[2],
            trimap.ctypes.data,
            int(trimap.dtype == np.float64),
            trimap.size,
            foreground or 0,
            background,
            int(foreground is not None),
            rgb.ctypes.data_as(dp),
            out.ctypes.data_as(dp),
        )
    )
    return rgb, out


def representation(
    image: np.ndarray, operation: int, *components: np.ndarray
) -> np.ndarray:
    if operation not in range(10):
        raise ValueError("Unknown color representation")
    source = _image(image, (4,) if operation == 1 else (3,))
    count = 4 if operation == 8 else 2 if operation == 9 else 0
    if len(components) != count or any(a.shape != source.shape[:2] for a in components):
        raise ValueError("Color representation components must match the image")
    buffers = [f32(a) for a in components]
    out = np.empty((*source.shape[:2], 4 if operation == 0 else 3), np.float32)
    pointers = [ptr(a) for a in buffers] + [None] * (4 - count)
    check(
        _api().cn_color_representation(
            ptr(source),
            ptr(out),
            source.shape[0] * source.shape[1],
            operation,
            *pointers,
        )
    )
    return out
