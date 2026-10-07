"""Scalar geometry and owned buffers for the complete C Blend Images canvas."""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import numpy as np

from .color.color import Color
from .native import check, f32, lib, ptr


class _Layer(ct.Structure):
    _fields_ = [
        ("data", ct.POINTER(ct.c_float)),
        ("height", ct.c_size_t),
        ("width", ct.c_size_t),
        ("channels", ct.c_size_t),
        ("constant", ct.c_int),
    ]


class _Geometry(ct.Structure):
    _fields_ = [
        (name, ct.c_size_t)
        for name in (
            "height",
            "width",
            "channels",
            "base_top",
            "base_left",
            "paste_top",
            "paste_left",
            "overlay_top",
            "overlay_left",
            "region_height",
            "region_width",
        )
    ] + [("padded", ct.c_int)]


@lru_cache(maxsize=1)
def _api():
    function = lib().cn_composite_canvas
    function.argtypes = [
        ct.POINTER(_Layer),
        ct.POINTER(_Layer),
        ct.POINTER(_Geometry),
        ct.POINTER(ct.c_float),
        ct.c_int,
    ]
    function.restype = ct.c_int
    return function


def dimensions(base: np.ndarray | Color, overlay: np.ndarray | Color):
    if isinstance(base, Color) and isinstance(overlay, Color):
        raise ValueError("At least one layer must be an image")
    reference = overlay if isinstance(base, Color) else base
    assert isinstance(reference, np.ndarray)
    for image in (base, overlay):
        if isinstance(image, np.ndarray) and (
            image.ndim not in (2, 3) or not all(image.shape)
        ):
            raise ValueError(
                "Blend images must be nonempty two- or three-dimensional arrays"
            )
    return tuple(
        (*reference.shape[:2], image.channels)
        if isinstance(image, Color)
        else (*image.shape[:2], image.shape[2] if image.ndim == 3 else 1)
        for image in (base, overlay)
    )


def canvas(
    base: np.ndarray | Color,
    overlay: np.ndarray | Color,
    mode: int,
    x: int,
    y: int,
    crop: bool,
) -> np.ndarray:
    (bh, bw, bc), (oh, ow, oc) = dimensions(base, overlay)
    if isinstance(base, Color) or isinstance(overlay, Color):
        crop = False
    for channels, name in ((bc, "base layer"), (oc, "overlay layer")):
        assert channels in (1, 3, 4), (
            f"The {name} has to be a grayscale, RGB, or RGBA image"
        )
    x, y = int(x), int(y)
    left, top = (0, 0) if crop else (max(0, -x), max(0, -y))
    right, bottom = (0, 0) if crop else (max(0, x + ow - bw), max(0, y + oh - bh))
    padded = any((left, top, right, bottom))
    if max(left, top, right, bottom) > 2**31 - 1:
        raise OverflowError("Blend canvas border exceeds OpenCV's integer range")
    h, w = bh + top + bottom, bw + left + right
    pt, pl = (max(0, y), max(0, x)) if crop else (y + top, x + left)
    rh, rw = (
        (max(0, min(y + oh, bh) - pt), max(0, min(x + ow, bw) - pl))
        if crop
        else (oh, ow)
    )
    if not rh or not rw:
        rh = rw = pt = pl = 0
    ot, ol = (max(0, -y), max(0, -x)) if crop and rh else (0, 0)
    cc = 4 if padded else bc
    channels = max(cc, oc) if rh else cc
    arrays = [
        np.array(value.value, np.float32) if isinstance(value, Color) else f32(value)
        for value in (base, overlay)
    ]
    layers = [
        _Layer(ptr(array), ih, iw, ic, isinstance(value, Color))
        for array, value, (ih, iw, ic) in zip(
            arrays, (base, overlay), ((bh, bw, bc), (oh, ow, oc)), strict=True
        )
    ]
    keep_singleton = isinstance(base, np.ndarray) and base.ndim == 3 and channels == 1
    shape = (h, w, channels) if channels != 1 or keep_singleton else (h, w)
    out = np.empty(shape, np.float32)
    geometry = _Geometry(h, w, channels, top, left, pt, pl, ot, ol, rh, rw, padded)
    check(
        _api()(
            ct.byref(layers[0]), ct.byref(layers[1]), ct.byref(geometry), ptr(out), mode
        )
    )
    return out
