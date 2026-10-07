"""Checked buffer ownership for resize and border-detection C kernels."""

from __future__ import annotations

import ctypes as ct
import operator
from functools import lru_cache

import numpy as np

from .native import check, f32, lib, ptr


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    fp, size, integer = ct.POINTER(ct.c_float), ct.c_size_t, ct.c_int
    signatures = {
        "cn_resample_nearest": [fp, fp, *([size] * 5)],
        "cn_resample_filtered": [fp, fp, *([size] * 5), integer, integer, integer],
        "cn_resample_alpha": [fp, fp, size, integer],
        "cn_resample_border": [
            fp,
            size,
            size,
            size,
            ct.c_float,
            integer,
            ct.POINTER(size),
            integer,
        ],
    }
    for name, signature in signatures.items():
        function = getattr(dll, name)
        function.argtypes, function.restype = signature, integer
    return dll


def _shape(image: object) -> tuple[int, int, int]:
    if not isinstance(image, np.ndarray) or image.dtype != np.float32:
        raise TypeError("C resampling requires a native float32 array")
    if image.ndim not in (2, 3) or 0 in image.shape:
        raise ValueError("Expected a nonempty two- or three-dimensional image")
    h, w = image.shape[:2]
    c = image.shape[2] if image.ndim == 3 else 1
    if c not in (1, 2, 3, 4):
        raise ValueError("Expected one to four channels")
    return h, w, c


def nearest(image: np.ndarray, dimensions: tuple[int, int]) -> np.ndarray:
    h, w, c = _shape(image)
    out_w, out_h = (operator.index(value) for value in dimensions)
    if min(out_h, out_w) < 1:
        raise ValueError("Resize dimensions must be positive")
    if max(h, w) > 2**31 - 1 or max(out_h, out_w) > 2**32 - 1:
        raise OverflowError("Resize dimensions exceed the native coordinate range")
    if out_h * out_w * c * 4 > 16 * 2**30:
        raise ValueError("Resize output exceeds the 16 GiB image limit")
    source = f32(image)
    out = np.empty((out_h, out_w, c), np.float32)
    check(_api().cn_resample_nearest(ptr(source), ptr(out), h, w, c, out_h, out_w))
    return out


def filtered(
    image: np.ndarray,
    dimensions: tuple[int, int],
    filter_id: int,
    gamma_correction: bool,
) -> np.ndarray:
    """Two-pass C reconstruction with the installed coefficient/gamma rules."""
    h, w, c = _shape(image)
    out_w, out_h = (operator.index(value) for value in dimensions)
    filter_id = operator.index(filter_id)
    if filter_id not in range(1, 12):
        raise ValueError("Unknown filtered resize method")
    if min(out_h, out_w) < 0:
        raise OverflowError("Resize dimensions cannot be negative")
    if max(h, w) > 2**31 - 1 or max(out_h, out_w) > 2**32 - 1:
        raise OverflowError("Resize dimensions exceed the native coordinate range")
    if out_h * out_w * c * 4 > 16 * 2**30:
        raise ValueError("Resize output exceeds the 16 GiB image limit")
    out = np.empty((out_h, out_w, c), np.float32)
    if out.size == 0:
        return out
    # The original binding chooses Vec4 only for strided RGBA upscales. Its
    # unordered clamp differs from scalar clamp; keep that observable behavior.
    vector_clip = (
        c == 4 and not image.flags.c_contiguous and (out_h * out_w) / (h * w) >= 1.99
    )
    source = f32(image)
    check(
        _api().cn_resample_filtered(
            ptr(source),
            ptr(out),
            h,
            w,
            c,
            out_h,
            out_w,
            filter_id,
            bool(gamma_correction),
            vector_clip,
        )
    )
    return out


def premultiply(image: np.ndarray) -> np.ndarray:
    h, w, c = _shape(image)
    if c != 4:
        raise ValueError("Alpha resampling requires four channels")
    source = f32(image)
    out = np.empty(source.shape, np.float32)
    check(_api().cn_resample_alpha(ptr(source), ptr(out), h * w, 0))
    return out


def finish_alpha_inplace(image: np.ndarray) -> np.ndarray:
    """Finish an owned resampler result; the caller must not pass graph input."""
    h, w, c = _shape(image)
    if c != 4 or not image.flags.writeable:
        raise ValueError("Expected a writable four-channel resampler result")
    out = f32(image)
    check(_api().cn_resample_alpha(ptr(out), ptr(out), h * w, 1))
    return out


def border_region(
    image: np.ndarray, tolerance: float, selection: int
) -> tuple[int, int, int, int]:
    if image.dtype != np.float32 or image.ndim not in (2, 3) or not all(image.shape):
        raise ValueError("Border detection requires a nonempty float32 image")
    h, w = image.shape[:2]
    c = image.shape[2] if image.ndim == 3 else 1
    if selection not in (1, 2, 3):
        raise ValueError("Unknown border section selection")
    source = f32(image)
    # Match the original difference array's keep-order layout. This tiny
    # allocation determines reduction traversal; it never copies image pixels.
    layout = np.empty_like(
        image, shape=tuple(min(n, 2) for n in image.shape), order="K"
    )
    pairwise = image.ndim == 2 or layout.strides[-1] == image.itemsize
    bounds = (ct.c_size_t * 4)()
    check(
        _api().cn_resample_border(
            ptr(source), h, w, c, tolerance, selection, bounds, pairwise
        )
    )
    return bounds[0], bounds[1], bounds[2], bounds[3]
