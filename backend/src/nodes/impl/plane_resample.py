"""Resampling of one float32 plane with zimg's semantics (VapourSynth's resize), so chroma
planes are scaled and sited as VapourSynth does it.

Each axis is a separate pass of per-tap gathers. Output sample i (centre i + 0.5) maps to
input position (i + 0.5) * src_width / width + src_left. When downscaling, the kernel is
widened by the scale factor; each output's taps are normalised to sum to 1, and taps
outside the plane are mirrored about its edge (then clamped to it). An axis whose size
and source window do not change is copied, as zimg skips it. Values are not clipped.

The passes run in C (native_planes.resample_axis) with each geometry's taps computed
once here and cached; native/tests/reference_yuv_planes keeps the NumPy passes they
match bit for bit."""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import Enum
from functools import lru_cache

import numpy as np

from . import native_planes


class Kernel(Enum):
    POINT = "point"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    SPLINE16 = "spline16"
    SPLINE36 = "spline36"
    LANCZOS = "lanczos"


KERNEL_LABELS = {
    Kernel.POINT: "Point",
    Kernel.BILINEAR: "Bilinear",
    Kernel.BICUBIC: "Bicubic (Catmull-Rom)",
    Kernel.SPLINE16: "Spline16",
    Kernel.SPLINE36: "Spline36",
    Kernel.LANCZOS: "Lanczos (3 taps)",
}


def _poly3(x: np.ndarray, c0: float, c1: float, c2: float, c3: float) -> np.ndarray:
    return c0 + x * (c1 + x * (c2 + x * c3))


def _point(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


def _bilinear(x: np.ndarray) -> np.ndarray:
    return np.maximum(1 - np.abs(x), 0)


# VapourSynth's default bicubic: b = 0, c = 0.5 (Catmull-Rom).
_B, _C = 0.0, 0.5


def _bicubic(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    near = _poly3(
        x, (6 - 2 * _B) / 6, 0, (-18 + 12 * _B + 6 * _C) / 6, (12 - 9 * _B - 6 * _C) / 6
    )
    far = _poly3(
        x,
        (8 * _B + 24 * _C) / 6,
        (-12 * _B - 48 * _C) / 6,
        (6 * _B + 30 * _C) / 6,
        (-_B - 6 * _C) / 6,
    )
    return np.where(x < 1, near, np.where(x < 2, far, 0))


def _spline16(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    return np.where(
        x < 1,
        _poly3(x, 1, -1 / 5, -9 / 5, 1),
        np.where(x < 2, _poly3(x - 1, 0, -7 / 15, 4 / 5, -1 / 3), 0),
    )


def _spline36(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    return np.where(
        x < 1,
        _poly3(x, 1, -3 / 209, -453 / 209, 13 / 11),
        np.where(
            x < 2,
            _poly3(x - 1, 0, -156 / 209, 270 / 209, -6 / 11),
            np.where(x < 3, _poly3(x - 2, 0, 26 / 209, -45 / 209, 1 / 11), 0),
        ),
    )


def _lanczos(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    return np.where(x < 3, np.sinc(x) * np.sinc(x / 3), 0)


# (support in input pixels, kernel)
_KERNELS: dict[Kernel, tuple[int, Callable[[np.ndarray], np.ndarray]]] = {
    Kernel.POINT: (0, _point),
    Kernel.BILINEAR: (1, _bilinear),
    Kernel.BICUBIC: (2, _bicubic),
    Kernel.SPLINE16: (2, _spline16),
    Kernel.SPLINE36: (3, _spline36),
    Kernel.LANCZOS: (3, _lanczos),
}


def _round_halfup(x: np.ndarray) -> np.ndarray:
    # zimg's bias keeps round(x - 1) == round(x) - 1.
    return np.where(x < 0, np.floor(x + 0.5), np.floor(x + 0.49999999999999994))


@lru_cache(maxsize=64)
def _taps(
    kernel: Kernel, src_dim: int, dst_dim: int, shift: float, width: float
) -> tuple[np.ndarray, np.ndarray]:
    """Input indices (int32) and float32 weights of zimg's compute_filter, each
    taps x dst_dim (tap-major, as the C passes read them). Cached, so read-only."""
    support, f = _KERNELS[kernel]
    scale = dst_dim / width
    step = min(scale, 1.0)
    size = max(math.ceil(support / step) * 2, 1)
    pos = (np.arange(dst_dim) + 0.5) / scale + shift
    xpos = (_round_halfup(pos - size / 2) + 0.5)[:, None] + np.arange(size)
    weights = f((xpos - pos[:, None]) * step)
    weights /= weights.sum(axis=1, keepdims=True)
    mirrored = np.where(
        xpos < 0, -xpos, np.where(xpos >= src_dim, 2.0 * src_dim - xpos, xpos)
    )
    np.clip(mirrored, 0.0, np.nextafter(src_dim, -np.inf), out=mirrored)
    index = np.ascontiguousarray(np.floor(mirrored).T, np.int32)
    weights = np.ascontiguousarray(weights.T, np.float32)
    index.flags.writeable = weights.flags.writeable = False
    return index, weights


def resample_plane(
    plane: np.ndarray,
    width: int,
    height: int,
    kernel: Kernel = Kernel.SPLINE36,
    src_left: float = 0.0,
    src_top: float = 0.0,
    src_width: float | None = None,
    src_height: float | None = None,
) -> np.ndarray:
    """A float32 HxW plane resampled to `width` x `height`: the source window
    `src_left`..`src_left + src_width` (and `src_top`..`src_top + src_height`), in input
    pixels, mapped onto the whole output (zimg's src_left/src_top/src_width/src_height).
    The window defaults to the whole plane."""
    return resample_plane_biased(
        plane, width, height, kernel, src_left, src_top, src_width, src_height, 0, 0
    )


def resample_plane_biased(
    plane: np.ndarray,
    width: int,
    height: int,
    kernel: Kernel,
    src_left: float,
    src_top: float,
    src_width: float | None,
    src_height: float | None,
    bias_in: float,
    bias_out: float,
) -> np.ndarray:
    """resample_plane of `plane - bias_in`, plus `bias_out` (each only when it is not
    0), both inside the C passes: the values of a separate float32 subtraction and
    addition, without their passes over the plane."""
    if plane.ndim != 2:
        raise ValueError(f"A plane has 2 dimensions, not {plane.ndim}.")
    if width < 1 or height < 1:
        raise ValueError(f"Cannot resample a plane to {width}x{height}.")
    h, w = plane.shape
    window_w = float(w if src_width is None else src_width)
    window_h = float(h if src_height is None else src_height)
    if window_w <= 0 or window_h <= 0:
        raise ValueError(f"The source window {window_w:g}x{window_h:g} is empty.")
    plane = plane.astype(np.float32, copy=False)
    passes = [
        (axis, size, shift, window)
        for axis, size, shift, window in (
            (1, width, src_left, window_w),
            (0, height, src_top, window_h),
        )
        if not (plane.shape[axis] == size and shift == 0 and window == size)
    ]
    if not passes:
        out = plane - np.float32(bias_in) if bias_in else plane.copy()
        return out + np.float32(bias_out) if bias_out else out
    out = plane
    for number, (axis, size, shift, window) in enumerate(passes):
        index, weights = _taps(kernel, out.shape[axis], size, shift, window)
        out = native_planes.resample_axis(
            out,
            axis == 0,
            index,
            weights,
            bias_in if number == 0 else 0,
            bias_out if number == len(passes) - 1 else 0,
        )
    return out
