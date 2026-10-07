"""Checked bridges for fused image construction and procedural noise kernels."""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import numpy as np

from .native import check, f32, lib, ptr


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    fp, dp, ip = (ct.POINTER(t) for t in (ct.c_float, ct.c_double, ct.c_int32))
    size, integer, real, void = ct.c_size_t, ct.c_int, ct.c_double, ct.c_void_p
    signatures = {
        "cn_image_fill": [fp, size, size, size, fp, fp, fp, integer, size],
        "cn_noise_combine": [fp, size, size, void, void, size, integer, integer, fp],
        "cn_procedural_noise": [
            dp,
            size,
            integer,
            ip,
            size,
            fp,
            size,
            dp,
            size,
            integer,
            integer,
            real,
            real,
            real,
            real,
            dp,
        ],
    }
    for name, signature in signatures.items():
        function = getattr(dll, name)
        function.argtypes, function.restype = signature, ct.c_int
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


def combine_noise(
    image: np.ndarray, noises: list[np.ndarray], operation: int
) -> np.ndarray:
    image = f32(image)
    if image.ndim not in (2, 3) or 0 in image.shape:
        raise ValueError("Expected a nonempty image")
    channels = image.shape[2] if image.ndim == 3 else 1
    if (
        channels == 2
        or operation not in (0, 1, 2)
        or len(noises) != (2 if operation == 2 else 1)
    ):
        raise ValueError("Invalid noise operation or channels")
    noise = noises[0]
    nc = noise.shape[2] if noise.ndim == 3 else 1
    if (
        noise.ndim not in (2, 3)
        or nc not in (1, 3)
        or noise.shape[:2] != image.shape[:2]
        or noise.dtype not in (np.dtype(np.float32), np.dtype(np.uint8))
        or any(n.shape != noise.shape or n.dtype != noise.dtype for n in noises)
    ):
        raise ValueError("Noise shape or dtype does not match image")
    buffers = [np.require(n, requirements=["C", "A"]) for n in noises]
    out_c = max(channels, nc)
    shape = (*image.shape[:2], out_c) if out_c > 1 else image.shape[:2]
    out = np.empty(shape, dtype=np.float32)
    check(
        _api().cn_noise_combine(
            ptr(image),
            image.shape[0] * image.shape[1],
            channels,
            buffers[0].ctypes.data,
            buffers[1].ctypes.data if operation == 2 else None,
            nc,
            int(noise.dtype == np.uint8),
            operation,
            ptr(out),
        )
    )
    return out


def procedural_supported(points: np.ndarray, dimensions: int) -> bool:
    # Preserve the uncommon float32/integer helper arithmetic and extreme-coordinate
    # cast semantics in the original implementation. Workflow coordinates are f64.
    return (
        points.dtype == np.float64
        and points.ndim == 2
        and points.shape[1] == dimensions
        and 1 <= dimensions <= 6
        and bool(np.all(np.isfinite(points)))
        and bool(np.all(np.abs(points) <= 1e8))
    )


def procedural_noise(
    points: np.ndarray,
    table: np.ndarray,
    *,
    values: np.ndarray | None = None,
    gradients: np.ndarray | None = None,
    smooth: bool = False,
    f: float = 0,
    g: float = 0,
    r2: float = 0,
    scale: float = 1,
) -> np.ndarray:
    if points.ndim != 2 or not procedural_supported(points, points.shape[1]):
        raise ValueError("Unsupported procedural noise coordinates")
    if (
        table.ndim != 1
        or table.size == 0
        or table.dtype.kind not in "iu"
        or np.any(table < 0)
        or np.any(table >= table.size)
    ):
        raise ValueError("Invalid permutation table")
    points = np.require(points, requirements=["C", "A"])
    table = np.require(table, dtype=np.int32, requirements=["C", "A"])
    simplex = gradients is not None
    if simplex:
        assert gradients is not None
        if (
            gradients.ndim != 2
            or gradients.shape[1] != points.shape[1]
            or gradients.shape[0] == 0
            or gradients.dtype != np.float64
        ):
            raise ValueError("Invalid gradient table")
        gradients = np.require(gradients, requirements=["C", "A"])
    else:
        if values is None or values.ndim != 1 or not values.size:
            raise ValueError("Invalid value table")
        values = f32(values)
    dp = ct.POINTER(ct.c_double)
    out = np.empty(points.shape[0], dtype=np.float64)
    check(
        _api().cn_procedural_noise(
            points.ctypes.data_as(dp),
            len(points),
            points.shape[1],
            table.ctypes.data_as(ct.POINTER(ct.c_int32)),
            table.size,
            ptr(values) if values is not None else None,
            values.size if values is not None else 0,
            gradients.ctypes.data_as(dp) if gradients is not None else None,
            len(gradients) if gradients is not None else 0,
            int(smooth),
            int(simplex),
            f,
            g,
            r2,
            scale,
            out.ctypes.data_as(dp),
        )
    )
    return out
