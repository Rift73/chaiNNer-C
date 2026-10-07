"""Typed bridges for the C gradient and normal-map kernels.

Workflow images are float32. The public helpers retain their original NumPy
implementation for other dtypes, broadcasting, and overlapping component
views, whose uncommon mutation semantics are outside this contiguous C ABI.
DLL loading errors are never treated as a reason to select that compatibility
path. Writable contiguous aligned buffers are passed directly to C. Strided
images are copied and in-place component results are copied back.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache

import numpy as np

from .native import check, f32, lib, ptr
from .native_numpy_simd import fma3, loop_target

XYZ = tuple[np.ndarray, np.ndarray, np.ndarray]


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    fp = ctypes.POINTER(ctypes.c_float)
    size = ctypes.c_size_t
    integer = ctypes.c_int
    real = ctypes.c_float
    signatures = {
        "cn_gradient": [
            fp,
            size,
            size,
            integer,
            ctypes.c_double,
            ctypes.c_double,
            integer,
        ],
        "cn_normalize_normals": [fp, fp, fp, size],
        "cn_normal_decode": [fp, size, size, integer, fp, fp, fp],
        "cn_normal_encode": [fp, fp, fp, size, integer, fp],
        "cn_normal_map": [fp, fp, size, size, integer, integer, integer],
        "cn_normal_add": [
            fp,
            fp,
            size,
            size,
            size,
            integer,
            real,
            real,
            integer,
            integer,
            fp,
            fp,
            fp,
        ],
    }
    for name, signature in signatures.items():
        function = getattr(dll, name)
        function.argtypes = signature
        function.restype = ctypes.c_int
    return dll


def image_supported(image: np.ndarray) -> bool:
    return image.dtype == np.float32 and image.ndim == 3 and image.shape[2] >= 3


def components_supported(*components: np.ndarray) -> bool:
    return (
        all(c.dtype == np.float32 and c.ndim == 2 for c in components)
        and all(c.shape == components[0].shape for c in components[1:])
        and not any(
            np.may_share_memory(c, other)
            for index, c in enumerate(components)
            for other in components[index + 1 :]
        )
    )


def factors_supported(*factors: float) -> bool:
    # NumPy 1.x promotes float32 arrays when a scalar does not fit float32.
    sample = np.empty(0, dtype=np.float32)
    return all(np.result_type(sample, factor) == np.float32 for factor in factors)


def fill_gradient(img: np.ndarray, kind: int, a: float = 0, b: float = 1) -> bool:
    if img.dtype != np.float32 or img.ndim != 2:
        return False
    if kind >= 2 and 0 in img.shape:
        return False
    wide_start = False
    if kind == 2:
        factor_dtype = np.result_type(np.empty(0, dtype=np.float32), b)
        object_integer = factor_dtype == np.dtype(object) and isinstance(b, int)
        if (
            factor_dtype not in (np.dtype(np.float32), np.dtype(np.float64))
            and not object_integer
        ):
            return False
        # For arbitrary Python integers, NumPy's object loop multiplies Python
        # float objects. They perform the same double operations; float(b) below
        # also preserves the original OverflowError beyond double's range.
        wide_start = factor_dtype == np.float64 or object_integer
    if kind == 2 and isinstance(a, np.floating) and a.dtype != np.float64:
        # A float32 angle selects NumPy's distinct float32 trigonometric kernel.
        return False
    direct = img.flags.c_contiguous and img.flags.aligned and img.flags.writeable
    result = img if direct else np.empty(img.shape, dtype=np.float32)
    check(
        _api().cn_gradient(
            ptr(result), *img.shape, kind, float(a), float(b), wide_start
        )
    )
    if not direct:
        img[...] = result
    return True


def _inplace_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.flags.writeable and y.flags.writeable:
        # The public entry points exclude overlapping x/y/z component views.
        return f32(x), f32(y)
    # If either component is read-only, delay writes until the original x-then-y
    # copyback order. In particular, x rejection must leave writable y unchanged.
    return (
        np.array(x, dtype=np.float32, order="C", copy=True),
        np.array(y, dtype=np.float32, order="C", copy=True),
    )


def normalize(x: np.ndarray, y: np.ndarray) -> XYZ:
    nx, ny = _inplace_xy(x, y)
    z = np.empty(x.shape, dtype=np.float32)
    check(_api().cn_normalize_normals(ptr(nx), ptr(ny), ptr(z), x.size))
    if nx is not x:
        x[...] = nx
    if ny is not y:
        y[...] = ny
    return x, y, z


def decode(image: np.ndarray, *, octahedral: bool = False) -> XYZ:
    image = f32(image)
    shape = image.shape[:2]
    x, y, z = (np.empty(shape, dtype=np.float32) for _ in range(3))
    xyz = (x, y, z)
    check(
        _api().cn_normal_decode(
            ptr(image),
            xyz[0].size,
            image.shape[2],
            int(octahedral),
            *(ptr(c) for c in xyz),
        )
    )
    return xyz


def encode(xyz: XYZ, *, octahedral: bool = False) -> np.ndarray:
    x, y, z = xyz
    # Octahedral encoding intentionally normalizes the caller's x/y in-place.
    nx, ny = _inplace_xy(x, y) if octahedral else (f32(x), f32(y))
    nz = f32(z)
    result = np.empty((*x.shape, 3), dtype=np.float32)
    check(
        _api().cn_normal_encode(
            ptr(nx), ptr(ny), ptr(nz), x.size, int(octahedral), ptr(result)
        )
    )
    if octahedral:
        if nx is not x:
            x[...] = nx
        if ny is not y:
            y[...] = ny
    return result


def add(
    n1: np.ndarray, n2: np.ndarray | None, method: int, f1: float, f2: float = 0
) -> XYZ:
    n1 = f32(n1)
    n2 = f32(n2) if n2 is not None else None
    x, y, z = (np.empty(n1.shape[:2], dtype=np.float32) for _ in range(3))
    xyz = (x, y, z)
    check(
        _api().cn_normal_add(
            ptr(n1),
            ptr(n2) if n2 is not None else None,
            xyz[0].size,
            n1.shape[2],
            n2.shape[2] if n2 is not None else 0,
            method,
            f1,
            f2,
            int(n2 is None),
            # cn_normal_add's sine mirrors np.sin's float32 loop: its FMA3 form
            # where NumPy dispatches one, else the CRT sinf.
            int(fma3(loop_target("sin", "ff"))),
            *(ptr(c) for c in xyz),
        )
    )
    return xyz


def normal_map(
    image: np.ndarray, mode: int, option_a: int = 0, option_b: int = 0
) -> np.ndarray:
    if mode == 1 and abs(image.strides[0]) < abs(image.strides[1]):
        # Balance's np.mean sums x and y in their memory order, which follows the
        # image's (a ufunc output's K order): column-major here. Every other step
        # is per pixel, and upstream's np.dstack returns a C-ordered image.
        transposed = normal_map(image.transpose(1, 0, 2), mode)
        return np.ascontiguousarray(transposed.transpose(1, 0, 2))
    image = f32(image)
    out = np.empty((*image.shape[:2], 3), dtype=np.float32)
    check(
        _api().cn_normal_map(
            ptr(image),
            ptr(out),
            out.size // 3,
            image.shape[2],
            mode,
            option_a,
            option_b,
        )
    )
    return out
