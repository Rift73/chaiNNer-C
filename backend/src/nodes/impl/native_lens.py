"""Exact C Lens Blur kernel generation, power, and component composition."""

from __future__ import annotations

import ctypes as ct
import operator

import numpy as np

from .native import check, f32, lib, ptr
from .native_numpy_simd import fma3, loop_target

_lib = lib()
_fp = ct.POINTER(ct.c_float)
_n = ct.c_size_t
_lib.cn_lens_kernel.argtypes = [_fp, _n, ct.c_float, ct.c_float, ct.c_float, ct.c_int]
_lib.cn_lens_power.argtypes = [_fp, _fp, _n, ct.c_float, ct.c_int]
_lib.cn_lens_compose.argtypes = [_fp] * 5 + [_n, ct.c_float, ct.c_float, ct.c_int]
for _name in ("kernel", "power", "compose"):
    getattr(_lib, "cn_lens_" + _name).restype = ct.c_int


def kernel_dispatch() -> bool:
    """Whether NumPy runs float32 exp, sin and cos on their FMA3 SIMD paths, which
    cn_lens_kernel's use_fma mirrors with one flag for the three loops."""
    paths = {fma3(loop_target(name, "ff")) for name in ("exp", "sin", "cos")}
    if len(paths) != 1:
        raise RuntimeError("NumPy runs float32 exp, sin and cos on different paths")
    return paths.pop()


def complex_kernel(radius: int, scale: float, a: float, b: float) -> np.ndarray:
    radius = operator.index(radius)
    if not 1 <= radius <= 1000:
        raise ValueError("Lens kernel radius must be between 1 and 1000")
    result = np.empty((1, radius * 2 + 1), np.complex64)
    use_fma = kernel_dispatch()
    check(
        _lib.cn_lens_kernel(
            ptr(result.view(np.float32)), radius, scale, a, b, int(use_fma)
        )
    )
    return result


def power(image: np.ndarray, exponent: float, *, finish: bool = False) -> np.ndarray:
    source = f32(image)
    result = np.empty_like(source)
    check(_lib.cn_lens_power(ptr(source), ptr(result), source.size, exponent, finish))
    return result


def compose(
    f1: np.ndarray,
    f2: np.ndarray,
    f3: np.ndarray,
    f4: np.ndarray,
    a: float,
    b: float,
    *,
    out: np.ndarray,
    accumulate: bool,
) -> None:
    inputs = [f32(array) for array in (f1, f2, f3, f4)]
    if any(array.shape != out.shape for array in inputs):
        raise ValueError("Lens component shapes must match")
    if (
        out.dtype != np.float32
        or not out.flags.c_contiguous
        or not out.flags.aligned
        or not out.flags.writeable
        or any(np.shares_memory(out, array) for array in (f1, f2, f3, f4))
    ):
        raise ValueError(
            "Lens output must be writable aligned contiguous nonoverlapping float32"
        )
    check(
        _lib.cn_lens_compose(
            *(ptr(array) for array in inputs),
            ptr(out),
            out.size,
            a,
            b,
            accumulate,
        )
    )
