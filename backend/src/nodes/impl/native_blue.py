"""C seeded void-and-cluster ranking, with the installed numerical contract."""

from __future__ import annotations

import ctypes as ct
import operator
from functools import lru_cache

import numpy as np

from .native import check, lib
from .native_noise import seeded_permutation


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    void, size, real = ct.c_void_p, ct.c_size_t, ct.c_double
    dll.cn_blue_from_permutation.argtypes = [void, void, size, size, real, real, real]
    dll.cn_blue_from_permutation.restype = ct.c_int
    dll.cn_blue_find_index.argtypes = [
        void,
        size,
        size,
        real,
        real,
        ct.c_int,
        ct.POINTER(size),
    ]
    dll.cn_blue_find_index.restype = ct.c_int
    dll.cn_blue_normalize.argtypes = [void, void, size, ct.c_int]
    dll.cn_blue_normalize.restype = ct.c_int
    return dll


def sigmas(sigma: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(sigma, (int, float, np.integer, np.floating)):
        return float(sigma), float(sigma)
    values = tuple(sigma)
    if len(values) != 2:
        raise RuntimeError("sequence argument must have length equal to input rank")
    return float(values[0]), float(values[1])


def create(
    shape: tuple[int, int],
    sigma: float | tuple[float, float],
    fraction: float,
    seed: int,
) -> np.ndarray:
    height, width = (operator.index(v) for v in shape)
    if height <= 0 or width <= 0:
        raise ValueError("Blue noise dimensions must be positive")
    # Preserve int conversion errors before the checked C boundary.
    int(height * width * fraction)
    permutation = seeded_permutation(seed, height * width)
    out = np.empty((height, width), dtype=np.int32)
    check(
        _api().cn_blue_from_permutation(
            permutation.ctypes.data,
            out.ctypes.data,
            height,
            width,
            *sigmas(sigma),
            fraction,
        )
    )
    return out


def find(
    pattern: np.ndarray, sigma: float | tuple[float, float], tightest: bool
) -> int:
    pattern = np.require(pattern, dtype=np.bool_, requirements=["C", "A"])
    if pattern.ndim != 2:
        raise ValueError("Expected a two-dimensional binary pattern")
    result = ct.c_size_t()
    check(
        _api().cn_blue_find_index(
            pattern.ctypes.data,
            *pattern.shape,
            *sigmas(sigma),
            int(tightest),
            ct.byref(result),
        )
    )
    return result.value


def normalize(ranks: np.ndarray) -> np.ndarray:
    ranks = np.require(ranks, dtype=np.int32, requirements=["C", "A"])
    dtype = np.result_type(np.empty(0, dtype=np.float32), ranks.size - 1)
    out = np.empty(ranks.shape, dtype=dtype)
    check(
        _api().cn_blue_normalize(
            ranks.ctypes.data, out.ctypes.data, ranks.size, int(dtype == np.float64)
        )
    )
    return out
