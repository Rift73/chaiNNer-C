"""Linear Light: an image's colour channels between encoded values and linear light
(native/src/linear_light_ops.c, the curves of native/include/linear_light.h)."""

from __future__ import annotations

import ctypes as ct
from enum import Enum
from functools import lru_cache

import numpy as np

from .native import check, f32, lib, ptr


class LightTransfer(Enum):
    """The transfer curve between encoded values and linear light; the values are
    linear_light.h's CN_LIGHT_* and the Transfer dropdown's."""

    GAMMA_24 = 0
    SRGB = 1
    GAMMA_22 = 2
    BT709_CAMERA = 3


@lru_cache(maxsize=1)
def _api():
    library = lib()
    size, integer = ct.c_size_t, ct.c_int
    library.cn_linear_light.argtypes = [
        ct.POINTER(ct.c_float),
        size,
        size,
        size,
        integer,
        integer,
        integer,
    ]
    library.cn_linear_light.restype = integer
    return library


def _convert(image: np.ndarray, transfer: LightTransfer, encode: bool) -> np.ndarray:
    """image (owned, C-contiguous) converted in place: every channel, or the first
    three of four (alpha as is)."""
    h, w = image.shape[:2]
    c = image.shape[2] if image.ndim == 3 else 1
    check(_api().cn_linear_light(ptr(image), h, w, c, transfer.value, encode, c == 4))
    return image


def linearize(image: np.ndarray, transfer: LightTransfer) -> np.ndarray:
    """A linear-light copy of image: its colour channels through transfer's inverse,
    alpha (a fourth channel) as is. A one-channel image is converted as a plane."""
    return _convert(np.array(f32(image), order="C"), transfer, False)


def encode_inplace(image: np.ndarray, transfer: LightTransfer) -> np.ndarray:
    """Encodes an owned linear-light result with transfer (alpha as is); the caller
    must not pass graph input."""
    if not image.flags.writeable:
        raise ValueError("Expected a writable linear-light result")
    return _convert(f32(image), transfer, True)
