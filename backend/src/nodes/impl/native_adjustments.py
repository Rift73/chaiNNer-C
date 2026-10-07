"""Array ownership and scalar argument preparation for C adjustment kernels.

adjust, levels and log_linear return np.clip's order-K layout, as upstream's
NumPy expressions do (Consult 11 D-16): the result is np.empty_like(image,
order="K"), and the kernels run flat over the image and the result in that memory
order, the channel-aware ones with the channel stride of that order.
"""

from __future__ import annotations

import ctypes as ct

import numpy as np

from .native import check, checked_f32, f32, in_order, lib, ptr

_lib = lib()
_float_p = ct.POINTER(ct.c_float)
_lib.cn_adjust_f32.argtypes = [
    _float_p,
    _float_p,
    ct.c_size_t,
    ct.c_size_t,
    ct.c_size_t,
    ct.c_int,
    ct.c_float,
    ct.c_float,
    ct.c_int,
]
_lib.cn_opacity_f32.argtypes = [
    _float_p,
    _float_p,
    ct.c_size_t,
    ct.c_size_t,
    ct.c_float,
]
_lib.cn_levels_f32.argtypes = [
    _float_p,
    _float_p,
    ct.c_size_t,
    ct.c_size_t,
    ct.c_size_t,
    ct.c_uint,
    ct.c_float,
    ct.c_float,
    ct.c_float,
    ct.c_float,
    ct.c_float,
]
_lib.cn_log_linear_f32.argtypes = [
    _float_p,
    _float_p,
    ct.c_size_t,
    ct.c_float,
    ct.c_float,
    ct.c_float,
    ct.c_float,
    ct.c_float,
    ct.c_int,
]
for _name in ("cn_adjust_f32", "cn_opacity_f32", "cn_levels_f32", "cn_log_linear_f32"):
    getattr(_lib, _name).restype = ct.c_int


def _channels(img: np.ndarray) -> int:
    if img.ndim not in (2, 3) or any(size == 0 for size in img.shape):
        raise ValueError(f"Invalid image shape {img.shape}")
    return img.shape[2] if img.ndim == 3 else 1


def _image(img: np.ndarray) -> tuple[np.ndarray, int]:
    channels = _channels(img)
    return f32(img), channels


def _in_order(
    img: np.ndarray, c_order: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """The kernel's source and target, the result, the channel count and the
    channel stride in elements: np.clip's order K, or with c_order a C-ordered
    result, as upstream's ndarray.copy() gives."""
    if c_order:
        source, channels = _image(img)
        result = np.empty_like(source)
        return source, result, result, channels, 1
    channels = _channels(img)
    source, target, result = in_order(checked_f32(img))
    stride = result.strides[2] // result.itemsize if channels > 1 else 1
    return source, target, result, channels, stride


def adjust(
    img: np.ndarray,
    op: int,
    a: float = 0,
    b: float = 0,
    *,
    clip: bool = False,
    c_order: bool = False,
) -> np.ndarray:
    source, target, result, channels, stride = _in_order(img, c_order)
    check(
        _lib.cn_adjust_f32(
            ptr(source),
            ptr(target),
            source.size // channels,
            channels,
            stride,
            op,
            a,
            b,
            clip,
        )
    )
    return result


def opacity(img: np.ndarray, value: float) -> np.ndarray:
    source, channels = _image(img)
    result = np.empty((*source.shape[:2], 4), dtype=np.float32)
    check(
        _lib.cn_opacity_f32(
            ptr(source), ptr(result), source.size // channels, channels, value
        )
    )
    return result


def levels(
    img: np.ndarray,
    mask: int,
    black: float,
    white: float,
    gamma: float,
    out_black: float,
    out_white: float,
) -> np.ndarray:
    # Upstream's as_3d copies a two-dimensional image (C order) and adds an axis.
    source, target, result, channels, stride = _in_order(img, img.ndim == 2)
    check(
        _lib.cn_levels_f32(
            ptr(source),
            ptr(target),
            source.size // channels,
            channels,
            stride,
            mask,
            black,
            white,
            max(0.001, gamma),
            out_black,
            out_white,
        )
    )
    # Color Levels returns a 3D single-channel array; ImageOutput flattens it.
    return result.reshape(*result.shape, 1) if result.ndim == 2 else result


def log_linear(
    img: np.ndarray, black: float, white: float, gamma: float, invert: bool
) -> np.ndarray:
    # Preserve Python's scalar exceptions for gamma=0 or equal black/white.
    offset = pow(10.0, (black - white) * 0.002 / gamma)
    gain = 1.0 / (1.0 - offset)
    source, target, result, _, _ = _in_order(img)
    check(
        _lib.cn_log_linear_f32(
            ptr(source),
            ptr(target),
            source.size,
            white,
            gamma,
            offset,
            gain,
            0.002 / gamma,
            invert,
        )
    )
    return result
