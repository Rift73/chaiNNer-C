"""Array ownership and ABI adaptation for the C blend implementation."""

import ctypes

import numpy as np

from .native import check, f32, lib, ptr

_library = lib()
_float_p = ctypes.POINTER(ctypes.c_float)
_library.cn_blend_mode.argtypes = [
    _float_p,
    _float_p,
    _float_p,
    ctypes.c_size_t,
    ctypes.c_int,
]
_library.cn_blend_mode.restype = ctypes.c_int
_library.cn_blend_images.argtypes = [
    _float_p,
    _float_p,
    _float_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
_library.cn_blend_images.restype = ctypes.c_int


def raw_blend(a: np.ndarray, b: np.ndarray, mode: int) -> np.ndarray:
    """Apply a mode to equally shaped float32 arrays, without output clipping."""
    if mode == 0:
        f32(a)
        return a
    if a.shape != b.shape:
        raise ValueError("Blend arrays must have the same shape.")
    a_buffer, b_buffer = f32(a), f32(b)
    result = np.empty(a.shape, dtype=np.float32)
    check(
        _library.cn_blend_mode(ptr(a_buffer), ptr(b_buffer), ptr(result), a.size, mode)
    )
    return result


def composite(
    overlay: np.ndarray,
    base: np.ndarray,
    mode: int,
    overlay_channels: int,
    base_channels: int,
) -> np.ndarray:
    """Validate the complete buffer shape before passing pixel counts to C."""
    for image, channels in ((overlay, overlay_channels), (base, base_channels)):
        if image.ndim not in (2, 3) or any(size == 0 for size in image.shape):
            raise ValueError(f"Invalid blend image shape {image.shape}")
        actual_channels = image.shape[2] if image.ndim == 3 else 1
        if channels not in (1, 3, 4) or channels != actual_channels:
            raise ValueError("Blend channel count does not match the image buffer")
    if overlay.shape[:2] != base.shape[:2]:
        raise ValueError("Blend images must have the same size")
    o_buffer, b_buffer = f32(overlay), f32(base)
    height, width = overlay.shape[:2]
    target = max(overlay_channels, base_channels)
    if mode == 0 and target < 4 and overlay_channels == target:
        return overlay[:, :, 0] if target == 1 and overlay.ndim == 3 else overlay
    shape = (height, width) if target == 1 else (height, width, target)
    result = np.empty(shape, dtype=np.float32)
    check(
        _library.cn_blend_images(
            ptr(o_buffer),
            ptr(b_buffer),
            ptr(result),
            height * width,
            overlay_channels,
            base_channels,
            mode,
        )
    )
    return result
