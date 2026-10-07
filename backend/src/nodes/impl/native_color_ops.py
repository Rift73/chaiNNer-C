"""Checked NumPy ownership bridge for C color, threshold and material kernels.

The helpers own pointwise arithmetic, threshold antialiasing, means and
histograms in C. Adaptive Threshold also uses C Box/Gaussian neighborhood filters.
"""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import numpy as np

from .native import check, f32, lib, ptr
from .native_buffers import converted_pixels


@lru_cache(maxsize=1)
def _api():
    library = lib()
    fp, up = ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint8)
    size, integer, real = ct.c_size_t, ct.c_int, ct.c_float
    signatures = {
        "cn_color_linear": [fp, fp, size, real, real],
        "cn_hls_adjust": [fp, size, real, real],
        "cn_threshold_f32": [fp, fp, fp, size, integer, real, real],
        "cn_quantize_u8": [fp, up, size],
        "cn_auto_threshold": [fp, size, size, integer, ct.POINTER(integer)],
        "cn_adaptive_apply": [up, up, up, size, integer, integer, integer],
        "cn_material_f32": [fp, fp, fp, fp, size, integer, real, real],
        "cn_height_map": [fp, fp, size, size, integer],
        "cn_height_prepare": [fp, fp, size, real, real],
        "cn_normal_output": [fp, fp, fp, fp, size, integer, integer, integer],
    }
    for name, signature in signatures.items():
        function = getattr(library, name)
        function.argtypes = signature
        function.restype = integer
    return library


def _image(image: np.ndarray) -> tuple[np.ndarray, int]:
    if image.ndim not in (2, 3) or any(size == 0 for size in image.shape):
        raise ValueError(f"Invalid image shape {image.shape}")
    return f32(image), image.shape[2] if image.ndim == 3 else 1


def _same_size(a: np.ndarray, b: np.ndarray):
    if a.shape[:2] != b.shape[:2]:
        raise ValueError("C color kernel inputs must have matching spatial dimensions")


def _u8ptr(image: np.ndarray):
    return image.ctypes.data_as(ct.POINTER(ct.c_uint8))


def linear(image: np.ndarray, factor: float, offset: float = 0) -> np.ndarray:
    source, _ = _image(image)
    result = np.empty_like(source)
    check(_api().cn_color_linear(ptr(source), ptr(result), source.size, factor, offset))
    return result


def hls_adjust(image: np.ndarray, hue: float, saturation: float) -> np.ndarray:
    source, channels = _image(image)
    if channels != 3:
        raise ValueError("HLS adjustment requires three channels")
    result = source.copy()
    check(_api().cn_hls_adjust(ptr(result), result.size // 3, hue, 1 + saturation))
    return result


def threshold(
    image: np.ndarray,
    value: float,
    maximum: float,
    kind: int,
    binary: np.ndarray | None = None,
) -> np.ndarray:
    source, channels = _image(image)
    if kind not in range(5):
        raise ValueError("Unknown threshold type")
    api = _api()
    if binary is None:
        from .native_threshold import hard_threshold

        return hard_threshold(source, value, maximum, kind)
    binary, binary_channels = _image(binary)
    _same_size(source, binary)
    if binary_channels != channels:
        raise ValueError("Threshold mask channel count must match its image")
    shape = source.shape[:2] if channels == 1 else source.shape
    result = np.empty(shape, dtype=np.float32)
    check(
        api.cn_threshold_f32(
            ptr(source),
            ptr(binary),
            ptr(result),
            source.size,
            kind,
            value,
            maximum,
        )
    )
    return result


def quantize_u8(image: np.ndarray) -> np.ndarray:
    source, _ = _image(image)
    result = np.empty(source.shape, dtype=np.uint8)
    check(_api().cn_quantize_u8(ptr(source), _u8ptr(result), source.size))
    return result


def auto_threshold(image: np.ndarray, method: int) -> float:
    source, channels = _image(image)
    if channels > 7:
        from .native_threshold import channel_mean

        source = channel_mean(image)  # upstream's np.mean reads the input's own layout
        channels = 1
    value = ct.c_int()
    check(
        _api().cn_auto_threshold(
            ptr(source), source.size // channels, channels, method, ct.byref(value)
        )
    )
    return value.value / 255 * 100


def adaptive_threshold(
    image: np.ndarray, kind: int, maximum: float, method: int, radius: int, delta: int
) -> np.ndarray:
    source, channels = _image(image)
    if channels != 1 or radius < 1 or kind not in (0, 1) or method not in (0, 1):
        raise ValueError("Invalid adaptive threshold shape or parameters")
    # Reuse the checked C conversion, including the original NumPy floating
    # warnings/errors for nonfinite and overflowing registered float32 inputs.
    source_u8 = converted_pixels(source, 8, normalized=True)
    assert source_u8 is not None
    source_u8 = source_u8.reshape(source.shape[:2])
    if method == 0:
        from .native_box import mean_u8

        mean = mean_u8(source_u8, radius)
    else:
        from .native_gaussian import mean_u8

        mean = mean_u8(source_u8, radius)
    result = np.empty_like(source_u8)
    maximum_u8 = max(0, min(255, round(maximum)))
    check(
        _api().cn_adaptive_apply(
            _u8ptr(source_u8),
            _u8ptr(mean),
            _u8ptr(result),
            result.size,
            delta,
            kind,
            maximum_u8,
        )
    )
    return result


def material(
    a: np.ndarray,
    op: int,
    b: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    minimum: float = 0,
    span: float = 1,
) -> np.ndarray:
    a, channels = _image(a)
    if channels != (1 if op == 0 else 3):
        raise ValueError("Invalid material input channel count")
    if b is not None:
        b, b_channels = _image(b)
        _same_size(a, b)
        if b_channels != 3:
            raise ValueError("Material color input must have three channels")
    if mask is not None:
        mask, mask_channels = _image(mask)
        _same_size(a, mask)
        if mask_channels != 1:
            raise ValueError("Metallic mask must have one channel")
    shape = a.shape[:2] if op == 3 else (*a.shape[:2], 3)
    result = np.empty(shape, dtype=np.float32)
    check(
        _api().cn_material_f32(
            ptr(a),
            ptr(b) if b is not None else None,
            ptr(mask) if mask is not None else None,
            ptr(result),
            a.shape[0] * a.shape[1],
            op,
            minimum,
            span,
        )
    )
    return result


def height_map(image: np.ndarray, mode: int) -> np.ndarray:
    source, channels = _image(image)
    result = np.empty(source.shape[:2], dtype=np.float32)
    check(_api().cn_height_map(ptr(source), ptr(result), result.size, channels, mode))
    return result


def prepare_height(image: np.ndarray, minimum: float, scale: float) -> np.ndarray:
    source, channels = _image(image)
    if channels != 1:
        raise ValueError("Height map must have one channel")
    result = np.empty_like(source)
    check(
        _api().cn_height_prepare(ptr(source), ptr(result), source.size, minimum, scale)
    )
    return result


def normal_output(
    dx: np.ndarray,
    dy: np.ndarray,
    alpha: np.ndarray | None,
    invert_r: bool,
    invert_g: bool,
    with_alpha: bool,
) -> np.ndarray:
    dx, dx_channels = _image(dx)
    dy, dy_channels = _image(dy)
    _same_size(dx, dy)
    if dx_channels != 1 or dy_channels != 1:
        raise ValueError("Normal derivatives must have one channel")
    if alpha is not None:
        alpha, alpha_channels = _image(alpha)
        _same_size(dx, alpha)
        if alpha_channels != 1:
            raise ValueError("Normal alpha must have one channel")
    channels = 4 if with_alpha else 3
    result = np.empty((*dx.shape[:2], channels), dtype=np.float32)
    check(
        _api().cn_normal_output(
            ptr(dx),
            ptr(dy),
            ptr(alpha) if alpha is not None else None,
            ptr(result),
            dx.size,
            invert_r,
            invert_g,
            channels,
        )
    )
    return result
