"""Checked median and dithering kernels with explicit array contracts."""

from __future__ import annotations

import ctypes as ct

import cv2
import numpy as np

from .native import check, f32, lib, ptr

_lib = lib()
_p = ct.POINTER(ct.c_float)
_n = ct.c_size_t
_lib.cn_neighborhood_median_f32.argtypes = [_p, _p, _n, _n, _n, _n, _n]
_lib.cn_neighborhood_median_u8.argtypes = [ct.c_void_p, ct.c_void_p, _n, _n, _n, _n]
_lib.cn_neighborhood_dither.argtypes = [
    _p,
    _p,
    _n,
    _n,
    _n,
    ct.c_uint32,
    ct.c_int,
    _n,
    ct.c_int,
    _p,
    _n,
    ct.POINTER(ct.c_int),
]
_lib.cn_neighborhood_uniform_riemersma.argtypes = [
    _p,
    _p,
    _n,
    _n,
    _n,
    ct.c_uint32,
    ct.c_uint32,
    ct.c_float,
]
_lib.cn_neighborhood_uniform_riemersma.restype = ct.c_int
_lib.cn_neighborhood_palette_riemersma.argtypes = [
    _p,
    _p,
    _n,
    _n,
    _n,
    _p,
    _n,
    ct.c_uint32,
    ct.c_float,
    ct.POINTER(ct.c_int),
]
_lib.cn_neighborhood_palette_riemersma.restype = ct.c_int
for _name in ("median_f32", "median_u8", "dither"):
    getattr(_lib, "cn_neighborhood_" + _name).restype = ct.c_int

DIFFUSION_IDS = {"FS": 0, "JJN": 1, "ST": 2, "A": 3, "B": 4, "S": 5, "S2": 6, "SL": 7}


def _image(image: np.ndarray) -> tuple[np.ndarray, int]:
    source = f32(image)
    if source.ndim not in (2, 3) or any(size == 0 for size in source.shape):
        raise ValueError(f"Invalid image shape {source.shape}")
    channels = source.shape[2] if source.ndim == 3 else 1
    return source, channels


def median(image: np.ndarray, radius: object) -> np.ndarray:
    source, channels = _image(image)
    if not isinstance(radius, int) or not 1 <= radius <= 1000:
        raise ValueError("Median radius must be an integer in [1,1000]")
    shape = source.shape[:2] if channels == 1 else source.shape
    if radius <= 2:
        if channels > 512:
            raise cv2.error("Median Blur supports at most 512 image channels.")
        result = np.empty(shape, dtype=np.float32)
        check(
            _lib.cn_neighborhood_median_f32(
                ptr(source),
                ptr(result),
                source.shape[0],
                source.shape[1],
                channels,
                radius,
                # OpenCV 4.8's installed x86 dispatch has AVX2 (8 lanes) or
                # baseline SSE2 (4 lanes). This queries CPU configuration only;
                # the comparator network and all pixel processing run in C.
                8 if cv2.checkHardwareSupport(11) else 4,
            )
        )
    else:
        from .image_utils import to_uint8

        source = np.require(to_uint8(source, normalized=True), requirements=["C", "A"])
        if channels not in (1, 3, 4):
            raise cv2.error(
                "Median Blur with radius 3 or larger requires 1, 3, or 4 channels."
            )
        result = np.empty(shape, dtype=np.uint8)
        check(
            _lib.cn_neighborhood_median_u8(
                source.ctypes.data,
                result.ctypes.data,
                source.shape[0],
                source.shape[1],
                channels,
                radius,
            )
        )
    return result


def uniform_dither(
    image: np.ndarray,
    colors: object,
    mode: int,
    *,
    map_size: int = 2,
    algorithm: int = 0,
    history_length: object = 16,
) -> np.ndarray:
    source, channels = _image(image)
    if channels not in (1, 3, 4):
        raise ValueError("Dithering requires 1, 3, or 4 channels")
    if not isinstance(colors, int) or not 2 <= colors <= 0xFFFFFFFF:
        raise ValueError("Dither colors must be an integer in [2,4294967295]")
    if mode not in (0, 1, 2, 3) or algorithm not in range(8):
        raise ValueError("Invalid dither mode or diffusion algorithm")
    if map_size not in (2, 4, 8, 16):
        raise ValueError("Bayer map must have size 2, 4, 8, or 16")
    result = np.empty((*source.shape[:2], channels), dtype=np.float32)
    if mode == 3:
        if not isinstance(history_length, int) or not 2 <= history_length <= 0xFFFFFFFF:
            raise ValueError("History length must be an integer in [2,4294967295]")
        check(
            _lib.cn_neighborhood_uniform_riemersma(
                ptr(source),
                ptr(result),
                source.shape[0],
                source.shape[1],
                channels,
                colors,
                history_length,
                1 / history_length,
            )
        )
        return result
    compatible = ct.c_int()
    check(
        _lib.cn_neighborhood_dither(
            ptr(source),
            ptr(result),
            source.shape[0],
            source.shape[1],
            channels,
            colors,
            mode,
            map_size,
            algorithm,
            None,
            0,
            ct.byref(compatible),
        )
    )
    return result


def palette_dither(
    image: np.ndarray,
    palette: np.ndarray,
    mode: int,
    *,
    algorithm: int = 0,
    history_length: object = 16,
) -> np.ndarray:
    """Exact float32 nearest lookup with deterministic palette tie ordering.

    Equal distances choose the first color ordered by luminance, then first
    input occurrence. This intentionally replaces upstream randomized equal-
    luminance order and traversal-dependent R-tree ties. Grayscale palettes
    of 300 or more unique colors are supported instead of triggering its
    one-dimensional R-tree panic. These large palettes require finite image
    and palette values; undefined nonfinite tree distances raise ValueError.
    """
    source, channels = _image(image)
    colors, palette_channels = _image(palette)
    if channels not in (1, 3, 4):
        raise ValueError("Dithering requires 1, 3, or 4 channels")
    if palette_channels != channels or colors.shape[0] != 1:
        raise ValueError("Palette must be one row with matching image channels")
    if mode not in (0, 2, 3) or algorithm not in range(8):
        raise ValueError("Invalid palette dither mode or diffusion algorithm")
    result = np.empty((*source.shape[:2], channels), dtype=np.float32)
    compatible = ct.c_int()
    if mode == 3:
        if not isinstance(history_length, int) or not 2 <= history_length <= 0xFFFFFFFF:
            raise ValueError("History length must be an integer in [2,4294967295]")
        check(
            _lib.cn_neighborhood_palette_riemersma(
                ptr(source),
                ptr(result),
                source.shape[0],
                source.shape[1],
                channels,
                ptr(colors),
                colors.shape[1],
                history_length,
                1 / history_length,
                ct.byref(compatible),
            )
        )
    else:
        check(
            _lib.cn_neighborhood_dither(
                ptr(source),
                ptr(result),
                source.shape[0],
                source.shape[1],
                channels,
                2,
                mode,
                2,
                algorithm,
                ptr(colors),
                colors.shape[1],
                ct.byref(compatible),
            )
        )
    if not compatible.value:
        raise ValueError(
            "Dithering with 300 or more unique palette colors requires "
            "finite image and palette values."
        )
    return result
