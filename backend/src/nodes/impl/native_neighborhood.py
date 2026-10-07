"""Checked median and dithering kernels with explicit array contracts."""

from __future__ import annotations

import ctypes as ct
import weakref

import cv2
import numpy as np

from .native import check, f32, lib, ptr
from .native_opencv_simd import float32_lanes, target
from .native_prepared import PreparedCache, float_state
from .native_versions import CV_CN_MAX

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
        if channels > CV_CN_MAX:
            raise cv2.error(f"Median Blur supports at most {CV_CN_MAX} image channels.")
        result = np.empty(shape, dtype=np.float32)
        check(
            _lib.cn_neighborhood_median_f32(
                ptr(source),
                ptr(result),
                source.shape[0],
                source.shape[1],
                channels,
                radius,
                # The vector width OpenCV's median_blur dispatch runs at; the
                # comparator network and all pixel processing run in C.
                float32_lanes(target("median_blur")),
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
            None,
        )
    )
    return result


class _PalettePlan:
    def __init__(self, colors: np.ndarray, channels: int):
        self.handle = ct.c_void_p()
        size = ct.c_size_t()
        create = _lib.cn_neighborhood_palette_create
        create.argtypes = [
            ct.c_void_p,
            ct.c_size_t,
            ct.c_size_t,
            ct.POINTER(ct.c_void_p),
            ct.POINTER(ct.c_size_t),
        ]
        create.restype = ct.c_int
        check(
            create(
                colors.ctypes.data,
                colors.shape[1],
                channels,
                ct.byref(self.handle),
                ct.byref(size),
            )
        )
        free = _lib.cn_neighborhood_palette_free
        free.argtypes = [ct.c_void_p]
        free.restype = None
        self._release = weakref.finalize(self, free, self.handle)
        self.bytes = size.value


_palette_plans = PreparedCache[_PalettePlan](4 * 1024 * 1024, 16)


def _prepared_palette(colors: np.ndarray, channels: int) -> _PalettePlan:
    payload = colors.tobytes()
    key = (channels, payload, float_state())

    def build():
        plan = _PalettePlan(colors, channels)
        return plan, plan.bytes + len(payload)

    return _palette_plans.get(key, build)


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
    one-dimensional R-tree panic. With 300 or more unique colors, a pixel with
    a nonfinite channel, or every pixel when the palette holds a nonfinite
    color, takes the linear rule used below 300 colors (Consult 8 D-5).
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
    if mode == 3 and (
        not isinstance(history_length, int) or not 2 <= history_length <= 0xFFFFFFFF
    ):
        raise ValueError("History length must be an integer in [2,4294967295]")
    plan = _prepared_palette(colors, channels)
    apply = _lib.cn_neighborhood_palette_apply
    apply.argtypes = (
        [ct.c_void_p, ct.c_void_p]
        + [ct.c_size_t] * 3
        + [
            ct.c_void_p,
            ct.c_int,
            ct.c_int,
            ct.c_uint32,
            ct.c_float,
            ct.POINTER(ct.c_int),
        ]
    )
    apply.restype = ct.c_int
    length = history_length if mode == 3 and isinstance(history_length, int) else 16
    check(
        apply(
            source.ctypes.data,
            result.ctypes.data,
            *source.shape[:2],
            channels,
            plan.handle,
            mode,
            algorithm,
            length,
            1 / length,
            None,
        )
    )
    return result
