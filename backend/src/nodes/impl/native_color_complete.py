"""Float32 color transforms preserving the installed OpenCV row dispatch."""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import cv2
import numpy as np

from .native import check, f32, lib, ptr
from .native_numpy_simd import fma3, loop_target
from .native_opencv_simd import float32_lanes, target
from .native_versions import cv2_is_tested


@lru_cache(maxsize=1)
def _api():
    library = lib()
    fp, size = ct.POINTER(ct.c_float), ct.c_size_t
    library.cn_color_convert_f32.argtypes = [fp, fp, size, size, size, size] + [
        ct.c_int
    ] * 4
    library.cn_color_convert_f32.restype = ct.c_int
    return library


# Operation, blue index, output channel count. OpenCV aliases share numbers.
_CODES = {
    cv2.COLOR_BGR2BGRA: (0, 0, 4),
    cv2.COLOR_BGRA2BGR: (0, 0, 3),
    cv2.COLOR_BGR2RGBA: (0, 2, 4),
    cv2.COLOR_BGRA2RGB: (0, 2, 3),
    cv2.COLOR_BGR2RGB: (0, 2, 3),
    cv2.COLOR_BGRA2RGBA: (0, 2, 4),
    cv2.COLOR_GRAY2BGR: (0, 0, 3),
    cv2.COLOR_GRAY2BGRA: (0, 0, 4),
    cv2.COLOR_BGR2GRAY: (1, 0, 1),
    cv2.COLOR_BGRA2GRAY: (1, 0, 1),
    cv2.COLOR_RGB2GRAY: (1, 2, 1),
    cv2.COLOR_RGBA2GRAY: (1, 2, 1),
    cv2.COLOR_BGR2YUV: (2, 0, 3),
    cv2.COLOR_RGB2YUV: (2, 2, 3),
    cv2.COLOR_YUV2BGR: (3, 0, 3),
    cv2.COLOR_YUV2RGB: (3, 2, 3),
    cv2.COLOR_BGR2HSV: (4, 0, 3),
    cv2.COLOR_RGB2HSV: (4, 2, 3),
    cv2.COLOR_HSV2BGR: (5, 0, 3),
    cv2.COLOR_HSV2RGB: (5, 2, 3),
    cv2.COLOR_BGR2HLS: (6, 0, 3),
    cv2.COLOR_RGB2HLS: (6, 2, 3),
    cv2.COLOR_HLS2BGR: (7, 0, 3),
    cv2.COLOR_HLS2RGB: (7, 2, 3),
    cv2.COLOR_BGR2YCrCb: (8, 0, 3),
    cv2.COLOR_RGB2YCrCb: (8, 2, 3),
    cv2.COLOR_YCrCb2BGR: (9, 0, 3),
    cv2.COLOR_YCrCb2RGB: (9, 2, 3),
    cv2.COLOR_BGR2LAB: (10, 0, 3),
    cv2.COLOR_RGB2LAB: (10, 2, 3),
    cv2.COLOR_LAB2BGR: (11, 0, 3),
    cv2.COLOR_LAB2RGB: (11, 2, 3),
}

# The dispatched OpenCV source holding each operation's float32 row loop (14 is
# the colorwheel's HSV2RGB). Lab (10, 11) lives in color_lab.cpp, which is not
# dispatched: it runs the baseline's four-lane routine.
_FAMILIES = {
    0: "color_rgb",
    1: "color_rgb",
    2: "color_yuv",
    3: "color_yuv",
    4: "color_hsv",
    5: "color_hsv",
    6: "color_hsv",
    7: "color_hsv",
    8: "color_yuv",
    9: "color_yuv",
    14: "color_hsv",
}


def _lanes(op: int) -> int:
    family = _FAMILIES.get(op)
    return float32_lanes(target(family) if family else "BASELINE")


def cvt_color(image: np.ndarray, code: int) -> np.ndarray:
    """Convert supported float32 images; retain other dtype/version contracts."""
    if image.dtype != np.float32 or code not in _CODES or not cv2_is_tested(__name__):
        return cv2.cvtColor(image, code)
    source = f32(image)
    if source.ndim not in (2, 3) or any(size == 0 for size in source.shape):
        raise ValueError("Color conversion requires a nonempty image")
    channels = source.shape[2] if source.ndim == 3 else 1
    op, blue, output_channels = _CODES[code]
    shape = (
        source.shape[:2]
        if output_channels == 1
        else (*source.shape[:2], output_channels)
    )
    result = np.empty(shape, np.float32)
    check(
        _api().cn_color_convert_f32(
            ptr(source),
            ptr(result),
            *source.shape[:2],
            channels,
            output_channels,
            op,
            blue,
            # The baseline's four lanes even with optimizations disabled; AVX2's
            # eight use explicitly fused multiply-add expressions.
            _lanes(op),
            int(cv2.ipp.useIPP()),
        )
    )
    return result


def lab_lch(image: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    """Convert normalized Lab/LCH including NumPy's selected trig arithmetic."""
    source = f32(image)
    if source.ndim != 3 or source.shape[2] != 3 or not source.size:
        raise ValueError("Lab/LCH conversion requires a nonempty three-channel image")
    # The sine/cosine mirror takes the FMA3 form of np.sin's float32 loop; np.cos
    # shares its dispatch source (loops_trigonometric), hence its target.
    use_fma = fma3(loop_target("sin", "ff"))
    result = np.empty(source.shape, np.float32)
    check(
        _api().cn_color_convert_f32(
            ptr(source),
            ptr(result),
            *source.shape[:2],
            3,
            3,
            13 if inverse else 12,
            0,
            4,
            int(use_fma),
        )
    )
    return result


def colorwheel(gradient: np.ndarray) -> np.ndarray:
    source = f32(gradient)
    if source.ndim != 2 or not source.size:
        raise ValueError("Colorwheel requires a nonempty single-channel gradient")
    result = np.empty((*source.shape, 3), np.float32)
    check(
        _api().cn_color_convert_f32(
            ptr(source), ptr(result), *source.shape, 1, 3, 14, 0, _lanes(14), 0
        )
    )
    return result
