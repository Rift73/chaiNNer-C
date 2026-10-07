"""OpenCV 4.8 float32 AREA downsampling and baseline LINEAR/CUBIC interpolation.

The installed IPP LINEAR/CUBIC implementations have different numerical behavior and
remains an explicit external primitive. AREA's default dispatch is entirely C.
AREA uses the original single-thread row initialization consistently; original
parallel stripe boundaries can otherwise change the signs of underflowed zeros.
"""

from __future__ import annotations

import ctypes as ct

import cv2
import numpy as np

from .native import check, f32, lib, ptr
from .native_versions import cv2_is_tested

_lib = lib()
_p = ct.POINTER(ct.c_float)
_n = ct.c_size_t
for _name in ("cn_cv_resize_area", "cn_cv_resize_linear", "cn_cv_resize_cubic"):
    _function = getattr(_lib, _name)
    _function.argtypes = [_p, _p, _n, _n, _n, _n, _n]
    _function.restype = ct.c_int


def resize(image: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
    """Preserve cv2.resize's HWC1 collapse and default installed dispatch."""
    source = f32(image)
    if source.ndim not in (2, 3) or not all(source.shape):
        raise ValueError("Resize requires a nonempty float32 image")
    if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("Resize dimensions must be positive integers")
    if max(size) > np.iinfo(np.int32).max:
        raise OverflowError("Resize dimensions exceed the OpenCV range")
    if interpolation not in (cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC):
        raise ValueError("This resize helper supports AREA, LINEAR and CUBIC only")
    h, w = source.shape[:2]
    channels = source.shape[2] if source.ndim == 3 else 1
    dw, dh = size
    compatible = cv2_is_tested(__name__) and channels <= 4
    area_down = interpolation == cv2.INTER_AREA and dh <= h and dw <= w
    # Non-default IPP modes, unsupported channels/versions, AREA upscaling and
    # the default closed IPP interpolation primitives preserve original behavior.
    if (
        not compatible
        or (
            interpolation == cv2.INTER_AREA
            and (not area_down or (cv2.ipp.useIPP() and cv2.ipp.useIPP_NotExact()))
        )
        or (interpolation in (cv2.INTER_LINEAR, cv2.INTER_CUBIC) and cv2.ipp.useIPP())
    ):
        return cv2.resize(source, size, interpolation=interpolation)
    shape = (dh, dw) if channels == 1 else (dh, dw, channels)
    result = np.empty(shape, np.float32)
    function = {
        cv2.INTER_AREA: _lib.cn_cv_resize_area,
        cv2.INTER_LINEAR: _lib.cn_cv_resize_linear,
        cv2.INTER_CUBIC: _lib.cn_cv_resize_cubic,
    }[interpolation]
    check(function(ptr(source), ptr(result), h, w, channels, dh, dw))
    return result
