"""Native byte-image Canny and FMM inpainting; quantization stays shared."""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import cv2
import numpy as np

from .native import check, lib
from .native_versions import CV_CN_MAX


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    pointer, size = ct.c_void_p, ct.c_size_t
    for name, arguments in {
        "cn_canny_u8": [pointer, pointer, size, size, size, ct.c_double, ct.c_double],
        "cn_canny_u8_policy": [
            pointer,
            pointer,
            size,
            size,
            size,
            ct.c_double,
            ct.c_double,
            ct.c_int,
        ],
        "cn_inpaint_u8": [
            pointer,
            pointer,
            pointer,
            size,
            size,
            size,
            ct.c_double,
            ct.c_int,
        ],
    }.items():
        function = getattr(dll, name)
        function.argtypes, function.restype = arguments, ct.c_int
    return dll


def canny(image: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("Canny expects an unsigned-byte HW or HWC image")
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    if not height or not width or not 1 <= channels <= CV_CN_MAX:
        # Outside the native contract the bridge defers to OpenCV exactly as
        # upstream's node calls cv2.Canny: OpenCV raises for unsupported channel
        # layouts and for images without columns or channels, and returns None
        # for an image without rows (test_repair_ops.py:
        # test_canny_channel_ceiling, test_canny_empty_images_defer_to_opencv).
        return cv2.Canny(image, lower, upper)
    source = np.ascontiguousarray(image)
    output = np.empty((height, width), np.uint8)
    check(
        _api().cn_canny_u8_policy(
            source.ctypes.data,
            output.ctypes.data,
            height,
            width,
            channels,
            lower,
            upper,
            int(
                channels == 1
                and height > 3
                and width > 3
                and cv2.getNumThreads() <= 1
                and cv2.ipp.useIPP()
            ),
        )
    )
    return output


def inpaint(
    image: np.ndarray, mask: np.ndarray, radius: float, method: int
) -> np.ndarray:
    if image.dtype != np.uint8 or mask.dtype != np.uint8:
        raise TypeError("Inpaint expects unsigned-byte image and mask buffers")
    if image.ndim not in (2, 3) or mask.ndim not in (2, 3):
        raise ValueError("Inpaint expects HW or HWC buffers")
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    if (
        not height
        or not width
        or channels not in (1, 3)
        or image.shape[:2] != mask.shape[:2]
        or (mask.ndim == 3 and mask.shape[2] != 1)
        or method not in (0, 1)
    ):
        raise ValueError(
            "Inpaint requires matching nonempty gray/BGR image and gray mask, using NS or Telea"
        )
    source, stencil = np.ascontiguousarray(image), np.ascontiguousarray(mask)
    output = np.empty(
        (height, width) if channels == 1 else (height, width, channels), np.uint8
    )
    check(
        _api().cn_inpaint_u8(
            source.ctypes.data,
            stencil.ctypes.data,
            output.ctypes.data,
            height,
            width,
            channels,
            radius,
            method,
        )
    )
    return output
