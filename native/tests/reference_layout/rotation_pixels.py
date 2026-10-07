"""Frozen original conversion helpers for independent rotation parity."""
from __future__ import annotations
import cv2
import numpy as np
from nodes.utils.utils import get_h_w_c

def convert_to_bgra(img: np.ndarray, in_c: int) -> np.ndarray:
    assert in_c in (1, 3, 4), f"Number of channels ({in_c}) unexpected"
    if in_c == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif in_c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    return img.copy()

def _get_iinfo(img: np.ndarray) -> np.iinfo | None:
    try:
        return np.iinfo(img.dtype)
    except Exception:
        return None

def normalize(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.float32:
        info = _get_iinfo(img)
        img = img.astype(np.float32)

        if info is not None:
            img /= info.max
            if info.min == 0:
                # we don't need to clip
                return img

        # we own `img`, so it's okay to write to it
        return np.clip(img, 0, 1, out=img)

    return np.clip(img, 0, 1)

def to_uint8(img: np.ndarray, normalized: bool = False) -> np.ndarray:
    """
    Returns a new uint8 image with the given image data.

    If `normalized` is `False`, then the image will be normalized before being converted to uint8.
    """
    if img.dtype == np.uint8:
        return img.copy()

    if not normalized or img.dtype != np.float32:
        img = normalize(img)

    return (img * 255).round().astype(np.uint8)
