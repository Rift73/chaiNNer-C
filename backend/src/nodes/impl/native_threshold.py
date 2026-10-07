"""Owned buffers for C analytic threshold antialiasing and channel reductions."""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import cv2
import numpy as np

from .native import check, f32, lib, ptr


@lru_cache(maxsize=1)
def _api():
    api = lib()
    fp, size = ct.POINTER(ct.c_float), ct.c_size_t
    api.cn_threshold_aa.argtypes = [fp, fp, size, size, size, ct.c_float, ct.c_float]
    api.cn_threshold_aa.restype = ct.c_int
    api.cn_threshold_channel_mean.argtypes = [fp, fp, size, size, size, ct.c_int]
    api.cn_threshold_channel_mean.restype = ct.c_int
    api.cn_threshold_hard.argtypes = [
        fp,
        fp,
        size,
        ct.c_int,
        ct.c_float,
        ct.c_float,
        ct.c_int,
    ]
    api.cn_threshold_hard.restype = ct.c_int
    return api


def _image(image: np.ndarray) -> tuple[np.ndarray, int]:
    source = f32(image)
    if source.ndim not in (2, 3) or not all(source.shape):
        raise ValueError("Threshold requires a nonempty image")
    return source, source.shape[2] if source.ndim == 3 else 1


def binary_threshold_aa(
    image: np.ndarray, threshold: float, smoothness: float
) -> np.ndarray:
    source, channels = _image(image)
    result = np.empty((*source.shape[:2], channels), np.float32)
    check(
        _api().cn_threshold_aa(
            ptr(source), ptr(result), *source.shape[:2], channels, threshold, smoothness
        )
    )
    return result


def _channels_outer(image: np.ndarray) -> bool:
    """Whether np.mean(image, axis=-1) iterates the channel axis outside another one.

    NumPy's iterator moves an axis inward only past a strictly larger stride (zero
    strides decide nothing), so the channel axis stays innermost unless another axis
    of length > 1 has a smaller nonzero stride (a planar CHW view as HWC).
    """
    if image.ndim != 3:
        return False
    channel = abs(image.strides[2])
    return any(
        length > 1 and 0 < abs(stride) < channel
        for length, stride in zip(image.shape[:2], image.strides[:2], strict=True)
    )


def channel_mean(image: np.ndarray) -> np.ndarray:
    """np.mean(image, axis=-1) of ``image`` in its own layout."""
    source, channels = _image(image)
    result = np.empty(source.shape[:2], np.float32)
    # An unaligned image is a cast operand: NumPy 2.5.3 buffers it and reduces each
    # pixel's channels in NPY_BUFSIZE (8192) calls; an aligned one in one call.
    block = 0 if image.flags.aligned else 8192
    check(
        _api().cn_threshold_channel_mean(
            ptr(source),
            ptr(result),
            result.size,
            channels,
            block,
            _channels_outer(image),
        )
    )
    return result


def hard_threshold(
    image: np.ndarray, threshold: float, maximum: float, kind: int
) -> np.ndarray:
    source, channels = _image(image)
    result = np.empty(source.shape[:2] if channels == 1 else source.shape, np.float32)
    check(
        _api().cn_threshold_hard(
            ptr(source),
            ptr(result),
            source.size,
            kind,
            threshold,
            maximum,
            cv2.ipp.useIPP(),
        )
    )
    return result
