"""Checked palette bridges; bucket and clustering algorithms execute in C."""

from __future__ import annotations

import ctypes as ct
import operator
from functools import lru_cache

import cv2
import numpy as np

from .native import check, f32, lib, ptr
from .native_analysis import mean
from .native_numpy_reduce import numpy_sum_block


@lru_cache(maxsize=1)
def _api():
    function = lib().cn_palette_median_cut
    fp = ct.POINTER(ct.c_float)
    function.argtypes = [
        fp,
        fp,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(ct.c_size_t),
    ]
    function.restype = ct.c_int
    return function


def median_cut(image: np.ndarray, num_colors: int) -> np.ndarray:
    source = f32(image)
    if source.ndim != 2 or not all(source.shape) or num_colors < 1:
        raise ValueError(
            "Median cut requires nonempty pixel rows and a positive palette size"
        )
    capacity = min(num_colors, source.shape[0])
    out = np.empty((1, capacity, source.shape[1]), dtype=np.float32)
    written = ct.c_size_t()
    # Upstream's root bucket is this array (its column's NumPy 2.5.3 reduce block);
    # each split makes fresh aligned arrays, which NumPy 2.5.3 sums as one tree.
    status = _api()(
        ptr(source),
        ptr(out),
        *source.shape,
        num_colors,
        *numpy_sum_block(image[:, 0]),
        0,
        ct.byref(written),
    )
    if status == 6:
        raise ValueError(
            "zero-size array to reduction operation minimum which has no identity"
        )
    check(status)
    rows, channels = source.shape
    if (
        written.value == 1
        and rows > 1
        and abs(image.strides[0]) < abs(image.strides[1])
    ):
        # The root bucket stayed whole. When its rows are the innermost axis (a
        # planar CHW view transposed to HWC), upstream's np.mean(axis=0) reduces
        # each column as one np.mean of that column, not row by row.
        out[0, 0] = [mean(image[:, channel]) for channel in range(channels)]
    return out[:, : written.value]


@lru_cache(maxsize=1)
def _pad_api():
    function = lib().cn_palette_pad_edge
    fp = ct.POINTER(ct.c_float)
    function.argtypes = [fp, fp, ct.c_size_t, ct.c_size_t, ct.c_size_t]
    function.restype = ct.c_int
    return function


def pad_edge(image: np.ndarray, colors: int) -> np.ndarray:
    source = f32(image)
    if source.ndim != 3 or source.shape[0] != 1 or not all(source.shape):
        raise ValueError("Palette padding requires one nonempty row of colors")
    if colors < source.shape[1]:
        raise ValueError("Palette padding cannot discard colors")
    output = np.empty((1, colors, source.shape[2]), np.float32)
    check(
        _pad_api()(ptr(source), ptr(output), source.shape[1], source.shape[2], colors)
    )
    return output


@lru_cache(maxsize=1)
def _kmeans_api():
    function = lib().cn_palette_kmeans
    fp = ct.POINTER(ct.c_float)
    function.argtypes = [fp, fp, ct.c_size_t, ct.c_size_t, ct.c_size_t]
    function.restype = ct.c_int
    return function


def kmeans(image: np.ndarray, num_colors: int) -> np.ndarray:
    source = f32(image)
    colors = operator.index(num_colors)
    if source.ndim != 2 or not all(source.shape):
        raise ValueError("K-means requires nonempty pixel rows")
    # OpenCV treats a one-row matrix as a column of scalar samples. Keep that
    # helper-level quirk; the public node bypasses clustering for one pixel.
    if source.shape[0] == 1:
        source = source.reshape(-1, 1)
    if colors <= 0 or colors > source.shape[0]:
        raise cv2.error("K-means palette size must be positive and not exceed samples")
    output = np.empty((colors, source.shape[1]), np.float32)
    status = _kmeans_api()(ptr(source), ptr(output), *source.shape, colors)
    if status == 5:
        raise cv2.error(
            "kmeans: can't update cluster center (check input for huge or NaN values)"
        )
    check(status)
    return output
