"""YUV plane maths in C: codes <-> plane values (colour_info.py's convention), one axis
of plane_resample.py's resampler, and yuv_rgb.py's matrix step.

Every kernel keeps the NumPy operation sequence in float32, so its results are the
NumPy forms' bits (native/tests/test_yuv_planes.py).
"""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import numpy as np

from .native import check, in_order, lib


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    p, z, i, f, u = ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_float, ct.c_uint32
    signatures = {
        "cn_plane_codes_to_f32": [p, i, z, z, z, f, f, i, p],
        "cn_plane_f32_to_codes": [p, z, z, z, f, f, i, u, i, p],
        "cn_plane_resample_axis": [p, z, z, z, i, p, p, z, z, f, f, p],
        "cn_yuv_to_bgr": [p, z, p, z, p, z, z, z, p, f, p],
        "cn_bgr_to_yuv": [p, z, z, z, p, f, p, p, p],
    }
    for name, signature in signatures.items():
        function = getattr(dll, name)
        function.argtypes, function.restype = signature, i
    return dll


def _row_stride(array: np.ndarray) -> int | None:
    """The row stride in elements of a 2-D aligned array whose rows are runs of
    contiguous elements that do not overlap (a crop of wider rows included), else
    None."""
    height, width = array.shape
    item = array.itemsize
    row, column = array.strides
    if width == 1:
        column = item
    if height == 1:
        row = width * item
    if (
        array.flags.aligned
        and column == item
        and row >= width * item
        and row % item == 0
    ):
        return row // item
    return None


def _plane(array: np.ndarray) -> tuple[np.ndarray, int]:
    """A 2-D array as rows a kernel reads: (array, row stride in elements) in place
    where _row_stride allows, else (a C-contiguous copy, width)."""
    stride = _row_stride(array)
    if stride is None:
        array = np.require(array, requirements=["C", "A"])
        stride = array.shape[1]
    return array, stride


def _rows(
    source: np.ndarray, dtype: type[np.generic]
) -> tuple[np.ndarray, int, int, int, np.ndarray, np.ndarray]:
    """(source, its row stride in elements, height, width, target, out) for an
    elementwise kernel that reads height rows of width contiguous elements and writes
    target, a C-contiguous view of out, which has NumPy's elementwise result layout
    (order K). A 2-D plane is read in place where _row_stride allows; any other array
    through in_order (copied only if it has gaps)."""
    stride = _row_stride(source) if source.ndim == 2 else None
    if stride is not None:
        out = np.empty(source.shape, dtype)
        height, width = source.shape
        return source, stride, height, width, out, out
    source, target, out = in_order(source, dtype)
    width = source.shape[-1] if source.ndim else 1
    height = source.size // width if width else 0
    return source, width, height, width, target, out


def codes_to_float(
    plane: np.ndarray, offset: float, scale: float, chroma: bool
) -> np.ndarray:
    """colour_info.code_to_float of uint8 or uint16 codes, with its (offset, scale)."""
    source, stride, height, width, target, out = _rows(plane, np.float32)
    if out.size:
        check(
            _api().cn_plane_codes_to_f32(
                source.ctypes.data,
                int(source.dtype == np.uint16),
                stride,
                height,
                width,
                offset,
                scale,
                int(chroma),
                target.ctypes.data,
            )
        )
    return out


def float_to_codes(
    plane: np.ndarray, offset: float, scale: float, chroma: bool, bit_depth: int
) -> np.ndarray:
    """colour_info.float_to_code of a plane (cast to float32 first, as it is), with
    its (offset, scale), for up to 16 bits."""
    wide = bit_depth > 8
    values = np.asarray(plane, np.float32)
    source, stride, height, width, target, out = _rows(
        values, np.uint16 if wide else np.uint8
    )
    if out.size:
        check(
            _api().cn_plane_f32_to_codes(
                source.ctypes.data,
                stride,
                height,
                width,
                offset,
                scale,
                int(chroma),
                (1 << bit_depth) - 1,
                int(wide),
                target.ctypes.data,
            )
        )
    return out


def resample_axis(
    plane: np.ndarray,
    vertical: bool,
    index: np.ndarray,
    weights: np.ndarray,
    bias_in: float,
    bias_out: float,
) -> np.ndarray:
    """One pass of plane_resample.py over a float32 plane, along its rows (vertical
    False) or its columns: output i is the sum, from +0 in tap order, of
    (input[index[t, i]] - bias_in) * weights[t, i], plus bias_out when it is not 0.
    index (int32) and weights (float32) are taps x outputs, C-contiguous."""
    source, stride = _plane(plane)
    height, width = source.shape
    taps, size = index.shape
    out = np.empty((size, width) if vertical else (height, size), np.float32)
    check(
        _api().cn_plane_resample_axis(
            source.ctypes.data,
            stride,
            height,
            width,
            int(vertical),
            index.ctypes.data,
            weights.ctypes.data,
            taps,
            size,
            bias_in,
            bias_out,
            out.ctypes.data,
        )
    )
    return out


def yuv_to_bgr(
    y: np.ndarray, u: np.ndarray, v: np.ndarray, matrix: np.ndarray, centre: float
) -> np.ndarray:
    """yuv_rgb.py's matrix step to RGB: float32 Y, U and V planes of one size as a
    float32 BGR image. matrix: 3 x 3 float32, rows R, G, B by Y, U, V; centre, when
    not 0, is subtracted from U and V first."""
    if not y.shape == u.shape == v.shape:
        raise ValueError("The Y, U and V planes must have one size for the matrix.")
    (y, y_stride), (u, u_stride), (v, v_stride) = (_plane(p) for p in (y, u, v))
    height, width = y.shape
    coefficients = np.ascontiguousarray(matrix, np.float32)
    out = np.empty((height, width, 3), np.float32)
    check(
        _api().cn_yuv_to_bgr(
            y.ctypes.data,
            y_stride,
            u.ctypes.data,
            u_stride,
            v.ctypes.data,
            v_stride,
            height,
            width,
            coefficients.ctypes.data,
            centre,
            out.ctypes.data,
        )
    )
    return out


def bgr_to_yuv(
    image: np.ndarray, matrix: np.ndarray, centre: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """yuv_rgb.py's matrix step from RGB: a BGR image (cast to float32) as float32
    Y, U and V planes. matrix: 3 x 3 float32, rows Y, U, V by R, G, B; centre, when
    not 0, is added to U and V last."""
    source = np.asarray(image, np.float32)
    height, width, _ = source.shape
    row, pixel, channel = source.strides
    if width == 1:
        pixel = 3 * source.itemsize
    if height == 1:
        row = 3 * width * source.itemsize
    if not (
        source.flags.aligned
        and channel == source.itemsize
        and pixel == 3 * source.itemsize
        and row >= 3 * width * source.itemsize
        and row % source.itemsize == 0
    ):
        source = np.require(source, requirements=["C", "A"])
        row = 3 * width * source.itemsize
    coefficients = np.ascontiguousarray(matrix, np.float32)
    y, u, v = (np.empty((height, width), np.float32) for _ in range(3))
    check(
        _api().cn_bgr_to_yuv(
            source.ctypes.data,
            row // source.itemsize,
            height,
            width,
            coefficients.ctypes.data,
            centre,
            y.ctypes.data,
            u.ctypes.data,
            v.ctypes.data,
        )
    )
    return y, u, v
