"""Checked C filter buffers; specialized OpenCV primitives remain explicit."""

from __future__ import annotations

import ctypes as ct
import operator

import cv2
import numpy as np

from .native import check, f32, lib, ptr
from .native_image_setup import exceptional
from .native_opencv_simd import float32_lanes, target
from .native_versions import CV_CN_MAX

_lib = lib()
_p = ct.POINTER(ct.c_float)
_n = ct.c_size_t
_lib.cn_filter_arithmetic.argtypes = [
    _p,
    _p,
    _p,
    _p,
    _p,
    _n,
    ct.c_int,
    ct.c_float,
    ct.c_float,
    ct.c_int,
]
_lib.cn_filter_lens_normalize.argtypes = [_p, ct.c_void_p, _p, _n, _n]
_lib.cn_filter_morphology.argtypes = [_p, _p, _n, _n, _n, _n, _n, ct.c_int, ct.c_int]
_lib.cn_morphology_complete.argtypes = [
    _p,
    _p,
    _n,
    _n,
    _n,
    _n,
    _n,
    ct.c_int,
    ct.c_int,
    _n,
]
_lib.cn_morphology_complete.restype = ct.c_int
_lib.cn_morphology_ellipse_spans.argtypes = [_n, ct.POINTER(_n)]
_lib.cn_morphology_ellipse_spans.restype = ct.c_int
_lib.cn_filter_quantize_reference.argtypes = [
    _p,
    _p,
    _p,
    _n,
    _n,
    _n,
    _n,
    _n,
    _n,
    ct.c_double,
]
for _name in (
    "arithmetic",
    "lens_normalize",
    "morphology",
    "quantize_reference",
):
    getattr(_lib, "cn_filter_" + _name).restype = ct.c_int


def _image(image: np.ndarray) -> tuple[np.ndarray, int]:
    source = f32(image)
    if source.ndim not in (2, 3) or any(n == 0 for n in source.shape):
        raise ValueError(f"Invalid image shape {source.shape}")
    return source, source.shape[2] if source.ndim == 3 else 1


def arithmetic(
    image: np.ndarray,
    operation: int,
    scale: float = 1,
    *,
    b: np.ndarray | None = None,
    c: np.ndarray | None = None,
    d: np.ndarray | None = None,
    second: float = 0,
    clip: bool = False,
    out: np.ndarray | None = None,
) -> np.ndarray:
    source, _ = _image(image)
    operands: list[np.ndarray | None] = [source]
    for value in (b, c, d):
        if value is None:
            operands.append(None)
        else:
            converted, _ = _image(value)
            if converted.shape != source.shape:
                raise ValueError("Filter operands must have identical shapes")
            operands.append(converted)
    if operation == 10 and out is None:
        raise ValueError("Accumulating a lens component requires initialized output")
    if out is None:
        result = np.empty_like(source)
    else:
        if (
            out.shape != source.shape
            or out.dtype != np.float32
            or not out.flags.c_contiguous
            or not out.flags.aligned
            or not out.flags.writeable
        ):
            raise ValueError(
                "Filter output must be writable aligned contiguous float32 of matching shape"
            )
        if any(
            value is not None and np.shares_memory(out, value)
            for value in (image, b, c, d)
        ):
            raise ValueError("Filter output must not overlap an input operand")
        result = out
    check(
        _lib.cn_filter_arithmetic(
            *(ptr(value) if value is not None else None for value in operands),
            ptr(result),
            source.size,
            operation,
            scale,
            second,
            clip,
        )
    )
    return result


def normalize_lens(
    kernels: list[np.ndarray],
    coefficients: list[tuple[float, float]] | np.ndarray,
) -> list[np.ndarray]:
    if not kernels or len(kernels) != len(coefficients):
        raise ValueError(
            "Lens kernels and coefficients must have matching nonzero counts"
        )
    shape = kernels[0].shape
    if len(shape) != 2 or shape[0] != 1 or not shape[1]:
        raise ValueError("Lens kernels must have shape (1,N)")
    if any(k.dtype != np.complex64 or k.shape != shape for k in kernels):
        raise ValueError("Lens kernels must be complex64 arrays of equal shape")
    source = np.require(np.stack(kernels), requirements=["C", "A"])
    parameters = np.require(coefficients, dtype=np.float64, requirements=["C", "A"])
    if parameters.shape != (len(kernels), 2):
        raise ValueError("Every lens kernel requires two coefficients")
    result = np.empty_like(source)
    check(
        _lib.cn_filter_lens_normalize(
            ptr(source.view(np.float32)),
            parameters.ctypes.data,
            ptr(result.view(np.float32)),
            len(kernels),
            shape[1],
        )
    )
    return [result[index] for index in range(len(kernels))]


def morphology(
    image: np.ndarray, shape: int, radius: int, iterations: int, *, maximum: bool
) -> np.ndarray:
    source, channels = _image(image)
    radius = operator.index(radius)
    iterations = operator.index(iterations)
    if radius < 0 or iterations < 0:
        raise ValueError("Morphology radius and iterations must be nonnegative")
    if radius == 0 or iterations == 0:
        return image
    if shape not in (cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE):
        raise ValueError("Unsupported morphology shape")
    if channels > CV_CN_MAX:
        raise cv2.error(f"Morphology supports at most {CV_CN_MAX} image channels.")
    # A one-channel result is allocated in the (h, w) shape it is returned in, so it
    # owns its data like every other result (the C entries see the same floats).
    result = np.empty(source.shape[:2] if channels == 1 else source.shape, np.float32)
    if shape == cv2.MORPH_ELLIPSE or exceptional(source):
        # Only OpenCV's morph dispatch width (its lane/tail selection) is
        # needed; C performs the kernel generation and every operation.
        lanes = float32_lanes(target("morph"))
        check(
            _lib.cn_morphology_complete(
                ptr(source),
                ptr(result),
                *source.shape[:2],
                channels,
                radius,
                iterations,
                shape,
                maximum,
                lanes,
            )
        )
        return result
    check(
        _lib.cn_filter_morphology(
            ptr(source),
            ptr(result),
            source.shape[0],
            source.shape[1],
            channels,
            radius,
            iterations,
            shape == cv2.MORPH_CROSS,
            maximum,
        )
    )
    return result


def quantize_reference(
    image: np.ndarray, reference: np.ndarray, radius: int, spatial: float
) -> np.ndarray:
    source, channels = _image(image)
    palette, palette_channels = _image(reference)
    if channels != palette_channels:
        raise ValueError("Image and reference must have matching channels")
    result = np.empty_like(source)
    check(
        _lib.cn_filter_quantize_reference(
            ptr(source),
            ptr(palette),
            ptr(result),
            source.shape[0],
            source.shape[1],
            palette.shape[0],
            palette.shape[1],
            channels,
            radius,
            spatial,
        )
    )
    return result


def binary_sdf(image_u8: np.ndarray, spread: float) -> np.ndarray:
    """C scalar chamfer; preserve the external IPP primitive's exact ordering."""
    if image_u8.dtype != np.uint8 or image_u8.ndim not in (2, 3):
        raise TypeError("Binary SDF requires an unsigned-byte grayscale image")
    if not image_u8.size or (image_u8.ndim == 3 and image_u8.shape[2] != 1):
        raise ValueError("Binary SDF requires a nonempty grayscale image")
    source = np.require(image_u8, requirements=["C", "A"])
    result = np.empty((*source.shape[:2], 1), dtype=np.float32)
    double_division = np.result_type(np.dtype(np.float32), spread) == np.float64
    if cv2.ipp.useIPP():
        # Intel's closed SIMD implementation chooses a different addition order
        # for some mathematically tied paths. Replacing it changed distances by
        # one ULP. Keep only that primitive; masks and composition remain C.
        masks = _lib.cn_distance_binary_masks
        masks.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, _n, _n]
        masks.restype = ct.c_int
        black_mask = np.empty(source.shape[:2], np.uint8)
        white_mask = np.empty(source.shape[:2], np.uint8)
        check(
            masks(
                source.ctypes.data,
                black_mask.ctypes.data,
                white_mask.ctypes.data,
                source.shape[0],
                source.shape[1],
            )
        )
        float_depth = cv2.CV_32F
        black = cv2.distanceTransform(black_mask, cv2.DIST_L2, 5, dstType=float_depth)
        white = cv2.distanceTransform(white_mask, cv2.DIST_L2, 5, dstType=float_depth)
        combine = _lib.cn_distance_binary_combine
        combine.argtypes = [ct.c_void_p, _p, _p, _p, _n, _n, ct.c_double, ct.c_int]
        combine.restype = ct.c_int
        check(
            combine(
                source.ctypes.data,
                ptr(black),
                ptr(white),
                ptr(result),
                source.shape[0],
                source.shape[1],
                spread,
                double_division,
            )
        )
        return result
    function = _lib.cn_distance_binary
    function.argtypes = [ct.c_void_p, _p, _n, _n, ct.c_double, ct.c_int]
    function.restype = ct.c_int
    check(
        function(
            source.ctypes.data,
            ptr(result),
            source.shape[0],
            source.shape[1],
            spread,
            double_division,
        )
    )
    return result


def subpixel_sdf(image: np.ndarray, radius: float) -> np.ndarray:
    """Complete C ESDF variant selected by the Distance Transform node."""
    source, channels = _image(image)
    if channels != 1:
        raise ValueError("Subpixel distance transform requires one image channel")
    function = _lib.cn_distance_esdf
    function.argtypes = [_p, _p, _n, _n, ct.c_float]
    function.restype = ct.c_int
    result = np.empty((*source.shape[:2], 1), dtype=np.float32)
    check(function(ptr(source), ptr(result), source.shape[0], source.shape[1], radius))
    return result
