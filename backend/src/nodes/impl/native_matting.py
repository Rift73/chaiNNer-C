"""Closed-form alpha and multilevel foreground matting in C (PyMatting 1.1.16).

The public NumPy OpenBLAS dot primitive retains the pinned CG reduction order.
The foreground solver starts from PyMatting 1.1.16's mean foreground and
background colours.
"""

from __future__ import annotations

import ctypes as ct
import operator
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

from .native import (
    check,
    empty_ordered,
    lib,
    numpy_order,
    ordered_layout,
    pixel_source,
    pixel_target,
)


@lru_cache(maxsize=1)
def _dot():
    libs = Path(np.__file__).resolve().parent.parent / "numpy.libs"
    paths = list(libs.glob("libscipy_openblas64_*.dll"))
    if len(paths) != 1:
        raise RuntimeError(
            "Alpha Matting requires NumPy's public ILP64 OpenBLAS library"
        )
    library = ct.CDLL(str(paths[0]))
    try:
        function = library.scipy_cblas_ddot64_
    except AttributeError as error:
        raise RuntimeError(
            "NumPy OpenBLAS lacks the public scipy_cblas_ddot64_ export"
        ) from error
    return library, ct.cast(function, ct.c_void_p)


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    p, z, i = ct.c_void_p, ct.c_size_t, ct.c_int
    for name, arguments in {
        "cn_matting_cf_stencil": [p, p, z, z, p, p, p, p],
        "cn_matting_cf_alpha": [p, p, z, z, p, p, p, p],
        "cn_matting_foreground": [p, p, z, z, p, p],
        "cn_matting_output": [p, p, z, i, p],
        "cn_matting_erode_mask_u8": [p, p, z, z, z],
        "cn_matting_trimap_output": [p, z, z, p, z, p, z, z],
    }.items():
        function = getattr(dll, name)
        function.argtypes, function.restype = arguments, i
    return dll


def _inputs(image: np.ndarray, matte: np.ndarray):
    if (
        image.dtype != np.float64
        or image.ndim != 3
        or image.shape[2] != 3
        or image.size == 0
        or matte.dtype != np.float64
        or matte.shape != image.shape[:2]
    ):
        raise ValueError(
            "Matting requires nonempty float64 RGB and matching 2D matte arrays"
        )
    return (
        np.require(image, requirements=["C", "A"]),
        np.require(matte, requirements=["C", "A"]),
    )


def _raise_solver_error(code: int):
    if code == 6:
        raise ZeroDivisionError("division by zero")
    if code:
        messages = {
            1: "Trimap did not contain background values (values <= 0.100000)",
            2: "Trimap did not contain foreground values (values >= 0.900000)",
            3: "Thresholded incomplete Cholesky decomposition failed due to insufficient positive-definiteness of matrix A and diagonal shifts did not help.",
            4: "Thresholded incomplete Cholesky decomposition failed because more than max_nnz non-zero elements were created. Try increasing max_nnz or discard_threshold.",
            5: "Conjugate gradient descent did not converge within 10000 iterations",
        }
        raise ValueError(messages[code])


def estimate_alpha(image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
    source, matte = _inputs(image, trimap)
    # These reductions preserve the original validation diagnostics only;
    # Laplacian, sparse construction, preconditioning and solving are in C.
    for name, array in (("Image", source), ("Trimap", matte)):
        minimum, maximum = array.min(), array.max()
        if minimum < 0:
            warnings.warn(
                f"{name} values should be in [0, 1], but {name.lower()}.min() is {minimum}.",
                stacklevel=2,
            )
        if maximum > 1:
            displayed = minimum if name == "Trimap" else maximum
            warnings.warn(
                f"{name} values should be in [0, 1], but {name.lower()}.max() is {displayed}.",
                stacklevel=2,
            )
    result = np.empty(matte.shape, np.float64)
    error, retries = ct.c_int(), ct.c_int()
    # Keep the library owner alive even during concurrent first-call setup.
    backend = _dot()
    check(
        _api().cn_matting_cf_alpha(
            source.ctypes.data,
            matte.ctypes.data,
            *matte.shape,
            result.ctypes.data,
            backend[1],
            ct.byref(error),
            ct.byref(retries),
        )
    )
    shifts = (0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 10, 100, 1e3, 1e4, 1e5)
    for shift in shifts[: retries.value]:
        print("PERFORMANCE WARNING:")
        print(
            "Thresholded incomplete Cholesky decomposition failed due to insufficient positive-definiteness of matrix A with parameters:"
        )
        print(f"    discard_threshold = {1e-4:e}")
        print(f"    shift = {shift:e}")
        print("Try decreasing discard_threshold or start with a larger shift")
        print("")
    _raise_solver_error(error.value)
    return result


def estimate_foreground(
    image: np.ndarray, alpha: np.ndarray, return_background: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    source, matte = _inputs(image, alpha)
    if matte.size == 1:
        raise ZeroDivisionError("division by zero")
    foreground = np.empty(source.shape, np.float32)
    background = np.empty(source.shape, np.float32) if return_background else None
    check(
        _api().cn_matting_foreground(
            source.ctypes.data,
            matte.ctypes.data,
            *matte.shape,
            foreground.ctypes.data,
            None if background is None else background.ctypes.data,
        )
    )
    return foreground if background is None else (foreground, background)


def output(
    foreground: np.ndarray, alpha: np.ndarray, *, float64_output: bool = True
) -> np.ndarray:
    if (
        foreground.dtype != np.float32
        or foreground.ndim != 3
        or foreground.shape[2] != 3
        or foreground.size == 0
        or alpha.dtype != np.float64
        or alpha.shape != foreground.shape[:2]
    ):
        raise ValueError(
            "Matting output requires float32 RGB and matching float64 alpha"
        )
    foreground, alpha = (
        np.require(a, requirements=["C", "A"]) for a in (foreground, alpha)
    )
    result = np.empty((*alpha.shape, 4), np.float64 if float64_output else np.float32)
    check(
        _api().cn_matting_output(
            foreground.ctypes.data,
            alpha.ctypes.data,
            alpha.size,
            float64_output,
            result.ctypes.data,
        )
    )
    return result


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = operator.index(radius)
    if mask.dtype != np.uint8 or mask.ndim != 2 or not mask.size:
        raise ValueError("Chroma erosion requires a nonempty uint8 plane")
    if not 0 <= radius <= 46340:
        raise ValueError("Invalid chroma erosion radius")
    if not radius:
        return mask
    source = np.require(mask, requirements=["C", "A"])
    result = np.empty_like(source)
    check(
        _api().cn_matting_erode_mask_u8(
            source.ctypes.data,
            result.ctypes.data,
            *source.shape,
            radius,
        )
    )
    return result


def trimap_output(image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
    if (
        image.dtype != np.float32
        or image.ndim != 3
        or image.shape[2] != 3
        or not image.size
        or trimap.dtype != np.float64
        or trimap.shape != image.shape[:2]
    ):
        raise ValueError(
            "Chroma trimap output requires float32 BGR and matching float64 trimap"
        )
    source, pixel, channel = pixel_source(image)
    matte = np.require(trimap, requirements=["C", "A"])
    # Upstream's np.dstack((img, trimap.astype(np.float32))), the trimap
    # C-ordered: img's order K.
    h, w = matte.shape
    result = empty_ordered(
        (h, w, 4), numpy_order(image, ordered_layout((h, w, 1), (0, 1, 2)))
    )
    target, out_pixel, out_channel = pixel_target(result)
    check(
        _api().cn_matting_trimap_output(
            source.ctypes.data,
            pixel,
            channel,
            matte.ctypes.data,
            matte.size,
            target.ctypes.data,
            out_pixel,
            out_channel,
        )
    )
    if target is not result:
        np.copyto(result, target)
    return result
