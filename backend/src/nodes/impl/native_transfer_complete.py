"""C color-transfer preparation with the installed public OpenBLAS ABI.

BLAS/LAPACK remain an external native dependency. Lab colorspace conversion
remains in OpenCV; neither is described as a handwritten C replacement.
"""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache
from pathlib import Path

import numpy as np

from .native import (
    check,
    empty_ordered,
    f32,
    lib,
    numpy_order,
    pixel_source,
    pixel_target,
    ptr,
)
from .native_analysis import channel_stats


class _Backend(ct.Structure):
    library: ct.CDLL | None = None
    _fields_ = [
        (name, ct.c_void_p) for name in ("zgemm", "syrk", "geev", "zgetrf", "zgesv")
    ]


@lru_cache(maxsize=1)
def _backend():
    libs = Path(np.__file__).resolve().parent.parent / "numpy.libs"
    paths = list(libs.glob("libscipy_openblas64_*.dll"))
    if len(paths) != 1:
        raise RuntimeError(
            "Color Transfer requires the installed NumPy ILP64 OpenBLAS library"
        )
    library = ct.CDLL(str(paths[0]))
    # NumPy 2 np.linalg.eig always returns complex128, so after the real
    # covariance (dsyrk) and eigensystem (dgeev) every product, determinant and
    # inverse is complex, as in NumPy.
    symbols = (
        "scipy_cblas_zgemm64_",
        "scipy_cblas_dsyrk64_",
        "scipy_dgeev_64_",
        "scipy_zgetrf_64_",
        "scipy_zgesv_64_",
    )
    try:
        addresses = [
            ct.cast(getattr(library, name), ct.c_void_p).value for name in symbols
        ]
    except AttributeError as error:
        raise RuntimeError(
            "NumPy OpenBLAS lacks the required public ILP64 exports"
        ) from error
    # Tie the owner to each table as well as the cache: simultaneous first
    # calls may legitimately construct more than one cached-function result.
    table = _Backend(*addresses)
    table.library = library
    return library, table


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    p, z, i, fp = ct.c_void_p, ct.c_size_t, ct.c_int, ct.POINTER(ct.c_float)
    signatures = {
        "cn_transfer_parameters": [p, p, p, p, z, z, z, z, z, z, i, i, p, p, p, p, p],
        "cn_transfer_gather_f32": [fp, p, z, fp, ct.POINTER(z)],
        "cn_transfer_mask": [p, z, z, z, z, i, p],
        "cn_transfer_alpha": [p, z, z, p, z, z, z, i, i, p, z, z],
        "cn_transfer_apply_complex": [p, z, z, z, i, p, p, p, p, z, z, i, p],
        "cn_analysis_transfer": [fp, fp, p, z, fp, fp, fp, i, i],
    }
    for name, arguments in signatures.items():
        fn = getattr(dll, name)
        fn.argtypes, fn.restype = arguments, i
    return dll


def supported(
    image: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    reference_mask: np.ndarray,
) -> bool:
    return (
        image.dtype in (np.dtype(np.float32), np.dtype(np.float64))
        and reference.dtype == image.dtype
        and image.ndim == reference.ndim == 3
        and image.shape[2] == reference.shape[2] == 3
        and image.size > 0
        and reference.size > 0
        and mask.dtype == reference_mask.dtype == np.bool_
        and mask.shape == image.shape[:2]
        and reference_mask.shape == reference.shape[:2]
    )


def parameters(
    image: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    reference_mask: np.ndarray,
    principal: bool,
):
    if not supported(image, reference, mask, reference_mask):
        raise ValueError(
            "Color Transfer requires matching native float images and boolean masks"
        )
    # Both images are read in place, interleaved or planar (pixel_source).
    source, pixel, channel = pixel_source(image)
    ref, ref_pixel, ref_channel = pixel_source(reference)
    valid, ref_valid = (
        np.require(a, requirements=["C", "A"]) for a in (mask, reference_mask)
    )
    means = np.empty(3, source.dtype)
    ref_means = np.empty(3, source.dtype)
    matrix = np.empty((3, 3), np.complex128)
    error = ct.c_int()
    check(
        _api().cn_transfer_parameters(
            source.ctypes.data,
            ref.ctypes.data,
            valid.ctypes.data,
            ref_valid.ctypes.data,
            valid.size,
            ref_valid.size,
            pixel,
            channel,
            ref_pixel,
            ref_channel,
            int(source.dtype == np.float64),
            principal,
            means.ctypes.data,
            ref_means.ctypes.data,
            matrix.ctypes.data,
            ct.byref(error),
            ct.byref(_backend()[1]),
        )
    )
    if error.value:
        messages = {
            1: "Array must not contain infs or NaNs",
            3: "Eigenvalues did not converge",
            4: "Singular matrix",
        }
        raise np.linalg.LinAlgError(messages[error.value])
    return source, means, ref_means, matrix


def transfer(
    image: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    reference_mask: np.ndarray,
    principal: bool,
) -> np.ndarray:
    source, means, ref_means, matrix = parameters(
        image, reference, mask, reference_mask, principal
    )
    source, pixel, channel = pixel_source(source)
    # Upstream's Linear result, transfer.dot((content - mu).T).T reshaped, is
    # planar whatever the image's layout; Principal's (content - mu).dot(...) is
    # C-ordered.
    result = empty_ordered(
        source.shape, (0, 1, 2) if principal else (2, 0, 1), np.complex128
    )
    out_pixel, out_channel = (3, 1) if principal else (1, source.size // 3)
    check(
        _api().cn_transfer_apply_complex(
            source.ctypes.data,
            pixel,
            channel,
            source.size // 3,
            int(source.dtype == np.float64),
            means.ctypes.data,
            ref_means.ctypes.data,
            matrix.ctypes.data,
            result.ctypes.data,
            out_pixel,
            out_channel,
            principal,
            ct.byref(_backend()[1]),
        )
    )
    return result


def mean_std_apply(
    image: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    reference_mask: np.ndarray,
    limits: tuple[float, ...],
    *,
    reciprocal: bool,
    scale: bool,
) -> np.ndarray:
    if (
        not supported(image, reference, mask, reference_mask)
        or image.dtype != np.float32
    ):
        raise ValueError(
            "Mean/Std requires matching native float32 images and boolean masks"
        )
    values = []
    for data, original_mask in ((image, mask), (reference, reference_mask)):
        source = f32(data)
        valid = np.require(original_mask, requirements=["C", "A"])
        gathered = np.empty((valid.size, 3), np.float32)
        count = ct.c_size_t()
        check(
            _api().cn_transfer_gather_f32(
                ptr(source),
                valid.ctypes.data,
                valid.size,
                ptr(gathered),
                ct.byref(count),
            )
        )
        values.append(channel_stats(gathered[: count.value]))
    source = f32(image)
    valid = np.require(mask, requirements=["C", "A"])
    bounds = np.require(limits, dtype=np.float32, requirements=["C", "A"])
    if bounds.shape != (6,):
        raise ValueError("Mean/Std requires six channel limits")
    result = np.empty_like(source)
    check(
        _api().cn_analysis_transfer(
            ptr(source),
            ptr(result),
            valid.ctypes.data,
            valid.size,
            ptr(values[0]),
            ptr(values[1]),
            ptr(bounds),
            reciprocal,
            scale,
        )
    )
    return result


def transfer_mask(image: np.ndarray) -> np.ndarray:
    if (
        image.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
        or image.ndim != 3
        or image.shape[2] not in (3, 4)
    ):
        raise ValueError(
            "Color Transfer mask requires a native float BGR or BGRA image"
        )
    source, pixel, channel = pixel_source(image)
    mask = np.empty(source.shape[:2], np.bool_)
    check(
        _api().cn_transfer_mask(
            source.ctypes.data,
            mask.size,
            source.shape[2],
            pixel,
            channel,
            int(source.dtype == np.float64),
            mask.ctypes.data,
        )
    )
    return mask


def transfer_with_alpha(image: np.ndarray, original: np.ndarray) -> np.ndarray:
    if (
        original.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
        or original.ndim != 3
        or original.shape[2] != 4
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[:2] != original.shape[:2]
        or image.dtype
        not in (np.dtype(np.float32), np.dtype(np.float64), np.dtype(np.complex128))
    ):
        raise ValueError(
            "Color Transfer alpha requires matching native float image buffers"
        )
    dtype = np.result_type(image.dtype, original.dtype)
    # Upstream's np.dstack((transfer, alpha)) keeps the transfer's order K.
    h, w, _ = image.shape
    result = empty_ordered((h, w, 4), numpy_order(image, original[:, :, 3:]), dtype)
    if image.dtype != dtype:
        image = image.astype(dtype, order="K")
    source, pixel, channel = pixel_source(image)
    original, original_pixel, original_channel = pixel_source(original)
    target, out_pixel, out_channel = pixel_target(result)
    kind = 2 if dtype == np.complex128 else int(dtype == np.float64)
    check(
        _api().cn_transfer_alpha(
            source.ctypes.data,
            pixel,
            channel,
            original.ctypes.data,
            original_pixel,
            original_channel,
            h * w,
            kind,
            int(original.dtype == np.float64),
            target.ctypes.data,
            out_pixel,
            out_channel,
        )
    )
    if target is not result:
        np.copyto(result, target)
    return result
