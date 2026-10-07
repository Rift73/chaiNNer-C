"""Checked boundary between NumPy-owned buffers and the C17 image kernels.

No Python implementation is selected on a load/ABI/kernel error: a broken native
installation must be visible, rather than silently running an unconverted path.
ctypes releases the GIL during calls. Callers retain all input/output arrays.
"""

from __future__ import annotations

import ctypes
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt

from . import native_profile

# cn_isa_get's levels by value; the requested level is -1 for auto.
ISA_LEVELS = ("scalar", "avx2", "avx512")


def library_path() -> Path:
    suffix = (
        ".dll"
        if sys.platform == "win32"
        else ".dylib"
        if sys.platform == "darwin"
        else ".so"
    )
    return Path(__file__).with_name("chainner_native" + suffix)


@lru_cache(maxsize=1)
def lib() -> ctypes.CDLL:
    path = library_path()
    try:
        library = (
            native_profile.ProfiledCDLL if native_profile.enabled() else ctypes.CDLL
        )(str(path))
    except OSError as error:
        # A library that refuses to initialize (WinError 1114) has written its reason
        # to stderr itself, e.g. a missing ucrtbase.dll export (cn_crt_math.h).
        raise RuntimeError(
            f"Cannot load chaiNNer C kernels at {path}: {error}. A line the library "
            "wrote to stderr while loading names the cause; without one, build "
            "native/CMakeLists.txt before starting this copy."
        ) from error
    library.cn_abi_version.argtypes = []
    library.cn_abi_version.restype = ctypes.c_int
    if library.cn_abi_version() != 2:
        raise RuntimeError(f"Incompatible chaiNNer C kernel ABI at {path}")
    try:
        isa_get = library.cn_isa_get
    except AttributeError as error:
        raise RuntimeError(
            f"Incompatible chaiNNer C kernel ABI at {path}: no cn_isa_get export"
        ) from error
    isa_get.argtypes = [ctypes.POINTER(ctypes.c_int)]
    isa_get.restype = ctypes.c_int
    if native_profile.enabled():
        # After the ABI checks: a DLL without the pool counters raises the ABI
        # error naming it.
        native_profile.attach_pool(str(path))
    return library


def isa_line() -> str:
    """The startup line: effective ISA level, the CHAINNER_C_ISA request, the CPU's."""
    state = (ctypes.c_int * 3)()
    check(lib().cn_isa_get(state))
    effective, requested, cpu = state
    request = "auto" if requested < 0 else ISA_LEVELS[requested]
    return f"native isa={ISA_LEVELS[effective]} (requested {request}, cpu {ISA_LEVELS[cpu]})"


def checked_f32(array: object) -> np.ndarray:
    """array itself, once it is a native-endian float32 ndarray."""
    if not isinstance(array, np.ndarray) or array.dtype != np.float32:
        raise TypeError("C image kernels require a native-endian float32 NumPy array")
    return array


def f32(array: object) -> np.ndarray:
    # ascontiguousarray alone can keep an unaligned array; the C ABI needs both.
    return np.require(checked_f32(array), dtype=np.float32, requirements=["C", "A"])


def ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def memory_order(image: np.ndarray) -> tuple[int, ...]:
    """image's axes from the largest stride to the smallest (stable on ties)."""
    return tuple(sorted(range(image.ndim), key=lambda axis: -image.strides[axis]))


Layout = tuple[tuple[int, ...], tuple[int, ...]]


def ordered_layout(shape: tuple[int, ...], order: tuple[int, ...]) -> Layout:
    """The shape and element strides of a new gapless array whose memory runs
    through its axes in order: (0, 1, 2) for a C-ordered one (np.full's, or cv2's
    result)."""
    strides = [0] * len(shape)
    size = 1
    for axis in reversed(order):
        strides[axis] = size
        size *= shape[axis]
    return shape, tuple(strides)


def numpy_order(*arrays: np.ndarray | Layout) -> tuple[int, ...]:
    """The memory order, slowest axis first, of the array NumPy computes from
    arrays (all of one rank, each an ndarray or a Layout) with order K:
    PyArray_CreateMultiSortedStridePerm, which np.concatenate uses and the ufunc
    iterator mirrors. A stable insertion sort by absolute stride; an array votes
    only on axes where it has more than one element, and where the arrays
    disagree C order wins."""
    layouts = [
        (array.shape, array.strides) if isinstance(array, np.ndarray) else array
        for array in arrays
    ]
    order = list(range(len(layouts[0][0])))
    for i0 in range(1, len(order)):
        moved, insert = order[i0], i0
        for i1 in range(i0 - 1, -1, -1):
            vote = None
            for shape, strides in layouts:
                if shape[moved] != 1 and shape[order[i1]] != 1:
                    swap = abs(strides[moved]) > abs(strides[order[i1]])
                    vote = swap if vote is None else vote and swap
            if vote is None:
                continue
            if not vote:
                break
            insert = i1
        order[insert + 1 : i0 + 1] = order[insert:i0]
        order[insert] = moved
    return tuple(order)


def empty_ordered(
    shape: tuple[int, ...], order: tuple[int, ...], dtype: npt.DTypeLike = np.float32
) -> np.ndarray:
    """A new gapless array of shape whose memory runs through its axes in order."""
    return np.empty([shape[axis] for axis in order], dtype).transpose(np.argsort(order))


def in_order(
    image: np.ndarray, dtype: npt.DTypeLike = np.float32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """np.clip's layout for an elementwise result of image: out is
    np.empty_like(image, dtype, order="K"). Returns image and out transposed into
    out's memory order, where both are C-contiguous (image is copied only if it
    has gaps), so a flat loop over them maps every element in place; then out."""
    out = np.empty_like(image, dtype=dtype, order="K", subok=False)
    order = memory_order(out)
    source = np.require(image.transpose(order), requirements=["C", "A"])
    return source, out.transpose(order), out


def pixel_strides(image: np.ndarray) -> tuple[int, int] | None:
    """The (pixel, channel) element strides of an aligned three-dimensional image
    whose pixels form a row-major grid, element (y, x, c) at (y * width + x) *
    pixel + c * channel: interleaved (channels, 1), planar (1, plane), or a channel
    slice of either. None for any other layout. An axis of length 1 takes the
    stride that fits."""
    if image.ndim != 3 or not image.flags.aligned:
        return None
    height, width, channels = image.shape
    row, pixel, channel = image.strides
    if width == 1:
        pixel = row
    if height == 1:
        row = width * pixel
    if channels == 1:
        channel = image.itemsize
    strides = (row, pixel, channel)
    if any(stride <= 0 or stride % image.itemsize for stride in strides):
        return None
    if row != width * pixel:
        return None
    return pixel // image.itemsize, channel // image.itemsize


def pixel_source(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    """A three-dimensional image as a pixel-grid kernel reads it, with its
    (pixel, channel) element strides: in place where pixel_strides allows,
    else a C-ordered copy."""
    strides = pixel_strides(image)
    if strides is None:
        image = np.require(image, requirements=["C", "A"])
        return image, image.shape[2], 1
    return image, *strides


def pixel_target(out: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Where a pixel-grid kernel writes out, with its (pixel, channel) element
    strides: out itself where pixel_strides allows, else a new C-ordered array
    that the caller copies into out."""
    strides = pixel_strides(out)
    if strides is None:
        target = np.empty(out.shape, out.dtype)
        return target, out.shape[2], 1
    return out, *strides


def channel_planes(
    image: np.ndarray, out: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """The channel planes of image and of out, when out is planar and image's
    channel planes are contiguous too (pixel_strides, pixel stride 1): each plane
    is a one-channel image that an interleaved kernel reads or writes in place.
    Else None."""
    if image.ndim != 3 or out.ndim != 3 or out.flags.c_contiguous:
        return None
    source, target = pixel_strides(image), pixel_strides(out)
    if source is None or target is None or source[0] != 1 or target[0] != 1:
        return None
    return [(image[:, :, c], out[:, :, c]) for c in range(image.shape[2])]


def check(status: int) -> None:
    if status == 1:
        raise ValueError("Invalid image dimensions or arguments passed to C kernels")
    if status == 2:
        raise OverflowError("Image size exceeds the C kernel addressable buffer range")
    if status == 3:
        raise MemoryError("Unable to allocate the native CPU worker pool or work item")
    if status != 0:
        raise RuntimeError(f"Unexpected C kernel status {status}")
