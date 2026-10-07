"""C pixel conversion and upscaler CPU buffers, independent of model runtimes."""

from __future__ import annotations

import ctypes as ct
import sys
import warnings
import weakref
from collections import OrderedDict, deque
from collections.abc import Callable
from functools import lru_cache
from threading import Lock
from typing import cast

import numpy as np

from . import native_profile
from .native import check, f32, in_order, lib, memory_order, ptr

_TYPES = {
    np.dtype(name): i
    for i, name in enumerate(
        (
            "float32",
            "float64",
            "uint8",
            "uint16",
            "int8",
            "int16",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "bool",
        )
    )
}


# lru_cache does not serialize first calls. Two at once would each set argtypes on the
# function objects their getattr misses create, and one thread's object can replace the
# other's on the library (SP3b audit A8), so the body runs one thread at a time.
_api_lock = Lock()


@lru_cache(maxsize=1)
def _api():
    with _api_lock:
        dll = lib()
        fp, size, integer, void = (
            ct.POINTER(ct.c_float),
            ct.c_size_t,
            ct.c_int,
            ct.c_void_p,
        )
        for name, signature in {
            "cn_pixels_convert": [void, void, size, integer, integer, integer],
            "cn_pixels_convert_checked": [
                void,
                void,
                size,
                integer,
                integer,
                integer,
                ct.POINTER(integer),
            ],
            "cn_pixels_normalized_f32": [fp, size, ct.POINTER(integer)],
            "cn_tile_mix": [fp, fp, fp, fp, size, size, size, integer, size, size],
            "cn_upscale_alpha": [fp, fp, fp, size, size, integer],
            "cn_constant_alpha": [fp, size, ct.POINTER(integer)],
            "cn_caption_compose": [fp, void, fp, size, size, size, size, integer],
            "cn_text_raster": [void, void, fp, size],
        }.items():
            function = getattr(dll, name)
            function.argtypes, function.restype = signature, integer
        return dll


def _floating_error(kind: str, operation: str, flag: int) -> None:
    mode = np.geterr()[kind]
    if mode == "ignore":
        return
    error = (
        "invalid value" if kind == "invalid" else "overflow" if kind == "over" else kind
    )
    message = f"{error} encountered in {operation}"
    if mode == "warn":
        # The caller of converted_pixels or freeze_normalized (via _report_events).
        warnings.warn(message, RuntimeWarning, stacklevel=4)
    elif mode == "raise":
        raise FloatingPointError(message)
    elif mode == "print":
        print("Warning: " + message, file=sys.stderr)
    else:
        handler = np.geterrcall()
        if handler is None:
            raise NameError(
                f"python callback specified for {error} (in {operation}) but no function found."
            )
        if mode == "call":
            cast(Callable[[str, int], object], handler)(error, flag)
        else:
            handler.write("Warning: " + message + "\n")


def _report_events(events: int) -> None:
    """cn_pixels_convert_checked's event bits, reported as NumPy reports its errors."""
    multiply_flag = (2 if events & 1 else 0) | (8 if events & 4 else 0)
    if events & 1:
        _floating_error("over", "multiply", multiply_flag)
    if events & 4:
        _floating_error("invalid", "multiply", multiply_flag)
    if events & 2:
        _floating_error("invalid", "cast", 8)


def _gapless(image: np.ndarray) -> np.ndarray | None:
    """image's axes reordered so that its memory is C-contiguous, when image is
    gapless: a permutation of a C-contiguous array, as np.empty_like(order="K")
    allocates for np.clip's output (positive strides, no gaps). Else None."""
    ordered = image.transpose(memory_order(image))
    return ordered if ordered.flags.c_contiguous else None


def converted_pixels(
    image: np.ndarray, bits: int = 0, normalized: bool = False
) -> np.ndarray | None:
    """Return None only for explicitly retained dtype/scalar conversion semantics.

    The result has np.clip's layout: np.empty_like(image, order="K"), so a node's
    permuted output (Lens Blur's transposed plane) keeps its strides, as upstream's
    ImageOutput passes them on. The conversion runs flat over both arrays in that
    order; the source is copied first only if it has gaps.
    """
    if bits not in (0, 8, 16):
        raise ValueError("Expected float32 normalization, uint8 or uint16 output")
    if image.dtype not in _TYPES or image.ndim == 0:
        return None
    source, target, out = in_order(
        image, {0: np.float32, 8: np.uint8, 16: np.uint16}[bits]
    )
    events = ct.c_int()
    check(
        _api().cn_pixels_convert_checked(
            source.ctypes.data,
            target.ctypes.data,
            source.size,
            _TYPES[source.dtype],
            {0: 0, 8: 1, 16: 2}[bits],
            int(not normalized or bits == 0),
            ct.byref(events),
        )
    )
    _report_events(events.value)
    return out


# CHAINNER_C_PROFILE: the bridge's own time (np.require, the output allocation,
# the event checks) is converted_pixels' exclusive time; the conversion it calls
# is timed as cn_pixels_convert_checked and subtracted. Off, nothing is wrapped.
if native_profile.enabled():
    converted_pixels = native_profile.timed("converted_pixels", converted_pixels)


_NORMALIZED_OWNER_LIMIT = 4096
_normalized_owners: OrderedDict[int, weakref.ReferenceType[np.ndarray]] = OrderedDict()
_normalized_owner_lock = Lock()
# A collection triggered inside a locked section can run a weakref callback on the
# same thread, so callbacks only queue their entry; locked sections remove it.
_normalized_expired: deque[tuple[int, weakref.ReferenceType[np.ndarray]]] = deque()


def _drain_normalized_expired() -> None:
    # Requires _normalized_owner_lock. A newer registration under a reused id stays.
    while _normalized_expired:
        key, reference = _normalized_expired.popleft()
        if _normalized_owners.get(key) is reference:
            del _normalized_owners[key]


def _not_float32_owner(image: np.ndarray) -> bool:
    """Anything but a base ndarray that owns aligned gapless float32 data (C order,
    or the order-K permutation converted_pixels allocates)."""
    return (
        type(image) is not np.ndarray
        or image.base is not None
        or not image.flags.owndata
        or image.dtype != np.float32
        or not image.flags.aligned
        or _gapless(image) is None
    )


def register_normalized_copy(image: np.ndarray) -> None:
    """Record only ImageOutput's newly allocated, frozen normalization result,
    or a bridge's fresh array frozen before return (freeze_normalized).

    A read-only flag alone does not prove ownership: older writable aliases may
    still exist. This registry records the boundary's own isolated copies. It
    never owns image memory; eviction or collection merely disables borrowing.
    Callers must continue to honor the enforced output's read-only contract.
    """
    if _not_float32_owner(image) or image.flags.writeable:
        return
    key = id(image)

    def expired(reference: weakref.ReferenceType[np.ndarray]) -> None:
        _normalized_expired.append((key, reference))

    reference = weakref.ref(image, expired)
    with _normalized_owner_lock:
        _drain_normalized_expired()
        _normalized_owners[key] = reference
        _normalized_owners.move_to_end(key)
        while len(_normalized_owners) > _NORMALIZED_OWNER_LIMIT:
            _normalized_owners.popitem(last=False)


def freeze_normalized(out: np.ndarray, *, clamp: bool = False) -> np.ndarray:
    """Freeze and register a node's or bridge's fresh output, so ImageOutput.enforce
    may borrow it instead of converting a copy; returns out.

    out must be the array the caller just allocated and filled: a writable float32
    ndarray that owns aligned gapless data (C order or a permutation of it), with no
    view of it kept anywhere (ValueError for any other array, which is left as it
    was). The caller may return a permuted view of it, as Lens Blur returns its
    plane's transpose(1, 2, 0); enforce borrows that view.

    clamp: first convert out in place with the output enforce's own conversion,
    cn_pixels_convert_checked(out, out, out.size, 0, 0, 1) (normalize's float32
    clip, which converted_pixels runs out of place), and report its events as
    converted_pixels does. Only for a node that returns out straight to its
    ImageOutput (Gaussian and Box Blur).

    Bits: freezing changes no value. enforce borrows out only when
    cn_pixels_normalized_f32 proves that its clip leaves every element's bits as
    they are (+0 and normal values in (0, 1]); otherwise it converts out exactly as
    it converts any unregistered array. With clamp the user still receives enforce's
    bits for the node's result: the executor runs the node and then its output
    enforce in one run_node call on one thread (process.py:115-124), so the clamp is
    enforce's own function under the MXCSR enforce runs under, directly before it;
    and the clip is idempotent, so a refused borrow re-clamps to the same bits.
    """
    if _not_float32_owner(out) or not out.flags.writeable:
        raise ValueError(
            "freeze_normalized needs a fresh writable float32 array that owns "
            "aligned gapless data"
        )
    if clamp:
        events = ct.c_int()
        check(
            _api().cn_pixels_convert_checked(
                out.ctypes.data, out.ctypes.data, out.size, 0, 0, 1, ct.byref(events)
            )
        )
        _report_events(events.value)
    out.setflags(write=False)
    register_normalized_copy(out)
    return out


# CHAINNER_C_PROFILE: freeze_normalized's own time (its checks, the flag, the
# registration and with clamp the event checks) is its exclusive time, like
# converted_pixels'; its in-place conversion is timed as cn_pixels_convert_checked.
# With the enforce call's cn_pixels_normalized_f32 it is a borrowed output's
# enforce path.
if native_profile.enabled():
    freeze_normalized = native_profile.timed("freeze_normalized", freeze_normalized)


def _known_normalized_owner(owner: object) -> bool:
    while True:
        if isinstance(owner, np.ndarray):
            if type(owner) is not np.ndarray or owner.flags.writeable:
                return False
            if owner.base is None:
                with _normalized_owner_lock:
                    _drain_normalized_expired()
                    reference = _normalized_owners.get(id(owner))
                    if reference is None or reference() is not owner:
                        return False
                return True
            owner = owner.base
        elif isinstance(owner, memoryview):
            if not owner.readonly:
                return False
            owner = owner.obj
        elif type(owner) is bytes:
            return True
        else:
            return False


def normalized_readonly(image: np.ndarray) -> np.ndarray | None:
    """Borrow known enforced copies or immutable bytes; keep unknown aliases isolated.

    A gapless permuted view of a known owner (Lens Blur's transposed plane) is
    borrowed as it is, keeping the strides np.clip's order K would give it; the
    scan reads its memory through the C-ordered transpose.
    """
    if (
        type(image) is not np.ndarray
        or image.dtype != np.float32
        or not image.flags.aligned
    ):
        return None
    ordered = _gapless(image)
    if ordered is None or not _known_normalized_owner(image):
        return None
    same = ct.c_int()
    check(_api().cn_pixels_normalized_f32(ptr(ordered), image.size, ct.byref(same)))
    return image if same.value else None


def tile_mix(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if a.ndim != 3 or a.shape != b.shape or weights.shape != a.shape or 0 in a.shape:
        raise ValueError(
            "Tile images and blend weights must have matching nonempty shapes"
        )
    if weights.dtype != np.float32:
        raise TypeError("Tile weights must be float32")

    def row_buffer(image: np.ndarray) -> np.ndarray:
        # Typical overlap views have contiguous pixels but a larger row stride.
        # Read them directly instead of copying both overlap images before mixing.
        if (
            image.dtype == np.float32
            and image.flags.aligned
            and image.strides[2] == 4
            and image.strides[1] == image.shape[2] * 4
            and image.strides[0] >= image.shape[1] * image.shape[2] * 4
            and image.strides[0] % 4 == 0
        ):
            return image
        return f32(image)

    a, b = row_buffer(a), row_buffer(b)
    if weights.strides[0] == weights.strides[2] == 0:
        values, axis = f32(weights[0, :, 0]), 1
    elif weights.strides[1] == weights.strides[2] == 0:
        values, axis = f32(weights[:, 0, 0]), 2
    else:
        values, axis = f32(weights), 0
    out = np.empty(a.shape, dtype=np.float32)
    # NumPy permits zero/negative strides on singleton axes even when C-contiguous.
    # The only row is dense after row_buffer, and its unused stride is canonical.
    row_width = a.shape[1] * a.shape[2]
    check(
        _api().cn_tile_mix(
            ptr(a),
            ptr(b),
            ptr(values),
            ptr(out),
            *a.shape,
            axis,
            row_width if a.shape[0] == 1 else a.strides[0] // 4,
            row_width if b.shape[0] == 1 else b.strides[0] // 4,
        )
    )
    return out


def alpha_backgrounds(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = f32(image)
    if source.ndim != 3 or source.shape[2] != 4:
        raise ValueError("Expected RGBA input")
    black, white = (np.empty((*source.shape[:2], 3), np.float32) for _ in range(2))
    check(
        _api().cn_upscale_alpha(
            ptr(source), ptr(black), ptr(white), black.size // 3, 4, 1
        )
    )
    return black, white


def flatten_alpha(image: np.ndarray) -> np.ndarray:
    source = f32(image)
    if source.ndim != 3 or not 1 <= source.shape[2] <= 7:
        raise ValueError("Expected one to seven channels")
    out = np.empty(source.shape[:2], np.float32)
    check(
        _api().cn_upscale_alpha(
            ptr(source), ptr(out), None, out.size, source.shape[2], 0
        )
    )
    return out


def constant_alpha(image: np.ndarray) -> tuple[bool, float]:
    if image.dtype != np.float32:
        values = np.unique(image[:, :, 3])
        return len(values) == 1, float(values[0]) if values.size else 0
    source = f32(image)
    if source.ndim != 3 or source.shape[2] != 4 or not source.size:
        raise ValueError("Expected nonempty RGBA input")
    constant = ct.c_int()
    check(_api().cn_constant_alpha(ptr(source), source.size // 4, ct.byref(constant)))
    return bool(constant.value), float(source[0, 0, 3])


def compose_caption(image: np.ndarray, caption: np.ndarray, top: bool) -> np.ndarray:
    source = f32(image)
    if source.ndim not in (2, 3) or not all(source.shape):
        raise ValueError("Caption requires a nonempty image")
    channels = source.shape[2] if source.ndim == 3 else 1
    if channels not in (1, 3, 4) or (source.ndim == 3 and channels == 1):
        raise ValueError("Caption expects two-dimensional grayscale or RGB/RGBA")
    if caption.dtype != np.uint8:
        raise TypeError("Caption raster must be uint8")
    if (
        caption.ndim != 2
        or not all(caption.shape)
        or caption.shape[1] != source.shape[1]
    ):
        raise ValueError("Caption raster must have the same width as the image")
    raster = np.require(caption, requirements=["C", "A"])
    out = np.empty((source.shape[0] + raster.shape[0], *source.shape[1:]), np.float32)
    check(
        _api().cn_caption_compose(
            ptr(source),
            raster.ctypes.data,
            ptr(out),
            *source.shape[:2],
            channels,
            raster.shape[0],
            int(top),
        )
    )
    return out


def colorize_text(mask: np.ndarray, ink: tuple[int, ...]) -> np.ndarray:
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("Text mask must be a uint8 plane")
    color = np.require(ink, dtype=np.uint8, requirements=["C", "A"])
    if color.shape != (3,):
        raise ValueError("Text color must have three uint8 components")
    raster = np.require(mask, requirements=["C", "A"])
    out = np.empty((*mask.shape, 4), np.float32)
    check(
        _api().cn_text_raster(
            raster.ctypes.data, color.ctypes.data, ptr(out), raster.size
        )
    )
    return out
