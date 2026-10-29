"""C descale: vapoursynth-descale's solver over the native pool (native/src/descale_ops.c)."""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import numpy as np

from .linear_light import LightTransfer, encode_inplace, linearize
from .native import check, f32, lib, ptr

_lib = lib()
_lib.cn_descale_core_create.argtypes = [
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_int,
    ct.POINTER(ct.c_int),
]
_lib.cn_descale_core_create.restype = ct.c_void_p
_lib.cn_descale_core_free.argtypes = [ct.c_void_p]
_lib.cn_descale_core_free.restype = None
_lib.cn_descale_workspace.argtypes = [
    ct.c_size_t,
    ct.c_size_t,
    ct.c_size_t,
    ct.c_void_p,
    ct.c_void_p,
]
_lib.cn_descale_workspace.restype = ct.c_size_t
_lib.cn_descale.argtypes = [
    ct.POINTER(ct.c_float),
    ct.c_size_t,
    ct.c_size_t,
    ct.c_size_t,
    ct.c_void_p,
    ct.c_void_p,
    ct.POINTER(ct.c_float),
    ct.c_void_p,
    ct.c_size_t,
]
_lib.cn_descale.restype = ct.c_int


# DescaleMode's names, 1 bilinear .. 7 point.
_KERNELS = (
    "Bilinear",
    "Bicubic",
    "Lanczos",
    "Spline16",
    "Spline36",
    "Spline64",
    "Point",
)


class _Core:
    """One direction's factorized system, freed when the cache and every caller drop it."""

    def __init__(self, handle: int):
        self.handle = handle

    def __del__(self):
        _lib.cn_descale_core_free(self.handle)


@lru_cache(maxsize=16)
def _core(
    src_dim: int,
    dst_dim: int,
    mode: int,
    taps: int,
    b: float,
    c: float,
    blur: float,
    shift: float,
    active_dim: float,
    border: int,
) -> _Core:
    # Building the system costs O(dim^2) (tens of ms for 4K), so a batch reuses it.
    refusal = ct.c_int(0)
    handle = _lib.cn_descale_core_create(
        src_dim,
        dst_dim,
        mode,
        taps,
        b,
        c,
        blur,
        shift,
        active_dim,
        border,
        ct.byref(refusal),
    )
    if not handle:
        if refusal.value == 1:
            raise ValueError(
                f"Descale: Blur {blur:g} is too narrow for the {_KERNELS[mode - 1]} kernel"
                " at this size: some target pixels get no weight (the result would be"
                " black). Use a larger Blur (1 is the kernel's own width)."
            )
        if refusal.value == 2:
            raise ValueError(
                "Descale: with this source window (Source Left, Top, Width, Height) some"
                f" target pixels get no weight from the {_KERNELS[mode - 1]} kernel (the"
                " result would be black). Use a window that covers the target."
            )
        if refusal.value == 3:
            raise ValueError(
                f"Descale: the {_KERNELS[mode - 1]} kernel's weights are not finite with"
                " these parameters (the result would not be finite either)."
            )
        raise ValueError("Invalid descale parameters")
    return _Core(handle)


def descale(
    image: np.ndarray,
    width: int,
    height: int,
    mode: int,
    taps: int,
    b: float,
    c: float,
    blur: float,
    shift: tuple[float, float],
    active: tuple[float, float],
    border: int,
    light: LightTransfer | None = None,
) -> np.ndarray:
    """image (H x W or H x W x C float32) descaled to width x height. mode and border take
    descale's DescaleMode and DescaleBorder values; shift and active are (left, top) and
    (width, height) of the source window in target pixels. A direction whose size, shift
    and window are unchanged is skipped. light descales in linear light with that curve:
    the colour channels (alpha, a fourth channel, as is) linearized before the solve and
    encoded after, never clipped."""
    source = f32(image)
    h, w = source.shape[:2]
    channels = 1 if source.ndim == 2 else source.shape[2]
    if not 1 <= width <= w or not 1 <= height <= h:
        raise ValueError(
            f"Descale needs a target no larger than the image ({w}x{h}), got {width}x{height}"
        )
    core_h = (
        _core(w, width, mode, taps, b, c, blur, shift[0], active[0], border)
        if width != w or shift[0] != 0 or active[0] != width
        else None
    )
    core_v = (
        _core(h, height, mode, taps, b, c, blur, shift[1], active[1], border)
        if height != h or shift[1] != 0 or active[1] != height
        else None
    )
    handles = (core_h.handle if core_h else None, core_v.handle if core_v else None)
    size = _lib.cn_descale_workspace(h, w, channels, *handles)
    if not size:
        raise ValueError("Invalid descale image or parameters")
    # From NumPy, whose pool keeps it warm across a batch: fresh pages cost more than the solve.
    workspace = np.empty(size, np.uint8)
    result = np.empty((height, width, channels), np.float32)
    if light is not None:
        source = linearize(source, light)
    check(
        _lib.cn_descale(
            ptr(source),
            h,
            w,
            channels,
            *handles,
            ptr(result),
            workspace.ctypes.data,
            size,
        )
    )
    if light is not None:
        encode_inplace(result, light)
    return result if source.ndim == 3 else result[:, :, 0]
