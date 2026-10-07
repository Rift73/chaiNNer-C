"""Checked ownership boundary for fused C layout and spatial block kernels."""

from __future__ import annotations

import ctypes as ct
from collections.abc import Sequence

import numpy as np

from .color.color import Color
from .image_utils import to_uint8
from .native import (
    channel_planes,
    check,
    empty_ordered,
    f32,
    lib,
    numpy_order,
    ordered_layout,
    ptr,
)

_lib = lib()
_p = ct.POINTER(ct.c_float)
_n = ct.c_size_t
_lib.cn_layout_pad.argtypes = [_p, _p, _p, *([_n] * 8), ct.c_int]
_lib.cn_layout_paste.argtypes = [_p, _p, *([_n] * 10), ct.c_int]
_lib.cn_layout_zstack.argtypes = [ct.POINTER(_p), _p, _n, _n, ct.c_int]
_lib.cn_layout_pixelate.argtypes = [_p, _p, *([_n] * 5)]
_lib.cn_layout_rotate_u8.argtypes = [ct.POINTER(ct.c_uint8), _p, *([_n] * 4), ct.c_int]
for _name in ("pad", "paste", "zstack", "pixelate", "rotate_u8"):
    getattr(_lib, "cn_layout_" + _name).restype = ct.c_int


def _shape(image: object) -> tuple[int, int, int]:
    if not isinstance(image, np.ndarray) or image.dtype != np.float32:
        raise TypeError("C image kernels require a native-endian float32 NumPy array")
    if image.ndim not in (2, 3) or any(size == 0 for size in image.shape):
        raise ValueError("Expected a nonempty two- or three-dimensional image")
    return image.shape[0], image.shape[1], image.shape[2] if image.ndim == 3 else 1


def _paste(
    source: np.ndarray,
    out: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    expand_gray: bool,
) -> None:
    h, w, c = _shape(source)
    oh, ow, oc = _shape(out)
    if (
        min(top, left) < 0
        or min(height, width) < 1
        or height > oh - top
        or width > ow - left
    ):
        raise ValueError("Invalid image placement")
    check(
        _lib.cn_layout_paste(
            ptr(source),
            ptr(out),
            h,
            w,
            c,
            oh,
            ow,
            oc,
            top,
            left,
            height,
            width,
            int(expand_gray),
        )
    )


def pad(
    image: np.ndarray,
    border: int,
    color: Color | None,
    top: int,
    right: int,
    bottom: int,
    left: int,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    h, w, c = _shape(image)
    if min(top, right, bottom, left) < 0:
        raise ValueError("Padding must not be negative")
    if max(top, right, bottom, left) > 2**31 - 1:
        raise OverflowError("Padding exceeds the original OpenCV integer range")
    out_h, out_w = output_size or (h + top + bottom, w + left + right)
    if min(out_h, out_w) < 1:
        raise ValueError("Padding must produce a nonempty image")
    if top == right == bottom == left == 0 and (out_h, out_w) == (h, w):
        return image
    out_c, mode = c, border
    values = (0.0,) * c
    if top or right or bottom or left:
        if border == 0 and c == 4:
            values = (0.0, 0.0, 0.0, 1.0)
        elif border == 5:
            out_c, mode, values = 4, 0, (0.0,) * 4
        elif border == 6:
            mode, values = 0, (1.0,) * c
        elif border == 7:
            assert color is not None, (
                "Creating a border with a custom color requires supplying a custom color."
            )
            out_c, mode = max(c, color.channels), 0
            values = color.value
            if color.channels < out_c:
                # Upstream widens a float32 color image then reconstructs Color.
                values = tuple(float(np.float32(v)) for v in values)
                if len(values) == 1 and out_c in (3, 4):
                    values = values * 3
                if len(values) == 3 and out_c == 4:
                    values += (1.0,)
                values = tuple(max(0.0, min(v, 1.0)) for v in values)
            if len(values) != out_c:
                raise ValueError(
                    f"Unable to convert {color.channels} channel image to {out_c} channel image"
                )
        if out_c != c and not ((c == 1 and out_c in (3, 4)) or (c == 3 and out_c == 4)):
            raise ValueError(
                f"Unable to convert {c} channel image to {out_c} channel image"
            )
    else:
        # Offset mode may crop an image without ever creating a border.
        mode = 0
    if mode not in (0, 1, 3, 4):
        raise ValueError("Unsupported border mode")
    source = f32(image)
    fill = np.array(values, dtype=np.float32)
    keep_singleton = c == 1 and image.ndim == 3 and top == right == bottom == left == 0
    shape = (
        (out_h, out_w) if out_c == 1 and not keep_singleton else (out_h, out_w, out_c)
    )
    out = np.empty(shape, dtype=np.float32)
    check(
        _lib.cn_layout_pad(
            ptr(source),
            ptr(out),
            ptr(fill),
            h,
            w,
            c,
            out_h,
            out_w,
            out_c,
            top,
            left,
            mode,
        )
    )
    return out


def stack(
    images: Sequence[np.ndarray | None], orientation: str, *, sequence: bool = False
) -> np.ndarray:
    items = [image for image in images if image is not None]
    if not items:
        raise ValueError("No images in sequence to stack")
    shapes = [_shape(image) for image in items]
    if sequence and len(items) == 1:
        return items[0]
    if orientation not in ("horizontal", "vertical"):
        raise AssertionError(f"Invalid orientation '{orientation}'")
    max_h, max_w = max(s[0] for s in shapes), max(s[1] for s in shapes)
    max_c = max(3 if s[2] == 1 else s[2] for s in shapes)
    targets = []
    for h, w, c in shapes:
        th, tw = h, w
        if orientation == "horizontal" and h < max_h:
            th, tw = max_h, int(np.floor(w * max_h / h + 0.5))
        elif orientation == "vertical" and w < max_w:
            th, tw = int(np.floor(h * max_w / w + 0.5)), max_w
        expanded_c = 3 if c == 1 else c
        if expanded_c < max_c and not sequence:
            # The installed Stack Images allocates the original maxima while
            # promoting channels. Preserve its broadcast/failure behavior.
            if th not in (1, max_h) or tw not in (1, max_w):
                raise ValueError(
                    f"could not broadcast input array from shape {(th, tw, expanded_c)} into shape {(max_h, max_w, expanded_c)}"
                )
            th, tw = max_h, max_w
        targets.append((th, tw))
    out_h = max_h if orientation == "horizontal" else sum(t[0] for t in targets)
    out_w = sum(t[1] for t in targets) if orientation == "horizontal" else max_w
    out = np.empty((out_h, out_w, max_c), dtype=np.float32)
    top = left = 0
    for image, (th, tw) in zip(items, targets, strict=False):
        _paste(f32(image), out, top, left, th, tw, True)
        if orientation == "horizontal":
            left += tw
        else:
            top += th
    return out


def merge_spritesheet(
    images: Sequence[np.ndarray], rows: int, columns: int, row_major: bool
) -> np.ndarray:
    if rows < 1 or columns < 1:
        raise ValueError("Rows and columns must be positive")
    items = list(images[: rows * columns])
    if not items:
        raise ValueError("need at least one array to concatenate")
    shapes = [_shape(image) for image in items]
    c, ndim = shapes[0][2], items[0].ndim
    if any(
        shape[2] != c or image.ndim != ndim
        for shape, image in zip(shapes, items, strict=False)
    ):
        raise ValueError("All sprites must have matching dimensions and channels")
    groups, count = (rows, columns) if row_major else (columns, rows)
    placements = []
    # Upstream concatenates each row (column) of sprites, then the rows (columns),
    # each np.concatenate in its inputs' order K (numpy_order).
    group_layouts = []
    primary = 0
    secondary_size = None
    for group in range(groups):
        indices = range(group * count, min((group + 1) * count, len(items)))
        if not indices:
            raise ValueError("need at least one array to concatenate")
        reference = shapes[indices.start][0 if row_major else 1]
        secondary = 0
        for index in indices:
            h, w, _ = shapes[index]
            if (h if row_major else w) != reference:
                raise ValueError(
                    "All sprites in a row/column must have matching dimensions"
                )
            top, left = (primary, secondary) if row_major else (secondary, primary)
            placements.append((index, top, left, h, w))
            secondary += w if row_major else h
        if secondary_size is not None and secondary_size != secondary:
            raise ValueError(
                "Sprite rows/columns must concatenate to the same dimensions"
            )
        size = (reference, secondary) if row_major else (secondary, reference)
        group_layouts.append(
            ordered_layout((*size, c)[:ndim], numpy_order(*(items[i] for i in indices)))
        )
        secondary_size = secondary
        primary += reference
    assert secondary_size is not None
    out_h, out_w = (primary, secondary_size) if row_major else (secondary_size, primary)
    shape = (out_h, out_w) if ndim == 2 else (out_h, out_w, c)
    out = empty_ordered(shape, numpy_order(*group_layouts))
    # A planar sheet is pasted one plane at a time from planar sprites.
    planes = [channel_planes(items[index], out) for index, *_ in placements]
    planar = [pairs for pairs in planes if pairs is not None]
    if len(planar) == len(placements):
        for pairs, (_, top, left, h, w) in zip(planar, placements, strict=True):
            for source, target in pairs:
                _paste(source, target, top, left, h, w, False)
        return out
    target = out if out.flags.c_contiguous else np.empty(shape, np.float32)
    for index, top, left, h, w in placements:
        _paste(f32(items[index]), target, top, left, h, w, False)
    if target is not out:
        np.copyto(out, target)
    return out


def z_stack(images: Sequence[np.ndarray | None], mode: str) -> np.ndarray:
    items = [image for image in images if image is not None]
    assert 2 <= len(items) <= 15, (
        f"Number of images must be between 2 and 15 ({len(items)})"
    )
    shapes = [_shape(image) for image in items]
    assert all(shape == shapes[0] for shape in shapes), (
        "All images must have the same dimensions and channels"
    )
    if any(image.shape != items[0].shape for image in items):
        raise ValueError("Stacked images must have the same shape")
    modes = {"mean": 0, "median": 1, "minimum": 2, "maximum": 3}
    if mode not in modes:
        raise AssertionError(f"Invalid expression '{mode}'")
    arrays = [f32(image) for image in items]
    pointers = (_p * len(arrays))(*(ptr(image) for image in arrays))
    out = np.empty_like(arrays[0])
    check(_lib.cn_layout_zstack(pointers, ptr(out), out.size, len(arrays), modes[mode]))
    return out


def pixelate(image: np.ndarray, size_x: int, size_y: int) -> np.ndarray:
    h, w, c = _shape(image)
    if not 1 <= size_x <= 1024 or not 1 <= size_y <= 1024:
        raise ValueError("Pixel blocks must be between 1 and 1024 pixels")
    source = f32(image)
    out = np.empty((h, w) if c == 1 else (h, w, c), dtype=np.float32)
    check(_lib.cn_layout_pixelate(ptr(source), ptr(out), h, w, c, size_x, size_y))
    return out


def rotate_quarters(image: np.ndarray, quarter: int, transparent: bool) -> np.ndarray:
    h, w, c = _shape(image)
    if quarter not in (0, 1, 2, 3) or c not in (1, 2, 3, 4):
        raise ValueError("Unsupported right-angle rotation")
    if transparent:
        assert c in (1, 3, 4), f"Number of channels ({c}) unexpected"
    out_c = 4 if transparent else c
    source = np.require(to_uint8(image, normalized=True), requirements=["C", "A"])
    out_h, out_w = (w, h) if quarter % 2 else (h, w)
    shape = (out_h, out_w) if out_c == 1 else (out_h, out_w, out_c)
    out = np.empty(shape, dtype=np.float32)
    check(
        _lib.cn_layout_rotate_u8(
            source.ctypes.data_as(ct.POINTER(ct.c_uint8)),
            ptr(out),
            h,
            w,
            c,
            out_c,
            quarter,
        )
    )
    return out
