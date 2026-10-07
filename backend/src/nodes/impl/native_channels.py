"""Validated, NumPy-owned buffers for native channel assembly and image layout."""

from __future__ import annotations

import ctypes as ct

import numpy as np

from .color.color import Color
from .native import (
    Layout,
    channel_planes,
    check,
    empty_ordered,
    f32,
    lib,
    numpy_order,
    ordered_layout,
    pixel_source,
    pixel_strides,
    ptr,
)
from .native_versions import CV_CN_MAX


class _Channel(ct.Structure):
    _fields_ = [
        ("data", ct.POINTER(ct.c_float)),
        ("channels", ct.c_size_t),
        ("channel", ct.c_size_t),
        ("constant", ct.c_float),
    ]


_lib = lib()
_float_ptr = ct.POINTER(ct.c_float)
_size = ct.c_size_t
_lib.cn_channels_assemble.argtypes = [ct.POINTER(_Channel), _float_ptr, _size, _size]
_lib.cn_channels_flip.argtypes = [_float_ptr, _float_ptr, _size, _size, _size, ct.c_int]
_lib.cn_channels_shift.argtypes = [
    _float_ptr,
    _float_ptr,
    _size,
    _size,
    _size,
    ct.c_int64,
    ct.c_int64,
    ct.c_int,
]
_lib.cn_channels_bbox.argtypes = [
    _float_ptr,
    _size,
    _size,
    _size,
    _size,
    ct.c_float,
    ct.POINTER(_size),
]
_lib.cn_channels_crop.argtypes = [_float_ptr, _float_ptr, *([_size] * 8)]
for _name in ("assemble", "flip", "shift", "bbox", "crop"):
    getattr(_lib, "cn_channels_" + _name).restype = ct.c_int


def _shape(image: object) -> tuple[int, int, int]:
    if not isinstance(image, np.ndarray) or image.dtype != np.float32:
        raise TypeError("C image kernels require a native-endian float32 NumPy array")
    if image.ndim not in (2, 3) or any(size == 0 for size in image.shape):
        raise ValueError("Expected a nonempty two- or three-dimensional image")
    return image.shape[0], image.shape[1], 1 if image.ndim == 2 else image.shape[2]


def _three_dimensional(value: np.ndarray) -> np.ndarray:
    """A checked image (_shape) with a channel axis, np.atleast_3d's view."""
    _shape(value)
    return value if value.ndim == 3 else value[:, :, np.newaxis]


def _channel(value: np.ndarray, pixel: int, plane: int, channel: int) -> _Channel:
    """A descriptor of a pixel_source array's channel, element p (a row-major pixel) at
    data[p * channels + channel]."""
    if plane * channel < pixel:
        return _Channel(ptr(value), pixel, plane * channel, 0)
    address = value.ctypes.data + plane * channel * value.itemsize
    return _Channel(ct.cast(address, _float_ptr), pixel, 0, 0)


def _layout(
    value: np.ndarray | Color, height: int, width: int, rgb: bool = False
) -> np.ndarray | Layout:
    """value as np.concatenate sees it (numpy_order): three-dimensional; for a
    Color, the C-ordered image Color.to_image makes. rgb: as_target_channels(value,
    3), which converts any other channel count with cv2 (C order)."""
    if isinstance(value, Color):
        return ordered_layout((height, width, value.channels), (0, 1, 2))
    if rgb and _shape(value)[2] != 3:
        return ordered_layout((height, width, 3), (0, 1, 2))
    return _three_dimensional(value)


def _assemble(
    inputs: tuple[np.ndarray | Color, ...],
    selections: list[tuple[int, int]],
    order: tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    """Copy selected channels or scalar colors in one pass without color images,
    into a new image whose memory runs through its axes in order (numpy_order).
    Interleaved and planar inputs are read in place; a planar result is written
    one plane at a time."""
    first = next(value for value in inputs if isinstance(value, np.ndarray))
    height, width, _ = _shape(first)
    readable = {
        index: pixel_source(_three_dimensional(value))
        for index, value in enumerate(inputs)
        if isinstance(value, np.ndarray)
    }
    descriptors = (_Channel * len(selections))()
    for output_channel, (input_index, input_channel) in enumerate(selections):
        value = inputs[input_index]
        if isinstance(value, np.ndarray):
            h, w, channels = _shape(value)
            if (h, w) != (height, width) or not 0 <= input_channel < channels:
                raise ValueError("Invalid source dimensions or channel selection")
            array, pixel, plane = readable[input_index]
            descriptors[output_channel] = _channel(array, pixel, plane, input_channel)
        else:
            descriptors[output_channel] = _Channel(
                None, 0, 0, value.value[input_channel]
            )
    out = empty_ordered((height, width, len(selections)), order)
    strides = pixel_strides(out)
    if strides == (len(selections), 1):
        check(
            _lib.cn_channels_assemble(
                descriptors, ptr(out), height * width, len(selections)
            )
        )
    elif strides is not None and strides[0] == 1:
        for c in range(len(selections)):
            check(
                _lib.cn_channels_assemble(
                    ct.pointer(descriptors[c]),
                    ptr(out[:, :, c]),
                    height * width,
                    1,
                )
            )
    else:
        np.copyto(out, _assemble(inputs, selections))
    return out


def _same_size(inputs: tuple[np.ndarray | Color, ...], error: str) -> None:
    shapes = [_shape(value)[:2] for value in inputs if isinstance(value, np.ndarray)]
    if not shapes:
        raise ValueError(error)
    assert all(shape == shapes[0] for shape in shapes), (
        "All channel images must have the same resolution"
    )


def combine_rgba(
    red: np.ndarray | Color,
    green: np.ndarray | Color,
    blue: np.ndarray | Color,
    alpha: np.ndarray | Color | None,
) -> np.ndarray:
    inputs = (blue, green, red, Color.gray(1) if alpha is None else alpha)
    _same_size(
        inputs,
        "At least one channels must be an image, but all given channels are colors.",
    )
    if any(
        (_shape(v)[2] if isinstance(v, np.ndarray) else v.channels) != 1 for v in inputs
    ):
        raise ValueError("All channel images must have a single channel")
    # Upstream's np.stack on a new channel axis: np.concatenate's order K.
    h, w = next(_shape(v)[:2] for v in inputs if isinstance(v, np.ndarray))
    order = numpy_order(*(_layout(value, h, w) for value in inputs))
    return _assemble(inputs, [(i, 0) for i in range(4)], order)


def combine_rgb_alpha(rgb: np.ndarray | Color, alpha: np.ndarray | Color) -> np.ndarray:
    inputs = (rgb, alpha)
    _same_size(
        inputs,
        "At least one input must be an image, but both RGB and Alpha are colors.",
    )
    channels = _shape(rgb)[2] if isinstance(rgb, np.ndarray) else rgb.channels
    alpha_channels = (
        _shape(alpha)[2] if isinstance(alpha, np.ndarray) else alpha.channels
    )
    if channels not in (1, 3, 4) or alpha_channels != 1:
        raise ValueError("Expected grayscale/RGB/RGBA color and a single alpha channel")
    # Upstream's np.dstack((as_target_channels(rgb, 3, narrowing=True), alpha)).
    h, w = next(_shape(v)[:2] for v in inputs if isinstance(v, np.ndarray))
    order = numpy_order(_layout(rgb, h, w, rgb=True), _layout(alpha, h, w))
    return _assemble(
        inputs, [(0, 0 if channels == 1 else i) for i in range(3)] + [(1, 0)], order
    )


def merge_channels(*images: np.ndarray | None) -> np.ndarray:
    inputs = tuple(image for image in images if image is not None)
    if not inputs:
        raise ValueError("At least one input image is required")
    h, w = _shape(inputs[0])[:2]
    assert all(_shape(image)[:2] == (h, w) for image in inputs), (
        "All images to be merged must be the same resolution"
    )
    selections = [
        (i, c) for i, image in enumerate(inputs) for c in range(_shape(image)[2])
    ][:4]
    # Upstream's np.concatenate on the channel axis, whose first four channels
    # are kept; two channels go through cv2.merge((b, g, g)) (C order).
    order = (0, 1, 2)
    if len(selections) == 2:
        selections.append(selections[1])
    else:
        order = numpy_order(*(_layout(image, h, w) for image in inputs))
    return _assemble(inputs, selections, order)


def flip(image: np.ndarray, axis: int) -> np.ndarray:
    height, width, channels = _shape(image)
    if axis == 2:
        return image
    if axis not in (-1, 0, 1):
        raise ValueError("Invalid flip axis")
    source = f32(image)
    shape = (height, width) if channels == 1 else image.shape
    out = np.empty(shape, dtype=np.float32)
    check(_lib.cn_channels_flip(ptr(source), ptr(out), height, width, channels, axis))
    return out


def shift(image: np.ndarray, dx: int, dy: int, fill: int) -> np.ndarray:
    height, width, channels = _shape(image)
    if fill == 2:
        dx, dy = dx % width, dy % height
        if dx == 0 and dy == 0:
            return image
        # Upstream's np.roll fills np.empty_like(image), np.clip's order K; a
        # planar result is rolled one plane at a time.
        out = np.empty_like(image, order="K")
        planes = channel_planes(image, out)
        if planes is not None:
            for source, target in planes:
                check(
                    _lib.cn_channels_shift(
                        ptr(source), ptr(target), height, width, 1, dx, dy, fill
                    )
                )
            return out
    else:
        if fill not in (-1, 0, 1) or height >= 32767 or width >= 32767:
            raise ValueError("Unsupported affine shift dimensions or fill")
        if fill == 1 and channels not in (1, 3, 4):
            raise AssertionError(f"Number of channels ({channels}) unexpected")
        if (fill == -1 and channels > 4) or channels > CV_CN_MAX:
            # The pinned original rejects these arrays before any pixel work:
            # AUTO produces a scalar longer than four entries; OpenCV's Mat
            # image representation has at most CV_CN_MAX channels. Retain its exact
            # exception diagnostics, without falling back for a supported image.
            from .image_utils import ShiftFill
            from .image_utils import shift as original_shift

            original_shift(image, dx, dy, ShiftFill(fill))
            raise RuntimeError("Expected the pinned OpenCV shift validation to fail")
        # Integer translations farther away than this cannot sample the image,
        # including OpenCV's zero-weight neighbor taps at the border.
        dx = max(-width - 1, min(width + 1, dx))
        dy = max(-height - 1, min(height + 1, dy))
        out_channels = 4 if fill == 1 else channels
        shape = (height, width) if out_channels == 1 else (height, width, out_channels)
        out = np.empty(shape, dtype=np.float32)
    target = out if out.flags.c_contiguous else np.empty(out.shape, np.float32)
    check(
        _lib.cn_channels_shift(
            ptr(f32(image)), ptr(target), height, width, channels, dx, dy, fill
        )
    )
    if target is not out:
        np.copyto(out, target)
    return out


def _bbox(
    image: np.ndarray, channel: int, threshold: float
) -> tuple[int, int, int, int]:
    height, width, channels = _shape(image)
    if not 0 <= channel < channels:
        raise ValueError("Invalid bounding-box channel")
    array, pixel, plane = pixel_source(_three_dimensional(image))
    source = _channel(array, pixel, plane, channel)
    bounds = (_size * 4)()
    check(
        _lib.cn_channels_bbox(
            source.data,
            height,
            width,
            source.channels,
            source.channel,
            threshold,
            bounds,
        )
    )
    return tuple(bounds)


def bounding_box(image: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    if _shape(image)[2] != 1:
        raise ValueError("Expected a grayscale bounding-box mask")
    bounds = _bbox(image, 0, min(threshold / 100, 0.99999))
    if bounds[2] == 0:
        raise RuntimeError("Resulting bounding box is empty.")
    return bounds


def crop_to_content(image: np.ndarray, threshold: float) -> np.ndarray:
    height, width, channels = _shape(image)
    if channels < 4:
        return image
    x, y, crop_width, crop_height = _bbox(image, 3, min(threshold / 100, 0.99999))
    if crop_width == 0:
        raise RuntimeError("Crop results in empty image.")
    # Upstream returns a view into np.copy(image), np.clip's order K, so the view
    # keeps the whole copy's strides. Only the crop's pixels are ever read, so
    # only they are copied, interleaved in one pass or plane by plane.
    out = np.empty_like(image, order="K")
    crop = (slice(y, y + crop_height), slice(x, x + crop_width))
    pairs = channel_planes(image, out)
    if pairs is None and out.flags.c_contiguous:
        pairs = [(f32(image), out)]
    if pairs is None:
        np.copyto(out[crop], image[crop])
        return out[crop]
    for source, target in pairs:
        count = 1 if target.ndim == 2 else channels
        check(
            _lib.cn_channels_crop(
                ptr(source),
                ptr(target[crop]),
                height,
                width,
                count,
                x,
                y,
                crop_width,
                crop_height,
                width,
            )
        )
    return out[crop]


def concatenate_channels(*images: np.ndarray) -> np.ndarray:
    """Join all image channels without Merge Channels' four-channel UI limit."""
    if not images:
        raise ValueError("At least one input image is required")
    shapes = [_shape(image) for image in images]
    if any(shape[:2] != shapes[0][:2] for shape in shapes):
        raise ValueError("Channel images must have matching dimensions")
    # Upstream's np.dstack and np.concatenate on the channel axis.
    h, w, _ = shapes[0]
    return _assemble(
        images,
        [
            (index, channel)
            for index, shape in enumerate(shapes)
            for channel in range(shape[2])
        ],
        numpy_order(*(_layout(image, h, w) for image in images)),
    )
