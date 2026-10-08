"""Native model image buffers, normalization, trimaps and cloth mask decisions.

Reshape/transpose/squeeze are metadata views; numeric casts, reconstruction,
normalization and mask arithmetic use the checked native implementations.
"""

from __future__ import annotations

import ctypes as ct
import sys
import warnings
from collections.abc import Callable
from functools import lru_cache
from typing import cast

import numpy as np

from .native import check, f32, lib
from .native_numpy_simd import fma3, loop_target
from .native_transfer import contrast_bounds


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    p, z, i, f = ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_float
    signatures = {
        "cn_framework_pad": [p, p, z, z, z, z, z, z, i],
        "cn_framework_ncnn": [p, p, z, z],
        "cn_framework_affine": [p, p, z, z, z, f, i, p],
        "cn_framework_mask_threshold": [p, p, z],
        "cn_framework_trimap": [p, p, z, z, f, f, z],
        "cn_framework_cloth_labels": [p, p, z, z, i, p],
        "cn_framework_cloth_masks": [p, p, z],
        "cn_planar_to_interleaved_f32": [p, i, z, z, z, z, z, i, p],
    }
    for name, signature in signatures.items():
        function = getattr(dll, name)
        function.argtypes, function.restype = signature, i
    return dll


def _report(events: int, operation: str) -> None:
    for flag, kind, description in (
        (1, "divide", "divide by zero"),
        (2, "over", "overflow"),
        (4, "under", "underflow"),
        (8, "invalid", "invalid value"),
    ):
        if not events & flag:
            continue
        mode = np.geterr()[kind]
        message = f"{description} encountered in {operation}"
        if mode == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        elif mode == "raise":
            raise FloatingPointError(message)
        elif mode == "print":
            print("Warning: " + message, file=sys.stderr)
        elif mode in ("call", "log"):
            handler = np.geterrcall()
            if handler is None:
                raise NameError(
                    f"python callback specified for {description} (in {operation}) but no function found."
                )
            if mode == "call":
                cast(Callable[[str, int], object], handler)(description, events)
            else:
                handler.write("Warning: " + message + "\n")


def reflect_pad(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim not in (2, 3) or not image.size:
        raise ValueError("Reflect padding requires a nonempty image")
    if image.dtype not in (
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    ):
        return np.pad(
            image,
            [(0, height), (0, width)] + ([(0, 0)] if image.ndim == 3 else []),
            "reflect",
        )
    if width < 0 or height < 0:
        raise ValueError("Padding must be nonnegative")
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    source = np.require(image, requirements=["C", "A"])
    order = "F" if image.flags.f_contiguous and not image.flags.c_contiguous else "C"
    shape = (
        (h + height, w + width, channels)
        if image.ndim == 3
        else (h + height, w + width)
    )
    result = np.empty(shape, image.dtype, order=order)
    check(
        _api().cn_framework_pad(
            source.ctypes.data,
            result.ctypes.data,
            h,
            w,
            channels,
            h + height,
            w + width,
            image.itemsize,
            int(order == "F"),
        )
    )
    return result


def swap_red_blue(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        return image
    if image.shape[2] == 3:
        return image[:, :, ::-1]
    if image.shape[2] == 4:
        # The same-type native caster retains half-model bit patterns.
        return _assemble_four(image)
    return image


def _assemble_four(image: np.ndarray) -> np.ndarray:
    # The typed C caster accepts arbitrary signed byte strides; use it for each
    # source plane and assign into an owned destination through the same ABI.
    from .native_tensors import _api as tensor_api
    from .native_tensors import _view

    h, w, channels = image.shape
    if h > 1 and abs(image.strides[0]) < abs(image.strides[1]):
        out = np.empty((w, h, channels), image.dtype).transpose(1, 0, 2)
    else:
        out = np.empty(image.shape, image.dtype)
    for target, source in enumerate((2, 1, 0, 3)):
        events = ct.c_int()
        check(
            tensor_api().cn_tensor_cast_typed(
                _view(image[:, :, source]), _view(out[:, :, target]), ct.byref(events)
            )
        )
        _report(events.value, "cast")
    return out


def planar_to_interleaved(planes: np.ndarray, reverse: bool) -> np.ndarray:
    """(C, H, W) float16 or float32 planes as a new (H, W, C) float32 image, every
    value exact (NumPy's cast). Rows may be cropped from a wider buffer (positive
    strides, contiguous within a row). reverse flips the channel order."""
    if planes.ndim != 3 or planes.dtype not in (
        np.dtype(np.float16),
        np.dtype(np.float32),
    ):
        raise ValueError("Planar conversion needs (C, H, W) float16 or float32 planes")
    item = planes.itemsize
    plane_stride, row_stride, column_stride = planes.strides
    if column_stride != item or plane_stride % item or row_stride % item:
        raise ValueError("Planar conversion needs rows of contiguous samples")
    channels, h, w = planes.shape
    result = np.empty((h, w, channels), np.float32)
    check(
        _api().cn_planar_to_interleaved_f32(
            planes.ctypes.data,
            int(planes.dtype == np.float16),
            plane_stride // item,
            row_stride // item,
            h,
            w,
            channels,
            int(reverse),
            result.ctypes.data,
        )
    )
    return result


def ncnn_input(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim not in (2, 3) or not image.size:
        raise ValueError("NCNN pixel conversion requires a nonempty uint8 image")
    channels = image.shape[2] if image.ndim == 3 else 1
    if channels not in (1, 3, 4):
        raise ValueError("NCNN pixel conversion supports one, three or four channels")
    source = np.require(image, requirements=["C", "A"])
    h, w = source.shape[:2]
    result = np.empty((channels, h, w), np.float32)
    check(
        _api().cn_framework_ncnn(
            source.ctypes.data, result.ctypes.data, h * w, channels
        )
    )
    return result


def _affine(
    source: np.ndarray,
    out: np.ndarray,
    channels: int,
    channel: int,
    scalar: float,
    divide: bool,
) -> None:
    events = ct.c_int()
    check(
        _api().cn_framework_affine(
            source.ctypes.data,
            out.ctypes.data,
            source.size // channels,
            channels,
            channel,
            scalar,
            int(divide),
            ct.byref(events),
        )
    )
    _report(events.value, "divide" if divide else "subtract")


def normalize_model(
    image: np.ndarray, mean: tuple[float, float, float], std: tuple[float, float, float]
) -> np.ndarray:
    source = f32(image)
    if source.ndim != 3 or source.shape[2] != 3 or not source.size:
        raise ValueError("Model normalization requires a nonempty three-channel image")
    batch = np.empty_like(np.expand_dims(source.transpose(2, 0, 1), 0))
    result = batch[0].transpose(1, 2, 0)
    for channel in range(3):
        _affine(source, result, 3, channel, mean[channel], False)
        _affine(result, result, 3, channel, std[channel], True)
    return batch


def normalize_prediction(prediction: np.ndarray) -> np.ndarray:
    if prediction.dtype != np.float32 or not prediction.size:
        maximum, minimum = np.max(prediction), np.min(prediction)
        return (prediction - minimum) / (maximum - minimum)
    minimum, maximum = contrast_bounds(prediction)
    minimum, maximum = np.float32(minimum), np.float32(maximum)
    source = f32(prediction)
    result = np.empty_like(source)
    _affine(source, result, 1, 0, float(minimum), False)
    span = maximum - minimum
    _affine(result, result, 1, 0, float(span), True)
    return result


def threshold_mask(mask: np.ndarray) -> np.ndarray:
    source = f32(mask)
    out = np.empty_like(source)
    check(
        _api().cn_framework_mask_threshold(
            source.ctypes.data, out.ctypes.data, source.size
        )
    )
    return out


def matting_trimap(
    mask: np.ndarray, foreground: int, background: int, size: int
) -> np.ndarray:
    source = f32(mask)
    if source.ndim != 2 or not source.size:
        raise ValueError("Trimap requires a nonempty grayscale mask")
    out = np.empty(source.shape, np.float64)
    check(
        _api().cn_framework_trimap(
            source.ctypes.data,
            out.ctypes.data,
            *source.shape,
            foreground / 255,
            background / 255,
            max(size, 0),
        )
    )
    return out


def cloth_labels(prediction: np.ndarray) -> np.ndarray:
    if prediction.dtype != np.float32:
        from scipy.special import log_softmax

        return np.squeeze(
            np.argmax(log_softmax(prediction, 1), axis=1, keepdims=True), (0, 1)
        ).astype(np.uint8)
    source = f32(prediction)
    if source.ndim != 4 or source.shape[0] != 1 or not source.size:
        raise ValueError(
            "Cloth model output must have shape (1, classes, height, width)"
        )
    _, classes, h, w = source.shape
    out = np.empty((h, w), np.uint8)
    events = (ct.c_int * 4)()
    # exp and log mirror np.exp/np.log's float32 loops (one dispatch source,
    # loops_exponent_log): the AVX2/FMA3 form where NumPy dispatches FMA3, else
    # the CRT. MSVC builds compile out the AVX-512F form.
    fused = fma3(loop_target("exp", "ff"))
    check(
        _api().cn_framework_cloth_labels(
            source.ctypes.data, out.ctypes.data, h * w, classes, int(fused), events
        )
    )
    for flag, operation in zip(
        events, ("subtract", "exp", "reduce", "subtract"), strict=True
    ):
        _report(flag, operation)
    return out


def cloth_masks(labels: np.ndarray) -> list[np.ndarray]:
    if labels.dtype != np.uint8 or labels.ndim != 2 or not labels.size:
        raise ValueError("Cloth labels must be a nonempty uint8 plane")
    source = np.require(labels, requirements=["C", "A"])
    out = np.empty((3, *source.shape), np.float32)
    check(
        _api().cn_framework_cloth_masks(
            source.ctypes.data, out.ctypes.data, source.size
        )
    )
    return [out[index] for index in range(3)]
