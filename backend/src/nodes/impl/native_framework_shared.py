"""Shared CPU image arithmetic, exact output layouts and owned tile writes."""

# Allocation stride metadata below is an altered adaptation of NumPy 1.24.4
# PyArray_CreateMultiSortedStridePerm (numpy/core/src/multiarray/shape.c).
# Copyright (c) 2005-2022, NumPy Developers. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the NumPy Developers nor the names of any contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import ctypes as ct
import math
from functools import lru_cache

import numpy as np

from .native import check, f32, lib
from .native_framework_images import _report
from .native_numpy_simd import fma3, loop_target
from .native_tensors import _allocate, _view, cast_numpy
from .native_tensors import _api as tensor_api


@lru_cache(maxsize=1)
def _api():
    library = lib()
    library.cn_framework_shared.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_int,
        ct.c_int,
        ct.c_float,
        ct.c_void_p,
    ]
    library.cn_framework_shared.restype = ct.c_int
    return library


def _run(
    source: np.ndarray | None,
    out: np.ndarray,
    mode: int,
    scalar: float = 0,
    second: np.ndarray | None = None,
) -> None:
    if not out.size:
        return
    events = (ct.c_int * 9)()
    # The sine curves mirror np.sin's float32 loop: its FMA3 form where NumPy
    # dispatches one, else the CRT sinf.
    fused = fma3(loop_target("sin", "ff"))
    check(
        _api().cn_framework_shared(
            None if source is None else source.ctypes.data,
            None if second is None else second.ctypes.data,
            out.ctypes.data,
            out.size,
            mode,
            int(fused),
            scalar,
            events,
        )
    )
    sine = ("multiply", "subtract", "sin", "add", "divide")
    half = ("multiply", "subtract", "clip")
    operations = {
        0: ("clip",),
        1: sine,
        2: (*half, *sine),
        3: ("subtract", "subtract"),
        4: ("fill",),
        5: ("divide",),
        6: ("divide", *sine),
        7: ("divide", *half, *sine),
    }[mode]
    for event, operation in zip(events[: len(operations)], operations, strict=True):
        _report(event, operation)


def copy_into(target: np.ndarray, source: np.ndarray) -> None:
    """Copy a tile view with the existing checked strided C tensor caster."""
    if target.dtype != np.float32 or source.dtype != np.float32:
        target[...] = source
        return
    if not target.flags.writeable:
        raise ValueError("assignment destination is read-only")
    while source.ndim > target.ndim and source.shape[0] == 1:
        source = source[0]
    try:
        source = np.broadcast_to(source, target.shape)
    except ValueError as error:
        source_shape = str(source.shape).replace(" ", "")
        target_shape = str(target.shape).replace(" ", "")
        raise ValueError(
            f"could not broadcast input array from shape {source_shape} into shape {target_shape}"
        ) from error
    if np.may_share_memory(target, source):
        source = cast_numpy(source, source.dtype)
    events = ct.c_int()
    check(
        tensor_api().cn_tensor_cast_typed(
            _view(source), _view(target), ct.byref(events)
        )
    )
    _report(events.value, "cast")


def clip_image(image: np.ndarray) -> np.ndarray:
    if type(image) is not np.ndarray or image.dtype != np.float32 or image.ndim == 0:
        return np.clip(image, 0, 1)
    source = np.require(image.ravel(order="K"), requirements=["C", "A"])
    out = np.empty_like(image)
    _run(source, out.ravel(order="K"), 0)
    return out


def blend_curve(image: np.ndarray, *, half: bool = False) -> np.ndarray:
    if type(image) is not np.ndarray or image.dtype != np.float32 or image.ndim == 0:
        if half:
            image = np.clip(image * 2 - 0.5, 0, 1)
        return (np.sin(image * math.pi - math.pi / 2) + 1) / 2
    source = np.require(image.ravel(order="K"), requirements=["C", "A"])
    out = np.empty_like(image)
    _run(source, out.ravel(order="K"), 2 if half else 1)
    return out


def blend_weights(size: int, *, half: bool = False) -> np.ndarray:
    out = np.empty(size, np.float32)
    if np.result_type(out, size - 1) != np.float32:
        return blend_curve(np.arange(size, dtype=np.float32) / (size - 1), half=half)
    _run(None, out, 7 if half else 6, float(size - 1))
    return out


def recover_alpha(white: np.ndarray, black: np.ndarray) -> np.ndarray:
    if white.dtype != np.float32 or black.dtype != np.float32:
        return 1 - (white - black)
    intermediate = _allocate(white, black, np.dtype(np.float32))
    out = np.empty_like(intermediate)
    del intermediate
    white, black = np.broadcast_arrays(white, black)
    result = np.empty(white.shape, np.float32)
    _run(f32(white), result, 3, second=f32(black))
    copy_into(out, result)
    return out


def assemble_alpha(rgb: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    if rgb.dtype != np.float32 or (
        isinstance(alpha, np.ndarray) and alpha.dtype != np.float32
    ):
        if not isinstance(alpha, np.ndarray):
            alpha = np.full(rgb.shape[:-1], alpha, np.float32)
        return np.dstack((rgb, alpha))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        if not isinstance(alpha, np.ndarray):
            alpha = np.full(rgb.shape[:-1], alpha, np.float32)
        return np.dstack((rgb, alpha))
    if not isinstance(alpha, np.ndarray):
        plane = np.empty(rgb.shape[:-1], np.float32)
        _run(None, plane, 4, alpha)
        alpha = plane
    if alpha.ndim != 2 or alpha.shape != rgb.shape[:2]:
        return np.dstack((rgb, alpha))
    h, w = rgb.shape[:2]
    # NumPy 1.24.4 PyArray_CreateMultiSortedStridePerm: concatenation uses a
    # stable descending stride order, with singleton ambiguity and C-order
    # precedence on conflicts. This is allocation metadata, never pixel work.
    arrays = (rgb, alpha[:, :, None])
    permutation = [0, 1, 2]
    for index in range(1, 3):
        position, axis = index, permutation[index]
        for previous in range(index - 1, -1, -1):
            ambiguous, swap = True, False
            other = permutation[previous]
            for array in arrays:
                if array.shape[axis] != 1 and array.shape[other] != 1:
                    if abs(array.strides[axis]) <= abs(array.strides[other]):
                        swap = False
                    elif ambiguous:
                        swap = True
                    ambiguous = False
            if not ambiguous:
                if swap:
                    position = previous
                else:
                    break
        if position != index:
            permutation.insert(position, permutation.pop(index))
    shape, strides, stride = (h, w, 4), [0, 0, 0], 4
    for axis in reversed(permutation):
        strides[axis] = stride
        stride *= shape[axis]
    out = np.ndarray(shape, np.float32, strides=tuple(strides))
    copy_into(out[:, :, :3], rgb)
    copy_into(out[:, :, 3], alpha)
    return out
