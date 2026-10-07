"""Typed C++ CPU coefficient arithmetic; no model execution or device transfers.

NumPy/Torch resolve dtype, broadcast and allocation metadata. Ordinary numeric
buffers, including scalar/complex/integer and half/bfloat weights, are processed
by native stages. Device, autograd, subclass, sparse/quantized and conjugate or
negative-view semantics remain explicitly owned by Torch.
"""

from __future__ import annotations

import ctypes as ct
import sys
import warnings
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numpy as np

from .native import check, lib

if TYPE_CHECKING:
    import torch

_TYPES = {
    np.dtype(name): i
    for i, name in enumerate(
        (
            "float16",
            "float32",
            "float64",
            "complex64",
            "complex128",
            "bool",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
        )
    )
}


class _View(ct.Structure):
    buffer_owner: np.ndarray | None = None
    _fields_ = [
        ("data", ct.c_void_p),
        ("count", ct.c_size_t),
        ("dimensions", ct.c_size_t),
        ("shape", ct.POINTER(ct.c_size_t)),
        ("strides", ct.POINTER(ct.c_ssize_t)),
        ("type", ct.c_int),
    ]


def _view(array: np.ndarray, type_code: int | None = None) -> _View:
    # ctypes pointer fields retain their backing arrays for the synchronous call.
    descriptor = _View(
        array.ctypes.data,
        array.size,
        array.ndim,
        (ct.c_size_t * array.ndim)(*array.shape),
        (ct.c_ssize_t * array.ndim)(*array.strides),
        _TYPES[array.dtype] if type_code is None else type_code,
    )
    descriptor.buffer_owner = array
    return descriptor


@lru_cache(maxsize=1)
def _api():
    library = lib()
    view, flag = ct.POINTER(_View), ct.POINTER(ct.c_int)
    for name, arguments in {
        "cn_tensor_scale_typed": [view, view, ct.c_double, ct.c_int, flag],
        "cn_tensor_add_typed": [view, view, view, flag],
        "cn_tensor_cast_typed": [view, view, flag],
        "cn_tensor_interpolate": [
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_size_t,
            ct.c_int,
            ct.c_double,
            ct.c_double,
        ],
    }.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, ct.c_int
    return library


def _report_fp(events: int, operation: str) -> None:
    for flag, kind, name in (
        (2, "over", "overflow"),
        (4, "under", "underflow"),
        (8, "invalid", "invalid value"),
    ):
        if not events & flag:
            continue
        mode = np.geterr()[kind]
        message = f"{name} encountered in {operation}"
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
                    f"python callback specified for {name} (in {operation}) but no function found."
                )
            if mode == "call":
                cast(Callable[[str, int], object], handler)(name, events)
            else:
                handler.write("Warning: " + message + "\n")


def _allocate(a: np.ndarray, b: np.ndarray, dtype: np.dtype) -> np.ndarray:
    # Resolve K-order output strides without advancing the iterator or computing
    # any array element. The native call performs all arithmetic.
    with np.nditer(
        (a, b, None),  # None: NumPy allocates this operand
        flags=["zerosize_ok"],
        op_flags=[["readonly"], ["readonly"], ["writeonly", "allocate"]],
        op_dtypes=[None, None, dtype],
        casting="unsafe",
    ) as iterator:
        return iterator.operands[2]


def _scale_numpy(a: np.ndarray, amount: float) -> np.ndarray:
    dtype = np.result_type(a, amount)
    # NEP 50: the weak Python coefficient converts to dtype before the element loop,
    # with its own error boundary. NumPy 2 reports that conversion's overflow but
    # never its underflow (NumPy 1.24's cast reported both).
    scalar = np.asarray(amount)
    if scalar.dtype in _TYPES:
        coefficient = np.empty((), dtype)
        converted = ct.c_int()
        check(
            _api().cn_tensor_cast_typed(
                _view(scalar), _view(coefficient), ct.byref(converted)
            )
        )
        _report_fp(converted.value & ~4, "cast")
    else:
        # A Python int beyond 64 bits is an object scalar: NumPy's cast converts it.
        coefficient = scalar.astype(dtype)
    # A zero-D operand adds no axis: the output's layout is a's.
    out = _allocate(a, coefficient, dtype)
    events = ct.c_int()
    check(
        _api().cn_tensor_scale_typed(
            _view(a), _view(out), float(coefficient.item().real), 0, ct.byref(events)
        )
    )
    _report_fp(events.value, "multiply")
    return out


def interpolate_numpy(
    a: np.ndarray, b: np.ndarray, amount_a: float, amount_b: float
) -> np.ndarray:
    if (
        type(a) is not np.ndarray
        or type(b) is not np.ndarray
        or a.dtype not in _TYPES
        or b.dtype not in _TYPES
        # Non-float coefficients can select integer ufunc loops. The node always
        # supplies float percentages; retain this external helper contract.
        or np.result_type(a, amount_a).kind not in "fc"
        or np.result_type(b, amount_b).kind not in "fc"
    ):
        return a * amount_a + b * amount_b
    left, right = _scale_numpy(a, amount_a), _scale_numpy(b, amount_b)
    # NEP 50 (NumPy 2): zero-D operands promote by dtype, as arrays do; only the
    # Python-float coefficients are weak, and _scale_numpy resolved those.
    dtype = np.result_type(left, right)
    out = _allocate(left, right, dtype)
    if left.ndim == 0 and left.dtype != dtype:
        left = cast_numpy(left, dtype)
    if right.ndim == 0 and right.dtype != dtype:
        right = cast_numpy(right, dtype)
    left, right = np.broadcast_arrays(left, right)
    events = ct.c_int()
    check(
        _api().cn_tensor_add_typed(
            _view(left), _view(right), _view(out), ct.byref(events)
        )
    )
    _report_fp(events.value, "scalar add" if out.ndim == 0 else "add")
    return out[()] if out.ndim == 0 else out


def cast_numpy(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Numeric cast for ONNX/NCNN serialization; no arithmetic delegated."""
    source = np.asarray(value)
    dtype = np.dtype(dtype)
    if source.dtype not in _TYPES or dtype not in _TYPES:
        return source.astype(dtype)
    if source.dtype.kind == "c" and dtype.kind not in "cb":
        warnings.warn(
            "Casting complex values to real discards the imaginary part",
            np.exceptions.ComplexWarning,
            stacklevel=2,
        )
    out = np.empty_like(source, dtype=dtype, order="K", subok=False)
    events = ct.c_int()
    check(_api().cn_tensor_cast_typed(_view(source), _view(out), ct.byref(events)))
    _report_fp(events.value, "cast")
    return out


@lru_cache(maxsize=1)
def _torch_types():
    import torch

    names = (
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "bool",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "bfloat16",
        "complex32",
    )
    return {
        dtype: code
        for code, name in enumerate(names)
        if (dtype := getattr(torch, name, None)) is not None
    }


def _torch_buffer(value: torch.Tensor) -> np.ndarray:
    import torch

    if _torch_types()[value.dtype] == 14:
        return value.view(torch.int16).numpy()
    if _torch_types()[value.dtype] == 15:
        return value.view(torch.int32).numpy()
    return value.numpy()


def _torch_storage(dtype: torch.dtype) -> np.dtype:
    if _torch_types()[dtype] == 14:
        return np.dtype("int16")
    if _torch_types()[dtype] == 15:
        return np.dtype("int32")
    return next(key for key, code in _TYPES.items() if code == _torch_types()[dtype])


def _scale_torch(value: torch.Tensor, amount: float) -> torch.Tensor:
    import torch

    dtype = torch.result_type(value, amount)
    source = _torch_buffer(value)
    result = torch.empty_like(value, dtype=dtype)
    out = _torch_buffer(result)
    check(
        _api().cn_tensor_scale_typed(
            _view(source, _torch_types()[value.dtype]),
            _view(out, _torch_types()[dtype]),
            amount,
            1,
            None,
        )
    )
    return result


def _torch_allocate(a: np.ndarray, b: np.ndarray, dtype: np.dtype) -> np.ndarray:
    # Torch gives the first non-broadcast operand precedence when layouts
    # disagree. This is allocation metadata, following TensorIterator's public
    # output-stride behavior; no operand values are read.
    a, b = np.broadcast_arrays(a, b)
    shape = a.shape
    order = list(reversed(range(a.ndim)))

    def compare(first: int, second: int) -> int:
        for source in (a, b):
            left, right = source.strides[first], source.strides[second]
            if left and right:
                if left != right:
                    return 1 if left > right else -1
                if shape[first] > shape[second]:
                    return 1
        return 0

    for current in range(1, len(order)):
        selected = current
        for prior in range(current - 1, -1, -1):
            comparison = compare(order[prior], order[selected])
            if comparison > 0:
                order[prior], order[selected] = order[selected], order[prior]
                selected = prior
            elif comparison < 0:
                break
    axes = list(reversed(order))
    storage = np.empty(tuple(shape[d] for d in axes), dtype=dtype)
    return storage.transpose(tuple(axes.index(d) for d in range(a.ndim)))


def interpolate_torch(
    a: torch.Tensor, b: torch.Tensor, amount_a: float, amount_b: float
):
    import torch

    if not (
        type(a) is torch.Tensor
        and type(b) is torch.Tensor
        and a.device.type == b.device.type == "cpu"
        and a.layout == b.layout == torch.strided
        and max(a.ndim, b.ndim) <= 32
        and a.dtype in _torch_types()
        and b.dtype in _torch_types()
        and isinstance(amount_a, float)
        and isinstance(amount_b, float)
        and not (
            a.requires_grad
            or b.requires_grad
            or a.is_conj()
            or b.is_conj()
            or a.is_neg()
            or b.is_neg()
        )
    ):
        return amount_a * a + amount_b * b
    left, right = _scale_torch(a, amount_a), _scale_torch(b, amount_b)
    dtype = torch.result_type(left, right)
    lhs, rhs = _torch_buffer(left), _torch_buffer(right)
    try:
        np.broadcast_shapes(lhs.shape, rhs.shape)
    except ValueError:
        # Error-only framework adapter: incompatible shapes fail before the
        # framework can compute any output element, preserving its exception.
        return left + right
    out = _torch_allocate(lhs, rhs, _torch_storage(dtype))
    if out.size == 0:
        return torch.empty(out.shape, dtype=dtype)
    result = None
    if left.is_contiguous() and right.is_contiguous():
        result = torch.empty(out.shape, dtype=dtype)
        out = _torch_buffer(result)
    elif tuple(left.shape) == out.shape:
        result = torch.empty_like(left, dtype=dtype)
        out = _torch_buffer(result)
    lhs, rhs = np.broadcast_arrays(lhs, rhs)
    check(
        _api().cn_tensor_add_typed(
            _view(lhs, _torch_types()[left.dtype]),
            _view(rhs, _torch_types()[right.dtype]),
            _view(out, _torch_types()[dtype]),
            None,
        )
    )
    return result if result is not None else torch.from_numpy(out).view(dtype)
