"""Independent pinned NumPy operator and installed normalization oracles."""

from __future__ import annotations

import copy
import operator
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
import pytest
from onnx import helper
from test_onnx_converter import current, model, old, snapshot

from nodes.impl.native_graph import graph

DTYPES = [
    np.bool_,
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]
OPS = [operator.lt, operator.le, operator.eq, operator.ne, operator.gt, operator.ge]


type ErrorPolicy = Literal["ignore", "warn", "raise", "call", "print", "log"]


def observed(function, policy: ErrorPolicy = "warn", warning_error=False):
    events = []

    class Sink:
        def write(self, message):
            events.append(message)

    callback = Sink() if policy == "log" else lambda *args: events.append(args)
    with (
        np.errstate(all=policy, call=callback),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("error" if warning_error else "always")
        try:
            value = function()
            result = (
                (
                    type(value),
                    value.dtype.str,
                    value.shape,
                    value.strides,
                    value.tobytes(),
                )
                if isinstance(value, np.ndarray)
                else (type(value), value)
            )
        except Exception as error:
            result = (type(error), str(error), error.args)
    return result, [(type(w.message), str(w.message)) for w in caught], events


def compare(left, right, op, policy: ErrorPolicy = "warn", warning_error=False):
    expected = observed(lambda: OPS[op](left, right), policy, warning_error)
    actual = observed(
        lambda: graph().onnx_converter_compare(left, right, op), policy, warning_error
    )
    assert actual == expected


def values(dtype):
    dt = np.dtype(dtype)
    if dt.kind in "iu":
        info = np.iinfo(dtype)
        return np.array([info.min, 0, 1, info.max], dtype=dtype)
    if dt.kind == "b":
        return np.array([False, True, True, False])
    return np.array([-np.inf, -0.0, 1.0, np.nan], dtype=dtype)


@pytest.mark.parametrize("left_type", DTYPES)
@pytest.mark.parametrize("right_type", DTYPES)
@pytest.mark.parametrize("op", range(6))
def test_numeric_promotion(left_type, right_type, op):
    compare(values(left_type)[:, None], values(right_type)[None, :], op)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "other", [0, 1, -1, 255, 256, 2**63, 2**64 - 1, 1e-50, 1e39, np.float64(0.1)]
)
@pytest.mark.parametrize("op", range(6))
def test_value_aware_scalar_promotion(dtype, other, op):
    compare(values(dtype), other, op)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (1, 7), (5, 1), (3, 7, 2)])
@pytest.mark.parametrize(
    "kind", ["ordinary", "fortran", "reverse", "unaligned", "readonly"]
)
@pytest.mark.parametrize("op", range(6))
def test_foreign_layouts(shape, kind, op):
    left = np.random.default_rng(74).normal(size=shape).astype(np.float32)
    if kind == "fortran":
        left = np.asfortranarray(left)
    elif kind == "reverse":
        left = left[::-1, ::-1]
    elif kind == "unaligned":
        foreign = np.ndarray(left.shape, left.dtype, bytearray(left.nbytes + 1), 1)
        foreign[:] = left
        left = foreign
    elif kind == "readonly":
        left.flags.writeable = False
    before = left.tobytes()
    compare(left, np.float32(0.3), op)
    assert left.tobytes() == before


@pytest.mark.parametrize(
    "dtype,uint,snan,qnan",
    [
        ("f2", "u2", 0x7C01, 0x7E01),
        ("f4", "u4", 0x7F800001, 0x7FC00001),
        ("f8", "u8", 0x7FF0000000000001, 0x7FF8000000000001),
    ],
)
@pytest.mark.parametrize("op", range(6))
@pytest.mark.parametrize("policy", ["warn", "raise", "call", "log", "ignore"])
def test_real_nan_payloads(dtype, uint, snan, qnan, op, policy):
    sign = 1 << (np.dtype(uint).itemsize * 8 - 1)
    left = np.array([snan, qnan, sign | snan, sign | qnan], uint).view(dtype)
    compare(left, 1, op, policy)


@pytest.mark.parametrize(
    "dtype,uint,snan,qnan",
    [
        ("c8", "u4", 0x7F800001, 0x7FC00001),
        ("c16", "u8", 0x7FF0000000000001, 0x7FF8000000000001),
    ],
)
@pytest.mark.parametrize("component", [0, 1])
@pytest.mark.parametrize("signaling", [False, True])
@pytest.mark.parametrize("other", [-1, 0, 1])
@pytest.mark.parametrize("op", range(6))
@pytest.mark.parametrize("policy", ["warn", "raise", "call", "log", "ignore"])
def test_complex_nan_events(
    dtype, uint, snan, qnan, component, signaling, other, op, policy
):
    left = np.zeros(1, dtype)
    left.view(uint)[component] = snan if signaling else qnan
    compare(left, other, op, policy)


@pytest.mark.parametrize("op", range(6))
@pytest.mark.parametrize("policy", ["warn", "raise", "call", "log", "ignore"])
@pytest.mark.parametrize(
    "other", [np.ones(4, np.float64), np.array(1, np.float64), np.float64(1e300), 1e300]
)
def test_promotion_cast_events(op, policy, other):
    left = np.array([0x7F800001, 0x7FC00123, 0xFF800001, 0x3F800000], np.uint32).view(
        np.float32
    )
    compare(left, other, op, policy)
    compare(other, left, op, policy)


@pytest.mark.parametrize("op", range(6))
@pytest.mark.parametrize("other", [np.ones((2, 4), np.float32), np.ones(7, np.float64)])
def test_broadcast_errors(op, other):
    compare(np.ones((3, 4), np.float32), other, op)


@pytest.mark.parametrize("op", range(6))
@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.int64, np.uint64]
)
def test_zero_dimensional_and_scalar_results(op, dtype):
    compare(np.array(1, dtype), np.array(2, dtype), op)
    compare(dtype(1), dtype(2), op)


@pytest.mark.parametrize("op", range(6))
def test_half_bit_domain(op):
    left = np.arange(65536, dtype=np.uint16).view(np.float16)
    compare(left, np.float16(0), op)
    compare(left, left[::-1], op)


@pytest.mark.parametrize("op", range(6))
@pytest.mark.parametrize("dtype", [">f4", ">f8", ">i8", ">u8", ">c8"])
def test_non_native_byte_order(op, dtype):
    compare(np.array([0, 1, 7, 15], dtype), np.array([1, 1, 1, 1], np.float32), op)


@pytest.mark.parametrize(
    "operation", ["GroupNorm", "InstanceNormalization", "LayerNorm"]
)
@pytest.mark.parametrize("policy", ["warn", "raise", "call", "log", "ignore"])
@pytest.mark.parametrize("bits", [0x7F800001, 0xFF800003, 0x7FC00123])
def test_original_normalization_graph(operation, policy, bits):
    scale = np.array([bits, 0x3F800000, 0x3F800000], np.uint32).view(np.float32)
    source = model(
        [
            helper.make_node(
                operation, ["x", "s", "b"], ["y"], channels=3, groups=1, affine=1
            )
        ],
        {"s": scale, "b": np.zeros(3, np.float32)},
    )

    def run(module):
        converted = module.Onnx2NcnnConverter(copy.deepcopy(source))
        value = converted.convert()
        return value.write_param(), value.serialize_weights(), snapshot(converted)

    assert observed(lambda: run(current), policy) == observed(lambda: run(old), policy)


def test_concurrent_independent_comparisons():
    def run(seed):
        left = np.random.default_rng(seed).random((37, 53), dtype=np.float32)
        compare(left, 0.5, seed % 6)

    with ThreadPoolExecutor(4) as executor:
        list(executor.map(run, range(24)))


@pytest.mark.parametrize("op", range(6))
@pytest.mark.parametrize("other", [2**65, -(2**65)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_unbounded_integer_promotion(op, other, dtype):
    compare(values(dtype), other, op)
    compare(other, values(dtype), op)


@pytest.mark.parametrize("op", range(6))
def test_noncanonical_bool_storage(op):
    compare(np.arange(256, dtype=np.uint8).view(np.bool_), np.ones(256, np.bool_), op)


@pytest.mark.parametrize("op", range(6))
@pytest.mark.parametrize("policy", ["warn", "raise"])
def test_warning_as_error_chaining(op, policy):
    left = np.array([0x7F800001], np.uint32).view(np.float32)
    right = np.ones(1, np.float64)

    def run(function):
        with np.errstate(all=policy), warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                function()
            except Exception as error:

                def snapshot_error(value):
                    if value is None:
                        return None
                    return type(value), str(value), value.args

                return (
                    snapshot_error(error),
                    snapshot_error(error.__cause__),
                    snapshot_error(error.__context__),
                )
        raise AssertionError("The source fixture must fail")

    assert run(lambda: graph().onnx_converter_compare(left, right, op)) == run(
        lambda: OPS[op](left, right)
    )


@pytest.mark.parametrize("op", range(6))
@pytest.mark.parametrize("policy", ["warn", "raise", "call", "log", "ignore"])
@pytest.mark.parametrize(
    "dtype,uint,snan", [("f4", "u4", 0x7F800001), ("f8", "u8", 0x7FF0000000000001)]
)
def test_object_promotion_signaling_nan(op, policy, dtype, uint, snan):
    left = np.array([snan], uint).view(dtype)
    compare(left, 2**65, op, policy)
