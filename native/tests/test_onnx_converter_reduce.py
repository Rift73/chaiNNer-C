"""RNN bias reduction preserves NumPy's dtype and accumulation tree."""

import itertools
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from nodes.impl.native_graph import graph

DTYPES = [
    np.bool_,
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.uint64,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]


def result(call: Callable[[], Any]) -> tuple[Any, list[tuple[str, str]]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = call()
        except Exception as error:
            value = (type(error).__name__, str(error))
    return value, [(type(w.message).__name__, str(w.message)) for w in caught]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "n,layout",
    itertools.product(
        [0, 1, 3, 7, 8, 9, 63, 64, 65, 127, 128, 129, 257, 8193],
        ["C", "F", "reverse", "slice", "3d"],
    ),
)
def test_bias_sum(dtype, n, layout):
    rng = np.random.default_rng(131)
    shape = (3, n, 2) if layout == "3d" else (3, n)
    source = rng.uniform(-100, 100, shape)
    if np.issubdtype(dtype, np.complexfloating):
        source = source + rng.uniform(-100, 100, shape) * 1j
    with np.errstate(all="ignore"):
        source = source.astype(dtype)
    if layout == "F":
        source = np.asfortranarray(source)
    elif layout == "reverse":
        source = source[::-1, ::-1]
    elif layout == "slice":
        source = source[:, ::2]
    expected, ew = result(lambda: np.sum(source, 1))
    actual, aw = result(lambda: graph().onnx_converter_sum_axis(source, 1))
    assert aw == ew
    if isinstance(expected, tuple):
        assert actual == expected
    else:
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize(
    "values", [[-0.0] * 9, [np.inf, -np.inf, 0, np.nan], [1e38] * 130, [1e-30] * 9]
)
@pytest.mark.parametrize("policy", ["ignore", "warn", "raise"])
def test_special_values(dtype, values, policy):
    with np.errstate(all="ignore"):
        source = np.asarray([values], dtype=dtype)
    with np.errstate(all=policy):
        expected, ew = result(lambda: np.sum(source, 1))
        actual, aw = result(lambda: graph().onnx_converter_sum_axis(source, 1))
    assert aw == ew
    if isinstance(expected, tuple):
        assert actual == expected
    else:
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "source",
    [
        np.array(1),
        np.ones(7),
        np.empty((0, 0, 3)),
        np.empty((3, 4, 0)),
        np.ones((3, 5), dtype=">f4"),
    ],
)
def test_shape_and_byte_order(source):
    expected, ew = result(lambda: np.sum(source, 1))
    actual, aw = result(lambda: graph().onnx_converter_sum_axis(source, 1))
    assert aw == ew
    if isinstance(expected, tuple):
        assert actual == expected
    else:
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize(
    "values", itertools.product([0.0, np.inf, -np.inf, np.nan, -np.nan], repeat=4)
)
def test_complex_nan_propagation(dtype, values):
    source = np.asarray([values], dtype=dtype)
    with np.errstate(all="ignore"):
        expected = np.sum(source, 1)
        actual = graph().onnx_converter_sum_axis(source, 1)
    assert actual.tobytes() == expected.tobytes()


def test_every_half_bit_pattern():
    source = np.arange(65536, dtype=np.uint16).view(np.float16).reshape(-1, 1)
    expected, ew = result(lambda: np.sum(source, 1))
    actual, aw = result(lambda: graph().onnx_converter_sum_axis(source, 1))
    assert aw == ew
    assert actual.tobytes() == expected.tobytes()
