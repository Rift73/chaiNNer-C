"""Compare fused transfer against frozen NumPy/BLAS implementations on CPU."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

from nodes.impl import native_transfer as native
from nodes.impl.color_transfer.linear_histogram import linear_histogram_transfer
from nodes.impl.color_transfer.principal_color import principal_color_transfer

REFERENCE = Path(__file__).with_name("reference_transfer")
ALGORITHMS = [
    (
        linear_histogram_transfer,
        run_path(str(REFERENCE / "linear_histogram.py"))["linear_histogram_transfer"],
    ),
    (
        principal_color_transfer,
        run_path(str(REFERENCE / "principal_color.py"))["principal_color_transfer"],
    ),
]


def inputs(dtype, shape, layout, seed):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-0.2, 1.2, (*shape, 3)).astype(dtype)
    if layout == "reverse":
        a = a[::-1, ::-1]
    elif layout == "fortran":
        a = np.asfortranarray(a)
    elif layout == "unaligned":
        storage = np.empty(a.nbytes + 1, dtype=np.uint8)
        unaligned = np.ndarray(a.shape, dtype=dtype, buffer=storage, offset=1)
        unaligned[:] = a
        a = unaligned
    elif layout == "readonly":
        a.setflags(write=False)
    b = rng.uniform(0.1, 0.9, (11, 13, 3)).astype(dtype)
    return a, b, rng.random(shape) > 0.25, rng.random((11, 13)) > 0.25


@pytest.mark.parametrize("functions", ALGORITHMS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(3, 5), (17, 23), (128, 129)])
@pytest.mark.parametrize(
    "layout", ["plain", "reverse", "fortran", "unaligned", "readonly"]
)
@pytest.mark.parametrize("seed", [41, 592, 1739])
def test_complete_transfer(functions, dtype, shape, layout, seed):
    actual_fn, original_fn = functions
    args = inputs(dtype, shape, layout, seed)
    before = [a.copy() for a in args]
    actual = actual_fn(*args)
    expected = original_fn(*args)
    # NumPy 2 np.linalg.eig always returns complex128, so the transfer is complex.
    assert actual.dtype == expected.dtype == np.complex128
    # The completed pipeline preserves the original public BLAS call ordering.
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.rint(actual * 255), np.rint(expected * 255))
    for prior, value in zip(before, args, strict=True):
        np.testing.assert_array_equal(prior, value)


@pytest.mark.parametrize("functions", ALGORITHMS)
@pytest.mark.parametrize(
    "mode", ["constant", "rank_one", "identical", "empty", "single"]
)
def test_singular_or_exception_contract(functions, mode):
    actual_fn, original_fn = functions
    a, b, mask, ref_mask = inputs(np.float32, (5, 7), "plain", 349)
    if mode == "constant":
        a[:] = 0.25
        b[:] = 0.75
    elif mode == "rank_one":
        a[:, :, 1:] = a[:, :, :1]
        b[:, :, 1:] = b[:, :, :1]
    elif mode == "identical":
        b, ref_mask = a.copy(), mask.copy()
    elif mode == "empty":
        mask[:] = False
    else:
        mask[:] = False
        mask[0, 0] = True
    args = (a, b, mask, ref_mask)
    with np.errstate(all="ignore"):
        try:
            expected = original_fn(*args)
        except (ValueError, np.linalg.LinAlgError) as error:
            with pytest.raises(type(error)):
                actual_fn(*args)
        else:
            actual = actual_fn(*args)
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("count", [1, 2, 7, 129, 8193])
@pytest.mark.parametrize(
    "order", ["random", "ascending", "descending", "constant", "quantized"]
)
@pytest.mark.parametrize("percentile", [None, 0, 1.5, 15.25, 49.99, 50])
def test_contrast_bounds(count, order, percentile):
    rng = np.random.default_rng(702)
    values = rng.random((1, count), dtype=np.float32)
    if order in ("ascending", "descending"):
        values.sort()
        if order == "descending":
            values = values[:, ::-1]
    elif order == "constant":
        values[:] = 0.57
    elif order == "quantized":
        values = np.rint(values * 9) / 9
    prior = values.copy()
    if percentile is None:
        expected = (float(values.min()), float(values.max()))
    else:
        expected = tuple(
            float(np.percentile(values, q)) for q in (percentile, 100 - percentile)
        )
    assert native.contrast_bounds(values, percentile) == expected
    np.testing.assert_array_equal(values, prior)


@pytest.mark.parametrize("percentile", [None, 0, 12.5, 50])
@pytest.mark.parametrize("special", [0.0, -0.0, np.inf, -np.inf, np.nan])
def test_contrast_exceptional_values(percentile, special):
    values = np.array([0, special, 1, 0.5], np.float32).reshape(2, 2)
    with np.errstate(all="ignore"):
        expected = (
            (float(np.min(values)), float(np.max(values)))
            if percentile is None
            else tuple(
                float(np.percentile(values, q)) for q in (percentile, 100 - percentile)
            )
        )
        result = native.contrast_bounds(values, percentile)
    np.testing.assert_array_equal(result, expected)
    for actual, original in zip(result, expected, strict=True):
        if actual == 0:
            assert np.signbit(actual) == np.signbit(original)


def test_contrast_concurrent_and_invalid():
    values = np.random.default_rng(548).random((197, 269, 3), dtype=np.float32)
    expected = native.contrast_bounds(values, 1.75)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(
            value == expected
            for value in pool.map(
                lambda _: native.contrast_bounds(values, 1.75), range(12)
            )
        )
    for q in (-1, 51, np.nan):
        with pytest.raises(ValueError):
            native.contrast_bounds(values, q)
    with pytest.raises(ValueError):
        native.contrast_bounds(values[:0])
    with pytest.raises(IndexError):
        native.contrast_bounds(values[:0], 1.5)
