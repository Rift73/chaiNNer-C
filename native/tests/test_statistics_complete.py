"""NumPy 2.5.3 statistics parity including signed-zero selection and reduction."""

from __future__ import annotations

import ctypes as ct
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from nodes.impl.native import lib, ptr
from nodes.impl.native_analysis import mean, statistics
from nodes.impl.native_numpy_reduce import numpy_sum_block
from nodes.impl.native_transfer import contrast_bounds


def exact(actual, expected):
    a = np.asarray(actual, np.float64)
    b = np.asarray(expected, np.float64)
    np.testing.assert_array_equal(a, b)
    valid = ~np.isnan(b)
    np.testing.assert_array_equal(a[valid].view(np.uint64), b[valid].view(np.uint64))


def reference(image, quantile):
    with np.errstate(all="ignore"):
        return (
            np.min(image),
            np.max(image),
            np.mean(image),
            np.percentile(image, quantile),
        )


@pytest.mark.parametrize(
    "size",
    [
        1,
        2,
        3,
        4,
        7,
        8,
        15,
        16,
        17,
        31,
        32,
        33,
        127,
        128,
        129,
        255,
        256,
        257,
        8191,
        8192,
        8193,
        16385,
    ],
)
@pytest.mark.parametrize("quantile", [0, 0.01, 12.5, 25, 33.33, 50, 75, 99.99, 100])
@pytest.mark.parametrize(
    "kind", ["zero", "mixed_zero", "duplicates", "random", "nonfinite"]
)
def test_statistics_exact(size, quantile, kind):
    rng = np.random.default_rng(913 + size)
    image = rng.standard_normal((1, size)).astype(np.float32)
    if kind == "zero":
        image.fill(-0.0)
    elif kind == "mixed_zero":
        image = np.copysign(np.zeros_like(image), image)
    elif kind == "duplicates":
        image = rng.choice(np.array([-1, -0.0, 0, 1], np.float32), (1, size))
    elif kind == "nonfinite":
        image = rng.choice(
            np.array([-np.inf, -1, -0.0, 0, 1, np.inf], np.float32), (1, size)
        )
    saved = image.copy()
    exact(statistics(image, quantile), reference(image, quantile))
    np.testing.assert_array_equal(image.view(np.uint32), saved.view(np.uint32))


@pytest.mark.parametrize(
    "layout", ["C", "F", "transpose", "reverse", "stride", "readonly", "unaligned"]
)
@pytest.mark.parametrize("shape", [(7, 9), (17, 19), (91, 93)])
@pytest.mark.parametrize("quantile", [0, 12.5, 25, 50, 75, 100])
@pytest.mark.parametrize("kind", ["random", "mixed_zero", "sparse_zero", "nonfinite"])
def test_statistics_layouts(layout, shape, quantile, kind):
    rng = np.random.default_rng(291)
    image = rng.standard_normal(shape).astype(np.float32)
    if kind == "mixed_zero":
        image = np.copysign(np.zeros_like(image), image)
    elif kind == "sparse_zero":
        image = rng.choice(np.array([-0.0, 0, 1], np.float32), shape)
    elif kind == "nonfinite":
        image.flat[::17] = np.nan
        image.flat[::31] = np.inf
    if layout == "F":
        image = np.asfortranarray(image)
    elif layout == "transpose":
        image = image.T
    elif layout == "reverse":
        image = image[::-1, ::-1]
    elif layout == "stride":
        image = image[::2, ::2]
    elif layout == "readonly":
        image.flags.writeable = False
    elif layout == "unaligned":
        other = np.ndarray(shape, np.float32, bytearray(image.nbytes + 1), 1)
        other[:] = image
        image = other
    exact(statistics(image, quantile), reference(image, quantile))


def test_concurrent_statistics():
    rng = np.random.default_rng(77)
    images = [
        rng.choice(np.array([-1, -0.0, 0, 1], np.float32), (311, 313)) for _ in range(8)
    ]
    expected = [reference(image, 53.3) for image in images]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda image: statistics(image, 53.3), images))
    for a, b in zip(actual, expected, strict=True):
        exact(a, b)


def test_statistics_abi_validation():
    function = lib().cn_analysis_statistics
    image = np.ones((3, 5), np.float32)
    result = np.full(4, 987, np.float64)
    output = result.ctypes.data_as(ct.POINTER(ct.c_double))
    assert function(None, ptr(image), image.size, 50, 16, 0, 0, 1, output) == 1
    assert function(ptr(image), ptr(image), 0, 50, 16, 0, 0, 1, output) == 1
    assert function(ptr(image), ptr(image), image.size, 50, 3, 0, 0, 1, output) == 1
    assert function(ptr(image), ptr(image), image.size, 50, 8, 0, 0, 2, output) == 1
    assert (
        function(ptr(image), ptr(image), image.size, float("nan"), 16, 0, 0, 1, output)
        == 1
    )
    assert (
        function(ptr(image), ptr(image), ct.c_size_t(-1).value, 50, 16, 0, 0, 1, output)
        == 2
    )
    assert (
        function(
            result.ctypes.data_as(ct.POINTER(ct.c_float)),
            ptr(image),
            image.size,
            50,
            16,
            0,
            0,
            1,
            output,
        )
        == 1
    )
    np.testing.assert_array_equal(result, 987)


def _extreme_values(rng, kind, n):
    values = np.where(rng.random(n) < 0.5, np.float32(-0.0), np.float32(0.0))
    if kind == "zeros_half":
        values[rng.random(n) < 0.3] = 0.5
    elif kind == "zeros_one":
        values[rng.random(n) < 0.1] = 1.0
        values[rng.random(n) < 0.1] = -1.0
    elif kind == "nan_payloads":
        values = rng.random(n).astype(np.float32)
        nan, payload = rng.integers(0, n, 2)
        values[nan] = np.float32(np.nan)
        values.view(np.uint32)[payload] = 0xFFC00001
    return values


def _extreme_layout(values, layout):
    n = values.size
    if layout == "contiguous":
        return values.reshape(1, n)
    if layout == "reversed":  # a -4 byte inner stride: NumPy's scalar path
        return values.reshape(1, n)[:, ::-1]
    if layout == "plane":  # a channel plane, unbuffered and strided
        planes = np.full((1, n, 3), 7, np.float32)
        planes[:, :, 0] = values
        return planes[:, :, 0]
    if layout == "unaligned":  # a CAST operand, buffered in 8192 blocks
        other = np.ndarray((1, n), np.float32, bytearray(values.nbytes + 1), 1)
        other[:] = values
        return other
    width = 61  # rows of a wider image: buffered in whole-row blocks
    rows = -(-n // width)
    wide = np.full((rows, width + 3), 7, np.float32)
    wide[:, :width] = np.resize(values, (rows, width))
    return wide[:, :width]


@pytest.mark.parametrize("kind", ["zeros", "zeros_half", "zeros_one", "nan_payloads"])
@pytest.mark.parametrize(
    "layout", ["contiguous", "reversed", "plane", "unaligned", "block"]
)
def test_extreme_large_signed_zero_and_nan_payloads(kind, layout):
    """np.min/np.max bits, NaN payloads included, for n up to 20000."""
    rng = np.random.default_rng(4049)
    sizes = [1, 2, 7, 8, 9, 63, 64, 65, 8191, 8192, 8193, 16385, 19999]
    for n in sizes + rng.integers(1, 20000, 16).tolist():
        image = _extreme_layout(_extreme_values(rng, kind, n), layout)
        with np.errstate(all="ignore"):
            expected = np.array([np.min(image), np.max(image)], np.float32)
        actual = np.array(statistics(image, 50)[:2], np.float32)
        np.testing.assert_array_equal(
            actual.view(np.uint32), expected.view(np.uint32), err_msg=f"n={n}"
        )


@pytest.mark.parametrize("disabled", ["", "X86_V3"])
def test_lower_numpy_simd_dispatch(disabled):
    # NumPy 2.5's MSVC wheels dispatch X86_V3 only; disabling it runs X86_V2.
    environment = os.environ.copy()
    environment["NPY_DISABLE_CPU_FEATURES"] = disabled
    script = """
import sys
sys.path.insert(0, 'backend/src')
import numpy as np
from nodes.impl.native_analysis import statistics
rng = np.random.default_rng(333)
for n in [3, 8, 9, 16, 17, 32, 33, 64, 65, 129, 8192, 8193, 16385]:
    a = rng.choice(np.array([-1, -0.0, 0, 1], np.float32), (1, n))
    for q in (0, 25, 50, 75, 100):
        expected = np.array([np.min(a), np.max(a), np.mean(a), np.percentile(a, q)], np.float64)
        actual = np.array(statistics(a, q), np.float64)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["roll", "organ_pipe", "sawtooth", "alternating"])
@pytest.mark.parametrize("quantile", [0.01, 25, 50, 75, 99.99])
def test_adversarial_introselect(kind, quantile):
    size = 65539
    image = np.arange(size, dtype=np.float32) - size // 2
    if kind == "roll":
        image = np.roll(image, size // 2)
    elif kind == "organ_pipe":
        image = np.minimum(image, image[::-1])
    elif kind == "sawtooth":
        image %= 127
    else:
        image[::2] = -image[::2]
    image[::3] = -0.0
    image[::5] = 0.0
    image = image.reshape(1, -1)
    exact(statistics(image, quantile), reference(image, quantile))


# Quiet and signaling NaNs of both signs, with and without payload bits.
NAN_PAYLOADS = np.array(
    [0x7FC00000, 0xFFC00000, 0x7FC12345, 0xFFE00001, 0x7F800001, 0xFFA5A5A5],
    np.uint32,
)


def _bits(value):
    return int(np.float64(value).view(np.uint64))


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 17, 129, 1000, 8193, 40000])
@pytest.mark.parametrize("layout", ["row", "transpose"])
def test_percentile_nan_payloads(size, layout):
    """np.percentile returns arr[-1] after its NaN-last partition: whichever NaN the
    selection moves there, payload included, on the weak and float64-q lerps."""
    rng = np.random.default_rng(6907 + size)
    seen = set()
    for trial in range(6):
        values = rng.standard_normal(size).astype(np.float32)
        if trial % 2:
            values = rng.choice(np.array([-1, -0.0, 0, 1], np.float32), size)
        count = int(rng.integers(1, min(size, 12) + 1))
        where = rng.choice(size, count, replace=False)
        values.view(np.uint32)[where] = rng.choice(NAN_PAYLOADS, count)
        image = values.reshape(1, size)
        if layout == "transpose":
            image = np.resize(values, (2, size)).T
        for q in (0.0, 0.01, 12.5, 33.33, 50.0, 75.0, 99.99, 100.0):
            for percentile in (q, np.float64(q)):
                with np.errstate(all="ignore"):
                    expected = np.percentile(image, percentile)
                    pair = (expected, np.percentile(image, 100 - percentile))
                assert _bits(statistics(image, percentile)[3]) == _bits(expected)
                seen.add(_bits(expected))
                if q <= 50:
                    actual = contrast_bounds(image, percentile)
                    assert [_bits(v) for v in actual] == [_bits(v) for v in pair]
    # The data must reach NaNs other than the default one for the test to bite.
    assert len(seen) > 1


def _period_view(name, kind, rng):
    """R2 I-1's two views, whose buffered reduce restarts its blocks at every outer
    row: Split Transparency's RGB of an RGBA image cropped left and right, strides
    (16W, 16, 4), and a flipped two-channel view, strides (-16W, 16, 4)."""
    shape = (5, 4000, 4) if name == "rgb_crop" else (3, 9000, 4)
    base = rng.random(shape, dtype=np.float32)
    if kind in ("zeros_min", "zeros_max"):
        base = np.where(rng.random(shape) < 0.5, np.float32(-0.0), np.float32(0.0))
        base[rng.random(shape) < 0.3] = 0.5 if kind == "zeros_min" else -0.5
    view = base[:, :, :3][:, 10:3990] if name == "rgb_crop" else base[::-1, :, :2]
    if kind.startswith("nan"):
        count = 1 if kind == "nan_one" else 3
        where = rng.choice(view.size, count, replace=False)
        for index, payload in zip(where, rng.choice(NAN_PAYLOADS, count), strict=True):
            view[np.unravel_index(index, view.shape)] = payload.view(np.float32)
    return view


@pytest.mark.parametrize("name", ["rgb_crop", "flipped"])
@pytest.mark.parametrize(
    "kind", ["random", "zeros_min", "zeros_max", "nan_one", "nan_many", "nan_tail"]
)
def test_reduce_restarts_at_every_outer_row(name, kind):
    """np.mean, np.min and np.max bits (NaN payloads and the sign of zero included)
    through Image Statistics, Image Metrics' mean and Stretch Contrast."""
    rng = np.random.default_rng(1949)
    expected_blocks = (8190, 11940) if name == "rgb_crop" else (8192, 18000)
    trials = 40 if kind.startswith("zeros") else 8
    for trial in range(trials):
        view = _period_view(name, "random" if kind == "nan_tail" else kind, rng)
        if kind == "nan_tail":
            # The last call's scalar tail (its last 6 elements on the restarted
            # blocks), where the NaN keeps its payload; a back-to-back cut would
            # put it in that call's vector part, giving the canonical NaN.
            view[-1, -1, 0] = NAN_PAYLOADS[2 + trial % 4].view(np.float32)
        assert numpy_sum_block(view) == expected_blocks
        with np.errstate(all="ignore"):
            expected = [np.min(view), np.max(view), np.mean(view)]
        actual = [*statistics(view, 50.0)[:3]]
        message = f"{name} {kind} trial {trial}"
        # Which of several distinct NaNs a sum returns follows the compilers'
        # operand order, not the blocks (reported separately): nan_many checks
        # min and max only.
        checked = 2 if kind == "nan_many" else 3
        assert [_bits(v) for v in actual[:checked]] == [
            _bits(v) for v in expected[:checked]
        ], message
        if checked == 3:
            assert _bits(mean(view)) == _bits(expected[2]), message
        bounds = contrast_bounds(view)
        assert [_bits(v) for v in bounds] == [_bits(v) for v in expected[:2]], message
        if kind == "nan_tail" and name == "rgb_crop":
            assert _bits(expected[0]) != _bits(np.float32(np.nan)), message
