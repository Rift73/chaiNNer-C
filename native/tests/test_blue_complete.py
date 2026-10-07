"""Void/cluster ranking must preserve every pixel rank, without tolerances."""

from __future__ import annotations

import ctypes as ct
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from nodes.impl import native_blue
from nodes.impl.native import check, lib
from nodes.impl.noise_functions import blue

path = Path(__file__).with_name("reference_generation") / "blue.py"
spec = importlib.util.spec_from_file_location("original_blue_noise", path)
assert spec and spec.loader
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def generate(shape, sigma, seed, fraction=0.1):
    return native_blue.create(shape, sigma, fraction, seed)


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 7), (7, 1), (3, 3), (4, 8), (7, 11), (17, 19)]
)
@pytest.mark.parametrize("sigma", [0, 0.5, 1, 1.5, 3, 100, (1.1, 2.7)])
@pytest.mark.parametrize("seed", [0, 1, 274])
def test_all_rankings_exact(shape, sigma, seed):
    expected = original.create_blue_noise(shape, sigma, seed=seed)
    actual = generate(shape, sigma, seed)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.sort(actual, axis=None), np.arange(actual.size))


@pytest.mark.parametrize("fraction", [-1, 0, 0.01, 0.25, 0.5, 1, 5])
@pytest.mark.parametrize("shape", [(1, 1), (3, 4), (7, 9)])
def test_initial_fraction_clamping(shape, fraction):
    expected = original.create_blue_noise(shape, 1.5, fraction, 32)
    actual = generate(shape, 1.5, 32, fraction)
    np.testing.assert_array_equal(actual, expected)


def test_concurrent_independent_rankings():
    expected = [original.create_blue_noise((13, 17), 1.7, seed=i) for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda i: generate((13, 17), 1.7, i), range(4)))
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, e)


def test_invalid_permutation_leaves_output_unchanged():
    generate((1, 1), 1.5, 0)
    source = np.arange(6, dtype=np.int64)
    out = np.full((2, 3), 71, dtype=np.int32)
    f = lib().cn_blue_from_permutation
    for invalid in (-1, 6, 1):
        source[0] = invalid
        assert f(source.ctypes.data, out.ctypes.data, 2, 3, 1.5, 1.5, 0.1) == 1
        np.testing.assert_array_equal(out, 71)


@pytest.mark.parametrize("shape", [(1, 1), (1, 127), (17, 19), (31, 37), (47, 53)])
@pytest.mark.parametrize("sigma", [1, 1.001, 1.5, 2.731, 7.3, 19.71, 50, 100])
@pytest.mark.parametrize("density", [0, 0.1, 0.5, 0.9, 1])
def test_full_periodic_fields_and_extrema(shape, sigma, density):
    pattern = np.random.default_rng(170).random(shape) < density
    effective = ~pattern if np.count_nonzero(pattern) * 2 >= pattern.size else pattern
    expected = np.fft.ifftn(
        ndimage.fourier_gaussian(np.fft.fftn(np.where(effective, 1.0, 0.0)), sigma)
    ).real.copy()
    out = np.empty(shape, np.float64)
    f = lib().cn_blue_filtered
    f.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_double,
        ct.c_double,
    ]
    f.restype = ct.c_int
    check(f(pattern.ctypes.data, out.ctypes.data, *shape, sigma, sigma))
    np.testing.assert_array_equal(out.view(np.uint64), expected.view(np.uint64))
    assert blue.find_largest_void(pattern, sigma) == original.find_largest_void(
        pattern, sigma
    )
    assert blue.find_tightest_cluster(pattern, sigma) == original.find_tightest_cluster(
        pattern, sigma
    )


@pytest.mark.parametrize("count", [1, 3, 255, 256, 65535, 65536, 65537, 262147])
def test_node_normalization_promotion(count):
    ranks = np.arange(count, dtype=np.int32).reshape(1, count)
    with np.errstate(all="ignore"):
        expected = ranks.astype(np.float32) / (count - 1)
        actual = native_blue.normalize(ranks)
    assert expected.dtype == actual.dtype
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    mask = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[mask].view(np.uint8), expected[mask].view(np.uint8)
    )


def test_node_path_does_not_call_numpy_rng_fft_or_scipy(monkeypatch):
    expected = original.create_blue_noise((17, 19), 1.5, seed=32)
    expected_void = original.find_largest_void(expected < 32, 1.5)
    expected_cluster = original.find_tightest_cluster(expected < 32, 1.5)

    def forbidden(*args, **kwargs):
        raise AssertionError("Retained RNG/FFT/Gaussian/selection pixel work invoked")

    with monkeypatch.context() as m:
        m.setattr(np.random, "default_rng", forbidden)
        m.setattr(np.fft, "fftn", forbidden)
        m.setattr(np.fft, "ifftn", forbidden)
        m.setattr(ndimage, "fourier_gaussian", forbidden)
        m.setattr(np, "argmin", forbidden)
        m.setattr(np, "argmax", forbidden)
        actual = blue.create_blue_noise((17, 19), 1.5, seed=32)
        void = blue.find_largest_void(expected < 32, 1.5)
        cluster = blue.find_tightest_cluster(expected < 32, 1.5)
    np.testing.assert_array_equal(actual, expected)
    assert void == expected_void and cluster == expected_cluster
