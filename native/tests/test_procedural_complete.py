"""Create Noise C pipeline against the frozen original node and generators."""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from test_generation import ORIGINAL_CREATE_NOISE as ORIGINAL

from nodes.impl import native_procedural as native
from nodes.utils.seed import Seed

API = native._api()  # exercise the raw C boundary

TILINGS = [(bool(i & 1), bool(i & 2), bool(i & 4)) for i in range(8)]


def reference(
    width,
    height,
    seed,
    method,
    scale,
    brightness,
    horizontal,
    vertical,
    spherical,
    layers,
    scale_ratio,
    brightness_ratio,
    increment,
):
    return ORIGINAL.create_noise_node(
        width,
        height,
        Seed(seed),
        list(ORIGINAL.NoiseMethod)[method],
        scale,
        brightness * 100,
        horizontal,
        vertical,
        spherical,
        ORIGINAL.FractalMethod.PINK_NOISE if layers else ORIGINAL.FractalMethod.NONE,
        layers,
        scale_ratio,
        brightness_ratio,
        increment,
        1.5,
    )


def exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    dtype = np.uint32 if actual.dtype == np.float32 else np.uint64
    mask = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[mask].view(dtype), expected[mask].view(dtype))


@pytest.mark.parametrize("method", range(3))
@pytest.mark.parametrize("tiling", TILINGS)
@pytest.mark.parametrize("seed", [0, 714, 2**32 - 1])
@pytest.mark.parametrize("shape", [(1, 1), (1, 37), (41, 1), (17, 23)])
@pytest.mark.parametrize(
    "layers,increment", [(0, False), (2, True), (7, False), (20, True)]
)
def test_all_node_paths(method, tiling, seed, shape, layers, increment):
    h, w = shape
    horizontal, vertical, spherical = tiling
    args = (
        w,
        h,
        seed,
        method,
        17.3,
        0.87,
        horizontal,
        vertical,
        spherical,
        layers,
        1.7,
        2.3,
        increment,
    )
    exact(native.create(*args), reference(*args))


@pytest.mark.parametrize("method", range(3))
@pytest.mark.parametrize("tiling", TILINGS)
@pytest.mark.parametrize(
    "scale,brightness,sr,br",
    [
        (1, 0, 1, 1),
        (1, 1, 16, 1.01),
        (1e100, 0.5, 1e10, 1e16),
        (1, 0.001, 1e16, 1),
        (17.3, 0.87, 2, 2),
    ],
)
def test_extreme_scales_and_brightness(method, tiling, scale, brightness, sr, br):
    horizontal, vertical, spherical = tiling
    args = (
        19,
        13,
        2**32 - 1,
        method,
        scale,
        brightness,
        horizontal,
        vertical,
        spherical,
        20,
        sr,
        br,
        True,
    )
    with np.errstate(all="ignore"):
        exact(native.create(*args), reference(*args))


@pytest.mark.parametrize("method", range(3))
@pytest.mark.parametrize("tiling", TILINGS[:4])
def test_parallel_and_concurrent_images(method, tiling):
    horizontal, vertical, spherical = tiling
    args = (
        137,
        71,
        714,
        method,
        1,
        1,
        horizontal,
        vertical,
        spherical,
        4,
        2.0,
        2.0,
        True,
    )
    expected = reference(*args)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for actual in pool.map(lambda _: native.create(*args), range(8)):
            exact(actual, expected)


def test_no_python_array_processing(monkeypatch):
    args = (31, 29, 0, 2, 1, 1, True, True, False, 4, 2.0, 2.0, True)
    expected = reference(*args)

    def forbidden(*args, **kwargs):
        raise AssertionError("Create Noise used Python array processing")

    for name in ["arange", "cos", "sin", "power", "sum", "concatenate", "zeros"]:
        monkeypatch.setattr(np, name, forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    actual = native.create(*args)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "index,value",
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 2**64 - 1),
        (4, -1),
        (4, 3),
        (5, 0),
        (5, float("nan")),
        (6, -0.1),
        (6, 1.1),
        (7, 2),
        (8, 2),
        (9, 2),
        (10, 1),
        (10, 21),
        (11, 0),
        (12, 0),
        (13, 2),
    ],
)
def test_checked_abi_unchanged_output(index, value):
    out = np.full((5, 7), 123.0, dtype=np.float64)
    args = [out.ctypes.data, 5, 7, 0, 2, 1.0, 1.0, 0, 0, 0, 0, 1.0, 1.0, 1]
    args[index] = value
    before = out.tobytes()
    assert API.cn_create_procedural(*args) == 1
    assert out.tobytes() == before


def test_checked_abi_alignment_and_overflow():
    out = np.full(128, 0xA5, dtype=np.uint8)
    args = [out.ctypes.data + 1, 1, 1, 0, 2, 1.0, 1.0, 0, 0, 0, 0, 1.0, 1.0, 1]
    assert API.cn_create_procedural(*args) == 1
    args[0] = out.ctypes.data
    args[1] = ct.c_size_t(-1).value
    args[2] = 2
    assert API.cn_create_procedural(*args) == 2
    args[:3] = [ct.c_size_t(-8).value, 1, 2]
    assert API.cn_create_procedural(*args) == 2
    assert np.all(out == 0xA5)


def test_scalar_power_overflow():
    with pytest.raises(OverflowError):
        native.create(1, 1, 0, 2, 1, 1, False, False, False, 20, 1e100, 2, True)


@pytest.mark.parametrize("method", range(3))
@pytest.mark.parametrize("tiling", TILINGS[:4])
@pytest.mark.parametrize("offset", [-512, -1, 0, 1, 512])
def test_windows_integer_lattice_boundaries(method, tiling, offset):
    ratio = ((2**31 - 1) + offset) / 7
    horizontal, vertical, spherical = tiling
    args = (
        19,
        13,
        714,
        method,
        1,
        1,
        horizontal,
        vertical,
        spherical,
        2,
        ratio,
        1,
        True,
    )
    with np.errstate(all="ignore"):
        exact(native.create(*args), reference(*args))
