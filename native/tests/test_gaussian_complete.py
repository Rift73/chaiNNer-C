"""Exact installed OpenCV kernel and separable Gaussian contracts."""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest
from reference_gaussian_image_utils import fast_gaussian_blur as original_fast_blur

from nodes.impl import native_gaussian as native
from nodes.impl.image_utils import fast_gaussian_blur

API = native._api()  # verify the checked C ABI


def exact(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    mask = ~np.isnan(expected)
    integer = np.uint64 if actual.dtype == np.float64 else np.uint32
    np.testing.assert_array_equal(
        actual[mask].view(integer), expected[mask].view(integer)
    )


@pytest.mark.parametrize(
    "size", [1, 2, 3, 5, 7, 9, 11, 17, 31, 64, 101, 999, 2001, 8001]
)
@pytest.mark.parametrize(
    "sigma", [-1, 0, 0.01, 0.1, 0.333, 1, 1.5, 2.3, 6.7, 17.1, 125, 1000]
)
@pytest.mark.parametrize("double", [False, True])
def test_kernel(size, sigma, double):
    expected = cv2.getGaussianKernel(
        size, sigma, cv2.CV_64F if double else cv2.CV_32F
    ).ravel()
    exact(native.kernel(size, sigma, double=double), expected)


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 17), (17, 1), (7, 7), (13, 19), (17, 31), (31, 37)]
)
@pytest.mark.parametrize("channels", [0, 1, 2, 3, 4, 7])
@pytest.mark.parametrize(
    "sigma", [(0.1, 0.1), (0.3, 0.3), (0.5, 0.7), (1.5, 1.5), (2.3, 3.1), (17.1, 13.3)]
)
@pytest.mark.parametrize(
    "border", [cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_REFLECT_101]
)
def test_gaussian(shape, channels, sigma, border):
    shape = (*shape, channels) if channels else shape
    image = np.random.default_rng(214).uniform(-0.5, 1.5, shape).astype(np.float32)
    before = image.copy()
    exact(
        native.gaussian(image, *sigma, border=border),
        cv2.GaussianBlur(image, (0, 0), sigma[0], sigmaY=sigma[1], borderType=border),
    )
    exact(image, before)


@pytest.mark.parametrize("width", [1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33])
@pytest.mark.parametrize("channels", [0, 3, 4])
@pytest.mark.parametrize("size", [(1, 1), (3, 3), (5, 5), (7, 11), (21, 21)])
@pytest.mark.parametrize("optimized", [False, True])
def test_nonfinite_and_signed_zero(width, channels, size, optimized):
    shape = (7, width, channels) if channels else (7, width)
    original = cv2.useOptimized()
    cv2.setUseOptimized(optimized)
    try:
        image = np.resize(
            np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 1.0, -1.0], np.float32), shape
        )
        with np.errstate(all="ignore"):
            exact(
                native.gaussian(image, 0, size=size, border=1),
                cv2.GaussianBlur(image, size, 0, borderType=1),
            )
    finally:
        cv2.setUseOptimized(original)


@pytest.mark.parametrize("layout", ["strided", "transpose", "readonly", "unaligned"])
def test_layouts(layout):
    image = np.random.default_rng(12).random((17, 31, 4), dtype=np.float32)
    if layout == "strided":
        image = image[::-2, ::-1]
    elif layout == "transpose":
        image = image.transpose(1, 0, 2)
    elif layout == "readonly":
        image.setflags(write=False)
    else:
        data = np.empty(image.nbytes + 1, np.uint8)
        view = np.ndarray(image.shape, np.float32, buffer=data, offset=1)
        view[:] = image
        image = view
    before = image.copy()
    exact(
        native.gaussian(image, 1.5), cv2.GaussianBlur(image, (0, 0), 1.5, borderType=2)
    )
    exact(image, before)


def test_concurrency():
    image = np.random.default_rng(72).random((257, 263, 3), dtype=np.float32)
    expected = cv2.GaussianBlur(image, (0, 0), 2.3, sigmaY=4.7, borderType=2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda _: native.gaussian(image, 2.3, 4.7), range(8)):
            exact(result, expected)


@pytest.mark.parametrize(
    "index,value",
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 2),
        (6, 2),
        (7, -1),
        (7, float("nan")),
        (8, float("inf")),
        (9, 3),
        (10, 2),
    ],
)
def test_invalid_abi(index, value):
    src = np.ones((7, 11), np.float32)
    out = np.full_like(src, 123)
    args = [src.ctypes.data, out.ctypes.data, 7, 11, 1, 3, 3, 1.0, 1.0, 2, 8]
    args[index] = value
    assert API.cn_gaussian_f32(*args) == 1
    assert np.all(out == 123)


def test_abi_ranges_and_overlap():
    src = np.ones((7, 11), np.float32)
    out = np.full_like(src, 123)
    args = [src.ctypes.data, out.ctypes.data, 7, 11, 1, 3, 3, 1.0, 1.0, 2, 8]
    args[0] += 1
    assert API.cn_gaussian_f32(*args) == 1
    args[0] = out.ctypes.data + 4
    assert API.cn_gaussian_f32(*args) == 1
    args[0] = ct.c_size_t(-4).value
    assert API.cn_gaussian_f32(*args) == 2
    args[0] = src.ctypes.data
    args[2:5] = [2**31 - 1, 2**31 - 1, 512]
    assert API.cn_gaussian_f32(*args) == 2
    assert np.all(out == 123)


@pytest.mark.parametrize("radius", [1, 2, 3, 4, 7, 16, 31, 100, 1000, 23170, 23171])
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (3, 5)])
def test_adaptive_gaussian_mean(radius, shape):
    image = np.random.default_rng(801).integers(0, 256, shape, dtype=np.uint8)
    size = radius * 2 + 1
    expected = cv2.convertScaleAbs(
        cv2.GaussianBlur(
            image.astype(np.float32),
            (size, size),
            0,
            borderType=cv2.BORDER_REPLICATE | cv2.BORDER_ISOLATED,
        )
    )
    np.testing.assert_array_equal(native.mean_u8(image, radius), expected)


@pytest.mark.parametrize("size", [46341, 46343, 65535])
@pytest.mark.parametrize("sigma", [0, 1, 1000])
@pytest.mark.parametrize("double", [False, True])
def test_large_original_kernel_integer_semantics(size, sigma, double):
    with np.errstate(all="ignore"):
        expected = cv2.getGaussianKernel(
            size, sigma, cv2.CV_64F if double else cv2.CV_32F
        ).ravel()
        exact(native.kernel(size, sigma, double=double), expected)


def test_ssim_window():
    weights = cv2.getGaussianKernel(11, 1.5)
    exact(native.ssim_window(), np.outer(weights, weights.transpose()))


@pytest.mark.parametrize("shape", [(1, 1), (1, 37), (31, 1), (19, 31), (137, 149)])
@pytest.mark.parametrize("channels", [0, 1, 3, 4])
@pytest.mark.parametrize(
    "sigma",
    [
        (0, 1),
        (0.1, 0.3),
        (10.9, 17),
        (11, 11),
        (14.9, 15),
        (15, 19.9),
        (20, 24.9),
        (25, 30),
        (49.9, 51),
        (50, 100),
        (99.9, 100),
        (100, 200),
        (199.9, 301),
        (200, 999.9),
        (1000, 1000),
    ],
)
def test_fast_gaussian_pipeline(shape, channels, sigma):
    shape = (*shape, channels) if channels else shape
    image = np.random.default_rng(773).uniform(-0.5, 1.5, shape).astype(np.float32)
    expected = original_fast_blur(image, *sigma)
    exact(fast_gaussian_blur(image, *sigma), expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.float64])
@pytest.mark.parametrize("sigma", [1.5, 11, 24.9, 50, 200])
def test_general_helper_compatibility(dtype, sigma):
    image = (
        np.random.default_rng(73)
        .integers(0, 256, (17, 31, 3), dtype=np.uint8)
        .astype(dtype)
    )
    np.testing.assert_array_equal(
        fast_gaussian_blur(image, sigma), original_fast_blur(image, sigma)
    )
