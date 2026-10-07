"""Exact CPU tests for integer/separable box and adaptive-mean kernels."""

from __future__ import annotations

import ctypes as ct
import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest
from test_convolution_ops import INSTALLED_BOX, MODULES
from test_normalized_outputs import RAW_BOX, assert_enforced_equal

from nodes.impl import native_box, native_convolution
from nodes.impl.native import lib
from nodes.impl.native_color_ops import adaptive_threshold, quantize_u8

ORIGINAL_BLUR = cv2.blur
ORIGINAL_BOX = cv2.boxFilter
ORIGINAL_SEPARABLE = cv2.sepFilter2D


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    if actual.dtype == np.float32:
        valid = ~np.isnan(expected)
        np.testing.assert_array_equal(
            actual[valid].view(np.uint32), expected[valid].view(np.uint32)
        )


def original_mean(image, radius):
    return ORIGINAL_BOX(
        image,
        -1,
        (radius * 2 + 1,) * 2,
        normalize=True,
        borderType=cv2.BORDER_REPLICATE | cv2.BORDER_ISOLATED,
    )


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 19), (17, 1), (3, 5), (13, 31), (19, 33)]
)
@pytest.mark.parametrize("radius", [0, 1, 2, 7, 8, 15, 127, 1447, 1448, 23169])
@pytest.mark.parametrize("pattern", ["random", "zero", "white", "checker"])
def test_mean_rounding_regimes(shape, radius, pattern):
    source = np.random.default_rng(749).integers(0, 256, shape, dtype=np.uint8)
    if pattern == "zero":
        source.fill(0)
    elif pattern == "white":
        source.fill(255)
    elif pattern == "checker":
        source[:] = (np.indices(shape).sum(axis=0) % 2) * 255
    exact(native_box.mean_u8(source, radius), original_mean(source, radius))


@pytest.mark.parametrize("radius", [1, 2, 7, 8, 15, 127, 1447, 1448])
@pytest.mark.parametrize("maximum", [0, 17.5, 127.5, 255])
@pytest.mark.parametrize("kind", [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV])
@pytest.mark.parametrize("delta", [-255, -1, 0, 1, 255])
def test_adaptive_mean_result(radius, maximum, kind, delta):
    source = np.random.default_rng(92).random((17, 35), dtype=np.float32)
    expected = cv2.adaptiveThreshold(
        quantize_u8(source),
        maximum,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        kind,
        radius * 2 + 1,
        delta,
    )
    exact(adaptive_threshold(source, kind, maximum, 0, radius, delta), expected)


@pytest.mark.parametrize(
    "radii",
    [(0, 1), (1, 0), (0, 3), (3, 0), (1, 1), (2, 3), (3, 5), (19, 15), (1000, 1000)],
)
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1), (7, 9), (13, 35)])
@pytest.mark.parametrize("channels", [1, 3, 4, 7])
@pytest.mark.parametrize("pattern", ["nan", "inf", "mixed", "negative_zero"])
def test_integer_nonfinite_and_signed_zero(radii, shape, channels, pattern):
    source = np.random.default_rng(29).random((*shape, channels), dtype=np.float32)
    if pattern == "nan":
        source.ravel()[::7] = np.nan
    elif pattern == "inf":
        source.ravel()[::11] = np.inf
    elif pattern == "mixed":
        source[:] = np.random.default_rng(39).choice(
            np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 0.25], np.float32),
            source.shape,
        )
    else:
        source.fill(-0.0)
    exact(
        native_convolution.box_blur(source, *radii),
        ORIGINAL_BLUR(source, tuple(r * 2 + 1 for r in radii)),
    )


@pytest.mark.parametrize("layout", ["reverse", "transpose", "readonly", "broadcast"])
def test_mean_foreign_buffers(layout):
    source = np.random.default_rng(835).integers(0, 256, (19, 23), dtype=np.uint8)
    if layout == "reverse":
        source = source[::-1, ::-1]
    elif layout == "transpose":
        source = source.T
    elif layout == "readonly":
        source.setflags(write=False)
    else:
        source = np.broadcast_to(source[:1, :1], source.shape)
    before = source.copy()
    result = native_box.mean_u8(source, 19)
    exact(result, original_mean(source, 19))
    exact(source, before)
    assert not np.shares_memory(result, source)
    assert result.flags.c_contiguous and result.flags.writeable


def test_paths_do_not_call_opencv_kernels(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Integer box and adaptive mean must compute in C")

    monkeypatch.setattr(cv2, "blur", forbidden)
    monkeypatch.setattr(cv2, "boxFilter", forbidden)
    source = np.random.default_rng(47).random((13, 17), dtype=np.float32)
    source[1, 2] = np.nan
    exact(native_convolution.box_blur(source, 3, 0), ORIGINAL_BLUR(source, (7, 1)))
    u8 = quantize_u8(source)
    exact(native_box.mean_u8(u8, 9), original_mean(u8, 9))


def test_checked_mean_abi():
    api = lib()
    source = np.zeros((3, 7), np.uint8)
    output = np.empty_like(source)
    pointer = ct.POINTER(ct.c_uint8)
    a, b = source.ctypes.data_as(pointer), output.ctypes.data_as(pointer)
    assert api.cn_box_mean_u8(None, b, 3, 7, 1) == 1
    assert api.cn_box_mean_u8(a, None, 3, 7, 1) == 1
    assert api.cn_box_mean_u8(a, a, 3, 7, 1) == 1
    shifted = ct.cast(source.ctypes.data + 1, pointer)
    assert api.cn_box_mean_u8(a, shifted, 3, 7, 1) == 1
    assert api.cn_box_mean_u8(a, b, ct.c_size_t(-1).value, 7, 1) == 1


def test_concurrent_mean_calls_are_exact_and_immutable():
    sources = [
        np.random.default_rng(i).integers(0, 256, (79, 83), dtype=np.uint8)
        for i in range(8)
    ]
    expected = [original_mean(source, 17) for source in sources]
    before = [source.copy() for source in sources]

    def run(i):
        exact(native_box.mean_u8(sources[i % 8], 17), expected[i % 8])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(32)))
    for actual, snapshot in zip(sources, before, strict=True):
        exact(actual, snapshot)


@pytest.mark.parametrize("radius", [23170, 32767, 32768, 1000000, (2**31 - 2) // 2])
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (7, 1), (2, 3), (3, 9)])
def test_wide_area_exact_integer_definition(radius, shape):
    image = np.random.default_rng(83).integers(0, 256, shape, dtype=np.uint8)
    height, width = shape
    rx, ry = (radius if width > 1 else 0), (radius if height > 1 else 0)
    area = (rx * 2 + 1) * (ry * 2 + 1)
    expected = np.empty_like(image)
    for y in range(height):
        for x in range(width):
            total = 0
            for yy in range(height):
                wy = int(y - ry <= yy <= y + ry)
                if yy == 0:
                    wy += max(0, ry - y)
                if yy == height - 1:
                    wy += max(0, y + ry - (height - 1))
                for xx in range(width):
                    wx = int(x - rx <= xx <= x + rx)
                    if xx == 0:
                        wx += max(0, rx - x)
                    if xx == width - 1:
                        wx += max(0, x + rx - (width - 1))
                    total += int(image[yy, xx]) * wx * wy
            expected[y, x] = (total + area // 2) // area
    exact(native_box.mean_u8(image, radius), expected)


def test_original_signed_area_overflow_is_corrected():
    image = np.full((2, 3), 173, np.uint8)
    # Independent evidence: an average of constant173 must remain173.
    # The installed4.8.0 implementation corrupts it to0 after signed area overflow.
    np.testing.assert_array_equal(original_mean(image, 23170), np.zeros_like(image))
    exact(native_box.mean_u8(image, 23170), image)


def original_separable(image, rx, ry):
    def kernel(radius):
        values = np.ones(math.ceil(radius) * 2 + 1, np.float32)
        if radius % 1:
            values[0] *= radius % 1
            values[-1] *= radius % 1
        values /= np.sum(values)
        return values

    return ORIGINAL_SEPARABLE(
        image, -1, kernel(rx), kernel(ry), borderType=cv2.BORDER_REFLECT_101
    )


@pytest.fixture(params=[False, True])
def optimized(request):
    before = cv2.useOptimized()
    cv2.setUseOptimized(request.param)
    try:
        yield request.param
    finally:
        cv2.setUseOptimized(before)


@pytest.mark.parametrize(
    "radii",
    [
        (0, 0),
        (0.1, 0.2),
        (1.5, 3.2),
        (15, 1.2),
        (2.1, 0.4),
        (3.7, 69.2),
        (69.9, 69.9),
        (100.5, 20.2),
        (200, 24.4),
        (0, 1.7),
        (1.7, 0),
        (1e-50, 0.1),
    ],
)
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1), (13, 23), (15, 32)])
@pytest.mark.parametrize("channels", [1, 2, 3, 4, 7])
def test_separable_box_options(radii, shape, channels, optimized):
    image = np.random.default_rng(830).random((*shape, channels), dtype=np.float32)
    exact(native_box.separable_box(image, *radii), original_separable(image, *radii))


@pytest.mark.parametrize(
    "radii", [(0.1, 0.2), (1.5, 3.2), (0, 1.3), (1.3, 0), (3.5, 7.2), (100.5, 20.2)]
)
@pytest.mark.parametrize("channels", [1, 3, 4, 7])
@pytest.mark.parametrize("pattern", ["negative_zero", "mixed", "subnormal", "dynamic"])
def test_separable_extreme_values(radii, channels, pattern, optimized):
    image = np.random.default_rng(82).random((13, 17, channels), dtype=np.float32)
    if pattern == "negative_zero":
        image.fill(-0.0)
    elif pattern == "mixed":
        image[:] = np.random.default_rng(31).choice(
            np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 0.25], np.float32),
            image.shape,
        )
    elif pattern == "subnormal":
        image = (image - np.float32(0.5)) * np.float32(1e-38)
    else:
        image = np.ldexp(
            image - np.float32(0.5),
            np.random.default_rng(5).integers(-140, 127, image.shape, dtype=np.int32),
        )
    with np.errstate(invalid="ignore", over="ignore"):
        exact(
            native_box.separable_box(image, *radii), original_separable(image, *radii)
        )


def test_separable_never_calls_opencv_kernel(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Separable Box must compute in C")

    monkeypatch.setattr(cv2, "sepFilter2D", forbidden)
    image = np.random.default_rng(11).random((19, 23, 3), dtype=np.float32)
    exact(
        native_box.separable_box(image, 6.2, 3.3), original_separable(image, 6.2, 3.3)
    )


@pytest.mark.parametrize(
    "layout", ["reverse", "transpose", "readonly", "broadcast", "unaligned"]
)
@pytest.mark.parametrize("channels", [1, 3, 7])
def test_separable_foreign_buffers(layout, channels):
    image = np.random.default_rng(12).random((19, 23, channels), dtype=np.float32)
    if layout == "reverse":
        image = image[::-1, ::-1]
    elif layout == "transpose":
        image = image.swapaxes(0, 1)
    elif layout == "readonly":
        image.setflags(write=False)
    elif layout == "broadcast":
        image = np.broadcast_to(image[:1, :1], image.shape)
    else:
        foreign = np.ndarray(
            image.shape, np.float32, bytearray(image.nbytes + 1), offset=1
        )
        foreign[:] = image
        image = foreign
    before = image.copy()
    result = native_box.separable_box(image, 5.3, 1.7)
    exact(result, original_separable(image, 5.3, 1.7))
    exact(image, before)
    assert not np.shares_memory(result, image)
    assert result.flags.c_contiguous and result.flags.writeable


@pytest.mark.parametrize("radius", [-1, 1000.1, np.nan, np.inf, -np.inf])
def test_separable_invalid_radii(radius):
    source = np.zeros((3, 7), np.float32)
    with pytest.raises(ValueError):
        native_box.separable_box(source, radius, 1.5)
    with pytest.raises(ValueError):
        native_box.separable_box(source, 1.5, radius)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (3,), (2, 3, 0), (2, 3, 4, 5)])
def test_separable_invalid_shapes(shape):
    with pytest.raises(ValueError):
        native_box.separable_box(np.empty(shape, np.float32), 1.5, 1.5)


def test_checked_separable_abi():
    api = lib()
    source = np.zeros((3, 7), np.float32)
    output = np.empty_like(source)
    pointer = ct.POINTER(ct.c_float)
    a, b = source.ctypes.data_as(pointer), output.ctypes.data_as(pointer)
    assert api.cn_box_separable(None, b, 3, 7, 1, 1.5, 1.5, 4) == 1
    assert api.cn_box_separable(a, None, 3, 7, 1, 1.5, 1.5, 4) == 1
    assert api.cn_box_separable(a, a, 3, 7, 1, 1.5, 1.5, 4) == 1
    shifted = ct.cast(source.ctypes.data + 4, pointer)
    assert api.cn_box_separable(a, shifted, 3, 7, 1, 1.5, 1.5, 4) == 1
    unaligned = ct.cast(source.ctypes.data + 1, pointer)
    assert api.cn_box_separable(unaligned, b, 3, 7, 1, 1.5, 1.5, 4) == 1
    assert api.cn_box_separable(a, b, 3, 7, 1, 1.5, 1.5, 3) == 1
    assert api.cn_box_separable(a, b, 3, 7, 0, 1.5, 1.5, 4) == 1
    assert api.cn_box_separable(a, b, 2**31 - 1, 2**31 - 1, 512, 1.5, 1.5, 4) == 2


def test_concurrent_separable_calls_are_exact_and_immutable():
    sources = [
        np.random.default_rng(i).random((79, 83, 3), dtype=np.float32) for i in range(8)
    ]
    expected = [original_separable(source, 15.3, 6.2) for source in sources]
    before = [source.copy() for source in sources]

    def run(i):
        exact(native_box.separable_box(sources[i % 8], 15.3, 6.2), expected[i % 8])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(32)))
    for actual, snapshot in zip(sources, before, strict=True):
        exact(actual, snapshot)


@pytest.mark.parametrize("radii", [(73.2, 169.7), (77.1, 102.3)])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_large_fractional_box_preserves_ipp_pixel_ties(radii, channels, monkeypatch):
    source = np.full((31, 37, channels), 0.5, np.float32)
    expected = INSTALLED_BOX.box_blur_node(source, *radii)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Large Box owns its filter algorithm in C")

    monkeypatch.setattr(cv2, "filter2D", forbidden)
    # As the user receives it (ImageOutput.enforce of both returns); the ties and
    # the kernel parity on the helpers' raw result (RAW_BOX) (SP4b Task 3b2).
    assert_enforced_equal(
        MODULES["box_blur"][0].box_blur_node(source, *radii), expected
    )
    raw = RAW_BOX.box_blur_node(source, *radii)
    exact(raw, expected)
    np.testing.assert_array_equal(quantize_u8(raw), quantize_u8(expected))
