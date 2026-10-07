"""Exact installed sorting-network parity and overflow-free wide medians."""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from nodes.impl import native_neighborhood as native
from nodes.impl.native import lib
from nodes.impl.native_versions import CV_CN_MAX


def equal_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if actual.dtype == np.float32:
        actual, expected = actual.view(np.uint32), expected.view(np.uint32)
    np.testing.assert_array_equal(actual, expected)


def samples(channels, shape, pattern, seed=917):
    rng = np.random.default_rng(seed)
    shape = (*shape, channels)
    if pattern == "ordinary":
        return rng.random(shape, dtype=np.float32)
    if pattern == "zeros":
        values = np.array([0, 0x80000000, 0x3E800000], np.uint32)
    elif pattern == "nonfinite":
        values = np.array(
            [0, 0x3E800000, 0x3F800000, 0x7F800000, 0xFF800000, 0x7FC00000], np.uint32
        )
    else:
        values = np.array(
            [0, 0x80000000, 0x3E800000, 0x7FC00000, 0xFFC00000, 0x7FC01234, 0xFFC04321],
            np.uint32,
        )
    return values[rng.integers(0, len(values), shape)].view(np.float32)


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5, 7, 31, CV_CN_MAX])
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1), (2, 3), (3, 2), (7, 19)])
@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize("pattern", ["ordinary", "zeros", "nonfinite", "nan_payloads"])
def test_small_median_complete_float_domain(channels, shape, radius, pattern):
    source = samples(channels, shape, pattern)
    saved = source.copy()
    expected = cv2.medianBlur(source, 2 * radius + 1)
    actual = native.median(source, radius)
    equal_bits(actual, expected)
    equal_bits(source, saved)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("radius", [1, 2])
def test_scalar_simd_and_tail_dispatch(optimized, radius):
    previous = cv2.useOptimized()
    cv2.setUseOptimized(optimized)
    try:
        for channels in (1, 3, 4, 7):
            for width in range(1, 34):
                source = samples(
                    channels, (7, width), "nan_payloads", seed=width + channels
                )
                equal_bits(
                    native.median(source, radius),
                    cv2.medianBlur(source, 2 * radius + 1),
                )
    finally:
        cv2.setUseOptimized(previous)


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize(
    "layout",
    ["strided", "transposed", "readonly", "unaligned", "broadcast", "two_dimensional"],
)
def test_small_median_array_contracts(channels, radius, layout):
    source = samples(channels, (11, 17), "nan_payloads")
    if layout == "strided":
        source = source[::-1, ::-2]
    elif layout == "transposed":
        source = source.swapaxes(0, 1)
    elif layout == "readonly":
        source.flags.writeable = False
    elif layout == "unaligned":
        destination = np.ndarray(
            source.shape, np.float32, buffer=bytearray(source.nbytes + 1), offset=1
        )
        destination[:] = source
        source = destination
    elif layout == "broadcast":
        source = np.broadcast_to(source[:1, :1], source.shape)
    elif channels == 1:
        source = source[:, :, 0]
    saved = source.copy()
    equal_bits(native.median(source, radius), cv2.medianBlur(source, 2 * radius + 1))
    equal_bits(source, saved)


def weighted_median_reference(source, radius):
    """Independent exhaustive histogram, weighted by replicate-border counts.

    This does not use sliding columns or the C lazy two-level algorithm. Counts
    are exact integers representable in float64 through the radius1000 limit.
    """
    height, width, channels = source.shape
    result = np.empty_like(source)
    y_weights = [
        np.bincount(
            np.clip(np.arange(y - radius, y + radius + 1), 0, height - 1),
            minlength=height,
        )
        for y in range(height)
    ]
    x_weights = [
        np.bincount(
            np.clip(np.arange(x - radius, x + radius + 1), 0, width - 1),
            minlength=width,
        )
        for x in range(width)
    ]
    threshold = (2 * radius + 1) ** 2 // 2 + 1
    for y in range(height):
        for x in range(width):
            weights = np.outer(y_weights[y], x_weights[x]).reshape(-1)
            for c in range(channels):
                histogram = np.bincount(
                    source[:, :, c].reshape(-1), weights=weights, minlength=256
                )
                result[y, x, c] = np.searchsorted(histogram.cumsum(), threshold)
    return result[:, :, 0] if channels == 1 else result


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (8, 1), (3, 5), (5, 3), (7, 9)])
@pytest.mark.parametrize("radius", [3, 7, 127, 128, 180, 181, 255, 256, 511, 1000])
def test_wide_median_exact_weighted_reference(channels, shape, radius):
    bytes_image = np.random.default_rng(43).integers(
        0, 256, (*shape, channels), np.uint8
    )
    source = bytes_image.astype(np.float32) / 255
    saved = source.copy()
    actual = native.median(source, radius)
    expected = weighted_median_reference(bytes_image, radius)
    equal_bits(actual, expected)
    equal_bits(source, saved)
    if radius <= 127:
        equal_bits(actual, cv2.medianBlur(bytes_image, 2 * radius + 1))


@pytest.mark.parametrize("radius", [128, 180, 181, 255, 256, 511, 1000])
@pytest.mark.parametrize("value", [0, 1, 71, 127, 128, 254, 255])
def test_constant_images_through_maximum_radius(radius, value):
    source = np.full((3, 7, 3), np.float32(value / 255), np.float32)
    np.testing.assert_array_equal(
        native.median(source, radius), np.full(source.shape, value, np.uint8)
    )


def test_original_saturated_histogram_wrong_pixels_are_corrected():
    original = np.tile(
        np.array([0, 0, 0, 0, 255, 255, 255, 255, 255], np.uint8), (7, 1)
    )
    expected = weighted_median_reference(original[:, :, None], 180)
    baseline = cv2.medianBlur(original, 361)
    assert np.count_nonzero(baseline != expected) == 14
    equal_bits(native.median(original.astype(np.float32) / 255, 180), expected)


def test_original_constant_large_window_assertion_is_corrected():
    original = np.array([[71]], np.uint8)
    with pytest.raises(cv2.error, match="k < 16"):
        cv2.medianBlur(original, 363)
    equal_bits(native.median(original.astype(np.float32) / 255, 181), original)


@pytest.mark.parametrize("channels", [2, 5, 7, CV_CN_MAX, CV_CN_MAX + 1])
@pytest.mark.parametrize("radius", [1, 2, 3, 1000])
def test_channel_contract(channels, radius):
    source = samples(channels, (3, 7), "ordinary")
    if radius > 2 or channels > CV_CN_MAX:
        with pytest.raises(cv2.error):
            native.median(source, radius)
        if radius <= 2:
            # The ceiling is OpenCV's own, on the float sort network.
            with pytest.raises(cv2.error):
                cv2.medianBlur(source, 2 * radius + 1)
    else:
        equal_bits(
            native.median(source, radius), cv2.medianBlur(source, 2 * radius + 1)
        )


def test_no_op_identity_and_no_op_does_not_validate(monkeypatch):
    from test_neighborhood_ops import MODULES

    module = MODULES["median_blur"][0]

    def forbidden(*args):
        raise AssertionError("Radius zero called processing")

    monkeypatch.setattr(module, "median", forbidden)
    source = np.empty((0, 0), np.float64)
    assert module.median_blur_node(source, 0) is source


def test_native_dispatch_without_opencv_algorithm(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Retained OpenCV median algorithm called")

    monkeypatch.setattr(cv2, "medianBlur", forbidden)
    for radius in (1, 2):
        result = native.median(samples(7, (5, 9), "nan_payloads"), radius)
        assert result.shape == (5, 9, 7)
    for radius in (128, 181, 1000):
        np.testing.assert_array_equal(
            native.median(np.ones((3, 5), np.float32), radius), 255
        )


def test_direct_c_boundaries():
    value = ct.c_float()
    p = ct.pointer(value)
    maximum = ct.c_size_t(-1).value
    library = lib()
    assert library.cn_neighborhood_median_f32(None, p, 1, 1, 1, 1, 8) == 1
    assert library.cn_neighborhood_median_f32(p, p, maximum, 2, 1, 1, 8) == 2
    assert library.cn_neighborhood_median_f32(p, p, 1, 1, 0, 1, 8) == 1
    assert library.cn_neighborhood_median_f32(p, p, 1, 1, 1, 1, 0) == 1
    assert library.cn_neighborhood_median_u8(p, p, maximum, 2, 3, 1000) == 2
    assert library.cn_neighborhood_median_u8(p, p, 1, 1, 1, 1001) == 1


def test_concurrent_native_median_requests():
    sources = [samples(c, (19, 31), "nan_payloads", seed=c) for c in (1, 3, 4, 7)]
    source_bytes = np.random.default_rng(96).integers(0, 256, (5, 9, 3), np.uint8)
    wide_source = source_bytes.astype(np.float32) / 255
    wide_expected = weighted_median_reference(source_bytes, 1000)
    expected = [
        [cv2.medianBlur(source, radius * 2 + 1) for radius in (1, 2)]
        for source in sources
    ]
    saved = [source.copy() for source in sources]

    def run(index):
        source = sources[index % len(sources)]
        return (
            native.median(source, 1),
            native.median(source, 2),
            native.median(wide_source, 1000),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run, range(24)))
    for i, result in enumerate(results):
        equal_bits(result[0], expected[i % len(sources)][0])
        equal_bits(result[1], expected[i % len(sources)][1])
        equal_bits(result[2], wide_expected)
    for source, before in zip(sources, saved, strict=True):
        equal_bits(source, before)
