"""Bit-exact installed OpenCV morphology contracts; CPU correctness only."""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from nodes.impl import native_filters as native
from nodes.impl.native import lib, ptr
from nodes.impl.native_versions import CV_CN_MAX

ORIGINAL_DILATE = cv2.dilate
ORIGINAL_ERODE = cv2.erode
ORIGINAL_KERNEL = cv2.getStructuringElement


def exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[valid].view(np.uint32), expected[valid].view(np.uint32)
    )


def source(shape=(13, 19), channels=3, pattern="random", seed=894):
    rng = np.random.default_rng(seed)
    result = rng.random((*shape, channels), dtype=np.float32)
    if pattern == "signed_zero":
        result[:] = rng.choice(np.array([-0.0, 0.0, 0.25], np.float32), result.shape)
    elif pattern == "nan":
        result.ravel()[::7] = np.nan
    elif pattern == "infinity":
        result.ravel()[::5] = np.inf
        result.ravel()[1::7] = -np.inf
    elif pattern == "mixed":
        result[:] = rng.choice(
            np.array(
                [
                    np.nan,
                    np.inf,
                    -np.inf,
                    -0.0,
                    0.0,
                    -np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    0.25,
                ],
                np.float32,
            ),
            result.shape,
        )
    elif pattern == "all_nan":
        result.fill(np.nan)
    elif pattern == "dynamic":
        result[:] = np.ldexp(
            result - np.float32(0.5),
            rng.integers(-145, 127, result.shape, dtype=np.int32),
        )
    return result


def original(image, shape, radius, iterations, maximum):
    element = ORIGINAL_KERNEL(shape, (radius * 2 + 1,) * 2)
    return (ORIGINAL_DILATE if maximum else ORIGINAL_ERODE)(
        image, element, iterations=iterations
    )


@pytest.fixture(params=[False, True])
def optimized(request):
    previous = cv2.useOptimized()
    cv2.setUseOptimized(request.param)
    try:
        yield request.param
    finally:
        cv2.setUseOptimized(previous)


@pytest.mark.parametrize("shape", [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE])
@pytest.mark.parametrize("maximum", [False, True])
@pytest.mark.parametrize("channels", [1, 2, 3, 4, 7])
@pytest.mark.parametrize(
    "pattern",
    ["random", "signed_zero", "nan", "infinity", "mixed", "all_nan", "dynamic"],
)
@pytest.mark.parametrize("radius,iterations", [(1, 1), (2, 2), (7, 3)])
def test_exact_morphology(
    shape, maximum, channels, pattern, radius, iterations, optimized
):
    image = source(channels=channels, pattern=pattern)
    before = image.copy()
    exact(
        native.morphology(image, shape, radius, iterations, maximum=maximum),
        original(image, shape, radius, iterations, maximum),
    )
    exact(image, before)


@pytest.mark.parametrize(
    "size",
    [
        (1, 1),
        (1, 17),
        (19, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (8, 9),
        (9, 8),
        (17, 31),
        (18, 33),
        (19, 35),
    ],
)
@pytest.mark.parametrize("shape", [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE])
@pytest.mark.parametrize("maximum", [False, True])
def test_border_and_simd_tails(size, shape, maximum, optimized):
    image = source(size, 3, "mixed")
    exact(
        native.morphology(image, shape, 3, 2, maximum=maximum),
        original(image, shape, 3, 2, maximum),
    )


@pytest.mark.parametrize("radius", [1, 2, 4, 16, 127, 128, 333, 1000])
@pytest.mark.parametrize("maximum", [False, True])
def test_ellipse_radius_extremes(radius, maximum):
    image = source((7, 9), 4)
    exact(
        native.morphology(image, cv2.MORPH_ELLIPSE, radius, 1, maximum=maximum),
        original(image, cv2.MORPH_ELLIPSE, radius, 1, maximum),
    )


def test_all_1000_ellipse_kernel_radii_exact():
    api = lib()
    for radius in range(1001):
        spans = np.empty(radius * 2 + 1, np.uintp)
        assert (
            api.cn_morphology_ellipse_spans(
                radius, spans.ctypes.data_as(ct.POINTER(ct.c_size_t))
            )
            == 0
        )
        element = ORIGINAL_KERNEL(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
        expected = (np.count_nonzero(element, axis=1) - 1) // 2
        np.testing.assert_array_equal(spans, expected, err_msg=f"radius={radius}")


@pytest.mark.parametrize("maximum", [False, True])
@pytest.mark.parametrize("shape", [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE])
def test_nonfinite_large_radius_compressed_borders(shape, maximum):
    image = source((3, 5), 3, "mixed")
    exact(
        native.morphology(image, shape, 103, 1, maximum=maximum),
        original(image, shape, 103, 1, maximum),
    )


@pytest.mark.parametrize(
    "kind",
    ["reverse", "transpose", "readonly", "unaligned", "broadcast", "grayscale2d"],
)
@pytest.mark.parametrize("shape", [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE])
@pytest.mark.parametrize("maximum", [False, True])
def test_foreign_layouts_immutable(kind, shape, maximum):
    image = source(pattern="mixed")
    if kind == "reverse":
        image = image[::-1, ::-1, ::-1]
    elif kind == "transpose":
        image = image.swapaxes(0, 1)
    elif kind == "readonly":
        image.setflags(write=False)
    elif kind == "unaligned":
        view = np.ndarray(
            image.shape, np.float32, buffer=bytearray(image.nbytes + 1), offset=1
        )
        view[:] = image
        image = view
    elif kind == "broadcast":
        image = np.broadcast_to(image[:1, :1], image.shape)
    elif kind == "grayscale2d":
        image = image[:, :, 0]
    before = image.copy()
    actual = native.morphology(image, shape, 3, 2, maximum=maximum)
    exact(actual, original(image, shape, 3, 2, maximum))
    exact(image, before)
    assert actual.flags.aligned and actual.flags.c_contiguous and actual.flags.writeable
    assert not np.shares_memory(actual, image)


@pytest.mark.parametrize("shape", [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE])
def test_all_options_run_in_c(shape, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Morphology kernels must run entirely in C")

    monkeypatch.setattr(cv2, "getStructuringElement", forbidden)
    monkeypatch.setattr(cv2, "dilate", forbidden)
    monkeypatch.setattr(cv2, "erode", forbidden)
    for pattern in ["random", "signed_zero", "mixed"]:
        image = source(pattern=pattern)
        for maximum in [False, True]:
            exact(
                native.morphology(image, shape, 3, 2, maximum=maximum),
                original(image, shape, 3, 2, maximum),
            )


def test_noop_preserves_identity():
    image = source()
    assert native.morphology(image, cv2.MORPH_ELLIPSE, 0, 3, maximum=True) is image
    assert native.morphology(image, cv2.MORPH_ELLIPSE, 3, 0, maximum=False) is image


@pytest.mark.parametrize("shape", [(3,), (3, 0, 4), (3, 4, 0), (2, 3, 4, 4)])
def test_invalid_shapes(shape):
    with pytest.raises(ValueError):
        native.morphology(
            np.empty(shape, np.float32), cv2.MORPH_ELLIPSE, 1, 1, maximum=True
        )


def test_checked_c_abi():
    api = lib()
    image = source((5, 7))
    output = np.empty_like(image)
    null = ct.POINTER(ct.c_float)()
    args = (5, 7, 3, 1, 1, cv2.MORPH_ELLIPSE, 1, 8)
    assert api.cn_morphology_complete(null, ptr(output), *args) == 1
    assert api.cn_morphology_complete(ptr(image), null, *args) == 1
    assert api.cn_morphology_complete(ptr(image), ptr(image), *args) == 1
    shifted = ct.cast(image.ctypes.data + 4, ct.POINTER(ct.c_float))
    assert api.cn_morphology_complete(ptr(image), shifted, *args) == 1
    unaligned = ct.cast(image.ctypes.data + 1, ct.POINTER(ct.c_float))
    assert api.cn_morphology_complete(unaligned, ptr(output), *args) == 1
    assert (
        api.cn_morphology_complete(
            ptr(image), ptr(output), ct.c_size_t(-1).value, *args[1:]
        )
        == 2
    )
    assert api.cn_morphology_complete(ptr(image), ptr(output), *args[:-1], 3) == 1
    assert api.cn_morphology_ellipse_spans(1, None) == 1
    bad_spans = ct.cast(image.ctypes.data + 1, ct.POINTER(ct.c_size_t))
    assert api.cn_morphology_ellipse_spans(1, bad_spans) == 1


def test_concurrent_shapes_and_input_immutability():
    fixtures = [
        (
            source((47, 61), 3, "mixed" if i % 2 else "random", i + 17),
            i % 3,
            bool(i % 2),
        )
        for i in range(12)
    ]
    expected = [
        original(image, shape, 7, 3, maximum) for image, shape, maximum in fixtures
    ]
    before = [image.copy() for image, _, _ in fixtures]

    def run(i):
        image, shape, maximum = fixtures[i % len(fixtures)]
        exact(
            native.morphology(image, shape, 7, 3, maximum=maximum),
            expected[i % len(fixtures)],
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(36)))
    for (actual, _, _), snapshot in zip(fixtures, before, strict=True):
        exact(actual, snapshot)


@pytest.mark.parametrize("channels", [31, 127, CV_CN_MAX, CV_CN_MAX + 1, 512])
@pytest.mark.parametrize("shape", [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE])
@pytest.mark.parametrize("maximum", [False, True])
def test_uncommon_channels_and_all_zero_ties(channels, shape, maximum, optimized):
    image = source((5, 3), channels)
    image[:] = np.random.default_rng(63).choice(
        np.array([-0.0, 0.0], np.float32), image.shape
    )
    if channels > CV_CN_MAX:
        # OpenCV's channel ceiling: both sides raise.
        with pytest.raises(cv2.error):
            native.morphology(image, shape, 2, 2, maximum=maximum)
        with pytest.raises(cv2.error):
            original(image, shape, 2, 2, maximum)
        return
    exact(
        native.morphology(image, shape, 2, 2, maximum=maximum),
        original(image, shape, 2, 2, maximum),
    )


@pytest.mark.parametrize("shape", [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE])
@pytest.mark.parametrize("maximum", [False, True])
@pytest.mark.parametrize("pattern", ["random", "mixed"])
def test_node_iteration_limit(shape, maximum, pattern):
    image = source((3, 5), 3, pattern)
    exact(
        native.morphology(image, shape, 1, 1000, maximum=maximum),
        original(image, shape, 1, 1000, maximum),
    )


@pytest.mark.parametrize("radius", [3, 11, 32])
@pytest.mark.parametrize("iterations", [1, 2])
@pytest.mark.parametrize("maximum", [False, True])
def test_finite_ellipse_curved_boundary(radius, iterations, maximum, optimized):
    image = source((67, 83), 4)
    exact(
        native.morphology(
            image, cv2.MORPH_ELLIPSE, radius, iterations, maximum=maximum
        ),
        original(image, cv2.MORPH_ELLIPSE, radius, iterations, maximum),
    )
