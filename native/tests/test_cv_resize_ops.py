"""Exact checks against the installed OpenCV 4.8 CPU resize primitive.

Default IPP LINEAR is intentionally retained: its arithmetic is distinct from
the open-source baseline. AREA downsampling and IPP-off LINEAR must be C.
"""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from nodes.impl import native_cv_resize
from nodes.impl.native import lib, ptr


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[valid].view(np.uint32), expected[valid].view(np.uint32)
    )


@pytest.fixture
def ipp_off():
    enabled = cv2.ipp.useIPP()
    cv2.ipp.setUseIPP(False)
    try:
        yield
    finally:
        cv2.ipp.setUseIPP(enabled)


SHAPES = [
    (1, 1),
    (1, 37),
    (31, 1),
    (2, 2),
    (2, 7),
    (7, 2),
    (3, 5),
    (5, 3),
    (8, 8),
    (12, 16),
    (16, 12),
    (17, 19),
    (31, 37),
    (64, 80),
    (73, 89),
]
FACTORS = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (4, 4), (8, 8), (2, 8), (8, 2)]


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("factors", FACTORS)
def test_area_dimensions(channels, shape, factors):
    h, w = shape
    fy, fx = factors
    data_shape = shape if channels == 1 else (*shape, channels)
    image = np.random.default_rng(813).random(data_shape, dtype=np.float32)
    size = ((w + fx - 1) // fx, (h + fy - 1) // fy)
    exact(
        native_cv_resize.resize(image, size, cv2.INTER_AREA),
        cv2.resize(image, size, interpolation=cv2.INTER_AREA),
    )


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("factors", [(1, 2), (2, 1), (2, 2), (3, 5), (8, 8)])
def test_baseline_linear_dimensions(channels, shape, factors, ipp_off):
    h, w = shape
    fy, fx = factors
    data_shape = shape if channels == 1 else (*shape, channels)
    image = np.random.default_rng(721).random(data_shape, dtype=np.float32)
    size = (w * fx - fx + 1, h * fy - fy + 1)
    exact(
        native_cv_resize.resize(image, size, cv2.INTER_LINEAR),
        cv2.resize(image, size, interpolation=cv2.INTER_LINEAR),
    )


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(8, 12), (7, 11), (1, 7), (7, 1)])
@pytest.mark.parametrize("kind", ["zeros", "signed_zeros", "special", "exponents"])
@pytest.mark.parametrize("mode", [cv2.INTER_AREA, cv2.INTER_LINEAR])
def test_nonfinite_and_zero(channels, shape, kind, mode, ipp_off):
    data_shape = shape if channels == 1 else (*shape, channels)
    rng = np.random.default_rng(876)
    if kind == "zeros":
        image = np.full(data_shape, -0.0, np.float32)
    elif kind == "signed_zeros":
        image = np.copysign(
            np.zeros(data_shape, np.float32), rng.standard_normal(data_shape)
        )
        image = image.astype(np.float32)
    elif kind == "special":
        image = np.resize(
            np.array([0, -0.0, np.nan, np.inf, -np.inf, 1, -1], np.float32), data_shape
        )
    else:
        image = np.ldexp(
            rng.uniform(-1, 1, data_shape).astype(np.float32),
            rng.integers(-140, 127, data_shape, dtype=np.int32),
        )
    h, w = shape
    size = (
        ((w + 1) // 2, (h + 1) // 2)
        if mode == cv2.INTER_AREA
        else (w * 3 + 1, h * 2 + 1)
    )
    exact(
        native_cv_resize.resize(image, size, mode),
        cv2.resize(image, size, interpolation=mode),
    )


@pytest.mark.parametrize(
    "layout", ["reversed", "strided", "readonly", "unaligned", "hwc1"]
)
@pytest.mark.parametrize("mode", [cv2.INTER_AREA, cv2.INTER_LINEAR])
def test_foreign_buffers(layout, mode, ipp_off):
    image = np.random.default_rng(993).random((17, 23, 3), dtype=np.float32)
    if layout == "reversed":
        image = image[::-1, ::-1, ::-1]
    elif layout == "strided":
        image = image[::2, ::2]
    elif layout == "readonly":
        image.flags.writeable = False
    elif layout == "unaligned":
        target = np.ndarray(image.shape, np.float32, bytearray(image.nbytes + 1), 1)
        target[:] = image
        image = target
    else:
        image = image[..., :1]
    saved = image.copy()
    size = (7, 5) if mode == cv2.INTER_AREA else (59, 41)
    exact(
        native_cv_resize.resize(image, size, mode),
        cv2.resize(image, size, interpolation=mode),
    )
    exact(image, saved)


@pytest.mark.parametrize("mode", [cv2.INTER_AREA, cv2.INTER_LINEAR])
def test_concurrent_large_image(mode, ipp_off):
    rng = np.random.default_rng(105)
    images = [rng.random((271, 313, c), dtype=np.float32) for c in (1, 2, 3, 4)]
    size = (119, 87) if mode == cv2.INTER_AREA else (467, 389)
    expected = [cv2.resize(image, size, interpolation=mode) for image in images]

    def worker(image):
        # OpenCV's IPP enable flag is thread-local: match the oracle's dispatch.
        enabled = cv2.ipp.useIPP()
        cv2.ipp.setUseIPP(False)
        try:
            return native_cv_resize.resize(image, size, mode)
        finally:
            cv2.ipp.setUseIPP(enabled)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, images))
    for actual, reference in zip(results, expected, strict=True):
        exact(actual, reference)


@pytest.mark.parametrize("mode", [cv2.INTER_AREA, cv2.INTER_LINEAR])
def test_native_path_has_no_opencv_resize(mode, monkeypatch, ipp_off):
    image = np.random.default_rng(711).random((31, 37, 4), dtype=np.float32)
    size = (13, 11) if mode == cv2.INTER_AREA else (73, 59)
    expected = cv2.resize(image, size, interpolation=mode)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Native resize called the original OpenCV primitive")

    monkeypatch.setattr(cv2, "resize", forbidden)
    exact(native_cv_resize.resize(image, size, mode), expected)


def test_default_ipp_linear_retention_is_explicit(monkeypatch):
    enabled = cv2.ipp.useIPP()
    cv2.ipp.setUseIPP(True)
    try:
        image = np.random.default_rng(899).random((7, 9, 3), dtype=np.float32)
        expected = cv2.resize(image, (37, 31), interpolation=cv2.INTER_LINEAR)
        original = cv2.resize
        calls = []

        def observed(*args, **kwargs):
            calls.append(kwargs.get("interpolation"))
            return original(*args, **kwargs)

        monkeypatch.setattr(cv2, "resize", observed)
        exact(native_cv_resize.resize(image, (37, 31), cv2.INTER_LINEAR), expected)
        assert calls == [cv2.INTER_LINEAR]
    finally:
        cv2.ipp.setUseIPP(enabled)


def test_c_validation_and_overlap():
    native = lib()
    source = np.ones((4, 5, 3), np.float32)
    output = np.full((2, 3, 3), 987, np.float32)
    for function in (native.cn_cv_resize_area, native.cn_cv_resize_linear):
        assert function(None, ptr(output), 4, 5, 3, 2, 3) == 1
        assert function(ptr(source), ptr(output), 0, 5, 3, 2, 3) == 1
        assert function(ptr(source), ptr(output), 4, 5, 5, 2, 3) == 1
        assert function(ptr(source), ptr(source), 4, 5, 3, 4, 5) == 1
        assert (
            function(ptr(source), ptr(output), ct.c_size_t(-1).value, 5, 3, 2, 3) == 2
        )
        np.testing.assert_array_equal(output, 987)


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(31, 37), (32, 40), (73, 89)])
@pytest.mark.parametrize("size", [(1, 1), (1, 3), (3, 1), (3, 5), (74, 3), (5, 146)])
def test_additional_baseline_ratios(channels, shape, size, ipp_off):
    image = np.random.default_rng(19).random((*shape, channels), dtype=np.float32)
    for mode in (cv2.INTER_AREA, cv2.INTER_LINEAR):
        exact(
            native_cv_resize.resize(image, size, mode),
            cv2.resize(image, size, interpolation=mode),
        )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("not_exact", [False, True])
def test_area_ipp_dispatch(enabled, not_exact, monkeypatch):
    old_enabled, old_not_exact = cv2.ipp.useIPP(), cv2.ipp.useIPP_NotExact()
    cv2.ipp.setUseIPP(enabled)
    cv2.ipp.setUseIPP_NotExact(not_exact)
    try:
        image = np.random.default_rng(98).random((31, 37, 4), dtype=np.float32)
        expected = cv2.resize(image, (11, 7), interpolation=cv2.INTER_AREA)
        original = cv2.resize
        calls = []

        def observed(*args, **kwargs):
            calls.append(kwargs.get("interpolation"))
            return original(*args, **kwargs)

        monkeypatch.setattr(cv2, "resize", observed)
        exact(native_cv_resize.resize(image, (11, 7), cv2.INTER_AREA), expected)
        assert calls == ([cv2.INTER_AREA] if enabled and not_exact else [])
    finally:
        cv2.ipp.setUseIPP(old_enabled)
        cv2.ipp.setUseIPP_NotExact(old_not_exact)


def test_ipp_linear_saved_image_rounding_evidence():
    """Replacing default IPP with the source baseline changes 111 saved bytes."""
    enabled = cv2.ipp.useIPP()
    try:
        image = np.full((7, 9, 3), 128.5 / 255, np.float32)
        cv2.ipp.setUseIPP(True)
        original = cv2.resize(image, (37, 31), interpolation=cv2.INTER_LINEAR)
        retained = native_cv_resize.resize(image, (37, 31), cv2.INTER_LINEAR)
        cv2.ipp.setUseIPP(False)
        baseline = native_cv_resize.resize(image, (37, 31), cv2.INTER_LINEAR)
        exact(retained, original)
        original_u8 = (original * 255).round().astype(np.uint8)
        baseline_u8 = (baseline * 255).round().astype(np.uint8)
        assert np.count_nonzero(original_u8 != baseline_u8) == 111
        assert np.max(np.abs(original - baseline)) == np.float32(2**-24)
        ok, original_png = cv2.imencode(".png", original_u8)
        assert ok
        ok, baseline_png = cv2.imencode(".png", baseline_u8)
        assert ok
        assert not np.array_equal(original_png, baseline_png)
    finally:
        cv2.ipp.setUseIPP(enabled)


@pytest.mark.parametrize("size", [(0, 1), (1, -1), (2**31, 1), (1.5, 2)])
def test_invalid_python_dimensions(size):
    image = np.ones((3, 5), np.float32)
    with pytest.raises((ValueError, OverflowError)):
        native_cv_resize.resize(image, size, cv2.INTER_AREA)


def test_bridge_validation():
    with pytest.raises(TypeError):
        native_cv_resize.resize(np.ones((3, 5), np.float64), (2, 2), cv2.INTER_AREA)
    with pytest.raises(ValueError):
        native_cv_resize.resize(np.ones((3,), np.float32), (2, 2), cv2.INTER_AREA)
    with pytest.raises(ValueError):
        native_cv_resize.resize(np.ones((3, 0), np.float32), (2, 2), cv2.INTER_AREA)
    with pytest.raises(ValueError):
        native_cv_resize.resize(np.ones((3, 5), np.float32), (2, 2), cv2.INTER_LANCZOS4)


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("threads", [1, 2, 32])
@pytest.mark.parametrize(
    "shape,size",
    [((7, 1), (1, 3)), ((7, 9), (5, 3)), ((700, 1601), (1601, 300))],
)
def test_fractional_underflow_canonical_zero_signs(channels, threads, shape, size):
    """Canonical one-thread reference avoids original scheduler-dependent zero signs."""
    old_threads = cv2.getNumThreads()
    try:
        image = np.full(
            (*shape, channels),
            -np.nextafter(np.float32(0), np.float32(1)),
            np.float32,
        )
        cv2.setNumThreads(1)
        canonical = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        cv2.setNumThreads(threads)
        exact(native_cv_resize.resize(image, size, cv2.INTER_AREA), canonical)
    finally:
        cv2.setNumThreads(old_threads)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_underflow_zero_signs_do_not_depend_on_concurrent_calls(workers):
    old_threads = cv2.getNumThreads()
    image = np.full(
        (700, 1601), -np.nextafter(np.float32(0), np.float32(1)), np.float32
    )
    try:
        cv2.setNumThreads(1)
        canonical = cv2.resize(image, (1601, 300), interpolation=cv2.INTER_AREA)
    finally:
        cv2.setNumThreads(old_threads)

    def worker(_):
        return native_cv_resize.resize(image, (1601, 300), cv2.INTER_AREA)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(worker, range(8)))
    for result in results:
        exact(result, canonical)


def test_original_area_scheduler_zero_sign_evidence():
    """The original thread-count difference is 9,606 zero signs, no nonzero pixels."""
    old_threads = cv2.getNumThreads()
    image = np.full(
        (700, 1601), -np.nextafter(np.float32(0), np.float32(1)), np.float32
    )
    try:
        cv2.setNumThreads(1)
        canonical = cv2.resize(image, (1601, 300), interpolation=cv2.INTER_AREA)
        cv2.setNumThreads(32)
        parallel = cv2.resize(image, (1601, 300), interpolation=cv2.INTER_AREA)
        bits = parallel.view(np.uint32)
        assert np.flatnonzero(bits[:, 0] == 0).tolist() == [
            0,
            43,
            86,
            129,
            171,
            214,
            257,
        ]
        assert np.count_nonzero(bits != canonical.view(np.uint32)) == 9606
        np.testing.assert_array_equal(parallel, canonical)
    finally:
        cv2.setNumThreads(old_threads)
