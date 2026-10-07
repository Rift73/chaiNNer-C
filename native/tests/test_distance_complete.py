"""Exact ESDF and binary-mask differential checks against original dependencies."""

from __future__ import annotations

import ctypes as ct
import runpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_filter_ops import MODULES
from test_resample_ops import LAYOUTS, with_layout

from nodes.impl import native_filters as native
from nodes.impl.native import ptr

current, reference = MODULES["distance_transform"]
pixels = runpy.run_path(
    str(Path(__file__).with_name("reference_layout") / "rotation_pixels.py")
)
reference.__dict__["to_uint8"] = pixels["to_uint8"]


def fixture(shape, kind):
    rng = np.random.default_rng(516)
    if kind == "random":
        return rng.random(shape, dtype=np.float32)
    if kind == "binary":
        return rng.integers(0, 2, shape).astype(np.float32)
    if kind == "zero":
        return np.zeros(shape, np.float32)
    if kind == "one":
        return np.ones(shape, np.float32)
    if kind == "gray":
        return np.full(shape, 0.5, np.float32)
    if kind == "single":
        out = np.ones(shape, np.float32)
        out[shape[0] // 2, shape[1] // 2] = 0
        return out
    if kind == "ramp":
        return np.broadcast_to(
            np.linspace(0, 1, shape[1], dtype=np.float32), shape
        ).copy()
    if kind == "antialiased":
        yy, xx = np.indices(shape, dtype=np.float32)
        return np.clip((xx - yy + 2) / 4, 0, 1)
    out = rng.random(shape, dtype=np.float32)
    out.flat[: min(out.size, 7)] = np.array(
        [np.nan, np.inf, -np.inf, -0.0, -1, 2, 1e-40], np.float32
    )[: min(out.size, 7)]
    return out


def compare(source, spread, subpixel):
    before = source.copy()
    with np.errstate(all="ignore"):
        expected = reference.distance_transform_node(source, spread, subpixel)
        actual = current.distance_transform_node(source, spread, subpixel)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(
        actual[finite].view(np.uint32), expected[finite].view(np.uint32)
    )
    np.testing.assert_array_equal(source, before)
    assert not np.shares_memory(source, actual)


@pytest.mark.parametrize("shape", [(1, 1), (1, 13), (11, 1), (3, 5), (19, 23)])
@pytest.mark.parametrize(
    "kind",
    [
        "zero",
        "one",
        "gray",
        "random",
        "binary",
        "single",
        "ramp",
        "antialiased",
        "special",
    ],
)
@pytest.mark.parametrize("spread", [1, 4, 19, 65536])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_esdf(shape, kind, spread, layout):
    compare(with_layout(fixture(shape, kind), layout), spread, True)


@pytest.mark.parametrize("shape", [(1, 1), (1, 13), (11, 1), (19, 23), (137, 149)])
@pytest.mark.parametrize(
    "kind", ["zero", "one", "random", "binary", "single", "antialiased", "special"]
)
@pytest.mark.parametrize("spread", [1, 4, 100, 65536, 1e40])
@pytest.mark.parametrize("ipp", [False, True])
def test_binary(shape, kind, spread, ipp):
    enabled = cv2.ipp.useIPP()
    try:
        cv2.ipp.setUseIPP(ipp)
        compare(fixture(shape, kind), spread, False)
    finally:
        cv2.ipp.setUseIPP(enabled)


@pytest.mark.parametrize("subpixel", [False, True])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_single_channel_axis(subpixel, layout):
    compare(
        with_layout(fixture((17, 23), "antialiased")[..., None], layout), 4, subpixel
    )


@pytest.mark.parametrize("subpixel", [False, True])
def test_concurrent_distance_fields(subpixel):
    jobs = [
        fixture((271, 277), kind)
        for kind in ("random", "binary", "antialiased", "ramp")
    ]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda source: compare(source, 4, subpixel), jobs))


def test_esdf_does_not_call_rust(monkeypatch):
    import chainner_ext

    def forbidden(*_args, **_kwargs):
        raise AssertionError("C ESDF called the Rust implementation")

    monkeypatch.setattr(chainner_ext, "esdf", forbidden)
    result = current.distance_transform_node(fixture((7, 11), "antialiased"), 4, True)
    assert result.shape == (7, 11, 1)


def test_scalar_binary_does_not_call_opencv(monkeypatch):
    enabled = cv2.ipp.useIPP()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("C scalar chamfer called OpenCV")

    try:
        cv2.ipp.setUseIPP(False)
        expected = reference.distance_transform_node(
            fixture((17, 23), "single"), 100, False
        )
        monkeypatch.setattr(cv2, "distanceTransform", forbidden)
        actual = current.distance_transform_node(
            fixture((17, 23), "single"), 100, False
        )
        np.testing.assert_array_equal(actual, expected)
    finally:
        cv2.ipp.setUseIPP(enabled)


def test_ipp_primitive_is_retained_exactly(monkeypatch):
    enabled = cv2.ipp.useIPP()
    original = cv2.distanceTransform
    calls = []

    def recorded(source, distance_type, mask_size, **keywords):
        # The port passes OpenCV's dstType keyword, and only that keyword.
        assert keywords.keys() == {"dstType"}
        assert source.dtype == np.uint8
        assert set(np.unique(source)).issubset({0, 255})
        calls.append((distance_type, mask_size))
        return original(source, distance_type, mask_size, **keywords)

    try:
        cv2.ipp.setUseIPP(True)
        expected = reference.distance_transform_node(
            fixture((137, 149), "single"), 100, False
        )
        monkeypatch.setattr(cv2, "distanceTransform", recorded)
        actual = current.distance_transform_node(
            fixture((137, 149), "single"), 100, False
        )
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        assert calls == [(cv2.DIST_L2, 5), (cv2.DIST_L2, 5)]
    finally:
        cv2.ipp.setUseIPP(enabled)


def test_c_boundaries():
    source = np.zeros((1, 1), np.float32)
    out = np.full((1, 1), -17, np.float32)
    native.subpixel_sdf(source, 2)
    dll = native._lib  # verify C boundary rejection before access
    for h, w, status in (
        (0, 1, 1),
        (1, 0, 1),
        (2**31, 1, 2),
        (1, 2**31, 2),
        (2**31 - 1, 2**31 - 1, 2),
    ):
        assert dll.cn_distance_esdf(ptr(source), ptr(out), h, w, 2) == status
        assert out[0, 0] == -17
    assert dll.cn_distance_esdf(None, ptr(out), 1, 1, 2) == 1
    assert dll.cn_distance_esdf(ptr(source), None, 1, 1, 2) == 1
    assert ct.sizeof(ct.c_size_t) in (4, 8)


@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 1), np.float32),
        np.ones((2, 3, 2), np.float32),
        np.ones((2, 3), np.float64),
    ],
)
def test_bad_esdf_images(image):
    with pytest.raises((TypeError, ValueError)):
        native.subpixel_sdf(image, 2)
