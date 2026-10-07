"""Exact median-cut and caption comparisons; no GPU or timing benchmarks."""

from __future__ import annotations

import ctypes as ct
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import ImageFont

from nodes.impl import caption, native_buffers, native_palette
from nodes.impl.dithering import palette

REFERENCE = Path(__file__).with_name("reference_palette")


def load(name, package):
    spec = importlib.util.spec_from_file_location(
        package + "._original_" + name, REFERENCE / (name + ".py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORIGINAL = load("palette", "nodes.impl.dithering")
ORIGINAL_CAPTION = load("caption", "nodes.impl")
# The original helper must not inherit the new C normalization it used to call.
_spec = importlib.util.spec_from_file_location(
    "nodes.impl._caption_original_image_utils",
    REFERENCE.parent / "reference_buffers/image_utils.py",
)
assert _spec is not None and _spec.loader is not None
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)
ORIGINAL_CAPTION.__dict__.update(normalize=_utils.normalize)


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("pixels", [1, 2, 7, 8, 33, 128, 129, 8201])
@pytest.mark.parametrize("colors", [1, 2, 8, 31])
def test_median_cut_exact(channels, pixels, colors):
    image = np.random.default_rng(91).random((1, pixels, channels), dtype=np.float32)
    np.testing.assert_array_equal(
        palette.median_cut_palette(image, colors),
        ORIGINAL.median_cut_palette(image, colors),
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("pattern", ["constant", "ties", "dominant", "signed_zero"])
@pytest.mark.parametrize(
    "layout", ["plain", "reversed", "transpose", "readonly", "unaligned"]
)
def test_median_cut_stable_partitions(channels, pattern, layout):
    rng = np.random.default_rng(27)
    image = rng.integers(0, 4, (37, 23, channels)).astype(np.float32) / 4
    if pattern == "constant":
        image[:] = 0.75
    elif pattern == "dominant":
        image[:29] = 1
    elif pattern == "signed_zero":
        image[image == 0] = -0.0
    if layout == "reversed":
        image = image[::-1, ::-2]
    elif layout == "transpose":
        image = image.swapaxes(0, 1)
    elif layout == "readonly":
        image.setflags(write=False)
    elif layout == "unaligned":
        view = np.ndarray(
            image.shape, np.float32, buffer=bytearray(image.nbytes + 1), offset=1
        )
        view[:] = image
        image = view
    before = image.copy()
    actual, expected = (
        palette.median_cut_palette(image, 19),
        ORIGINAL.median_cut_palette(image, 19),
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("channels", [3, 4])
def test_median_cut_planar_root_bucket(channels):
    # A node's CHW tensor transposed to HWC: upstream's (h * w, c) root bucket stays a
    # view with contiguous columns, so its np.mean(axis=0) reduces each column alone.
    image = (
        np.random.default_rng(65)
        .random((channels, 181, 211), dtype=np.float32)
        .transpose(1, 2, 0)
    )
    assert image.reshape(-1, channels).flags.f_contiguous
    for colors in (1, 2, 16):
        actual = palette.median_cut_palette(image, colors)
        expected = ORIGINAL.median_cut_palette(image, colors)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_median_cut_dtype(dtype):
    image = np.random.default_rng(39).integers(0, 251, (31, 17, 3)).astype(dtype)
    np.testing.assert_array_equal(
        palette.median_cut_palette(image, 13), ORIGINAL.median_cut_palette(image, 13)
    )


def test_median_cut_concurrency():
    image = np.random.default_rng(14).random((131, 97, 3), dtype=np.float32)
    expected = ORIGINAL.median_cut_palette(image, 37)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(
            pool.map(lambda _: palette.median_cut_palette(image, 37), range(12))
        )
    for result in results:
        np.testing.assert_array_equal(result, expected)


def test_median_cut_bad_buffers():
    with pytest.raises(ValueError):
        native_palette.median_cut(np.zeros((3, 4), np.float32), 0)
    with pytest.raises(TypeError):
        native_palette.median_cut(np.zeros((3, 4), np.float64), 3)
    with pytest.raises(ValueError):
        native_palette.median_cut(np.zeros((0, 4), np.float32), 3)
    fn = native_palette._api()
    assert fn(None, None, 1, 1, 1, 0, 0, 0, None) == 1
    source = np.zeros(4, np.float32)
    p = native_palette.ptr(source)
    written = ct.c_size_t()
    assert fn(p, p, ct.c_size_t(-1).value, 4, 1, 0, 0, 0, ct.byref(written)) == 2
    source[0] = np.nan
    # Expanded C contract preserves the original median-cut one-bucket NaN
    # result; nonfinite inputs are no longer delegated to Python buckets.
    assert fn(p, p, 1, 4, 1, 0, 0, 0, ct.byref(written)) == 0
    assert np.isnan(source[0])


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("position", ["TOP", "BOTTOM"])
@pytest.mark.parametrize("text", ["", "Hello!", "Wide caption 123", "Alpha\nBeta"])
@pytest.mark.parametrize("layout", ["plain", "reversed", "readonly", "float64"])
def test_caption_exact(monkeypatch, channels, position, text, layout):
    font_path = (
        Path(__file__).resolve().parents[2] / "backend/src/fonts/Roboto-Light.ttf"
    )
    for module in (caption, ORIGINAL_CAPTION):
        monkeypatch.setattr(
            module, "get_font", lambda size: ImageFont.truetype(str(font_path), size)
        )
    shape = (31, 73) if channels == 1 else (31, 73, channels)
    image = np.random.default_rng(26).random(shape, dtype=np.float32)
    image.flat[:3] = [np.nan, np.inf, -0.0]
    if layout == "reversed":
        image = image[::-1, ::-1]
    elif layout == "readonly":
        image.setflags(write=False)
    elif layout == "float64":
        image = image.astype(np.float64)
    before = image.copy()
    actual = caption.add_caption(
        image, text, 31, getattr(caption.CaptionPosition, position)
    )
    expected = ORIGINAL_CAPTION.add_caption(
        image, text, 31, getattr(ORIGINAL_CAPTION.CaptionPosition, position)
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(image, before)


def test_caption_concurrency_and_guards():
    image = np.random.default_rng(16).random((129, 517, 4), dtype=np.float32)
    raster = np.arange(37 * 517, dtype=np.uint8).reshape(37, 517)[:, ::-1]
    expected_caption = np.repeat(
        (raster.astype(np.float32) / 255)[:, :, None], 4, axis=2
    )
    expected_caption[:, :, 3] = 1
    expected = np.vstack((expected_caption, image))
    with ThreadPoolExecutor(max_workers=6) as pool:
        outputs = list(
            pool.map(
                lambda _: native_buffers.compose_caption(image, raster, True), range(12)
            )
        )
    for output in outputs:
        np.testing.assert_array_equal(output, expected)
    with pytest.raises(ValueError):
        native_buffers.compose_caption(image, raster[:, :1], True)
    with pytest.raises(TypeError):
        native_buffers.compose_caption(image, raster.astype(np.float32), True)
    with pytest.raises(ValueError):
        native_buffers.compose_caption(image[:, :, :1], raster, False)
    fn = native_buffers._api().cn_caption_compose
    assert fn(None, None, None, 1, 1, 3, 1, 0) == 1
    assert (
        fn(
            native_buffers.ptr(image),
            raster.ctypes.data,
            native_buffers.ptr(image),
            ct.c_size_t(-1).value,
            1,
            4,
            1,
            0,
        )
        == 2
    )
