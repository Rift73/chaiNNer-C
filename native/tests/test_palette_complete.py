"""Palette clustering parity, ownership and concurrency against installed OpenCV."""

from __future__ import annotations

import ast
import ctypes as ct
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from nodes.impl import native_palette
from nodes.impl.dithering import palette
from nodes.impl.native import lib, ptr
from nodes.impl.native_image_setup import distinct_exceeds

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).with_name("reference_palette") / "palette.py"
_spec = importlib.util.spec_from_file_location(
    "nodes.impl.dithering._complete_palette_reference", REFERENCE
)
assert _spec and _spec.loader
ORIGINAL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ORIGINAL)


def reference(data, colors):
    cv2.setRNGSeed(0)
    return cv2.kmeans(data, colors, None, (3, 10, 1.0), 10, cv2.KMEANS_PP_CENTERS)[2]  # pyright: ignore[reportCallIssue, reportArgumentType] -- OpenCV's generated stub forbids None for the optional bestLabels output that the binding accepts; upstream's palette.py carries the same ignore


def exact(actual, expected):
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 7, 15, 16, 17, 32, 65])
@pytest.mark.parametrize("pixels", [7, 35, 129, 1001])
@pytest.mark.parametrize("colors", [1, 2, 5])
@pytest.mark.parametrize("seed", [0, 1, 13])
def test_kmeans_all_distance_lanes(channels, pixels, colors, seed):
    data = np.random.default_rng(seed).random((pixels, channels), dtype=np.float32)
    before = data.copy()
    exact(native_palette.kmeans(data, colors), reference(data, colors))
    exact(data, before)


@pytest.mark.parametrize("channels", [1, 3, 4, 17])
@pytest.mark.parametrize("colors", [1, 2, 7, 31])
@pytest.mark.parametrize(
    "pattern", ["constant", "duplicates", "ties", "zero", "negative", "large", "tiny"]
)
def test_empty_clusters_ties_and_numeric_order(channels, colors, pattern):
    rng = np.random.default_rng(101)
    data = rng.random((113, channels), dtype=np.float32)
    if pattern == "constant":
        data[:] = 0.7
    elif pattern == "duplicates":
        data[:] = data[np.arange(113) % 3]
    elif pattern == "ties":
        data[:] = rng.integers(0, 3, data.shape)
    elif pattern == "zero":
        data[:] = 0
        data[::2] = -0.0
    elif pattern == "negative":
        data -= 0.75
    elif pattern == "large":
        data *= np.float32(1e15)
    elif pattern == "tiny":
        data *= np.float32(1e-22)
    exact(native_palette.kmeans(data, colors), reference(data, colors))


@pytest.mark.parametrize("dtype", [np.float32, np.uint8, np.uint16])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize(
    "layout", ["plain", "reverse", "transpose", "readonly", "unaligned"]
)
def test_original_helper_dtypes_and_layouts(dtype, channels, layout):
    rng = np.random.default_rng(144)
    image = rng.random((19, 23, channels), dtype=np.float32)
    if dtype != np.float32:
        image = (image * np.iinfo(dtype).max).astype(dtype)
    if layout == "reverse":
        image = image[::-1, ::-2]
    elif layout == "transpose":
        image = image.swapaxes(0, 1)
    elif layout == "readonly":
        image.setflags(write=False)
    elif layout == "unaligned":
        other = np.ndarray(
            image.shape, dtype, buffer=bytearray(image.nbytes + 1), offset=1
        )
        other[:] = image
        image = other
    before = image.copy()
    result = palette.kmeans_palette(image, 8)
    exact(result, ORIGINAL.kmeans_palette(image, 8))
    np.testing.assert_array_equal(image, before)
    assert not np.shares_memory(result, image)


def load_node(path, helpers):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = []
    for item in tree.body:
        if isinstance(item, ast.ImportFrom) and item.module == "__future__":
            definitions.append(item)
        if isinstance(item, (ast.FunctionDef, ast.ClassDef)):
            item.decorator_list = []
            definitions.append(item)
    tree.body = definitions
    namespace: dict[str, Any] = {
        "np": np,
        "Enum": Enum,
        "MAX_COLORS": 4096,
        "pad_edge": native_palette.pad_edge,
    }
    namespace.update(
        {
            name: getattr(helpers, name)
            for name in (
                "distinct_colors_palette",
                "kmeans_palette",
                "median_cut_palette",
            )
        }
    )
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


NODE_PATH = (
    "packages/chaiNNer_standard/image_utility/miscellaneous/palette_from_image.py"
)
NODE = load_node(ROOT / "backend/src" / NODE_PATH, palette)
# Match the production module import; frozen original helpers remain independent.
NODE["distinct_exceeds"] = distinct_exceeds
ORIGINAL_NODE = load_node(
    REFERENCE.parent.parent / "reference_analysis" / NODE_PATH, ORIGINAL
)


@pytest.mark.parametrize("method", ["ALL", "KMEANS", "MEDIAN_CUT"])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("case", ["singleton", "few_colors", "random", "one_row"])
def test_complete_palette_node_modes(method, channels, case):
    image = np.random.default_rng(82).random((23, 17, channels), dtype=np.float32)
    if case == "singleton":
        image = image[:1, :1]
    elif case == "few_colors":
        image[:] = (image > 0.5).astype(np.float32)
    elif case == "one_row":
        image = image[:1]
    expected = ORIGINAL_NODE["palette_from_image_node"](
        image, ORIGINAL_NODE["PaletteExtractionMethod"][method], 11
    )
    actual = NODE["palette_from_image_node"](
        image, NODE["PaletteExtractionMethod"][method], 11
    )
    exact(actual, expected)


def test_kmeans_never_calls_opencv(monkeypatch):
    image = np.random.default_rng(83).random((13, 17, 4), dtype=np.float32)
    expected = ORIGINAL.kmeans_palette(image, 9)

    def forbidden(*args, **kwargs):
        raise AssertionError("The C clustering path must not call OpenCV K-means")

    monkeypatch.setattr(cv2, "kmeans", forbidden)
    monkeypatch.setattr(cv2, "setRNGSeed", forbidden)
    exact(palette.kmeans_palette(image, 9), expected)


def test_calls_are_independent_of_other_rng_consumers():
    data = np.random.default_rng(31).random((2001, 4), dtype=np.float32)
    expected = reference(data, 11)
    cv2.setRNGSeed(8192)
    target = np.zeros((37, 13), np.float32)
    cv2.randu(target, 0, 1)  # pyright: ignore[reportCallIssue, reportArgumentType] -- OpenCV's generated stub forbids the scalar bounds that the binding accepts; the call only consumes OpenCV's global RNG
    exact(native_palette.kmeans(data, 11), expected)
    for _ in range(4):
        exact(native_palette.kmeans(data, 11), expected)


def test_concurrent_clustering_owns_all_state():
    arrays = [
        np.random.default_rng(i).random((4101, 4), dtype=np.float32) for i in range(4)
    ]
    expected = [reference(x, 9) for x in arrays]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda i: native_palette.kmeans(arrays[i % 4], 9), range(16))
        )
    for i, result in enumerate(results):
        exact(result, expected[i % 4])


@pytest.mark.parametrize("colors", [-1, 0, 4])
def test_invalid_palette_count_raises_original_error_type(colors):
    source = np.zeros((3, 3), np.float32)
    with pytest.raises(cv2.error):
        reference(source, colors)
    with pytest.raises(cv2.error):
        native_palette.kmeans(source, colors)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, np.finfo(np.float32).max])
def test_invalid_clustering_does_not_return_uninitialized_centers(value):
    source = np.full((7, 3), value, np.float32)
    with pytest.raises(cv2.error):
        native_palette.kmeans(source, 2)


def test_abi_validation_preserves_output():
    function = lib().cn_palette_kmeans
    source = np.ones(16, np.float32)
    target = np.full(16, 123, np.float32)
    assert function(None, ptr(target), 2, 3, 1) == 1
    assert function(ptr(source), ptr(target), 0, 3, 1) == 1
    assert function(ptr(source), ptr(target), 2, 3, 3) == 1
    assert function(ptr(source), ptr(target), ct.c_size_t(-1).value, 4, 2) == 2
    assert function(ptr(source), ptr(target), 2, ct.c_size_t(-1).value, 2) == 2
    np.testing.assert_array_equal(target, 123)


@pytest.mark.parametrize("colors", [1, 3])
def test_one_row_opencv_matrix_interpretation(colors):
    source = np.array([[0.1, 0.5, 0.9]], np.float32)
    exact(native_palette.kmeans(source, colors), reference(source, colors))


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("colors", [1, 2, 5, 17])
@pytest.mark.parametrize(
    "pattern", ["nan", "all_nan", "inf", "all_inf", "both_inf", "huge", "one_nan"]
)
def test_median_cut_nonfinite_contract_in_c(channels, colors, pattern):
    data = np.random.default_rng(392).random((1, 13, channels), dtype=np.float32)
    if pattern == "nan":
        data[0, 6, -1] = np.nan
    elif pattern == "all_nan":
        data[:] = np.nan
    elif pattern == "inf":
        data[0, 6, -1] = np.inf
    elif pattern == "all_inf":
        data[:] = np.inf
    elif pattern == "both_inf":
        data[0, 2, 0] = np.inf
        data[0, 8, 0] = -np.inf
    elif pattern == "huge":
        data *= np.finfo(np.float32).max
    elif pattern == "one_nan":
        data = data[:, :1]
        data[:] = np.nan
    with np.errstate(all="ignore"):
        try:
            expected = ORIGINAL.median_cut_palette(data, colors)
        except ValueError:
            with pytest.raises(ValueError):
                palette.median_cut_palette(data, colors)
        else:
            actual = palette.median_cut_palette(data, colors)
            np.testing.assert_array_equal(actual, expected)
            # NaN payloads are not part of NumPy's reduction contract.
            exact(actual[~np.isnan(actual)], expected[~np.isnan(expected)])


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 17])
@pytest.mark.parametrize("size", [1, 3, 7, 4096])
@pytest.mark.parametrize("extra", [0, 1, 9])
def test_padding_has_original_values_and_owned_storage(channels, size, extra):
    source = np.random.default_rng(285).random((1, size, channels), dtype=np.float32)
    source[0, 0, 0] = -0.0
    source.flags.writeable = False
    expected = np.pad(source, [(0, 0), (0, extra), (0, 0)], mode="edge")
    result = native_palette.pad_edge(source, size + extra)
    exact(result, expected)
    assert not np.shares_memory(source, result)


def test_median_and_padding_do_not_call_numpy_algorithms(monkeypatch):
    source = np.random.default_rng(918).random((1, 27, 3), dtype=np.float32)
    expected = ORIGINAL.median_cut_palette(source, 5)
    expected_padding = np.pad(expected, [(0, 0), (0, 8), (0, 0)], mode="edge")

    def forbidden(*args, **kwargs):
        raise AssertionError("Palette algorithms must execute in C")

    for name in ("min", "max", "median", "mean", "pad"):
        monkeypatch.setattr(np, name, forbidden)
    actual = palette.median_cut_palette(source, 5)
    padded = native_palette.pad_edge(actual, 13)
    exact(actual, expected)
    exact(padded, expected_padding)
