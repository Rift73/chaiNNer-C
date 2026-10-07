"""Exact CPU regressions for previously retained sharpening/contrast branches."""

from __future__ import annotations

import ast
import ctypes as ct
import math
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from nodes.impl import native_transfer, native_unsharp
from nodes.impl.native import ptr
from nodes.utils.utils import get_h_w_c

ROOT = Path(__file__).resolve().parents[2]


def load_body(path: Path) -> types.ModuleType:
    tree = ast.parse(path.read_text("utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.ImportFrom)
        or (
            node.level == 0
            and node.module not in {"api", "nodes.groups"}
            and not (node.module or "").startswith("nodes.properties")
        )
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    result = types.ModuleType(path.stem)
    exec(compile(tree, str(path), "exec"), result.__dict__)
    return result


STRETCH = load_body(
    ROOT
    / "backend/src/packages/chaiNNer_standard/image_adjustment/adjustments/stretch_contrast.py"
)
ORIGINAL_STRETCH = load_body(
    Path(__file__).with_name("reference_adjustments") / "stretch_contrast.py"
)
UNSHARP = load_body(
    ROOT / "backend/src/packages/chaiNNer_standard/image_filter/sharpen/unsharp_mask.py"
)
ORIGINAL_UNSHARP = load_body(
    Path(__file__).with_name("reference_analysis")
    / "packages/chaiNNer_standard/image_filter/sharpen/unsharp_mask.py"
)
_gaussian_path = (
    Path(__file__).with_name("reference_normal_complete") / "image_utils.py"
)
_gaussian_tree = ast.parse(_gaussian_path.read_text("utf-8"))
_gaussian_tree.body = [
    node
    for node in _gaussian_tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "fast_gaussian_blur"
]
_gaussian_globals = {"np": np, "cv2": cv2, "math": math, "get_h_w_c": get_h_w_c}
exec(compile(_gaussian_tree, str(_gaussian_path), "exec"), _gaussian_globals)
ORIGINAL_UNSHARP.__dict__.update(
    fast_gaussian_blur=_gaussian_globals["fast_gaussian_blur"]
)


def exact(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    keep = ~np.isnan(expected)
    kind = np.uint32 if actual.dtype == np.float32 else np.uint64
    np.testing.assert_array_equal(actual.view(kind)[keep], expected.view(kind)[keep])


def layout(image: np.ndarray, kind: str) -> np.ndarray:
    if kind == "reverse":
        return image[::-1, ::-1]
    if kind == "fortran":
        return np.asfortranarray(image)
    if kind == "rows":
        return np.repeat(image, 2, axis=0)[::2]
    if kind == "columns":
        return np.repeat(image, 2, axis=1)[:, ::2]
    if kind == "unaligned":
        result = np.ndarray(image.shape, np.float32, bytearray(image.nbytes + 1), 1)
        result[:] = image
        return result
    image.flags.writeable = False
    return image


@pytest.mark.parametrize("optimized", [True, False])
@pytest.mark.parametrize("channels", [1, 3, 4, 5, 17])
@pytest.mark.parametrize("amount", [0.1, 0.3, 1, 1.3, 9.7, 100])
@pytest.mark.parametrize(
    "kind", ["plain", "rows", "columns", "reverse", "fortran", "unaligned"]
)
def test_weighted_exact(optimized, channels, amount, kind):
    rng = np.random.default_rng(1429)
    shape = (9, 19) if channels == 1 else (9, 19, channels)
    image = layout(rng.uniform(-2, 3, shape).astype(np.float32), kind)
    blurred = layout(rng.uniform(-2, 3, shape).astype(np.float32), kind)
    prior = cv2.useOptimized()
    cv2.setUseOptimized(optimized)
    try:
        expected = cv2.addWeighted(image, amount + 1, blurred, -amount, 0)
        exact(native_unsharp.weighted(image, blurred, amount), expected)
    finally:
        cv2.setUseOptimized(prior)


@pytest.mark.parametrize("amount", [0, 0.1, 0.3, 1, 1.7, 100])
@pytest.mark.parametrize("width", [1, 3, 7, 8, 15, 16, 17, 35, 257])
def test_weighted_extremes(amount, width):
    values = np.array(
        [
            0,
            -0.0,
            1,
            -1,
            np.inf,
            -np.inf,
            np.nan,
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
            np.nextafter(np.float32(0), np.float32(1)),
        ],
        np.float32,
    )
    rng = np.random.default_rng(175)
    a, b = (values[rng.integers(0, len(values), (3, width))] for _ in range(2))
    exact(
        native_unsharp.weighted(a, b, amount),
        cv2.addWeighted(a, amount + 1, b, -amount, 0),
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("radius", [0.1, 1.3, 17.5])
@pytest.mark.parametrize("amount", [0.3, 1, 9.7])
def test_unsharp_whole_node(channels, radius, amount):
    shape = (11, 19) if channels == 1 else (11, 19, channels)
    image = np.random.default_rng(102).random(shape, dtype=np.float32)
    expected = ORIGINAL_UNSHARP.unsharp_mask_node(image, radius, amount, 0)
    exact(UNSHARP.unsharp_mask_node(image, radius, amount, 0), expected)


@pytest.mark.parametrize("count", [1, 2, 7, 31, 129, 8191, 8192, 8193, 16387])
@pytest.mark.parametrize("percentile", [None, 0, 0.1, 1.75, 13.3, 49.99, 50])
@pytest.mark.parametrize(
    "values", [[-0.0, 0.0], [-np.inf, -1, -0.0, 0.0, 1, np.inf], [0, np.nan, 1]]
)
def test_bounds_exceptional(count, percentile, values):
    rng = np.random.default_rng(135)
    image = np.array(values, np.float32)[rng.integers(0, len(values), (1, count))]
    before = image.copy()
    with np.errstate(all="ignore"):
        expected = (
            (np.min(image), np.max(image))
            if percentile is None
            else tuple(np.percentile(image, q) for q in (percentile, 100 - percentile))
        )
        actual = native_transfer.contrast_bounds(image, percentile)
    exact(np.array(actual, np.float64), np.array(expected, np.float64))
    exact(image, before)


@pytest.mark.parametrize(
    "kind", ["plain", "rows", "columns", "reverse", "fortran", "unaligned"]
)
@pytest.mark.parametrize("percentile", [None, 0, 12.5, 50])
def test_bounds_foreign_layouts(kind, percentile):
    rng = np.random.default_rng(258)
    values = np.array([-0.0, 0.0], np.float32)
    image = layout(values[rng.integers(0, 2, (47, 39, 3))], kind)
    expected = (
        (np.min(image), np.max(image))
        if percentile is None
        else tuple(np.percentile(image, q) for q in (percentile, 100 - percentile))
    )
    exact(
        np.array(native_transfer.contrast_bounds(image, percentile)),
        np.array(expected, np.float64),
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("keep_colors", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        [-0.0, 0.0],
        [-np.inf, -1, -0.0, 0.0, 1, np.inf],
        [0, np.nan, 1],
        [-3.4e38, -1, 1, 3.4e38],
    ],
)
def test_stretch_node_exceptional(channels, mode, keep_colors, values):
    shape = (17, 23) if channels == 1 else (17, 23, channels)
    rng = np.random.default_rng(812)
    image = np.array(values, np.float32)[rng.integers(0, len(values), shape)]
    with np.errstate(all="ignore"):
        expected = ORIGINAL_STRETCH.stretch_contrast_node(
            image, ORIGINAL_STRETCH.StretchMode(mode), keep_colors, 1.5, 0, 255
        )
        actual = STRETCH.stretch_contrast_node(
            image, STRETCH.StretchMode(mode), keep_colors, 1.5, 0, 255
        )
    exact(actual, expected)


@pytest.mark.parametrize(
    "kind", ["plain", "rows", "columns", "reverse", "fortran", "unaligned"]
)
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("wide_channel", [0, 1, 2])
def test_stretch_mixed_precision_channels(kind, mode, wide_channel):
    image = np.random.default_rng(134).random((11, 19, 4), dtype=np.float32)
    image[:, ::2, wide_channel] = -3.4e38
    image[:, 1::2, wide_channel] = 3.4e38
    image = layout(image, kind)
    prior = image.copy()
    with np.errstate(all="ignore"):
        expected = ORIGINAL_STRETCH.stretch_contrast_node(
            image, ORIGINAL_STRETCH.StretchMode(mode), False, 0, 0, 255
        )
        actual = STRETCH.stretch_contrast_node(
            image, STRETCH.StretchMode(mode), False, 0, 0, 255
        )
    exact(actual, expected)
    exact(image, prior)


def test_concurrent_and_no_retained_image_computation(monkeypatch):
    a = np.random.default_rng(442).random((231, 267, 3), dtype=np.float32)
    b = a[::-1]
    expected = native_unsharp.weighted(a, b, 0.3)
    bounds = native_transfer.contrast_bounds(a, 1.75)

    def blocked(*_args, **_kwargs):
        pytest.fail("Valid float32 computation returned to an unconverted primitive")

    for name in ("percentile", "min", "max", "dstack"):
        monkeypatch.setattr(np, name, blocked)
    monkeypatch.setattr(cv2, "addWeighted", blocked)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(
            pool.map(
                lambda _: (
                    native_unsharp.weighted(a, b, 0.3),
                    native_transfer.contrast_bounds(a, 1.75),
                ),
                range(8),
            )
        )
    for image, result in outputs:
        exact(image, expected)
        assert result == bounds


def test_checked_abis():
    a = np.ones((5, 7), np.float32)
    b = a.copy()
    output = np.empty_like(a)
    call = native_unsharp._api().cn_unsharp_weighted
    args = (ptr(a), ptr(b), ptr(output), a.size, a.size, 1.3, 8)
    assert call(None, *args[1:]) == 1
    assert call(ptr(a), ptr(b), ptr(a), *args[3:]) == 1
    shifted = ct.cast(a.ctypes.data + 4, ct.POINTER(ct.c_float))
    assert call(ptr(a), ptr(b), shifted, *args[3:]) == 1
    unaligned = ct.cast(a.ctypes.data + 1, ct.POINTER(ct.c_float))
    assert call(unaligned, *args[1:]) == 1
    assert call(*args[:3], ct.c_size_t(-1).value, 1, 1.3, 8) == 2
    assert call(*args[:4], 0, 1.3, 8) == 1
    result = np.empty(2, np.float64)
    bounds = native_transfer._lib.cn_contrast_bounds_complete
    dp = result.ctypes.data_as(ct.POINTER(ct.c_double))
    assert bounds(None, a.size, 0, 0, 16, 0, 0, dp) == 1
    assert bounds(unaligned, a.size, 0, 0, 16, 0, 0, dp) == 1
    assert bounds(ptr(a), ct.c_size_t(-1).value, 0, 0, 16, 0, 0, dp) == 2
    assert (
        bounds(
            ptr(a),
            a.size,
            0,
            0,
            16,
            0,
            0,
            ct.cast(a.ctypes.data, ct.POINTER(ct.c_double)),
        )
        == 1
    )


def test_wide_checked_abis():
    source = np.ones((5, 7), np.float32)
    output = np.empty(source.shape, np.float64)
    double = ct.POINTER(ct.c_double)
    dp = output.ctypes.data_as(double)
    stretch = native_transfer._lib.cn_contrast_stretch_wide
    assert stretch(None, dp, source.size, 0, 1) == 1
    assert (
        stretch(ptr(source), ct.cast(source.ctypes.data, double), source.size, 0, 1)
        == 1
    )
    assert stretch(ptr(source), dp, ct.c_size_t(-1).value, 0, 1) == 2
    assert (
        stretch(ptr(source), ct.cast(output.ctypes.data + 1, double), source.size, 0, 1)
        == 1
    )
    merge = native_transfer._lib.cn_contrast_merge_wide
    pointers = (ct.c_void_p * 1)(source.ctypes.data)
    kinds = (ct.c_int * 1)(0)
    assert merge(pointers, kinds, dp, source.size, 0) == 1
    assert merge(pointers, kinds, dp, ct.c_size_t(-1).value, 1) == 2
    assert (
        merge(pointers, kinds, ct.cast(source.ctypes.data, double), source.size, 1) == 1
    )
    pointers[0] = source.ctypes.data + 1
    assert merge(pointers, kinds, dp, source.size, 1) == 1
