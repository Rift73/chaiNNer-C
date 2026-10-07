"""Frozen Apply Palette parity for unsigned casts, index errors and buffer layouts."""

from __future__ import annotations

import ast
import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nodes.impl.native import lib, ptr
from nodes.impl.native_analysis import apply_palette
from nodes.utils.utils import get_h_w_c

PATH = (
    Path(__file__).with_name("reference_analysis")
    / "packages/chaiNNer_standard/image_utility/miscellaneous/apply_palette.py"
)
tree = ast.parse(PATH.read_text(encoding="utf-8"))
tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
for node in tree.body:
    assert isinstance(node, ast.FunctionDef)
    node.decorator_list = []
namespace: dict[str, Any] = {"np": np, "get_h_w_c": get_h_w_c}
exec(compile(tree, str(PATH), "exec"), namespace)
ORIGINAL = namespace["apply_palette_node"]


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def compare(image, palette):
    with np.errstate(all="ignore"):
        try:
            expected = ORIGINAL(image, palette)
        except IndexError:
            with pytest.raises(IndexError):
                apply_palette(image, palette)
        else:
            exact(apply_palette(image, palette), expected)


@pytest.mark.parametrize("levels", [1, 2, 3, 7, 16, 255, 256, 257, 65535, 65536, 65537])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("length", [1, 7, 8, 17, 64, 8193])
@pytest.mark.parametrize("kind", ["normal", "exceptional", "wrapped"])
def test_apply_palette_original_cast(levels, channels, length, kind):
    rng = np.random.default_rng(945)
    image = rng.random((1, length), dtype=np.float32)
    if kind == "exceptional":
        image = np.resize(
            np.array([np.nan, np.inf, -np.inf, -1e38, 1e38, -0.0, 0.0], np.float32),
            (1, length),
        )
    elif kind == "wrapped":
        image = np.resize(
            np.array([-1, 2, -1e7, 1e7, -1e10, 1e10, 0.5], np.float32), (1, length)
        )
    shape = (1, levels) if channels == 1 else (1, levels, channels)
    palette = rng.standard_normal(shape).astype(np.float32)
    palette.flat[: min(5, palette.size)] = [-0.0, 0.0, np.nan, -np.inf, np.inf][
        : min(5, palette.size)
    ]
    compare(image, palette)


@pytest.mark.parametrize(
    "layout", ["reverse", "strided", "transpose", "readonly", "unaligned", "hwc1"]
)
@pytest.mark.parametrize("levels", [1, 2, 255, 256, 257, 65537])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_palette_foreign_buffers(layout, levels, channels):
    rng = np.random.default_rng(283)
    image = rng.random((13, 19), dtype=np.float32)
    shape = (3, levels) if channels == 1 else (3, levels, channels)
    palette = rng.standard_normal(shape).astype(np.float32)
    image.flat[::3] = np.nan
    if layout == "reverse":
        image, palette = image[::-1, ::-1], palette[::-1, ::-1]
    elif layout == "strided":
        image, palette = image[::2, ::2], palette[::2]
    elif layout == "transpose":
        image = image.T
        palette = np.asfortranarray(palette)
    elif layout == "readonly":
        image.flags.writeable = False
        palette.flags.writeable = False
    elif layout == "unaligned":
        other = np.ndarray(image.shape, np.float32, bytearray(image.nbytes + 1), 1)
        other[:] = image
        image = other
        other = np.ndarray(palette.shape, np.float32, bytearray(palette.nbytes + 1), 1)
        other[:] = palette
        palette = other
    else:
        image = image[..., None]
        if channels == 1:
            palette = palette[..., None]
    saved, saved_palette = image.copy(), palette.copy()
    compare(image, palette)
    exact(image, saved)
    exact(palette, saved_palette)


def test_palette_concurrent_casts():
    rng = np.random.default_rng(228)
    image = rng.random((257, 263), dtype=np.float32)
    image.flat[::3] = np.inf
    image.flat[::5] = np.nan
    palette = rng.random((1, 256, 4), dtype=np.float32)
    with np.errstate(invalid="ignore"):
        expected = ORIGINAL(image, palette)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: apply_palette(image, palette), range(8)))
    for actual in results:
        exact(actual, expected)


def test_palette_invalid_indices_do_not_write():
    function = lib().cn_analysis_palette
    image = np.array([[0, 1, -1]], np.float32)
    palette = np.ones((1, 7, 3), np.float32)
    output = np.full((1, 3, 3), 987, np.float32)
    assert function(ptr(image), ptr(palette), ptr(output), 3, 7, 3) == 1
    assert function(ptr(image), ptr(palette), ptr(palette), 3, 7, 3) == 1
    assert function(None, ptr(palette), ptr(output), 3, 7, 3) == 1
    assert (
        function(ptr(image), ptr(palette), ptr(output), ct.c_size_t(-1).value, 7, 3)
        == 2
    )
    np.testing.assert_array_equal(output, 987)
