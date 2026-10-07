"""CPU parity checks against frozen pre-conversion resize/border code."""

from __future__ import annotations

import ast
import ctypes as ct
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from nodes.impl import native_resample as native
from nodes.impl import resize as current
from nodes.impl.native import lib, ptr

ROOT = Path(__file__).resolve().parents[2]
REF = Path(__file__).with_name("reference_resample")
NODES = ROOT / "backend/src/packages/chaiNNer_standard"


def load_body(path, name, package=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if package is None:
        body = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                if node.level or node.module in ("api", "nodes.groups"):
                    continue
                if (node.module or "").startswith("nodes.properties"):
                    continue
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
            body.append(node)
        tree.body = body
    module = types.ModuleType(name)
    module.__package__ = package
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


reference = load_body(REF / "resize.py", "reference_resample_resize", "nodes.impl")
border = load_body(
    NODES / "image_dimension/crop/crop_border.py", "current_resample_border"
)
border_ref = load_body(REF / "crop_border.py", "reference_resample_border")
resize_node = load_body(
    NODES / "image_dimension/resize/resize.py", "current_resize_node"
)
resize_node_ref = load_body(REF / "resize_node.py", "reference_resize_node")
side_node = load_body(
    NODES / "image_dimension/resize/resize_to_side.py", "current_resize_side"
)
side_node_ref = load_body(REF / "resize_to_side.py", "reference_resize_side")
for module in (resize_node_ref, side_node_ref):
    module.__dict__.update(resize=reference.resize)
    module.__dict__.update(ResizeFilter=reference.ResizeFilter)

LAYOUTS = ("contiguous", "strided", "transpose", "readonly", "unaligned")


def with_layout(data, layout):
    if layout == "strided":
        backing = np.empty(
            (data.shape[0] * 2, data.shape[1] * 2, *data.shape[2:]), data.dtype
        )
        backing[::2, ::2] = data
        data = backing[::2, ::2][::-1]
    elif layout == "transpose":
        data = data.swapaxes(0, 1)
    elif layout == "readonly":
        data.setflags(write=False)
    elif layout == "unaligned":
        out = np.ndarray(
            data.shape, data.dtype, buffer=bytearray(data.nbytes + 1), offset=1
        )
        out[:] = data
        data = out
    return data


def image(channels=4, layout="contiguous", shape=(7, 11), seed=17):
    shape = shape if channels == 0 else (*shape, channels)
    data = np.random.default_rng(seed).random(shape, dtype=np.float32)
    if channels == 4:
        data[::3, ::2, 3] = 0
        data[1::3, 1::2, 3] = np.float32(0.0001)
    return with_layout(data, layout)


def equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("channels", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dimensions", [(1, 1), (3, 5), (13, 19), (22, 14)])
@pytest.mark.parametrize("gamma", [False, True])
def test_nearest_images(channels, layout, dimensions, gamma):
    source = image(channels, layout)
    before = source.copy()
    expected = reference.resize(
        source, dimensions, reference.ResizeFilter.NEAREST, False, gamma
    )
    actual = current.resize(
        source, dimensions, current.ResizeFilter.NEAREST, False, gamma
    )
    equal(actual, expected)
    equal(source, before)
    assert not np.shares_memory(actual, source)


@pytest.mark.parametrize("source_width", range(1, 34))
@pytest.mark.parametrize("out_width", range(1, 34))
def test_nearest_fixed_point_ties(source_width, out_width):
    source = np.arange(source_width, dtype=np.float32).reshape(1, source_width, 1)
    equal(
        native.nearest(source, (out_width, 1)),
        reference.native_resize(
            source, (out_width, 1), reference.NativeResizeFilter.Nearest, False
        ),
    )


@pytest.mark.parametrize("filter_value", [f.value for f in current.ResizeFilter])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dimensions", [(3, 5), (13, 19), (11, 7)])
@pytest.mark.parametrize("gamma", [False, True])
@pytest.mark.parametrize("separate", [False, True])
def test_all_resize_filters_alpha(filter_value, layout, dimensions, gamma, separate):
    source = image(4, layout)
    before = source.copy()
    expected = reference.resize(
        source, dimensions, reference.ResizeFilter(filter_value), separate, gamma
    )
    actual = current.resize(
        source, dimensions, current.ResizeFilter(filter_value), separate, gamma
    )
    equal(actual, expected)
    equal(source, before)
    assert not np.shares_memory(actual, source)


@pytest.mark.parametrize("channels", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("filter_value", [0, 4])
def test_same_size_copy_semantics(channels, filter_value):
    source = image(channels, "readonly")
    actual = current.resize(
        source, (source.shape[1], source.shape[0]), current.ResizeFilter(filter_value)
    )
    equal(actual, source)
    assert actual is not source
    assert not np.shares_memory(actual, source)
    assert actual.flags.writeable


@pytest.mark.parametrize("layout", LAYOUTS)
def test_special_values_alpha_and_nearest(layout):
    data = np.array(
        [
            [np.nan, -np.inf, np.inf, 0],
            [-0.0, 0.25, 2, -0.0],
            [1, -1, 0.125, np.nan],
            [1, 2, 3, np.inf],
            [1, 2, 3, -np.inf],
            [-1, 0.25, 2, 0.00001],
            [1, 1, 1, np.nextafter(np.float32(0.0001), np.float32(0))],
        ],
        np.float32,
    ).reshape(1, 7, 4)
    data = with_layout(data, layout)
    with np.errstate(all="ignore"):
        expected = data.copy()
        for channel in range(3):
            expected[..., channel] *= data[..., 3]
        actual = native.premultiply(data)
        equal(actual, expected)
        alpha_r = 1 / np.maximum(expected[..., 3], 0.0001)
        for channel in range(3):
            expected[..., channel] *= alpha_r
        np.minimum(expected, 1, out=expected)
        finished = native.finish_alpha_inplace(actual)
        equal(finished, expected)
        np.testing.assert_array_equal(np.signbit(finished), np.signbit(expected))
        equal(
            native.nearest(data, (13, 9)),
            reference.resize(data, (13, 9), reference.ResizeFilter.NEAREST),
        )


@pytest.mark.parametrize("selection", [1, 2, 3])
@pytest.mark.parametrize("channels", [0, 2, 3, 4])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("tolerance", [0, 5, 30])
@pytest.mark.parametrize("padding", [0, 2, 1000])
def test_border_sections(selection, channels, layout, tolerance, padding):
    shape = (23, 31) if channels == 0 else (23, 31, channels)
    source = np.full(shape, 0.125, np.float32)
    source[2:4, 2:7] = 0.75
    source[8:19, 10:22] = 1
    source[19:21, 25:27] = 0.6
    source = with_layout(source, layout)
    before = source.copy()
    actual = border.crop_border_node(
        source, tolerance, border.SelectMode(selection), padding
    )
    expected = border_ref.crop_border_node(
        source, tolerance, border_ref.SelectMode(selection), padding
    )
    equal(actual, expected)
    equal(source, before)
    assert np.shares_memory(actual, source)
    assert (actual is source) == (expected is source)


@pytest.mark.parametrize("selection", [1, 2, 3])
@pytest.mark.parametrize("channels", [0, 2, 3, 4])
@pytest.mark.parametrize("shape", [(1, 1), (1, 11), (13, 1), (5, 7), (7, 11)])
@pytest.mark.parametrize("kind", ["random", "constant", "nonfinite", "threshold"])
def test_border_median_boundaries(selection, channels, shape, kind):
    source = image(channels, shape=shape)
    if kind == "constant":
        source.fill(-0.0)
    elif kind == "nonfinite":
        source.flat[::7] = np.nan
        source.flat[1::13] = np.inf
    elif kind == "threshold":
        source.fill(0)
        source[1:-1, 1:-1] = np.nextafter(np.float32(0.05), np.float32(1))
    with np.errstate(all="ignore"):
        actual = border.crop_border_node(source, 5, border.SelectMode(selection), 0)
        expected = border_ref.crop_border_node(
            source, 5, border_ref.SelectMode(selection), 0
        )
    equal(actual, expected)
    assert (actual is source) == (expected is source)


@pytest.mark.parametrize("side", list(side_node.SideSelection))
@pytest.mark.parametrize("condition", list(side_node.ResizeCondition))
@pytest.mark.parametrize("target", [3, 20])
def test_resize_to_side_adapter(side, condition, target):
    source = image()
    actual = side_node.resize_to_side_node(
        source, target, side, condition, current.ResizeFilter.NEAREST
    )
    expected = side_node_ref.resize_to_side_node(
        source,
        target,
        side_node_ref.SideSelection(side.value),
        side_node_ref.ResizeCondition(condition.value),
        reference.ResizeFilter.NEAREST,
    )
    equal(actual, expected)


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("separate", [False, True])
def test_resize_adapter(mode, separate):
    source = image()
    actual = resize_node.resize_node(
        source,
        resize_node.ImageResizeMode(mode),
        175.25,
        17,
        3,
        current.ResizeFilter.LINEAR,
        separate,
    )
    expected = resize_node_ref.resize_node(
        source,
        resize_node_ref.ImageResizeMode(mode),
        175.25,
        17,
        3,
        reference.ResizeFilter.LINEAR,
        separate,
    )
    equal(actual, expected)


@pytest.mark.parametrize(
    "data",
    [
        np.zeros((2, 3), np.float64),
        np.zeros((0, 3), np.float32),
        np.zeros((3,), np.float32),
        np.zeros((2, 3, 4, 5), np.float32),
        np.zeros((2, 3, 5), np.float32),
    ],
)
@pytest.mark.parametrize(
    "operation", ["nearest", "premultiply", "finish_alpha_inplace", "border_region"]
)
def test_invalid_shapes_before_dispatch(data, operation, monkeypatch):
    if operation == "border_region" and data.shape == (2, 3, 5):
        assert native.border_region(data, 0.05, 1) == (0, 0, 3, 2)
        return
    monkeypatch.setattr(native, "_api", lambda: pytest.fail("Invalid input reached C"))
    arguments = {
        "nearest": ((3, 4),),
        "premultiply": (),
        "finish_alpha_inplace": (),
        "border_region": (0.05, 1),
    }[operation]
    with pytest.raises((TypeError, ValueError)):
        getattr(native, operation)(data, *arguments)


@pytest.mark.parametrize(
    "dimensions", [(0, 1), (-1, 2), (2**32, 1), (1, 2**32), (2**20, 2**20), (1.5, 3)]
)
def test_invalid_dimensions_before_dispatch(dimensions, monkeypatch):
    monkeypatch.setattr(native, "_api", lambda: pytest.fail("Invalid size reached C"))
    with pytest.raises((TypeError, ValueError, OverflowError)):
        native.nearest(image(), dimensions)


def test_inplace_rejects_readonly():
    with pytest.raises(ValueError):
        native.finish_alpha_inplace(image(layout="readonly"))


@pytest.mark.parametrize("dimensions", [(0, 1), (1, 0), (0, 0)])
def test_empty_helper_resize_retains_rust_semantics(dimensions):
    source = image()
    equal(
        current.resize(source, dimensions, current.ResizeFilter.NEAREST),
        reference.resize(source, dimensions, reference.ResizeFilter.NEAREST),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_singleton_channel_border_retains_original_assertion(dtype):
    source = np.zeros((3, 5, 1), dtype)
    for module in (border, border_ref):
        with pytest.raises(AssertionError):
            module.crop_border_node(source, 5, module.SelectMode.ALL_SECTIONS, 0)


@pytest.mark.parametrize("channels", [0, 2, 3, 4, 8])
def test_float64_border_retained(channels):
    source = image(channels).astype(np.float64)
    equal(
        border.crop_border_node(source, 5, border.SelectMode.LARGEST_SECTION, 0),
        border_ref.crop_border_node(
            source, 5, border_ref.SelectMode.LARGEST_SECTION, 0
        ),
    )


def test_large_channel_border_retained():
    source = image(8)
    equal(
        border.crop_border_node(source, 5, border.SelectMode.LARGEST_SECTION, 0),
        border_ref.crop_border_node(
            source, 5, border_ref.SelectMode.LARGEST_SECTION, 0
        ),
    )


def test_raw_c_boundaries():
    data = np.zeros((1, 1, 4), np.float32)
    native.nearest(data, (1, 1))
    dll = lib()
    p = ptr(data)
    assert dll.cn_resample_nearest(p, p, 1, 1, 4, 0, 1) == 1
    assert dll.cn_resample_nearest(p, p, 2**31, 1, 1, 1, 1) == 2
    assert dll.cn_resample_alpha(p, p, 1, 2) == 1
    assert dll.cn_resample_alpha(p, p, ct.c_size_t(-1).value, 0) == 2
    bounds = (ct.c_size_t * 4)()
    assert dll.cn_resample_border(p, 1, 1, 4, 0, 0, bounds, 1) == 1
    assert dll.cn_resample_border(p, 2**62, 2, 4, 0, 1, bounds, 1) == 2


def test_parallel_calls_and_large_worker_ranges():
    source = image(shape=(257, 263))
    expected_resize = reference.resize(
        source, (401, 269), reference.ResizeFilter.NEAREST
    )
    border_image = np.full((271, 277, 4), 0.125, np.float32)
    border_image[7:-9, 11:-13] = 0.875
    expected_border = border_ref.crop_border_node(
        border_image, 5, border_ref.SelectMode.LARGEST_SECTION, 2
    )

    def run(index):
        if index % 2:
            equal(
                current.resize(source, (401, 269), current.ResizeFilter.NEAREST),
                expected_resize,
            )
        else:
            equal(
                border.crop_border_node(
                    border_image, 5, border.SelectMode.LARGEST_SECTION, 2
                ),
                expected_border,
            )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(run, range(24)))


@pytest.mark.parametrize(
    "original,relative",
    [
        ("crop_border.py", "image_dimension/crop/crop_border.py"),
        ("resize_node.py", "image_dimension/resize/resize.py"),
        ("resize_to_side.py", "image_dimension/resize/resize_to_side.py"),
    ],
)
def test_installed_metadata_unchanged(original, relative):
    a = ast.parse((REF / original).read_text(encoding="utf-8"))
    b = ast.parse((NODES / relative).read_text(encoding="utf-8"))
    for before in a.body:
        if isinstance(before, ast.FunctionDef) and before.decorator_list:
            after = next(
                node
                for node in b.body
                if isinstance(node, ast.FunctionDef) and node.name == before.name
            )
            assert ast.dump(before.args) == ast.dump(after.args)
            assert [ast.dump(d) for d in before.decorator_list] == [
                ast.dump(d) for d in after.decorator_list
            ]


def test_border_mean_past_8192_channels():
    # NumPy 2.5.3 means each pixel's channel differences in one inner call: one
    # pairwise tree, even past 8192 channels (NumPy 1.24 cut them into 8192-element
    # buffers). Seed 34's pixel is where the two differ, so a tolerance one float
    # below the tree's mean separates them.
    source = np.zeros((5, 6, 20000), np.float32)
    source[2, 3] = np.random.default_rng(34).random(20000, dtype=np.float32)
    tree = np.mean(source, axis=-1)[2, 3]
    for tolerance in (tree, np.nextafter(tree, np.float32(0))):
        percent = float(tolerance) * 100
        assert np.float32(percent / 100) == tolerance
        actual = border.crop_border_node(source, percent, border.SelectMode(1), 0)
        expected = border_ref.crop_border_node(
            source, percent, border_ref.SelectMode(1), 0
        )
        equal(actual, expected)
