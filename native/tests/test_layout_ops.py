"""Differential layout tests against untouched original node snapshots."""

from __future__ import annotations

import ast
import ctypes as ct
import types
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from nodes.impl import native_layout as native
from nodes.impl.color.color import Color
from nodes.impl.image_utils import BorderType
from nodes.impl.native import lib, ptr

ROOT = Path(__file__).resolve().parents[2]
NODES = ROOT / "backend/src/packages/chaiNNer_standard"
REF = Path(__file__).with_name("reference_layout")
# Loading the entire input package imports optional Torch backends. This enum
# has no dependencies beyond Enum, so execute only its actual source definition.
_enum_tree = ast.parse(
    (ROOT / "backend/src/nodes/properties/inputs/generic_inputs.py").read_text(
        encoding="utf-8"
    )
)
_enum_class = next(
    node
    for node in _enum_tree.body
    if isinstance(node, ast.ClassDef) and node.name == "OrderEnum"
)
_enum_namespace = {"Enum": Enum}
exec(
    compile(ast.Module(body=[_enum_class], type_ignores=[]), "OrderEnum", "exec"),
    _enum_namespace,
)
PATHS = {
    "pad": "image_dimension/border/pad.py",
    "stack_images": "image_utility/compositing/stack_images.py",
    "z_stack_images": "image_utility/compositing/z_stack_images.py",
    "merge_spritesheet": "image/batch_processing/merge_spritesheet.py",
    "pixelate": "image_filter/miscellaneous/pixelate.py",
    "rotate": "image_utility/modification/rotate.py",
}


def load_body(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module == "nodes.groups":
                continue
            if node.module == "api" or (node.module or "").startswith(
                "nodes.properties"
            ):
                node.names = [
                    name
                    for name in node.names
                    if node.module == "api" and name.name == "Collector"
                ]
                if not node.names:
                    continue
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
        body.append(node)
    tree.body = body
    module = types.ModuleType(path.stem)
    module.__dict__.update(OrderEnum=_enum_namespace["OrderEnum"])
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


MODULES = {
    name: (load_body(NODES / path), load_body(REF / Path(path).name))
    for name, path in PATHS.items()
}
_original_rotate = MODULES["rotate"][1].rotate
_frozen_rotate = next(
    node
    for node in ast.parse((REF / "pil_utils.py").read_text(encoding="utf-8")).body
    if isinstance(node, ast.FunctionDef) and node.name == "rotate"
)
_rotate_globals = dict(_original_rotate.__globals__)
exec(
    compile(
        ast.Module(body=[_frozen_rotate], type_ignores=[]), "original_rotate", "exec"
    ),
    _rotate_globals,
)
_original_rotate = _rotate_globals["rotate"]
_pixels = load_body(REF / "rotation_pixels.py")
MODULES["rotate"][1].__dict__.update(
    rotate=types.FunctionType(
        _original_rotate.__code__,
        {
            **_original_rotate.__globals__,
            "to_uint8": _pixels.to_uint8,
            "normalize": _pixels.normalize,
            "convert_to_bgra": _pixels.convert_to_bgra,
        },
    )
)
LAYOUTS = ("contiguous", "strided", "transpose", "readonly", "unaligned")


def image(channels=3, layout="contiguous", shape=(11, 13), seed=177):
    shape = shape if channels == 1 else (*shape, channels)
    data = np.random.default_rng(seed).random(shape, dtype=np.float32)
    if data.size > 1:
        data.flat[0], data.flat[-1] = 0, 1
    if layout == "strided":
        data = data[::-2, ::2]
    elif layout == "transpose":
        data = data.swapaxes(0, 1)
    elif layout == "readonly":
        data.setflags(write=False)
    elif layout == "unaligned":
        unaligned = np.ndarray(
            data.shape, np.float32, buffer=bytearray(data.nbytes + 1), offset=1
        )
        unaligned[:] = data
        data = unaligned
    return data


def equal(actual, expected, inputs):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    assert not any(np.shares_memory(actual, item) for item in inputs)


@pytest.mark.parametrize("name", PATHS)
def test_metadata_unchanged(name):
    original = ast.parse((REF / Path(PATHS[name]).name).read_text(encoding="utf-8"))
    current = ast.parse((NODES / PATHS[name]).read_text(encoding="utf-8"))
    a = [node for node in original.body if isinstance(node, ast.FunctionDef)][-1]
    b = [node for node in current.body if isinstance(node, ast.FunctionDef)][-1]
    assert ast.dump(a.args) == ast.dump(b.args)
    assert [ast.dump(d) for d in a.decorator_list] == [
        ast.dump(d) for d in b.decorator_list
    ]


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("border", list(BorderType))
@pytest.mark.parametrize(
    "color",
    [
        Color.gray(0.37),
        Color.bgr((0.13, 0.41, 0.87)),
        Color.bgra((0.19, 0.31, 0.79, 0.63)),
    ],
)
@pytest.mark.parametrize("mode", ["BORDER", "EDGES", "OFFSETS"])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "readonly"])
def test_pad(channels, border, color, mode, layout):
    data = image(channels, layout)
    before = data.copy()
    modules = MODULES["pad"]
    results = [
        module.pad_node(
            data, border, color, getattr(module.BorderMode, mode), 3, 2, 3, 4, 5, 12, 10
        )
        for module in modules
    ]
    actual, expected = results
    equal(actual, expected, [data])
    np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("border", list(BorderType))
def test_pad_zero_identity(channels, border):
    data = image(channels, "readonly")
    for module in MODULES["pad"]:
        assert (
            module.pad_node(
                data,
                border,
                Color.bgra((1, 0, 1, 0)),
                module.BorderMode.BORDER,
                0,
                0,
                0,
                0,
                0,
                13,
                11,
            )
            is data
        )


@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (4, 1), (2, 3)])
@pytest.mark.parametrize("border", list(BorderType))
def test_far_reflection_and_offset_crop(shape, border):
    data = image(4, shape=shape)
    for offset in [(0, 0), (21, 17)]:
        results = [
            module.pad_node(
                data,
                border,
                Color.gray(0.29),
                module.BorderMode.OFFSETS,
                0,
                offset[0],
                offset[1],
                0,
                0,
                2,
                3,
            )
            for module in MODULES["pad"]
        ]
        if results[0] is data:
            assert results[1] is data
        else:
            actual, expected = results
            equal(actual, expected, [data])


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("orientation", ["HORIZONTAL", "VERTICAL"])
@pytest.mark.parametrize("channels", [(1, 1), (3, 3), (4, 4), (1, 3, 4), (2, 4), (3,)])
@pytest.mark.parametrize("different_sizes", [False, True])
def test_stack(layout, orientation, channels, different_sizes):
    inputs = [
        image(
            c, layout, (7 + i * 2, 11 - i * 2) if different_sizes else (9, 13), 181 + i
        )
        for i, c in enumerate(channels)
    ]
    saved = [item.copy() for item in inputs]
    current, reference = MODULES["stack_images"]
    try:
        expected = reference.stack_images_node(
            getattr(reference.Orientation, orientation), *inputs
        )
    except ValueError as error:
        with pytest.raises(ValueError, match="broadcast"):
            current.stack_images_node(
                getattr(current.Orientation, orientation), *inputs
            )
        assert "broadcast" in str(error)
    else:
        actual = current.stack_images_node(
            getattr(current.Orientation, orientation), *inputs
        )
        equal(actual, expected, inputs)
    for item, before in zip(inputs, saved, strict=False):
        np.testing.assert_array_equal(item, before)


@pytest.mark.parametrize("orientation", ["HORIZONTAL", "VERTICAL"])
def test_stack_nearest_rounding(orientation):
    current, reference = MODULES["stack_images"]
    for small in range(1, 18):
        for large in (small, small + 1, 23, 29):
            inputs = [image(3, shape=(small, 7)), image(3, shape=(large, 11))]
            if orientation == "VERTICAL":
                inputs = [item.swapaxes(0, 1) for item in inputs]
            expected = reference.stack_images_node(
                getattr(reference.Orientation, orientation), *inputs
            )
            actual = current.stack_images_node(
                getattr(current.Orientation, orientation), *inputs
            )
            equal(actual, expected, inputs)


@pytest.mark.parametrize("mode", ["MEAN", "MEDIAN", "MIN", "MAX"])
@pytest.mark.parametrize("count", [2, 3, 7, 8, 9, 14, 15])
@pytest.mark.parametrize(
    "channels,shape", [(1, (1, 1)), (1, (11, 13)), (3, (11, 13)), (4, (11, 13))]
)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "readonly"])
def test_z_stack(mode, count, channels, shape, layout):
    inputs = [image(channels, layout, shape, 191 + i) for i in range(count)]
    results = [
        module.z_stack_images_node(getattr(module.Expression, mode), *inputs)
        for module in MODULES["z_stack_images"]
    ]
    actual, expected = results
    equal(actual, expected, inputs)


@pytest.mark.parametrize("mode", ["MEAN", "MEDIAN", "MIN", "MAX"])
@pytest.mark.parametrize("special", [np.nan, np.inf, -np.inf, -0.0])
def test_z_stack_nonfinite(mode, special):
    inputs = [image(4, seed=199 + i) for i in range(8)]
    inputs[3][4, 5] = special
    with np.errstate(all="ignore"):
        results = [
            module.z_stack_images_node(getattr(module.Expression, mode), *inputs)
            for module in MODULES["z_stack_images"]
        ]
    actual, expected = results
    equal(actual, expected, inputs)


@pytest.mark.parametrize(
    ("mode", "nans"),
    # Which of several NaNs a sum returns follows the compiler's operand order
    # (out of contract), so MEAN takes one NaN per lane.
    [("MEAN", 1), ("MEDIAN", 1), ("MEDIAN", 3), ("MIN", 1), ("MIN", 3), ("MAX", 3)],
)
@pytest.mark.parametrize("count", [2, 3, 4, 7, 8, 15])
def test_z_stack_nan_payload_bits(mode, nans, count):
    # np.median returns the NaN its partition leaves last, as stored; minimum and
    # maximum keep the first NaN they meet.
    payloads = np.array([0x7FC12345, 0xFFE00001, 0x7F800001], np.uint32)
    rng = np.random.default_rng(613 + count)
    inputs = [image(3, seed=640 + i) for i in range(count)]
    for pixel in range(40):
        y, x, c = pixel % 11, pixel // 11, pixel % 3
        for layer in rng.choice(count, min(nans, count), replace=False):
            inputs[layer][y, x, c] = rng.choice(payloads).view(np.float32)
    with np.errstate(all="ignore"):
        actual, expected = (
            module.z_stack_images_node(getattr(module.Expression, mode), *inputs)
            for module in MODULES["z_stack_images"]
        )
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("mode", ["MEAN", "MEDIAN", "MIN", "MAX"])
@pytest.mark.parametrize("count", [2, 3, 8, 9, 15])
def test_z_stack_negative_zero_bits(mode, count):
    inputs = [np.full((2, 3), -0.0, dtype=np.float32) for _ in range(count)]
    results = [
        module.z_stack_images_node(getattr(module.Expression, mode), *inputs)
        for module in MODULES["z_stack_images"]
    ]
    np.testing.assert_array_equal(
        results[0].view(np.uint32), results[1].view(np.uint32)
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("grid", [(1, 1), (1, 3), (3, 1), (2, 3)])
@pytest.mark.parametrize("order", ["ROW_MAJOR", "COLUMN_MAJOR"])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "readonly"])
def test_spritesheet_collector(channels, grid, order, layout):
    inputs = [
        image(channels, layout, (7, 11), 211 + i) for i in range(grid[0] * grid[1] + 1)
    ]
    results = []
    for module in MODULES["merge_spritesheet"]:
        collector = module.merge_spritesheet_node(
            None, *grid, getattr(module.OrderEnum, order)
        )
        for item in inputs:
            collector.on_iterate(item)
        results.append(collector.on_complete())
    actual, expected = results
    equal(actual, expected, inputs)


@pytest.mark.parametrize("order", ["ROW_MAJOR", "COLUMN_MAJOR"])
def test_spritesheet_nonuniform_tiles(order):
    shapes = [(3, 5), (3, 7), (4, 6), (4, 6)]
    if order == "COLUMN_MAJOR":
        shapes = [(w, h) for h, w in shapes]
    inputs = [image(3, shape=shape) for shape in shapes]
    results = []
    for module in MODULES["merge_spritesheet"]:
        collector = module.merge_spritesheet_node(
            None, 2, 2, getattr(module.OrderEnum, order)
        )
        for item in inputs:
            collector.on_iterate(item)
        results.append(collector.on_complete())
    actual, expected = results
    equal(actual, expected, inputs)


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "sizes", [(1, 1), (2, 3), (3, 2), (8, 8), (13, 11), (16, 7), (129, 5), (5, 129)]
)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_pixelate(channels, sizes, layout):
    data = image(channels, layout)
    saved = data.copy()
    results = [module.pixelate_node(data, *sizes) for module in MODULES["pixelate"]]
    actual, expected = results
    equal(actual, expected, [data])
    np.testing.assert_array_equal(data, saved)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (9, 1)])
@pytest.mark.parametrize("sizes", [(1, 8), (8, 1), (129, 129), (1024, 2), (2, 1024)])
def test_pixelate_degenerate(shape, sizes):
    data = image(1, shape=shape)
    actual, expected = (
        module.pixelate_node(data, *sizes) for module in MODULES["pixelate"]
    )
    equal(actual, expected, [data])


@pytest.mark.parametrize("special", [np.nan, np.inf, -np.inf, -0.0])
def test_pixelate_nonfinite(special):
    data = image(4)
    data[0, 0] = special
    with np.errstate(all="ignore"):
        actual, expected = (
            module.pixelate_node(data, 5, 3) for module in MODULES["pixelate"]
        )
        equal(actual, expected, [data])


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("angle", [0, 90, 180, 270, 360, -90])
@pytest.mark.parametrize("shape", [(7, 11), (9, 9), (1, 1), (1, 7)])
@pytest.mark.parametrize("expand", ["EXPAND", "CROP"])
@pytest.mark.parametrize("fill", ["AUTO", "BLACK", "TRANSPARENT"])
def test_rotate_right_angles(channels, angle, shape, expand, fill):
    data = image(channels, "readonly", shape)
    results = [
        module.rotate_node(
            data,
            angle,
            module.RotationInterpolationMethod.CUBIC,
            getattr(module.RotateSizeChange, expand),
            getattr(module.FillColor, fill),
        )
        for module in MODULES["rotate"]
    ]
    actual, expected = results
    equal(actual, expected, [data])


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("interpolation", ["NEAREST", "LINEAR", "CUBIC"])
def test_rotate_arbitrary_retained(layout, interpolation):
    data = image(4, layout)
    results = [
        module.rotate_node(
            data,
            37.1,
            getattr(module.RotationInterpolationMethod, interpolation),
            module.RotateSizeChange.EXPAND,
            module.FillColor.TRANSPARENT,
        )
        for module in MODULES["rotate"]
    ]
    actual, expected = results
    equal(actual, expected, [data])


@pytest.mark.parametrize("special", [np.nan, np.inf, -np.inf, -0.0, -1.0, 2.0])
def test_rotate_quantization_edges(special):
    data = image(4)
    data[0, 0] = special
    with np.errstate(all="ignore"):
        results = [
            module.rotate_node(
                data,
                90,
                module.RotationInterpolationMethod.NEAREST,
                module.RotateSizeChange.EXPAND,
                module.FillColor.AUTO,
            )
            for module in MODULES["rotate"]
        ]
    actual, expected = results
    equal(actual, expected, [data])


@pytest.mark.parametrize(
    "data",
    [
        np.zeros((), np.float32),
        np.zeros(3, np.float32),
        np.zeros((0, 3), np.float32),
        np.zeros((2, 3, 0), np.float32),
        np.zeros((2, 3, 1, 4), np.float32),
        np.zeros((2, 3), np.float64),
        np.zeros((2, 3), ">f4"),
    ],
)
def test_bad_buffers_do_not_dispatch(data, monkeypatch):
    def forbidden(*args):
        raise AssertionError("Malformed image reached C")

    for name in ("pad", "paste", "pixelate", "zstack", "rotate_u8"):
        monkeypatch.setattr(lib(), "cn_layout_" + name, forbidden)
    calls = [
        lambda: native.pad(data, 0, None, 1, 1, 1, 1),
        lambda: native.stack([data], "horizontal"),
        lambda: native.merge_spritesheet([data], 1, 1, True),
        lambda: native.z_stack([data, data], "mean"),
        lambda: native.pixelate(data, 3, 5),
        lambda: native.rotate_quarters(data, 1, False),
    ]
    for call in calls:
        with pytest.raises((ValueError, TypeError)):
            call()


def test_native_extent_validation():
    value = np.ones(1, np.float32)
    p = ptr(value)
    huge = ct.c_size_t(-1).value
    assert lib().cn_layout_pad(p, p, p, huge, 2, 4, 1, 1, 4, 0, 0, 0) == 2
    assert lib().cn_layout_paste(p, p, 1, 1, 1, 1, 1, 1, 0, 0, huge, 1, 0) == 1
    assert lib().cn_layout_pixelate(p, p, 1, 1, 1, 0, 1) == 1
    pointers = (ct.POINTER(ct.c_float) * 2)(p, p)
    assert lib().cn_layout_zstack(pointers, p, huge, 2, 0) == 2


def test_huge_padding_cannot_wrap_ctypes_integer(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Unrepresentable padding reached C")

    monkeypatch.setattr(lib(), "cn_layout_pad", forbidden)
    with pytest.raises(OverflowError):
        native.pad(image(), 4, None, 2**64, 0, 0, 0, (2, 3))


def test_concurrent_layout_calls():
    data = image(4, "readonly", (257, 263))

    def operation(number):
        if number % 3 == 0:
            results = [
                module.pixelate_node(data, 7, 5) for module in MODULES["pixelate"]
            ]
        elif number % 3 == 1:
            results = [
                module.z_stack_images_node(module.Expression.MEDIAN, data, data, data)
                for module in MODULES["z_stack_images"]
            ]
        else:
            results = [
                module.stack_images_node(module.Orientation.HORIZONTAL, data, data)
                for module in MODULES["stack_images"]
            ]
        actual, expected = results
        equal(actual, expected, [data])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(operation, range(16)))
