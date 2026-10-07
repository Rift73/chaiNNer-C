"""Differential node-body tests against the unmodified upstream snapshots.

Decorators alone are stripped to avoid registering a second copy of each node.
The actual original functions and their helper imports execute unchanged.
"""

from __future__ import annotations

import ast
import ctypes as ct
import types
from pathlib import Path

import numpy as np
import pytest

from nodes.impl.native import f32, lib

ROOT = Path(__file__).resolve().parents[2]
NODES = ROOT / "backend/src/packages/chaiNNer_standard/image_adjustment"
REFERENCE = Path(__file__).with_name("reference_adjustments")
PATHS = {
    "add": "arithmetic/add.py",
    "multiply": "arithmetic/multiply.py",
    "divide": "arithmetic/divide.py",
    "invert_color": "adjustments/invert_color.py",
    "clamp": "adjustments/clamp.py",
    "brightness_and_contrast": "adjustments/brightness_and_contrast.py",
    "opacity": "adjustments/opacity.py",
    "color_levels": "adjustments/color_levels.py",
    "premultiplied_alpha": "arithmetic/premultiplied_alpha.py",
    "stretch_contrast": "adjustments/stretch_contrast.py",
    "log_to_linear": "gamma/log_to_linear.py",
}


def load_body(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
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
    module = types.ModuleType(path.stem)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


MODULES = {
    name: (load_body(NODES / path), load_body(REFERENCE / Path(path).name))
    for name, path in PATHS.items()
}


def source(channels, layout):
    shape = (11, 13) if channels == 1 else (11, 13, channels)
    img = np.random.default_rng(173).random(shape, dtype=np.float32)
    img.flat[:6] = [0, 1, 0.5, 0.001, 0.9999, 0.3333]
    if layout == "strided":
        img = img[::2, ::-2]
    elif layout == "transpose":
        img = img.swapaxes(0, 1)
    elif layout == "readonly":
        img.setflags(write=False)
    elif layout == "unaligned":
        view = np.ndarray(
            shape, dtype=np.float32, buffer=bytearray(img.nbytes + 1), offset=1
        )
        view[:] = img
        img = view
    return img


def compare(name, img, args, reference_args=None, *, rtol=0, atol=0):
    native, original = MODULES[name]
    saved = img.copy()
    with np.errstate(all="ignore"):
        # A copy in img's own memory order, whose layout the result follows.
        expected = getattr(original, name + "_node")(
            img.copy(order="K"), *(reference_args or args)
        )
        actual = getattr(native, name + "_node")(img, *args)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.float32
    # Upstream's memory order, as np.clip's order K passes it on (Consult 11 D-16).
    assert np.empty_like(actual, order="K").strides == (
        np.empty_like(expected, order="K").strides
    )
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    np.testing.assert_array_equal(img, saved)
    return actual


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5])
@pytest.mark.parametrize(
    "layout", ["plain", "strided", "transpose", "readonly", "unaligned"]
)
@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("add", [-100]),
        ("add", [13.7]),
        ("multiply", [0]),
        ("multiply", [3.9187]),
        ("divide", [0.0001]),
        ("divide", [3.9187]),
        ("invert_color", []),
        ("clamp", [0.2, 0.8]),
        ("clamp", [0.8, 0.2]),
        ("brightness_and_contrast", [0, 0]),
        ("brightness_and_contrast", [-100, -100]),
        ("brightness_and_contrast", [100, 100]),
        ("brightness_and_contrast", [13.7, -23.4]),
    ],
)
def test_adjustment_nodes(channels, layout, name, args):
    compare(name, source(channels, layout), args)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("value", [0, 37.3, 100])
@pytest.mark.parametrize("layout", ["plain", "strided", "readonly", "unaligned"])
def test_opacity(channels, value, layout):
    compare("opacity", source(channels, layout), [value])


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("mask", [0, 1, 5, 7, 8, 15])
@pytest.mark.parametrize(
    "values",
    [
        [0, 1, 1, 0, 1],
        [0.13, 0.91, 2.2, 0.07, 0.8],
        [0.5, 0.5, 0, 0, 1],
        [0.8, 0.2, 0.9, 0.9, 0.1],
    ],
)
def test_levels(channels, mask, values):
    args = [bool(mask & 4), bool(mask & 2), bool(mask & 1), bool(mask & 8), *values]
    compare("color_levels", source(channels, "strided"), args)


@pytest.mark.parametrize("name", ["PREMULTIPLY_RGB", "UNPREMULTIPLY_RGB"])
def test_alpha_association(name):
    native, original = MODULES["premultiplied_alpha"]
    img = source(4, "readonly").copy()
    img[0, 0, :] = 0
    img[0, 1, 3] = 0
    compare(
        "premultiplied_alpha",
        img,
        [getattr(native.AlphaAssociation, name)],
        [getattr(original.AlphaAssociation, name)],
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("mode", ["AUTO", "PERCENTILE", "MANUAL"])
@pytest.mark.parametrize("keep", [True, False])
@pytest.mark.parametrize("constant", [True, False])
def test_stretch(channels, mode, keep, constant):
    native, original = MODULES["stretch_contrast"]
    img = source(channels, "strided")
    if constant:
        img[:] = 0.5
    args = [keep, 1.5, 20, 200]
    compare(
        "stretch_contrast",
        img,
        [getattr(native.StretchMode, mode), *args],
        [getattr(original.StretchMode, mode), *args],
    )


@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("values", [[95, 685, 0.6], [0, 1023, 1], [120, 670, 0.31]])
def test_log_linear(invert, channels, values):
    compare(
        "log_to_linear",
        source(channels, "strided"),
        [*values, invert],
    )


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("add", [0]),
        ("multiply", [1]),
        ("divide", [1]),
        ("clamp", [0, 1]),
        ("brightness_and_contrast", [0, 0]),
        ("opacity", [100]),
    ],
)
def test_noop_identity(name, args):
    img = source(4, "readonly")
    assert compare(name, img, args) is img


def test_bridge_alignment_and_dtype():
    img = source(3, "unaligned")
    assert not img.flags.aligned
    assert f32(img).flags.aligned
    with pytest.raises(TypeError):
        f32(img.astype(np.float64))
    with pytest.raises(TypeError):
        f32(img.astype(">f4"))


def test_bad_c_arguments_do_not_write():
    dll = lib()
    value = ct.c_float(123)
    pointer = ct.pointer(value)
    assert dll.cn_adjust_f32(pointer, pointer, 1, 0, 1, 1, 0, 0, 0) == 1
    assert dll.cn_adjust_f32(pointer, pointer, 1, 3, 1, 5, 0, 0, 0) == 1
    # A channel stride of 0, or one the pixel count is no multiple of.
    assert dll.cn_adjust_f32(pointer, pointer, 1, 4, 0, 1, 0, 0, 0) == 1
    assert dll.cn_adjust_f32(pointer, pointer, 3, 4, 2, 1, 0, 0, 0) == 1
    assert dll.cn_levels_f32(pointer, pointer, 3, 4, 2, 15, 0, 1, 1, 0, 1) == 1
    assert (
        dll.cn_adjust_f32(pointer, pointer, ct.c_size_t(-1).value, 4, 1, 1, 0, 0, 0)
        == 2
    )
    assert value.value == 123


@pytest.mark.parametrize(
    "name,tail",
    [
        ("cn_adjust_f32", [16, 4, 1, 0, 0.1, 0.0, 0]),
        ("cn_adjust_f32", [16, 4, 4, 6, 0.0, 0.0, 0]),
        ("cn_opacity_f32", [16, 4, 0.7]),
        ("cn_levels_f32", [16, 4, 1, 15, 0.0, 1.0, 2.2, 0.0, 1.0]),
        ("cn_levels_f32", [16, 4, 16, 5, 0.0, 1.0, 2.2, 0.0, 1.0]),
        ("cn_log_linear_f32", [64, 685.0, 0.6, 0.01, 1.01, 0.003, 0]),
    ],
)
def test_adjustment_abi_ranges_and_inplace(name, tail):
    source = np.linspace(0.1, 0.9, 64, dtype=np.float32)
    output = np.full(64, 123, dtype=np.float32)
    pointer = ct.POINTER(ct.c_float)
    function = getattr(lib(), name)
    before = output.tobytes()
    for address, expected in [
        (source.ctypes.data + 1, 1),
        (output.ctypes.data + 4, 1),
        (ct.c_size_t(-4).value, 2),
    ]:
        assert (
            function(ct.cast(address, pointer), output.ctypes.data_as(pointer), *tail)
            == expected
        )
        assert output.tobytes() == before
    assert (
        function(source.ctypes.data_as(pointer), output.ctypes.data_as(pointer), *tail)
        == 0
    )
    inplace = source.copy()
    assert (
        function(
            inplace.ctypes.data_as(pointer), inplace.ctypes.data_as(pointer), *tail
        )
        == 0
    )
    np.testing.assert_array_equal(inplace.view(np.uint32), output.view(np.uint32))


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("clamp", [0.0, 0.8]),
        ("clamp", [-0.0, 0.8]),
        ("clamp", [0.0, -0.0]),
        ("clamp", [-0.0, 0.0]),
        ("brightness_and_contrast", [-50, 50]),
        ("brightness_and_contrast", [50, 50]),
        ("color_levels", [True, True, True, True, 0.0, 1.0, 1.0, -0.0, 1.0]),
        ("color_levels", [False, False, False, False, 0.0, 1.0, 1.0, 0.0, 1.0]),
    ],
)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_numpy_clip_signed_zero_at_every_c_caller(name, args, channels):
    values = np.array([-0.0, 0.0, -np.inf, np.inf, np.nan, 0.5], np.float32)
    img = np.resize(values, (7, 9, channels))
    native, original = MODULES[name]
    with np.errstate(all="ignore"):
        actual = getattr(native, name + "_node")(img, *args)
        expected = getattr(original, name + "_node")(img, *args)
    np.testing.assert_array_equal(actual, expected)
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[valid].view(np.uint32), expected[valid].view(np.uint32)
    )
