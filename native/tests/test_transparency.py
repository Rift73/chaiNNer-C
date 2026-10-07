"""Compare transparency/color node behavior with frozen pre-port sources."""

from __future__ import annotations

import ast
import ctypes as ct
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import DTypeLike

from nodes.impl import native_transparency as native
from nodes.impl.color import convert as conversion
from nodes.impl.color import convert_data
from nodes.impl.color.color import Color

ROOT = Path(__file__).resolve().parents[2]
REF = Path(__file__).with_name("reference_transparency")
NODES = ROOT / "backend/src/packages/chaiNNer_standard/image_channel/misc"


def reference_helper(name):
    full_name = "nodes.impl.color._reference_transparency_" + name
    tree = ast.parse((REF / (name + ".py")).read_text("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "convert_data":
            node.module = "_reference_transparency_convert_data"
        elif isinstance(node, ast.ImportFrom) and node.module == "logger":
            # The frozen source logged through a module the shipped tree lacks.
            node.module = "sanic.log"
    module = types.ModuleType(full_name)
    module.__package__ = "nodes.impl.color"
    sys.modules[full_name] = module
    exec(compile(tree, str(REF / (name + ".py")), "exec"), module.__dict__)
    return module


def node_body(path):
    tree = ast.parse(path.read_text("utf-8"))
    tree.body = [
        n
        for n in tree.body
        if not isinstance(n, ast.ImportFrom)
        or (
            n.level == 0
            and n.module != "nodes.groups"
            and not (n.module or "").startswith("nodes.properties")
        )
    ]
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            n.decorator_list = []
    module = types.ModuleType(path.stem)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


REFERENCE_DATA = reference_helper("convert_data")
REFERENCE_CONVERT = reference_helper("convert")
CHROMA = node_body(NODES / "chroma_key.py")
ORIGINAL_CHROMA = node_body(REF / "chroma_key.py")
MATTING = node_body(NODES / "alpha_matting.py")
ORIGINAL_MATTING = node_body(REF / "alpha_matting.py")


def image(channels=3, layout="plain", shape=(9, 13), dtype: DTypeLike = np.float32):
    out = np.random.default_rng(792).random((*shape, channels)).astype(dtype)
    return arrange(out, layout)


def arrange(out, layout):
    if layout == "strided":
        out = out[::-2, ::2]
    elif layout == "readonly":
        out.flags.writeable = False
    elif layout == "unaligned":
        other = np.ndarray(
            out.shape, dtype=out.dtype, buffer=bytearray(out.nbytes + 1), offset=1
        )
        other[...] = out
        out = other
    return out


def same(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("source", range(14))
@pytest.mark.parametrize("target", range(14))
@pytest.mark.parametrize("layout", ["plain", "strided", "readonly"])
def test_color_conversion_graph(source, target, layout):
    src = conversion.color_space_from_id(source)
    dst = conversion.color_space_from_id(target)
    data = image(src.channels, layout)
    if src.channels == 1:
        data = data[:, :, 0]
    before = data.copy()
    actual = conversion.convert(data, src, dst)
    expected = REFERENCE_CONVERT.convert(
        data,
        REFERENCE_CONVERT.color_space_from_id(source),
        REFERENCE_CONVERT.color_space_from_id(target),
    )
    same(actual, expected)
    same(data, before)
    if source == target:
        assert actual is data


@pytest.mark.parametrize(
    "name,channels",
    [
        ("__rgb_to_cmyk", 3),
        ("__cmyk_to_rgb", 4),
        ("__lab_to_lch", 3),
        ("__lch_to_lab", 3),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["plain", "strided", "readonly", "unaligned"])
def test_color_arithmetic_edges(name, channels, dtype, layout):
    data = image(channels, dtype=dtype)
    edge = np.array(
        [np.nan, np.inf, -np.inf, -0.0, 0.0, 0.001, -0.5, 1.5, 0.5, 1, 1e20, -1e20],
        dtype=dtype,
    )
    data.flat[: len(edge)] = edge
    data = arrange(data, layout)
    with np.errstate(all="ignore"):
        same(getattr(convert_data, name)(data), getattr(REFERENCE_DATA, name)(data))


@pytest.mark.parametrize("layout", ["plain", "strided", "readonly", "unaligned"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 13), (17, 1), (9, 13)])
@pytest.mark.parametrize("threshold", [0, 0.001, 0.1, 1, np.nan])
def test_binary_chroma(layout, shape, threshold):
    data = image(layout=layout, shape=shape)
    key = Color.bgr((0.12, 0.43, 0.88))
    same(
        CHROMA.binary_keying(data, key, threshold),
        ORIGINAL_CHROMA.binary_keying(data, key, threshold),
    )


@pytest.mark.parametrize("bg,fg", [(0, 0.2), (0.2, 0.4), (0.8, 0.1)])
@pytest.mark.parametrize("confusion", [(0, 0), (1, 3), (3, 1)])
@pytest.mark.parametrize("layout", ["plain", "strided", "readonly"])
def test_trimap_chroma(bg, fg, confusion, layout):
    data = image(layout=layout)
    key = Color.bgr((0.12, 0.43, 0.88))
    args = (data, key, bg, fg, *confusion, True)
    same(
        CHROMA.trimap_matting_keying(*args),
        ORIGINAL_CHROMA.trimap_matting_keying(*args),
    )


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["plain", "strided", "readonly", "unaligned"])
@pytest.mark.parametrize("singleton", [False, True])
def test_matting_solver_boundary(monkeypatch, channels, dtype, layout, singleton):
    data = image(channels, layout)
    trimap = np.linspace(0, 1, data.shape[0] * data.shape[1], dtype=dtype).reshape(
        data.shape[:2]
    )
    trimap.flat[:5] = [15 / 255, 240 / 255, np.nan, np.inf, -np.inf]
    if singleton:
        trimap = trimap[:, :, None]
    captures = []

    def estimate_alpha(img, mask):
        captures.append((img.copy(), mask.copy()))
        return mask.reshape(mask.shape[:2]).copy()

    def estimate_foreground(img, alpha):
        return img.astype(np.float32)

    monkeypatch.setattr(MATTING.pymatting, "estimate_alpha_cf", estimate_alpha)
    monkeypatch.setattr(
        MATTING.pymatting, "estimate_foreground_ml", estimate_foreground
    )
    monkeypatch.setattr(MATTING, "estimate_alpha", estimate_alpha)
    monkeypatch.setattr(MATTING, "estimate_foreground", estimate_foreground)
    actual = MATTING.alpha_matting_node(data, trimap, 240, 15)
    expected = ORIGINAL_MATTING.alpha_matting_node(data, trimap, 240, 15)
    same(actual, expected)
    for a, b in zip(captures[0], captures[1], strict=True):
        same(a, b)


@pytest.fixture
def real_solver_calls(monkeypatch):
    """Compare the C solvers' inputs with the live PyMatting 1.1.16's."""
    calls = {"alpha": [], "foreground": []}
    estimate_alpha = MATTING.pymatting.estimate_alpha_cf
    estimate_foreground = MATTING.pymatting.estimate_foreground_ml
    c_alpha, c_foreground = MATTING.estimate_alpha, MATTING.estimate_foreground

    def alpha(img, trimap):
        calls["alpha"].append((img.copy(), trimap.copy()))
        return estimate_alpha(img, trimap)

    def foreground(img, matte):
        calls["foreground"].append((img.copy(), matte.copy()))
        return estimate_foreground(img, matte)

    def native_alpha(img, trimap):
        calls["alpha"].append((img.copy(), trimap.copy()))
        return c_alpha(img, trimap)

    def native_foreground(img, matte):
        calls["foreground"].append((img.copy(), matte.copy()))
        return c_foreground(img, matte)

    monkeypatch.setattr(MATTING.pymatting, "estimate_alpha_cf", alpha)
    monkeypatch.setattr(MATTING.pymatting, "estimate_foreground_ml", foreground)
    for module in (MATTING, CHROMA):
        monkeypatch.setattr(module, "estimate_alpha", native_alpha)
        monkeypatch.setattr(module, "estimate_foreground", native_foreground)
    return calls


def same_real_matting(actual, expected, calls):
    """Assert exact preparation, alpha and foreground outputs."""
    same(actual, expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.isfinite(actual[:, :, :3]).all()
    assert np.isfinite(expected[:, :, :3]).all()
    for stages in calls.values():
        assert len(stages) == 2
        for a, b in zip(stages[0], stages[1], strict=True):
            same(a, b)


def test_real_small_matting(real_solver_calls):
    data = image(shape=(12, 13))
    trimap = np.full(data.shape[:2], 0.5, np.float32)
    trimap[:, :3] = 0
    trimap[:, -3:] = 1
    same_real_matting(
        MATTING.alpha_matting_node(data, trimap, 240, 15),
        ORIGINAL_MATTING.alpha_matting_node(data, trimap, 240, 15),
        real_solver_calls,
    )


def test_real_small_chroma_matting(real_solver_calls):
    data = image(shape=(12, 13))
    key = Color.bgr((0.12, 0.43, 0.88))
    data[:, :3] = key.value
    data[:, -3:] = [1, 1, 0]
    args = (data, key, 0.01, 0.4, 0, 0, False)
    same_real_matting(
        CHROMA.trimap_matting_keying(*args),
        ORIGINAL_CHROMA.trimap_matting_keying(*args),
        real_solver_calls,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_chroma_nonfinite_and_threshold_boundary(dtype):
    data = image(dtype=dtype)
    data.flat[:6] = [np.nan, np.inf, -np.inf, 0, 0.5, 1]
    key = Color.bgr((0.2, 0.3, 0.4))
    for threshold in (0, np.nextafter(np.float32(0.1), np.float32(1)), 0.5):
        same(
            CHROMA.binary_keying(data, key, threshold),
            ORIGINAL_CHROMA.binary_keying(data, key, threshold),
        )
    args = (data, key, 0.1, 0.3, 1, 1, True)
    same(
        CHROMA.trimap_matting_keying(*args),
        ORIGINAL_CHROMA.trimap_matting_keying(*args),
    )


def test_native_load_failure_never_falls_back(monkeypatch):
    def broken():
        raise RuntimeError("test native load failure")

    monkeypatch.setattr(native, "_api", broken)
    with pytest.raises(RuntimeError, match="test native load failure"):
        CHROMA.binary_keying(image(), Color.bgr((0, 0, 0)), 0.1)
    with pytest.raises(RuntimeError, match="test native load failure"):
        MATTING.alpha_matting_node(image(), np.zeros((9, 13), np.float32), 240, 15)
    with pytest.raises(RuntimeError, match="test native load failure"):
        conversion.convert(image(), convert_data.RGB, convert_data.CMYK)


def test_concurrent_large_transparency_and_color():
    data = image(shape=(257, 513))
    key = Color.bgr((0.2, 0.3, 0.4))
    functions = [
        lambda: CHROMA.binary_keying(data, key, 0.17),
        lambda: CHROMA.trimap_matting_keying(data, key, 0.1, 0.2, 1, 2, True),
        lambda: conversion.convert(data, convert_data.RGB, convert_data.CMYK),
    ]
    expected = [
        ORIGINAL_CHROMA.binary_keying(data, key, 0.17),
        ORIGINAL_CHROMA.trimap_matting_keying(data, key, 0.1, 0.2, 1, 2, True),
        REFERENCE_CONVERT.convert(data, REFERENCE_DATA.RGB, REFERENCE_DATA.CMYK),
    ]
    with ThreadPoolExecutor(max_workers=6) as pool:
        outputs = list(pool.map(lambda n: functions[n % 3](), range(12)))
    for n, output in enumerate(outputs):
        same(output, expected[n % 3])


def test_checked_boundaries():
    with pytest.raises(ValueError):
        native.chroma_key(image(4), (0, 0, 0), 0.1)
    with pytest.raises(TypeError):
        native.chroma_key(image(dtype=np.float64), (0, 0, 0), 0.1)
    with pytest.raises(ValueError):
        native.chroma_key(image(), (0, 0), 0.1)
    with pytest.raises(ValueError):
        native.chroma_trimap(np.zeros((2, 2), np.uint8), np.zeros((3, 2), np.uint8))
    with pytest.raises(ValueError):
        native.matting_input(image(), np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError):
        native.representation(image(), 10)
    with pytest.raises(ValueError):
        native.representation(image(), 8, np.zeros((2, 2), np.float32))
    dll = native._api()  # validate the raw native boundary
    assert dll.cn_chroma_key(None, None, 1, 3, 1, 0, 0, 0, None, 4, 1, None, None) == 1
    data = np.zeros(4, np.float32)
    p = native.ptr(data)
    # Zero strides, of the image or of out.
    assert dll.cn_chroma_key(p, p, 1, 0, 1, 0, 0, 0, p, 4, 1, None, None) == 1
    assert dll.cn_chroma_key(p, p, 1, 3, 1, 0, 0, 0, p, 4, 0, None, None) == 1
    too_large = (1 << (ct.sizeof(ct.c_size_t) * 8)) - 1
    assert dll.cn_chroma_key(p, p, too_large, 3, 1, 0, 0, 0, p, 4, 1, None, None) == 2
    assert dll.cn_color_representation(p, p, 1, 8, None, None, None, None) == 1
    assert dll.cn_color_representation(p, p, too_large, 0, None, None, None, None) == 2
    np.testing.assert_array_equal(data, 0)


@pytest.mark.parametrize("name", ["chroma_key", "alpha_matting"])
def test_node_metadata_and_signature_unchanged(name):
    def node(path):
        tree = ast.parse(path.read_text("utf-8"))
        return next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == name + "_node"
        )

    actual, original = node(NODES / (name + ".py")), node(REF / (name + ".py"))
    assert ast.dump(actual.args) == ast.dump(original.args)
    assert [ast.dump(x) for x in actual.decorator_list] == [
        ast.dump(x) for x in original.decorator_list
    ]


@pytest.mark.parametrize("layout", ["reversed", "unaligned"])
@pytest.mark.parametrize("foreground", [None, 0.3])
def test_chroma_foreign_key_buffers(layout, foreground):
    key = np.array([0.2, 0.6, 0.8], dtype=np.float32)
    if layout == "reversed":
        key = key[::-1]
        assert key.strides[0] < 0
    else:
        buffer = np.ndarray(
            key.shape, np.float32, buffer=bytearray(key.nbytes + 1), offset=1
        )
        buffer[:] = key
        key = buffer
        assert not key.flags.aligned
    image = np.random.default_rng(31).random((11, 13, 3), dtype=np.float32)
    saved_key, saved_image = key.copy(), image.copy()
    actual = native.chroma_key(image, key, 0.1, foreground)
    expected = native.chroma_key(image, tuple(key), 0.1, foreground)
    if foreground is None:
        np.testing.assert_array_equal(actual, expected)
    else:
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(key, saved_key)
    np.testing.assert_array_equal(image, saved_image)
