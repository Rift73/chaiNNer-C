"""Independent installed Adaptive Threshold node and enforced-input contracts."""

from __future__ import annotations

import ast
import ctypes as ct
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from test_adjustment_complete import layout, load_body

from nodes.impl import native_color_ops as native
from nodes.impl.color.color import Color
from nodes.utils.utils import get_h_w_c

ROOT = Path(__file__).resolve().parents[2]
NODE = (
    ROOT
    / "backend/src/packages/chaiNNer_standard/image_adjustment/threshold/threshold_adaptive.py"
)
REFERENCE = Path(__file__).with_name("reference_color_ops") / "threshold_adaptive.py"
CURRENT, ORIGINAL = load_body(NODE), load_body(REFERENCE)

# Freeze the old float32 byte-conversion body too: the production image_utils
# now delegates this operation to C and must not be used by the reference.
_path = Path(__file__).with_name("reference_normal_complete") / "image_utils.py"
_tree = ast.parse(_path.read_text("utf-8"))
_tree.body = [
    node
    for node in _tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "to_uint8"
]
_globals = {"np": np}
exec(compile(_tree, str(_path), "exec"), _globals)
ORIGINAL.__dict__.update(to_uint8=_globals["to_uint8"])

# Exercise the actual registered ImageInput enforcement body without importing
# unrelated optional model providers from the input-registry package.
_input_path = ROOT / "backend/src/nodes/properties/inputs/numpy_inputs.py"
_input_tree = ast.parse(_input_path.read_text("utf-8"))
_input_class = next(
    node
    for node in _input_tree.body
    if isinstance(node, ast.ClassDef) and node.name == "ImageInput"
)
_input_tree.body = [
    node
    for node in _input_class.body
    if isinstance(node, ast.FunctionDef) and node.name == "enforce"
]
_input_globals: dict[str, Any] = {"np": np, "Color": Color, "get_h_w_c": get_h_w_c}
exec(compile(_input_tree, str(_input_path), "exec"), _input_globals)


def enforce(image: np.ndarray) -> np.ndarray:
    return _input_globals["enforce"](
        types.SimpleNamespace(channels=[1], allow_colors=False, label="Image"), image
    )


def call(module, image, method, kind, radius, maximum=100, delta: float = 0):
    return module.threshold_adaptive_node(
        enforce(image),
        module.AdaptiveThresholdType(kind),
        maximum,
        module.AdaptiveMethod(method),
        radius,
        delta,
    )


def source(shape, pattern):
    rng = np.random.default_rng(887)
    if pattern == "random":
        return rng.random(shape, dtype=np.float32)
    if pattern == "ties":
        return (rng.integers(0, 255, shape).astype(np.float32) + 0.5) / 255
    if pattern == "constant":
        return np.full(shape, 0.5, np.float32)
    values = np.array(
        [-0.0, 0.0, np.nan, np.inf, -np.inf, -2, 3, np.finfo(np.float32).max],
        np.float32,
    )
    return values[rng.integers(0, len(values), shape)]


@pytest.mark.parametrize("method", [0, 1])
@pytest.mark.parametrize("kind", [0, 1])
@pytest.mark.parametrize("radius", [1, 2, 7, 8, 15, 100])
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1), (11, 35)])
@pytest.mark.parametrize("pattern", ["random", "ties", "constant", "exceptional"])
def test_registered_node_exact(method, kind, radius, shape, pattern):
    image = source(shape, pattern)
    before = image.copy()
    with np.errstate(all="ignore"):
        expected = call(ORIGINAL, image, method, kind, radius, 70, -3.3)
        actual = call(CURRENT, image, method, kind, radius, 70, -3.3)
    assert actual.dtype == expected.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("method", [0, 1])
@pytest.mark.parametrize("kind", [0, 1])
@pytest.mark.parametrize(
    "maximum,delta", [(0, -100), (10, -0.2), (30, 0), (50, 0.2), (90, 3.3), (100, 100)]
)
def test_scalar_rounding(method, kind, maximum, delta):
    image = source((13, 29), "ties")
    np.testing.assert_array_equal(
        call(CURRENT, image, method, kind, 2, maximum, delta),
        call(ORIGINAL, image, method, kind, 2, maximum, delta),
    )


@pytest.mark.parametrize("method", [0, 1])
@pytest.mark.parametrize(
    "kind", ["plain", "rows", "columns", "reverse", "fortran", "unaligned"]
)
@pytest.mark.parametrize("optimized", [True, False])
def test_foreign_singleton_channel_and_dispatch(method, kind, optimized):
    image = layout(source((13, 29, 1), "random"), kind)
    saved = cv2.useOptimized()
    cv2.setUseOptimized(optimized)
    try:
        np.testing.assert_array_equal(
            call(CURRENT, image, method, 0, 7), call(ORIGINAL, image, method, 0, 7)
        )
    finally:
        cv2.setUseOptimized(saved)


@pytest.mark.parametrize("method", [0, 1])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, np.finfo(np.float32).max])
def test_enforced_exceptional_inputs_preserve_errors_and_warnings(method, value):
    image = np.full((3, 5, 1), value, np.float32)
    assert enforce(image).shape == (3, 5)
    results = []
    for module in (ORIGINAL, CURRENT):
        with np.errstate(all="raise"), pytest.raises(FloatingPointError) as error:
            call(module, image, method, 0, 1)
        with warnings.catch_warnings(record=True) as records, np.errstate(all="warn"):
            warnings.simplefilter("always")
            result = call(module, image, method, 0, 1)
        results.append(
            (str(error.value), [(r.category, str(r.message)) for r in records], result)
        )
    assert results[0][:2] == results[1][:2]
    np.testing.assert_array_equal(results[0][2], results[1][2])


def test_all_bulk_computation_native_and_concurrent(monkeypatch):
    image = source((257, 269), "random")
    expected = [call(ORIGINAL, image, method, 1, 3, 90, 0.2) for method in (0, 1)]

    def forbidden(*_args, **_kwargs):
        pytest.fail("Adaptive Threshold may not delegate its image algorithm")

    for name in (
        "adaptiveThreshold",
        "GaussianBlur",
        "blur",
        "boxFilter",
        "getGaussianKernel",
        "convertScaleAbs",
        "sepFilter2D",
    ):
        monkeypatch.setattr(cv2, name, forbidden)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda i: call(CURRENT, image, i % 2, 1, 3, 90, 0.2), range(8))
        )
    for i, result in enumerate(results):
        np.testing.assert_array_equal(result, expected[i % 2])


def test_registered_schema_unchanged():
    def registered(path):
        return next(
            node
            for node in ast.parse(path.read_text("utf-8")).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "threshold_adaptive_node"
        )

    a, b = registered(NODE), registered(REFERENCE)
    assert ast.dump(a.args) == ast.dump(b.args)
    assert [ast.dump(x) for x in a.decorator_list] == [
        ast.dump(x) for x in b.decorator_list
    ]


def test_adaptive_foreign_boundaries():
    p = ct.POINTER(ct.c_uint8)
    a = np.ones(9, np.uint8)
    b = np.zeros(9, np.uint8)
    out = np.empty(9, np.uint8)
    ap, bp, op = (x.ctypes.data_as(p) for x in (a, b, out))
    fn = native._api().cn_adaptive_apply
    assert fn(None, bp, op, 9, 0, 0, 255) == 1
    assert fn(ap, bp, ap, 9, 0, 0, 255) == 1
    assert fn(ap, bp, ct.cast(a.ctypes.data + 1, p), 9, 0, 0, 255) == 1
    assert fn(ap, bp, op, ct.c_size_t(-1).value, 0, 0, 255) == 2
    assert fn(ap, bp, op, 9, 0, 2, 255) == 1
