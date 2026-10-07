"""Exact frozen-source Normal Map kernels and whole-node CPU comparisons."""

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

from nodes.impl import native_normal_kernel as native
from nodes.impl.normals import edge_filter
from nodes.utils.utils import get_h_w_c

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).with_name("reference_normal_complete")


def load_body(path: Path) -> types.ModuleType:
    tree = ast.parse(path.read_text("utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.ImportFrom)
        or (
            node.level == 0
            and node.module not in {"nodes.groups"}
            and not (node.module or "").startswith("nodes.properties")
        )
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    module = types.ModuleType(path.stem)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


ORIGINAL_KERNEL = load_body(REFERENCE / "edge_filter.py")
CURRENT_NODE = load_body(
    ROOT
    / "backend/src/packages/chaiNNer_standard/material_textures/normal_map/normal_map_generator.py"
)
ORIGINAL_NODE = load_body(REFERENCE / "normal_map_generator.py")
ORIGINAL_NODE.__dict__.update(EdgeFilter=ORIGINAL_KERNEL.EdgeFilter)
ORIGINAL_NODE.__dict__.update(get_filter_kernels=ORIGINAL_KERNEL.get_filter_kernels)
_tree = ast.parse((REFERENCE / "image_utils.py").read_text("utf-8"))
_tree.body = [
    node
    for node in _tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "fast_gaussian_blur"
]
_globals = {"np": np, "cv2": cv2, "math": math, "get_h_w_c": get_h_w_c}
exec(compile(_tree, str(REFERENCE / "image_utils.py"), "exec"), _globals)
ORIGINAL_NODE.__dict__.update(fast_gaussian_blur=_globals["fast_gaussian_blur"])


def exact(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    mask = ~np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    integer = np.uint32 if actual.dtype == np.float32 else np.uint64
    np.testing.assert_array_equal(
        actual.view(integer)[mask], expected.view(integer)[mask]
    )


PARAMETERS = [
    [],
    [(0.25, 0)],
    [(0.25, 1)],
    [(0.5, 0.333)],
    [(1, 1.25)],
    [(2, 10)],
    [(4, 0.003)],
    [(8, 1)],
    [(16, 1)],
    [(32, 0.1)],
    [
        (0.25, 0.25),
        (0.5, 0.5),
        (1, 0.3),
        (2, 0.25),
        (4, 0.2),
        (8, 0.15),
        (16, 0.1),
        (32, 0.1),
    ],
    [(0.25, 1), (0.5, -1)],
    [(0.25, -1)],
    [(-0.5, 1)],
    [(0.25, np.nan)],
    [(np.inf, 0), (0.25, 1)],
    [(np.nan, 0), (0.25, 1)],
]


@pytest.mark.parametrize("parameters", PARAMETERS)
def test_gaussian_coefficients(parameters: list[tuple[float, float]]) -> None:
    with np.errstate(all="ignore"):
        expected = ORIGINAL_KERNEL.create_gauss_kernel(parameters)
        actual = edge_filter.create_gauss_kernel(parameters)
    exact(actual, expected)


@pytest.mark.parametrize("name", [item.value for item in edge_filter.EdgeFilter])
def test_fixed_filter_pair(name: str) -> None:
    parameters = [(0.25, 0.5), (1, 1)]
    actual = edge_filter.get_filter_kernels(edge_filter.EdgeFilter(name), parameters)
    expected = ORIGINAL_KERNEL.get_filter_kernels(
        ORIGINAL_KERNEL.EdgeFilter(name), parameters
    )
    for a, b in zip(actual, expected, strict=True):
        exact(a, b)


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 3), (5, 5), (7, 7), (9, 9), (33, 257), (65, 257)]
)
@pytest.mark.parametrize("layout", ["plain", "reverse", "unaligned", "readonly"])
@pytest.mark.parametrize("normalize", [False, True])
def test_pair_generic(shape: tuple[int, int], layout: str, normalize: bool) -> None:
    source = np.random.default_rng(73).random(shape)
    if layout == "reverse":
        source = source[::-1, ::-1]
    elif layout == "unaligned":
        values = source
        source = np.ndarray(
            shape, dtype=np.float64, buffer=bytearray(values.nbytes + 1), offset=1
        )
        source[:] = values
    elif layout == "readonly":
        source.flags.writeable = False
    before = source.copy()
    with np.errstate(all="ignore"):
        x = source / np.sum(source[:, : shape[1] // 2]) if normalize else source
    y = np.rot90(x, -1)
    actual_x, actual_y = native.kernel_pair(source, normalize=normalize)
    exact(actual_x, x)
    exact(actual_y, y)
    exact(source, before)


@pytest.mark.parametrize("kind", ["random", "nonfinite", "zeros", "extremes"])
@pytest.mark.parametrize("optimized", [False, True])
def test_sharpen(kind: str, optimized: bool) -> None:
    rng = np.random.default_rng(75)
    a, b = (rng.random((29, 41), dtype=np.float32) for _ in range(2))
    if kind != "random":
        options = {
            "nonfinite": [-np.inf, np.inf, np.nan, 0.5, 1],
            "zeros": [-0.0, 0.0],
            "extremes": [
                -np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).tiny,
                -np.finfo(np.float32).tiny,
            ],
        }
        values = np.asarray(options[kind], np.float32)
        a, b = (values[rng.integers(0, len(values), a.shape)] for _ in range(2))
    a, b = a[::-1, ::-1], b[::-1, ::-1]
    a.flags.writeable = b.flags.writeable = False
    original = cv2.useOptimized()
    try:
        cv2.setUseOptimized(optimized)
        exact(native.sharpen(a, b), cv2.addWeighted(a, 2, b, -1, 0))
    finally:
        cv2.setUseOptimized(original)


@pytest.mark.parametrize("name", [item.value for item in edge_filter.EdgeFilter])
@pytest.mark.parametrize("tileable", [False, True])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("blur", [-1.3, 0, 1.7])
def test_normal_node(name: str, tileable: bool, channels: int, blur: float) -> None:
    source = np.random.default_rng(233).random((9, 13, channels), dtype=np.float32)
    if channels == 1:
        source = source[:, :, 0]
    source = source[::-1, ::-1]
    before = source.copy()
    source.flags.writeable = False
    scales = [0.25, 0.5, 0.3, 0.25, 0, 0, 0, 0]
    args = [
        tileable,
        CURRENT_NODE.HeightSource.SCREEN_RGB,
        blur,
        0.2,
        1.7,
        CURRENT_NODE.EdgeFilter(name),
        *scales,
        True,
        False,
        CURRENT_NODE.AlphaOutput.HEIGHT,
    ]
    expected_args = args.copy()
    expected_args[5] = ORIGINAL_NODE.EdgeFilter(name)
    expected_args[-1] = ORIGINAL_NODE.AlphaOutput.HEIGHT
    expected = ORIGINAL_NODE.normal_map_generator_node(source, *expected_args)
    actual = CURRENT_NODE.normal_map_generator_node(source, *args)
    exact(actual, expected)
    exact(source, before)


def test_parallel_concurrency() -> None:
    parameters = [[(0.25, 0.1 * i), (2, 0.5)] for i in range(1, 9)]
    expected = [ORIGINAL_KERNEL.create_gauss_kernel(p) for p in parameters]
    with ThreadPoolExecutor(max_workers=8) as executor:
        actual = list(executor.map(native.gaussian_kernel, parameters))
    for a, b in zip(actual, expected, strict=True):
        exact(a, b)


def test_raw_boundaries() -> None:
    source = np.ones(32, np.float64)
    x, y = np.empty_like(source), np.empty_like(source)
    p = ct.POINTER(ct.c_double)
    function = native._api().cn_normal_kernel_pair
    args = [native._double(source), native._double(x), native._double(y), 2, 3, 1]
    assert function(*args) == 0
    for index, value in [(0, p()), (1, p()), (2, p()), (3, 0), (4, 0), (5, 2)]:
        bad = args.copy()
        bad[index] = value
        assert function(*bad) == 1
    for index in (1, 2):
        for offset in (1, 8):
            bad = args.copy()
            bad[index] = ct.cast(source.ctypes.data + offset, p)
            assert function(*bad) == 1
    bad = args.copy()
    bad[3] = ct.c_size_t(-1).value
    assert function(*bad) == 2
    args[1] = native._double(source)
    assert function(*args) == 0


@pytest.mark.parametrize(
    "parameters", [[(0.0, 1)], [(1e-300, 1)], [(np.inf, 1)], [(np.nan, 1)]]
)
def test_public_parameter_errors(parameters: list[tuple[float, float]]) -> None:
    with pytest.raises((ZeroDivisionError, OverflowError, ValueError)) as expected:
        ORIGINAL_KERNEL.create_gauss_kernel(parameters)
    with pytest.raises(type(expected.value)):
        native.gaussian_kernel(parameters)


@pytest.mark.parametrize("active", [0, 4, 7, None])
@pytest.mark.parametrize("tileable", [False, True])
def test_large_and_zero_gaussian_normal_node(
    active: int | None, tileable: bool
) -> None:
    source = np.random.default_rng(257).random((5, 7, 4), dtype=np.float32)
    scales = [0.0] * 8
    if active is not None:
        scales[active] = 0.731
    args = [
        tileable,
        CURRENT_NODE.HeightSource.AVERAGE_RGB,
        0,
        0,
        1.9,
        CURRENT_NODE.EdgeFilter.MULTI_GAUSS,
        *scales,
        False,
        True,
        CURRENT_NODE.AlphaOutput.UNCHANGED,
    ]
    expected_args = args.copy()
    expected_args[5] = ORIGINAL_NODE.EdgeFilter.MULTI_GAUSS
    expected_args[-1] = ORIGINAL_NODE.AlphaOutput.UNCHANGED
    exact(
        CURRENT_NODE.normal_map_generator_node(source, *args),
        ORIGINAL_NODE.normal_map_generator_node(source, *expected_args),
    )


def test_gaussian_raw_boundaries() -> None:
    parameters = np.array([[0.25, 1]], np.float64)
    result = np.empty((5, 5), np.float64)
    function = native._api().cn_normal_gaussian_kernel
    dp = ct.POINTER(ct.c_double)
    assert function(native._double(parameters), 1, native._double(result), 2) == 0
    assert function(dp(), 1, native._double(result), 2) == 1
    assert function(native._double(parameters), 1, dp(), 2) == 1
    assert function(native._double(parameters), 1, native._double(result), 1) == 1
    assert function(native._double(result), 1, native._double(result), 2) == 1
    assert (
        function(
            native._double(parameters), ct.c_size_t(-1).value, native._double(result), 2
        )
        == 2
    )
    assert (
        function(
            native._double(parameters), 1, native._double(result), ct.c_size_t(-1).value
        )
        == 2
    )
    assert (
        function(ct.cast(parameters.ctypes.data + 1, dp), 1, native._double(result), 2)
        == 1
    )
    assert function(dp(), 0, native._double(result), 0) == 0


def test_normal_spatial_path_does_not_call_old_kernels(monkeypatch) -> None:
    image = np.random.default_rng(81).random((17, 19, 4), dtype=np.float32)
    scales = [0.25, 0.5, 0.3, 0.25, 0, 0, 0, 0]
    args = [
        True,
        CURRENT_NODE.HeightSource.RED,
        0,
        0,
        1.3,
        CURRENT_NODE.EdgeFilter.MULTI_GAUSS,
        *scales,
        False,
        False,
        CURRENT_NODE.AlphaOutput.ONE,
    ]
    expected_args = args.copy()
    expected_args[5] = ORIGINAL_NODE.EdgeFilter.MULTI_GAUSS
    expected_args[-1] = ORIGINAL_NODE.AlphaOutput.ONE
    expected = ORIGINAL_NODE.normal_map_generator_node(image, *expected_args)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Normal C path called an original kernel")

    for name in ("filter2D", "copyMakeBorder", "addWeighted"):
        monkeypatch.setattr(cv2, name, forbidden)
    for name in ("exp", "sum", "rot90"):
        monkeypatch.setattr(np, name, forbidden)
    exact(CURRENT_NODE.normal_map_generator_node(image, *args), expected)
