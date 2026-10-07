"""Exact installed Rust parity for Gamma and Fill Alpha color extension."""

from __future__ import annotations

import ast
import ctypes as ct
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from chainner_ext import fast_gamma as original_gamma
from chainner_ext import fill_alpha_extend_color as original_extend
from nodes.impl import native_alpha_gamma as native
from nodes.impl.native import lib, ptr

ROOT = Path(__file__).resolve().parents[2]
NODES = ROOT / "backend/src/packages/chaiNNer_standard"
REF = Path(__file__).with_name("reference_alpha_gamma")
PATHS = {
    "gamma": "image_adjustment/gamma/gamma.py",
    "fill_alpha": "image_channel/misc/fill_alpha.py",
}


def load_body(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.ImportFrom)
        or (not node.level and not (node.module or "").startswith("nodes.properties"))
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    module = types.ModuleType(path.stem)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


MODULES = {
    name: (load_body(NODES / path), load_body(REF / Path(path).name))
    for name, path in PATHS.items()
}
LAYOUTS = ("contiguous", "strided", "transpose", "readonly", "unaligned")


def layout(data, kind):
    if kind == "strided":
        backing = np.empty(
            (data.shape[0] * 2, data.shape[1] * 2, *data.shape[2:]), data.dtype
        )
        backing[::2, ::2] = data
        return backing[::2, ::2][::-1]
    if kind == "transpose":
        return data.swapaxes(0, 1)
    if kind == "readonly":
        data.setflags(write=False)
    if kind == "unaligned":
        result = np.ndarray(
            data.shape, data.dtype, buffer=bytearray(data.nbytes + 1), offset=1
        )
        result[:] = data
        return result
    return data


def image(channels=4, shape=(13, 17), seed=411):
    shape = (*shape, channels) if channels else shape
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


def exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[finite].view(np.uint32), expected[finite].view(np.uint32)
    )


@pytest.mark.parametrize("channels", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("kind", LAYOUTS)
@pytest.mark.parametrize("gamma", [0.01, 0.125, 0.5, 1, 2, 2.2, 7.31, 100])
@pytest.mark.parametrize("invert", [False, True])
def test_gamma_node_exact(channels, kind, gamma, invert):
    source = layout(image(channels), kind)
    before = source.copy()
    current, original = MODULES["gamma"]
    actual = current.gamma_node(source, gamma, invert)
    expected = original.gamma_node(source, gamma, invert)
    exact(actual, expected)
    exact(source, before)
    assert (actual is source) == (gamma == 1)
    if gamma != 1:
        assert not np.shares_memory(actual, source)
    if channels == 4:
        np.testing.assert_array_equal(
            actual[..., 3].view(np.uint32), source[..., 3].view(np.uint32)
        )


@pytest.mark.parametrize(
    "count", [0, 1, 7, 8, 9, 15, 16, 17, 8191, 8192, 8193, 16385, 65539]
)
@pytest.mark.parametrize("gamma", [0, -1, 0.3, 2.2, 100, np.nan, np.inf, -np.inf])
def test_gamma_chunk_boundaries_and_float_edges(count, gamma):
    values = np.array(
        [
            -np.inf,
            -2,
            -1,
            -0.0,
            0.0,
            np.nextafter(np.float32(0), np.float32(1)),
            np.finfo(np.float32).tiny,
            0.01,
            0.3,
            0.5,
            0.75,
            np.nextafter(np.float32(1), np.float32(0)),
            1,
            np.nextafter(np.float32(1), np.float32(2)),
            2,
            np.inf,
            np.nan,
        ],
        np.float32,
    )
    source = np.resize(values, count).reshape((1, count))
    exact(native.fast_gamma(source, gamma), original_gamma(source, gamma))


@pytest.mark.parametrize("channels", [1, 3, 4, 7])
@pytest.mark.parametrize("gamma", [0.01, 0.12345, 0.5, 1.71, 2.2, 3.1415, 19.9, 100])
def test_gamma_many_float_bit_patterns(channels, gamma):
    rng = np.random.default_rng(491)
    source = rng.integers(0, 0xFFFFFFFF, (31, 67, channels), dtype=np.uint32).view(
        np.float32
    )
    exact(native.fast_gamma(source, gamma), original_gamma(source, gamma))


@pytest.mark.parametrize("channels", [0, 1, 3, 4, 7])
@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (7, 0)])
def test_gamma_empty(channels, shape):
    source = image(channels, shape)
    exact(native.fast_gamma(source, 2.2), original_gamma(source, 2.2))


def test_gamma_noop_keeps_original_validation_and_zero_invert():
    for module in MODULES["gamma"]:
        source = np.ones((3, 5), np.float64)
        assert module.gamma_node(source, 1, True) is source
        with pytest.raises(ZeroDivisionError):
            module.gamma_node(image(), 0, True)


def fill_image(shape, pattern):
    source = image(4, shape)
    if pattern == "random":
        source[..., 3] = (source[..., 3] > 0.73).astype(np.float32)
    elif pattern == "center":
        source[..., 3] = 0
        source[shape[0] // 2, shape[1] // 2, 3] = 1
    elif pattern == "opaque":
        source[..., 3] = 1
    elif pattern == "transparent":
        source[..., 3] = 0
    elif pattern == "edge":
        source[..., 3] = 0
        source[:, -1, 3] = 1
    elif pattern == "threshold":
        values = np.array(
            [
                0.0,
                np.nextafter(np.float32(0.05), np.float32(0)),
                0.05,
                np.nextafter(np.float32(0.05), np.float32(1)),
                0.7,
                1,
            ],
            np.float32,
        )
        source[..., 3] = np.resize(values, shape)
    elif pattern == "signed_zero":
        source[..., :3] = -0.0
        source[..., 3] = (source[..., 3] > 0.6).astype(np.float32)
    elif pattern == "nonfinite":
        source[..., 3] = (source[..., 3] > 0.6).astype(np.float32)
        source.reshape(-1, 4)[::7, :3] = [np.nan, np.inf, -np.inf]
        source.reshape(-1, 4)[::11, 3] = np.nan
    return source


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 17), (19, 1), (7, 9), (8, 8), (9, 17), (25, 31), (33, 35)]
)
@pytest.mark.parametrize(
    "pattern",
    [
        "random",
        "center",
        "opaque",
        "transparent",
        "edge",
        "threshold",
        "signed_zero",
        "nonfinite",
    ],
)
@pytest.mark.parametrize("iterations", [0, 1, 2, 8, 9, 100000])
def test_extend_color_exact(shape, pattern, iterations):
    source = fill_image(shape, pattern)
    before = source.copy()
    exact(
        native.fill_alpha_extend_color(source, 0.05, iterations),
        original_extend(source, 0.05, iterations),
    )
    exact(source, before)


@pytest.mark.parametrize("kind", LAYOUTS)
@pytest.mark.parametrize("pattern", ["random", "center", "threshold", "signed_zero"])
@pytest.mark.parametrize("method", [1, 2, 3])
def test_fill_node_modes_and_alpha_view(kind, pattern, method):
    source = layout(fill_image((25, 29), pattern), kind)
    before = source.copy()
    current, original = MODULES["fill_alpha"]
    actual_rgb, actual_alpha = current.fill_alpha_node(
        source, current.AlphaFillMethod(method)
    )
    expected_rgb, expected_alpha = original.fill_alpha_node(
        source, original.AlphaFillMethod(method)
    )
    exact(actual_rgb, expected_rgb)
    exact(actual_alpha, expected_alpha)
    assert np.shares_memory(actual_alpha, source)
    assert not np.shares_memory(actual_rgb, source)
    exact(source, before)


@pytest.mark.parametrize("threshold", [-1, 0, 0.05, 0.8, 1, 2, np.nan])
def test_extend_thresholds(threshold):
    source = image(4, (27, 23))
    exact(
        native.fill_alpha_extend_color(source, threshold, 100),
        original_extend(source, threshold, 100),
    )


@pytest.mark.parametrize("shape", [(0, 0, 4), (0, 3, 4), (7, 0, 4)])
def test_extend_empty(shape):
    source = np.empty(shape, np.float32)
    exact(
        native.fill_alpha_extend_color(source, 0.05, 100),
        original_extend(source, 0.05, 100),
    )


def test_parallel_kernels_concurrency():
    source = fill_image((259, 263), "random")
    rgb = source[..., :3].copy()
    expected_gamma = original_gamma(rgb, 2.2)
    expected_fill = original_extend(source, 0.05, 100000)

    def run(index):
        if index % 2:
            exact(native.fast_gamma(rgb, 2.2), expected_gamma)
        else:
            exact(native.fill_alpha_extend_color(source, 0.05, 100000), expected_fill)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(24)))


@pytest.mark.parametrize(
    "source",
    [
        np.ones((3,), np.float32),
        np.ones((2, 3, 4, 5), np.float32),
        np.ones((2, 3, 0), np.float32),
        np.ones((2, 3), np.float64),
    ],
)
@pytest.mark.parametrize("operation", ["fast_gamma", "fill_alpha_extend_color"])
def test_invalid_shape_before_c(source, operation, monkeypatch):
    monkeypatch.setattr(
        native, "_api", lambda: pytest.fail("Invalid input dispatched to C")
    )
    arguments = (2.2,) if operation == "fast_gamma" else (0.05, 3)
    with pytest.raises((TypeError, ValueError)):
        getattr(native, operation)(source, *arguments)


@pytest.mark.parametrize("iterations", [-1, 2**32, 1.5])
def test_invalid_iteration_before_c(iterations, monkeypatch):
    monkeypatch.setattr(
        native, "_api", lambda: pytest.fail("Invalid iteration dispatched to C")
    )
    with pytest.raises((TypeError, ValueError)):
        native.fill_alpha_extend_color(image(), 0.05, iterations)


def test_raw_c_overflow_and_cpu_dispatch():
    data = image(1, (1, 8))
    native.fast_gamma(data, 2.2)
    dll = lib()
    p = ptr(data)
    assert dll.cn_alpha_gamma(p, p, ct.c_size_t(-1).value, 4, 2.2) == 2
    assert dll.cn_alpha_gamma(p, p, 1, 0, 2.2) == 1
    assert dll.cn_alpha_extend(p, p, ct.c_size_t(-1).value, 4, 0.05, 1) == 2
    assert dll.cn_alpha_extend(None, p, 1, 1, 0.05, 1) == 1
    # The original's vector path clamps values above one; scalar powf does not.
    data.fill(2)
    vector_selected = original_gamma(data, 2.2)[0, 0, 0] == 1
    assert bool(dll.cn_gamma_uses_approximation()) == vector_selected


@pytest.fixture
def gamma_api():
    native.fast_gamma(np.empty((0, 0), np.float32), 2.2)
    return lib()


@pytest.mark.parametrize("shift", [-4, -1, 1, 4])
def test_gamma_raw_shifted_overlap_and_alignment(shift, gamma_api):
    source = np.arange(34, dtype=np.float32)[1:33]
    before = source.copy()
    output = ct.cast(source.ctypes.data + shift, ct.POINTER(ct.c_float))
    assert gamma_api.cn_alpha_gamma(ptr(source), output, 32, 1, 2.2) == 1
    exact(source, before)


def test_gamma_raw_pointer_wrap_and_unaligned_input(gamma_api):
    source = np.zeros((2, 3), np.float32)
    output = np.empty_like(source)
    pointer = ct.POINTER(ct.c_float)
    unaligned = ct.cast(source.ctypes.data + 1, pointer)
    near_end = ct.cast(ct.c_size_t(-4).value, pointer)
    assert gamma_api.cn_alpha_gamma(unaligned, ptr(output), 6, 1, 2.2) == 1
    assert gamma_api.cn_alpha_gamma(near_end, ptr(output), 6, 1, 2.2) == 2
    assert gamma_api.cn_alpha_gamma(ptr(source), near_end, 6, 1, 2.2) == 2


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_gamma_raw_exact_alias_for_resize(channels, gamma_api):
    source = image(channels, (17, 19))
    expected = original_gamma(source, 1 / 2.2)
    assert (
        gamma_api.cn_alpha_gamma(ptr(source), ptr(source), 17 * 19, channels, 1 / 2.2)
        == 0
    )
    exact(source, expected)


@pytest.mark.parametrize("name", PATHS)
def test_node_schema_unchanged(name):
    current = ast.parse((NODES / PATHS[name]).read_text(encoding="utf-8"))
    original = ast.parse((REF / Path(PATHS[name]).name).read_text(encoding="utf-8"))
    for before in original.body:
        if isinstance(before, ast.FunctionDef) and before.decorator_list:
            after = next(
                node
                for node in current.body
                if isinstance(node, ast.FunctionDef) and node.name == before.name
            )
            assert ast.dump(before.args) == ast.dump(after.args)
            assert [ast.dump(d) for d in before.decorator_list] == [
                ast.dump(d) for d in after.decorator_list
            ]
