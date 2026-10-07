"""Exact luminance, f64 morphology, and frozen original CAS/High Boost checks."""

from __future__ import annotations

import ast
import ctypes as ct
import importlib.util
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from nodes.impl import cas, native_cas
from nodes.impl.native import lib
from nodes.impl.native_analysis import cas_mask

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).with_name("reference_analysis")


def load_original_cas():
    spec = importlib.util.spec_from_file_location(
        "nodes.impl.reference_cas_complete", REFERENCE / "nodes/impl/cas.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_node(path):
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


ORIGINAL = load_original_cas()
RELATIVE = "packages/chaiNNer_standard/image_filter/sharpen/high_boost_filter.py"
BOOST = load_node(ROOT / "backend/src" / RELATIVE)
BOOST_REF = load_node(REFERENCE / RELATIVE)
BOOST_REF.__dict__.update(cas_mix=ORIGINAL.cas_mix)


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    finite = ~np.isnan(expected)
    integer = np.uint32 if actual.dtype == np.float32 else np.uint64
    np.testing.assert_array_equal(
        actual[finite].view(integer), expected[finite].view(integer)
    )


@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1), (17, 31), (259, 263)])
@pytest.mark.parametrize("channels", [3, 4, 7])
@pytest.mark.parametrize("pattern", ["random", "signed", "special", "exponents"])
def test_luminance_dot_order(shape, channels, pattern):
    rng = np.random.default_rng(869)
    image = rng.random((*shape, channels), dtype=np.float32)
    if pattern == "signed":
        image = image * 2 - 1
    elif pattern == "special":
        image = rng.choice(
            np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 1, -1], np.float32),
            image.shape,
        )
    elif pattern == "exponents":
        image = np.ldexp(image, rng.integers(-149, 127, image.shape, dtype=np.int32))
    with np.errstate(all="ignore"):
        expected = np.dot(image[..., :3], [0.2126, 0.7152, 0.0722])
    exact(native_cas.luminance(image), expected)


KERNELS = [
    np.ones(shape, np.uint8)
    for shape in [(1, 1), (1, 3), (3, 1), (3, 3), (2, 4), (5, 7)]
]
KERNELS += [
    cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
    np.array([[0, 1, 0, 2], [1, 0, 0, 1]], np.uint8),
]


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (9, 1), (2, 3), (7, 8), (19, 17)])
@pytest.mark.parametrize(
    "pattern", ["random", "negative_zero", "nonfinite", "extremes"]
)
def test_double_morphology_all_orders(kernel, shape, pattern):
    rng = np.random.default_rng(587)
    image = rng.uniform(-1, 1, shape)
    if pattern == "negative_zero":
        image = rng.choice(np.array([0.0, -0.0], np.float64), shape)
    elif pattern == "nonfinite":
        image = rng.choice(
            np.array([np.nan, np.inf, -np.inf, 0, 1, -1], np.float64), shape
        )
    elif pattern == "extremes":
        image = rng.choice(
            np.array(
                [np.finfo(np.float64).max, -np.finfo(np.float64).max, np.inf, -np.inf],
                np.float64,
            ),
            shape,
        )
    low, high = native_cas.extrema(image, kernel)
    exact(low, cv2.erode(image, kernel))
    exact(high, cv2.dilate(image, kernel))


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(1, 1), (1, 13), (17, 1), (13, 19)])
@pytest.mark.parametrize("kind", [cv2.MORPH_RECT, cv2.MORPH_CROSS])
@pytest.mark.parametrize("bias", [1, 1.3, 2, 3])
@pytest.mark.parametrize("pattern", ["random", "special"])
def test_frozen_cas_mask_and_mix(channels, shape, kind, bias, pattern):
    rng = np.random.default_rng(644)
    full_shape = shape if channels == 1 else (*shape, channels)
    image = rng.random(full_shape, dtype=np.float32)
    if pattern == "special":
        image = rng.choice(
            np.array([np.nan, np.inf, -np.inf, -0.0, 0, 0.5, 1], np.float32), full_shape
        )
    sharpened = rng.uniform(-1, 2, full_shape).astype(np.float32)
    kernel = cv2.getStructuringElement(kind, (3, 3))
    with np.errstate(all="ignore"):
        expected_mask = ORIGINAL.create_cas_mask(image, kernel, bias)
        expected_mix = ORIGINAL.cas_mix(image, sharpened, kernel, bias)
        exact(cas.create_cas_mask(image, kernel, bias), expected_mask)
        exact(cas.cas_mix(image, sharpened, kernel, bias), expected_mix)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("kind", [0, 1])
@pytest.mark.parametrize("amount", [0, 0.1, 2, 100])
@pytest.mark.parametrize("bias", [1, 1.3, 2, 3])
def test_high_boost_entire_cas_path(channels, kind, amount, bias):
    shape = (13, 17) if channels == 1 else (13, 17, channels)
    image = np.random.default_rng(111).random(shape, dtype=np.float32)
    actual = BOOST.high_boost_filter_node(
        image, BOOST.KernelType(kind), amount, True, bias
    )
    expected = BOOST_REF.high_boost_filter_node(
        image, BOOST_REF.KernelType(kind), amount, True, bias
    )
    exact(actual, expected)
    if amount == 0:
        assert actual is image


@pytest.mark.parametrize("layout", ["reverse", "transpose", "readonly", "unaligned"])
def test_foreign_buffers(layout):
    image = np.random.default_rng(235).random((17, 19, 4), dtype=np.float32)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    if layout == "reverse":
        image, kernel = image[::-1, ::-1], kernel[::-1, ::-1]
    elif layout == "transpose":
        image, kernel = image.swapaxes(0, 1), kernel.T
    elif layout == "readonly":
        image.setflags(write=False)
        kernel.setflags(write=False)
    else:
        raw = np.ndarray(image.shape, image.dtype, bytearray(image.nbytes + 1), 1)
        raw[:] = image
        image = raw
    saved = image.copy()
    exact(
        cas.create_cas_mask(image, kernel, 1.3),
        ORIGINAL.create_cas_mask(image, kernel, 1.3),
    )
    exact(image, saved)
    double = np.dot(image[..., :3], [0.2126, 0.7152, 0.0722])
    if layout == "unaligned":
        raw_double = np.ndarray(
            double.shape, double.dtype, bytearray(double.nbytes + 1), 1
        )
        raw_double[:] = double
        double = raw_double
    else:
        double = double[::-1, ::-1]
    low, high = native_cas.extrema(double, kernel)
    exact(low, cv2.erode(double, kernel))
    exact(high, cv2.dilate(double, kernel))


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_high_boost_no_opencv_or_numpy_pixel_arithmetic(channels, monkeypatch):
    shape = (13, 17) if channels == 1 else (13, 17, channels)
    image = np.random.default_rng(394).random(shape, dtype=np.float32)
    expected = BOOST_REF.high_boost_filter_node(
        image, BOOST_REF.KernelType.STRONG, 2, True, 2
    )

    def prohibited(*_args, **_kwargs):
        pytest.fail("Converted CAS processing called its original primitive")

    for name in ("dot", "power"):
        monkeypatch.setattr(np, name, prohibited)
    for name in ("erode", "dilate", "filter2D"):
        monkeypatch.setattr(cv2, name, prohibited)
    exact(
        BOOST.high_boost_filter_node(image, BOOST.KernelType.STRONG, 2, True, 2),
        expected,
    )


def test_concurrent_double_extrema():
    rng = np.random.default_rng(921)
    images = [rng.random((131, 137)) for _ in range(8)]
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    expected = [
        (cv2.erode(image, kernel), cv2.dilate(image, kernel)) for image in images
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda image: native_cas.extrema(image, kernel), images))
    for pair, reference in zip(actual, expected, strict=True):
        for got, wanted in zip(pair, reference, strict=True):
            exact(got, wanted)


def test_invalid_c_counts_leave_output_untouched():
    d = ct.c_double(987)
    p = ct.pointer(d)
    f = ct.pointer(ct.c_float())
    k = ct.pointer(ct.c_uint8(1))
    native = lib()
    assert native.cn_cas_luminance(None, p, 1, 3, 1) == 1
    assert native.cn_cas_luminance(f, p, ct.c_size_t(-1).value, 3, 1) == 2
    assert native.cn_cas_extrema_f64(p, p, p, 0, 1, k, 1, 1) == 1
    assert native.cn_cas_extrema_f64(p, p, p, 2**31, 1, k, 1, 1) == 2
    assert d.value == 987


def test_double_power_uses_installed_public_runtime_rounding():
    low = np.array([[0.0770269491314888]], np.float64)
    high = np.array([[0.7824119388580322]], np.float64)
    expected = np.power(np.minimum(1 - high, low) / (high + 1e-8), 1 / 2.3)
    exact(cas_mask(low, high, 2.3), expected)
