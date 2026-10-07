"""Exact Lens arithmetic and remaining filter/morphology node comparisons."""

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

from nodes.impl import native_convolution, native_lens
from nodes.impl.native import lib
from nodes.impl.native_filters import normalize_lens

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/src/packages/chaiNNer_standard/image_filter"
REFERENCES = Path(__file__).parent


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


LENS = load_body(BASE / "blur/lens_blur.py")
LENS_REF = load_body(REFERENCES / "reference_filters/blur/lens_blur.py")
EDGE = load_body(BASE / "miscellaneous/edge_detection.py")
EDGE_REF = load_body(REFERENCES / "reference_filters/miscellaneous/edge_detection.py")
BOOST = load_body(BASE / "sharpen/high_boost_filter.py")
BOOST_REF = load_body(
    REFERENCES
    / "reference_analysis/packages/chaiNNer_standard/image_filter/sharpen/high_boost_filter.py"
)
_cas_spec = importlib.util.spec_from_file_location(
    "nodes.impl.reference_lens_cas",
    REFERENCES / "reference_analysis/nodes/impl/cas.py",
)
assert _cas_spec is not None and _cas_spec.loader is not None
_cas_reference = importlib.util.module_from_spec(_cas_spec)
_cas_spec.loader.exec_module(_cas_reference)
BOOST_REF.__dict__.update(cas_mix=_cas_reference.cas_mix)
PARAMETERS = [
    (scale, params["a"], params["b"])
    for count in range(1, 7)
    for params, scale in [
        (p, LENS_REF.get_parameters(count)[1])
        for p in LENS_REF.get_parameters(count)[0]
    ]
]


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    if actual.dtype == np.complex64:
        actual, expected = actual.view(np.float32), expected.view(np.float32)
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[finite].view(np.uint32), expected[finite].view(np.uint32)
    )


@pytest.mark.parametrize(
    "radius", [*range(1, 33), 63, 64, 65, 127, 255, 511, 999, 1000]
)
@pytest.mark.parametrize("parameters", PARAMETERS)
def test_all_component_kernel_parameters(radius, parameters):
    exact(
        LENS.complex_kernel_1d(radius, *parameters),
        LENS_REF.complex_kernel_1d(radius, *parameters),
    )


@pytest.mark.parametrize("radius", [1, 3, 7])
@pytest.mark.parametrize("count", range(1, 7))
def test_kernel_normalization_is_exact(radius, count):
    params, scale = LENS_REF.get_parameters(count)
    kernels = [
        LENS_REF.complex_kernel_1d(radius, scale, p["a"], p["b"]) for p in params
    ]
    actual = normalize_lens(kernels, [(p["A"], p["B"]) for p in params])
    expected = LENS_REF.normalize_kernels(kernels, params)
    for got, wanted in zip(actual, expected, strict=True):
        exact(got, wanted)


# Every path power_range picks once per range: FLOAT_power's -1, 0, 0.5, 1 and 2
# fast paths (-0 takes the 0 path) and powf (NaN included). The finishing -1 is
# test_forced_paths.py's: its maxss(+0, x) input clamp keeps -0, which np.clip does not.
@pytest.mark.parametrize(
    "gamma,finish",
    [
        (gamma, finish)
        for gamma in (0.01, 0.1, 0.3, 0.5, 1, 1.5, 2, 3, 5, 12, 100, 0.0, -0.0, np.nan)
        for finish in (False, True)
    ]
    + [(-1.0, False)],
)
@pytest.mark.parametrize("pattern", ["random", "signed", "special", "exponents"])
def test_power_exact(gamma, finish, pattern):
    rng = np.random.default_rng(758)
    image = rng.random((17, 19), dtype=np.float32)
    if pattern == "signed":
        image = image * 2 - 1
    elif pattern == "special":
        image[:] = np.resize(
            np.array(
                [np.nan, np.inf, -np.inf, -0.0, 0, 1, -1, np.finfo(np.float32).tiny],
                np.float32,
            ),
            image.shape,
        )
    elif pattern == "exponents":
        image = np.ldexp(image, rng.integers(-149, 127, image.shape, dtype=np.int32))
    with np.errstate(all="ignore"):
        expected = np.power(np.clip(image, 0, None) if finish else image, gamma)
        if finish:
            expected = np.clip(expected, 0, 1)
    exact(native_lens.power(image, gamma, finish=finish), expected)


@pytest.mark.parametrize("pattern", ["random", "signed_zero", "nonfinite"])
@pytest.mark.parametrize("accumulate", [False, True])
def test_component_complex_arithmetic(pattern, accumulate):
    rng = np.random.default_rng(593)
    f1, f2, f3, f4 = rng.uniform(-1, 1, (4, 13, 17)).astype(np.float32)
    if pattern == "signed_zero":
        values = np.array([-0.0, 0.0], np.float32)
        f1, f2, f3, f4 = values[rng.integers(0, 2, (4, 13, 17))]
    elif pattern == "nonfinite":
        values = np.array([np.inf, -np.inf, np.nan, -0.0, 0.0, -1, 1], np.float32)
        f1, f2, f3, f4 = values[rng.integers(0, 7, (4, 13, 17))]
    out = rng.random((13, 17), dtype=np.float32)
    a, b = 0.767583, 1.862321
    with np.errstate(all="ignore"):
        component = f1 - f4 + 1j * (f2 + f3)
        expected = component.real * a + component.imag * b
        if accumulate:
            expected = out + expected
    native_lens.compose(f1, f2, f3, f4, a, b, out=out, accumulate=accumulate)
    exact(out, expected)


@pytest.mark.parametrize("radius", [1, 3, 7])
@pytest.mark.parametrize("count", [1, 3, 6])
@pytest.mark.parametrize("gamma", [0.01, 0.3, 0.5, 1, 2, 5, 100])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_full_lens_against_frozen_original(radius, count, gamma, channels):
    shape = (7, 9) if channels == 1 else (7, 9, channels)
    image = np.random.default_rng(857).random(shape, dtype=np.float32)
    exact(
        LENS.lens_blur_node(image, radius, count, gamma),
        LENS_REF.lens_blur_node(image, radius, count, gamma),
    )


@pytest.mark.parametrize("radius", [24, 25, 64, 65])
@pytest.mark.parametrize("optimized", [False, True])
def test_lens_dft_transition(radius, optimized):
    previous = cv2.useOptimized()
    try:
        cv2.setUseOptimized(optimized)
        image = np.random.default_rng(749).random((7, 9, 3), dtype=np.float32)
        exact(
            LENS.lens_blur_node(image, radius, 1, 1),
            LENS_REF.lens_blur_node(image, radius, 1, 1),
        )
    finally:
        cv2.setUseOptimized(previous)


@pytest.mark.parametrize("shape", [(1, 1), (1, 13), (17, 1), (9, 11, 1), (9, 11, 4)])
@pytest.mark.parametrize("pattern", ["random", "zero", "negative_zero", "nonfinite"])
def test_lens_dimensions_and_special_values(shape, pattern):
    image = np.random.default_rng(746).random(shape, dtype=np.float32)
    if pattern == "zero":
        image.fill(0)
    elif pattern == "negative_zero":
        image.fill(-0.0)
    elif pattern == "nonfinite":
        image.ravel()[::7] = np.nan
        image.ravel()[1::13] = np.inf
    with np.errstate(all="ignore"):
        exact(
            LENS.lens_blur_node(image, 3, 2, 1), LENS_REF.lens_blur_node(image, 3, 2, 1)
        )


@pytest.mark.parametrize("algorithm", list(EDGE.Algorithm))
@pytest.mark.parametrize("gradient", list(EDGE.GradientComponent))
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_edge_modes(algorithm, gradient, channels):
    image = np.random.default_rng(83).random(
        (9, 13) if channels == 1 else (9, 13, channels), dtype=np.float32
    )
    exact(
        EDGE.edge_detection_node(image, 1, algorithm, gradient, 1, 2),
        EDGE_REF.edge_detection_node(
            image,
            1,
            EDGE_REF.Algorithm(algorithm.value),
            EDGE_REF.GradientComponent(gradient.value),
            1,
            2,
        ),
    )


@pytest.mark.parametrize("kind", list(BOOST.KernelType))
@pytest.mark.parametrize("amount", [0.1, 2, 100])
@pytest.mark.parametrize("cas", [False, True])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_high_boost_modes(kind, amount, cas, channels):
    image = np.random.default_rng(19).random(
        (9, 13) if channels == 1 else (9, 13, channels), dtype=np.float32
    )
    exact(
        BOOST.high_boost_filter_node(image, kind, amount, cas, 2),
        BOOST_REF.high_boost_filter_node(
            image, BOOST_REF.KernelType(kind.value), amount, cas, 2
        ),
    )


@pytest.mark.parametrize("layout", ["strided", "transposed", "readonly", "unaligned"])
def test_lens_foreign_images(layout):
    image = np.random.default_rng(9).random((9, 13, 4), dtype=np.float32)
    if layout == "strided":
        image = image[::-1, ::-1]
    elif layout == "transposed":
        image = image.swapaxes(0, 1)
    elif layout == "readonly":
        image.setflags(write=False)
    else:
        array = np.ndarray(image.shape, image.dtype, bytearray(image.nbytes + 1), 1)
        array[:] = image
        image = array
    before = image.copy()
    exact(
        LENS.lens_blur_node(image, 3, 3, 0.3), LENS_REF.lens_blur_node(image, 3, 3, 0.3)
    )
    exact(image, before)


def test_filter_paths_use_c(monkeypatch):
    image = np.random.default_rng(937).random((13, 17, 4), dtype=np.float32)
    lens_expected = LENS_REF.lens_blur_node(image, 3, 3, 5)
    edge_expected = EDGE_REF.edge_detection_node(
        image,
        1,
        EDGE_REF.Algorithm.LAPLACIAN_DENOISE,
        EDGE_REF.GradientComponent.MAGNITUDE,
        1,
        2,
    )
    boost_expected = BOOST_REF.high_boost_filter_node(
        image, BOOST_REF.KernelType.STRONG, 2, False, 2
    )

    def prohibited(*_args, **_kwargs):
        pytest.fail("A converted Lens or spatial filtering primitive was called")

    for name in ("filter2D", "dilate", "erode"):
        monkeypatch.setattr(cv2, name, prohibited)
    for name in ("exp", "sin", "cos", "power"):
        monkeypatch.setattr(np, name, prohibited)
    exact(LENS.lens_blur_node(image, 3, 3, 5), lens_expected)
    exact(
        EDGE.edge_detection_node(
            image,
            1,
            EDGE.Algorithm.LAPLACIAN_DENOISE,
            EDGE.GradientComponent.MAGNITUDE,
            1,
            2,
        ),
        edge_expected,
    )
    exact(
        BOOST.high_boost_filter_node(image, BOOST.KernelType.STRONG, 2, False, 2),
        boost_expected,
    )


def test_concurrent_lens_images():
    rng = np.random.default_rng(184)
    images = [rng.random((17, 19, 4), dtype=np.float32) for _ in range(8)]
    expected = [LENS_REF.lens_blur_node(image, 3, 3, 5) for image in images]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(
            pool.map(lambda image: LENS.lens_blur_node(image, 3, 3, 5), images)
        )
    for got, wanted in zip(actual, expected, strict=True):
        exact(got, wanted)


@pytest.mark.parametrize("radius", [0, -1, 1001, 1.5])
def test_kernel_invalid_radius(radius):
    with pytest.raises((ValueError, TypeError)):
        native_lens.complex_kernel(radius, 1.2, 1, 2)


def test_native_rejects_malformed_inputs():
    dll = lib()
    p = ct.pointer(ct.c_float())
    assert dll.cn_lens_kernel(None, 1, 1, 1, 1, 1) == 1
    assert dll.cn_lens_kernel(p, 1001, 1, 1, 1, 1) == 1
    assert dll.cn_lens_power(None, p, 1, 1, 0) == 1
    assert dll.cn_lens_power(p, p, ct.c_size_t(-1).value, 1, 0) == 2
    assert dll.cn_lens_compose(p, p, p, p, p, 1, 1, 1, 2) == 1


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 17), (13, 1), (7, 19), (7, 19, 3), (7, 19, 4)]
)
@pytest.mark.parametrize("kernel_shape", [(1, 1), (3, 3), (4, 3), (1, 49)])
@pytest.mark.parametrize(
    "pattern",
    [
        "input_nan",
        "input_inf",
        "kernel_nan",
        "kernel_inf",
        "negative_zero",
        "mixed",
        "all_nan",
        "all_inf",
        "double_overflow",
    ],
)
def test_spatial_nonfinite_remains_in_c(
    optimized, shape, kernel_shape, pattern, monkeypatch
):
    original_optimized = cv2.useOptimized()
    rng = np.random.default_rng(758)
    image = rng.uniform(-1, 1, shape).astype(np.float32)
    kernel = rng.uniform(-1, 1, kernel_shape)
    if pattern == "input_nan":
        image.ravel()[::7] = np.nan
    elif pattern == "input_inf":
        image.ravel()[::7] = np.inf
    elif pattern == "kernel_nan":
        kernel.ravel()[::3] = np.nan
    elif pattern == "kernel_inf":
        kernel.ravel()[::3] = np.inf
    elif pattern == "negative_zero":
        image.fill(-0.0)
    elif pattern == "mixed":
        image[:] = rng.choice(
            np.array([np.nan, np.inf, -np.inf, -0.0, 0, 1, -1], np.float32), shape
        )
    elif pattern == "all_nan":
        kernel.fill(np.nan)
    elif pattern == "all_inf":
        kernel.fill(np.inf)
    else:
        kernel.ravel()[::3] = 1e300

    def prohibited(*_args, **_kwargs):
        pytest.fail("Supported spatial convolution used OpenCV")

    try:
        cv2.setUseOptimized(optimized)
        expected = cv2.filter2D(image, -1, kernel)
        monkeypatch.setattr(cv2, "filter2D", prohibited)
        exact(native_convolution.convolve(image, kernel, 0), expected)
    finally:
        cv2.setUseOptimized(original_optimized)
