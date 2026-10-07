"""Frozen-body filter comparisons, including CPU concurrency and bad buffers."""

from __future__ import annotations

import ast
import ctypes as ct
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from nodes.impl import native_filters as native
from nodes.impl.native import lib

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "backend/src/packages/chaiNNer_standard/image_filter"
REFERENCE = Path(__file__).with_name("reference_filters")
PATHS = {
    "lens_blur": "blur/lens_blur.py",
    "edge_detection": "miscellaneous/edge_detection.py",
    "dilate": "miscellaneous/dilate.py",
    "erode": "miscellaneous/erode.py",
    "distance_transform": "miscellaneous/distance_transform.py",
    "quantize_to_reference": "quantize/quantize_to_reference.py",
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
    name: (load_body(SOURCE / relative), load_body(REFERENCE / relative))
    for name, relative in PATHS.items()
}


def source(channels=3, layout="contiguous", height=19, width=23, seed=917):
    shape = (height, width) if channels == 1 else (height, width, channels)
    image = np.random.default_rng(seed).random(shape, dtype=np.float32)
    if layout == "strided":
        image = image[::-1, ::2]
    elif layout == "transposed":
        image = image.swapaxes(0, 1)
    elif layout == "readonly":
        image.flags.writeable = False
    elif layout == "unaligned":
        other = np.ndarray(
            shape, np.float32, buffer=bytearray(image.nbytes + 1), offset=1
        )
        other[:] = image
        image = other
    return image


def arrays(actual, expected, *, rtol=0, atol=0):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("operation", ["dilate", "erode"])
@pytest.mark.parametrize("shape", ["RECTANGLE", "CROSS", "ELLIPSE"])
@pytest.mark.parametrize(
    "radius,iterations", [(0, 1), (1, 0), (1, 1), (2, 3), (19, 2), (1000, 1)]
)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_morphology(operation, shape, radius, iterations, channels):
    image = source(channels, "readonly")
    saved = image.copy()
    actual, original = MODULES[operation]
    results = [
        getattr(module, operation + "_node")(
            image, getattr(module.MorphShape, shape), radius, iterations
        )
        for module in (actual, original)
    ]
    arrays(*results)
    arrays(image, saved)
    if radius == 0 or iterations == 0:
        assert results[0] is image


@pytest.mark.parametrize("layout", ["strided", "transposed", "unaligned"])
@pytest.mark.parametrize("size", [(1, 1), (1, 17), (17, 1), (19, 29), (257, 279)])
@pytest.mark.parametrize("maximum", [False, True])
def test_morphology_layout_and_deque_bounds(layout, size, maximum):
    image = source(3, layout, height=size[0], width=size[1])
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
    original = cv2.dilate if maximum else cv2.erode
    arrays(
        native.morphology(image, cv2.MORPH_CROSS, 3, 2, maximum=maximum),
        original(image, kernel, iterations=2),
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("maximum", [False, True])
def test_morphology_nonfinite_preserves_opencv(value, maximum):
    image = source(3)
    image[2, 4, 1] = value
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    original = cv2.dilate if maximum else cv2.erode
    arrays(
        native.morphology(image, cv2.MORPH_RECT, 1, 2, maximum=maximum),
        original(image, kernel, iterations=2),
    )


@pytest.mark.parametrize("components", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("radius", [1, 3, 9])
def test_lens_normalization(components, radius):
    actual, original = MODULES["lens_blur"]
    parameters, scale = original.get_parameters(components)
    kernels = [
        original.complex_kernel_1d(radius, scale, p["a"], p["b"]) for p in parameters
    ]
    expected = original.normalize_kernels(kernels, parameters)
    result = actual.normalize_kernels(kernels, parameters)
    for a, b in zip(result, expected, strict=True):
        arrays(a, b)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("components", [1, 3, 6])
@pytest.mark.parametrize("gamma", [0.3, 1, 5, 12])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "readonly"])
def test_lens_image(channels, components, gamma, layout):
    image = source(channels, layout)
    saved = image.copy()
    actual, original = MODULES["lens_blur"]
    # The C kernel, power, composition and filter paths preserve the installed
    # float32 operation order; saved-image rounding must not hide discrepancies.
    arrays(
        actual.lens_blur_node(image, 2, components, gamma),
        original.lens_blur_node(image, 2, components, gamma),
    )
    arrays(image, saved)


@pytest.mark.parametrize("algorithm", list(range(1, 10)))
@pytest.mark.parametrize("component", [1, 2, 3])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("amount", [0, 1, 7.3])
def test_edge_detection(algorithm, component, channels, amount):
    image = source(channels, "readonly")
    saved = image.copy()
    actual, original = MODULES["edge_detection"]
    results = [
        module.edge_detection_node(
            image,
            amount,
            module.Algorithm(algorithm),
            module.GradientComponent(component),
            0.7,
            2.1,
        )
        for module in (actual, original)
    ]
    arrays(*results)
    arrays(image, saved)


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("radius", [1, 2, 5])
@pytest.mark.parametrize("spatial", [0, 0.1, 35.7, 100])
@pytest.mark.parametrize("scale_y,scale_x", [(1, 1), (2, 2), (3, 3), (2, 3), (3, 2)])
def test_reference_quantization(channels, radius, spatial, scale_y, scale_x):
    reference = source(channels, "readonly", height=3, width=4)
    image = source(channels, "readonly", height=3 * scale_y, width=4 * scale_x, seed=23)
    saved, saved_ref = image.copy(), reference.copy()
    actual, original = MODULES["quantize_to_reference"]
    arrays(
        actual.quantize_to_reference_node(image, reference, radius, spatial),
        original.quantize_to_reference_node(image, reference, radius, spatial),
    )
    arrays(image, saved)
    arrays(reference, saved_ref)


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1)])
@pytest.mark.parametrize("spatial", [0, 25, 100])
def test_reference_single_axes_and_ties(shape, spatial):
    reference = np.zeros((*shape, 3), dtype=np.float32)
    reference[..., 0] = (
        np.arange(reference.shape[0] * reference.shape[1]).reshape(shape) % 2
    )
    image = np.full((shape[0] * 3, shape[1] * 3, 3), 0.5, dtype=np.float32)
    actual, original = MODULES["quantize_to_reference"]
    arrays(
        actual.quantize_to_reference_node(image, reference, 5, spatial),
        original.quantize_to_reference_node(image, reference, 5, spatial),
    )


@pytest.mark.parametrize(
    "size", [(1, 1), (1, 15), (15, 1), (19, 23), (137, 149), (263, 257)]
)
@pytest.mark.parametrize("pattern", ["zero", "one", "random", "center"])
@pytest.mark.parametrize("spread", [1, 4, 19, 100])
def test_binary_distance(size, pattern, spread):
    image = source(1, height=size[0], width=size[1])
    if pattern == "zero":
        image.fill(0)
    elif pattern == "one":
        image.fill(1)
    elif pattern == "center":
        image.fill(1)
        image[size[0] // 2, size[1] // 2] = 0
    saved = image.copy()
    actual, original = MODULES["distance_transform"]
    arrays(
        actual.distance_transform_node(image, spread, False),
        original.distance_transform_node(image, spread, False),
    )
    arrays(image, saved)


def test_distance_scalar_opencv_path():
    image = source(1)
    actual, original = MODULES["distance_transform"]
    enabled = cv2.ipp.useIPP()
    try:
        cv2.ipp.setUseIPP(False)
        arrays(
            actual.distance_transform_node(image, 4, False),
            original.distance_transform_node(image, 4, False),
        )
    finally:
        cv2.ipp.setUseIPP(enabled)


@pytest.mark.parametrize("shape", [(4,), (0, 4), (3, 5, 0), (2, 3, 4, 5)])
def test_invalid_image_shape(shape):
    image = np.empty(shape, dtype=np.float32)
    for function in (
        lambda: native.arithmetic(image, 6),
        lambda: native.morphology(image, cv2.MORPH_RECT, 1, 1, maximum=True),
        lambda: native.quantize_reference(image, image, 1, 0),
    ):
        with pytest.raises(ValueError):
            function()


def test_malformed_secondary_buffers():
    image = source(3)
    with pytest.raises(ValueError):
        native.arithmetic(image, 0, b=image[:-1])
    with pytest.raises(ValueError):
        native.arithmetic(image, 9, b=image, c=image, d=image, out=image[..., :1])
    with pytest.raises(ValueError):
        native.quantize_reference(image, image[..., :2], 1, 0)
    with pytest.raises(ValueError):
        native.normalize_lens([np.ones((3, 3), np.complex64)], [(1, 2)])
    with pytest.raises(ValueError):
        native.binary_sdf(np.zeros((3, 3, 3), np.uint8), 4)
    with pytest.raises(TypeError):
        native.arithmetic(image.astype(np.float64), 6)


def test_checked_c_counts():
    dll = lib()
    sentinel = ct.c_float(731)
    p = ct.pointer(sentinel)
    assert dll.cn_filter_arithmetic(p, None, None, None, p, 1, 0, 1, 0, 0) == 1
    assert dll.cn_filter_morphology(p, p, ct.c_size_t(-1).value, 2, 3, 1, 1, 0, 1) == 2
    assert dll.cn_filter_quantize_reference(p, p, p, 1, 1, 1, 1, 3, 0, 0) == 1
    assert sentinel.value == 731


def test_concurrent_filters():
    image = source(3, "readonly", height=132, width=156)
    reference = source(3, "readonly", height=44, width=52)

    def operations():
        return (
            native.morphology(image, cv2.MORPH_RECT, 19, 3, maximum=True),
            native.quantize_reference(image, reference, 2, 17),
            native.arithmetic(image, 7, 2.3),
        )

    expected = operations()
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: operations(), range(16)))
    for result in results:
        for actual, original in zip(result, expected, strict=True):
            arrays(actual, original)


@pytest.mark.parametrize("layout", ["reversed", "unaligned"])
def test_lens_foreign_coefficient_buffers(layout):
    original = MODULES["lens_blur"][1]
    parameters, scale = original.get_parameters(2)
    kernels = [original.complex_kernel_1d(3, scale, p["a"], p["b"]) for p in parameters]
    values = np.array([[p["A"], p["B"]] for p in parameters], dtype=np.float64)
    if layout == "reversed":
        values = values[::-1]
        assert values.strides[0] < 0
    else:
        buffer = np.ndarray(
            values.shape, np.float64, buffer=bytearray(values.nbytes + 1), offset=1
        )
        buffer[:] = values
        values = buffer
        assert not values.flags.aligned
    saved = values.copy()
    expected = native.normalize_lens(kernels, values.tolist())
    actual = native.normalize_lens(kernels, values)
    for a, b in zip(actual, expected, strict=True):
        arrays(a, b)
    arrays(values, saved)


def test_filter_accumulation_requires_initialized_output():
    image = source(1)
    with pytest.raises(ValueError, match="initialized output"):
        native.arithmetic(image, 10, b=image, c=image, d=image)


@pytest.mark.parametrize("operand", ["image", "b", "c", "d"])
@pytest.mark.parametrize("layout", ["same", "forward", "backward", "strided"])
def test_filter_rejects_output_aliases(operand, layout):
    storage = np.arange(33, dtype=np.float32)
    if layout == "same":
        value = storage[:16].reshape(4, 4)
        out = value
    elif layout == "forward":
        value = storage[:16].reshape(4, 4)
        out = storage[1:17].reshape(4, 4)
    elif layout == "backward":
        value = storage[1:17].reshape(4, 4)
        out = storage[:16].reshape(4, 4)
    else:
        value = storage[:32:2].reshape(4, 4)
        out = storage[:16].reshape(4, 4)
    inputs: dict[str, np.ndarray] = {
        name: np.ones((4, 4), np.float32) for name in ("image", "b", "c", "d")
    }
    inputs[operand] = value
    saved = storage.copy()
    with pytest.raises(ValueError, match="overlap"):
        native.arithmetic(
            image=inputs["image"],
            b=inputs["b"],
            c=inputs["c"],
            d=inputs["d"],
            operation=10,
            out=out,
        )
    arrays(storage, saved)
