"""Frozen-node comparisons for reflected blur and padded spatial convolution.

Box Blur is compared as the user receives its output, ImageOutput.enforce of both
nodes' returns, and its kernel parity stays exact at the helpers: RAW_BOX, the body
whose returns are box_blur's, separable_box's or filter2d's raw results, against the
reference's OpenCV results. Both comparisons are bitwise (SP4b Task 3b2; owner,
2026-10-04: a raw pre-enforce return is an internal detail).
"""

from __future__ import annotations

import ast
import ctypes as ct
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_adjustment_complete import exact
from test_normalized_outputs import RAW_BOX, assert_enforced_equal, node_schema

from nodes.impl import native_convolution as native
from nodes.impl.native import lib

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "backend/src/packages/chaiNNer_standard/image_filter"
REFERENCE = Path(__file__).with_name("reference_convolution")
PATHS = {"box_blur": "blur/box_blur.py", "convolve": "miscellaneous/convolve.py"}


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
INSTALLED_BOX = load_body(REFERENCE / "blur/box_blur_installed.py")


def image(channels=3, height=17, width=23, layout="normal", seed=571):
    shape = (height, width) if channels == 1 else (height, width, channels)
    source = np.random.default_rng(seed).random(shape, dtype=np.float32)
    if layout == "strided":
        source = source[::-1, ::2]
    elif layout == "transposed":
        source = source.swapaxes(0, 1)
    elif layout == "readonly":
        source.flags.writeable = False
    elif layout == "unaligned":
        other = np.ndarray(
            shape, np.float32, buffer=bytearray(source.nbytes + 1), offset=1
        )
        other[:] = source
        source = other
    elif layout == "singleton" and channels == 1:
        source = source[:, :, None]
    return source


def equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def box_parity(source, radii, reference):
    """Box Blur against the reference node: the outputs as the user receives them
    (ImageOutput.enforce of both returns) and the helpers' raw results (RAW_BOX)
    against the reference's return, each bitwise."""
    expected = reference.box_blur_node(source, *radii)
    assert_enforced_equal(
        MODULES["box_blur"][0].box_blur_node(source, *radii), expected
    )
    exact(RAW_BOX.box_blur_node(source, *radii), expected)


RADII = [(0, 0), (1, 0), (0, 3), (1, 1), (2, 3), (7, 4), (15, 19), (1000, 1000)]


@pytest.mark.parametrize("radii", RADII)
@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (7, 1), (13, 17)])
def test_box_integer(radii, channels, shape):
    source = image(channels, *shape)
    saved = source.copy()
    actual, original = MODULES["box_blur"]
    box_parity(source, radii, original)
    equal(source, saved)
    if radii == (0, 0):
        assert actual.box_blur_node(source, *radii) is source


@pytest.mark.parametrize(
    "radii", [(0.1, 0.2), (1.5, 3.2), (15, 1.2), (1.3, 16), (201.3, 1.7), (70.5, 80.2)]
)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_box_fractional_original_paths(radii, channels):
    source = image(channels, 15, 23)
    box_parity(source, radii, MODULES["box_blur"][1])


@pytest.mark.parametrize("radii", [*RADII, (1.2, 15), (203.6, 201.5)])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_box_installed_original(radii, channels):
    box_parity(image(channels), radii, INSTALLED_BOX)


def test_box_schema_unchanged():
    relative = PATHS["box_blur"]
    assert node_schema(SOURCE / relative) == node_schema(REFERENCE / relative)


@pytest.mark.parametrize("radii", [(1, 1), (2, 3), (7, 4), (15, 19)])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("kind", ["subnormal", "subnormal_signed", "halfway"])
def test_box_tiny_and_halfway_values(radii, channels, kind):
    source = image(channels, 13, 17)
    if kind == "halfway":
        half = np.float32(0.5)
        values = np.array(
            [np.nextafter(half, np.float32(0)), half, np.nextafter(half, np.float32(1))]
        )
        source = values[np.random.default_rng(19).integers(0, 3, size=source.shape)]
    elif kind == "subnormal":
        source *= np.float32(1e-38)
    else:
        source = (source * 2 - 1) * np.float32(1e-38)
    box_parity(source, radii, INSTALLED_BOX)


@pytest.mark.parametrize(
    "layout", ["strided", "transposed", "readonly", "unaligned", "singleton"]
)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_box_foreign(layout, channels):
    source = image(channels, layout=layout)
    saved = source.copy()
    box_parity(source, (3, 5), MODULES["box_blur"][1])
    equal(source, saved)


@pytest.mark.parametrize(
    "pattern",
    [
        "constant",
        "signed_zero",
        "checkerboard",
        "quantization_half",
        "signed",
        "exponent_range",
        "nan",
        "inf",
    ],
)
@pytest.mark.parametrize("radii", [(1, 1), (2, 3), (19, 15)])
def test_box_numeric_boundaries(pattern, radii):
    source = image(3, 13, 17)
    if pattern == "constant":
        source.fill(np.float32(0.1))
    elif pattern == "signed_zero":
        source[source < 0.5] = -0.0
        source[source >= 0.5] = 0.0
    elif pattern == "checkerboard":
        source = np.repeat(
            (np.indices((13, 17)).sum(axis=0) % 2).astype(np.float32)[:, :, None],
            3,
            axis=2,
        )
    elif pattern == "quantization_half":
        source = (np.floor(source * 255) + np.float32(0.5)) / np.float32(255)
    elif pattern == "signed":
        source = source * 2 - 1
    elif pattern == "exponent_range":
        source = np.ldexp(
            source,
            np.random.default_rng(773).integers(
                -120, 120, size=source.shape, dtype=np.int32
            ),
        )
    elif pattern == "nan":
        source[6, 9, 2] = np.nan
    else:
        source[6, 9, 2] = np.inf
    box_parity(source, radii, MODULES["box_blur"][1])


def kernel_text(kernel):
    return "\n".join(" ".join(repr(float(value)) for value in row) for row in kernel)


KERNELS = [
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 3),
    (5, 5),
    (11, 11),
    (1, 129),
    (1, 130),
    (13, 13),
]


@pytest.mark.parametrize("shape", KERNELS)
@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("padding", [0, 2, 9])
def test_convolve_shapes(shape, channels, padding):
    source = image(channels, 7, 17)
    kernel = np.random.default_rng(832).random(shape) - 0.5
    text = kernel_text(kernel)
    actual, original = MODULES["convolve"]
    saved = source.copy()
    equal(
        actual.convolve_node(source, text, padding),
        original.convolve_node(source, text, padding),
    )
    equal(source, saved)


@pytest.mark.parametrize("width", list(range(1, 18)))
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_convolve_vector_tails(width, channels):
    source = image(channels, 5, width)
    text = "0.1243 0.9732 -0.562\n-0.231 0.154 0.232\n0.18 0.625 -0.372"
    actual, original = MODULES["convolve"]
    equal(
        actual.convolve_node(source, text, 0), original.convolve_node(source, text, 0)
    )


@pytest.mark.parametrize(
    "layout", ["strided", "transposed", "readonly", "unaligned", "singleton"]
)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_convolve_foreign(layout, channels):
    source = image(channels, layout=layout)
    saved = source.copy()
    actual, original = MODULES["convolve"]
    text = "0.25 -0.125 0.2\n-0.5 0.8 0.0\n0.1 0.4 0.2"
    equal(
        actual.convolve_node(source, text, 2), original.convolve_node(source, text, 2)
    )
    equal(source, saved)


@pytest.mark.parametrize("layout", ["reversed", "transposed", "readonly", "unaligned"])
def test_convolve_foreign_kernel(layout):
    kernel = np.random.default_rng(853).random((3, 4)) - 0.5
    if layout == "reversed":
        kernel = kernel[::-1, ::-1]
    elif layout == "transposed":
        kernel = kernel.T
    elif layout == "readonly":
        kernel.flags.writeable = False
    else:
        raw = np.ndarray(
            kernel.shape, np.float64, buffer=bytearray(kernel.nbytes + 1), offset=1
        )
        raw[:] = kernel
        kernel = raw
    source = image()
    before = kernel.copy()
    equal(native.convolve(source, kernel, 0), cv2.filter2D(source, -1, kernel))
    equal(kernel, before)


@pytest.mark.parametrize(
    "pattern",
    [
        "zero",
        "sparse",
        "derivative",
        "zero_sign",
        "subnormal",
        "overflow",
        "nan",
        "inf",
    ],
)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_convolve_numeric_boundaries(pattern, channels):
    source = image(channels, 13, 17)
    kernel = np.array([[0.25, 0.125, 0.2], [-0.5, 0.8, 0], [0.1, 0.4, 0.2]], np.float64)
    if pattern == "zero":
        kernel.fill(0)
    elif pattern == "sparse":
        kernel[[0, 2], :] = 0
    elif pattern == "derivative":
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
    elif pattern == "zero_sign":
        source[source < 0.5] = -0.0
        source[source >= 0.5] = 0.0
        kernel *= -1
    elif pattern == "subnormal":
        source *= np.float32(1e-38)
    elif pattern == "overflow":
        kernel[0, 0] = 1e300
    elif pattern == "nan":
        source.reshape(-1)[10] = np.nan
    else:
        kernel[0, 0] = np.inf
    actual, original = MODULES["convolve"]
    text = kernel_text(kernel)
    a, b = (
        actual.convolve_node(source, text, 0),
        original.convolve_node(source, text, 0),
    )
    equal(a, b)
    if pattern == "zero_sign":
        equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.parametrize("enabled", [True, False])
def test_convolve_opencv_optimization_contract(enabled):
    before = cv2.useOptimized()
    source = image()
    kernel = np.random.default_rng(888).random((3, 3)) - 0.5
    try:
        cv2.setUseOptimized(enabled)
        equal(native.convolve(source, kernel, 0), cv2.filter2D(source, -1, kernel))
        assert cv2.useOptimized() == enabled
    finally:
        cv2.setUseOptimized(before)


@pytest.mark.parametrize("shape", [(), (7,), (0, 4), (4, 0), (2, 3, 0), (2, 3, 4, 1)])
def test_reject_malformed_images(shape):
    source = np.zeros(shape, np.float32)
    with pytest.raises(ValueError):
        native.box_blur(source, 1, 1)
    with pytest.raises(ValueError):
        native.convolve(source, np.ones((3, 3), np.float32), 0)


@pytest.mark.parametrize("dtype", [np.float64, np.uint8, np.dtype(">f4")])
def test_reject_image_dtype(dtype):
    source = np.zeros((3, 4), dtype)
    with pytest.raises(TypeError):
        native.box_blur(source, 1, 1)
    with pytest.raises(TypeError):
        native.convolve(source, np.ones((3, 3), np.float32), 0)


def test_checked_c_boundaries():
    library = lib()
    p = ct.pointer(ct.c_float())
    maximum = ct.c_size_t(-1).value
    assert library.cn_convolution_box(None, p, 1, 1, 1, 1, 1) == 1
    assert library.cn_convolution_box(p, p, 1, 1, 1, 1001, 1) == 1
    assert library.cn_convolution_box(p, p, maximum, 1, 1, 1, 1) == 1
    assert library.cn_convolution_spatial(p, p, 1, 1, 1, p, maximum, 1, 0, 8) == 1
    assert library.cn_convolution_spatial(p, p, 1, 1, 1, p, 1, 1, maximum, 8) == 2
    assert library.cn_convolution_spatial(p, p, 1, 1, 1, p, 1, 1, 0, 16) == 1


def test_concurrent_requests_and_parallel_work():
    sources = [image(3, 173, 193, seed=seed) for seed in range(4)]
    saved = [source.copy() for source in sources]
    actual_box, original_box = MODULES["box_blur"]
    actual_conv, original_conv = MODULES["convolve"]
    text = "0.12 0.1243 0.223\n0.06 0.09 0.132\n0.042 0.093 0.034"
    expected = [
        (
            original_box.box_blur_node(source, 3, 7),
            original_conv.convolve_node(source, text, 1),
        )
        for source in sources
    ]

    def run(index):
        source = sources[index % len(sources)]
        return (
            actual_box.box_blur_node(source, 3, 7),
            RAW_BOX.box_blur_node(source, 3, 7),
            actual_conv.convolve_node(source, text, 1),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run, range(16)))
    for index, (box, raw_box, conv) in enumerate(results):
        wanted_box, wanted_conv = expected[index % len(sources)]
        assert_enforced_equal(box, wanted_box)
        exact(raw_box, wanted_box)
        equal(conv, wanted_conv)
        for actual, reference in ((raw_box, wanted_box), (conv, wanted_conv)):
            # Saved PNG precision must not hide a floating-point discrepancy.
            for levels in (255, 65535):
                equal(
                    np.rint(np.clip(actual, 0, 1) * levels),
                    np.rint(np.clip(reference, 0, 1) * levels),
                )
    for source, before in zip(sources, saved, strict=True):
        equal(source, before)
