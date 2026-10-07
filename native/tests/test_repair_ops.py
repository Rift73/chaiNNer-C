"""Exact CPU oracles from OpenCV 5.0; no models, GPU, or timing benchmarks."""

from __future__ import annotations

import ast
import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from nodes.impl import native_repair as native
from nodes.impl.image_utils import to_uint8
from nodes.impl.native_versions import CV_CN_MAX


def image(shape, layout="plain", seed=791):
    source = np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)
    if layout == "readonly":
        source.flags.writeable = False
    elif layout == "strided":
        source = source[::-1, ::2]
    elif layout == "fortran":
        source = np.asfortranarray(source)
    return source


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 17), (13, 1), (2, 3), (3, 3), (9, 13), (31, 37), (129, 131)]
)
@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5, 17])
@pytest.mark.parametrize(
    "thresholds",
    [(0, 0), (100, 300), (300, 100), (25.9, 84.1), (1020, 2040), (1e100, 1e100)],
)
def test_canny_exact(shape, channels, thresholds):
    source = image((*shape, channels) if channels > 1 else shape)
    expected = cv2.Canny(source, *thresholds)
    np.testing.assert_array_equal(native.canny(source, *thresholds), expected)


@pytest.mark.parametrize("method", [cv2.INPAINT_NS, cv2.INPAINT_TELEA])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("shape", [(2, 2), (2, 7), (7, 2), (9, 13), (27, 31)])
@pytest.mark.parametrize(
    "radius",
    [0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 7.5, 100.0, 101.0, float("inf"), float("nan")],
)
@pytest.mark.parametrize("mask_kind", ["empty", "full", "random", "center", "border"])
def test_inpaint_exact(method, channels, shape, radius, mask_kind):
    source = image((*shape, channels) if channels > 1 else shape)
    mask = np.zeros(shape, np.uint8)
    if mask_kind == "full":
        mask[:] = 255
    elif mask_kind == "random":
        mask = np.where(image(shape, seed=182) < 60, 255, 0).astype(np.uint8)
    elif mask_kind == "center":
        mask[shape[0] // 2, shape[1] // 2] = 1
    elif mask_kind == "border":
        mask[0] = 255
        mask[:, -1] = 255
    expected = cv2.inpaint(source, mask, radius, method)
    actual = native.inpaint(source, mask, radius, method)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("layout", ["plain", "strided", "readonly", "fortran"])
@pytest.mark.parametrize("channels", [1, 3])
def test_foreign_layout_immutability(layout, channels):
    source = image((17, 19, channels) if channels > 1 else (17, 19), layout)
    mask = np.where(image(source.shape[:2], seed=181) < 80, 1, 0).astype(np.uint8)
    if layout == "readonly":
        mask.flags.writeable = False
    before = (source.tobytes(), mask.tobytes())
    for method in (0, 1):
        np.testing.assert_array_equal(
            native.inpaint(source, mask, 3, method),
            cv2.inpaint(source, mask, 3, method),
        )
    np.testing.assert_array_equal(
        native.canny(source, 83, 173), cv2.Canny(source, 83, 173)
    )
    assert before == (source.tobytes(), mask.tobytes())


def test_concurrent_repeatability():
    source = image((43, 47, 3))
    mask = np.where(image((43, 47), seed=871) < 70, 255, 0).astype(np.uint8)
    expected = [cv2.inpaint(source, mask, 4, method) for method in (0, 1)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        output = list(
            pool.map(lambda i: native.inpaint(source, mask, 4, i % 2), range(24))
        )
    for i, result in enumerate(output):
        np.testing.assert_array_equal(result, expected[i % 2])


def test_checked_abi_and_python_shape_rejection():
    dll = native._api()
    a = np.zeros((3, 4, 3), np.uint8)
    mask = np.zeros((3, 4), np.uint8)
    out = np.full_like(a, 93)
    assert dll.cn_canny_u8(None, out.ctypes.data, 3, 4, 3, 0.0, 1.0) == 1
    assert dll.cn_canny_u8(a.ctypes.data, a.ctypes.data, 3, 4, 3, 0.0, 1.0) == 1
    assert (
        dll.cn_canny_u8(a.ctypes.data, out.ctypes.data, 3, 4, CV_CN_MAX + 1, 0.0, 1.0)
        == 1
    )
    assert (
        dll.cn_canny_u8(a.ctypes.data, out.ctypes.data, ct.c_size_t(-1), 4, 3, 0.0, 1.0)
        == 2
    )
    assert (
        dll.cn_inpaint_u8(
            a.ctypes.data, mask.ctypes.data, out.ctypes.data, 3, 4, 2, 2.0, 0
        )
        == 1
    )
    assert (
        dll.cn_inpaint_u8(
            a.ctypes.data, mask.ctypes.data, a.ctypes.data, 3, 4, 3, 2.0, 0
        )
        == 1
    )
    assert (
        dll.cn_inpaint_u8(
            a.ctypes.data, mask.ctypes.data, out.ctypes.data, 3, 4, 3, 2.0, 9
        )
        == 1
    )
    np.testing.assert_array_equal(out, 93)
    with pytest.raises(ValueError):
        native.inpaint(a, mask[:2], 3.0, 0)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).with_name("reference_repair")


def function(path, name):
    tree = ast.parse(path.read_text("utf-8"))
    definition = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    definition.decorator_list = []
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        definition,
    ]
    context: dict[str, Any] = {
        "np": np,
        "cv2": cv2,
        "to_uint8": to_uint8,
        "canny": native.canny,
        "inpaint": native.inpaint,
    }
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), context)
    return context[name]


@pytest.mark.parametrize("channels", [1, 3])
def test_nodes_and_metadata(channels):
    paths = [
        (
            "image_filter/miscellaneous/canny_edge_detection.py",
            "canny_edge_detection_node",
        ),
        ("image_utility/miscellaneous/inpaint.py", "inpaint_node"),
    ]
    img = (
        image((19, 23, channels) if channels > 1 else (19, 23)).astype(np.float32) / 255
    )
    mask = (image((19, 23), seed=561) < 100).astype(np.float32)
    for relative, name in paths:
        current = ROOT / "backend/src/packages/chaiNNer_standard" / relative
        original = REFERENCE / Path(relative).name
        for path in (current, original):
            definition = next(
                n
                for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            contract = (
                ast.dump(definition.args),
                [ast.dump(d) for d in definition.decorator_list],
            )
            if path == current:
                actual_contract = contract
            else:
                assert contract == actual_contract
        args = (
            (img, 100, 300)
            if name.startswith("canny")
            else (img, mask, SimpleNamespace(value=1), 3.0)
        )
        np.testing.assert_array_equal(
            function(current, name)(*args), function(original, name)(*args)
        )


@pytest.mark.parametrize("method", [0, 1])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize(
    "radius",
    [
        *list(range(101)),
        np.nextafter(1.5, 0),
        np.nextafter(1.5, 2),
        -1.0,
        -np.inf,
        2**31,
    ],
)
def test_inpaint_radius_rounding_domain(method, channels, radius):
    source = image((7, 9, channels) if channels == 3 else (7, 9))
    mask = np.zeros((7, 9), np.uint8)
    mask[2:5, 3:6] = 255
    np.testing.assert_array_equal(
        native.inpaint(source, mask, radius, method),
        cv2.inpaint(source, mask, radius, method),
    )


@pytest.mark.parametrize("channels", [1, 3, 4, CV_CN_MAX])
@pytest.mark.parametrize(
    "threshold",
    [
        -np.inf,
        -1e100,
        -2147483649.0,
        -2147483648.0,
        -0.1,
        0.0,
        1.0,
        1020.0,
        2147483647.0,
        2147483648.0,
        np.inf,
        np.nan,
    ],
)
def test_canny_threshold_conversion(channels, threshold):
    source = image((11, 13, channels) if channels > 1 else (11, 13))
    np.testing.assert_array_equal(
        native.canny(source, threshold, threshold),
        cv2.Canny(source, threshold, threshold),
    )


@pytest.mark.parametrize("channels", [CV_CN_MAX + 1, 512])
@pytest.mark.parametrize("shape", [(0, 13), (11, 13)])
def test_canny_channel_ceiling(channels, shape):
    source = image((*shape, channels))
    with pytest.raises(cv2.error) as expected:
        cv2.Canny(source, 1.0, 1.0)
    with pytest.raises(cv2.error) as actual:
        native.canny(source, 1.0, 1.0)
    assert str(actual.value) == str(expected.value)


def canny_outcome(canny, source):
    """Canny's result, or the cv2.error message it raises."""
    try:
        return canny(source, 1.0, 1.0)
    except cv2.error as error:
        return f"cv2.error: {error}"


@pytest.mark.parametrize(
    ("shape", "raises"),
    [((0, 13), False), ((0, 0), False), ((11, 0), True), ((11, 13, 0), True)],
)
def test_canny_empty_images_defer_to_opencv(shape, raises):
    # OpenCV never accepts an empty image: it returns None for one without rows and
    # raises for one without columns or channels. The bridge returns the same.
    source = image(shape)
    expected = canny_outcome(cv2.Canny, source)
    if raises:
        assert isinstance(expected, str)
        assert expected.startswith("cv2.error: ")
    else:
        assert expected is None
    assert canny_outcome(native.canny, source) == expected


@pytest.mark.parametrize("method", [0, 1])
@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (1, 7), (7, 1)])
@pytest.mark.parametrize("channels", [1, 3])
def test_tiny_inpaint_corrected_bounds_repeatability(method, shape, channels):
    # Baseline reads outside its allocation for these dimensions. Test the
    # corrected invariants, not arbitrary bytes from an allocator-dependent run.
    source = image((*shape, channels) if channels == 3 else shape)
    mask = np.zeros(shape, np.uint8)
    mask.reshape(-1)[np.prod(shape) // 2] = 255
    expected = native.inpaint(source, mask, 3, method)
    np.testing.assert_array_equal(expected[mask == 0], source[mask == 0])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: native.inpaint(source, mask, 3, method), range(16))
        )
    for result in results:
        np.testing.assert_array_equal(result, expected)
    if method == 0 and np.prod(shape) > 1:
        constant = np.full_like(source, 71)
        np.testing.assert_array_equal(
            native.inpaint(constant, mask, 3, method), constant
        )


@pytest.mark.parametrize("kind", ["constant", "ramp", "checker", "tie_channels"])
def test_canny_exact_suppression_ties(kind):
    y, x = np.indices((25, 37))
    if kind == "constant":
        source = np.full((25, 37), 127, np.uint8)
    elif kind == "ramp":
        source = (x * 7).astype(np.uint8)
    elif kind == "checker":
        source = ((x + y) % 2 * 255).astype(np.uint8)
    else:
        source = np.stack([(x % 2 * 255), (y % 2 * 255), (x % 2 * 255)], axis=2).astype(
            np.uint8
        )
    for thresholds in [
        (0, 0),
        (1, 1),
        (255, 255),
        (509, 510),
        (510, 511),
        (1019, 1020),
    ]:
        np.testing.assert_array_equal(
            native.canny(source, *thresholds), cv2.Canny(source, *thresholds)
        )


@pytest.mark.parametrize("shape", [(8, 11, 1), (8, 11, 3)])
def test_mask_nonbinary_and_gray_singleton(shape):
    source = image(shape)
    mask = image(shape[:2], seed=173)
    for method in (0, 1):
        expected = cv2.inpaint(source, mask, 3, method)
        actual = native.inpaint(source, mask[..., None], 3, method)
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)


def test_canny_parallel_and_repeated():
    source = image((257, 259, 4))
    expected = cv2.Canny(source, 71, 191)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: native.canny(source, 71, 191), range(12)))
    for result in results:
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("threads,ipp", [(1, True), (2, True), (1, False)])
@pytest.mark.parametrize(
    "thresholds",
    [
        (399.99999, 399.99999),
        (400.0, 400.0),
        (0.0, 1e100),
        (0.0, np.nan),
        (np.nan, 0.0),
        (-1.0, 1e100),
        (-1e-100, 1e100),
    ],
)
def test_canny_original_configuration_threshold_policy(threads, ipp, thresholds):
    old_threads, old_ipp = cv2.getNumThreads(), cv2.ipp.useIPP()
    source = np.zeros((9, 13), np.uint8)
    source[:, 7:] = 100
    try:
        cv2.setNumThreads(threads)
        cv2.ipp.setUseIPP(ipp)
        np.testing.assert_array_equal(
            native.canny(source, *thresholds), cv2.Canny(source, *thresholds)
        )
    finally:
        cv2.setNumThreads(old_threads)
        cv2.ipp.setUseIPP(old_ipp)
