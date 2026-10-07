"""Exact OpenCV cubic source path and independently frozen Average Color Fix."""

from __future__ import annotations

import ast
import importlib.util
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from nodes.impl import native_cv_resize

ROOT = Path(__file__).resolve().parents[2]
RELATIVE = "packages/chaiNNer_standard/image_filter/correction/average_color_fix.py"


def load_node(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.ImportFrom)
        or (node.level == 0 and not (node.module or "").startswith("nodes.properties"))
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    module = types.ModuleType(path.stem)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


ACTUAL = load_node(ROOT / "backend/src" / RELATIVE)
ORIGINAL = load_node(Path(__file__).with_name("reference_analysis") / RELATIVE)
spec = importlib.util.spec_from_file_location(
    "nodes.impl.reference_average_resize",
    Path(__file__).with_name("reference_resample") / "resize.py",
)
assert spec is not None and spec.loader is not None
RESIZE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RESIZE)
ORIGINAL.__dict__.update(resize=RESIZE.resize)
ORIGINAL.__dict__.update(ResizeFilter=RESIZE.ResizeFilter)


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[valid].view(np.uint32), expected[valid].view(np.uint32)
    )


@pytest.fixture(params=[False, True])
def ipp(request):
    old = cv2.ipp.useIPP()
    cv2.ipp.setUseIPP(request.param)
    try:
        yield request.param
    finally:
        cv2.ipp.setUseIPP(old)


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (2, 2), (3, 5), (17, 19)])
@pytest.mark.parametrize(
    "size", [(1, 1), (2, 3), (3, 2), (7, 5), (16, 17), (31, 37), (40, 33)]
)
def test_cubic_shapes(channels, shape, size, ipp):
    image = np.random.default_rng(472).random((*shape, channels), dtype=np.float32)
    exact(
        native_cv_resize.resize(image, size, cv2.INTER_CUBIC),
        cv2.resize(image, size, interpolation=cv2.INTER_CUBIC),
    )


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(1, 7), (7, 1), (2, 2), (3, 5), (17, 19)])
@pytest.mark.parametrize("kind", ["signed_zero", "nonfinite", "exponents"])
def test_cubic_special_values(channels, shape, kind, ipp):
    rng = np.random.default_rng(443)
    image = rng.standard_normal((*shape, channels)).astype(np.float32)
    if kind == "signed_zero":
        image = np.copysign(np.zeros_like(image), image)
    elif kind == "nonfinite":
        image.flat[::7] = np.nan
        image.flat[::13] = np.inf
        image.flat[::17] = -np.inf
    else:
        image = np.ldexp(image, rng.integers(-140, 123, image.shape, dtype=np.int32))
    exact(
        native_cv_resize.resize(image, (37, 31), cv2.INTER_CUBIC),
        cv2.resize(image, (37, 31), interpolation=cv2.INTER_CUBIC),
    )


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("ref_channels", [3, 4])
@pytest.mark.parametrize("scale", [0, 0.01, 12.5, 50, 100])
@pytest.mark.parametrize(
    "shape,ref_shape", [((7, 9), (2, 3)), ((31, 37), (7, 9)), ((73, 67), (19, 23))]
)
def test_average_full_node(channels, ref_channels, scale, shape, ref_shape, ipp):
    rng = np.random.default_rng(914)
    image = rng.random((*shape, channels), dtype=np.float32)
    reference = rng.random((*ref_shape, ref_channels), dtype=np.float32)
    if channels == 4:
        image[::2, ::3, 3] = 0
    if ref_channels == 4:
        reference[::2, ::3, 3] = 0
    image.flags.writeable = reference.flags.writeable = False
    exact(
        ACTUAL.average_color_fix_node(image, reference, scale),
        ORIGINAL.average_color_fix_node(image, reference, scale),
    )


@pytest.mark.parametrize("layout", ["strided", "reverse", "transpose", "unaligned"])
@pytest.mark.parametrize("channels,ref_channels", [(3, 3), (3, 4), (4, 3), (4, 4)])
def test_average_foreign_buffers(layout, channels, ref_channels, ipp):
    rng = np.random.default_rng(589)
    image = rng.random((37, 41, channels), dtype=np.float32)
    reference = rng.random((13, 17, ref_channels), dtype=np.float32)
    if layout == "strided":
        image, reference = image[::2, ::2], reference[::2, ::2]
    elif layout == "reverse":
        image, reference = image[::-1, ::-1], reference[::-1, ::-1]
    elif layout == "transpose":
        image, reference = image.swapaxes(0, 1), reference.swapaxes(0, 1)
    else:

        def unaligned(value):
            out = np.ndarray(value.shape, np.float32, bytearray(value.nbytes + 1), 1)
            out[:] = value
            return out

        image, reference = unaligned(image), unaligned(reference)
    saved, saved_ref = image.copy(), reference.copy()
    exact(
        ACTUAL.average_color_fix_node(image, reference, 50),
        ORIGINAL.average_color_fix_node(image, reference, 50),
    )
    exact(image, saved)
    exact(reference, saved_ref)


def test_cubic_saved_png_evidence():
    old = cv2.ipp.useIPP()
    try:
        image = np.full((7, 9, 3), 0.5, np.float32)
        cv2.ipp.setUseIPP(True)
        original = cv2.resize(image, (37, 31), interpolation=cv2.INTER_CUBIC)
        exact(native_cv_resize.resize(image, (37, 31), cv2.INTER_CUBIC), original)
        cv2.ipp.setUseIPP(False)
        baseline = native_cv_resize.resize(image, (37, 31), cv2.INTER_CUBIC)
        first = (original * 255).round().astype(np.uint8)
        second = (baseline * 255).round().astype(np.uint8)
        # OpenCV 5.0.0's IPP cubic and its plain path still differ, now in 1269
        # bytes by at most 3 * 2**-25 (OpenCV 4.8: 1428 bytes, 2**-23); the C
        # mirror equals cv2 with IPP on above and off here.
        assert np.count_nonzero(first != second) == 1269
        assert np.max(np.abs(original - baseline)) == np.float32(3 * 2**-25)
        assert (
            cv2.imencode(".png", first)[1].tobytes()
            != cv2.imencode(".png", second)[1].tobytes()
        )
    finally:
        cv2.ipp.setUseIPP(old)


def test_complete_average_has_no_original_pixel_primitives(monkeypatch):
    rng = np.random.default_rng(526)
    image = rng.random((71, 79, 4), dtype=np.float32)
    reference = rng.random((17, 19, 4), dtype=np.float32)
    old = cv2.ipp.useIPP()
    cv2.ipp.setUseIPP(False)
    try:
        expected = ORIGINAL.average_color_fix_node(image, reference, 12.5)

        def forbidden(*_args, **_kwargs):
            pytest.fail("Average Color Fix used its original array primitive")

        monkeypatch.setattr(cv2, "resize", forbidden)
        monkeypatch.setattr(np, "concatenate", forbidden)
        exact(ACTUAL.average_color_fix_node(image, reference, 12.5), expected)
    finally:
        cv2.ipp.setUseIPP(old)


def test_concurrent_cubic():
    image = np.random.default_rng(616).random((131, 137, 4), dtype=np.float32)
    old = cv2.ipp.useIPP()
    cv2.ipp.setUseIPP(False)
    try:
        expected = cv2.resize(image, (317, 289), interpolation=cv2.INTER_CUBIC)
    finally:
        cv2.ipp.setUseIPP(old)

    def worker(_):
        saved = cv2.ipp.useIPP()
        cv2.ipp.setUseIPP(False)
        try:
            return native_cv_resize.resize(image, (317, 289), cv2.INTER_CUBIC)
        finally:
            cv2.ipp.setUseIPP(saved)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(8)))
    for actual in results:
        exact(actual, expected)


@pytest.mark.parametrize("channels,ref_channels", [(3, 3), (3, 4), (4, 3), (4, 4)])
@pytest.mark.parametrize("kind", ["nonfinite", "signed_zero"])
@pytest.mark.parametrize("scale", [12.5, 100])
def test_average_special_values(channels, ref_channels, kind, scale, ipp):
    rng = np.random.default_rng(629)
    image = rng.random((13, 17, channels), dtype=np.float32)
    reference = rng.random((7, 9, ref_channels), dtype=np.float32)
    if kind == "nonfinite":
        image.flat[::37] = np.nan
        image.flat[::57] = np.inf
        reference.flat[::7] = -np.inf
        reference.flat[::17] = np.nan
    else:
        image = np.copysign(np.zeros_like(image), image - np.float32(0.5))
        reference = np.copysign(np.zeros_like(reference), reference - np.float32(0.5))
    with np.errstate(all="ignore"):
        exact(
            ACTUAL.average_color_fix_node(image, reference, scale),
            ORIGINAL.average_color_fix_node(image, reference, scale),
        )
