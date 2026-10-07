"""Exact Surface Blur C dispatch, retained IPP evidence, and checked buffer ABI."""

from __future__ import annotations

import ast
import ctypes as ct
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_adjustment_complete import exact, layout, load_body

from nodes.impl import native_bilateral as native

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).with_name("reference_stage2_blur")
NODE_PATH = (
    ROOT / "backend/src/packages/chaiNNer_standard/image_filter/blur/surface_blur.py"
)
NODE = load_body(NODE_PATH)
ORIGINAL = load_body(REFERENCE / "surface_blur.py")


@pytest.fixture
def exposed():
    optimized, ipp = cv2.useOptimized(), cv2.ipp.useIPP()
    cv2.ipp.setUseIPP(False)
    try:
        yield
    finally:
        cv2.setUseOptimized(optimized)
        cv2.ipp.setUseIPP(ipp)


@pytest.mark.usefixtures("exposed")
@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("width", [1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33])
@pytest.mark.parametrize("channels", [0, 1, 3, 4])
@pytest.mark.parametrize(
    "parameters", [(1, 1, 1), (2, 25, 25), (4, 255, 1000), (7, 1000, 3)]
)
def test_exact_exposed_dispatch(width, channels, parameters, optimized):
    cv2.setUseOptimized(optimized)
    cv2.ipp.setUseIPP(False)
    shape = (7, width, channels) if channels else (7, width)
    image = np.random.default_rng(614).uniform(-0.5, 1.5, shape).astype(np.float32)
    before = image.copy()
    exact(
        NODE.surface_blur_node(image, *parameters),
        ORIGINAL.surface_blur_node(image, *parameters),
    )
    exact(image, before)


@pytest.mark.usefixtures("exposed")
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (3, 5)])
@pytest.mark.parametrize("channels", [0, 3, 4])
@pytest.mark.parametrize("radius", [16, 100])
def test_large_radius_thin_images(shape, channels, radius):
    shape = (*shape, channels) if channels else shape
    image = np.random.default_rng(44).random(shape, dtype=np.float32)
    exact(
        NODE.surface_blur_node(image, radius, 25, 25),
        ORIGINAL.surface_blur_node(image, radius, 25, 25),
    )


@pytest.mark.usefixtures("exposed")
@pytest.mark.parametrize("width", [1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33])
@pytest.mark.parametrize("channels", [0, 3, 4])
@pytest.mark.parametrize(
    "pattern", ["nan", "all_nan", "zero", "near_constant", "subnormal"]
)
def test_nan_zero_and_small_ranges(width, channels, pattern):
    shape = (7, width, channels) if channels else (7, width)
    image = np.random.default_rng(616).random(shape, dtype=np.float32)
    if pattern == "nan":
        image[2:4, :2] = np.nan
    elif pattern == "all_nan":
        image[:] = np.nan
    elif pattern == "zero":
        image = np.where(image > 0.5, np.float32(-0.0), np.float32(0.0))
    elif pattern == "near_constant":
        image = np.float32(0.5) + image * np.float32(1e-6)
    else:
        image *= np.float32(1e-40)
    exact(
        NODE.surface_blur_node(image, 4, 25, 25),
        ORIGINAL.surface_blur_node(image, 4, 25, 25),
    )


@pytest.mark.usefixtures("exposed")
@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("width", [1, 3, 8, 9, 17, 33])
@pytest.mark.parametrize("channels", [0, 3])
@pytest.mark.parametrize("pattern", ["block", "column"])
def test_nan_range_of_strided_rows(width, channels, pattern, optimized):
    # OpenCV's minMaxIdx scans a Mat whose rows are apart one row at a time,
    # which changes the range wherever NaN meets its vector lanes.
    cv2.setUseOptimized(optimized)
    cv2.ipp.setUseIPP(False)
    shape = (7, width, channels) if channels else (7, width)
    image = np.random.default_rng(616).random(shape, dtype=np.float32)
    if pattern == "block":
        image[2:4, :2] = np.nan
    else:
        image[1:, 0] = np.nan
    image = layout(image, "rows")
    exact(
        NODE.surface_blur_node(image, 4, 25, 25),
        ORIGINAL.surface_blur_node(image, 4, 25, 25),
    )


@pytest.mark.parametrize("ipp", [False, True])
@pytest.mark.parametrize("channels", [0, 3, 4])
@pytest.mark.parametrize(
    "kind", ["reverse", "fortran", "rows", "columns", "unaligned", "readonly"]
)
def test_foreign_buffers(ipp, channels, kind):
    previous = cv2.ipp.useIPP()
    cv2.ipp.setUseIPP(ipp)
    try:
        shape = (7, 13, channels) if channels else (7, 13)
        image = layout(np.random.default_rng(201).random(shape, dtype=np.float32), kind)
        before = image.copy()
        exact(
            NODE.surface_blur_node(image, 4, 25, 25),
            ORIGINAL.surface_blur_node(image, 4, 25, 25),
        )
        exact(image, before)
    finally:
        cv2.ipp.setUseIPP(previous)


def test_default_ipp_saved_pixel_dependency():
    previous = cv2.ipp.useIPP()
    try:
        image = np.float32(0.5) + (
            np.random.default_rng(14).random((13, 19), dtype=np.float32)
            - np.float32(0.5)
        ) * np.float32(1e-6)
        cv2.ipp.setUseIPP(True)
        if not cv2.ipp.useIPP():
            pytest.skip("IPP is unavailable on this host")
        expected = ORIGINAL.surface_blur_node(image, 4, 25, 25)
        exact(NODE.surface_blur_node(image, 4, 25, 25), expected)
        cv2.ipp.setUseIPP(False)
        alternative = ORIGINAL.surface_blur_node(image, 4, 25, 25)
        assert np.any(np.rint(expected * 255) != np.rint(alternative * 255))
        exact(NODE.surface_blur_node(image, 4, 25, 25), alternative)
    finally:
        cv2.ipp.setUseIPP(previous)


@pytest.mark.usefixtures("exposed")
def test_concurrency_without_bilateral_runtime(monkeypatch):
    image = np.random.default_rng(777).random((71, 83, 4), dtype=np.float32)
    expected = ORIGINAL.surface_blur_node(image, 4, 25, 25)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Exposed float32 bilateral arithmetic must execute in C")

    def run(_):
        # OpenCV's IPP setting is thread-local; configure the worker exactly as
        # the reference caller instead of inheriting the worker's default IPP.
        previous = cv2.ipp.useIPP()
        cv2.ipp.setUseIPP(False)
        try:
            return NODE.surface_blur_node(image, 4, 25, 25)
        finally:
            cv2.ipp.setUseIPP(previous)

    monkeypatch.setattr(cv2, "bilateralFilter", forbidden)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for actual in pool.map(run, range(8)):
            exact(actual, expected)


@pytest.mark.parametrize("parameters", [(0, 25, 25), (4, 0, 25), (4, 25, 0)])
def test_noop_alias(parameters):
    image = np.zeros((3, 7, 4), np.float32)
    assert NODE.surface_blur_node(image, *parameters) is image


@pytest.mark.usefixtures("exposed")
@pytest.mark.parametrize("sigmas", [(1e-6, 25.0), (0.1, 1e-6)])
def test_tiny_sigma_copies_like_opencv(sigmas):
    image = np.random.default_rng(3).random((7, 9, 3), dtype=np.float32)
    expected = cv2.bilateralFilter(image, 9, *sigmas, borderType=cv2.BORDER_REFLECT_101)
    exact(expected, image)
    exact(native.bilateral(image, 4, *sigmas), expected)


def test_schema_and_signature_unchanged():
    functions = []
    for path in (REFERENCE / "surface_blur.py", NODE_PATH):
        tree = ast.parse(path.read_text("utf-8"))
        functions.append(
            next(item for item in tree.body if isinstance(item, ast.FunctionDef))
        )
    assert ast.dump(functions[0].args) == ast.dump(functions[1].args)
    assert [ast.dump(item) for item in functions[0].decorator_list] == [
        ast.dump(item) for item in functions[1].decorator_list
    ]


@pytest.mark.parametrize(
    "index,value",
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 2),
        (5, 0),
        (5, 101),
        (6, 0),
        (6, float("nan")),
        (7, float("inf")),
        (8, 2),
        (9, 16),
        (10, 2),
    ],
)
def test_invalid_abi(index, value):
    image = np.ones((7, 9), np.float32)
    out = np.full_like(image, 123)
    args = [image.ctypes.data, out.ctypes.data, 7, 9, 1, 4, 25 / 255, 25, 16, 8, 0]
    args[index] = value
    assert native._api().cn_bilateral_f32(*args) == 1
    assert np.all(out == 123)


def test_abi_alignment_overlap_and_address_overflow():
    image = np.ones((7, 9), np.float32)
    out = np.full_like(image, 123)
    args = [image.ctypes.data, out.ctypes.data, 7, 9, 1, 4, 25 / 255, 25, 16, 8, 0]
    args[0] += 1
    assert native._api().cn_bilateral_f32(*args) == 1
    args[0] = out.ctypes.data + 4
    assert native._api().cn_bilateral_f32(*args) == 1
    args[0] = ct.c_size_t(-4).value
    assert native._api().cn_bilateral_f32(*args) == 2
    args[0] = image.ctypes.data
    args[2:5] = [2**31 - 1, 2**31 - 1, 3]
    assert native._api().cn_bilateral_f32(*args) == 2
    assert np.all(out == 123)


@pytest.mark.usefixtures("exposed")
@pytest.mark.parametrize(
    "kind", ["positive_infinity", "negative_infinity", "huge_range"]
)
def test_invalid_original_lut_range_preserves_exception_without_invalid_memory(kind):
    image = np.linspace(0, 1, 63, dtype=np.float32).reshape(7, 9)
    if kind == "positive_infinity":
        image[1, 2] = np.inf
    elif kind == "negative_infinity":
        image[1, 2] = -np.inf
    else:
        image[1, 2] = np.float32(3.4e38)
        image[3, 4] = np.float32(-3.4e38)
    # The pinned build translates its invalid gather into a cv2.error. Run that
    # original in an isolated process: SEH still trips pytest's fault handler.
    # The C implementation validates before allocation or output writes.
    probe = """
import ast,json,sys
from pathlib import Path
import cv2,numpy as np
tree=ast.parse(Path(sys.argv[2]).read_text('utf-8'))
tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
for node in tree.body:node.decorator_list=[]
scope={'np':np,'cv2':cv2,'get_h_w_c':lambda a:(*a.shape,1)}
exec(compile(tree,sys.argv[2],'exec'),scope)
image=np.linspace(0,1,63,dtype=np.float32).reshape(7,9)
if sys.argv[1]=='positive_infinity':image[1,2]=np.inf
elif sys.argv[1]=='negative_infinity':image[1,2]=-np.inf
else:image[1,2]=np.float32(3.4e38);image[3,4]=np.float32(-3.4e38)
cv2.ipp.setUseIPP(False)
try:scope['surface_blur_node'](image,4,25,25)
except cv2.error as error:print(json.dumps({'type':'cv2.error','message':str(error)}))
else:raise AssertionError('Original invalid LUT range unexpectedly succeeded')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe, kind, str(REFERENCE / "surface_blur.py")],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    original_error = json.loads(result.stdout)
    assert original_error == {
        "type": "cv2.error",
        "message": "Unknown C++ exception from OpenCV code",
    }
    with pytest.raises(cv2.error, match="Unknown C\\+\\+ exception"):
        NODE.surface_blur_node(image, 4, 25, 25)
    out = np.full_like(image, 123)
    assert (
        native._api().cn_bilateral_f32(
            image.ctypes.data, out.ctypes.data, 7, 9, 1, 4, 25 / 255, 25, 16, 8, 0
        )
        == 4
    )
    assert np.all(out == 123)


@pytest.mark.parametrize("channels", [0, 3, 4])
def test_general_uint8_helper_contract(channels):
    shape = (7, 13, channels) if channels else (7, 13)
    image = np.random.default_rng(656).integers(0, 256, shape, dtype=np.uint8)
    np.testing.assert_array_equal(
        NODE.surface_blur_node(image, 4, 25, 25),
        ORIGINAL.surface_blur_node(image, 4, 25, 25),
    )


@pytest.mark.parametrize("channels", [2, 5])
def test_unsupported_channel_exception(channels):
    image = np.ones((7, 13, channels), np.float32)
    with pytest.raises(cv2.error) as original:
        ORIGINAL.surface_blur_node(image, 4, 25, 25)
    with pytest.raises(cv2.error) as actual:
        NODE.surface_blur_node(image, 4, 25, 25)
    assert actual.value.code == original.value.code
