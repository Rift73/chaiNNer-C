"""Stage-two whole-node Gaussian checks against the untouched installed source.

The node is compared as the user receives its output: ImageOutput.enforce of both
nodes' returns, bitwise (SP4b Task 3b2; owner, 2026-10-04: a raw pre-enforce return
is an internal detail). The blur's kernel parity stays exact at the helper:
fast_gaussian_blur's raw result (normalized=False, its default) against the
installed one. The schema guard compares the register(...) call and the signature.
"""

from __future__ import annotations

import ast
import math
import operator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_adjustment_complete import exact, layout, load_body
from test_normalized_outputs import assert_enforced_equal, node_schema

from nodes.utils.utils import get_h_w_c

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).with_name("reference_stage2_blur")
NODE_PATH = (
    ROOT / "backend/src/packages/chaiNNer_standard/image_filter/blur/gaussian_blur.py"
)
NODE = load_body(NODE_PATH)
ORIGINAL = load_body(REFERENCE / "gaussian_blur.py")
_path = REFERENCE / "image_utils.py"
_tree = ast.parse(_path.read_text("utf-8"))
_tree.body = [
    item
    for item in _tree.body
    if isinstance(item, ast.FunctionDef) and item.name == "fast_gaussian_blur"
]
_globals = {"cv2": cv2, "np": np, "math": math, "get_h_w_c": get_h_w_c}
exec(compile(_tree, str(_path), "exec"), _globals)
vars(ORIGINAL).update(fast_gaussian_blur=_globals["fast_gaussian_blur"])

SIGMAS = [
    (0, 0),
    (0, 1000),
    (1000, 0),
    (0.1, 0.1),
    (0.1, 7),
    (1, 1),
    (2.3, 4.7),
    (10.9, 11),
    (11, 11),
    (14.9, 15),
    (15, 15),
    (19.9, 20),
    (20, 20),
    (24.9, 25),
    (25, 25),
    (29.9, 30),
    (30, 30),
    (49.9, 50),
    (50, 50),
    (99.9, 100),
    (100, 100),
    (199.9, 200),
    (200, 200),
    (1000, 1000),
]


def assert_node_parity(image, sigmas):
    """The node against the installed one as the user receives it (ImageOutput.enforce
    of both returns), and fast_gaussian_blur's raw result (normalized=False, its
    default) against the installed fast_gaussian_blur's, both bitwise."""
    assert_enforced_equal(
        NODE.gaussian_blur_node(image, *sigmas),
        ORIGINAL.gaussian_blur_node(image, *sigmas),
    )
    exact(
        NODE.fast_gaussian_blur(image, *sigmas),
        ORIGINAL.fast_gaussian_blur(image, *sigmas),
    )


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (13, 17)])
@pytest.mark.parametrize("channels", [0, 1, 3, 4, 7])
@pytest.mark.parametrize("sigmas", SIGMAS)
def test_all_scale_boundaries(shape, channels, sigmas):
    shape = (*shape, channels) if channels else shape
    image = np.random.default_rng(342).uniform(-0.5, 1.5, shape).astype(np.float32)
    before = image.copy()
    try:
        ORIGINAL.gaussian_blur_node(image, *sigmas)
    except cv2.error as original_error:
        # OpenCV AREA downsampling rejects >4 channels in its fractional mode.
        # These accepted image descriptors must retain the original exception.
        with pytest.raises(cv2.error) as raised:
            NODE.gaussian_blur_node(image, *sigmas)
        # cv2.error's code and func, which OpenCV's type stubs do not declare.
        site = operator.attrgetter("code", "func")
        assert site(raised.value) == site(original_error)
    else:
        assert_node_parity(image, sigmas)
    exact(image, before)


@pytest.mark.parametrize(
    "kind", ["reverse", "fortran", "rows", "columns", "unaligned", "readonly"]
)
@pytest.mark.parametrize("sigmas", [(0, 0), (0, 5), (1.5, 2.3), (11, 20), (1000, 1000)])
def test_foreign_buffers(kind, sigmas):
    image = layout(
        np.random.default_rng(102).random((7, 13, 4), dtype=np.float32), kind
    )
    before = image.copy()
    assert_node_parity(image, sigmas)
    exact(image, before)


@pytest.mark.parametrize("sigmas", [(0, 0), (0, 5), (1.5, 2.3), (11, 20), (200, 250)])
@pytest.mark.parametrize(
    "pattern", ["negative_zero", "nan", "infinity", "subnormal", "extreme"]
)
def test_exceptional_values(sigmas, pattern):
    image = np.random.default_rng(404).random((7, 13, 3), dtype=np.float32)
    if pattern == "negative_zero":
        image[:] = -0.0
    elif pattern == "nan":
        image[2, 3] = np.nan
    elif pattern == "infinity":
        image[2, 3] = np.inf
        image[3, 5] = -np.inf
    elif pattern == "subnormal":
        image *= np.float32(1e-40)
    else:
        image *= np.float32(3e38)
    with np.errstate(all="ignore"):
        assert_node_parity(image, sigmas)


def test_noop_alias_and_schema_unchanged():
    image = np.zeros((3, 7, 4), np.float32)
    assert NODE.gaussian_blur_node(image, 0, 0) is image
    assert NODE.gaussian_blur_node(image, 0, 1) is not image
    assert node_schema(NODE_PATH) == node_schema(REFERENCE / "gaussian_blur.py")


def test_concurrent_calls_and_no_gaussian_runtime(monkeypatch):
    image = np.random.default_rng(199).random((63, 71, 4), dtype=np.float32)
    sigmas = [(1, 2), (10, 9), (11, 17), (51, 71)] * 2
    expected = [
        (
            ORIGINAL.gaussian_blur_node(image, *pair),
            ORIGINAL.fast_gaussian_blur(image, *pair),
        )
        for pair in sigmas
    ]

    def forbidden(*_args, **_kwargs):
        pytest.fail("float32 Gaussian filtering must execute the C implementation")

    def run(pair):
        return (
            NODE.gaussian_blur_node(image, *pair),
            NODE.fast_gaussian_blur(image, *pair),
        )

    monkeypatch.setattr(cv2, "GaussianBlur", forbidden)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, sigmas))
    for (node, raw), (wanted_node, wanted_raw) in zip(results, expected, strict=True):
        assert_enforced_equal(node, wanted_node)
        exact(raw, wanted_raw)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("ipp", [False, True])
@pytest.mark.parametrize("sigmas", [(1, 2.3), (11, 17), (25, 31), (1000, 1000)])
def test_node_cpu_dispatch_settings(optimized, ipp, sigmas):
    before = cv2.useOptimized(), cv2.ipp.useIPP()
    try:
        cv2.setUseOptimized(optimized)
        cv2.ipp.setUseIPP(ipp)
        image = np.random.default_rng(496).random((17, 23, 4), dtype=np.float32)
        assert_node_parity(image, sigmas)
    finally:
        cv2.setUseOptimized(before[0])
        cv2.ipp.setUseIPP(before[1])
