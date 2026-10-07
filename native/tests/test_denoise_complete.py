"""Exact OpenCV 4.8 CPU NLM and linear-BGR Lab8 oracle, with frozen node I/O.

The byte conversion oracle is pinned separately from the production image
helpers. No timing, model inference, or GPU workload is performed here.
"""

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

from nodes.impl import native_denoise

TESTS = Path(__file__).parent
ROOT = TESTS.parents[1]
NODE = ROOT / "backend/src/packages/chaiNNer_standard/image_filter/noise/denoise.py"


def load_node(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        item
        for item in tree.body
        if not isinstance(item, ast.ImportFrom)
        or (
            not item.level
            and item.module != "nodes.groups"
            and not (item.module or "").startswith("nodes.properties")
        )
    ]
    for item in tree.body:
        if isinstance(item, ast.FunctionDef):
            item.decorator_list = []
    module = types.ModuleType(path.stem)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


CURRENT = load_node(NODE)
REFERENCE = load_node(TESTS / "reference_denoise.py")
SPEC = importlib.util.spec_from_file_location(
    "nodes.impl.reference_denoise_image", TESTS / "reference_buffers/image_utils.py"
)
assert SPEC is not None and SPEC.loader is not None
ORIGINAL_IMAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORIGINAL_IMAGE)
REFERENCE.__dict__.update(to_uint8=ORIGINAL_IMAGE.to_uint8)


def image(shape, channels, seed=11):
    dimensions = shape if channels == 1 else (*shape, channels)
    return np.random.default_rng(seed).integers(0, 256, dimensions, np.uint8)


def exact(actual, expected):
    assert actual.dtype == expected.dtype == np.uint8
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def baseline_lab(source, inverse):
    code = cv2.COLOR_LAB2LBGR if inverse else cv2.COLOR_LBGR2LAB
    result = cv2.cvtColor(source[..., :3], code)
    if source.shape[2] == 4:
        result = np.dstack((result, source[..., 3]))
    return result


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("shape", [(1, 1), (1, 33), (33, 1), (17, 31), (259, 263)])
def test_lab_exact(inverse, channels, shape):
    source = image(shape, channels)
    exact(native_denoise.lab(source, inverse=inverse), baseline_lab(source, inverse))


@pytest.mark.parametrize("inverse", [False, True])
def test_lab_all_channel_levels(inverse):
    # Every byte in each channel, with all combinations of boundary anchors.
    levels = np.arange(256, dtype=np.uint8)
    axes = np.array([0, 1, 15, 16, 20, 21, 127, 128, 129, 239, 254, 255], np.uint8)
    arrays = [
        np.stack(np.meshgrid(levels, axes, axes, indexing="ij"), -1).reshape(-1, 3)
    ]
    source = np.concatenate([np.roll(arrays[0], i, axis=1) for i in range(3)]).reshape(
        -1, 256, 3
    )
    exact(native_denoise.lab(source, inverse=inverse), baseline_lab(source, inverse))


@pytest.mark.parametrize("inverse", [False, True])
def test_lab_exhaustive_byte_domain(inverse):
    """Every one of 256**3 triples; bounded blocks keep memory small."""
    level = np.arange(256, dtype=np.uint8)
    for first in range(0, 256, 8):
        source = np.stack(
            np.meshgrid(
                np.arange(first, first + 8, dtype=np.uint8), level, level, indexing="ij"
            ),
            axis=-1,
        ).reshape(-1, 256, 3)
        exact(
            native_denoise.lab(source, inverse=inverse), baseline_lab(source, inverse)
        )


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("strength", [0, 0.1, 3, 17.1, 50])
@pytest.mark.parametrize("patch,search", [(1, 1), (3, 10), (10, 3), (30, 1), (1, 30)])
def test_nlm_exact(channels, strength, patch, search):
    source = image((5, 7), channels)
    expected = cv2.fastNlMeansDenoising(
        source,
        h=strength,
        templateWindowSize=patch * 2 + 1,
        searchWindowSize=search * 2 + 1,
    )
    exact(native_denoise.nlm(source, strength, patch, search), expected)


@pytest.mark.parametrize("channels", [1, 2])
def test_every_registered_strength_step(channels):
    # Both L and AB weight tables. Low-contrast samples exercise weights instead
    # of degenerating to identity over most of the strength slider.
    source = image((3, 5), channels) // 8 + 64
    for step in range(501):
        strength = step / 10
        expected = cv2.fastNlMeansDenoising(
            source, h=strength, templateWindowSize=3, searchWindowSize=5
        )
        exact(native_denoise.nlm(source, strength, 1, 2), expected)


@pytest.mark.parametrize("patch", range(1, 31))
@pytest.mark.parametrize("channels", [1, 2])
def test_every_registered_window_radius(patch, channels):
    search = 31 - patch
    source = image((2, 3), channels) // 4 + 96
    expected = cv2.fastNlMeansDenoising(
        source,
        h=17.3,
        templateWindowSize=patch * 2 + 1,
        searchWindowSize=search * 2 + 1,
    )
    exact(native_denoise.nlm(source, 17.3, patch, search), expected)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (9, 1), (2, 3), (17, 19), (33, 31)])
@pytest.mark.parametrize(
    "strength,color_strength", [(0, 0), (3, 3), (0.1, 49.9), (50, 0)]
)
def test_node_exact(channels, shape, strength, color_strength):
    source = image(shape, channels).astype(np.float32) / 255
    before = source.copy()
    expected = REFERENCE.denoise_node(source, strength, color_strength, 3, 10)
    actual = CURRENT.denoise_node(source, strength, color_strength, 3, 10)
    exact(actual, expected)
    np.testing.assert_array_equal(source, before)
    assert not np.shares_memory(actual, source)
    assert actual.flags.writeable


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize(
    "layout", ["reverse", "transpose", "fortran", "readonly", "unaligned"]
)
def test_foreign_buffers(channels, layout):
    source = image((17, 19), channels).astype(np.float32) / 255
    if layout == "reverse":
        source = source[::-1, ::-1]
    elif layout == "transpose":
        source = source.swapaxes(0, 1)
    elif layout == "fortran":
        source = np.asfortranarray(source)
    elif layout == "readonly":
        source.flags.writeable = False
    else:
        other = np.ndarray(
            source.shape, np.float32, buffer=bytearray(source.nbytes + 1), offset=1
        )
        other[...] = source
        source = other
    exact(
        CURRENT.denoise_node(source, 17.1, 23.7, 2, 3),
        REFERENCE.denoise_node(source, 17.1, 23.7, 2, 3),
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_exceptional_input_quantization(channels):
    source = image((7, 11), channels).astype(np.float32) / 255
    source.flat[:10] = [
        np.nan,
        np.inf,
        -np.inf,
        -0.0,
        -1,
        2,
        0.5 / 255,
        1.5 / 255,
        1e-40,
        1e30,
    ]
    with np.errstate(all="ignore"):
        exact(
            CURRENT.denoise_node(source, 3, 3, 1, 2),
            REFERENCE.denoise_node(source, 3, 3, 1, 2),
        )


@pytest.mark.parametrize("shape", [(1, 1), (2, 3)])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_both_maximum_windows(shape, channels):
    source = image(shape, channels).astype(np.float32) / 255
    exact(
        CURRENT.denoise_node(source, 50, 50, 30, 30),
        REFERENCE.denoise_node(source, 50, 50, 30, 30),
    )


def test_singleton_channel():
    source = image((7, 11), 1).astype(np.float32)[..., None] / 255
    exact(
        CURRENT.denoise_node(source, 3, 3, 3, 10),
        REFERENCE.denoise_node(source, 3, 3, 3, 10),
    )


def test_concurrent_owned_tiles():
    inputs = [image((67, 73), c, seed=21 + c) for c in (1, 3, 4)]
    expected = [
        REFERENCE.denoise_node(a.astype(np.float32) / 255, 23.7, 17.1, 2, 3)
        for a in inputs
    ]

    def run(index):
        which = index % len(inputs)
        exact(native_denoise.denoise(inputs[which], 23.7, 17.1, 2, 3), expected[which])

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(12)))


def test_no_cv2_algorithm_calls(monkeypatch):
    source = image((13, 17), 4).astype(np.float32) / 255
    expected = REFERENCE.denoise_node(source, 17, 29, 2, 3)

    def forbidden(*args, **kwargs):
        raise AssertionError("Denoise must run its own CPU algorithm")

    for name in ("cvtColor", "fastNlMeansDenoising", "fastNlMeansDenoisingColored"):
        monkeypatch.setattr(cv2, name, forbidden)
    exact(CURRENT.denoise_node(source, 17, 29, 2, 3), expected)


@pytest.mark.parametrize("shape", [(0, 3), (3,), (1, 1, 5), (1, 1, 0), (1, 1, 1, 1)])
def test_invalid_image(shape):
    with pytest.raises(ValueError):
        native_denoise.denoise(np.empty(shape, np.uint8), 3, 3, 1, 1)


def test_invalid_dtype():
    with pytest.raises(TypeError):
        native_denoise.nlm(np.zeros((3, 3), np.float32), 3, 1, 1)


@pytest.mark.parametrize("radius", [0, -1, 31, 2**32 + 1])
def test_radius_validation_prevents_ctypes_wrap(radius):
    with pytest.raises(ValueError):
        native_denoise.nlm(np.zeros((3, 3), np.uint8), 3, radius, 1)


@pytest.mark.parametrize("strength", [np.nan, np.inf, -np.inf, -3])
def test_non_ui_strength_retains_original_arithmetic(strength):
    source = image((5, 7), 2)
    expected = cv2.fastNlMeansDenoising(
        source, h=strength, templateWindowSize=3, searchWindowSize=5
    )
    exact(native_denoise.nlm(source, strength, 1, 2), expected)


@pytest.mark.parametrize(
    "case", ["null", "overlap", "zero", "channels", "overflow", "patch", "search"]
)
def test_abi_rejects_before_write(case):
    source = image((3, 5), 3)
    target = np.full_like(source, 173)
    args = [source.ctypes.data, target.ctypes.data, 3, 5, 3, 3.0, 3.0, 1, 1]
    expected = 1
    if case == "null":
        args[0] = 0
    elif case == "overlap":
        args[0] = target.ctypes.data + 1
    elif case == "zero":
        args[2] = 0
    elif case == "channels":
        args[4] = 2
    elif case == "overflow":
        args[3] = ct.c_size_t(-1).value
        expected = 2
    elif case == "patch":
        args[7] = 31
    else:
        args[8] = 0
    # An intentionally malformed ABI call must not reach any worker.
    assert native_denoise._api().cn_denoise_u8(*args) == expected
    assert (target == 173).all()


def test_registration_unchanged():
    actual = ast.parse(NODE.read_text(encoding="utf-8"))
    expected = ast.parse((TESTS / "reference_denoise.py").read_text(encoding="utf-8"))
    a = next(x for x in actual.body if isinstance(x, ast.FunctionDef))
    b = next(x for x in expected.body if isinstance(x, ast.FunctionDef))
    assert ast.dump(a.decorator_list[0]) == ast.dump(b.decorator_list[0])
