"""CPU-only differential checks against the unmodified upstream algorithms."""

from __future__ import annotations

import ctypes
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import DTypeLike

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend" / "src"))

import reference_gradients
import reference_normals_addition as reference_addition
import reference_normals_util as reference_util

from nodes.impl import gradients, native_geometry
from nodes.impl.normals import addition, util


def assert_equivalent(actual, expected):
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        for a, e in zip(actual, expected, strict=False):
            assert_equivalent(a, e)
        return
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    np.testing.assert_array_equal(actual, expected)
    valid = ~np.isnan(expected)
    unsigned = np.uint32 if expected.dtype == np.float32 else np.uint64
    np.testing.assert_array_equal(
        actual[valid].view(unsigned), expected[valid].view(unsigned)
    )


GRADIENT_CASES = [
    ("horizontal_gradient", ()),
    ("vertical_gradient", ()),
    ("diagonal_gradient", (np.pi / 4, 100)),
    ("diagonal_gradient", (0, 0)),
    ("diagonal_gradient", (-np.pi / 3, 2.7)),
    ("radial_gradient", ()),
    ("radial_gradient", (0.25, 0.25)),
    ("radial_gradient", (0.8, 0.1)),
    ("conic_gradient", ()),
    ("conic_gradient", (np.pi,)),
    ("conic_gradient", (-7 * np.pi,)),
    ("conic_gradient", (7 * np.pi,)),
]


@pytest.mark.parametrize("name,args", GRADIENT_CASES)
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (7, 1), (8, 11), (64, 64)])
@pytest.mark.parametrize("strided", [False, True])
def test_gradients(name, args, shape, strided):
    actual = np.empty(shape, dtype=np.float32)
    if strided:
        actual = np.empty((shape[0] * 2, shape[1] * 2), dtype=np.float32)[::2, ::-2]
    expected = np.empty(shape, dtype=np.float32)
    with np.errstate(all="ignore"):
        assert getattr(gradients, name)(actual, *args) is None
        assert getattr(reference_gradients, name)(expected, *args) is None
    assert_equivalent(actual, expected)


@pytest.mark.parametrize("name,args", GRADIENT_CASES)
def test_gradient_float64_compatibility(name, args):
    actual = np.empty((7, 11), dtype=np.float64)
    expected = np.empty_like(actual)
    with np.errstate(all="ignore"):
        getattr(gradients, name)(actual, *args)
        getattr(reference_gradients, name)(expected, *args)
    np.testing.assert_array_equal(actual, expected)


def test_diagonal_float32_angle_compatibility():
    actual = np.empty((13, 17), dtype=np.float32)
    expected = np.empty_like(actual)
    angle = np.float32(1.23)
    gradients.diagonal_gradient(actual, angle, 11)  # pyright: ignore[reportArgumentType] -- deliberate: a NumPy float32 angle pins float32 trig parity with upstream; the signature is upstream's
    reference_gradients.diagonal_gradient(expected, angle, 11)  # pyright: ignore[reportArgumentType] -- deliberate: a NumPy float32 angle pins float32 trig parity with upstream; the signature is upstream's
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("name,args", GRADIENT_CASES)
def test_readonly_gradient_rejected(name, args):
    actual = np.zeros((7, 11), dtype=np.float32)
    actual.flags.writeable = False
    with np.errstate(all="ignore"):
        with pytest.raises(ValueError):
            getattr(gradients, name)(actual, *args)
        with pytest.raises(ValueError):
            getattr(reference_gradients, name)(actual, *args)


@pytest.mark.parametrize("name", ["horizontal_gradient", "vertical_gradient"])
@pytest.mark.parametrize("shape", [(0, 3), (4, 0), (0, 0)])
def test_empty_axis_gradients(name, shape):
    actual = np.empty(shape, dtype=np.float32)
    expected = actual.copy()
    getattr(gradients, name)(actual)
    getattr(reference_gradients, name)(expected)
    assert_equivalent(actual, expected)


def normal_image(shape=(13, 17, 3), dtype: DTypeLike = np.float32):
    return np.random.default_rng(719).uniform(-0.1, 1.1, shape).astype(dtype)


@pytest.mark.parametrize("name", ["gr_to_xyz", "octahedral_gr_to_xyz"])
@pytest.mark.parametrize("shape", [(1, 1, 3), (11, 7, 4), (0, 4, 3), (4, 0, 4)])
@pytest.mark.parametrize("strided", [False, True])
def test_decode(name, shape, strided):
    image = normal_image(shape)
    if strided:
        image = image[::-1, ::-1, :]
    image.flags.writeable = False
    original = image.copy()
    actual = getattr(util, name)(image)
    expected = getattr(reference_util, name)(image)
    assert_equivalent(actual, expected)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("name", ["gr_to_xyz", "octahedral_gr_to_xyz"])
def test_nonfinite_decode(name):
    image = np.array(
        [[[0, np.nan, 0.5], [1, np.inf, 0], [0, 0.5, -np.inf], [0, 0.5, 0.5]]],
        dtype=np.float32,
    )
    with np.errstate(all="ignore"):
        assert_equivalent(
            getattr(util, name)(image), getattr(reference_util, name)(image)
        )


@pytest.mark.parametrize("strided", [False, True])
def test_normalize_mutates_and_returns_inputs(strided):
    x = normal_image((8, 11)) * 3 - 1.5
    y = normal_image((8, 11)) * 2 - 1
    if strided:
        x, y = x[::-1, ::2], y[::-1, ::2]
    expected_x, expected_y = x.copy(), y.copy()
    expected = reference_util.normalize_normals(expected_x, expected_y)
    actual = util.normalize_normals(x, y)
    assert actual[0] is x
    assert actual[1] is y
    assert_equivalent(actual, expected)
    assert_equivalent(x, expected_x)
    assert_equivalent(y, expected_y)


def test_normalize_alias_compatibility():
    actual = np.full((3, 4), 0.9, dtype=np.float32)
    expected = actual.copy()
    expected_result = reference_util.normalize_normals(expected, expected)
    actual_result = util.normalize_normals(actual, actual)
    assert actual_result[0] is actual_result[1] is actual
    assert_equivalent(actual_result, expected_result)


@pytest.mark.parametrize("name", ["normalize_normals", "xyz_to_octahedral_bgr"])
@pytest.mark.parametrize("readonly_index", [0, 1])
def test_inplace_readonly_rejection_and_partial_mutation(name, readonly_index):
    components = tuple(np.full((3, 4), v, dtype=np.float32) for v in [0.9, 0.7, 0.1])
    reference = tuple(c.copy() for c in components)
    components[readonly_index].flags.writeable = False
    reference[readonly_index].flags.writeable = False
    for module, args in [(util, components), (reference_util, reference)]:
        with pytest.raises(ValueError):
            if name == "normalize_normals":
                getattr(module, name)(*args[:2])
            else:
                getattr(module, name)(args)
    for actual, expected in zip(components, reference, strict=False):
        assert_equivalent(actual, expected)


@pytest.mark.parametrize("name", ["xyz_to_bgr", "xyz_to_octahedral_bgr"])
@pytest.mark.parametrize("strided", [False, True])
def test_encode_and_mutation(name, strided):
    xyz = util.gr_to_xyz(normal_image())
    if strided:
        xyz = tuple(c[::-1, ::2] for c in xyz)
    reference = tuple(c.copy() for c in xyz)
    actual = getattr(util, name)(xyz)
    expected = getattr(reference_util, name)(reference)
    assert_equivalent(actual, expected)
    for a, e in zip(xyz, reference, strict=False):
        assert_equivalent(a, e)


def test_octahedral_zero_vector_nan_and_alias_compatibility():
    for alias in [False, True]:
        x = np.zeros((1, 1), dtype=np.float32)
        actual_xyz = (x, x if alias else x.copy(), x.copy())
        e = x.copy()
        expected_xyz = (e, e if alias else e.copy(), e.copy())
        with np.errstate(all="ignore"):
            actual = util.xyz_to_octahedral_bgr(actual_xyz)
            expected = reference_util.xyz_to_octahedral_bgr(expected_xyz)
        assert_equivalent(actual, expected)
        assert actual[0, 0, 0] == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", [0, 1])
@pytest.mark.parametrize("factors", [(1, 1), (0, 0), (0.25, 2), (-0.5, 1.5)])
def test_addition(dtype, method, factors):
    n1 = normal_image((13, 17, 4), dtype)[::-1, ::-1]
    n2 = normal_image((13, 17, 3), dtype)
    n1.flags.writeable = False
    n2.flags.writeable = False
    actual = addition.add_normals(addition.AdditionMethod(method), n1, n2, *factors)
    expected = reference_addition.add_normals(
        reference_addition.AdditionMethod(method), n1, n2, *factors
    )
    assert_equivalent(actual, expected)


@pytest.mark.parametrize("method", [0, 1])
@pytest.mark.parametrize("factor", [0, 0.25, 1, 2, -0.5])
def test_strengthen(method, factor):
    image = normal_image((13, 17, 4))[::2, ::-1]
    image.flags.writeable = False
    actual = addition.strengthen_normals(addition.AdditionMethod(method), image, factor)
    expected = reference_addition.strengthen_normals(
        reference_addition.AdditionMethod(method), image, factor
    )
    assert_equivalent(actual, expected)


@pytest.mark.parametrize(
    "name",
    [
        "normalize_normals",
        "gr_to_xyz",
        "octahedral_gr_to_xyz",
        "xyz_to_bgr",
        "xyz_to_octahedral_bgr",
    ],
)
def test_normal_float64_compatibility(name):
    image = normal_image(dtype=np.float64)
    if name.endswith("gr_to_xyz") or name == "gr_to_xyz":
        actual = getattr(util, name)(image)
        expected = getattr(reference_util, name)(image)
    else:
        xyz = reference_util.gr_to_xyz(image)
        reference = tuple(c.copy() for c in xyz)
        if name == "normalize_normals":
            actual = getattr(util, name)(*xyz[:2])
            expected = getattr(reference_util, name)(*reference[:2])
        else:
            actual = getattr(util, name)(xyz)
            expected = getattr(reference_util, name)(reference)
    for a, e in zip(actual, expected, strict=False):
        np.testing.assert_array_equal(a, e)


def test_native_error_is_not_a_fallback(monkeypatch):
    def broken_api():
        raise RuntimeError("broken C library")

    monkeypatch.setattr(native_geometry, "_api", broken_api)
    with pytest.raises(RuntimeError, match="broken C library"):
        gradients.horizontal_gradient(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(RuntimeError, match="broken C library"):
        util.gr_to_xyz(normal_image())


def test_c_boundary_rejects_invalid_sizes_and_modes():
    api = native_geometry._api()  # verify the raw C argument boundary
    maximum = ctypes.c_size_t(-1).value
    assert api.cn_gradient(None, maximum, 2, 0, 0, 1, 0) == 2
    assert api.cn_gradient(None, 1, 1, 0, 0, 1, 0) == 1
    assert api.cn_gradient(None, 0, 0, 5, 0, 1, 0) == 1
    assert api.cn_gradient(None, 0, 0, 0, 0, 1, 1) == 1
    unaligned = ctypes.cast(ctypes.c_void_p(1), ctypes.POINTER(ctypes.c_float))
    assert api.cn_gradient(unaligned, 1, 1, 0, 0, 1, 0) == 1
    end_address = ctypes.cast(
        ctypes.c_void_p(maximum - 3), ctypes.POINTER(ctypes.c_float)
    )
    assert api.cn_gradient(end_address, 1, 1, 0, 0, 1, 0) == 2
    assert api.cn_normal_decode(None, 0, 2, 0, None, None, None) == 1
    assert api.cn_normal_decode(None, maximum, 3, 0, None, None, None) == 2
    assert api.cn_normalize_normals(None, None, None, 1) == 1
    assert api.cn_normal_encode(None, None, None, maximum, 0, None) == 2


@pytest.mark.parametrize("method", [0, 1])
def test_normal_workflow_bit_identity(method):
    # This includes many normals at the hemisphere boundary, where a one-ULP
    # sin/asin difference would otherwise be amplified by reconstruction of Z.
    rng = np.random.default_rng(274)
    n1 = rng.uniform(-0.1, 1.1, (64, 64, 4)).astype(np.float32)
    n2 = rng.uniform(-0.1, 1.1, (64, 64, 3)).astype(np.float32)
    for f1, f2 in [(1, 1), (0.25, 2), (0, 0), (-0.5, 1.5), (1.7, 0.3)]:
        pairs = [
            (
                addition.add_normals(addition.AdditionMethod(method), n1, n2, f1, f2),
                reference_addition.add_normals(
                    reference_addition.AdditionMethod(method), n1, n2, f1, f2
                ),
            ),
            (
                addition.strengthen_normals(addition.AdditionMethod(method), n1, f1),
                reference_addition.strengthen_normals(
                    reference_addition.AdditionMethod(method), n1, f1
                ),
            ),
        ]
        for actual, expected in pairs:
            for a, e in zip(actual, expected, strict=False):
                np.testing.assert_array_equal(a, e)
    for decode in ["gr_to_xyz", "octahedral_gr_to_xyz"]:
        actual_xyz = getattr(util, decode)(n1)
        expected_xyz = getattr(reference_util, decode)(n1)
        for a, e in zip(actual_xyz, expected_xyz, strict=False):
            np.testing.assert_array_equal(a, e)
        for encode in ["xyz_to_bgr", "xyz_to_octahedral_bgr"]:
            actual = getattr(util, encode)(tuple(c.copy() for c in actual_xyz))
            expected = getattr(reference_util, encode)(
                tuple(c.copy() for c in expected_xyz)
            )
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "name,args",
    [
        ("horizontal_gradient", ()),
        ("vertical_gradient", ()),
        ("diagonal_gradient", (np.pi / 3, 137)),
        ("radial_gradient", (0.1, 0.9)),
        ("conic_gradient", (np.pi / 4,)),
    ],
)
def test_large_gradient_parallel_parity(name, args):
    # More than two 65536-pixel blocks, with an incomplete final block.
    actual = np.empty((257, 513), dtype=np.float32)
    expected = np.empty_like(actual)
    getattr(gradients, name)(actual, *args)
    getattr(reference_gradients, name)(expected, *args)
    assert_equivalent(actual, expected)


def test_concurrent_large_normal_calls():
    def compare(seed: int):
        rng = np.random.default_rng(seed)
        n1 = rng.uniform(-0.1, 1.1, (257, 513, 4)).astype(np.float32)
        n2 = rng.uniform(-0.1, 1.1, (257, 513, 3)).astype(np.float32)
        method = seed % 2
        actual_xyz = addition.add_normals(
            addition.AdditionMethod(method), n1, n2, 0.7, 1.2
        )
        expected_xyz = reference_addition.add_normals(
            reference_addition.AdditionMethod(method), n1, n2, 0.7, 1.2
        )
        for a, e in zip(actual_xyz, expected_xyz, strict=False):
            np.testing.assert_array_equal(a, e)
        for name in ["gr_to_xyz", "octahedral_gr_to_xyz"]:
            actual_xyz = getattr(util, name)(n1)
            expected_xyz = getattr(reference_util, name)(n1)
            for a, e in zip(actual_xyz, expected_xyz, strict=False):
                np.testing.assert_array_equal(a, e)
            # Exercise in-place C buffers and separate output buffers concurrently.
            actual = util.xyz_to_octahedral_bgr(actual_xyz)
            expected = reference_util.xyz_to_octahedral_bgr(expected_xyz)
            np.testing.assert_array_equal(actual, expected)
            for a, e in zip(actual_xyz, expected_xyz, strict=False):
                np.testing.assert_array_equal(a, e)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(compare, range(4)))
