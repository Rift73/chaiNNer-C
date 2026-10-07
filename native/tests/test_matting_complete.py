"""Exact solver parity with the live PyMatting 1.1.16."""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pymatting
import pytest

from nodes.impl import native_matting as native


def inputs(shape, seed=1):
    random = np.random.default_rng(seed)
    image = random.random((*shape, 3))
    trimap = np.full(shape, 0.5)
    trimap[:, 0] = 0
    trimap[:, -1] = 1
    return image, trimap


@pytest.mark.parametrize("shape", [(3, 3), (4, 5), (12, 13), (17, 9), (33, 35)])
@pytest.mark.parametrize("seed", range(3))
def test_laplacian(shape, seed):
    image, trimap = inputs(shape, seed)
    values = np.empty((trimap.size, 25), np.float64)
    mapping = np.empty(trimap.size, np.int64)
    unknown, error = ct.c_size_t(), ct.c_int()
    result = native._api().cn_matting_cf_stencil(
        image.ctypes.data,
        trimap.ctypes.data,
        *shape,
        values.ctypes.data,
        mapping.ctypes.data,
        ct.byref(unknown),
        ct.byref(error),
    )
    assert result == error.value == 0
    known = (trimap <= 0.1) | (trimap >= 0.9)
    reference = pymatting.cf_laplacian(image, is_known=known)
    np.testing.assert_array_equal(values.ravel(), reference.data)
    assert unknown.value == np.count_nonzero(~known)


@pytest.mark.parametrize("shape", [(3, 3), (4, 5), (12, 13), (17, 9), (33, 35)])
@pytest.mark.parametrize("seed", range(3))
def test_alpha(shape, seed):
    image, trimap = inputs(shape, seed)
    expected = pymatting.estimate_alpha_cf(image, trimap)
    actual = native.estimate_alpha(image, trimap)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "shape", [(1, 2), (2, 1), (3, 3), (4, 5), (12, 13), (17, 9), (33, 35)]
)
@pytest.mark.parametrize("seed", range(3))
def test_foreground(shape, seed):
    image, _ = inputs(shape, seed)
    alpha = np.random.default_rng(seed + 19).random(shape)
    expected = pymatting.estimate_foreground_ml(image, alpha, return_background=True)
    actual = native.estimate_foreground(image, alpha, return_background=True)
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, e)


@pytest.mark.parametrize("known", ["none", "foreground", "background", "thresholds"])
def test_foreground_mean_initialization(known):
    # PyMatting 1.1.16 starts from the mean colours of the alpha > 0.9 and the
    # alpha < 0.1 pixels; a colour without such pixels is 0 / (0 + 1e-5).
    image, _ = inputs((9, 11), 47)
    alpha = np.full((9, 11), 0.5)
    if known == "foreground":
        alpha[::2] = 1
    elif known == "background":
        alpha[::2] = 0
    elif known == "thresholds":
        # float32 values on either side of 0.9 and 0.1, compared as doubles.
        edges = [np.float32(0.9), np.float32(0.1)]
        edges += [np.nextafter(v, np.float32(d)) for v in edges for d in (0, 1)]
        alpha.flat[: len(edges) * 7] = np.repeat(edges, 7)
    actual = native.estimate_foreground(image, alpha, return_background=True)
    expected = pymatting.estimate_foreground_ml(image, alpha, return_background=True)
    for a, e in zip(actual, expected, strict=True):
        assert a.tobytes() == e.tobytes()


def arrange(array, layout):
    if layout == "strided":
        return array[::-1, ::-1]
    if layout == "readonly":
        array.flags.writeable = False
    if layout == "unaligned":
        result = np.ndarray(
            array.shape, array.dtype, buffer=bytearray(array.nbytes + 1), offset=1
        )
        result[...] = array
        return result
    return array


@pytest.mark.parametrize("shape", [(3, 7), (5, 4), (11, 13), (32, 33), (35, 34)])
@pytest.mark.parametrize(
    "pattern", ["constant", "gray", "gradient", "random", "sparse"]
)
def test_alpha_image_patterns(shape, pattern):
    image, trimap = inputs(shape, 723)
    if pattern == "constant":
        image.fill(0.3)
    elif pattern == "gray":
        image[:] = image[:, :, :1]
    elif pattern == "gradient":
        image[:] = np.linspace(0, 1, shape[1])[None, :, None]
    elif pattern == "sparse":
        trimap[:] = 0.5
        trimap[0, 0], trimap[-1, -1] = 0, 1
    np.testing.assert_array_equal(
        native.estimate_alpha(image, trimap), pymatting.estimate_alpha_cf(image, trimap)
    )


@pytest.mark.parametrize("layout", ["plain", "strided", "readonly", "unaligned"])
@pytest.mark.parametrize("seed", range(6))
def test_readonly_foreign_buffers_and_immutability(layout, seed):
    image, trimap = [arrange(a, layout) for a in inputs((13, 17), seed)]
    original_image, original_trimap = image.copy(), trimap.copy()
    alpha = native.estimate_alpha(image, trimap)
    np.testing.assert_array_equal(
        alpha, pymatting.estimate_alpha_cf(original_image, original_trimap)
    )
    matte = arrange(alpha, layout)
    foreground = native.estimate_foreground(image, matte)
    expected = pymatting.estimate_foreground_ml(original_image, matte.copy())
    np.testing.assert_array_equal(foreground, expected)
    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(trimap, original_trimap)


@pytest.mark.parametrize(
    "value", [0.0, 1.0, 0.5, -0.0, -1.0, 2.0, np.nan, np.inf, -np.inf]
)
@pytest.mark.parametrize("which", ["image", "alpha"])
def test_foreground_extremes(value, which):
    image, alpha = inputs((7, 9), 86)
    if which == "image":
        image.flat[13] = value
    else:
        alpha.flat[13] = value
    actual = native.estimate_foreground(image, alpha, True)
    expected = pymatting.estimate_foreground_ml(image, alpha, return_background=True)
    for a, e in zip(actual, expected, strict=True):
        assert a.tobytes() == e.tobytes()


@pytest.mark.parametrize(
    "kind", ["no_background", "no_foreground", "all_known", "thin", "nan", "inf"]
)
def test_original_failures(kind):
    image, trimap = inputs((3, 5))
    if kind == "no_background":
        trimap.fill(1)
    elif kind == "no_foreground":
        trimap.fill(0)
    elif kind == "all_known":
        trimap[:, 1:] = 1
    elif kind == "thin":
        image, trimap = image[:2], trimap[:2]
    elif kind in ("nan", "inf"):
        image[1, 2, 0] = float(kind)
    with pytest.raises((ValueError, ZeroDivisionError)) as expected:
        pymatting.estimate_alpha_cf(image, trimap)
    with pytest.raises(type(expected.value)) as actual:
        native.estimate_alpha(image, trimap)
    assert str(actual.value) == str(expected.value)


def test_concurrent_repeatable_solvers():
    cases = [inputs((11, 13), seed) for seed in range(3)]
    for image, trimap in cases:
        image.flags.writeable = trimap.flags.writeable = False

    def run(index):
        image, trimap = cases[index % len(cases)]
        alpha = native.estimate_alpha(image, trimap)
        foreground = native.estimate_foreground(image, alpha)
        assert isinstance(foreground, np.ndarray)
        return native.output(foreground, alpha)

    expected = [run(index) for index in range(3)]
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(run, range(24)))
    for index, actual in enumerate(results):
        assert actual.tobytes() == expected[index % 3].tobytes()


@pytest.mark.parametrize("shape", [(3, 5, 4), (3, 5), (0, 5, 3), (2, 3, 5, 3)])
def test_image_contract(shape):
    image = np.zeros(shape, np.float64)
    with pytest.raises(ValueError):
        native.estimate_alpha(image, np.zeros((3, 5), np.float64))
    with pytest.raises(ValueError):
        native.estimate_foreground(image, np.zeros((3, 5), np.float64))


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((3, 5, 1), np.float64),
        ((3, 4), np.float64),
        ((3, 5), np.float32),
        ((3, 5), ">f8"),
    ],
)
def test_matte_contract(shape, dtype):
    image, _ = inputs((3, 5))
    matte = np.zeros(shape, dtype)
    with pytest.raises(ValueError):
        native.estimate_alpha(image, matte)
    with pytest.raises(ValueError):
        native.estimate_foreground(image, matte)


def test_raw_abi_bounds_no_output_mutation():
    dll = native._api()
    data = np.full(100, 0.5, np.float64)
    saved = data.copy()
    p = data.ctypes.data
    maximum = ct.c_size_t(-1).value
    assert dll.cn_matting_cf_alpha(None, p, 1, 1, p, p, p, p) == 1
    assert dll.cn_matting_cf_alpha(p, p, maximum, 2, p, p, p, p) == 2
    assert dll.cn_matting_cf_stencil(p, p, maximum, 2, p, p, p, p) == 2
    assert dll.cn_matting_foreground(p, p, maximum, 2, p, p) == 2
    assert dll.cn_matting_foreground(p, p, 0, 2, p, p) == 1
    assert dll.cn_matting_output(p, p, maximum, 1, p) == 2
    assert dll.cn_matting_output(p, p, 1, 2, p) == 1
    assert data.tobytes() == saved.tobytes()


@pytest.mark.parametrize(
    "thresholds", [(1, 0), (255, 254), (240, 15), (128, 127), (255, 0)]
)
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("layout", ["plain", "strided", "readonly", "unaligned"])
def test_alpha_node_thresholds_exact(thresholds, channels, layout):
    from test_transparency import MATTING, ORIGINAL_MATTING

    random = np.random.default_rng(874)
    image = arrange(random.random((9, 11, channels), dtype=np.float32), layout)
    trimap = np.linspace(0, 1, 99, dtype=np.float32).reshape(9, 11)
    trimap.flat[2:6] = [thresholds[0] / 255, thresholds[1] / 255, 0.1, 0.9]
    trimap = arrange(trimap, layout)
    # Near-adjacent thresholds can remove every unknown pixel. Keep the
    # original convergence exception as part of the contract in that case.
    try:
        expected = ORIGINAL_MATTING.alpha_matting_node(image, trimap, *thresholds)
    except ValueError as error:
        with pytest.raises(ValueError) as actual:
            MATTING.alpha_matting_node(image, trimap, *thresholds)
        assert str(actual.value) == str(error)
    else:
        actual = MATTING.alpha_matting_node(image, trimap, *thresholds)
        assert actual.dtype == expected.dtype == np.float64
        assert actual.tobytes() == expected.tobytes()
        # Alpha node normalization downstream also has identical float32 bits.
        assert (
            actual.astype(np.float32).tobytes() == expected.astype(np.float32).tobytes()
        )


@pytest.mark.parametrize(
    "threshold_bg,threshold_fg", [(0.0, 0.3), (0.01, 0.4), (0.2, 0.2), (0.8, 0.1)]
)
@pytest.mark.parametrize("confusion", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_chroma_node_solver_exact(threshold_bg, threshold_fg, confusion):
    from test_transparency import CHROMA, ORIGINAL_CHROMA, Color

    image = np.random.default_rng(31).random((13, 17, 3), dtype=np.float32)
    key = Color.bgr((0.12, 0.43, 0.88))
    image[:, :5] = key.value
    image[:, -5:] = [1, 1, 0]
    arguments = (image, key, threshold_bg, threshold_fg, *confusion, False)
    try:
        expected = ORIGINAL_CHROMA.trimap_matting_keying(*arguments)
    except ValueError as error:
        with pytest.raises(ValueError) as actual:
            CHROMA.trimap_matting_keying(*arguments)
        assert str(actual.value) == str(error)
    else:
        actual = CHROMA.trimap_matting_keying(*arguments)
        assert actual.dtype == expected.dtype == np.float32
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "which,value", [("image", -0.2), ("image", 1.3), ("trimap", -0.5), ("trimap", 2.0)]
)
def test_validation_warnings_match(which, value):
    import warnings

    image, trimap = inputs((7, 9))
    (image if which == "image" else trimap).flat[10] = value
    with warnings.catch_warnings(record=True) as expected:
        warnings.simplefilter("always")
        reference = pymatting.estimate_alpha_cf(image, trimap)
    with warnings.catch_warnings(record=True) as actual:
        warnings.simplefilter("always")
        result = native.estimate_alpha(image, trimap)
    assert [str(item.message) for item in actual] == [
        str(item.message) for item in expected
    ]
    np.testing.assert_array_equal(result, reference)


def test_cholesky_retry_diagnostics_match(capsys):
    image, trimap = inputs((2, 5))
    for function in (pymatting.estimate_alpha_cf, native.estimate_alpha):
        with pytest.raises(ValueError, match="did not converge"):
            function(image, trimap)
        capture = capsys.readouterr().out
        if function is pymatting.estimate_alpha_cf:
            expected = capture
        else:
            assert capture == expected


def test_native_load_failure_is_not_hidden(monkeypatch):
    def unavailable():
        raise RuntimeError("matting DLL unavailable")

    monkeypatch.setattr(native, "_api", unavailable)
    image, trimap = inputs((3, 5))
    with pytest.raises(RuntimeError, match="matting DLL unavailable"):
        native.estimate_alpha(image, trimap)
    with pytest.raises(RuntimeError, match="matting DLL unavailable"):
        native.estimate_foreground(image, trimap)


@pytest.mark.parametrize("seed", range(20))
def test_irregular_masks_with_unknown_index_zero(seed):
    image, _ = inputs((7 + seed % 4, 9 + seed % 3), seed)
    random = np.random.default_rng(923 + seed)
    trimap = random.choice([0.05, 0.95, 0.2, 0.5, 0.8], image.shape[:2])
    trimap[0, 0] = np.nan if seed % 2 else 0.5
    trimap[1, 1], trimap[1, 2] = 0.05, 0.95
    expected = pymatting.estimate_alpha_cf(image, trimap)
    actual = native.estimate_alpha(image, trimap)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("radius", range(101))
@pytest.mark.parametrize("binary", [False, True])
def test_chroma_ellipse_erosion_every_node_radius(radius, binary):
    random = np.random.default_rng(842)
    mask = random.integers(0, 256, (9, 17), np.uint8)
    if binary:
        mask = (mask > 127).astype(np.uint8) * 255
    expected = cv2.erode(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    )
    actual = native.erode_mask(mask, radius)
    np.testing.assert_array_equal(actual, expected)
    if not radius:
        assert actual is mask


@pytest.mark.parametrize("shape", [(1, 1), (1, 23), (31, 1), (31, 37), (129, 137)])
@pytest.mark.parametrize("radius", [1, 2, 7, 19, 100])
@pytest.mark.parametrize("layout", ["plain", "strided", "readonly", "unaligned"])
def test_chroma_ellipse_edges_and_foreign_buffers(shape, radius, layout):
    mask = arrange(np.random.default_rng(5).integers(0, 256, shape, np.uint8), layout)
    saved = mask.copy()
    expected = cv2.erode(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    )
    np.testing.assert_array_equal(native.erode_mask(mask, radius), expected)
    np.testing.assert_array_equal(mask, saved)


@pytest.mark.parametrize("layout", ["plain", "strided", "readonly", "unaligned"])
def test_trimap_packing_special_values(layout):
    image = np.random.default_rng(56).random((7, 11, 3), dtype=np.float32)
    matte = np.random.default_rng(63).random((7, 11))
    image.flat[:6] = [0, -0.0, np.nan, np.inf, -np.inf, 1]
    matte.flat[:6] = [0, -0.0, np.nan, np.inf, -np.inf, 1]
    image, matte = arrange(image, layout), arrange(matte, layout)
    expected = np.dstack((image, matte.astype(np.float32)))
    assert native.trimap_output(image, matte).tobytes() == expected.tobytes()


def test_chroma_new_buffer_contracts():
    for invalid in (
        np.zeros((2, 2), np.float32),
        np.zeros((2, 2, 1), np.uint8),
        np.zeros((0, 3), np.uint8),
    ):
        with pytest.raises(ValueError):
            native.erode_mask(invalid, 1)
    for radius in (-1, 46341):
        with pytest.raises(ValueError):
            native.erode_mask(np.zeros((2, 2), np.uint8), radius)
    image = np.zeros((2, 2, 3), np.float32)
    for matte in (
        np.zeros((2, 2), np.float32),
        np.zeros((2, 2, 1), np.float64),
        np.zeros((1, 2), np.float64),
    ):
        with pytest.raises(ValueError):
            native.trimap_output(image, matte)
    dll = native._api()
    buffer = np.zeros(32, np.float64)
    p, maximum = buffer.ctypes.data, ct.c_size_t(-1).value
    assert dll.cn_matting_erode_mask_u8(None, p, 1, 1, 1) == 1
    assert dll.cn_matting_erode_mask_u8(p, p, maximum, 1, 1) == 2
    assert dll.cn_matting_erode_mask_u8(p, p, 2**31, 1, 1) == 2
    assert dll.cn_matting_erode_mask_u8(p, p, 1, 1, 46341) == 1
    assert dll.cn_matting_trimap_output(None, 3, 1, p, 1, p, 4, 1) == 1
    assert dll.cn_matting_trimap_output(p, 0, 1, p, 1, p, 4, 1) == 1
    assert dll.cn_matting_trimap_output(p, 3, 1, p, 1, p, 4, 0) == 1
    assert dll.cn_matting_trimap_output(p, 3, 1, p, maximum, p, 4, 1) == 2
    assert not buffer.any()


def test_concurrent_chroma_masks_and_packing():
    mask = np.random.default_rng(9).integers(0, 256, (65, 79), np.uint8)
    image = np.random.default_rng(9).random((65, 79, 3), dtype=np.float32)
    mask.flags.writeable = image.flags.writeable = False
    matte = np.full(mask.shape, 0.5)
    expected = [
        (native.erode_mask(mask, r), native.trimap_output(image, matte))
        for r in (1, 10, 100)
    ]

    def run(index):
        return native.erode_mask(mask, (1, 10, 100)[index % 3]), native.trimap_output(
            image, matte
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        result = list(executor.map(run, range(18)))
    for i, pair in enumerate(result):
        for a, e in zip(pair, expected[i % 3], strict=True):
            assert a.tobytes() == e.tobytes()
