"""Exact installed Rust parity for Fragment Blur and spatial Nearest Color.

No tolerance, model execution, GPU use or performance measurements are involved.
NaN payloads are unspecified; every non-NaN result is compared by float32 bits.
"""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from chainner_ext import fill_alpha_fragment_blur as original_fragment
from chainner_ext import fill_alpha_nearest_color as original_nearest
from nodes.impl import native_alpha_gamma as native
from nodes.impl.native import ptr


def exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[finite].view(np.uint32), expected[finite].view(np.uint32)
    )


def source(shape=(17, 23), pattern="random", seed=1741):
    rng = np.random.default_rng(seed)
    result = rng.random((*shape, 4), dtype=np.float32)
    result[:, :, 3] = rng.choice(np.array([0, 0.04, 0.05, 0.5, 1], np.float32), shape)
    if pattern == "sparse":
        result[:, :, 3] = 0
        for y, x in [(0, 0), (-1, -1), (shape[0] // 2, shape[1] // 2)]:
            result[y, x, 3] = 1
    elif pattern == "opaque":
        result[:, :, 3] = 1
    elif pattern == "transparent":
        result[:, :, 3] = 0
    elif pattern == "signed_zero":
        result[:, :, :3] = rng.choice(
            np.array([-0.0, 0.0, 0.25], np.float32), (*shape, 3)
        )
    elif pattern == "nonfinite":
        result.ravel()[::13] = np.nan
        result.ravel()[1::29] = np.inf
        result.ravel()[2::31] = -np.inf
    elif pattern == "dynamic":
        exponent = rng.integers(-145, 122, (*shape, 3), dtype=np.int32)
        result[:, :, :3] = np.ldexp(result[:, :, :3] - np.float32(0.5), exponent)
    return result


@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1), (7, 9), (33, 41)])
@pytest.mark.parametrize("iterations", [0, 1, 2, 6, 8, 33])
@pytest.mark.parametrize("count", [1, 2, 5, 8, 31, 255])
def test_fragment_options_exact(shape, iterations, count):
    image = source(shape)
    exact(
        native.fill_alpha_fragment_blur(image, 0.05, iterations, count),
        original_fragment(image, 0.05, iterations, count),
    )


@pytest.mark.parametrize("count", range(1, 256))
def test_all_fragment_counts_exact(count):
    image = source((9, 11), "sparse")
    exact(
        native.fill_alpha_fragment_blur(image, 0.05, 4, count),
        original_fragment(image, 0.05, 4, count),
    )


@pytest.mark.parametrize(
    "pattern",
    ["sparse", "opaque", "transparent", "signed_zero", "nonfinite", "dynamic"],
)
@pytest.mark.parametrize(
    "threshold", [0, 0.05, 0.5, 1, 2, float("nan"), float("inf"), -float("inf")]
)
def test_fragment_extreme_values_exact(pattern, threshold):
    image = source(pattern=pattern)
    before = image.copy()
    exact(
        native.fill_alpha_fragment_blur(image, threshold, 6, 5),
        original_fragment(image, threshold, 6, 5),
    )
    exact(image, before)


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 17), (19, 1), (7, 9), (8, 8), (17, 23), (41, 65)]
)
@pytest.mark.parametrize("radius", [0, 1, 7, 8, 9, 16, 32, 100000])
@pytest.mark.parametrize("antialias", [False, True])
def test_nearest_options_exact(shape, radius, antialias):
    image = source(shape)
    exact(
        native.fill_alpha_nearest_color(image, 0.05, radius, antialias),
        original_nearest(image, 0.05, radius, antialias),
    )


@pytest.mark.parametrize(
    "pattern",
    ["sparse", "opaque", "transparent", "signed_zero", "nonfinite", "dynamic"],
)
@pytest.mark.parametrize(
    "threshold", [0, 0.05, 0.5, 1, 2, float("nan"), float("inf"), -float("inf")]
)
@pytest.mark.parametrize("antialias", [False, True])
def test_nearest_extreme_values_exact(pattern, threshold, antialias):
    image = source(pattern=pattern)
    before = image.copy()
    exact(
        native.fill_alpha_nearest_color(image, threshold, 100000, antialias),
        original_nearest(image, threshold, 100000, antialias),
    )
    exact(image, before)


@pytest.mark.parametrize("radius", [0, 1, 7, 8, 9, 16, 24, 32])
@pytest.mark.parametrize("antialias", [False, True])
def test_radius_grid_word_seams_exact(radius, antialias):
    image = source((41, 1057), "transparent")
    for x in [8, 62 * 8, 63 * 8, 64 * 8, 65 * 8, 126 * 8, 128 * 8]:
        image[20, x, 3] = 1
    exact(
        native.fill_alpha_nearest_color(image, 0.05, radius, antialias),
        original_nearest(image, 0.05, radius, antialias),
    )


@pytest.mark.parametrize("antialias", [False, True])
@pytest.mark.parametrize("seed", range(12))
def test_spatial_ties_and_candidate_circles(antialias, seed):
    image = source((51, 71), "transparent", seed)
    # Lattice creates many exact ties, including across tree/cell boundaries.
    image[1::8, 1::10, 3] = 1
    exact(
        native.fill_alpha_nearest_color(image, 0.05, 100000, antialias),
        original_nearest(image, 0.05, 100000, antialias),
    )


@pytest.mark.parametrize("method", ["fragment", "nearest"])
@pytest.mark.parametrize(
    "layout", ["reversed", "transpose", "readonly", "unaligned", "broadcast"]
)
def test_foreign_buffers_are_owned_and_immutable(method, layout):
    image = source()
    if layout == "reversed":
        image = image[::-1, ::-1, ::-1]
    elif layout == "transpose":
        image = image.swapaxes(0, 1)
    elif layout == "readonly":
        image.setflags(write=False)
    elif layout == "unaligned":
        unaligned = np.ndarray(
            image.shape, np.float32, buffer=bytearray(image.nbytes + 1), offset=1
        )
        unaligned[:] = image
        image = unaligned
    elif layout == "broadcast":
        image = np.broadcast_to(image[:1, :1], image.shape)
    before = image.copy()
    if method == "fragment":
        actual = native.fill_alpha_fragment_blur(image, 0.05, 6, 5)
        expected = original_fragment(image, 0.05, 6, 5)
    else:
        actual = native.fill_alpha_nearest_color(image, 0.05, 100000, True)
        expected = original_nearest(image, 0.05, 100000, True)
    exact(actual, expected)
    exact(image, before)
    assert actual.flags.c_contiguous and actual.flags.aligned and actual.flags.writeable
    assert not np.shares_memory(actual, image)


@pytest.mark.parametrize("shape", [(0, 0, 4), (0, 3, 4), (7, 0, 4)])
def test_empty_images(shape):
    image = np.empty(shape, np.float32)
    exact(
        native.fill_alpha_fragment_blur(image, 0.05, 6, 5),
        original_fragment(image, 0.05, 6, 5),
    )
    exact(
        native.fill_alpha_nearest_color(image, 0.05, 100000, True),
        original_nearest(image, 0.05, 100000, True),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": -1, "fragment_count": 5},
        {"iterations": 2**32, "fragment_count": 5},
        {"iterations": 1.0, "fragment_count": 5},
        {"iterations": 6, "fragment_count": -1},
        {"iterations": 6, "fragment_count": 2**32},
        {"iterations": 6, "fragment_count": 0},
        {"iterations": 6, "fragment_count": 256},
    ],
)
def test_fragment_error_contract(kwargs):
    image = source((1, 1))
    with pytest.raises(BaseException) as original:
        original_fragment(image, 0.05, **kwargs)
    with pytest.raises(type(original.value)) as converted:
        native.fill_alpha_fragment_blur(image, 0.05, **kwargs)
    assert str(converted.value) == str(original.value)


@pytest.mark.parametrize("count", [0, 256, 2**32 - 1])
def test_zero_iterations_do_not_validate_fragment_count(count):
    image = source()
    exact(
        native.fill_alpha_fragment_blur(image, 0.05, 0, count),
        original_fragment(image, 0.05, 0, count),
    )


@pytest.mark.parametrize(
    "radius,antialias",
    [(-1, True), (2**32, True), (1.0, True), (0, 1), (0, None), (0, np.bool_(True))],
)
def test_nearest_error_contract(radius, antialias):
    image = source((1, 1))
    with pytest.raises(BaseException) as original:
        original_nearest(image, 0.05, radius, antialias)
    with pytest.raises(type(original.value)) as converted:
        native.fill_alpha_nearest_color(image, 0.05, radius, antialias)
    assert str(converted.value) == str(original.value)


@pytest.mark.parametrize("shape", [(4,), (3, 4), (3, 4, 3), (2, 3, 4, 4), (3, 4, 0)])
def test_invalid_shapes_rejected_before_c(shape, monkeypatch):
    def forbidden():
        raise AssertionError("C must not receive malformed arrays")

    monkeypatch.setattr(native, "_api", forbidden)
    image = np.zeros(shape, np.float32)
    with pytest.raises(ValueError):
        native.fill_alpha_fragment_blur(image, 0.05, 6, 5)
    with pytest.raises(ValueError):
        native.fill_alpha_nearest_color(image, 0.05, 100000, True)


def test_checked_c_boundaries():
    api = native._api()  # exercise the checked C ABI directly
    image = source((4, 8))
    output = np.empty_like(image)
    null = ct.POINTER(ct.c_float)()
    maximum = ct.c_size_t(-1).value
    for function, options in [
        (api.cn_alpha_fragment, (6, 5)),
        (api.cn_alpha_nearest, (100000, 1)),
    ]:
        assert function(ptr(image), ptr(output), maximum, 2, 0.05, *options) == 2
        assert function(null, ptr(output), 4, 8, 0.05, *options) == 1
        assert function(ptr(image), null, 4, 8, 0.05, *options) == 1
        assert function(ptr(image), ptr(image), 4, 8, 0.05, *options) == 1
        shifted = ct.cast(image.ctypes.data + 4, ct.POINTER(ct.c_float))
        assert function(ptr(image), shifted, 4, 7, 0.05, *options) == 1
        unaligned = ct.cast(image.ctypes.data + 1, ct.POINTER(ct.c_float))
        assert function(unaligned, ptr(output), 4, 7, 0.05, *options) == 1
    assert api.cn_alpha_fragment(ptr(image), ptr(output), 4, 8, 0.05, 1, 0) == 1
    assert api.cn_alpha_nearest(ptr(image), ptr(output), 4, 8, 0.05, 0, 2) == 1


def test_parallel_calls_are_exact_and_isolated():
    images = [
        source((73, 83), "random" if i % 2 else "sparse", 730 + i) for i in range(12)
    ]
    originals = [im.copy() for im in images]
    expected_fragment = [original_fragment(im, 0.05, 6, 5) for im in images]
    expected_nearest = [original_nearest(im, 0.05, 100000, True) for im in images]

    def run(i):
        im = images[i % len(images)]
        if i % 2:
            exact(
                native.fill_alpha_fragment_blur(im, 0.05, 6, 5),
                expected_fragment[i % len(images)],
            )
        else:
            exact(
                native.fill_alpha_nearest_color(im, 0.05, 100000, True),
                expected_nearest[i % len(images)],
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(48)))
    for actual, expected in zip(images, originals, strict=True):
        exact(actual, expected)


def test_valid_parameters_never_call_rust(monkeypatch):
    import chainner_ext

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Every valid Fill Alpha option must run in C")

    monkeypatch.setattr(chainner_ext, "fill_alpha_fragment_blur", forbidden)
    monkeypatch.setattr(chainner_ext, "fill_alpha_nearest_color", forbidden)
    image = source()
    for iterations, count in [(0, 0), (0, 256), (1, 1), (6, 5), (3, 255)]:
        exact(
            native.fill_alpha_fragment_blur(image, 0.05, iterations, count),
            original_fragment(image, 0.05, iterations, count),
        )
    for radius in [0, 1, 8, 100000, 2**32 - 1]:
        for antialias in [False, True]:
            exact(
                native.fill_alpha_nearest_color(image, 0.05, radius, antialias),
                original_nearest(image, 0.05, radius, antialias),
            )
