"""Exact CPU oracle checks for every filtered resize and gamma path.

The frozen reference imports installed chainner_ext 0.3.10. It is never used
by production: these comparisons cover the replacement coefficient builder,
both convolution passes, overshoot clipping, and before/after gamma stages.
"""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from test_resample_ops import LAYOUTS, image, reference, with_layout

from nodes.impl import native_resample as native
from nodes.impl import resize as current
from nodes.impl.native import ptr

FILTERS = tuple(range(1, 12))


def exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(
        actual[finite].view(np.uint32), expected[finite].view(np.uint32)
    )


def compare(source, dimensions, filter_id, gamma=False, separate=True):
    before = source.copy()
    with np.errstate(all="ignore"):
        expected = reference.resize(
            source, dimensions, reference.ResizeFilter(filter_id), separate, gamma
        )
        actual = current.resize(
            source, dimensions, current.ResizeFilter(filter_id), separate, gamma
        )
    exact(actual, expected)
    exact(source, before)
    assert not np.shares_memory(actual, source)
    assert actual.flags.writeable
    return actual


@pytest.mark.parametrize("filter_id", FILTERS)
@pytest.mark.parametrize("channels", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("gamma", [False, True])
@pytest.mark.parametrize(
    ("shape", "dimensions"),
    [((1, 1), (3, 7)), ((1, 17), (5, 1)), ((13, 1), (3, 19)), ((9, 11), (7, 5))],
)
def test_complete_filters(filter_id, channels, layout, gamma, shape, dimensions):
    compare(image(channels, layout, shape), dimensions, filter_id, gamma)


@pytest.mark.parametrize("filter_id", FILTERS)
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("gamma", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_nonfinite_and_gamma_tail(filter_id, channels, gamma, layout):
    values = np.array(
        [
            0,
            -0.0,
            -1,
            2,
            np.nan,
            np.inf,
            -np.inf,
            np.finfo(np.float32).tiny,
            np.nextafter(np.float32(0), np.float32(1)),
            0.5,
            1,
            0.125,
            0.75,
        ],
        dtype=np.float32,
    )
    source = np.resize(values, (3, 13, channels)).copy()
    compare(with_layout(source, layout), (31, 7), filter_id, gamma)


@pytest.mark.parametrize("filter_id", FILTERS)
@pytest.mark.parametrize("gamma", [False, True])
def test_coefficient_recycling_across_axes(filter_id, gamma):
    # Width and height use a shared cache, keyed by rounded float32 offsets.
    # Large destination/source ratios produce repeated keys; axis dimensions
    # differ so the second axis must reuse the first axis' matching entries.
    compare(image(3, shape=(19, 17)), (173, 211), filter_id, gamma)


@pytest.mark.parametrize("filter_id", FILTERS)
@pytest.mark.parametrize("gamma", [False, True])
def test_varied_ratios_and_image_ranges(filter_id, gamma):
    rng = np.random.default_rng(705)
    for index in range(40):
        h, w, out_h, out_w = (int(v) for v in rng.integers(1, 97, 4))
        channels = index % 5
        shape = (h, w) if channels == 0 else (h, w, channels)
        data = rng.random(shape, dtype=np.float32)
        if index % 4 == 0:
            data = data * np.float32(5) - np.float32(2)
        elif index % 4 == 1:
            data.fill(0)
            data[0, 0] = 1
            data[-1, -1] = 0.5
        elif index % 4 == 2:
            data[::2] = np.finfo(np.float32).max
            data[1::2] = -np.finfo(np.float32).max
        compare(
            with_layout(data, LAYOUTS[index % len(LAYOUTS)]),
            (out_w, out_h),
            filter_id,
            gamma,
            separate=bool(index % 2),
        )


@pytest.mark.parametrize("filter_id", [-1, 0, *FILTERS])
@pytest.mark.parametrize("dimensions", [(0, 7), (5, 0), (0, 0)])
@pytest.mark.parametrize("channels", [0, 2, 4])
@pytest.mark.parametrize("gamma", [False, True])
def test_empty_destinations(filter_id, dimensions, channels, gamma):
    compare(image(channels), dimensions, filter_id, gamma, separate=False)


def test_concurrent_filters_and_gamma():
    jobs = [
        (image(c, shape=(143, 151), seed=f + c), (179, 191), f, bool(f % 2))
        for f in FILTERS
        for c in (1, 3, 4)
    ]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda args: compare(*args), jobs))


@pytest.mark.parametrize("filter_id", FILTERS)
def test_no_rust_resize_dispatch(monkeypatch, filter_id):
    import chainner_ext

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Filtered resize called the old Rust implementation")

    monkeypatch.setattr(chainner_ext, "resize", forbidden)
    actual = current.resize(
        image(4), (19, 13), current.ResizeFilter(filter_id), False, True
    )
    assert actual.shape == (13, 19, 4)


@pytest.mark.parametrize("dtype", [np.float64, np.int32, np.uint8, ">f4"])
def test_reject_unsupported_dtype(dtype):
    with pytest.raises(TypeError):
        native.filtered(np.ones((3, 5, 3), dtype), (7, 9), 2, False)


@pytest.mark.parametrize("shape", [(3,), (3, 5, 6), (0, 5, 3), (3, 5, 1, 1)])
def test_reject_malformed_shape(shape):
    with pytest.raises(ValueError):
        native.filtered(np.ones(shape, np.float32), (7, 9), 2, False)


@pytest.mark.parametrize("dimensions", [(-1, 2), (2**32, 1), (2**30, 2**30)])
def test_reject_malformed_dimensions(dimensions):
    with pytest.raises((OverflowError, ValueError)):
        native.filtered(image(3), dimensions, 2, False)


def test_c_rejects_dimensions_before_dereference():
    # Deliberately tiny buffers with impossible descriptors must be rejected
    # before memory access/allocation, including both intermediate dimensions.
    source, out = np.zeros(1, np.float32), np.full(1, -17, np.float32)
    dll = native._api()  # verify the C boundary directly
    args = [ptr(source), ptr(out), 1, 1, 1, 1, 1, 2, 0, 0]
    for index, value, expected in (
        (2, 0, 1),
        (3, 0, 1),
        (4, 0, 1),
        (4, 5, 1),
        (5, 0, 1),
        (6, 0, 1),
        (7, 0, 1),
        (7, 12, 1),
        (8, 2, 1),
        (9, 2, 1),
        (2, 2**31, 2),
        (3, 2**31, 2),
        (5, 2**32, 2),
        (6, 2**32, 2),
    ):
        modified = args.copy()
        modified[index] = value
        assert dll.cn_resample_filtered(*modified) == expected
        assert out[0] == -17
    assert dll.cn_resample_filtered(None, *args[1:]) == 1
    assert dll.cn_resample_filtered(args[0], None, *args[2:]) == 1
    if ct.sizeof(ct.c_size_t) == 8:
        assert (
            dll.cn_resample_filtered(
                ptr(source), ptr(out), 2**31 - 1, 2**31 - 1, 4, 1, 1, 2, 0, 0
            )
            == 2
        )
