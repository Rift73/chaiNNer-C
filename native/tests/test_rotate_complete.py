"""Exact affine rotation against frozen helper + installed Pillow 9.2.0.

The original helper and original pixel conversions are bound by test_layout_ops;
the oracle therefore never follows a newly converted production helper.
"""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image
from test_layout_ops import LAYOUTS, MODULES, image

from nodes.impl import native_rotate
from nodes.impl.native import ptr

current, reference = MODULES["rotate"]


def compare(source, angle, interpolation, expand, fill, *, helper=False):
    before = source.copy()
    output = []
    with np.errstate(all="ignore"):
        for module in (reference, current):
            fn = module.rotate if helper else module.rotate_node
            output.append(
                fn(
                    source,
                    angle,
                    module.RotationInterpolationMethod(interpolation),
                    module.RotateSizeChange(int(expand)),
                    module.FillColor(fill),
                )
            )
    expected, actual = output
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(source, before)
    assert not np.shares_memory(actual, source)
    assert actual.flags.writeable
    return actual


@pytest.mark.parametrize("interpolation", [0, 2, 3])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("fill", [-1, 0, 1])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize(
    "angle", [0.1, 13.7, 45, 89.9, 90, 149.3, 180, 269.9, 270, 359.9, -414.7]
)
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (9, 1), (7, 11)])
def test_all_affine_modes(interpolation, expand, fill, channels, angle, shape):
    compare(image(channels, shape=shape), angle, interpolation, expand, fill)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("interpolation", [0, 2, 3])
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("angle", [0, 37.1, 90, 180, 270])
def test_direct_helper_and_foreign_buffers(layout, interpolation, channels, angle):
    compare(image(channels, layout), angle, interpolation, True, -1, helper=True)


@pytest.mark.parametrize("interpolation", [0, 2, 3])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("fill", [-1, 0])
def test_two_channels(interpolation, expand, fill):
    source = image(2)
    source[::3, ::2, 1] = 0
    source[1::3, 1::2, 1] = 1 / 255
    compare(source, 27.2, interpolation, expand, fill)


@pytest.mark.parametrize("interpolation", [0, 2, 3])
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.int16, np.float32, np.float64]
)
def test_quantization_and_dtype(interpolation, channels, dtype):
    source = image(channels)
    if np.issubdtype(dtype, np.integer):
        source = (source * np.iinfo(dtype).max).astype(dtype)
    else:
        source = source.astype(dtype)
        source.flat[:7] = [np.nan, np.inf, -np.inf, -0.0, -1, 2, 0.5 / 255]
    compare(source, 43.4, interpolation, True, 0)


@pytest.mark.parametrize("interpolation", [0, 2, 3])
@pytest.mark.parametrize("angle", [0.000001, 0.1, 0.7, 89.9, 179.9, 270.1])
@pytest.mark.parametrize("shape", [(1, 32769), (32769, 1)])
def test_large_coordinate_float_path(interpolation, angle, shape):
    # Thin images cross Pillow's signed 16.16 range without a large fixture.
    compare(image(1, shape=shape), angle, interpolation, False, -1)


@pytest.mark.parametrize("interpolation", [2, 3])
@pytest.mark.parametrize("channels", [2, 4])
def test_all_alpha_bytes(interpolation, channels):
    source = np.empty((17, 256, channels), np.uint8)
    source[..., -1] = np.arange(256, dtype=np.uint8)
    for channel in range(channels - 1):
        source[..., channel] = (
            np.arange(256, dtype=np.uint16) * (channel + 2) % 256
        ).astype(np.uint8)
    compare(source, 11.1, interpolation, True, -1)


@pytest.mark.parametrize("interpolation", [0, 2, 3])
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_rounding_to_identity_is_not_transpose_shortcut(interpolation, channels):
    # A nonzero angle can round to an identity matrix. Pillow still applies
    # interpolation's byte alpha conversions in this case.
    compare(image(channels), 1e-15, interpolation, False, -1)


@pytest.mark.parametrize("interpolation", [0, 2, 3])
def test_single_channel_axis_promoted_by_transparent_fill(interpolation):
    compare(image(1)[..., None], 23.7, interpolation, True, 1)


def test_concurrent_affine_sampling():
    jobs = [
        (image(channels, shape=(271, 257)), 37.1 + channels, interpolation, True, fill)
        for channels in (1, 3, 4)
        for interpolation in (0, 2, 3)
        for fill in (-1, 0, 1)
    ]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda args: compare(*args), jobs))


def test_oracle_is_frozen_pillow_and_production_does_not_call_it(monkeypatch):
    assert reference.rotate.__code__.co_filename == "original_rotate"
    expected = reference.rotate_node(
        image(4),
        23.4,
        reference.RotationInterpolationMethod.CUBIC,
        reference.RotateSizeChange.EXPAND,
        reference.FillColor.AUTO,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("C rotate called a Pillow image operation")

    monkeypatch.setattr(Image.Image, "rotate", forbidden)
    monkeypatch.setattr(Image.Image, "transform", forbidden)
    actual = current.rotate_node(
        image(4),
        23.4,
        current.RotationInterpolationMethod.CUBIC,
        current.RotateSizeChange.EXPAND,
        current.FillColor.AUTO,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(3, 5, 1), (3, 5, 5), (3, 5, 3, 1)])
def test_original_unsupported_shapes(shape):
    for module in (reference, current):
        with pytest.raises((TypeError, ValueError)):
            module.rotate(
                np.ones(shape, np.float32),
                13.7,
                module.RotationInterpolationMethod.CUBIC,
                module.RotateSizeChange.EXPAND,
                module.FillColor.AUTO,
            )


@pytest.mark.parametrize("angle", [np.nan, np.inf, -np.inf])
def test_nonfinite_angles(angle):
    for module in (reference, current):
        with pytest.raises(ValueError):
            module.rotate(
                image(3),
                angle,
                module.RotationInterpolationMethod.CUBIC,
                module.RotateSizeChange.EXPAND,
                module.FillColor.AUTO,
            )


def test_c_validation():
    dll = native_rotate._api()  # verify raw pointer boundary validation
    source = np.zeros(1, np.uint8)
    out = np.full(1, -17, np.float32)
    matrix = (ct.c_double * 6)(1, 0, 0, 0, 1, 0)
    fill = (ct.c_uint8 * 4)(0, 0, 0, 0)
    arguments: list[object] = [
        source.ctypes.data_as(ct.POINTER(ct.c_uint8)),
        ptr(out),
        1,
        1,
        1,
        1,
        1,
        matrix,
        3,
        fill,
        -1,
    ]
    for index, value, status in (
        (0, None, 1),
        (1, None, 1),
        (2, 0, 1),
        (3, 0, 1),
        (4, 0, 1),
        (4, 5, 1),
        (5, 0, 1),
        (6, 0, 1),
        (7, None, 1),
        (8, 1, 1),
        (9, None, 1),
        (10, 4, 1),
        (2, 2**31, 2),
        (3, 2**31, 2),
        (5, 2**31, 2),
        (6, 2**31, 2),
    ):
        bad = arguments.copy()
        bad[index] = value
        assert dll.cn_rotate_affine_u8(*bad) == status
        assert out[0] == -17
    for coefficient in (np.nan, np.inf, -np.inf, 2):
        matrix[0] = coefficient
        assert dll.cn_rotate_affine_u8(*arguments) == 1
        assert out[0] == -17
