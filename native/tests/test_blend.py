"""Frozen blend algorithms with an explicit F26 integer-conversion adapter."""

import ctypes
import itertools

import numpy as np
import pytest
import reference_blend as reference

from nodes.impl.blend import (
    BlendMode,
    ImageBlender,
    blend_images,
    blend_mode_normalized,
)
from nodes.impl.native import lib, ptr
from nodes.impl.native_blend import composite

# The modes of the frozen snapshot; test_blend_non_separable.py has the others.
SEPARABLE = [BlendMode(mode.value) for mode in reference.BlendMode]


@pytest.fixture(autouse=True)
def saturating_xor_reference(monkeypatch):
    # The frozen XOR body imports today's to_uint8. Give its test namespace an
    # independent NumPy conversion with the approved saturation and rounding.
    def to_uint8(img, normalized=False):
        return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)

    monkeypatch.setattr(reference, "to_uint8", to_uint8)


def assert_parity(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.float32
    # Strict float32 operation ordering gives exact bits on the supported Windows
    # build, the sign of zero included. NaNs must be in the same places; which
    # payload survives where two NaNs meet is a don't-care (ARCHITECTURE section 7).
    nan = np.isnan(actual)
    np.testing.assert_array_equal(nan, np.isnan(expected))
    np.testing.assert_array_equal(
        actual.view(np.uint32)[~nan], expected.view(np.uint32)[~nan]
    )


@pytest.mark.parametrize(
    ("mode", "overlay", "base", "expected"),
    [
        (BlendMode.COLOR_BURN, 0.0, 1.0, 1.0),
        (BlendMode.COLOR_BURN, 0.0, 0.5, 0.0),
        (BlendMode.COLOR_BURN, 0.5, 1.0, 1.0),
        (BlendMode.COLOR_DODGE, 1.0, 0.0, 0.0),
        (BlendMode.COLOR_DODGE, 1.0, 0.5, 1.0),
        (BlendMode.COLOR_DODGE, 0.5, 0.0, 0.0),
        (BlendMode.VIVID_LIGHT, 0.0, 1.0, 1.0),
        (BlendMode.VIVID_LIGHT, 1.0, 0.0, 0.0),
    ],
)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_burn_keeps_white_and_dodge_keeps_black(mode, overlay, base, expected, channels):
    # The W3C compositing spec's and ISO 32000-2's definitions at their 0 / 0 corners.
    shape = (1, 1) if channels == 1 else (1, 1, channels)
    a = np.full(shape, overlay, np.float32)
    b = np.full(shape, base, np.float32)
    if channels == 4:
        a[:, :, 3] = b[:, :, 3] = 1
    result = blend_images(a, b, mode)
    assert (result if channels == 1 else result[:, :, :3]).ravel().tolist() == [expected] * min(channels, 3)


@pytest.mark.parametrize("base_alpha", [None, 1.0, 0.5])
def test_divide_mixes_its_clipped_quotient(base_alpha):
    # A black overlay at 2 % over gray: 2 % of white and the rest of the base, source-over. Unclipped, the quotient
    # 0.5 / 0.0001 turned every soft edge of such an overlay white.
    overlay = np.array([[[0, 0, 0, 0.02]]], np.float32)
    base = np.full((1, 1, 3), 0.5, np.float32)
    o_a, b_a = 0.02, 1.0
    if base_alpha is not None:
        b_a = base_alpha
        base = np.dstack([base, np.full((1, 1), base_alpha, np.float32)])
    result = blend_images(overlay, base, BlendMode.DIVIDE)
    final_a = o_a + b_a - o_a * b_a
    both = o_a * b_a
    expected = (both * 1.0 + (o_a - both) * 0.0 + (b_a - both) * 0.5) / final_a
    np.testing.assert_allclose(result[0, 0, :3], expected, atol=1e-6)


@pytest.mark.parametrize("shape", [(2, 2, 3, 0), (2, 2, 3, 1), (0, 2, 3), (2, 0, 3)])
@pytest.mark.parametrize("bad_overlay", [True, False])
def test_rejects_malformed_images_before_native_dispatch(
    shape, bad_overlay, monkeypatch
):
    from nodes.impl import native_blend

    def forbidden(*args):
        raise AssertionError("Malformed buffers must never reach C")

    monkeypatch.setattr(native_blend._library, "cn_blend_images", forbidden)
    malformed = np.empty(shape, dtype=np.float32)
    valid = np.ones((*shape[:2], 3), dtype=np.float32)
    a, b = (malformed, valid) if bad_overlay else (valid, malformed)
    with pytest.raises(ValueError, match="Invalid blend image shape"):
        blend_images(a, b, BlendMode.MULTIPLY)


def test_adapter_checks_declared_channels_and_both_spatial_shapes():
    image = np.ones((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="channel count"):
        composite(image, image, 1, 4, 3)
    with pytest.raises(ValueError, match="same size"):
        composite(image, image[:1], 1, 3, 3)


def image(rng, channels, layout, height=7, width=11):
    shape = (height * 2, width * 2) + (() if channels == 1 else (channels,))
    values = rng.random(shape, dtype=np.float32)
    values = values[::2, ::2]
    if layout == "contiguous":
        values = values.copy()
    elif layout == "reversed":
        values = values[::-1, ::-1]
    elif layout == "readonly":
        values.setflags(write=False)
    return values


@pytest.mark.parametrize("mode", SEPARABLE)
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed", "readonly"])
def test_raw_modes(mode, channels, layout):
    rng = np.random.default_rng(731)
    a, b = image(rng, channels, layout), image(rng, channels, layout)
    before_a, before_b = a.copy(), b.copy()
    expected = reference.ImageBlender().apply_blend(
        a, b, reference.BlendMode(mode.value)
    )
    actual = ImageBlender().apply_blend(a, b, mode)
    assert_parity(actual, expected)
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)


@pytest.mark.parametrize("mode", SEPARABLE)
@pytest.mark.parametrize("channels", list(itertools.product([1, 3, 4], repeat=2)))
@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed", "readonly"])
def test_all_composition_paths(mode, channels, layout):
    rng = np.random.default_rng(883)
    a, b = image(rng, channels[0], layout), image(rng, channels[1], layout)
    before_a, before_b = a.copy(), b.copy()
    expected = reference.blend_images(a, b, reference.BlendMode(mode.value))
    actual = blend_images(a, b, mode)
    assert_parity(actual, expected)
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)


@pytest.mark.parametrize("mode", SEPARABLE)
@pytest.mark.parametrize("channels", list(itertools.product([1, 3, 4], repeat=2)))
def test_boundaries_and_transparency(mode, channels):
    # Exercise branch cutoffs, epsilon denominators, and exactly transparent and
    # opaque alpha, including all combinations of foreground/background alpha.
    levels = np.array([0, 0.00001, 0.0001, 0.5, 0.9999, 1], dtype=np.float32)
    a, b = np.meshgrid(levels, levels)
    if channels[0] > 1:
        a = np.stack([np.roll(a, i, axis=0) for i in range(channels[0])], axis=2)
    if channels[1] > 1:
        b = np.stack([np.roll(b, i, axis=1) for i in range(channels[1])], axis=2)
    with np.errstate(all="ignore"):
        expected = reference.blend_images(a, b, reference.BlendMode(mode.value))
        actual = blend_images(a, b, mode)
    assert_parity(actual, expected)


@pytest.mark.parametrize("mode", SEPARABLE)
@pytest.mark.parametrize("channels", list(itertools.product([1, 3, 4], repeat=2)))
def test_composition_signed_zero_and_specials(mode, channels):
    # Consult 11 D-17.1: every clip is np.clip(x, 0, 1), which keeps -0 and NaN.
    levels = np.array(
        [0.0, -0.0, 0.0001, 0.5, 1.0, -0.5, 1.5, np.nan, np.inf, -np.inf],
        dtype=np.float32,
    )
    a, b = np.meshgrid(levels, levels)
    if channels[0] > 1:
        a = np.stack([np.roll(a, i, axis=0) for i in range(channels[0])], axis=2)
    if channels[1] > 1:
        b = np.stack([np.roll(b, i, axis=1) for i in range(channels[1])], axis=2)
    with np.errstate(all="ignore"):
        expected = reference.blend_images(a, b, reference.BlendMode(mode.value))
        actual = blend_images(a, b, mode)
    assert_parity(actual, expected)


@pytest.mark.parametrize("mode", SEPARABLE)
def test_raw_nan_infinity_and_signed_zero(mode):
    levels = np.array(
        [0.0, -0.0, 0.0001, 0.5, 1.0, np.nan, np.inf, -np.inf], dtype=np.float32
    )
    a, b = np.meshgrid(levels, levels)
    with np.errstate(all="ignore"):
        expected = reference.ImageBlender().apply_blend(
            a, b, reference.BlendMode(mode.value)
        )
        actual = ImageBlender().apply_blend(a, b, mode)
    assert_parity(actual, expected)


def test_xor_bankers_rounding():
    values = ((np.arange(255, dtype=np.float32) + 0.5) / 255).reshape(15, 17)
    other = np.flip(values, axis=1)
    expected = reference.ImageBlender().apply_blend(
        values, other, reference.BlendMode.XOR
    )
    actual = ImageBlender().apply_blend(values, other, BlendMode.XOR)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(1, 1), (1, 11), (11, 1), (1, 1, 1)])
@pytest.mark.parametrize("mode", SEPARABLE)
def test_single_pixel_dimensions_and_grayscale_flattening(shape, mode):
    a, b = np.full(shape, 0.2, np.float32), np.full(shape, 0.7, np.float32)
    assert_parity(
        blend_images(a, b, mode),
        reference.blend_images(a, b, reference.BlendMode(mode.value)),
    )


def test_normal_alias_semantics():
    for shape in [(3, 7), (3, 7, 1), (3, 7, 3)]:
        a = np.ones(shape, dtype=np.float32)
        b = np.zeros(shape, dtype=np.float32)
        a.setflags(write=False)
        assert ImageBlender().apply_blend(a, b, BlendMode.NORMAL) is a
        result = blend_images(a, b, BlendMode.NORMAL)
        assert np.shares_memory(result, a)
        assert not result.flags.writeable


def test_normalization_metadata_unchanged():
    for mode in SEPARABLE:
        assert blend_mode_normalized(mode) == reference.blend_mode_normalized(
            reference.BlendMode(mode.value)
        )


def test_shape_and_channel_errors():
    with pytest.raises(AssertionError, match="same size"):
        blend_images(
            np.zeros((2, 3), np.float32), np.zeros((3, 2), np.float32), BlendMode.NORMAL
        )
    with pytest.raises(AssertionError, match="overlay layer"):
        blend_images(
            np.zeros((2, 3, 2), np.float32),
            np.zeros((2, 3), np.float32),
            BlendMode.NORMAL,
        )
    with pytest.raises(AssertionError, match="base layer"):
        blend_images(
            np.zeros((2, 3), np.float32),
            np.zeros((2, 3, 2), np.float32),
            BlendMode.NORMAL,
        )
    with pytest.raises(TypeError, match="float32"):
        blend_images(
            np.zeros((2, 3), np.float64), np.zeros((2, 3), np.float32), BlendMode.NORMAL
        )


def test_c_argument_validation():
    library = lib()
    a = np.ones(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    null = ctypes.POINTER(ctypes.c_float)()
    assert library.cn_blend_mode(null, ptr(a), ptr(out), 4, 0) == 1
    assert library.cn_blend_mode(ptr(a), ptr(a), ptr(out), 4, 23) == 1
    assert (
        library.cn_blend_mode(ptr(a), ptr(a), ptr(out), ctypes.c_size_t(-1).value, 0)
        == 2
    )
    assert library.cn_blend_images(ptr(a), ptr(a), ptr(out), 1, 2, 4, 0) == 1
    assert (
        library.cn_blend_images(
            ptr(a), ptr(a), ptr(out), ctypes.c_size_t(-1).value, 4, 4, 0
        )
        == 2
    )
    np.testing.assert_array_equal(out, 0)
