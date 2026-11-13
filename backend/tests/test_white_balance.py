"""White Balance: a planted cast on a neutral scene comes off in each mode; gray, RGB and RGBA;
transparent pixels; a linear image; black, one-color and tiny inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nodes.impl.color.color import Color
from nodes.impl.linear_light import LightTransfer, encode_inplace, linearize
from nodes.impl.native_analysis import channel_sums, scale_channels
from packages.chaiNNer_standard.image_adjustment import adjustments_group
from packages.chaiNNer_standard.image_adjustment.adjustments.white_balance import (
    WhiteBalanceMode,
    planckian_bgr,
    white_balance_node,
)

M = WhiteBalanceMode
LUMINANCE = np.array([0.0722, 0.7152, 0.2126])
NODE = next(
    node
    for node in adjustments_group.nodes
    if node.schema_id == "chainner:image:white_balance"
)
# A warm light, as BGR factors in linear light.
LIGHT = np.array([0.55, 0.9, 1.25], dtype=np.float32)
GRAY = Color.bgr((0.5, 0.5, 0.5))


def balance(
    img: np.ndarray,
    mode: WhiteBalanceMode = M.GRAY_WORLD,
    percent: float = 1.0,
    color: Color = GRAY,
    temperature: int = 6500,
    tint: float = 0,
    linear_light: bool = True,
    transfer: LightTransfer = LightTransfer.SRGB,
) -> np.ndarray:
    return white_balance_node(
        img, mode, percent, color, temperature, tint, linear_light, transfer
    )


def gray_world_scene(height: int = 48, width: int = 64) -> np.ndarray:
    """Linear BGR values whose mean color is neutral."""
    rng = np.random.default_rng(11)
    scene = rng.uniform(0.05, 0.6, (height, width, 3)).astype(np.float32)
    scene *= scene.mean() / scene.mean(axis=(0, 1))
    return scene


def encoded(linear: np.ndarray) -> np.ndarray:
    return encode_inplace(np.array(linear, order="C"), LightTransfer.SRGB)


def luminance(bgr: np.ndarray) -> float:
    return float(bgr @ LUMINANCE)


def test_gray_world_takes_a_planted_cast_off():
    scene = gray_world_scene()
    result = balance(encoded(scene * LIGHT))
    recovered = linearize(result, LightTransfer.SRGB)
    # The scene comes back up to one factor: the brightness of the cast scene stays.
    factor = luminance(np.mean(scene * LIGHT, axis=(0, 1))) / luminance(
        np.mean(scene, axis=(0, 1))
    )
    assert np.allclose(recovered, scene * factor, atol=2e-4)
    mean = recovered.mean(axis=(0, 1))
    assert mean[0] == pytest.approx(mean[1], abs=1e-4)
    assert mean[2] == pytest.approx(mean[1], abs=1e-4)


def test_white_patch_takes_a_planted_cast_off():
    scene = np.clip(gray_world_scene(), 0, 0.5)
    # 1 % of the pixels, and no more, are a white surface.
    scene[:4, :8] = 0.9
    result = balance(encoded(scene * LIGHT), M.WHITE_PATCH, percent=1.0)
    recovered = linearize(result, LightTransfer.SRGB)
    white = recovered[:4, :8]
    assert np.allclose(white, white[0, 0, 1], atol=2e-4)
    expected = 0.9 * luminance(LIGHT)
    assert white[0, 0, 1] == pytest.approx(expected, abs=2e-4)
    assert np.allclose(
        recovered[4:], scene[4:] * LIGHT * expected / (0.9 * LIGHT), atol=2e-4
    )


def test_neutral_color_makes_that_color_gray():
    scene = gray_world_scene()
    scene[10, 20] = (0.2, 0.2, 0.2)
    cast = encoded(scene * LIGHT)
    picked = Color.bgr(cast[10, 20])
    result = balance(cast, M.NEUTRAL_COLOR, color=picked)
    pixel = linearize(result, LightTransfer.SRGB)[10, 20]
    assert pixel[0] == pytest.approx(pixel[1], abs=1e-4)
    assert pixel[2] == pytest.approx(pixel[1], abs=1e-4)
    assert pixel[1] == pytest.approx(0.2 * luminance(LIGHT), abs=1e-4)


def test_a_neutral_color_picked_from_rgba_counts_without_its_alpha():
    # Pick Color gives four channels on an RGBA image.
    img = np.array([[[0.2, 0.3, 0.5]]], np.float32)
    three = balance(img, M.NEUTRAL_COLOR, color=Color.bgr((0.2, 0.3, 0.5)))
    four = balance(img, M.NEUTRAL_COLOR, color=Color.bgra((0.2, 0.3, 0.5, 0.25)))
    assert np.array_equal(three, four)
    assert NODE.inputs[3].enforce(Color.bgra((0.2, 0.3, 0.5, 0.25))).channels == 4


def test_gray_world_of_a_cast_image_equals_gray_world_of_the_image():
    # Whatever the scene's own color, a cast changes only the mean the gains come from.
    rng = np.random.default_rng(12)
    scene = rng.uniform(0.02, 0.5, (32, 40, 3)).astype(np.float32) * [0.7, 1.0, 0.8]
    scene = scene.astype(np.float32)
    plain = linearize(balance(encoded(scene)), LightTransfer.SRGB)
    cast = linearize(balance(encoded(scene * LIGHT)), LightTransfer.SRGB)
    factor = np.median(cast / plain)
    assert np.allclose(cast, plain * factor, atol=3e-4)


@pytest.mark.parametrize(
    ("color", "name"),
    (
        ((0.01, 0.5, 0.9), "blue"),
        ((0.5, 0.01, 0.9), "green"),
        ((0.5, 0.9, 0.01), "red"),
    ),
)
def test_a_neutral_color_with_an_almost_empty_channel_is_refused(
    color: tuple[float, float, float], name: str
):
    # 0.01 is 0.0008 in linear light: its factor would be in the hundreds.
    img = np.full((2, 2, 3), 0.25, np.float32)
    with pytest.raises(ValueError, match=f"almost no {name}.*multiplied by [0-9]{{3}}"):
        balance(img, M.NEUTRAL_COLOR, color=Color.bgr(color))


def test_the_refusal_starts_at_a_64th_of_the_luminance():
    # With green and red at 0.5, the luminance is 0.4644 and its 64th 0.00726.
    img = np.full((2, 2, 3), 0.1, np.float32)
    below = Color.bgr((0.0071, 0.5, 0.5))
    with pytest.raises(ValueError, match="almost no blue"):
        balance(img, M.NEUTRAL_COLOR, color=below, linear_light=False)
    # Above it there is no cap: the factor is what the color asks for.
    above = balance(
        img, M.NEUTRAL_COLOR, color=Color.bgr((0.0073, 0.5, 0.5)), linear_light=False
    )
    luminance = 0.0722 * 0.0073 + 0.9278 * 0.5
    assert above[0, 0, 0] == pytest.approx(0.1 * luminance / 0.0073, rel=1e-4)
    assert above[0, 0, 0] > 6


def test_a_channel_without_any_light_is_not_refused():
    # It has no factor to refuse: its samples stay as they are.
    img = np.array([[[0.3, 0.5, 0.5]]], np.float32)
    result = balance(
        img, M.NEUTRAL_COLOR, color=Color.bgr((0.0, 0.5, 0.5)), linear_light=False
    )
    assert result[0, 0, 0] == pytest.approx(0.3, abs=1e-6)


@pytest.mark.parametrize("mode", (M.GRAY_WORLD, M.WHITE_PATCH))
def test_an_image_with_almost_none_of_a_channel_is_refused(mode: WhiteBalanceMode):
    img = np.full((4, 5, 3), (0.001, 0.2, 0.9), np.float32)
    with pytest.raises(ValueError, match="almost no blue"):
        balance(img, mode, linear_light=False)


@pytest.mark.parametrize("temperature", range(2000, 15001, 500))
def test_the_whole_temperature_slider_is_accepted(temperature: int):
    img = np.full((2, 2, 3), 0.25, np.float32)
    result = balance(img, M.TEMPERATURE, temperature=temperature)
    assert np.all(np.isfinite(result))


def test_a_temperature_and_tint_beyond_the_limit_are_refused():
    img = np.full((2, 2, 3), 0.25, np.float32)
    with pytest.raises(ValueError, match="almost no blue"):
        balance(img, M.TEMPERATURE, temperature=2000, tint=100)


# CIE 1931 chromaticity of black bodies, from published tables of the Planckian locus.
@pytest.mark.parametrize(
    ("temperature", "x", "y"),
    ((3000, 0.4369, 0.4041), (5000, 0.3451, 0.3516), (10000, 0.2807, 0.2884)),
)
def test_the_black_body_color_has_its_chromaticity(
    temperature: int, x: float, y: float
):
    b, g, r = planckian_bgr(temperature)
    linear_to_xyz = np.array(
        [
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ]
    )
    big_x, big_y, big_z = linear_to_xyz @ np.array([r, g, b])
    total = big_x + big_y + big_z
    assert big_x / total == pytest.approx(x, abs=5e-4)
    assert big_y / total == pytest.approx(y, abs=5e-4)


@pytest.mark.parametrize("temperature", (2000, 3000, 5500, 9000, 15000))
def test_temperature_takes_the_light_of_that_temperature_off(temperature: int):
    # A gray card under the light: its color is the light's, relative to the neutral 6500 K.
    light = (planckian_bgr(temperature) / planckian_bgr(6500)).astype(np.float32)
    light *= 0.2 / light.max()
    card = np.full((3, 3, 3), light, np.float32)
    result = balance(encoded(card), M.TEMPERATURE, temperature=temperature)
    pixel = linearize(result, LightTransfer.SRGB)[1, 1]
    assert pixel[0] == pytest.approx(pixel[1], rel=2e-2)
    assert pixel[2] == pytest.approx(pixel[1], rel=2e-2)
    assert pixel[1] == pytest.approx(luminance(light), rel=2e-2)


def test_the_temperature_direction():
    img = np.full((2, 2, 3), 0.25, np.float32)
    warm = balance(img, M.TEMPERATURE, temperature=9000)[0, 0]
    cool = balance(img, M.TEMPERATURE, temperature=3000)[0, 0]
    assert warm[2] > 0.25 > warm[0]
    assert cool[0] > 0.25 > cool[2]


def test_tint_scales_green_by_its_stops():
    img = np.full((2, 2, 3), 0.25, np.float32)
    result = balance(img, M.TEMPERATURE, tint=100, linear_light=False)[0, 0]
    # The reference's green doubles: its luminance is 1.7152 and green's factor half of that.
    assert result[1] == pytest.approx(0.25 * 1.7152 / 2, abs=1e-6)
    assert result[0] == pytest.approx(0.25 * 1.7152, abs=1e-6)
    magenta = balance(img, M.TEMPERATURE, tint=-100)[0, 0]
    assert magenta[1] > magenta[0]


@pytest.mark.parametrize(
    ("mode", "options"),
    (
        (M.GRAY_WORLD, {}),
        (M.WHITE_PATCH, {}),
        (M.NEUTRAL_COLOR, {"color": Color.bgr((0.1, 0.5, 0.9))}),
        (M.TEMPERATURE, {"temperature": 3000, "tint": 20}),
    ),
)
@pytest.mark.parametrize("size", ((1, 1), (1, 4), (4, 1), (5, 6)))
def test_grayscale_stays_as_it_is(
    mode: WhiteBalanceMode, options: dict[str, Any], size: tuple[int, int]
):
    img = np.random.default_rng(5).random(size, dtype=np.float32)
    assert np.array_equal(balance(img, mode, **options), img)


@pytest.mark.parametrize("mode", M)
@pytest.mark.parametrize("linear_light", (True, False))
def test_a_neutral_image_stays_as_it_is(mode: WhiteBalanceMode, linear_light: bool):
    # Every pixel is gray: no mode with its default settings has a cast to take off.
    value = np.random.default_rng(6).random((8, 9), dtype=np.float32)
    img = np.dstack((value, value, value))
    assert np.array_equal(balance(img, mode, linear_light=linear_light), img)


@pytest.mark.parametrize("mode", (M.GRAY_WORLD, M.WHITE_PATCH))
def test_a_one_pixel_image_becomes_gray_of_its_luminance(mode: WhiteBalanceMode):
    img = np.array([[[0.2, 0.4, 0.8]]], np.float32)
    result = balance(img, mode, percent=50, linear_light=False)
    expected = luminance(img[0, 0])
    assert np.allclose(result, expected, atol=1e-6)
    assert result.shape == (1, 1, 3)


def test_alpha_stays_and_transparent_pixels_do_not_count():
    scene = gray_world_scene(8, 8)
    garbage = np.array([1.0, 0.0, 0.0], np.float32)
    rgba = np.zeros((16, 8, 4), np.float32)
    rgba[:8, :, :3] = scene * LIGHT
    rgba[:8, :, 3] = 1
    rgba[8:, :, :3] = garbage
    rgba[8:, :, 3] = 0
    opaque = balance(rgba[:8, :, :3], linear_light=False)
    for mode in (M.GRAY_WORLD, M.WHITE_PATCH):
        whole = balance(rgba, mode, percent=25, linear_light=False)
        assert np.array_equal(whole[:, :, 3], rgba[:, :, 3])
        expected = balance(rgba[:8, :, :3], mode, percent=25, linear_light=False)
        assert np.allclose(whole[:8, :, :3], expected, atol=1e-6)
    assert not np.allclose(opaque, rgba[:8, :, :3], atol=1e-3)


def test_half_transparent_pixels_count_by_their_alpha():
    # Mean of (0.2 at alpha 1) and (0.8 at alpha 0.25) in blue, green stays 0.5 and red 0.5.
    rgba = np.array([[[0.2, 0.5, 0.5, 1.0], [0.8, 0.5, 0.5, 0.25]]], dtype=np.float32)
    result = balance(rgba, linear_light=False)
    blue_mean = (0.2 + 0.8 * 0.25) / 1.25
    reference = np.array([blue_mean, 0.5, 0.5])
    gain = luminance(reference) / blue_mean
    assert result[0, 0, 0] == pytest.approx(0.2 * gain, abs=1e-6)
    assert result[0, 1, 0] == pytest.approx(0.8 * gain, abs=1e-6)
    assert result[0, 0, 3] == 1 and result[0, 1, 3] == 0.25


@pytest.mark.parametrize("mode", (M.GRAY_WORLD, M.WHITE_PATCH))
def test_a_fully_transparent_image_stays_as_it_is(mode: WhiteBalanceMode):
    rgba = np.random.default_rng(7).random((4, 5, 4), dtype=np.float32)
    rgba[:, :, 3] = 0
    assert np.array_equal(balance(rgba, mode), rgba)


@pytest.mark.parametrize("mode", (M.GRAY_WORLD, M.WHITE_PATCH))
def test_black_stays_black(mode: WhiteBalanceMode):
    img = np.zeros((4, 5, 3), np.float32)
    assert np.array_equal(balance(img, mode), img)


@pytest.mark.parametrize("mode", (M.GRAY_WORLD, M.WHITE_PATCH))
def test_a_channel_without_light_gives_finite_values(mode: WhiteBalanceMode):
    img = np.zeros((4, 5, 3), np.float32)
    img[:, :, 2] = 0.5
    result = balance(img, mode)
    assert np.all(np.isfinite(result))
    assert np.all(result[:, :, :2] == 0)
    assert np.all(result[:, :, 2] > 0)


def test_linear_light_off_scales_the_values_as_they_are():
    img = np.array([[[0.1, 0.2, 0.4], [0.3, 0.4, 0.2]]], np.float32)
    result = balance(img, linear_light=False)
    mean = img.mean(axis=(0, 1)).astype(np.float64)
    gains = luminance(mean) / mean
    assert np.allclose(result, img * gains.astype(np.float32), atol=1e-6)


def test_the_transfer_curve_is_used_for_the_neutral_color_too():
    # With the same curve for the image and the color, the color maps to gray for each curve.
    img = np.array([[[0.2, 0.3, 0.5]]], np.float32)
    color = Color.bgr((0.2, 0.3, 0.5))
    for transfer in LightTransfer:
        result = balance(img, M.NEUTRAL_COLOR, color=color, transfer=transfer)
        assert result[0, 0, 0] == pytest.approx(result[0, 0, 1], abs=1e-5)
        assert result[0, 0, 2] == pytest.approx(result[0, 0, 1], abs=1e-5)


def test_the_input_is_not_written():
    img = encoded(gray_world_scene(8, 8) * LIGHT)
    before = img.copy()
    img.flags.writeable = False
    for linear_light in (True, False):
        balance(img, linear_light=linear_light)
    assert np.array_equal(img, before)


def test_highlights_reach_the_image_clipped():
    img = np.array([[[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]]], np.float32)
    enforced = NODE.outputs[0].enforce(
        balance(img, M.NEUTRAL_COLOR, color=Color.bgr((0.1, 0.5, 0.9)))
    )
    assert enforced.max() == 1 and enforced.min() >= 0


@pytest.mark.parametrize("mode", M)
def test_a_permuted_image_gives_the_same_pixels(mode: WhiteBalanceMode):
    scene = np.dstack(
        (encoded(gray_world_scene(8, 9) * LIGHT), np.linspace(0, 1, 72).reshape(8, 9))
    ).astype(np.float32)
    permuted = np.moveaxis(scene, 2, 0).copy().transpose(1, 2, 0)
    assert not permuted.flags.c_contiguous
    temperature = 4000 if mode is M.TEMPERATURE else 6500
    assert np.array_equal(
        balance(permuted, mode, temperature=temperature),
        balance(scene, mode, temperature=temperature),
    )


def wide_range_image(shape: tuple[int, ...]) -> np.ndarray:
    """Float32 values whose float64 sums round, so the order of the additions shows."""
    rng = np.random.default_rng(11)
    exponents = rng.integers(-149, 40, shape)
    values = (rng.random(shape) + 0.5) * np.exp2(exponents.astype(np.float64))
    return (values * rng.choice((-1.0, 1.0), shape)).astype(np.float32)


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (17, 31), (300, 301), (1000, 37)])
@pytest.mark.parametrize("channels", [3, 4])
def test_channel_sums_are_numpys_reduce(shape: tuple[int, int], channels: int):
    image = wide_range_image((*shape, channels))
    rgb = image[:, :, :3]
    expected = np.sum(rgb, axis=(0, 1), dtype=np.float64)
    assert channel_sums(rgb).tobytes() == expected.tobytes()
    assert channel_sums(image).tobytes() == expected.tobytes()


def test_channel_sums_start_from_zero():
    image = np.full((1, 2, 3), -0.0, dtype=np.float32)
    expected = np.sum(image, axis=(0, 1), dtype=np.float64)
    assert channel_sums(image).tobytes() == expected.tobytes()


@pytest.mark.parametrize("shape", [(1, 1), (17, 31), (300, 301)])
@pytest.mark.parametrize("channels", [3, 4])
def test_scale_channels_is_the_in_place_product(shape: tuple[int, int], channels: int):
    image = wide_range_image((*shape, channels))
    image[0, 0, 0] = np.nan
    image[-1, -1, 2] = np.inf
    gains = np.array([1.25, 1e-30, 3.5], dtype=np.float32)
    expected = image.copy()
    expected[:, :, :3] *= gains
    scale_channels(image, gains)
    assert image.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "gains",
    [np.ones(2, np.float32), np.ones((3, 3), np.float32), np.ones(3, np.float64)],
)
def test_scale_channels_takes_three_float32_gains_only(gains: np.ndarray):
    # The C kernel reads three floats.
    image = np.ones((2, 2, 3), np.float32)
    with pytest.raises((ValueError, TypeError)):
        scale_channels(image, gains)
    assert np.array_equal(image, np.ones((2, 2, 3), np.float32))
