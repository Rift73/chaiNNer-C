"""Is Grayscale against a separate reading of the same rule in whole 8-bit levels: an image is
grayscale when, in every pixel, its colour channels lie within Tolerance levels of each other."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from packages.chaiNNer_standard.image_utility import miscellaneous_group
from packages.chaiNNer_standard.image_utility.miscellaneous.is_grayscale import (
    is_grayscale_node,
)

TOLERANCES = (0, 1, 2, 10, 127, 254, 255)


def as_image(levels: np.ndarray) -> np.ndarray:
    return (levels.astype(np.float32) / 255).astype(np.float32)


def judged(levels: np.ndarray, tolerance: int) -> bool:
    """The rule on the integer levels, which have no rounding to absorb."""
    if levels.ndim == 2 or levels.shape[2] == 1:
        return True
    colour = levels[:, :, :3].astype(np.int32)
    return bool((colour.max(axis=2) - colour.min(axis=2) <= tolerance).all())


def pixel(*channels: int) -> np.ndarray:
    return np.array([[channels]], np.uint8)


def test_the_output_is_a_bool_named_as_conditional_reads_it():
    node = next(
        node
        for node in miscellaneous_group.nodes
        if node.schema_id == "chainner:image:is_grayscale"
    )
    assert [output.label for output in node.outputs] == ["Is Grayscale"]
    for levels in (pixel(9, 9, 9), pixel(9, 9, 10), np.array([[9]], np.uint8)):
        assert type(is_grayscale_node(as_image(levels), 0)) is bool


@pytest.mark.parametrize("shape", [(1, 1), (1, 1, 1), (4, 5), (4, 5, 1)])
@pytest.mark.parametrize("tolerance", TOLERANCES)
def test_gray_is_grayscale(shape: tuple[int, ...], tolerance: int):
    levels = np.random.default_rng(1).integers(0, 256, shape, np.uint8)
    assert is_grayscale_node(as_image(levels), tolerance) is True


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("height,width", [(1, 1), (6, 7)])
def test_equal_colour_channels_are_grayscale(channels: int, height: int, width: int):
    rng = np.random.default_rng(2)
    gray = rng.integers(0, 256, (height, width), np.uint8)
    levels = np.dstack([gray] * 3)
    if channels == 4:
        # Alpha is not a colour channel, whatever it holds.
        levels = np.dstack([levels, rng.integers(0, 256, (height, width), np.uint8)])
    assert is_grayscale_node(as_image(levels), 0) is True


@pytest.mark.parametrize("height,width", [(1, 1), (6, 7)])
def test_colour_shows_with_any_alpha(height: int, width: int):
    levels = np.full((height, width, 4), 100, np.uint8)
    levels[height // 2, width // 2, 2] = 101
    assert is_grayscale_node(as_image(levels), 0) is False
    assert is_grayscale_node(as_image(levels), 1) is True
    levels[:, :, 3] = 0
    assert is_grayscale_node(as_image(levels), 0) is False


@pytest.mark.parametrize("tolerance", TOLERANCES)
@pytest.mark.parametrize("alpha", [None, 7])
def test_a_spread_at_the_tolerance_passes_and_one_level_more_fails(
    tolerance: int, alpha: int | None
):
    for spread, expected in ((tolerance, True), (tolerance + 1, False)):
        if spread > 255:
            continue
        low = (255 - spread) // 2
        for order in permutations((low, low + spread // 2, low + spread)):
            levels = pixel(*order, *(() if alpha is None else (alpha,)))
            assert judged(levels, tolerance) is expected
            assert is_grayscale_node(as_image(levels), tolerance) is expected, order


def test_the_first_channel_between_the_others_still_counts_their_spread():
    # 10 levels from the first channel to each other one, 20 between those two.
    for order in ((128, 138, 118), (128, 118, 138)):
        assert is_grayscale_node(as_image(pixel(*order)), 10) is False
        assert is_grayscale_node(as_image(pixel(*order)), 19) is False
        assert is_grayscale_node(as_image(pixel(*order)), 20) is True


def test_one_level_apart_at_every_level_and_in_every_pair():
    # float32 rounding makes some of these differences slightly more than 1/255.
    for level in range(1, 256):
        for order in set(permutations((level, level, level - 1))) | set(
            permutations((level, level - 1, level - 1))
        ):
            image = as_image(pixel(*order))
            assert is_grayscale_node(image, 1) is True, order
            assert is_grayscale_node(image, 0) is False, order


def test_one_pixel_decides_a_large_image():
    levels = np.full((512, 640, 3), 200, np.uint8)
    assert is_grayscale_node(as_image(levels), 0) is True
    levels[511, 639, 1] = 197
    assert is_grayscale_node(as_image(levels), 2) is False
    assert is_grayscale_node(as_image(levels), 3) is True


def test_the_smallest_sixteen_bit_step_shows_at_tolerance_zero():
    image = np.full((2, 2, 3), 30000 / 65535, np.float32)
    image[1, 1, 2] = 30001 / 65535
    assert is_grayscale_node(image, 0) is False


@pytest.mark.parametrize(
    "colour,grayscale",
    [
        ((np.inf, np.inf, np.inf), True),
        ((-np.inf, -np.inf, -np.inf), True),
        ((np.nan, np.nan, np.nan), True),
        ((np.inf, np.inf, 0.5), False),
        ((0.5, np.inf, np.inf), False),
        ((np.nan, np.nan, 0.5), False),
        ((0.5, np.nan, np.nan), False),
        ((np.inf, -np.inf, np.inf), False),
        ((np.inf, np.nan, np.inf), False),
    ],
)
@pytest.mark.parametrize("alpha", [(), (0.5,), (np.nan,)])
def test_channels_that_are_not_finite(
    colour: tuple[float, float, float], grayscale: bool, alpha: tuple[float, ...]
):
    image = np.array([[(*colour, *alpha)]], np.float32)
    for tolerance in (0, 255):
        assert is_grayscale_node(image, tolerance) is grayscale
    # Beside finite pixels that agree, and beside one that does not.
    pixels = np.array([[(0.25, 0.25, 0.25, *alpha), (*colour, *alpha)]], np.float32)
    assert is_grayscale_node(pixels, 0) is grayscale
    pixels[0, 0, 1] = 0.75
    assert is_grayscale_node(pixels, 0) is False


@pytest.mark.parametrize("tolerance", TOLERANCES)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_random_near_gray_images_match(tolerance: int, channels: int):
    rng = np.random.default_rng(tolerance * 10 + channels)
    for _ in range(200):
        height, width = rng.integers(1, 5, 2)
        base = rng.integers(0, 256, (height, width, 1))
        noise = rng.integers(
            -tolerance // 2 - 1, tolerance // 2 + 2, (height, width, 3)
        )
        colour = np.clip(base + noise, 0, 255)
        if channels == 1:
            levels = colour[:, :, 0].astype(np.uint8)
        elif channels == 3:
            levels = colour.astype(np.uint8)
        else:
            alpha = rng.integers(0, 256, (height, width, 1))
            levels = np.dstack([colour, alpha]).astype(np.uint8)
        assert is_grayscale_node(as_image(levels), tolerance) is judged(
            levels, tolerance
        ), levels.tolist()
