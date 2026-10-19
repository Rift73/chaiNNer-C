"""Native image_utility entry points (native/src/run/nodes/image_utility.cpp) against their Python bodies."""

import numpy as np
import pytest
from native_differential import assert_native_matches
from native_images import image, seed_of

from nodes.impl.blend import BlendMode
from nodes.impl.color.color import Color
from nodes.impl.image_utils import FlipAxis
from nodes.impl.native_graph import graph

SIZES = [(1, 1), (3, 5), (17, 31), (64, 257)]
CHANNELS = [1, 3, 4]


@pytest.mark.parametrize("layout", ["c", "permuted"])
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("axis", [a.value for a in FlipAxis])
def test_flip_matches(axis, size, channels, layout):
    height, width = size
    img = image(seed_of(axis, size, channels, layout), height, width, channels, layout)
    assert_native_matches("chainner:image:flip", [img, axis])


def test_flip_none_returns_the_input_itself():
    img = image(1, 9, 11, 3, "c")
    img.setflags(write=False)
    assert graph().run_native("chainner:image:flip", [img, FlipAxis.NONE])[0] is img


@pytest.mark.parametrize("shape", [(0, 5, 3), (4, 0), (0, 0, 4)])
def test_flip_empty_image_error(shape):
    outcome = assert_native_matches(
        "chainner:image:flip", [np.zeros(shape, np.float32), 1]
    )
    assert outcome == (
        "error",
        ("ValueError", "Expected a nonempty two- or three-dimensional image"),
    )


BLEND = "chainner:image:blend"
MODES = [mode.value for mode in BlendMode]
POSITIONS = [
    "top_left",
    "top_centered",
    "top_right",
    "centered_left",
    "centered",
    "centered_right",
    "bottom_left",
    "bottom_centered",
    "bottom_right",
    "percent_offset",
    "pixel_offset",
]


def blend_inputs(
    base: object,
    overlay: object,
    mode: int = 0,
    position: str = "centered",
    percent: tuple[int, int] = (0, 0),
    pixels: tuple[int, int] = (0, 0),
    crop: bool = False,
) -> list[object]:
    return [base, overlay, mode, position, *percent, *pixels, crop]


@pytest.mark.parametrize("overlay_size", [(5, 7), (17, 31)])
@pytest.mark.parametrize("overlay_channels", CHANNELS)
@pytest.mark.parametrize("base_channels", CHANNELS)
@pytest.mark.parametrize("mode", MODES)
def test_blend_modes_match(mode, base_channels, overlay_channels, overlay_size):
    key = (BLEND, mode, base_channels, overlay_channels, overlay_size)
    base = image(seed_of(*key, "base"), 17, 31, base_channels)
    oh, ow = overlay_size
    overlay = image(seed_of(*key), oh, ow, overlay_channels)
    assert_native_matches(BLEND, blend_inputs(base, overlay, mode))


@pytest.mark.parametrize("crop", [False, True])
@pytest.mark.parametrize("overlay_size", [(5, 8), (17, 31), (40, 51)])
@pytest.mark.parametrize("channels", [(3, 4), (1, 3), (4, 1)])
@pytest.mark.parametrize("position", POSITIONS)
def test_blend_positions_match(position, channels, overlay_size, crop):
    key = (BLEND, position, channels, overlay_size, crop)
    base = image(seed_of(*key, "base"), 17, 31, channels[0])
    oh, ow = overlay_size
    overlay = image(seed_of(*key), oh, ow, channels[1])
    inputs = blend_inputs(base, overlay, 12, position, (37, -50), (-3, 5), crop)
    assert_native_matches(BLEND, inputs)


@pytest.mark.parametrize("crop", [False, True])
@pytest.mark.parametrize("y", [-1000, -40, -3, 0, 5, 16, 30])
@pytest.mark.parametrize("x", [-1000, -40, -3, 0, 5, 30, 1000])
def test_blend_pixel_offsets_match(x, y, crop):
    base = image(seed_of(BLEND, "px base"), 17, 31, 3)
    overlay = image(seed_of(BLEND, "px", x, y), 9, 12, 4)
    inputs = blend_inputs(base, overlay, 1, "pixel_offset", (0, 0), (x, y), crop)
    assert_native_matches(BLEND, inputs)


@pytest.mark.parametrize("percent", [(-200, 200), (-37, 13), (50, 50), (200, -200)])
@pytest.mark.parametrize("overlay_size", [(6, 9), (16, 30), (33, 41)])
def test_blend_percent_offsets_match(percent, overlay_size):
    # Odd size differences make the offsets ties for round().
    base = image(seed_of(BLEND, "% base"), 17, 31, 4)
    oh, ow = overlay_size
    overlay = image(seed_of(BLEND, "%", percent, overlay_size), oh, ow, 3)
    inputs = blend_inputs(base, overlay, 0, "percent_offset", percent)
    assert_native_matches(BLEND, inputs)


COLORS = [Color((0.25,)), Color((0.1, 0.5, 0.9)), Color((0.2, 0.4, 0.6, 0.5))]


@pytest.mark.parametrize("mode", [0, 1, 12, 15])
@pytest.mark.parametrize("image_channels", CHANNELS)
@pytest.mark.parametrize("color", COLORS)
@pytest.mark.parametrize("color_is_base", [True, False])
def test_blend_color_layers_match(color_is_base, color, image_channels, mode):
    img = image(seed_of(BLEND, color.value, image_channels, mode), 7, 9, image_channels)
    layers = (color, img) if color_is_base else (img, color)
    inputs = blend_inputs(*layers, mode, "pixel_offset", (0, 0), (3, -2), True)
    assert_native_matches(BLEND, inputs)


def test_blend_two_colors_error():
    outcome = assert_native_matches(BLEND, blend_inputs(COLORS[1], COLORS[2]))
    assert outcome == ("error", ("ValueError", "At least one layer must be an image"))


@pytest.mark.parametrize("empty_base", [True, False])
def test_blend_empty_image_error(empty_base):
    empty, img = np.zeros((0, 4, 3), np.float32), image(1, 3, 4, 3)
    layers = (empty, img) if empty_base else (img, empty)
    outcome = assert_native_matches(BLEND, blend_inputs(*layers))
    message = "Blend images must be nonempty two- or three-dimensional arrays"
    assert outcome == ("error", ("ValueError", message))


@pytest.mark.parametrize("pixels", [(2**31 + 10, 0), (0, -(2**31) - 5), (-(2**40), 0)])
def test_blend_border_overflow_error(pixels):
    base, overlay = image(1, 5, 6, 3), image(2, 2, 2, 3)
    inputs = blend_inputs(base, overlay, 0, "pixel_offset", (0, 0), pixels)
    outcome = assert_native_matches(BLEND, inputs)
    message = "Blend canvas border exceeds OpenCV's integer range"
    assert outcome == ("error", ("OverflowError", message))


def test_blend_crop_far_offsets_match():
    base, overlay = image(1, 5, 6, 3), image(2, 2, 2, 4)
    # Precision 0 rounds through a float: 2**63 - 1 would become 2**63, beyond int64.
    for pixels in [(2**62, 0), (-(2**62), 3), (2**62, -(2**63))]:
        inputs = blend_inputs(base, overlay, 0, "pixel_offset", (0, 0), pixels, True)
        assert_native_matches(BLEND, inputs)


@pytest.mark.parametrize(
    "pixels",
    [
        # Beyond int64.
        (2**64, 0),
        (0, -(2**70)),
        # A canvas side beyond the engine's int32 images.
        (2**31 - 1, 0),
    ],
)
def test_blend_declines(pixels):
    base, overlay = image(1, 5, 6, 3), image(2, 2, 2, 3)
    inputs = blend_inputs(base, overlay, 0, "pixel_offset", (0, 0), pixels)
    assert_native_matches(BLEND, inputs, declines=True)


PALETTE = "chainner:image:palette_from_image"


def quantized(seed: int, height: int, width: int, channels: int, levels: int):
    """An image of at most levels ** channels distinct colors."""
    values = np.floor(image(seed, height, width, channels, edges=False) * levels)
    return (values / levels).astype(np.float32)


@pytest.mark.parametrize("palette_size", [2, 3, 8, 16, 300, 4096])
@pytest.mark.parametrize("method", ["all", "k-means", "median"])
@pytest.mark.parametrize("kind", ["noise", "edges", "few", "one"])
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("size", [(1, 1), (3, 5), (17, 31), (40, 57)])
def test_palette_from_image_matches(size, channels, kind, method, palette_size):
    height, width = size
    key = (PALETTE, size, channels, kind)
    if kind == "few":
        img = quantized(seed_of(*key), height, width, channels, 3)
    elif kind == "one":
        shape = (height, width) if channels == 1 else (height, width, channels)
        img = np.full(shape, 0.4, np.float32)
    else:
        img = image(seed_of(*key), height, width, channels, edges=kind == "edges")
    assert_native_matches(PALETTE, [img, method, palette_size])


@pytest.mark.parametrize("channels", CHANNELS)
def test_palette_from_image_too_many_distinct_colors(channels):
    img = image(seed_of(PALETTE, "many", channels), 70, 70, channels, edges=False)
    outcome = assert_native_matches(PALETTE, [img, "all", 8])
    assert outcome[0] == "error"


@pytest.mark.parametrize("method", ["all", "k-means", "median"])
@pytest.mark.parametrize("shape", [(0, 5, 3), (4, 0), (0, 0, 4)])
def test_palette_from_image_empty_image(shape, method):
    outcome = assert_native_matches(PALETTE, [np.zeros(shape, np.float32), method, 8])
    assert outcome[0] == "error"
