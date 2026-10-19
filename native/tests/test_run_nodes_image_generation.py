"""Native image creation entry points (native/src/run/nodes/image_generation.cpp) against their Python bodies."""

import numpy as np
import pytest
from native_differential import assert_native_matches
from native_images import image, seed_of

from nodes.impl.color.color import Color

COLORS = [
    Color.gray(0.0),
    Color.gray(0.37),
    Color.bgr((0.1, 0.5, 0.9)),
    Color.bgra((0.2, 0.4, 0.6, 0.8)),
    Color((1.5,)),
    Color((-0.25, 0.3, 2.0)),
    Color((0.1, float("nan"), 0.123456789, -0.0)),
]
SIZES = [(1, 1), (5, 3), (31, 17), (257, 64)]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("color", COLORS)
def test_create_color_matches(color, size):
    assert_native_matches("chainner:image:create_color", [color, *size])


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("square", [1, 2, 7, 32, 10**6])
@pytest.mark.parametrize("pair", [(0, 1), (2, 3), (1, 2), (3, 0), (4, 5), (6, 4), (2, 2), (5, 3)])
def test_create_checkerboard_matches(pair, square, size):
    width, height = size
    first, second = COLORS[pair[0]], COLORS[pair[1]]
    assert_native_matches("chainner:image:create_checkerboard", [width, height, first, second, square])


# (style, diagonal angle, diagonal width, inner radius, outer radius, conic rotation)
STYLES = [
    ("Horizontal", 45, 100, 0, 100, 0),
    ("Vertical", 45, 100, 0, 100, 0),
    ("Diagonal", 45, 100, 0, 100, 0),
    ("Diagonal", 0, 1, 0, 100, 0),
    ("Diagonal", 359, 0, 0, 100, 0),
    ("Diagonal", 133, 10**9, 0, 100, 0),
    ("Radial", 45, 100, 0, 100, 0),
    ("Radial", 45, 100, 37, 81, 0),
    ("Radial", 45, 100, 50, 50, 0),
    ("Radial", 45, 100, 100, 0, 0),
    ("Conic", 45, 100, 0, 100, 0),
    ("Conic", 45, 100, 0, 100, 181),
    ("Conic", 45, 100, 0, 100, 360),
]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("pair", [(0, 1), (2, 3), (1, 3), (5, 0), (6, 4)])
@pytest.mark.parametrize("style", STYLES)
def test_create_gradient_matches(style, pair, reverse, size):
    width, height = size
    name, *parameters = style
    first, second = COLORS[pair[0]], COLORS[pair[1]]
    assert_native_matches("chainner:image:create_gradient", [width, height, first, second, reverse, name, *parameters])


@pytest.mark.parametrize("size", [1, 2, 3, 64, 255])
def test_create_colorwheel_matches(size):
    assert_native_matches("chainner:image:create_colorwheel", [size])


def test_sizes_beyond_an_image_decline():
    assert_native_matches("chainner:image:create_color", [COLORS[2], 2**40, 1], declines=True)
    assert_native_matches("chainner:image:create_colorwheel", [2**64], declines=True)


@pytest.mark.parametrize(
    "color, width, height",
    [
        (COLORS[2], 2**22, 2**22),  # 192 TiB
        (COLORS[1], 2**23, 2**23),  # 256 TiB, a 2-D shape
        (COLORS[3], 2**24, 2**24),  # 4.00 PiB
        (COLORS[1], 262144000, 2**20),  # 1000 TiB: no decimals
    ],
)
def test_out_of_memory_is_numpys_memory_error(color, width, height):
    outcome = assert_native_matches("chainner:image:create_color", [color, width, height])
    assert outcome[0] == "error" and outcome[1][0] == "numpy._core._exceptions._ArrayMemoryError", outcome


# (method, scale, brightness, tile h, tile v, tile spherical, fractal, layers, scale ratio, brightness ratio,
#  increment seed, standard deviation)
NOISE = [
    ("Simplex", 50.0, 100.0, False, False, False, "None", 3, 2.0, 2.0, True, 1.5),
    ("Value Noise", 7.5, 63.25, True, False, False, "None", 3, 2.0, 2.0, True, 1.5),
    ("Smooth Value Noise", 1.0, 0.0, False, True, False, "Pink noise", 2, 1.0, 1.0, False, 1.5),
    ("Simplex", 123.4, 100.0, True, True, True, "Pink noise", 20, 2.5, 1.75, True, 1.5),
    ("Value Noise", 30.0, 50.0, False, False, True, "Pink noise", 5, 3.0, 2.0, True, 1.5),
    ("Simplex", 50.0, 100.0, False, False, False, "Pink noise", 20, 1e300, 2.0, True, 1.5),
    ("Simplex", 50.0, 100.0, False, False, False, "Pink noise", 3, 2.0, 1e200, True, 1.5),
    ("Blue Noise", 50.0, 100.0, False, False, False, "None", 3, 2.0, 2.0, True, 1.5),
    ("Blue Noise", 50.0, 100.0, False, False, False, "None", 3, 2.0, 2.0, True, 1.0),
    ("Blue Noise", 50.0, 100.0, False, False, False, "None", 3, 2.0, 2.0, True, 37.0),
]


@pytest.mark.parametrize("size", [(1, 1), (5, 3), (31, 17), (64, 64)])
@pytest.mark.parametrize("seed", [0, 12345, 2**32 - 1])
@pytest.mark.parametrize("noise", NOISE)
def test_create_noise_matches(noise, seed, size):
    width, height = size
    assert_native_matches("chainner:image:create_noise", [width, height, seed, *noise])


# NaN is rejected by the input enforce before the body runs.
@pytest.mark.parametrize("ratio", [float("inf"), float("-inf")])
@pytest.mark.parametrize("ratios", [(None, 2.0), (2.0, None)])
@pytest.mark.parametrize("layers", [2, 3])
def test_create_noise_non_finite_ratios_match(layers, ratios, ratio):
    scale_ratio, brightness_ratio = (ratio if r is None else r for r in ratios)
    fractal = ("Pink noise", layers, scale_ratio, brightness_ratio, True, 1.5)
    noise = ("Simplex", 10.0, 50.0, False, False, False, *fractal)
    assert_native_matches("chainner:image:create_noise", [7, 5, 3, *noise])


def test_create_noise_declines_seeds_outside_u32():
    for seed in (-1, 2**32, 2**40):
        assert_native_matches("chainner:image:create_noise", [8, 8, seed, *NOISE[0]], declines=True)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize(("size", "rows", "columns"), [((6, 8), 2, 4), ((6, 8), 1, 1), ((17, 31), 17, 1),
                                                       ((12, 12), 3, 2), ((1, 1), 1, 1), ((6, 8), 4, 4),
                                                       ((6, 8), 3, 3), ((0, 8), 2, 4), ((6, 0), 3, 1)])
def test_split_spritesheet_matches(size, rows, columns, order, channels):
    height, width = size
    sheet = image(seed_of("sprites", size, channels), height, width, channels)
    assert_native_matches("chainner:image:split_spritesheet", [sheet, rows, columns, order])


def test_split_spritesheet_declines_huge_counts():
    sheet = np.zeros((4, 4, 3), np.float32)
    assert_native_matches("chainner:image:split_spritesheet", [sheet, 2**64, 1, 0], declines=True)
