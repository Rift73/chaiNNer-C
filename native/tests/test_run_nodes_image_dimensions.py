"""Native image_dimension entry points (native/src/run/nodes/image_dimensions.cpp) against their Python bodies."""

import numpy as np
import pytest
from native_differential import assert_native_matches
from native_images import CHANNELS, EMPTY_SHAPES, SIZES, image, seed_of

from nodes.impl.color.color import Color
from nodes.impl.native_graph import graph

CASES = [(size, channels) for size in SIZES for channels in CHANNELS]
COLORS = [Color.gray(0.4), Color.bgr((0.1, 0.5, 0.9)), Color.bgra((0.2, 0.4, 0.6, 0.8)), Color((1.5,)),
          Color((-0.5, 2.0, 0.25))]
BORDERS = [0, 1, 3, 4, 5, 6, 7]


def make(key: object, size: tuple[int, int], channels: int) -> np.ndarray:
    height, width = size
    return image(seed_of(key, size, channels), height, width, channels)


# (mode, amount, left, top, right, bottom, width, height)
PAD = [
    (0, 0, 0, 0, 0, 0, 100, 100),
    (0, 1, 0, 0, 0, 0, 100, 100),
    (0, 7, 0, 0, 0, 0, 100, 100),
    (1, 0, 1, 2, 3, 4, 100, 100),
    (1, 0, 0, 5, 0, 0, 100, 100),
    (1, 0, 0, 0, 0, 0, 100, 100),
    (2, 0, 2, 3, 0, 0, 10, 12),
    (2, 0, 0, 0, 0, 0, 2, 2),
    (2, 0, 0, 0, 0, 0, 5, 3),
    (2, 0, 9, 1, 0, 0, 1, 1),
    (2, 0, 0, 0, 0, 0, 40, 1),
]


@pytest.mark.parametrize(("size", "channels"), [(s, c) for s in SIZES[:3] for c in CHANNELS])
@pytest.mark.parametrize("border", BORDERS)
@pytest.mark.parametrize("parameters", PAD)
def test_pad_matches(parameters, border, size, channels):
    img = make(("pad", parameters, border), size, channels)
    mode, *numbers = parameters
    for color in COLORS if border == 7 else COLORS[:1]:
        assert_native_matches("chainner:image:pad", [img, border, color, mode, *numbers])


def test_pad_errors_and_declines():
    img = image(1, 4, 5, 3)
    color = COLORS[1]
    overflow = assert_native_matches("chainner:image:pad", [img, 0, color, 0, 2**31, 0, 0, 0, 0, 1, 1])
    assert overflow == ("error", ("OverflowError", "Padding exceeds the original OpenCV integer range"))
    assert_native_matches("chainner:image:pad", [img, 0, color, 1, 0, 0, 2**31 - 1, 0, 0, 1, 1], declines=True)
    assert_native_matches("chainner:image:pad", [img, 0, color, 0, 2**64, 0, 0, 0, 0, 1, 1], declines=True)
    for shape in EMPTY_SHAPES:
        for mode in (0, 1, 2):
            assert_native_matches("chainner:image:pad", [np.zeros(shape, np.float32), 7, color, mode, 2, 1, 1, 1, 1, 3, 3])


# (mode, amount, left, top, right, bottom, width, height)
CROP = [
    (0, 0, 0, 0, 0, 0, 100, 100),
    (0, 1, 0, 0, 0, 0, 100, 100),
    (0, 8, 0, 0, 0, 0, 100, 100),
    (0, 2**62, 0, 0, 0, 0, 100, 100),
    (1, 0, 0, 0, 0, 0, 100, 100),
    (1, 0, 1, 2, 3, 4, 100, 100),
    (1, 0, 0, 1, 0, 1, 100, 100),
    (1, 0, 2, 0, 0, 0, 100, 100),
    (1, 0, 30, 0, 2, 0, 100, 100),
    (2, 0, 0, 0, 0, 0, 100, 100),
    (2, 0, 1, 2, 0, 0, 3, 4),
    (2, 0, 0, 1, 0, 0, 10**12, 2),
    (2, 0, 0, 0, 0, 0, 1, 1),
    (2, 0, 4, 2, 0, 0, 2**62, 2**62),
    (2, 0, 2, 17, 0, 0, 1, 1),
]


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize("parameters", CROP)
def test_crop_matches(parameters, size, channels):
    assert_native_matches("chainner:image:crop", [make(("crop", parameters), size, channels), *parameters])


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("parameters", CROP)
def test_crop_empty_image(parameters, shape):
    assert_native_matches("chainner:image:crop", [np.zeros(shape, np.float32), *parameters])


def test_crop_declines_integers_beyond_int64():
    # The enforce rounds 2**63 - 1 through a float to 2**63.
    img = image(4, 6, 7, 3)
    assert_native_matches("chainner:image:crop", [img, 2, 0, 1, 1, 0, 0, 2**63 - 1, 3], declines=True)
    assert_native_matches("chainner:image:crop", [img, 0, 2**63, 0, 0, 0, 0, 1, 1], declines=True)


def bordered(seed: int, height: int, width: int, channels: int, border: int) -> np.ndarray:
    """Noise framed by a flat border of the given width, with a flat caption strip at the bottom."""
    img = image(seed, height, width, channels, edges=False)
    frame = np.full_like(img, 0.25)
    frame[border : height - border, border : width - border] = img[border : height - border, border : width - border]
    if height > 4 * border + 4:
        frame[height - border - 3 : height - border - 1, border + 1 : width - border - 1] = 0.9
    return frame


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize("select", [1, 2, 3])
@pytest.mark.parametrize(("tolerance", "padding"), [(5.0, 0), (0.0, 3), (30.0, 1000), (12.5, 2)])
def test_crop_border_matches(tolerance, padding, select, size, channels):
    height, width = size
    img = bordered(seed_of(tolerance, padding, select, size, channels), height, width, channels, min(height, width) // 4)
    assert_native_matches("chainner:image:remove_border", [img, tolerance, select, padding])
    noise = make(("noise", tolerance, select), size, channels)
    assert_native_matches("chainner:image:remove_border", [noise, tolerance, select, padding])


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_crop_border_empty_image(shape):
    assert_native_matches("chainner:image:remove_border", [np.zeros(shape, np.float32), 5.0, 1, 0])


def framed_alpha(seed: int, height: int, width: int, channels: int) -> np.ndarray:
    img = image(seed, height, width, channels)
    if channels == 4:
        img[:, :, 3] = 0
        img[height // 4 : height - height // 4 + 1, width // 3 : width - width // 5, 3] = 0.6
    return img


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize("threshold", [0.0, 50.0, 59.9, 60.0, 100.0])
def test_crop_to_content_matches(threshold, size, channels):
    height, width = size
    img = framed_alpha(seed_of("content", threshold, size, channels), height, width, channels)
    assert_native_matches("chainner:image:crop_content", [img, threshold])
    assert_native_matches("chainner:image:crop_content", [make(("raw", threshold), size, channels), threshold])


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("threshold", [0.0, 50.0, 59.9, 60.0, 100.0])
def test_get_bounding_box_matches(threshold, size):
    height, width = size
    mask = framed_alpha(seed_of("bbox", threshold, size), height, width, 4)[:, :, 3].copy()
    assert_native_matches("chainner:image:get_bbox", [mask, threshold])
    assert_native_matches("chainner:image:get_bbox", [make(("bbox raw", threshold), size, 1), threshold])


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_content_and_bbox_empty_image(shape):
    assert_native_matches("chainner:image:crop_content", [np.zeros(shape, np.float32), 10.0])
    if len(shape) == 2:
        assert_native_matches("chainner:image:get_bbox", [np.zeros(shape, np.float32), 10.0])


@pytest.mark.parametrize("shape", [*EMPTY_SHAPES, (3, 5), (2, 7, 3), (1, 1, 4)])
def test_get_dimensions_matches(shape):
    assert_native_matches("chainner:image:get_dims", [np.zeros(shape, np.float32)])


ALGORITHMS = ["adv_mame2x", "adv_mame3x", "adv_mame4x", "eagle2x", "eagle3x", "super_eagle2x", "sai2x",
              "super_sai2x", "hqx2x", "hqx3x", "hqx4x"]


@pytest.mark.parametrize(("size", "channels"), [(s, c) for s in SIZES[:3] for c in CHANNELS])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_resize_pixel_art_matches(algorithm, size, channels):
    img = make(("pixel art", algorithm), size, channels)
    # Pixel art: few distinct colours, so the pattern rules fire.
    img = np.round(img * 2) / 2
    assert_native_matches("chainner:image:resize_pixel_art", [img, algorithm])


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_resize_pixel_art_empty_image(shape):
    assert_native_matches("chainner:image:resize_pixel_art", [np.zeros(shape, np.float32), "hqx2x"])


def test_noop_returns_the_input_itself():
    img = image(5, 9, 11, 3)
    img.setflags(write=False)
    color = COLORS[0]
    assert graph().run_native("chainner:image:pad", [img, 0, color, 1, 0, 0, 0, 0, 0, 1, 1])[0] is img
    assert graph().run_native("chainner:image:crop", [img, 0, 0, 0, 0, 0, 0, 1, 1])[0] is img
    assert graph().run_native("chainner:image:crop_content", [img, 0.0])[0] is img


SIDES = ["width", "height", "shorter side", "longer side"]
CONDITIONS = ["both", "upscale", "downscale"]
FILTERS = [-1, 0, 4, 2, 3, 1, 5, 6, 7, 8, 9, 10, 11, 12]
DPID = 12


@pytest.mark.parametrize(("size", "channels"), [(s, c) for s in SIZES[:3] for c in CHANNELS])
@pytest.mark.parametrize("target", [1, 4, 17, 64])
@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("side", SIDES)
def test_resize_to_side_matches(side, condition, target, size, channels):
    img = make(("to side", side, condition, target), size, channels)
    for resize_filter in FILTERS if target == 17 else [-1, 4, DPID]:
        assert_native_matches("chainner:image:resize_to_side", [img, target, side, condition, resize_filter, 1.0])


@pytest.mark.parametrize("dpid_lambda", [0.0, 0.35, 2.5, 10.0])
@pytest.mark.parametrize("channels", CHANNELS)
def test_resize_to_side_dpid_lambda_matches(channels, dpid_lambda):
    img = make(("to side dpid", dpid_lambda), (64, 257), channels)
    for target, side in [(17, "width"), (40, "height"), (100, "longer side")]:
        outcome = assert_native_matches("chainner:image:resize_to_side", [img, target, side, "both", DPID, dpid_lambda])
        assert outcome[0] == "outputs"


def test_resize_to_side_edge_cases():
    for shape in EMPTY_SHAPES:
        for side in SIDES:
            for resize_filter in (-1, DPID):
                inputs = [np.zeros(shape, np.float32), 8, side, "both", resize_filter, 1.0]
                assert_native_matches("chainner:image:resize_to_side", inputs)
    img = image(6, 3, 4, 3)
    # The enforce rounds through a float: 2**53 + 1 arrives as 2**53 (computed), 2**53 + 2 as itself (declined).
    assert_native_matches("chainner:image:resize_to_side", [img, 2**53 + 1, "width", "both", 0, 1.0])
    assert_native_matches("chainner:image:resize_to_side", [img, 2**53 + 2, "width", "both", 0, 1.0], declines=True)
    assert_native_matches("chainner:image:resize_to_side", [img, 2**53 + 2, "height", "upscale", 0, 1.0], declines=True)
    assert_native_matches("chainner:image:resize_to_side", [img, 2**40, "width", "downscale", 0, 1.0])


# (width, height, b, c, taps, blur, left, top, source width, source height)
DESCALES = [
    (8, 6, 0.0, 0.5, 3, 1.0, 0.0, 0.0, 0.0, 0.0),
    (17, 31, 0.0, 0.5, 3, 1.0, 0.0, 0.0, 0.0, 0.0),
    (5, 9, 1 / 3, 1 / 3, 4, 1.25, 0.25, -0.5, 0.0, 0.0),
    (12, 20, -0.5, 0.75, 2, 0.8, 0.0, 0.0, 12.5, 19.0),
    (16, 30, 0.0, 0.0, 6, 1.0, 1.5, 0.0, 0.0, 30.5),
    (3, 2, 0.0, 0.5, 1, 10.0, 0.0, 0.0, 0.0, 0.0),
]


@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("border", [0, 1, 2])
@pytest.mark.parametrize("kernel", [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("parameters", DESCALES)
def test_descale_matches(parameters, kernel, border, channels):
    width, height, *rest = parameters
    img = make(("descale", parameters), (31, 17), channels)
    assert_native_matches(
        "chainner:image:descale", [img, width, height, kernel, rest[0], rest[1], rest[2], *rest[3:], border, False, 0]
    )


# Linear Light (13) off with each Transfer (14), then on.
@pytest.mark.parametrize("light", [(False, 1), (False, 3), (True, 0), (True, 1), (True, 2), (True, 3)])
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("kernel", [1, 2, 3, 6])
@pytest.mark.parametrize("parameters", DESCALES[:4])
def test_descale_linear_light_matches(parameters, kernel, channels, light):
    width, height, *rest = parameters
    img = make(("descale light", parameters), (31, 17), channels)
    inputs = [img, width, height, kernel, rest[0], rest[1], rest[2], *rest[3:], 2, *light]
    assert assert_native_matches("chainner:image:descale", inputs)[0] == "outputs"


@pytest.mark.parametrize("light", [(False, 0), (True, 1)])
def test_descale_errors(light):
    img = image(3, 10, 12, 3)
    cases: list[list[object]] = [
        [img, 13, 10, 1, 0.0, 0.5, 3, 1.0, 0.0, 0.0, 0.0, 0.0, 0],
        [img, 12, 11, 1, 0.0, 0.5, 3, 1.0, 0.0, 0.0, 0.0, 0.0, 0],
        [img, 2, 2, 3, 0.0, 0.5, 32, 1.0, 0.0, 0.0, 0.0, 0.0, 0],
        [img, 6, 5, 2, 0.0, 0.5, 3, 1.0, 900.0, 0.0, 0.0, 0.0, 0],
        [np.zeros((0, 5, 3), np.float32), 1, 1, 1, 0.0, 0.5, 3, 1.0, 0.0, 0.0, 0.0, 0.0, 0],
    ]
    for inputs in cases:
        assert_native_matches("chainner:image:descale", [*inputs, *light])
