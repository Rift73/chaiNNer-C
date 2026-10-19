"""Native image_utility entry points (native/src/run/nodes/image_utilities.cpp) against their Python bodies."""

import numpy as np
import pytest

from image_node_cases import CHANNELS, EMPTY_SHAPES, SIZES, image, seed_of
from native_differential import assert_native_matches
from nodes.impl.native_graph import graph

CASES = [(size, channels) for size in SIZES for channels in CHANNELS]


def make(key: object, size: tuple[int, int], channels: int) -> np.ndarray:
    height, width = size
    return image(seed_of(key, size, channels), height, width, channels)


# (mode, rel_x, rel_y, abs_x, abs_y)
SHIFTS = [(0, 0, 0, 0, 0), (0, 37, -63, 0, 0), (0, -100, 100, 0, 0), (1, 0, 0, 3, -2), (1, 0, 0, -1000, 7),
          (1, 0, 0, 10**12, -(10**12)), (1, 0, 0, 1, 0), (1, 0, 0, 0, 0)]


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize("fill", [-1, 0, 1, 2])
@pytest.mark.parametrize("shift", SHIFTS)
def test_shift_matches(shift, fill, size, channels):
    assert_native_matches("chainner:image:shift", [make(("shift", shift, fill), size, channels), *shift, fill])


def test_shift_edge_cases():
    for shape in EMPTY_SHAPES:
        for fill in (-1, 1, 2):
            assert_native_matches("chainner:image:shift", [np.zeros(shape, np.float32), 1, 0, 0, 2, 3, fill])
    tall = image(3, 32767, 1, 1)
    for fill in (0, 2):
        assert_native_matches("chainner:image:shift", [tall, 1, 0, 0, 1, 1, fill])
    assert_native_matches("chainner:image:shift", [image(4, 5, 6, 3), 1, 0, 0, 2**64, 0, 0], declines=True)


# (coordinate type, x_rel, y_rel, x, y)
PICKS = [(1, 0, 0, 0, 0), (1, 100, 100, 0, 0), (1, 37, 63, 0, 0), (2, 0, 0, 0, 0), (2, 0, 0, 2, 1),
         (2, 0, 0, 30, 0), (2, 0, 0, 0, 30), (2, 0, 0, 300, 300)]


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize("pick", PICKS)
def test_pick_color_matches(pick, size, channels):
    assert_native_matches("chainner:image:pick_color", [make(("pick", pick), size, channels), *pick])


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("pick", PICKS)
def test_pick_color_empty_image(pick, shape):
    assert_native_matches("chainner:image:pick_color", [np.zeros(shape, np.float32), *pick])


def test_pick_color_edge_values():
    img = np.array([[np.nan, -0.0, np.inf, -np.inf, 1.5, 1e-45, 0.25]], np.float32)
    for x in range(img.shape[1]):
        assert_native_matches("chainner:image:pick_color", [img, 2, 0, 0, x, 0])


# Each stacked image as (height, width, channels).
STACKS = [
    [(5, 7, 3), (5, 7, 3)],
    [(5, 7, 1), (9, 4, 3), (2, 2, 4)],
    [(3, 3, 3), (1, 6, 4)],
    [(4, 6, 4), (6, 1, 3), (1, 1, 1)],
    [(8, 3, 1), (8, 3, 1), (2, 9, 1), (5, 5, 1)],
    [(17, 31, 4)],
    [(3, 5, 3), (7, 2, 4)],
]


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
@pytest.mark.parametrize("shapes", STACKS)
def test_stack_matches(shapes, orientation):
    images = [image(seed_of(shapes, i), h, w, c) for i, (h, w, c) in enumerate(shapes)]
    inputs: list[object] = [orientation, *images, *[None] * (14 - len(images))]
    assert_native_matches("chainner:image:stack", inputs)


def test_stack_empty_images():
    a = image(1, 3, 4, 3)
    for shape in EMPTY_SHAPES:
        empty = np.zeros(shape, np.float32)
        for images in ([empty], [a, None, empty], [empty, a]):
            assert_native_matches("chainner:image:stack", ["vertical", *images, *[None] * (14 - len(images))])


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize("expression", ["median", "mean", "minimum", "maximum"])
@pytest.mark.parametrize("count", [2, 3, 4, 14])
def test_z_stack_matches(count, expression, size, channels):
    images = [make(("z", expression, count, i), size, channels) for i in range(count)]
    inputs: list[object] = [expression, *images[:2], None, *images[2:], *[None] * (13 - count)]
    assert_native_matches("chainner:image:z_stack", inputs[:15])


def test_z_stack_errors():
    a, b = image(1, 4, 5, 3), image(2, 4, 5, 1)
    empty = np.zeros((0, 5, 3), np.float32)
    cases: list[list[object]] = [[a, b], [a, a, b], [a, empty], [empty, a]]
    for images in cases:
        assert_native_matches("chainner:image:z_stack", ["mean", *images, *[None] * (14 - len(images))])


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("palette_channels", CHANNELS)
@pytest.mark.parametrize("levels", [1, 2, 7, 256, 1000])
def test_apply_palette_matches(levels, palette_channels, size):
    height, width = size
    img = image(seed_of("lut", levels, size), height, width, 1, edges=False)
    palette = image(seed_of("palette", levels, palette_channels), 1, levels, palette_channels)
    assert_native_matches("chainner:image:lut", [img, palette])
    edged = image(seed_of("lut edges", levels, size), height, width, 1)
    assert_native_matches("chainner:image:lut", [edged, palette])


def test_apply_palette_edge_cases():
    palette = image(3, 2, 5, 3)
    for img in (np.zeros((0, 4), np.float32), np.array([[1.5, -0.5]], np.float32), np.array([[0.0, 1.0]], np.float32)):
        assert_native_matches("chainner:image:lut", [img, palette])
    assert_native_matches("chainner:image:lut", [np.zeros((2, 2), np.float32), np.zeros((0, 3, 3), np.float32)])


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("percentile", [0.0, 50.0, 99.99, 100.0, 37.25, 1])
@pytest.mark.parametrize("edges", [False, True])
def test_image_statistics_matches(edges, percentile, size):
    height, width = size
    img = image(seed_of("stats", percentile, size), height, width, 1, edges=edges)
    assert_native_matches("chainner:image:image_statistics", [img, percentile])


@pytest.mark.parametrize("shape", [(0, 5), (4, 0)])
def test_image_statistics_empty_image(shape):
    assert_native_matches("chainner:image:image_statistics", [np.zeros(shape, np.float32), 50.0])


ANGLES = [0.0, 90.0, 180.0, 270.0, 360.0, 45.0, 30.5, 359.9, 123.4, 0.1, 89.9]


@pytest.mark.parametrize(("size", "channels"), [((1, 1), 3), ((3, 5), 1), ((17, 31), 4), ((17, 31), 3), ((9, 9), 1)])
@pytest.mark.parametrize("fill", [-1, 0, 1])
@pytest.mark.parametrize("expand", [0, 1])
@pytest.mark.parametrize("interpolation", [0, 2, 3])
@pytest.mark.parametrize("angle", ANGLES)
def test_rotate_matches(angle, interpolation, expand, fill, size, channels):
    img = make(("rotate", angle), size, channels)
    assert_native_matches("chainner:image:rotate", [img, angle, interpolation, expand, fill])


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("fill", [-1, 0, 1])
@pytest.mark.parametrize("angle", [0.0, 90.0, 45.0])
def test_rotate_empty_image(angle, fill, shape):
    assert_native_matches("chainner:image:rotate", [np.zeros(shape, np.float32), angle, 3, 0, fill])


def test_view_returns_the_input_itself():
    img = image(5, 9, 11, 3)
    img.setflags(write=False)
    assert graph().run_native("chainner:image:view", [img])[0] is img
    assert_native_matches("chainner:image:view", [img])


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("method", [0, 1])
@pytest.mark.parametrize("radius", [0.0, 1.0, 3.5, 12.0])
def test_inpaint_matches(radius, method, channels, size):
    height, width = size
    img = image(seed_of("inpaint", size, channels), height, width, channels)
    mask = (image(seed_of("mask", size), height, width, 1, edges=False) > 0.7).astype(np.float32)
    assert_native_matches("chainner:image:inpaint", [img, mask, method, radius])


def test_inpaint_errors():
    cases: list[list[object]] = [
        [image(1, 4, 5, 3), image(2, 5, 4, 1), 0, 3.0],
        [np.zeros((0, 5, 3), np.float32), np.zeros((0, 5), np.float32), 1, 3.0],
        [np.zeros((4, 0), np.float32), np.zeros((4, 0), np.float32), 0, 3.0],
    ]
    for inputs in cases:
        assert_native_matches("chainner:image:inpaint", inputs)


@pytest.mark.parametrize("size", [*SIZES, (11, 11), (12, 40)])
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("relation", ["noise", "same", "close", "edges"])
def test_image_metrics_matches(relation, channels, size):
    height, width = size
    original = image(seed_of("metrics", size, channels), height, width, channels, edges=relation == "edges")
    if relation == "same":
        comparison = original.copy()
    elif relation == "close":
        comparison = (original + np.float32(1e-3)).astype(np.float32)
    else:
        comparison = image(seed_of("compare", size, channels), height, width, channels, edges=False)
    assert_native_matches("chainner:image:image_metrics", [original, comparison])


def test_image_metrics_errors():
    a = image(1, 6, 7, 3)
    cases: list[list[object]] = [
        [a, image(2, 7, 6, 3)],
        [a, image(2, 6, 7, 4)],
        [np.zeros((0, 5, 3), np.float32), np.zeros((0, 5, 3), np.float32)],
        [np.zeros((4, 0), np.float32), np.zeros((4, 0), np.float32)],
        [np.full((12, 12), np.inf, np.float32), np.zeros((12, 12), np.float32)],
    ]
    for inputs in cases:
        assert_native_matches("chainner:image:image_metrics", inputs)


COLOR_INPUTS = [1000, 1001, 1002, 1003, 1004, 1005, 0, 6]
# The output dropdown lists the spaces without their alpha partners (alpha is the toggle).
COLOR_OUTPUTS = [0, 1, 3, 4, 5, 6, 10, 12]


@pytest.mark.parametrize("size", [(1, 1), (17, 31)])
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("output", COLOR_OUTPUTS)
@pytest.mark.parametrize("source", COLOR_INPUTS)
def test_change_color_model_matches(source, output, channels, size):
    height, width = size
    for edges in (False, True):
        img = image(seed_of("colour", source, output, channels, size, edges), height, width, channels, edges=edges)
        for alpha in (False, True):
            assert_native_matches("chainner:image:change_colorspace", [img, source, output, alpha])


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("output", COLOR_OUTPUTS)
def test_change_color_model_empty_image(output, shape):
    for source in COLOR_INPUTS:
        assert_native_matches("chainner:image:change_colorspace", [np.zeros(shape, np.float32), source, output, True])
