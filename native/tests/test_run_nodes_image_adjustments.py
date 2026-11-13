"""Native image_adjustment entry points (native/src/run/nodes/image_adjustments.cpp) against their Python bodies."""

import errno
import os

import numpy as np
import pytest
from native_differential import assert_native_matches
from native_images import CHANNELS, EMPTY_SHAPES, SIZES, image, seed_of

from nodes.impl.native_graph import graph
from packages.chaiNNer_standard.image_adjustment.adjustments.stretch_contrast import (
    StretchMode,
    stretch_contrast_node,
)

CASES = [(size, channels) for size in SIZES for channels in CHANNELS]

# (schema id, parameter sets after the image); the first set of each is the node's default.
NODES = {
    "chainner:image:brightness_and_contrast": [
        (0.0, 0.0),
        (50.0, 0.0),
        (0.0, 50.0),
        (-100.0, -100.0),
        (100.0, 100.0),
        (-37.5, 99.9),
        (12.3, -45.6),
        (100.0, -100.0),
    ],
    "chainner:image:clamp": [
        (0.0, 1.0),
        (0.0, 0.5),
        (0.25, 1.0),
        (0.3, 0.2),
        (1.0, 0.0),
        (0.0001, 0.9999),
    ],
    "chainner:image:color_levels": [
        (True, True, True, False, 0.0, 1.0, 1.0, 0.0, 1.0),
        (True, False, True, True, 0.2, 0.8, 2.2, 0.1, 0.9),
        (False, True, False, True, 0.5, 0.5, 1.0, 0.0, 1.0),
        (True, True, True, True, 0.9, 0.1, 0.0, 1.0, 0.0),
        (False, False, False, False, 0.0, 1.0, 10.0, 0.0, 1.0),
        (False, False, True, False, 0.001, 0.999, 0.0005, 0.25, 0.75),
    ],
    "chainner:image:invert": [()],
    "chainner:image:opacity": [(100.0,), (0.0,), (50.0,), (33.3,)],
    "chainner:image:add": [(0.0,), (-100.0,), (100.0,), (12.5,), (-0.1,)],
    "chainner:image:divide": [(1.0,), (0.0001,), (4.0,), (0.3333,)],
    "chainner:image:multiply": [(1.0,), (0.0,), (4.0,), (0.0001,), (2.5,)],
    "chainner:image:gamma": [
        (1.0, False),
        (2.2, False),
        (2.2, True),
        (0.01, True),
        (100.0, False),
        (0.5, True),
    ],
    "chainner:image:log2lin": [
        (95.0, 685.0, 0.6, False),
        (95.0, 685.0, 0.6, True),
        (0.0, 1023.0, 1.0, False),
        (1023.0, 0.0, 0.5, True),
        (0.0, 1023.0, 0.0001, True),
        (1023.0, 0.0, 0.0001, False),  # pow(10.0, 20460.0): OverflowError
        (500.0, 500.0, 0.5, False),  # black == white: ZeroDivisionError
    ],
    "chainner:image:generate_threshold": [(0,), (1,)],
}
PARAMETERS = [(schema, values) for schema, sets in NODES.items() for values in sets]


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize(("schema", "values"), PARAMETERS)
def test_adjustment_matches(schema, values, size, channels):
    height, width = size
    img = image(seed_of(schema, values, size, channels), height, width, channels)
    assert_native_matches(schema, [img, *values])


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("association", ["Straight -> Premultiplied", "Premultiplied -> Straight"])
def test_premultiplied_alpha_matches(association, size):
    height, width = size
    img = image(seed_of(association, size), height, width, 4)
    assert_native_matches("chainner:image:premultiplied_alpha", [img, association])


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("schema", list(NODES))
def test_adjustment_empty_image(schema, shape):
    for values in NODES[schema]:
        assert_native_matches(schema, [np.zeros(shape, np.float32), *values])


@pytest.mark.parametrize("shape", [(0, 5, 4), (3, 0, 4)])
def test_premultiplied_alpha_empty_image(shape):
    assert_native_matches("chainner:image:premultiplied_alpha", [np.zeros(shape, np.float32), "Premultiplied -> Straight"])


@pytest.mark.parametrize(
    ("schema", "values", "channels"),
    [
        ("chainner:image:brightness_and_contrast", (0.0, -0.0), 3),
        ("chainner:image:clamp", (0.0, 1.0), 1),
        ("chainner:image:opacity", (100.0,), 4),
        ("chainner:image:add", (0.0,), 4),
        ("chainner:image:divide", (1.0,), 3),
        ("chainner:image:multiply", (1.0,), 1),
        ("chainner:image:gamma", (1.0, True), 3),
    ],
)
def test_noop_returns_the_input_itself(schema, values, channels):
    img = image(1, 9, 11, channels)
    img.setflags(write=False)
    assert graph().run_native(schema, [img, *values])[0] is img


def test_log_to_linear_errors():
    img = image(2, 5, 7, 3)
    overflow = assert_native_matches("chainner:image:log2lin", [img, 1023.0, 0.0, 0.0001, False])
    # CPython's text for ERANGE is the C library's
    range_error = f"({errno.ERANGE}, {os.strerror(errno.ERANGE)!r})"
    assert overflow == ("error", ("OverflowError", range_error))
    # Equal Black and White was "division by zero".
    zero = assert_native_matches("chainner:image:log2lin", [img, 300.0, 300.0, 0.25, True])
    assert zero == ("error", ("ValueError", "Black and White must differ"))
    assert assert_native_matches("chainner:image:log2lin", [img, 300.0, 300.1, 0.25, True])[0] == "outputs"


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize(("hue", "saturation", "lightness"), [(0.0, 0.0, 0.0), (37.0, 0.0, 0.0), (0.0, -50.0, 0.0),
                                                              (0.0, 0.0, 40.0), (-180.0, 100.0, -100.0),
                                                              (180.0, -100.0, 100.0), (12.5, 33.3, -12.0)])
def test_hue_and_saturation_matches(hue, saturation, lightness, size, channels):
    height, width = size
    img = image(seed_of("hls", hue, saturation, lightness, size, channels), height, width, channels)
    assert_native_matches("chainner:image:hue_and_saturation", [img, hue, saturation, lightness])


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_hue_and_saturation_empty_image(shape):
    for values in ((37.0, 0.0, 0.0), (0.0, 0.0, 40.0), (0.0, 0.0, 0.0)):
        assert_native_matches("chainner:image:hue_and_saturation", [np.zeros(shape, np.float32), *values])


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize("kind", [0, 1, 2, 3, 4])
@pytest.mark.parametrize(("value", "maximum", "anti_aliasing", "smoothness"), [(50.0, 100.0, False, 0), (0.0, 37.5, False, 0),
                                                                               (100.0, 0.0, False, 0), (42.5, 80.0, True, 0),
                                                                               (50.0, 100.0, True, 10), (13.0, 60.0, True, 3)])
def test_threshold_matches(value, maximum, anti_aliasing, smoothness, kind, size, channels):
    height, width = size
    img = image(seed_of("threshold", value, anti_aliasing, kind, size, channels), height, width, channels)
    assert_native_matches("chainner:image:threshold", [img, value, kind, maximum, anti_aliasing, smoothness])


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_threshold_empty_image(shape):
    for anti_aliasing in (False, True):
        assert_native_matches("chainner:image:threshold", [np.zeros(shape, np.float32), 50.0, 0, 100.0, anti_aliasing, 0])


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("kind", [0, 1])
@pytest.mark.parametrize("method", [0, 1])
@pytest.mark.parametrize(("maximum", "radius", "constant"), [(100.0, 1, 0.0), (37.3, 3, 12.5), (0.0, 7, -100.0),
                                                             (100.0, 40, 100.0), (50.0, 2, 0.2), (50.0, 2, -0.6)])
def test_threshold_adaptive_matches(maximum, radius, constant, method, kind, size):
    height, width = size
    img = image(seed_of("adaptive", maximum, radius, method, kind, size), height, width, 1)
    assert_native_matches("chainner:image:threshold_adaptive", [img, kind, maximum, method, radius, constant])


def test_threshold_adaptive_edge_cases():
    img = image(4, 9, 11, 1)
    for method in (0, 1):
        assert_native_matches("chainner:image:threshold_adaptive", [img, 0, 100.0, method, 2**30, 0.0])
        assert_native_matches("chainner:image:threshold_adaptive", [np.zeros((0, 4), np.float32), 0, 100.0, method, 1, 0.0])
    assert_native_matches("chainner:image:threshold_adaptive", [img, 0, 100.0, 0, 2**70, 0.0], declines=True)


# (mode, keep colours, percentile, manual minimum, manual maximum)
STRETCHES = [(0, True, 1.0, 0.0, 255.0), (0, False, 1.0, 0.0, 255.0), (1, True, 1.0, 0.0, 255.0),
             (1, False, 0.0, 0.0, 255.0), (1, False, 37.5, 0.0, 255.0), (1, True, 50.0, 0.0, 255.0),
             (2, True, 1.0, 12.5, 200.0), (2, False, 1.0, 100.0, 100.0), (2, True, 1.0, 200.0, 100.0)]


@pytest.mark.parametrize(("size", "channels"), CASES + [((100, 100), 3), ((9000, 3), 4)])
@pytest.mark.parametrize("stretch", STRETCHES)
@pytest.mark.parametrize("edges", [False, True])
def test_stretch_contrast_matches(edges, stretch, size, channels):
    height, width = size
    img = image(seed_of("stretch", stretch, size, channels), height, width, channels, edges=edges)
    # The edge values start with a NaN, whose range only the Python body finds.
    assert_native_matches("chainner:image:stretch_contrast", [img, *stretch], declines=edges and stretch[0] != 2)


AUTOMATIC_STRETCHES = [(0, True, 1.0, 0.0, 255.0), (0, False, 1.0, 0.0, 255.0), (1, True, 1.0, 0.0, 255.0), (1, False, 1.0, 0.0, 255.0)]


@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("value", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("stretch", AUTOMATIC_STRETCHES)
def test_stretch_contrast_leaves_a_flat_image(stretch, value, channels):
    img = np.full((3, 4, channels) if channels > 1 else (3, 4), value, np.float32)
    assert_native_matches("chainner:image:stretch_contrast", [img, *stretch])
    result = stretch_contrast_node(img, StretchMode(stretch[0]), *stretch[1:])
    np.testing.assert_array_equal(result, img)


def test_stretch_contrast_keeps_a_flat_channel():
    img = np.random.default_rng(16).random((6, 7, 3), dtype=np.float32)
    img[:, :, 1] = 1
    stretch = (0, False, 1.0, 0.0, 255.0)
    assert_native_matches("chainner:image:stretch_contrast", [img, *stretch])
    result = stretch_contrast_node(img, StretchMode.AUTO, *stretch[1:])
    np.testing.assert_array_equal(result[:, :, 1], 1)
    for channel in (0, 2):
        assert result[:, :, channel].min() == 0 and result[:, :, channel].max() == 1


@pytest.mark.parametrize("stretch", AUTOMATIC_STRETCHES)
@pytest.mark.parametrize("channels", CHANNELS)
def test_stretch_contrast_takes_its_range_from_the_values_beside_a_nan(stretch, channels):
    shape = (16, 17, channels) if channels > 1 else (16, 17)
    img = np.random.default_rng(17).random(shape, dtype=np.float32)
    colour = img[..., :3] if channels == 4 else img
    colour[(3, 3, 0) if channels > 1 else (3, 3)] = np.nan
    assert_native_matches("chainner:image:stretch_contrast", [img, *stretch], declines=True)
    result = stretch_contrast_node(img, StretchMode(stretch[0]), *stretch[1:])
    resultcolour = result[..., :3] if channels == 4 else result
    assert np.isnan(resultcolour).sum() == 1

    def expected(values):
        low, high = np.nanpercentile(values, [stretch[2], 100 - stretch[2]]) if stretch[0] == 1 else (np.nanmin(values), np.nanmax(values))
        return (values.astype(np.float64) - low) / (high - low)

    keep_together = stretch[1] or channels == 1
    wanted = expected(colour) if keep_together else np.dstack([expected(colour[:, :, i]) for i in range(3)])
    np.testing.assert_allclose(resultcolour, wanted, atol=1e-4)


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("stretch", STRETCHES)
def test_stretch_contrast_empty_image(stretch, shape):
    declines = stretch[0] != 2
    assert_native_matches("chainner:image:stretch_contrast", [np.zeros(shape, np.float32), *stretch], declines=declines)
