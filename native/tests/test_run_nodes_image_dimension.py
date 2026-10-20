"""Native image_dimension entry points (native/src/run/nodes/image_dimension.cpp, image_resize.cpp) against their
Python bodies."""

import numpy as np
import pytest
from native_differential import assert_native_matches
from native_images import image, seed_of

from nodes.impl.native_graph import graph
from nodes.impl.resize import ResizeFilter

RESIZE = "chainner:image:resize"
SIZES = [(1, 1), (3, 5), (17, 31), (64, 257)]
FILTERS = [f.value for f in ResizeFilter]
PERCENT, ABSOLUTE = 0, 1


def resize_inputs(
    img: np.ndarray,
    mode: int,
    scale: float,
    target: tuple[int, int],
    filter_value: int,
    separate_alpha: bool,
    dpid_lambda: float = 1.0,
) -> list[object]:
    """Resize's inputs; target is (width, height) for Absolute."""
    width, height = target
    return [img, mode, scale, width, height, filter_value, separate_alpha, dpid_lambda]


@pytest.mark.parametrize("separate_alpha", [False, True])
@pytest.mark.parametrize("filter_value", FILTERS)
@pytest.mark.parametrize("scale", [0, 12.5, 50, 100, 150, 233.3333])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("size", SIZES)
def test_resize_percentage_matches(size, channels, scale, filter_value, separate_alpha):
    height, width = size
    key = ("resize%", size, channels, scale, filter_value, separate_alpha)
    img = image(seed_of(*key), height, width, channels)
    inputs = resize_inputs(img, PERCENT, scale, (1, 1), filter_value, separate_alpha)
    assert_native_matches(RESIZE, inputs)


@pytest.mark.parametrize("separate_alpha", [False, True])
@pytest.mark.parametrize("filter_value", FILTERS)
@pytest.mark.parametrize(
    "target", [(1, 1), (5, 3), (31, 17), (2, 40), (40, 2), (64, 64)]
)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_resize_absolute_matches(channels, target, filter_value, separate_alpha):
    key = ("resize", channels, target, filter_value, separate_alpha)
    img = image(seed_of(*key), 17, 31, channels)
    inputs = resize_inputs(img, ABSOLUTE, 100, target, filter_value, separate_alpha)
    assert_native_matches(RESIZE, inputs)


@pytest.mark.parametrize("dpid_lambda", [0.0, 0.35, 2.5, 10.0])
@pytest.mark.parametrize("separate_alpha", [False, True])
@pytest.mark.parametrize("target", [(5, 3), (31, 17), (2, 40), (13, 9), (64, 64)])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_resize_dpid_lambda_matches(channels, target, separate_alpha, dpid_lambda):
    key = ("resize dpid", channels, target, separate_alpha, dpid_lambda)
    img = image(seed_of(*key), 37, 53, channels)
    inputs = resize_inputs(
        img, ABSOLUTE, 100, target, ResizeFilter.DPID.value, separate_alpha, dpid_lambda
    )
    assert assert_native_matches(RESIZE, inputs)[0] == "outputs"


@pytest.mark.parametrize(
    "filter_value",
    [
        ResizeFilter.NEAREST.value,
        ResizeFilter.BOX.value,
        ResizeFilter.AUTO.value,
        ResizeFilter.DPID.value,
    ],
)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_resize_same_size_returns_the_input_itself(channels, filter_value):
    img = image(seed_of("same", channels, filter_value), 9, 11, channels)
    img.setflags(write=False)
    inputs = resize_inputs(img, ABSOLUTE, 100, (11, 9), filter_value, False)
    assert graph().run_native(RESIZE, inputs)[0] is img
    assert_native_matches(RESIZE, inputs)


@pytest.mark.parametrize(
    ("shape", "mode", "scale", "target"),
    [
        ((0, 5, 3), PERCENT, 50, (1, 1)),
        ((4, 0), ABSOLUTE, 100, (3, 3)),
        ((0, 0, 4), ABSOLUTE, 100, (2, 2)),
        # Percentage: a side of 0 pixels still gives one.
        ((0, 7), PERCENT, 0, (1, 1)),
    ],
)
@pytest.mark.parametrize("filter_value", FILTERS)
def test_resize_empty_image_error(shape, mode, scale, target, filter_value):
    inputs = resize_inputs(
        np.zeros(shape, np.float32), mode, scale, target, filter_value, False
    )
    outcome = assert_native_matches(RESIZE, inputs)
    assert outcome == (
        "error",
        ("ValueError", "Expected a nonempty two- or three-dimensional image"),
    )


@pytest.mark.parametrize(
    ("shape", "mode", "scale", "target"),
    [
        # 2^24 x 257 x 1 x 4 bytes = 16.0625 GiB: round(.., 3) ties to even (16.062).
        ((1, 1), ABSOLUTE, 100, (2**24, 257)),
        ((2, 2, 3), ABSOLUTE, 100, (100000, 100000)),
        ((2, 2, 4), ABSOLUTE, 100, (2**31 - 1, 2**31 - 1)),
        ((2, 2, 3), ABSOLUTE, 100, (2**62, 3)),
        ((3, 5, 3), PERCENT, 1e8, (1, 1)),
        # Sides beyond int64: the message in exponent notation, then a quotient beyond a double.
        ((3, 5, 3), PERCENT, 1e150, (1, 1)),
        ((3, 5, 4), PERCENT, 1e300, (1, 1)),
        ((3, 5), PERCENT, float("inf"), (1, 1)),
        # 0 * inf is NaN.
        ((3, 0), PERCENT, float("inf"), (1, 1)),
    ],
)
def test_resize_size_errors(shape, mode, scale, target):
    img = np.full(shape, 0.5, np.float32)
    inputs = resize_inputs(img, mode, scale, target, ResizeFilter.LANCZOS.value, False)
    outcome = assert_native_matches(RESIZE, inputs)
    assert outcome[0] == "error"


def test_resize_memory_message_text():
    inputs = resize_inputs(
        np.zeros((1, 1), np.float32), ABSOLUTE, 100, (2**24, 257), 1, False
    )
    outcome = assert_native_matches(RESIZE, inputs)
    message = "Resize would require 16.062 GB of memory, but only 16 GB are allowed."
    assert outcome == ("error", ("RuntimeError", message))


def test_resize_coordinate_limit_error():
    # Exactly 16 GiB passes the memory check; the bridge refuses a side above 2^32 - 1.
    inputs = resize_inputs(
        np.zeros((1, 1), np.float32), ABSOLUTE, 100, (2**32, 1), 0, False
    )
    outcome = assert_native_matches(RESIZE, inputs)
    message = "Resize dimensions exceed the native coordinate range"
    assert outcome == ("error", ("OverflowError", message))


@pytest.mark.parametrize(
    ("width", "height", "filter_value"),
    [
        # A Python int beyond int64.
        (2**70, 1, 1),
        (1, 2**64, 0),
        # A side beyond the engine's int32 images (the body would return it).
        (2**31, 1, 0),
        (1, 2**31, 2),
    ],
)
def test_resize_declines(width, height, filter_value):
    inputs = resize_inputs(
        np.zeros((1, 1), np.float32),
        ABSOLUTE,
        100,
        (width, height),
        filter_value,
        False,
    )
    assert_native_matches(RESIZE, inputs, declines=True)
