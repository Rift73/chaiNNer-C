"""Native image_channel entry points (native/src/run/nodes/image_channel.cpp) against their Python bodies."""

import numpy as np
import pytest
from native_differential import assert_native_matches
from native_images import image, seed_of

from nodes.impl.native_graph import graph

SIZES = [(1, 1), (3, 5), (17, 31), (64, 257)]


@pytest.mark.parametrize("layout", ["c", "permuted"])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("size", SIZES)
def test_split_channels_matches(size, channels, layout):
    height, width = size
    img = image(
        seed_of("split", size, channels, layout), height, width, channels, layout
    )
    assert_native_matches("chainner:image:split_channels", [img])


def test_split_channels_grey_returns_the_input_and_one_image_of_ones():
    img = image(1, 9, 11, 1)
    img.setflags(write=False)
    red, green, blue, alpha = graph().run_native("chainner:image:split_channels", [img])
    assert red is img
    assert np.shares_memory(green, blue) and np.shares_memory(green, alpha)
    assert green.shape == (9, 11) and np.all(green == 1)


@pytest.mark.parametrize("shape", [(0, 5, 3), (4, 0, 4), (0, 0)])
def test_split_channels_empty_image_error(shape):
    outcome = assert_native_matches(
        "chainner:image:split_channels", [np.zeros(shape, np.float32)]
    )
    assert outcome[0] == "error"
