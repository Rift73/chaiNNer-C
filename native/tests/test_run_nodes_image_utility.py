"""Native image_utility entry points (native/src/run/nodes/image_utility.cpp) against their Python bodies."""

import zlib

import numpy as np
import pytest

from native_differential import assert_native_matches
from nodes.impl.image_utils import FlipAxis
from nodes.impl.native_graph import graph

SIZES = [(1, 1), (3, 5), (17, 31), (64, 257)]
CHANNELS = [1, 3, 4]
# NaN with two payloads, both zeros, both infinities, a denormal, out-of-range values.
EDGES = np.array(
    [np.nan, -0.0, 0.0, np.inf, -np.inf, 1e-45, -0.25, 1.75, 1.0], dtype=np.float32
)
EDGES = np.concatenate([EDGES, np.array([0x7FC00123, 0xFFC00001], np.uint32).view(np.float32)])


def image(seed: int, height: int, width: int, channels: int, layout: str) -> np.ndarray:
    """Seeded noise with every edge value, as an enforced image: C order, or Lens's planar transpose."""
    rng = np.random.default_rng(seed)
    values = rng.random((height, width, channels), dtype=np.float32)
    flat = values.reshape(-1)
    flat[: min(flat.size, EDGES.size)] = EDGES[: flat.size]
    if layout == "permuted":
        values = np.ascontiguousarray(values.transpose(2, 0, 1)).transpose(1, 2, 0)
    return values[:, :, 0] if channels == 1 else values


@pytest.mark.parametrize("layout", ["c", "permuted"])
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("axis", [a.value for a in FlipAxis])
def test_flip_matches(axis, size, channels, layout):
    seed = zlib.crc32(repr((axis, size, channels, layout)).encode())
    height, width = size
    img = image(seed, height, width, channels, layout)
    assert_native_matches("chainner:image:flip", [img, axis])


def test_flip_none_returns_the_input_itself():
    img = image(1, 9, 11, 3, "c")
    img.setflags(write=False)
    assert graph().run_native("chainner:image:flip", [img, FlipAxis.NONE])[0] is img


@pytest.mark.parametrize("shape", [(0, 5, 3), (4, 0), (0, 0, 4)])
def test_flip_empty_image_error(shape):
    outcome = assert_native_matches("chainner:image:flip", [np.zeros(shape, np.float32), 1])
    assert outcome == ("error", ("ValueError", "Expected a nonempty two- or three-dimensional image"))
