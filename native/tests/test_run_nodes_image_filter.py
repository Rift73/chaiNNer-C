"""Native image_filter entry points (native/src/run/nodes/image_filter.cpp, image_blur.cpp) against their Python
bodies."""

import numpy as np
import pytest
from native_differential import assert_native_matches
from native_images import image, seed_of

from nodes.impl.native_graph import graph

GAUSSIAN = "chainner:image:gaussian_blur"
SIZES = [(1, 1), (3, 5), (17, 31), (64, 257)]


@pytest.mark.parametrize(
    "sigmas",
    [
        (0, 2.5),
        (2.5, 0),
        (2.5, 1.25),
        (3, 3),
        (6, 6),
        (0.1, 10.9),
        (10.9, 11),
        (11, 10.9),
        (1000, 0.5),
    ],
)
@pytest.mark.parametrize("edges", [True, False])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("size", SIZES)
def test_gaussian_blur_matches(size, channels, edges, sigmas):
    height, width = size
    key = ("gaussian", size, channels, edges, sigmas)
    img = image(seed_of(*key), height, width, channels, edges=edges)
    assert_native_matches(GAUSSIAN, [img, *sigmas])


@pytest.mark.parametrize("sigmas", [(11, 11), (25, 1000), (1000, 1000)])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_gaussian_blur_one_pixel_downsampling_sigmas(channels, sigmas):
    # A 1x1 image is not downsampled: its post blur uses sigma size / size_down = 1.
    img = image(seed_of("gaussian1", channels, sigmas), 1, 1, channels)
    assert_native_matches(GAUSSIAN, [img, *sigmas])


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_gaussian_blur_zero_radii_return_the_input_itself(channels):
    img = image(seed_of("gaussian0", channels), 9, 11, channels)
    img.setflags(write=False)
    assert graph().run_native(GAUSSIAN, [img, 0.0, 0.0])[0] is img
    assert_native_matches(GAUSSIAN, [img, 0, 0])


@pytest.mark.parametrize(
    ("size", "sigmas"),
    [
        ((17, 31), (11, 11)),
        ((1, 31), (11, 12)),
        ((31, 1), (30, 200)),
        ((64, 257), (1000, 1000)),
    ],
)
def test_gaussian_blur_downsampling_declines(size, sigmas):
    # The body upsamples with cv2.resize (IPP LINEAR): the Python body runs.
    height, width = size
    img = image(seed_of("gaussian-decline", size, sigmas), height, width, 3)
    assert_native_matches(GAUSSIAN, [img, *sigmas], declines=True)


@pytest.mark.parametrize(
    "sigmas", [(0, 0), (0, 2.5), (2.5, 2.5), (11, 11), (11, 2), (20, 30)]
)
@pytest.mark.parametrize("shape", [(0, 5, 3), (4, 0), (0, 0, 4)])
def test_gaussian_blur_empty_image(shape, sigmas):
    assert_native_matches(GAUSSIAN, [np.zeros(shape, np.float32), *sigmas])


MORPHOLOGY = ["chainner:image:dilate", "chainner:image:erode"]


@pytest.mark.parametrize("iterations", [0, 1, 2, 3])
@pytest.mark.parametrize("radius", [0, 1, 2, 3, 7])
@pytest.mark.parametrize("shape", [0, 1, 2])
@pytest.mark.parametrize("edges", [True, False])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("schema_id", MORPHOLOGY)
def test_morphology_matches(
    schema_id, size, channels, edges, shape, radius, iterations
):
    height, width = size
    key = (schema_id, size, channels, edges, shape, radius, iterations)
    img = image(seed_of(*key), height, width, channels, edges=edges)
    assert_native_matches(schema_id, [img, shape, radius, iterations])


@pytest.mark.parametrize("schema_id", MORPHOLOGY)
def test_morphology_large_radius(schema_id):
    img = image(seed_of(schema_id, "large"), 40, 50, 3, edges=False)
    for shape in (0, 1, 2):
        assert_native_matches(schema_id, [img, shape, 1000, 2])
        assert_native_matches(schema_id, [img, shape, 30, 1000])


@pytest.mark.parametrize("schema_id", MORPHOLOGY)
def test_morphology_zero_radius_returns_the_input_itself(schema_id):
    img = image(seed_of(schema_id, "zero"), 9, 11, 3)
    img.setflags(write=False)
    assert graph().run_native(schema_id, [img, 0, 0, 3])[0] is img
    assert graph().run_native(schema_id, [img, 2, 3, 0])[0] is img


@pytest.mark.parametrize("radius", [0, 2])
@pytest.mark.parametrize("shape", [(0, 5, 3), (4, 0), (0, 0, 4)])
@pytest.mark.parametrize("schema_id", MORPHOLOGY)
def test_morphology_empty_image(schema_id, shape, radius):
    assert_native_matches(schema_id, [np.zeros(shape, np.float32), 2, radius, 1])


LENS = "chainner:image:lens_blur"


@pytest.mark.parametrize("gamma", [0.01, 0.5, 1, 2, 3, 100])
@pytest.mark.parametrize("components", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("radius", [1, 2, 5, 30, 64])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("size", [(1, 1), (3, 5), (17, 31), (40, 57)])
def test_lens_blur_matches(size, channels, radius, components, gamma):
    height, width = size
    key = (LENS, size, channels, radius, components, gamma)
    img = image(seed_of(*key), height, width, channels)
    assert_native_matches(LENS, [img, radius, components, gamma])


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_lens_blur_zero_radius_returns_the_input_itself(channels):
    img = image(seed_of(LENS, "zero", channels), 9, 11, channels)
    img.setflags(write=False)
    assert graph().run_native(LENS, [img, 0, 3, 5.0])[0] is img
    assert_native_matches(LENS, [img, 0, 3, 5])


@pytest.mark.parametrize("radius", [65, 100, 1000])
def test_lens_blur_spectral_radius_declines(radius):
    # Kernels of 130 taps or more take the body's cv2.dft path: the Python body runs.
    img = image(seed_of(LENS, "decline", radius), 17, 31, 3)
    assert_native_matches(LENS, [img, radius, 3, 3], declines=True)


@pytest.mark.parametrize("radius", [0, 3])
@pytest.mark.parametrize("shape", [(0, 5, 3), (4, 0), (0, 0, 4)])
def test_lens_blur_empty_image(shape, radius):
    assert_native_matches(LENS, [np.zeros(shape, np.float32), radius, 3, 3])


DITHER = "chainner:image:palette_dither"
MAPS = ["FS", "JJN", "ST", "A", "B", "S", "S2", "SL"]
ALGORITHMS = [("None", "FS", 16)]
ALGORITHMS += [("Diffusion", m, 16) for m in MAPS]
ALGORITHMS += [("Riemersma", "FS", h) for h in (2, 3, 16, 1000)]


def palette(seed: int, rows: int, colors: int, channels: int, edges: bool):
    return image(seed, rows, colors, channels, edges=edges)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("colors", [1, 2, 16, 300])
@pytest.mark.parametrize("palette_edges", [False, True])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("size", [(1, 1), (3, 5), (17, 31), (40, 57)])
def test_palette_dither_matches(size, channels, palette_edges, colors, algorithm):
    height, width = size
    key = (DITHER, size, channels, palette_edges, colors, algorithm)
    img = image(seed_of(*key), height, width, channels)
    # Two rows: only the first is the palette.
    colors_image = palette(seed_of(*key, "palette"), 2, colors, channels, palette_edges)
    name, diffusion_map, history = algorithm
    inputs: list[object] = [img, colors_image, name, diffusion_map, history]
    assert_native_matches(DITHER, inputs)


@pytest.mark.parametrize(
    ("image_channels", "palette_channels"), [(3, 4), (1, 3), (4, 1)]
)
def test_palette_dither_channel_mismatch(image_channels, palette_channels):
    img = image(1, 4, 5, image_channels)
    colors = palette(2, 1, 4, palette_channels, False)
    outcome = assert_native_matches(DITHER, [img, colors, "None", "FS", 16])
    message = "Image and palette must have the same number of channels."
    assert outcome == ("error", ("AssertionError", message))


@pytest.mark.parametrize(
    ("image_shape", "palette_shape"),
    [
        ((0, 5, 3), (1, 4, 3)),
        ((4, 0), (1, 4)),
        ((3, 4, 4), (0, 4, 4)),
        ((3, 4), (2, 0)),
    ],
)
def test_palette_dither_empty_error(image_shape, palette_shape):
    img = np.full(image_shape, 0.5, np.float32)
    colors = np.full(palette_shape, 0.25, np.float32)
    outcome = assert_native_matches(DITHER, [img, colors, "Diffusion", "FS", 16])
    assert outcome[0] == "error"


@pytest.mark.parametrize("history", [682541920, 2**32, 2**40, 2**70])
def test_palette_dither_history_range_error(history):
    # Above 682541919 Riemersma's float32 decay base rounds to 1 (native_neighborhood).
    img, colors = image(1, 4, 5, 3), palette(2, 1, 4, 3, False)
    outcome = assert_native_matches(DITHER, [img, colors, "Riemersma", "FS", history])
    message = "History length must be an integer in [2,682541919]"
    assert outcome == ("error", ("ValueError", message))
