"""Normal to Height: planted surfaces with analytic slopes come back, in each format, tileable
or not; the Normal Map Generator round trip; flat maps, RGBA and tiny inputs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from nodes.impl.image_utils import NormalMapType
from nodes.impl.normals.edge_filter import EdgeFilter
from nodes.impl.normals.height import HeightSource
from nodes.impl.normals.util import xyz_to_octahedral_bgr
from packages.chaiNNer_standard.material_textures import normal_map_group
from packages.chaiNNer_standard.material_textures.normal_map.normal_map_generator import (
    AlphaOutput,
    normal_map_generator_node,
)
from packages.chaiNNer_standard.material_textures.normal_map.normal_to_height import (
    normal_to_height_node,
)

FORMATS = tuple(NormalMapType)
NODE = next(
    node
    for node in normal_map_group.nodes
    if node.schema_id == "chainner:image:normal_to_height"
)


def height_map(
    img: np.ndarray, normal_format: NormalMapType, tileable: bool
) -> np.ndarray:
    return normal_to_height_node(img, normal_format, tileable)[0]


def stretched(height: np.ndarray) -> np.ndarray:
    return (height - height.min()) / (height.max() - height.min())


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a - b))))


def encode(
    slope_x: np.ndarray, slope_y: np.ndarray, normal_format: NormalMapType
) -> np.ndarray:
    """The BGR normal map of a surface from its slopes per column and per row."""
    length = np.sqrt(np.square(slope_x) + np.square(slope_y) + 1)
    x = (-slope_x / length).astype(np.float32)
    y = (-slope_y / length).astype(np.float32)
    z = (1 / length).astype(np.float32)
    if normal_format is NormalMapType.OCTAHEDRAL:
        return xyz_to_octahedral_bgr((x, y, z))
    if normal_format is NormalMapType.OPENGL:
        y = -y
    return np.dstack((z, (y + 1) / 2, (x + 1) / 2))


def waves(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A surface that wraps around, and its slopes per column and per row."""
    row, column = np.mgrid[:height, :width].astype(np.float64)
    a = 2 * np.pi * (3 * column / width + 2 * row / height)
    b = 2 * np.pi * (column / width - 5 * row / height)
    surface = 4 * np.sin(a) + 2 * np.cos(b)
    slope_x = 4 * np.cos(a) * 2 * np.pi * 3 / width - 2 * np.sin(b) * 2 * np.pi / width
    slope_y = (
        4 * np.cos(a) * 2 * np.pi * 2 / height + 2 * np.sin(b) * 2 * np.pi * 5 / height
    )
    return surface, slope_x, slope_y


def bumps(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A surface that does not wrap around, with bumps cut by the edges, its overall tilt
    taken out, and its slopes per column and per row."""
    row, column = np.mgrid[:height, :width].astype(np.float64)
    surface = np.zeros((height, width))
    slope_x = np.zeros((height, width))
    slope_y = np.zeros((height, width))
    for y0, x0, sigma, amplitude in (
        (0.1, 0.0, 0.16, 18),
        (0.95, 0.35, 0.12, -14),
        (0.5, 1.0, 0.2, 22),
        (0.45, 0.45, 0.1, 9),
    ):
        y_dist = row - y0 * height
        x_dist = column - x0 * width
        s = sigma * width
        bump = amplitude * np.exp(
            -(np.square(y_dist) + np.square(x_dist)) / (2 * s * s)
        )
        surface += bump
        slope_x += -x_dist / (s * s) * bump
        slope_y += -y_dist / (s * s) * bump
    mean_x, mean_y = np.mean(slope_x), np.mean(slope_y)
    surface -= mean_x * column + mean_y * row
    return surface, slope_x - mean_x, slope_y - mean_y


@pytest.mark.parametrize("normal_format", FORMATS)
@pytest.mark.parametrize("size", ((64, 96), (65, 96), (64, 97), (65, 97)))
def test_a_surface_that_wraps_around_comes_back(
    normal_format: NormalMapType, size: tuple[int, int]
):
    surface, slope_x, slope_y = waves(*size)
    normal_map = encode(slope_x, slope_y, normal_format)
    result = height_map(normal_map, normal_format, True)
    assert result.dtype == np.float32 and result.shape == size
    assert rms(result, stretched(surface)) < 1e-3


@pytest.mark.parametrize("normal_format", FORMATS)
@pytest.mark.parametrize("size", ((128, 192), (129, 192), (128, 193)))
def test_a_surface_cut_by_the_edges_comes_back(
    normal_format: NormalMapType, size: tuple[int, int]
):
    surface, slope_x, slope_y = bumps(*size)
    normal_map = encode(slope_x, slope_y, normal_format)
    truth = stretched(surface)
    mirrored = rms(height_map(normal_map, normal_format, False), truth)
    wrapped = rms(height_map(normal_map, normal_format, True), truth)
    assert mirrored < 0.02
    # Wrapping around ties the edges together, which this surface does not have.
    assert wrapped > 2 * mirrored


def test_height_range_gives_the_surface_in_pixels():
    surface, slope_x, slope_y = waves(64, 96)
    normal_map = encode(slope_x, slope_y, NormalMapType.DIRECTX)
    _, height_range = normal_to_height_node(normal_map, NormalMapType.DIRECTX, True)
    assert height_range == pytest.approx(np.ptp(surface), rel=2e-3)

    surface, slope_x, slope_y = bumps(128, 192)
    normal_map = encode(slope_x, slope_y, NormalMapType.DIRECTX)
    _, height_range = normal_to_height_node(normal_map, NormalMapType.DIRECTX, False)
    assert height_range == pytest.approx(np.ptp(surface), rel=2e-3)


def test_height_range_tells_tiles_of_one_material_apart():
    # The same shape at a third of the height: the same image, a third of the range.
    surface, slope_x, slope_y = waves(64, 96)
    tall = encode(slope_x, slope_y, NormalMapType.DIRECTX)
    low = encode(slope_x / 3, slope_y / 3, NormalMapType.DIRECTX)
    tall_image, tall_range = normal_to_height_node(tall, NormalMapType.DIRECTX, True)
    low_image, low_range = normal_to_height_node(low, NormalMapType.DIRECTX, True)
    assert rms(tall_image, low_image) < 1e-3
    assert tall_range == pytest.approx(3 * low_range, rel=2e-3)
    # Height Range turns the stretched image back into the surface in pixels.
    assert rms(tall_image * tall_range, surface - surface.min()) < 0.02


@pytest.mark.parametrize("tileable", (False, True))
@pytest.mark.parametrize("normal_format", FORMATS)
def test_a_flat_map_has_no_height_range(normal_format: NormalMapType, tileable: bool):
    normal_map = np.dstack(
        (
            np.ones((16, 24), np.float32),
            np.full((16, 24), 128 / 255, np.float32),
            np.full((16, 24), 128 / 255, np.float32),
        )
    )
    image, height_range = normal_to_height_node(normal_map, normal_format, tileable)
    assert height_range == 0
    assert np.all(image == 0.5)
    _, height_range = normal_to_height_node(
        np.zeros((16, 24, 3), np.float32), normal_format, tileable
    )
    assert height_range == 0


def test_a_wrapped_height_map_continues_across_its_edges():
    surface, slope_x, slope_y = waves(64, 96)
    normal_map = encode(slope_x, slope_y, NormalMapType.DIRECTX)
    result = height_map(normal_map, NormalMapType.DIRECTX, True)
    across = np.abs(result[:, 0] - result[:, -1]).max()
    expected = np.abs(stretched(surface)[:, 0] - stretched(surface)[:, -1]).max()
    assert across == pytest.approx(expected, abs=2e-3)


@pytest.mark.parametrize("axis", (0, 1))
def test_a_surface_alternating_at_the_sampling_limit_comes_back(axis: int):
    # Along its alternation it has no slope at the samples, so only the other axis sees it.
    # The plain wave beside it shows the alternation's amplitude after the stretch to 0-1.
    alternating, along = np.mgrid[:32, :48].astype(np.float64)
    if axis == 1:
        along, alternating = np.mgrid[:48, :32].astype(np.float64)
    frequency = 2 * np.pi * 3 / 48
    sign = 1 - 2 * (alternating % 2)
    surface = 5 * sign * np.sin(frequency * along) + 9 * np.cos(frequency * along)
    slope = frequency * (
        5 * sign * np.cos(frequency * along) - 9 * np.sin(frequency * along)
    )
    other = np.zeros_like(slope)
    slope_x, slope_y = (slope, other) if axis == 0 else (other, slope)
    normal_map = encode(slope_x, slope_y, NormalMapType.DIRECTX)
    result = height_map(normal_map, NormalMapType.DIRECTX, True)
    assert rms(result, stretched(surface)) < 1e-3


def test_8_bit_values_keep_the_shape():
    surface, slope_x, slope_y = bumps(128, 192)
    normal_map = encode(slope_x, slope_y, NormalMapType.OPENGL)
    normal_map = (np.round(normal_map * 255) / 255).astype(np.float32)
    result = height_map(normal_map, NormalMapType.OPENGL, False)
    assert rms(result, stretched(surface)) < 0.03


@pytest.mark.parametrize(
    ("written", "read"),
    (
        (NormalMapType.DIRECTX, NormalMapType.OPENGL),
        (NormalMapType.OPENGL, NormalMapType.DIRECTX),
    ),
)
def test_the_other_format_does_not_give_the_surface(
    written: NormalMapType, read: NormalMapType
):
    surface, slope_x, slope_y = waves(64, 96)
    normal_map = encode(slope_x, slope_y, written)
    assert rms(height_map(normal_map, read, True), stretched(surface)) > 0.1


def bright_center(size: int) -> np.ndarray:
    row, column = np.mgrid[:size, :size].astype(np.float32)
    middle = (size - 1) / 2
    radius = np.square(row - middle) + np.square(column - middle)
    return np.exp(-radius / (2 * (size / 6) ** 2)).astype(np.float32)


def generated(source: np.ndarray, invert_g: bool) -> np.ndarray:
    return normal_map_generator_node(
        source,
        False,
        HeightSource.AVERAGE_RGB,
        0,
        0,
        4,
        EdgeFilter.SOBEL,
        *(0.0,) * 8,
        False,
        invert_g,
        AlphaOutput.NONE,
    )


def test_the_default_format_undoes_normal_map_generator():
    source = bright_center(128)
    default = NODE.inputs[1].default
    result = height_map(generated(source, False), NormalMapType(default), False)
    assert rms(result, stretched(source)) < 0.03
    assert result[64, 64] > 0.9 and result[0, 0] < 0.2


def test_inverted_g_of_the_generator_is_opengl():
    source = bright_center(128)
    result = height_map(generated(source, True), NormalMapType.OPENGL, False)
    assert rms(result, stretched(source)) < 0.03


@pytest.mark.parametrize("tileable", (False, True))
@pytest.mark.parametrize("normal_format", FORMATS)
@pytest.mark.parametrize(
    "value",
    # Flat normal maps as stored in 8 bits: their bias is 0, +1/255 or -1/255.
    (0.5, 128 / 255, 127 / 255),
)
def test_a_flat_normal_map_gives_middle_gray(
    normal_format: NormalMapType, tileable: bool, value: float
):
    normal_map = np.dstack(
        (
            np.ones((32, 48), np.float32),
            np.full((32, 48), value, np.float32),
            np.full((32, 48), value, np.float32),
        )
    )
    result = height_map(normal_map, normal_format, tileable)
    assert np.all(result == 0.5)


@pytest.mark.parametrize("tileable", (False, True))
@pytest.mark.parametrize("normal_format", FORMATS)
def test_alpha_and_blue_have_no_effect(normal_format: NormalMapType, tileable: bool):
    _, slope_x, slope_y = waves(32, 48)
    normal_map = encode(slope_x, slope_y, normal_format)
    rng = np.random.default_rng(3)
    noise = rng.random((32, 48), dtype=np.float32)
    other_blue = normal_map.copy()
    other_blue[:, :, 0] = noise
    rgba = np.dstack((normal_map, noise))
    expected = height_map(normal_map, normal_format, tileable)
    assert np.array_equal(height_map(other_blue, normal_format, tileable), expected)
    assert np.array_equal(height_map(rgba, normal_format, tileable), expected)


def test_grayscale_is_refused_as_on_the_other_normal_nodes():
    assert NODE.inputs[0].channels == [3, 4]
    with pytest.raises(ValueError, match="only supports"):
        NODE.inputs[0].enforce(np.zeros((4, 4), np.float32))


@pytest.mark.parametrize("tileable", (False, True))
@pytest.mark.parametrize("channels", (3, 4))
@pytest.mark.parametrize(
    "size", ((1, 1), (1, 2), (2, 1), (2, 2), (1, 7), (7, 1), (2, 3), (3, 2), (3, 3))
)
def test_tiny_inputs(size: tuple[int, int], channels: int, tileable: bool):
    rng = np.random.default_rng(size[0] * 10 + size[1])
    normal_map = rng.random((*size, channels), dtype=np.float32)
    result = height_map(normal_map, NormalMapType.OPENGL, tileable)
    assert result.shape == size
    assert np.all(np.isfinite(result))
    assert result.min() >= 0 and result.max() <= 1
    if size == (1, 1):
        assert result[0, 0] == 0.5


def test_a_ramp_of_one_pixel_rows_comes_back_when_not_tiled():
    # Up then down along a single row.
    surface = np.array([0, 1, 2, 3, 2, 1, 0], np.float64)
    slope = np.gradient(surface)[None]
    normal_map = encode(slope, np.zeros_like(slope), NormalMapType.OPENGL)
    result = height_map(normal_map, NormalMapType.OPENGL, False)[0]
    assert result.argmax() == 3
    assert result[0] < 0.2 and result[6] < 0.2


@pytest.mark.parametrize("tileable", (False, True))
def test_normals_in_the_image_plane_stay_finite(tileable: bool):
    # R at 1 and G in the middle is a normal with no Z left.
    normal_map = np.full((16, 16, 3), 0.5, np.float32)
    normal_map[4:8, 4:8, 2] = 1
    normal_map[10:12, 10:12, 2] = 0
    result = height_map(normal_map, NormalMapType.OPENGL, tileable)
    assert np.all(np.isfinite(result))
    assert result.min() == 0 and result.max() == 1


def test_a_normal_in_the_image_plane_is_a_step_of_the_slope_limit():
    # R at 0 and G in the middle, on an otherwise flat row: a slope of 10 on one pixel.
    normal_map = np.full((1, 65, 3), 0.5, np.float32)
    normal_map[0, 32, 2] = 0
    _, height_range = normal_to_height_node(normal_map, NormalMapType.DIRECTX, False)
    assert 8 < height_range < 14


# UV padding of game textures: the channels of a pixel that holds no normal.
NO_NORMALS = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "red and green": (0.0, 1.0, 1.0),
    "green only": (0.0, 1.0, 0.0),
}


@pytest.mark.parametrize("normal_format", (NormalMapType.DIRECTX, NormalMapType.OPENGL))
@pytest.mark.parametrize("value", NO_NORMALS.values(), ids=NO_NORMALS.keys())
@pytest.mark.parametrize(("size", "limit"), ((1, 0.002), (4, 0.01), (16, 0.05)))
def test_pixels_that_are_no_normals_leave_the_surface(
    normal_format: NormalMapType, value: tuple[float, ...], size: int, limit: float
):
    surface, slope_x, slope_y = bumps(128, 192)
    normal_map = encode(slope_x, slope_y, normal_format)
    normal_map[:size, :size] = value
    result = height_map(normal_map, normal_format, False)
    assert rms(result, stretched(surface)) < limit


@pytest.mark.parametrize(("size", "limit"), ((1, 0.002), (4, 0.01), (16, 0.05)))
def test_black_is_no_normal_in_octahedral_either(size: int, limit: float):
    # Every octahedral code decodes to a normal; black's lies in the image plane.
    surface, slope_x, slope_y = bumps(128, 192)
    normal_map = encode(slope_x, slope_y, NormalMapType.OCTAHEDRAL)
    normal_map[:size, :size] = 0
    result = height_map(normal_map, NormalMapType.OCTAHEDRAL, False)
    assert rms(result, stretched(surface)) < limit


def test_a_tileable_surface_keeps_its_shape_around_black_pixels():
    surface, slope_x, slope_y = waves(64, 96)
    normal_map = encode(slope_x, slope_y, NormalMapType.DIRECTX)
    normal_map[10, 20] = 0
    result = height_map(normal_map, NormalMapType.DIRECTX, True)
    assert rms(result, stretched(surface)) < 0.01


@pytest.mark.parametrize("tileable", (False, True))
@pytest.mark.parametrize("value", (128 / 255, 127 / 255))
@pytest.mark.parametrize("normal_format", (NormalMapType.DIRECTX, NormalMapType.OPENGL))
def test_a_flat_map_with_padding_gives_middle_gray(
    normal_format: NormalMapType, value: float, tileable: bool
):
    # The tilt comes from the normals alone: the bias of 8 bits must not become a ramp.
    normal_map = np.full((32, 48, 3), value, np.float32)
    normal_map[:, :, 0] = 1
    normal_map[:, 30:] = 0
    result = height_map(normal_map, normal_format, tileable)
    assert np.all(result == 0.5)


@pytest.mark.parametrize("tileable", (False, True))
@pytest.mark.parametrize("normal_format", FORMATS)
def test_a_map_of_black_gives_middle_gray(normal_format: NormalMapType, tileable: bool):
    normal_map = np.zeros((16, 24, 3), np.float32)
    assert np.all(height_map(normal_map, normal_format, tileable) == 0.5)


@pytest.mark.parametrize(
    "steps",
    # 8 bits per channel, and the 5 and 6 bits of BC1 textures' red and green.
    ((255, 255), (31, 63)),
)
@pytest.mark.parametrize("degrees", range(0, 360, 5))
def test_rounding_does_not_turn_a_steep_normal_into_no_normal(
    degrees: int, steps: tuple[int, int]
):
    # A normal in the image plane, in any direction, after the rounding of its channels.
    angle = np.deg2rad(degrees)
    red = np.round((np.cos(angle) + 1) / 2 * steps[0]) / steps[0]
    green = np.round((np.sin(angle) + 1) / 2 * steps[1]) / steps[1]
    normal_map = np.full((8, 8, 3), 0.5, np.float32)
    normal_map[4, 4, 1] = green
    normal_map[4, 4, 2] = red
    result = height_map(normal_map, NormalMapType.DIRECTX, False)
    assert np.any(result != 0.5)


@pytest.mark.parametrize("tileable", (False, True))
def test_a_permuted_image_gives_the_same_pixels(tileable: bool):
    _, slope_x, slope_y = waves(32, 48)
    normal_map = np.dstack(
        (encode(slope_x, slope_y, NormalMapType.OPENGL), np.ones((32, 48), np.float32))
    )
    permuted = np.moveaxis(normal_map, 2, 0).copy().transpose(1, 2, 0)
    assert not permuted.flags.c_contiguous
    assert np.array_equal(
        height_map(permuted, NormalMapType.OPENGL, tileable),
        height_map(normal_map, NormalMapType.OPENGL, tileable),
    )


def test_concurrent_runs_give_the_same_pixels():
    # The node may run ahead on the worker threads, and SciPy's transforms share a plan cache.
    inputs = [
        (encode(*waves(height, width)[1:], NormalMapType.DIRECTX), tileable)
        for height, width in ((64, 96), (65, 97), (48, 64), (97, 33))
        for tileable in (False, True)
    ] * 4
    expected = [
        height_map(img, NormalMapType.DIRECTX, tileable) for img, tileable in inputs
    ]
    with ThreadPoolExecutor(8) as pool:
        results = list(
            pool.map(
                lambda item: height_map(item[0], NormalMapType.DIRECTX, item[1]),
                inputs,
            )
        )
    for result, single in zip(results, expected):
        assert np.array_equal(result, single)
