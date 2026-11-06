"""Normal Map Generator against planted truth: slopes whose normal is known, a tileable
texture whose map must not change where the tile's edges fall, and Scale's limit."""

from __future__ import annotations

import numpy as np
import pytest

from nodes.impl.normals.edge_filter import EdgeFilter
from nodes.impl.normals.height import HeightSource
from packages.chaiNNer_standard.material_textures.normal_map.normal_map_generator import (
    AlphaOutput,
    normal_map_generator_node,
)

GAUSS = (0.25, 0.5, 0.3, 0.25, 0.2, 0.15, 0.1, 0.1)
Texture = np.ndarray


def normals(
    image: Texture,
    *,
    tileable: bool = False,
    blur_sharp: float = 0,
    scale: float = 1,
    edge_filter: EdgeFilter = EdgeFilter.SOBEL,
    alpha: AlphaOutput = AlphaOutput.NONE,
) -> np.ndarray:
    return normal_map_generator_node(
        image,
        tileable,
        HeightSource.AVERAGE_RGB,
        blur_sharp,
        0,
        scale,
        edge_filter,
        *GAUSS,
        False,
        False,
        alpha,
    )


def noise(height: int = 64, width: int = 48) -> Texture:
    return np.random.default_rng(3).random((height, width), dtype=np.float32)


@pytest.mark.parametrize("edge_filter", [EdgeFilter.SOBEL, EdgeFilter.FOUR_SAMPLE])
@pytest.mark.parametrize("scale", [0.5, 1, 4])
def test_a_slope_gives_its_normal(edge_filter, scale):
    slope = 1 / 128
    ramp = np.tile(np.arange(64, dtype=np.float32) * slope, (64, 1))
    # The filters measure h(x - 1) - h(x + 1), and z is 2 before normalizing.
    dx = -2 * slope * scale
    length = np.hypot(dx, 2)
    expected = (2 / length, 0.5, 0.5 + dx / length / 2)
    across = normals(ramp, scale=scale, edge_filter=edge_filter)
    np.testing.assert_allclose(
        across[8:-8, 8:-8], np.full((48, 48, 3), expected), atol=2e-5
    )
    down = normals(ramp.T, scale=scale, edge_filter=edge_filter)
    np.testing.assert_allclose(
        down[8:-8, 8:-8],
        np.full((48, 48, 3), expected[:1] + expected[:0:-1]),
        atol=2e-5,
    )


def test_scale_zero_is_flat():
    image = noise()
    flat = normals(image, scale=0, alpha=AlphaOutput.HEIGHT)
    np.testing.assert_array_equal(
        flat, np.full((*image.shape, 4), (1, 0.5, 0.5, 0), np.float32)
    )
    # The limit of the small scales, which the slider reaches continuously.
    near = normals(image, scale=1e-5)
    np.testing.assert_allclose(near, flat[:, :, :3], atol=1e-4)


TILED = [
    ("sobel", {}),
    ("sobel-like-9", {"edge_filter": EdgeFilter.SOBEL_LIKE_9}),
    ("4-sample", {"edge_filter": EdgeFilter.FOUR_SAMPLE}),
    ("multi-gauss", {"edge_filter": EdgeFilter.MULTI_GAUSS}),
    ("blur 0.4", {"blur_sharp": -0.4}),
    ("blur 2", {"blur_sharp": -2}),
    (
        "blur 7.5 sobel-like-9",
        {"blur_sharp": -7.5, "edge_filter": EdgeFilter.SOBEL_LIKE_9},
    ),
    ("sharpen 0.4", {"blur_sharp": 0.4}),
    ("sharpen 3 scharr", {"blur_sharp": 3, "edge_filter": EdgeFilter.SCHARR}),
]


@pytest.mark.parametrize(("label", "options"), TILED, ids=[t[0] for t in TILED])
@pytest.mark.parametrize("size", [(64, 48), (5, 3), (1, 7)])
def test_a_tileable_map_is_the_map_of_the_endless_texture(label, options, size):
    # The middle tile of many tiles (past the blur's and the filter's reach) is the truth.
    image = noise(*size)
    repeats = (2 * 80 // size[0] + 3) | 1, (2 * 80 // size[1] + 3) | 1
    many = normals(np.tile(image, repeats), **options)
    top, left = repeats[0] // 2 * size[0], repeats[1] // 2 * size[1]
    truth = many[top : top + size[0], left : left + size[1]]
    np.testing.assert_allclose(
        normals(image, tileable=True, **options), truth, atol=2e-6
    )


@pytest.mark.parametrize("blur_sharp", [-11, -20, 15])
def test_a_tileable_map_stays_seamless_under_the_fast_blur(blur_sharp):
    # A blur past sigma 11 downsamples, on a grid that moves with the padded size.
    image = noise()
    shift = (17, 23)
    moved = normals(
        np.roll(image, shift, axis=(0, 1)), tileable=True, blur_sharp=blur_sharp
    )
    plain = normals(image, tileable=True, blur_sharp=blur_sharp)
    np.testing.assert_allclose(np.roll(plain, shift, axis=(0, 1)), moved, atol=1e-3)
