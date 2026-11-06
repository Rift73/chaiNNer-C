"""Crop to Size: the part each anchor keeps, an odd excess, short sides kept or padded, on
planted coordinate images in gray, RGB and RGBA, down to 1x1."""

from __future__ import annotations

import numpy as np
import pytest

from nodes.impl.color.color import Color
from nodes.impl.image_utils import BorderType
from nodes.properties.inputs import Anchor
from packages.chaiNNer_standard.image_dimension.crop.crop_to_size import (
    CropIfSmaller,
    crop_to_size_node,
)

CHANNELS = (1, 3, 4)
GRAY = Color.gray(0.5)

# Where the kept part starts when 11 columns and 6 rows have to go.
STARTS = {
    Anchor.TOP_LEFT: (0, 0),
    Anchor.TOP: (5, 0),
    Anchor.TOP_RIGHT: (11, 0),
    Anchor.LEFT: (0, 3),
    Anchor.CENTER: (5, 3),
    Anchor.RIGHT: (11, 3),
    Anchor.BOTTOM_LEFT: (0, 6),
    Anchor.BOTTOM: (5, 6),
    Anchor.BOTTOM_RIGHT: (11, 6),
}


def coordinates(height: int, width: int, channels: int) -> np.ndarray:
    """Each sample's value names its row, column and channel."""
    y, x = np.mgrid[:height, :width]
    img = np.dstack([(y * width + x + 1) * (c + 1) for c in range(channels)])
    img = (img / (height * width * channels + 1)).astype(np.float32)
    return img[:, :, 0] if channels == 1 else img


def crop(
    img: np.ndarray,
    width: int,
    height: int,
    anchor: Anchor = Anchor.CENTER,
    border: BorderType = BorderType.BLACK,
    color: Color = GRAY,
) -> np.ndarray:
    return crop_to_size_node(
        img, width, height, anchor, CropIfSmaller.KEEP, border, color
    )


def pad(
    img: np.ndarray,
    width: int,
    height: int,
    anchor: Anchor = Anchor.CENTER,
    border: BorderType = BorderType.BLACK,
    color: Color = GRAY,
) -> np.ndarray:
    return crop_to_size_node(
        img, width, height, anchor, CropIfSmaller.PAD, border, color
    )


@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("anchor", Anchor)
def test_each_anchor_keeps_its_part(anchor: Anchor, channels: int):
    img = coordinates(26, 31, channels)
    left, top = STARTS[anchor]
    expected = img[top : top + 20, left : left + 20]
    assert np.array_equal(crop(img, 20, 20, anchor), expected)
    assert np.array_equal(pad(img, 20, 20, anchor), expected)


def test_an_odd_excess_gives_the_requested_size():
    # 831 to 800 by hand-computed edges gives 799 or 801.
    img = coordinates(4, 831, 3)
    result = crop(img, 800, 4)
    assert result.shape == (4, 800, 3)
    assert np.array_equal(result, img[:, 15:815])


@pytest.mark.parametrize("channels", CHANNELS)
def test_keep_leaves_a_short_side(channels: int):
    img = coordinates(8, 5, channels)
    assert np.array_equal(crop(img, 100, 4), img[2:6])
    assert np.array_equal(crop(img, 3, 100), img[:, 1:4])
    assert np.array_equal(crop(img, 100, 100), img)


def test_pad_places_a_short_image_by_the_anchor():
    img = np.array([[0.25, 0.5, 0.75], [0.125, 0.375, 0.625]], np.float32)
    expected = np.zeros((4, 6), np.float32)
    expected[2:, 3:] = img
    assert np.array_equal(pad(img, 6, 4, Anchor.BOTTOM_RIGHT), expected)
    expected = np.zeros((4, 6), np.float32)
    expected[:2, :3] = img
    assert np.array_equal(pad(img, 6, 4, Anchor.TOP_LEFT), expected)
    # Centered, an odd amount gives its extra row and column to the bottom and right.
    expected = np.zeros((5, 6), np.float32)
    expected[1:3, 1:4] = img
    assert np.array_equal(pad(img, 6, 5), expected)


@pytest.mark.parametrize("channels", CHANNELS)
def test_one_side_crops_while_the_other_pads(channels: int):
    img = coordinates(6, 3, channels)
    result = pad(img, 5, 2, Anchor.TOP, BorderType.WHITE)
    assert result.shape[:2] == (2, 5)
    assert np.array_equal(result[:, 1:4], img[:2])
    assert np.all(result[:, 0] == 1) and np.all(result[:, 4] == 1)


@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("border", BorderType)
def test_a_one_pixel_image_pads_with_every_border(border: BorderType, channels: int):
    img = coordinates(1, 1, channels)
    color = Color.bgra((0.1, 0.2, 0.3, 0.4))
    result = pad(img, 3, 3, Anchor.CENTER, border, color)
    output_channels = (
        4 if border in (BorderType.TRANSPARENT, BorderType.CUSTOM_COLOR) else channels
    )
    assert result.shape == ((3, 3) if output_channels == 1 else (3, 3, output_channels))
    if output_channels == channels:
        assert np.array_equal(result[1, 1], img[0, 0])
    if border is BorderType.CUSTOM_COLOR:
        assert np.array_equal(result[0, 0], np.float32(color.value))
    elif border is BorderType.TRANSPARENT:
        assert np.array_equal(result[0, 0], np.zeros(4, np.float32))
    elif border in (BorderType.REFLECT_MIRROR, BorderType.WRAP, BorderType.REPLICATE):
        assert np.array_equal(result[0, 0], img[0, 0])


@pytest.mark.parametrize("channels", CHANNELS)
def test_a_one_pixel_image_is_kept(channels: int):
    img = coordinates(1, 1, channels)
    assert np.array_equal(crop(img, 1, 1), img)
    assert np.array_equal(crop(img, 7, 9), img)
    assert np.array_equal(pad(img, 1, 1), img)


def test_a_full_size_pad_adds_no_channel():
    # No border is made, so Transparent cannot widen the image.
    img = coordinates(4, 5, 3)
    assert np.array_equal(pad(img, 5, 4, Anchor.CENTER, BorderType.TRANSPARENT), img)
    assert np.array_equal(
        pad(img, 2, 2, Anchor.CENTER, BorderType.TRANSPARENT), img[1:3, 1:3]
    )


@pytest.mark.parametrize("anchor", (Anchor.CENTER, Anchor.BOTTOM_RIGHT))
def test_a_permuted_image_gives_the_same_pixels(anchor: Anchor):
    # Lens hands over a planar array's transpose: its result is not C-contiguous.
    img = coordinates(9, 7, 4)
    permuted = np.moveaxis(img, 2, 0).copy().transpose(1, 2, 0)
    assert not permuted.flags.c_contiguous
    for width, height in ((3, 4), (10, 4), (3, 12), (10, 12)):
        assert np.array_equal(
            pad(permuted, width, height, anchor), pad(img, width, height, anchor)
        )
        assert np.array_equal(
            crop(permuted, width, height, anchor), crop(img, width, height, anchor)
        )
