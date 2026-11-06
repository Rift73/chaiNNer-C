"""Pad to Square: the square's size, where each anchor puts the image, an odd amount of
padding and each border, on planted images in gray, RGB and RGBA, down to 1x1."""

from __future__ import annotations

import numpy as np
import pytest

from nodes.impl.color.color import Color
from nodes.impl.image_utils import BorderType
from nodes.properties.inputs import Anchor
from packages.chaiNNer_standard.image_dimension.border.pad_to_square import (
    pad_to_square_node,
)

CHANNELS = (1, 3, 4)
GRAY = Color.gray(0.5)

# The row a 4x7 image starts at in its 7x7 square, and the column a 7x4 image starts at.
WIDE_ROWS = {"top": 0, "centered": 1, "bottom": 3}
TALL_COLUMNS = {"left": 0, "centered": 1, "right": 3}


def planted(height: int, width: int, channels: int) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    img = np.dstack([(y * width + x + 1) * (c + 1) for c in range(channels)])
    img = (img / (height * width * channels + 1)).astype(np.float32)
    return img[:, :, 0] if channels == 1 else img


def square(
    img: np.ndarray,
    anchor: Anchor = Anchor.CENTER,
    border: BorderType = BorderType.BLACK,
    color: Color = GRAY,
) -> np.ndarray:
    return pad_to_square_node(img, anchor, border, color)


def rows_for(anchor: Anchor) -> int:
    return WIDE_ROWS[
        next(
            part
            for part in ("top", "bottom", "centered")
            if anchor.value.startswith(part)
        )
    ]


def columns_for(anchor: Anchor) -> int:
    return TALL_COLUMNS[
        next(
            part
            for part in ("left", "right", "centered")
            if anchor.value.endswith(part)
        )
    ]


@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("anchor", Anchor)
def test_a_wide_image_gets_rows(anchor: Anchor, channels: int):
    img = planted(4, 7, channels)
    result = square(img, anchor)
    assert result.shape[:2] == (7, 7)
    top = rows_for(anchor)
    assert np.array_equal(result[top : top + 4], img)
    # Only rows were added: the rest is the border.
    first = np.atleast_3d(result)[:, :, 0]
    assert np.count_nonzero(np.delete(first, range(top, top + 4), axis=0)) == 0


@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("anchor", Anchor)
def test_a_tall_image_gets_columns(anchor: Anchor, channels: int):
    img = planted(7, 4, channels)
    result = square(img, anchor)
    assert result.shape[:2] == (7, 7)
    left = columns_for(anchor)
    assert np.array_equal(result[:, left : left + 4], img)
    first = np.atleast_3d(result)[:, :, 0]
    assert np.count_nonzero(np.delete(first, range(left, left + 4), axis=1)) == 0


@pytest.mark.parametrize("channels", CHANNELS)
def test_a_square_stays_as_it_is(channels: int):
    for side in (1, 5):
        img = planted(side, side, channels)
        for border in BorderType:
            color = Color.bgra((0.1, 0.2, 0.3, 0.4))
            assert np.array_equal(square(img, Anchor.TOP_LEFT, border, color), img)


@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("border", BorderType)
def test_a_one_row_image_pads_with_every_border(border: BorderType, channels: int):
    img = planted(1, 3, channels)
    color = Color.bgra((0.1, 0.2, 0.3, 0.4))
    result = square(img, Anchor.CENTER, border, color)
    output_channels = (
        4 if border in (BorderType.TRANSPARENT, BorderType.CUSTOM_COLOR) else channels
    )
    assert result.shape == ((3, 3) if output_channels == 1 else (3, 3, output_channels))
    if output_channels == channels:
        assert np.array_equal(result[1], img[0])
    if border is BorderType.REPLICATE:
        assert np.array_equal(result[0], img[0])
        assert np.array_equal(result[2], img[0])
    elif border is BorderType.WHITE:
        assert np.all(result[0] == 1)
    elif border is BorderType.CUSTOM_COLOR:
        assert np.array_equal(result[0, 0], np.float32(color.value))
    elif border is BorderType.TRANSPARENT:
        assert np.all(result[0] == 0)
        # The image's own samples stay opaque when it widens to RGBA.
        if channels != 4:
            assert np.all(result[1, :, 3] == 1)


def test_reflect_and_wrap_give_their_rows():
    # Rows a, b, c of a 3x5 image in a 5x5 square: one row above and one below. Reflect
    # mirrors around the edge row without repeating it.
    img = planted(3, 5, 1)
    a, b, c = img
    assert np.array_equal(
        square(img, Anchor.CENTER, BorderType.REFLECT_MIRROR),
        np.stack((b, a, b, c, b)),
    )
    assert np.array_equal(
        square(img, Anchor.CENTER, BorderType.WRAP), np.stack((c, a, b, c, a))
    )


def test_a_permuted_image_gives_the_same_pixels():
    img = planted(5, 8, 4)
    permuted = np.moveaxis(img, 2, 0).copy().transpose(1, 2, 0)
    assert not permuted.flags.c_contiguous
    assert np.array_equal(square(permuted), square(img))
