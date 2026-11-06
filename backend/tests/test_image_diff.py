"""Image Diff: planted differences in gray, RGB and RGBA, down to 1x1, read back from each
mode's image and from the numbers."""

from __future__ import annotations

import numpy as np
import pytest

from packages.chaiNNer_standard.image_utility import miscellaneous_group
from packages.chaiNNer_standard.image_utility.miscellaneous.image_diff import (
    DiffMode,
    image_diff_node,
)

CHANNELS = (1, 3, 4)
OUTPUT = next(
    node
    for node in miscellaneous_group.nodes
    if node.schema_id == "chainner:image:image_diff"
).outputs[0]


def blank(channels: int, height: int = 4, width: int = 5) -> np.ndarray:
    shape = (height, width) if channels == 1 else (height, width, channels)
    return np.full(shape, 100 / 255, np.float32)


def planted(channels: int) -> tuple[np.ndarray, np.ndarray]:
    """B is A but for two pixels: at (1, 2) the first channel is 10 levels darker, and at
    (3, 0) every channel is 3 levels brighter."""
    a = blank(channels)
    b = a.copy()
    if channels == 1:
        b[1, 2] = 90 / 255
        b[3, 0] = 103 / 255
    else:
        b[1, 2, 0] = 90 / 255
        b[3, 0, :] = 103 / 255
    return a, b


def diff(
    a: np.ndarray,
    b: np.ndarray,
    mode: DiffMode = DiffMode.ABSOLUTE,
    amplify: float = 1.0,
    tolerance: float = 0.0,
) -> tuple[np.ndarray, float, float, int]:
    return image_diff_node(a, b, mode, amplify, tolerance)


@pytest.mark.parametrize("channels", CHANNELS)
def test_absolute_gives_each_channels_difference(channels: int):
    a, b = planted(channels)
    image, *_ = diff(a, b)
    assert image.shape[:2] == a.shape[:2]
    expected = np.zeros(a.shape, np.float32)
    if channels == 1:
        expected[1, 2], expected[3, 0] = 10 / 255, 3 / 255
    else:
        expected[1, 2, 0] = 10 / 255
        expected[3, 0, :] = 3 / 255
    if channels == 4:
        assert image.shape[2] == 3
        expected = expected[:, :, :3]
    assert np.allclose(image, expected, atol=1e-6)


@pytest.mark.parametrize("channels", CHANNELS)
def test_signed_centers_on_middle_gray(channels: int):
    a, b = planted(channels)
    image, *_ = diff(a, b, DiffMode.SIGNED, 2.0)
    expected = np.full(a.shape, 0.5, np.float32)
    if channels == 1:
        expected[1, 2], expected[3, 0] = 0.5 + 10 / 255, 0.5 - 3 / 255
    else:
        expected[1, 2, 0] = 0.5 + 10 / 255
        expected[3, 0, :] = 0.5 - 3 / 255
    if channels == 4:
        expected = expected[:, :, :3]
    assert np.allclose(image, expected, atol=1e-6)


@pytest.mark.parametrize("channels", CHANNELS)
def test_mask_and_count_follow_the_tolerance(channels: int):
    a, b = planted(channels)
    for tolerance, positions in (
        (0, [(1, 2), (3, 0)]),
        (2.5, [(1, 2), (3, 0)]),
        (3, [(1, 2)]),
        (9.5, [(1, 2)]),
        (10, []),
        (255, []),
    ):
        image, _, _, count = diff(a, b, DiffMode.MASK, 1.0, tolerance)
        expected = np.zeros(a.shape[:2], np.float32)
        for position in positions:
            expected[position] = 1
        assert image.dtype == np.float32
        assert np.array_equal(image, expected)
        assert count == len(positions)


@pytest.mark.parametrize("channels", CHANNELS)
def test_the_numbers_are_in_levels(channels: int):
    a, b = planted(channels)
    _, largest, mean, count = diff(a, b)
    assert largest == 10
    assert mean == round((10 + 3 * channels) / (4 * 5 * channels), 4)
    assert count == 2


def test_one_level_of_any_8_bit_value_is_within_a_tolerance_of_one():
    # In float32, some of these differences come out above 1/255.
    a = (np.arange(255, dtype=np.float64) / 255).astype(np.float32)[None]
    b = (np.arange(1, 256, dtype=np.float64) / 255).astype(np.float32)[None]
    assert np.any(np.abs(a - b) > np.float32(1 / 255))
    assert diff(a, b, tolerance=1.0)[3] == 0
    assert diff(a, b, tolerance=0.0)[3] == 255
    assert diff(a, b)[1] == 1


@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("size", ((1, 1), (1, 3), (3, 1), (4, 5)))
def test_equal_images_have_no_difference(channels: int, size: tuple[int, int]):
    a = blank(channels, *size)
    absolute, largest, mean, count = diff(a, a.copy())
    assert np.all(absolute == 0)
    assert (largest, mean, count) == (0, 0, 0)
    assert np.all(diff(a, a.copy(), DiffMode.SIGNED)[0] == 0.5)
    assert np.all(diff(a, a.copy(), DiffMode.MASK)[0] == 0)


@pytest.mark.parametrize("channels", CHANNELS)
def test_a_one_pixel_difference(channels: int):
    a = blank(channels, 1, 1)
    b = np.full(a.shape, 0.75, np.float32)
    image, largest, mean, count = diff(a, b)
    assert largest == mean == round((0.75 - 100 / 255) * 255, 4)
    assert count == 1
    assert image.size == (3 if channels == 4 else channels)


def test_a_difference_of_alpha_alone():
    a = blank(4)
    b = a.copy()
    b[2, 2, 3] = 0
    absolute, largest, _, count = diff(a, b)
    assert count == 1 and largest == 100
    assert np.allclose(absolute[2, 2], [100 / 255] * 3, atol=1e-6)
    absolute[2, 2] = 0
    assert np.all(absolute == 0)
    # Signed shows the colour channels only.
    assert np.all(diff(a, b, DiffMode.SIGNED)[0] == 0.5)
    assert diff(a, b, DiffMode.MASK)[0][2, 2] == 1


def test_amplified_values_reach_the_image_clipped():
    a, b = planted(3)
    image, *_ = diff(a, b, DiffMode.ABSOLUTE, 100.0)
    enforced = OUTPUT.enforce(image)
    assert enforced[1, 2, 0] == 1 and enforced[3, 0, 0] == 1
    signed, *_ = diff(a, b, DiffMode.SIGNED, 100.0)
    enforced = OUTPUT.enforce(signed)
    assert enforced[1, 2, 0] == 1 and enforced[3, 0, 0] == 0


@pytest.mark.parametrize(
    ("a", "b"),
    (
        (np.zeros((4, 5), np.float32), np.zeros((4, 5, 3), np.float32)),
        (np.zeros((4, 5, 3), np.float32), np.zeros((4, 5, 4), np.float32)),
        (np.zeros((4, 5), np.float32), np.zeros((5, 4), np.float32)),
    ),
)
def test_different_shapes_are_refused(a: np.ndarray, b: np.ndarray):
    with pytest.raises(AssertionError, match="same dimensions"):
        diff(a, b)


@pytest.mark.parametrize("mode", DiffMode)
def test_permuted_and_mixed_layouts_give_the_same_result(mode: DiffMode):
    a, b = planted(4)
    expected = diff(a, b, mode, 3.0, 1.0)
    for first, second in (
        (np.moveaxis(a, 2, 0).copy().transpose(1, 2, 0), b),
        (a, np.moveaxis(b, 2, 0).copy().transpose(1, 2, 0)),
        (
            np.moveaxis(a, 2, 0).copy().transpose(1, 2, 0),
            np.moveaxis(b, 2, 0).copy().transpose(1, 2, 0),
        ),
    ):
        result = diff(first, second, mode, 3.0, 1.0)
        assert np.array_equal(result[0], expected[0])
        assert result[1:] == expected[1:]
