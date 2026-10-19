"""Native image_channel entry points (native/src/run/nodes/image_channels.cpp) against their Python bodies."""

import numpy as np
import pytest

from image_node_cases import SIZES, image, seed_of
from native_differential import assert_native_matches
from nodes.impl.color.color import Color

GREY = Color.gray(0.3)
COLORS = [Color.gray(0.25), Color.bgr((0.1, 0.5, 0.9)), Color.bgra((0.2, 0.4, 0.6, 0.8))]


def grey(key: object, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    return image(seed_of(key, size), height, width, 1)


# Each R, G, B, A input: "i" an image, "c" a colour, "-" unconnected (A only).
COMBINE = ["iiii", "iii-", "ciic", "cci-", "icc-", "ccic", "cccc", "ccc-"]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("pattern", COMBINE)
def test_combine_rgba_matches(pattern, size):
    inputs: list[object] = [
        grey((pattern, i), size) if kind == "i" else Color.gray(0.1 * (i + 1)) if kind == "c" else None
        for i, kind in enumerate(pattern)
    ]
    assert_native_matches("chainner:image:combine_rgba", inputs)


def test_combine_rgba_errors():
    a, b = grey("a", (4, 5)), grey("b", (5, 4))
    empty = np.zeros((0, 3), np.float32)
    cases: list[tuple[list[object], tuple[str, str]]] = [
        ([a, b, a, None], ("AssertionError", "All channel images must have the same resolution")),
        ([a, GREY, empty, None], ("ValueError", "Expected a nonempty two- or three-dimensional image")),
        ([empty, b, a, a], ("ValueError", "Expected a nonempty two- or three-dimensional image")),
        ([a, a, a, empty], ("ValueError", "Expected a nonempty two- or three-dimensional image")),
        (
            [GREY, GREY, GREY, None],
            ("ValueError", "At least one channels must be an image, but all given channels are colors."),
        ),
    ]
    for inputs, error in cases:
        assert assert_native_matches("chainner:image:combine_rgba", inputs) == ("error", error)


# Channel counts of the four inputs (0: unconnected).
MERGE = [(1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 1, 1), (3, 0, 0, 0), (3, 1, 0, 0), (1, 3, 0, 0),
         (4, 0, 0, 0), (3, 3, 0, 0), (4, 4, 4, 4), (1, 0, 1, 0), (3, 0, 0, 1), (1, 0, 0, 1), (1, 1, 3, 4)]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("counts", MERGE)
def test_merge_channels_matches(counts, size):
    height, width = size
    inputs: list[object] = [image(seed_of(counts, size, i), height, width, c) if c else None for i, c in enumerate(counts)]
    assert_native_matches("chainner:image:merge_channels", inputs)


def test_merge_channels_errors():
    a, b = image(1, 4, 5, 3), image(2, 5, 4, 1)
    empty = np.zeros((0, 3), np.float32)
    cases: list[list[object]] = [[a, b, None, None], [a, None, b, empty], [a, empty, b, None],
                                 [empty, a, None, None], [a, a, a, empty], [empty, None, None, None]]
    for inputs in cases:
        assert_native_matches("chainner:image:merge_channels", inputs)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("alpha_kind", ["image", "color"])
@pytest.mark.parametrize("rgb_kind", ["1", "3", "4", "grey", "bgr", "bgra"])
def test_merge_transparency_matches(rgb_kind, alpha_kind, size):
    height, width = size
    rgb = (
        image(seed_of(rgb_kind, size), height, width, int(rgb_kind))
        if rgb_kind.isdigit()
        else COLORS[["grey", "bgr", "bgra"].index(rgb_kind)]
    )
    alpha = grey((rgb_kind, "alpha"), size) if alpha_kind == "image" else Color.gray(0.75)
    if isinstance(rgb, Color) and isinstance(alpha, Color):
        alpha = grey((rgb_kind, "alpha"), size)
    assert_native_matches("chainner:image:merge_transparency", [rgb, alpha])


def test_merge_transparency_errors():
    a, b = image(1, 4, 5, 3), grey("b", (5, 4))
    empty3, empty1 = np.zeros((0, 4, 3), np.float32), np.zeros((4, 0), np.float32)
    cases: list[list[object]] = [[a, b], [empty3, b], [a, empty1], [COLORS[1], empty1], [COLORS[1], GREY]]
    for inputs in cases:
        assert_native_matches("chainner:image:merge_transparency", inputs)


def transparent(seed: int, height: int, width: int) -> np.ndarray:
    """RGBA noise whose alpha is zero over two thirds of the image, with edge values elsewhere."""
    img = image(seed, height, width, 4)
    alpha = img[:, :, 3]
    alpha[np.random.default_rng(seed).random((height, width)) < 2 / 3] = 0
    return img


@pytest.mark.parametrize("size", [*SIZES, (40, 23)])
@pytest.mark.parametrize("method", [1, 2, 3])
def test_fill_alpha_matches(method, size):
    height, width = size
    assert_native_matches("chainner:image:fill_alpha", [transparent(seed_of(method, size), height, width), method])


@pytest.mark.parametrize("method", [1, 2, 3])
def test_fill_alpha_edge_cases(method):
    opaque = image(3, 6, 7, 4)
    opaque[:, :, 3] = 1
    clear = np.zeros((6, 7, 4), np.float32)
    for img in (opaque, clear, np.zeros((0, 3, 4), np.float32), np.zeros((3, 0, 4), np.float32)):
        assert_native_matches("chainner:image:fill_alpha", [img, method])


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_split_transparency_matches(channels, size):
    height, width = size
    assert_native_matches("chainner:image:split_transparency", [image(seed_of("split", size, channels), height, width, channels)])


@pytest.mark.parametrize("shape", [(0, 5, 3), (4, 0), (0, 0, 4), (3, 0, 3)])
def test_split_transparency_empty_image(shape):
    assert_native_matches("chainner:image:split_transparency", [np.zeros(shape, np.float32)])


# (method, binary threshold, background threshold, foreground threshold, confusion bg, confusion fg, output trimap)
CHROMA = [(1, 1.0, 2.0, 10.0, 4, 4, False), (1, 30.5, 2.0, 10.0, 4, 4, False), (1, 0.0, 2.0, 10.0, 4, 4, False),
          (2, 1.0, 2.0, 10.0, 4, 4, True), (2, 1.0, 20.0, 35.0, 0, 3, True), (2, 1.0, 50.0, 5.0, 1, 0, True),
          (2, 1.0, 12.5, 60.0, 100, 7, True)]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("key", [Color.bgr((0.0, 1.0, 0.0)), Color.bgr((0.2, 0.5, 0.7))])
@pytest.mark.parametrize("chroma", CHROMA)
def test_chroma_key_matches(chroma, key, size):
    height, width = size
    img = image(seed_of("chroma", size), height, width, 3)
    img[: height // 2, : width // 2] = np.array(key.value, np.float32)
    assert_native_matches("chainner:image:chroma_key", [img, key, *chroma])


def test_chroma_key_edge_cases():
    key = Color.bgr((0.0, 1.0, 0.0))
    assert_native_matches("chainner:image:chroma_key", [image(1, 8, 8, 3), key, 2, 1.0, 2.0, 10.0, 4, 4, False], declines=True)
    for chroma in CHROMA[:4]:
        assert_native_matches("chainner:image:chroma_key", [np.zeros((0, 5, 3), np.float32), key, *chroma])
