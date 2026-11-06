"""Resize's Hamming and Hann filters end at their support: reductions keep flat areas flat and match a float64 reference
and Pillow. Their formulas go on past 1, and the sum that normalizes the weights reads taps out to the radius, which
the frozen reference (chainner_ext 0.3.10) counts: its reductions turn 0.5 gray into up to 1.0. Enlargements are that
reference's bits (test_resample_complete)."""

import math

import numpy as np
import pytest
from native_differential import assert_native_matches
from PIL import Image

from nodes.impl.resize import ResizeFilter, resize

WINDOWS = {
    ResizeFilter.HAMMING: (0.54, 0.46),
    ResizeFilter.HANN: (0.5, 0.5),
}
SIZES = [
    ((64, 64), (63, 63)),
    ((64, 64), (48, 48)),
    ((64, 64), (43, 21)),
    ((31, 17), (10, 7)),
    ((9, 16), (3, 1)),
    ((5, 3), (2, 2)),
    ((256, 100), (100, 37)),
]


def kernel(filter_, x: np.ndarray) -> np.ndarray:
    a, b = WINDOWS[filter_]
    return np.where(np.abs(x) < 1, np.sinc(x) * (a + b * np.cos(np.pi * x)), 0.0)


def weights(source: int, target: int, filter_) -> np.ndarray:
    ratio = source / target
    scale = max(ratio, 1.0)
    out = np.zeros((target, source))
    positions = np.arange(source)
    for x in range(target):
        center = (x + 0.5) * ratio - 0.5
        row = kernel(filter_, (positions - center) / scale)
        out[x] = row / row.sum()
    return out


def reference(img: np.ndarray, width: int, height: int, filter_) -> np.ndarray:
    across = weights(img.shape[1], width, filter_)
    down = weights(img.shape[0], height, filter_)
    wide = np.einsum("xw,hw...->hx...", across, img.astype(np.float64))
    return np.clip(np.einsum("yh,hx...->yx...", down, wide), 0, 1)


def noise(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


@pytest.mark.parametrize("filter_", WINDOWS)
@pytest.mark.parametrize(("source", "target"), SIZES)
@pytest.mark.parametrize("value", [0.5, 0.1, 1.0])
def test_flat_image_stays_flat(filter_, source, target, value):
    img = np.full((*source, 3), value, np.float32)
    out = resize(img, target[::-1], filter_)
    assert np.abs(out - value).max() < 2e-6


@pytest.mark.parametrize("filter_", WINDOWS)
@pytest.mark.parametrize(("source", "target"), SIZES)
@pytest.mark.parametrize("channels", [1, 3])
def test_reductions_match_a_float64_reference(filter_, source, target, channels):
    shape = source if channels == 1 else (*source, channels)
    img = noise(shape, math.prod(source) + channels)
    out = resize(img, target[::-1], filter_)
    out = out[:, :, 0] if channels == 1 else out
    expected = reference(img, target[1], target[0], filter_)
    assert np.abs(out - expected).max() < 2e-5


@pytest.mark.parametrize(("source", "target"), SIZES)
def test_hamming_reductions_match_pillow(source, target):
    img = noise(source, math.prod(target))
    out = resize(img, target[::-1], ResizeFilter.HAMMING)[:, :, 0]
    pil = Image.fromarray(img, mode="F").resize(target[::-1], Image.Resampling.HAMMING)
    assert np.abs(out - np.clip(np.asarray(pil), 0, 1)).max() < 2e-5


@pytest.mark.parametrize("filter_", WINDOWS)
@pytest.mark.parametrize(("source", "target"), SIZES)
def test_both_executors_agree(filter_, source, target):
    img = noise((*source, 3), 7)
    inputs = [img, 1, 100.0, target[1], target[0], filter_.value, False, 1.0, False, 1]
    assert assert_native_matches("chainner:image:resize", inputs)[0] == "outputs"
