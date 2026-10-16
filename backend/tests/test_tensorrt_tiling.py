from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from nodes.impl.tensorrt.tiling import (
    ShapeBounds,
    axis_weights,
    tile_starts,
    tiled_upscale,
)

SCALE = 4
STATIC_512 = ShapeBounds(512, 512, 512, 512)
DYNAMIC_64_512 = ShapeBounds(64, 64, 512, 512)


def fake_model(img: np.ndarray) -> np.ndarray:
    """A context-dependent 'model': 5x5 box blur, then nearest-neighbour 4x."""
    padded = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode="reflect")
    h, w = img.shape[:2]
    blurred = np.zeros_like(img)
    for dy in range(5):
        for dx in range(5):
            blurred += padded[dy : dy + h, dx : dx + w]
    blurred /= 25
    return blurred.repeat(SCALE, axis=0).repeat(SCALE, axis=1)


class FakeEngine:
    """Runs `fake_model` and enforces the engine's optimization profile."""

    def __init__(self, bounds: ShapeBounds) -> None:
        self.bounds = bounds
        self.queue: deque[np.ndarray] = deque()
        self.shapes: list[tuple[int, int]] = []

    def submit(self, tile: np.ndarray) -> None:
        h, w = tile.shape[:2]
        b = self.bounds
        assert b.min_h <= h <= b.max_h and b.min_w <= w <= b.max_w, (h, w)
        assert len(self.queue) < 2
        self.shapes.append((h, w))
        self.queue.append(fake_model(tile))

    def collect(self, height: int, width: int) -> np.ndarray:
        return self.queue.popleft()[:height, :width]


def _image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0.1, 1.0, (1024, 1024, 3)).astype(np.float32)


@pytest.mark.parametrize(
    "bounds", [STATIC_512, DYNAMIC_64_512], ids=["static512", "dynamic64-512"]
)
def test_tiled_1024_on_512_engine_equals_single_pass(bounds: ShapeBounds):
    img = _image()
    engine = FakeEngine(bounds)

    out = tiled_upscale(img, SCALE, bounds, engine)

    assert out.shape == (4096, 4096, 3)
    assert len(engine.shapes) > 1
    assert all(shape == (512, 512) for shape in engine.shapes)
    np.testing.assert_allclose(out, fake_model(img), rtol=0, atol=1e-5)
    for border in (out[0], out[-1], out[:, 0], out[:, -1]):
        assert border.min() >= 0.1 - 1e-5


def test_fixed_engine_of_unaligned_size_gets_only_its_own_shape():
    # A fixed engine accepts one shape (upstream #3347), here not a multiple of 64.
    bounds = ShapeBounds(250, 200, 250, 200)
    for size in ((600, 500), (100, 90)):
        img = _image()[: size[0], : size[1]]
        engine = FakeEngine(bounds)

        out = tiled_upscale(img, SCALE, bounds, engine)

        assert out.shape == (size[0] * SCALE, size[1] * SCALE, 3)
        assert all(shape == (250, 200) for shape in engine.shapes)
        if size == (600, 500):
            assert len(engine.shapes) > 1
            np.testing.assert_allclose(out, fake_model(img), rtol=0, atol=1e-5)


def test_without_overlap_the_seam_test_fails():
    img = _image()
    out = tiled_upscale(img, SCALE, STATIC_512, FakeEngine(STATIC_512), overlap=0)

    assert out.shape == (4096, 4096, 3)
    assert np.abs(out - fake_model(img)).max() > 1e-2


def test_small_image_is_padded_to_profile_minimum_and_cropped():
    img = np.full((10, 7, 3), 0.5, dtype=np.float32)
    engine = FakeEngine(DYNAMIC_64_512)

    out = tiled_upscale(img, SCALE, DYNAMIC_64_512, engine)

    assert out.shape == (40, 28, 3)
    assert engine.shapes == [(64, 64)]
    np.testing.assert_allclose(out, 0.5, atol=1e-6)


def test_single_tile_is_the_cropped_model_output():
    img = _image()[:300, :200]
    engine = FakeEngine(DYNAMIC_64_512)

    out = tiled_upscale(img, SCALE, DYNAMIC_64_512, engine)

    assert engine.shapes == [(320, 256)]
    padded = np.pad(img, ((0, 20), (0, 56), (0, 0)), mode="reflect")
    np.testing.assert_array_equal(out, fake_model(padded)[:1200, :800])


def test_an_engines_alignment_sets_the_padding_and_tile_size():
    # A dynamic DUAL engine takes multiples of 4: a 300x202 image runs unpadded but for
    # 2 columns, and a 4-aligned maximum is the tile size.
    bounds = ShapeBounds(64, 64, 1084, 1916, alignment=4)
    engine = FakeEngine(bounds)
    img = _image()[:300, :202]

    out = tiled_upscale(img, SCALE, bounds, engine)

    assert engine.shapes == [(300, 204)]
    padded = np.pad(img, ((0, 0), (0, 2), (0, 0)), mode="reflect")
    np.testing.assert_array_equal(out, fake_model(padded)[:1200, :808])
    engine = FakeEngine(bounds)
    tiled_upscale(np.zeros((2000, 3000, 3), np.float32), SCALE, bounds, engine)
    assert set(engine.shapes) == {(1084, 1916)}


def test_an_engine_too_small_to_tile_still_serves_images_up_to_its_size():
    # Build Engine allows a 32 px fixed engine; tiles need over 3 * 16 px.
    bounds = ShapeBounds(32, 32, 32, 32)
    engine = FakeEngine(bounds)

    out = tiled_upscale(_image()[:20, :32], SCALE, bounds, engine)

    assert out.shape == (80, 128, 3)
    with pytest.raises(ValueError) as error:
        tiled_upscale(_image()[:20, :40], SCALE, bounds, FakeEngine(bounds))
    assert str(error.value) == (
        "This image (40 px) is larger than the TensorRT engine's input (32 px), and"
        " tiling it needs an engine input over 48 px. The engine still upscales"
        " images up to 32 px; for larger ones, build it with a size over 48 px."
    )


def test_axis_weights_partition_of_unity():
    for length in (513, 1000, 1024, 1500, 2049):
        starts = tile_starts(length, 512)
        assert starts[-1] + 512 == length
        total = np.zeros(length * SCALE, dtype=np.float64)
        for s, w in zip(starts, axis_weights(starts, 512, length, SCALE)):
            total[s * SCALE : s * SCALE + w.size] += w
        np.testing.assert_allclose(total, 1.0, atol=1e-6)
