from __future__ import annotations

import contextlib
import ctypes
import warnings
from collections import deque
from collections.abc import Callable, Iterator
from typing import Literal

import numpy as np
import pytest

# The blend runs in the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from nodes.impl.native_framework_shared import accumulate_tile
from nodes.impl.tensorrt import tiling
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


def numpy_blend(
    result: np.ndarray | None,
    tile_out: np.ndarray,
    y0: int,
    x0: int,
    wy: np.ndarray,
    wx: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """The blend as NumPy computed it before the native pass."""
    if result is None:
        result = np.zeros((out_h, out_w, tile_out.shape[2]), dtype=np.float32)
    weight = wy[:, None, None] * wx[None, :, None]
    result[y0 : y0 + wy.size, x0 : x0 + wx.size] += tile_out * weight
    return result


class ExceptionalEngine(FakeEngine):
    """FakeEngine whose tiles also hold a NaN with the tile's own payload at every
    pixel (so two NaNs meet in each overlap), an infinity, -0 and the largest
    float32, at seeded positions."""

    def __init__(self, bounds: ShapeBounds) -> None:
        super().__init__(bounds)
        self.tiles = 0

    def submit(self, tile: np.ndarray) -> None:
        super().submit(tile)
        out = self.queue[-1]
        self.tiles += 1
        rng = np.random.default_rng(self.tiles)
        bits = out.view(np.uint32).reshape(-1)
        bits[rng.choice(bits.size, 4000, replace=False)] = 0x7FC00000 + self.tiles
        bits[rng.choice(bits.size, 50, replace=False)] = 0x7F800000
        bits[rng.choice(bits.size, 50, replace=False)] = 0x80000000
        bits[rng.choice(bits.size, 50, replace=False)] = 0x7F7FFFFF
        bits[:: bits.size // 7] = 0x7FC0FFFF  # the same pixels in every tile


@pytest.mark.parametrize(
    "bounds", [STATIC_512, DYNAMIC_64_512], ids=["static512", "dynamic64-512"]
)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_native_blend_is_numpys_bit_for_bit(
    monkeypatch: pytest.MonkeyPatch, bounds: ShapeBounds, channels: int
):
    img = np.ascontiguousarray(np.resize(_image()[:700, :1000], (700, 1000, channels)))

    def run() -> tuple[np.ndarray, list[str]]:
        with warnings.catch_warnings(record=True) as caught, np.errstate(all="warn"):
            warnings.simplefilter("always")
            out = tiled_upscale(img, SCALE, bounds, ExceptionalEngine(bounds))
        return out, [str(w.message) for w in caught]

    actual, actual_warnings = run()
    monkeypatch.setattr(tiling, "_blend", numpy_blend)
    expected, expected_warnings = run()
    assert actual.tobytes() == expected.tobytes()
    assert actual_warnings == expected_warnings
    assert "invalid value encountered in multiply" in actual_warnings


@contextlib.contextmanager
def flushed(flush: bool) -> Iterator[None]:
    """DAZ and FTZ on the calling thread (the CRT's _DN_FLUSH) when flush, which the
    native pool's helpers take; the default afterwards."""
    control = ctypes.CDLL("ucrtbase.dll")["_controlfp_s"]
    control.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
    word = ctypes.c_uint()
    if flush:
        assert control(ctypes.byref(word), 0x01000000, 0x03000000) == 0
    try:
        yield
    finally:
        assert control(ctypes.byref(word), 0, 0x03000000) == 0


def random_floats(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Every float32 bit pattern: NaN payloads, infinities, denormals, -0."""
    words = np.random.default_rng(seed).integers(0, 2**32, shape, dtype=np.uint64)
    return words.astype(np.uint32).view(np.float32)


@pytest.mark.parametrize("channels", [1, 3, 4, 7])
@pytest.mark.parametrize("mode", ["warn", "raise", "ignore"])
@pytest.mark.parametrize("flush", [False, True], ids=["default", "daz_ftz"])
def test_accumulate_tile_is_the_numpy_expression(
    channels: int, mode: Literal["warn", "raise", "ignore"], flush: bool
):
    # Any bits in the window, tile and weights; a strided tile, a window inside a
    # wider image; denormals in both MXCSR states.
    h, w = 37, 1531  # several pool chunks; rows of several 512-element blocks
    image = random_floats((h + 3, w + 5, channels), 1)
    tile = random_floats((h + 2, w + 9, channels), 2)[1:-1, 4:-5]
    wy, wx = random_floats((h,), 3), random_floats((w,), 4)
    wy[::3] = np.random.default_rng(5).random(wy[::3].shape, np.float32)
    wx[::2] = 0
    expected = image.copy()

    def outcome(call: Callable[[], object]) -> tuple[str | None, list[str]]:
        with warnings.catch_warnings(record=True) as caught, np.errstate(all=mode):
            warnings.simplefilter("always")
            try:
                call()
                failure = None
            except FloatingPointError as error:
                failure = str(error)
        return failure, [str(item.message) for item in caught]

    window = (slice(2, 2 + h), slice(3, 3 + w))
    with flushed(flush):
        actual_outcome = outcome(lambda: accumulate_tile(image[window], tile, wy, wx))
        expected_outcome = outcome(
            lambda: expected[window].__iadd__(
                tile * (wy[:, None, None] * wx[None, :, None])
            )
        )
    assert actual_outcome == expected_outcome
    if mode != "raise":
        assert image.tobytes() == expected.tobytes()
