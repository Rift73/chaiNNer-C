"""convenient_upscale's transparency path runs the model over the image on a black and on
a white background. Over opaque pixels the two differ at most in float32's lowest bits,
which an FP16 or BF16 input does not hold: on a fully opaque tile such a model gets the
same input bits in both passes, so its white pass output is its black pass output. The
black pass keeps those outputs for the white pass. A backend whose tile loop takes part
asks for `current_white_pass`; the others run both passes in full, with the same result.

An FP16 model that takes FP32 up to its first convolution (pytorch/input_export.py) does get
those lowest bits on a dark opaque tile, but they are the white background's rounding and
no difference in the image. For it a fully opaque tile counts as the same input, and a tile
with any other alpha has to match bit for bit."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import psutil

_current = threading.local()

# Asking how much memory is free takes 2 to 6 ms on Windows, as long as a small model takes
# for a tile, so it is asked at most once in this many seconds and once per image. Each tile
# kept since counts against the last answer, so a burst keeps no more than that answer has
# room for; only what other programs take in between goes unseen for that long.
MEMORY_QUERY_SECONDS = 1.0


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    # Not their values: -0 equals 0 and NaN does not equal itself.
    unsigned = np.dtype(f"u{a.dtype.itemsize}")
    return np.array_equal(a.view(unsigned), b.view(unsigned))


class WhitePass:
    def __init__(self, white: np.ndarray, opaque: np.ndarray) -> None:
        self._white = white
        self._opaque = opaque
        self._kept: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._started = False
        self._queried_at: float | None = None
        self._available = 0

    def start(self) -> None:
        self._started = True

    def keep(
        self,
        x: int,
        y: int,
        tile: np.ndarray,
        output: np.ndarray,
        model_input: Callable[[np.ndarray], np.ndarray],
        opaque_is_same: bool = False,
    ) -> None:
        """In the black pass: keeps the model's `output` for the `tile` at (x, y) when
        `model_input` (the tile as the model gets it, its order of channels aside) makes
        the white image's tile there the same bits. An image that is one tile is not
        opaque throughout (convenient_upscale's constant alpha), so it is not compared.
        `opaque_is_same`: the model takes FP32 and rounds after its first convolution."""
        if self._started or tile.shape == self._white.shape:
            return
        h, w = tile.shape[:2]
        twin = self._white[y : y + h, x : x + w]
        if twin.shape != tile.shape:
            return
        same = opaque_is_same and bool(self._opaque[y : y + h, x : x + w].all())
        if same or _same_bits(model_input(tile), model_input(twin)):
            # Only with memory to spare for what is still to come: the white pass's
            # result (three floats a pixel) and the RGBA result (four).
            scale = output.shape[0] // h
            pixels = self._white.shape[0] * self._white.shape[1] * scale * scale
            if self._has_room(output.nbytes, pixels * 7 * 4):
                self._kept[x, y, h, w] = (twin, output)

    def _has_room(self, kept: int, still_to_come: int) -> bool:
        now = time.monotonic()
        if self._queried_at is None or now - self._queried_at >= MEMORY_QUERY_SECONDS:
            self._queried_at = now
            self._available = psutil.virtual_memory().available
        if self._available < kept + still_to_come:
            return False
        self._available -= kept
        return True

    def reuse(self, x: int, y: int, tile: np.ndarray) -> np.ndarray | None:
        """In the white pass: the black pass's output for the `tile` at (x, y), if it kept
        one for these bits. A tile layout that changed (out of memory) finds none: a tile
        of another size where one was kept shows the change, and every kept tile goes."""
        if not self._started:
            return None
        size = tile.shape[:2]
        kept = self._kept.pop((x, y, *size), None)
        if kept is None:
            if any(key[:2] == (x, y) for key in self._kept):
                self._kept.clear()
            return None
        if not _same_bits(tile, kept[0]):
            return None
        return kept[1]


@contextmanager
def white_pass(white: np.ndarray, opaque: np.ndarray) -> Iterator[WhitePass]:
    """`white`: the white pass's image, as the upscale will get it. `opaque`: where the
    image's alpha is exactly 1."""
    current = WhitePass(white, opaque)
    _current.white_pass = current
    try:
        yield current
    finally:
        _current.white_pass = None


def current_white_pass() -> WhitePass | None:
    """The transparency path this thread is in, if any."""
    return getattr(_current, "white_pass", None)
