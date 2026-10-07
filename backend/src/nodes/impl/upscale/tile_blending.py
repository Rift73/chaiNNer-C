from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from nodes.utils.utils import get_h_w_c

from ..native_buffers import tile_mix
from ..native_framework_shared import blend_curve, blend_weights, copy_into


def sin_blend_fn(x: np.ndarray) -> np.ndarray:
    return blend_curve(x)


def half_sin_blend_fn(i: np.ndarray) -> np.ndarray:
    # only use half the overlap
    return blend_curve(i, half=True)


class BlendDirection(Enum):
    X = 0
    Y = 1


@dataclass(frozen=True)
class TileOverlap:
    start: int
    end: int

    @property
    def total(self) -> int:
        return self.start + self.end


def _fast_mix(a: np.ndarray, b: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """
    Returns `a * (1 - blend) + b * blend`
    """
    if (
        a.dtype == b.dtype == blend.dtype == np.float32
        and a.ndim == 3
        and a.shape == b.shape == blend.shape
        and 0 not in a.shape
    ):
        return tile_mix(a, b, blend)
    # a * (1 - blend) + b * blend
    # a - a * blend + b * blend
    r = b * blend
    r += a
    r -= a * blend  # type: ignore
    return r


class TileBlender:
    def __init__(
        self,
        width: int,
        height: int,
        channels: int,
        direction: BlendDirection,
        blend_fn: Callable[[np.ndarray], np.ndarray] = sin_blend_fn,
        _prev: TileBlender | None = None,
    ) -> None:
        self.direction: BlendDirection = direction
        self.blend_fn: Callable[[np.ndarray], np.ndarray] = blend_fn
        self.offset: int = 0
        self.last_end_overlap: int = 0
        self._last_blend: np.ndarray | None = None

        if (
            _prev is not None
            and _prev.direction == direction
            and _prev.width == width
            and _prev.height == height
            and _prev.channels == channels
        ):
            if _prev.blend_fn == blend_fn:
                # reuse blend
                self._last_blend = _prev._last_blend  # noqa: SLF001
            result = _prev.result
        else:
            result = np.zeros((height, width, channels), dtype=np.float32)
        self.result: np.ndarray = result

    @property
    def width(self) -> int:
        return self.result.shape[1]

    @property
    def height(self) -> int:
        return self.result.shape[0]

    @property
    def channels(self) -> int:
        return self.result.shape[2]

    def _get_blend(self, blend_size: int) -> np.ndarray:
        if self.direction == BlendDirection.X:
            if self._last_blend is not None and self._last_blend.shape[1] == blend_size:
                return self._last_blend

            blend = self._weights(blend_size)
            blend = blend.reshape((1, blend_size, 1))
            blend = np.broadcast_to(blend, (self.height, blend_size, self.channels))
        else:
            if self._last_blend is not None and self._last_blend.shape[0] == blend_size:
                return self._last_blend

            blend = self._weights(blend_size)
            blend = blend.reshape((blend_size, 1, 1))
            blend = np.broadcast_to(blend, (blend_size, self.width, self.channels))

        self._last_blend = blend
        return blend

    def _weights(self, size: int) -> np.ndarray:
        if self.blend_fn is sin_blend_fn or self.blend_fn is half_sin_blend_fn:
            return blend_weights(size, half=self.blend_fn is half_sin_blend_fn)
        return self.blend_fn(np.arange(size, dtype=np.float32) / (size - 1))

    def add_tile(self, tile: np.ndarray, overlap: TileOverlap) -> None:
        h, w, c = get_h_w_c(tile)
        assert c == self.channels
        o = overlap

        if self.direction == BlendDirection.X:
            assert h == self.height
            assert w > o.total

            if self.offset == 0:
                # the first tile is copied in as is
                copy_into(self.result[:, :w, ...], tile)

                assert o.start == 0
                self.offset += w - o.end
                self.last_end_overlap = o.end

            else:
                assert self.offset < self.width, "All tiles were filled in already"

                if self.last_end_overlap < o.start:
                    # we can't use all the overlap of the current tile, so we have to cut it off
                    diff = o.start - self.last_end_overlap
                    tile = tile[:, diff:, ...]
                    h, w, c = get_h_w_c(tile)
                    o = TileOverlap(self.last_end_overlap, o.end)

                # copy over the part that doesn't need blending (yet)
                copy_into(
                    self.result[
                        :, self.offset + o.start : self.offset + w - o.start, ...
                    ],
                    tile[:, o.start * 2 :, ...],
                )

                # blend the overlapping part
                blend_size = o.start * 2
                blend = self._get_blend(blend_size)

                left = self.result[
                    :, self.offset - o.start : self.offset + o.start, ...
                ]
                right = tile[:, :blend_size, ...]

                copy_into(
                    self.result[:, self.offset - o.start : self.offset + o.start, ...],
                    _fast_mix(left, right, blend),
                )

                self.offset += w - o.total
                self.last_end_overlap = o.end
        else:
            assert w == self.width
            assert h > o.total

            if self.offset == 0:
                # the first tile is copied in as is
                copy_into(self.result[:h, :, ...], tile)

                assert o.start == 0
                self.offset += h - o.end
                self.last_end_overlap = o.end

            else:
                assert self.offset < self.height, "All tiles were filled in already"

                if self.last_end_overlap < o.start:
                    # we can't use all the overlap of the current tile, so we have to cut it off
                    diff = o.start - self.last_end_overlap
                    tile = tile[diff:, :, ...]
                    h, w, c = get_h_w_c(tile)
                    o = TileOverlap(self.last_end_overlap, o.end)

                # copy over the part that doesn't need blending
                copy_into(
                    self.result[
                        self.offset + o.start : self.offset + h - o.start, :, ...
                    ],
                    tile[o.start * 2 :, :, ...],
                )

                # blend the overlapping part
                blend_size = o.start * 2
                blend = self._get_blend(blend_size)

                left = self.result[
                    self.offset - o.start : self.offset + o.start, :, ...
                ]
                right = tile[: o.start * 2, :, ...]

                copy_into(
                    self.result[self.offset - o.start : self.offset + o.start, :, ...],
                    _fast_mix(left, right, blend),
                )

                self.offset += h - o.total
                self.last_end_overlap = o.end

    def get_result(self) -> np.ndarray:
        if self.direction == BlendDirection.X:
            assert self.offset == self.width
        else:
            assert self.offset == self.height

        return self.result
