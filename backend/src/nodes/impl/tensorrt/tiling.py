"""Automatic overlapped tiling inside a TensorRT engine's optimization profile.

The user never chooses a tile size: tiles are as large as the engine's profile
allows and always overlap their neighbours. At every edge shared with a
neighbour, a tile's outermost `overlap` pixels (where the model lacks context)
are discarded, and the rest of the shared band is cross-faded, so no hard tile
edge reaches the output. Tiles at the right and bottom edges are moved inward,
so every tile is full-size and holds only real image content; padding (then
cropping) is used only when the image is smaller than one tile, and it reflects
the image, never adds black.

This module does not import TensorRT, so it can be tested without a GPU.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from api import Progress

TILE_ALIGNMENT = 64
# Context margin discarded at each shared tile edge; the cross-fade band has
# the same width, so neighbouring tiles overlap by at least 3 * TILE_OVERLAP.
TILE_OVERLAP = 16


@dataclass(frozen=True)
class ShapeBounds:
    """The input height/width range of an engine's optimization profile."""

    min_h: int
    min_w: int
    max_h: int
    max_w: int


class TileRunner(Protocol):
    """Runs tiles in submission order; at most two tiles are in flight."""

    def submit(self, tile: np.ndarray) -> None:
        """Start upscaling one HWC float32 tile."""
        ...

    def collect(self, height: int, width: int) -> np.ndarray:
        """Wait for the oldest submitted tile and return the top-left `height` x
        `width` pixels of its HWC float32 result as an array the caller owns."""
        ...


def _align_up(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def _tile_extent(max_size: int, min_size: int) -> int:
    aligned = max_size // TILE_ALIGNMENT * TILE_ALIGNMENT
    return aligned if aligned >= min_size else max_size


def _padded_extent(length: int, min_size: int, tile: int) -> int:
    """The engine input size used for a tile of `length` real pixels."""
    return min(max(_align_up(length, TILE_ALIGNMENT), min_size), tile)


def tile_starts(length: int, tile: int, overlap: int = TILE_OVERLAP) -> list[int]:
    """Start offsets of tiles that cover `length` pixels.

    Every tile is `tile` pixels long (unless the whole axis is shorter), and
    neighbouring tiles overlap by at least `3 * overlap` pixels.
    """
    if length <= tile:
        return [0]
    step = tile - 3 * overlap
    if step <= 0:
        raise ValueError(f"Tile size {tile} is too small for an overlap of {overlap}.")
    starts = [0]
    while starts[-1] + tile < length:
        starts.append(min(starts[-1] + step, length - tile))
    return starts


def axis_weights(
    starts: list[int], tile: int, length: int, scale: int, overlap: int = TILE_OVERLAP
) -> list[np.ndarray]:
    """Blend weights (in output pixels) for the tiles of one axis.

    At each edge shared with a neighbour, the outer `overlap` input pixels get
    weight zero and the rest of the shared band is a linear ramp. The weights
    are normalised to sum to exactly one at every output pixel.
    """
    raw: list[np.ndarray] = []
    for i, start in enumerate(starts):
        n = min(tile, length - start) * scale
        w = np.ones(n, dtype=np.float32)
        if i > 0:
            shared = (starts[i - 1] + tile - start) * scale
            cut = overlap * scale
            band = shared - 2 * cut
            w[:cut] = 0
            w[cut : cut + band] *= (np.arange(band, dtype=np.float32) + 0.5) / band
        if i < len(starts) - 1:
            shared = (start + tile - starts[i + 1]) * scale
            cut = overlap * scale
            band = shared - 2 * cut
            w[n - cut :] = 0
            w[n - cut - band : n - cut] *= (
                np.arange(band, 0, -1, dtype=np.float32) - 0.5
            ) / band
        raw.append(w)

    total = np.zeros(length * scale, dtype=np.float32)
    for start, w in zip(starts, raw):
        total[start * scale : start * scale + w.size] += w
    return [w / total[s * scale : s * scale + w.size] for s, w in zip(starts, raw)]


def _pad_tile(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    h, w = tile.shape[:2]
    if (h, w) == (height, width):
        return tile
    pad = ((0, height - h), (0, width - w), (0, 0))
    # reflect needs at least two pixels along a padded axis
    mode = "reflect" if (h > 1 or height == h) and (w > 1 or width == w) else "edge"
    return np.pad(tile, pad, mode=mode)


def _blend(
    result: np.ndarray | None,
    tile_out: np.ndarray,
    y0: int,
    x0: int,
    wy: np.ndarray,
    wx: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    if result is None:
        result = np.zeros((out_h, out_w, tile_out.shape[2]), dtype=np.float32)
    weight = wy[:, None, None] * wx[None, :, None]
    result[y0 : y0 + wy.size, x0 : x0 + wx.size] += tile_out * weight
    return result


def tiled_upscale(
    img: np.ndarray,
    scale: int,
    bounds: ShapeBounds,
    runner: TileRunner,
    progress: Progress | None = None,
    overlap: int = TILE_OVERLAP,
) -> np.ndarray:
    """Upscale an HWC float32 image by tiling it within the engine's profile."""
    h, w = img.shape[:2]
    tile_h = _tile_extent(bounds.max_h, bounds.min_h)
    tile_w = _tile_extent(bounds.max_w, bounds.min_w)
    ys = tile_starts(h, tile_h, overlap)
    xs = tile_starts(w, tile_w, overlap)
    if len(ys) == 1 and len(xs) == 1:
        # One tile: every blend weight is exactly 1, so the cropped output is
        # the result (blending would only re-read it).
        if progress is not None:
            progress.check_aborted()
        runner.submit(
            _pad_tile(
                img,
                _padded_extent(h, bounds.min_h, tile_h),
                _padded_extent(w, bounds.min_w, tile_w),
            )
        )
        single = runner.collect(h * scale, w * scale)
        if progress is not None:
            progress.set_progress(1)
        return single

    wys = axis_weights(ys, tile_h, h, scale, overlap)
    wxs = axis_weights(xs, tile_w, w, scale, overlap)
    jobs = [(iy, ix) for iy in range(len(ys)) for ix in range(len(xs))]

    result: np.ndarray | None = None
    pending: deque[tuple[int, int]] = deque()
    for done, (iy, ix) in enumerate(jobs):
        if progress is not None:
            progress.check_aborted()
        y, x = ys[iy], xs[ix]
        th, tw = min(tile_h, h - y), min(tile_w, w - x)
        tile = _pad_tile(
            img[y : y + th, x : x + tw],
            _padded_extent(th, bounds.min_h, tile_h),
            _padded_extent(tw, bounds.min_w, tile_w),
        )
        runner.submit(tile)
        pending.append((iy, ix))
        while len(pending) >= 2 or (pending and done == len(jobs) - 1):
            py, px = pending.popleft()
            result = _blend(
                result,
                runner.collect(wys[py].size, wxs[px].size),
                ys[py] * scale,
                xs[px] * scale,
                wys[py],
                wxs[px],
                h * scale,
                w * scale,
            )
        if progress is not None:
            progress.set_progress((done + 1) / len(jobs))

    if result is None:
        raise ValueError("Cannot upscale an empty image.")
    return result
