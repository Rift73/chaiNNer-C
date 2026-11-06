"""Automatic overlapped tiling inside a TensorRT engine's optimization profile.

The user never chooses a tile size: tiles are as large as the engine's profile
allows and always overlap their neighbours. At every edge shared with a
neighbour, a tile's outermost `overlap` pixels (where the model lacks context)
are discarded, and the rest of the shared band is cross-faded, so no hard tile
edge reaches the output. Tiles at the right and bottom edges are moved inward,
so every tile is full-size and holds only real image content; padding (then
cropping) is used only when the image is smaller than one tile, and it reflects
the image, never adds black.

A caller may ask for balanced tiles: the same number of tiles, each as small as the
engine's sizes allow, spread evenly (`balanced_tiles`). They change the output.

This module does not import TensorRT, so it can be tested without a GPU.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from api import Progress

from ..native_buffers import freeze_normalized
from ..native_framework_shared import accumulate_tile
from ..upscale.white_pass import current_white_pass

TILE_ALIGNMENT = 64
# Context margin discarded at each shared tile edge; the cross-fade band has
# the same width, so neighbouring tiles overlap by at least 3 * TILE_OVERLAP.
TILE_OVERLAP = 16


@dataclass(frozen=True)
class ShapeBounds:
    """The input height/width range of an engine's optimization profile, and the
    multiple its input sizes must be (a DUAL or DRFT engine names it; others 64)."""

    min_h: int
    min_w: int
    max_h: int
    max_w: int
    alignment: int = TILE_ALIGNMENT


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


def _tile_extent(max_size: int, min_size: int, alignment: int) -> int:
    aligned = max_size // alignment * alignment
    return aligned if aligned >= min_size else max_size


def _padded_extent(length: int, min_size: int, tile: int, alignment: int) -> int:
    """The engine input size used for a tile of `length` real pixels."""
    return min(max(_align_up(length, alignment), min_size), tile)


def tile_starts(length: int, tile: int, overlap: int = TILE_OVERLAP) -> list[int]:
    """Start offsets of tiles that cover `length` pixels.

    Every tile is `tile` pixels long (unless the whole axis is shorter), and
    neighbouring tiles overlap by at least `3 * overlap` pixels.
    """
    if length <= tile:
        return [0]
    step = tile - 3 * overlap
    if step <= 0:
        raise ValueError(
            f"This image ({length} px) is larger than the TensorRT engine's input"
            f" ({tile} px), and tiling it needs an engine input over"
            f" {3 * overlap} px. The engine still upscales images up to {tile} px;"
            f" for larger ones, build it with a size over {3 * overlap} px."
        )
    starts = [0]
    while starts[-1] + tile < length:
        starts.append(min(starts[-1] + step, length - tile))
    return starts


def balanced_tiles(
    length: int, tile: int, min_size: int, alignment: int, overlap: int = TILE_OVERLAP
) -> tuple[int, list[int]]:
    """The tile size and start offsets of one axis that use as many tiles as `tile_starts`,
    each the smallest the engine allows, and spread them evenly: neighbours overlap by at
    least `3 * overlap` pixels, and what the size leaves over goes to every seam alike."""
    starts = tile_starts(length, tile, overlap)
    count = len(starts)
    if count == 1:
        return tile, starts
    needed = _align_up(length + (count - 1) * 3 * overlap, count) // count
    size = _padded_extent(needed, min_size, tile, alignment)
    return size, [i * (length - size) // (count - 1) for i in range(count)]


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


def one_tile(height: int, width: int, bounds: ShapeBounds) -> bool:
    """Whether an image of this size runs as one tile."""
    return height <= _tile_extent(
        bounds.max_h, bounds.min_h, bounds.alignment
    ) and width <= _tile_extent(bounds.max_w, bounds.min_w, bounds.alignment)


def tile_input(height: int, width: int, bounds: ShapeBounds) -> tuple[int, int]:
    """The engine input (height, width) of the tiles of an image of this size."""
    tile_h = _tile_extent(bounds.max_h, bounds.min_h, bounds.alignment)
    tile_w = _tile_extent(bounds.max_w, bounds.min_w, bounds.alignment)
    return (
        _padded_extent(min(height, tile_h), bounds.min_h, tile_h, bounds.alignment),
        _padded_extent(min(width, tile_w), bounds.min_w, tile_w, bounds.alignment),
    )


def smaller_tiles(bounds: ShapeBounds, height: int, width: int) -> ShapeBounds | None:
    """Bounds for tiles half the size of a `height` x `width` tile that ran out of memory,
    on each axis as far as the engine's sizes and the tiles' overlap allow; None when
    neither axis can go smaller."""

    def smaller(size: int, min_size: int) -> int:
        smallest = _align_up(max(min_size, 3 * TILE_OVERLAP + 1), bounds.alignment)
        return min(size, max(_align_up(size // 2, bounds.alignment), smallest))

    h, w = smaller(height, bounds.min_h), smaller(width, bounds.min_w)
    if (h, w) == (height, width):
        return None
    return ShapeBounds(bounds.min_h, bounds.min_w, h, w, bounds.alignment)


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
    # result[window] += tile_out * (wy[:, None, None] * wx[None, :, None]), as NumPy
    # computes it, in one multithreaded pass
    accumulate_tile(result[y0 : y0 + wy.size, x0 : x0 + wx.size], tile_out, wy, wx)
    return result


def tiled_upscale(
    img: np.ndarray,
    scale: int,
    bounds: ShapeBounds,
    runner: TileRunner,
    progress: Progress | None = None,
    overlap: int = TILE_OVERLAP,
    *,
    model_input: Callable[[np.ndarray], np.ndarray] | None = None,
    opaque_is_same: bool = False,
    clip: bool = False,
    balanced: bool = False,
    before_clip: Callable[[np.ndarray], None] | None = None,
) -> np.ndarray:
    """Upscale an HWC float32 image, or a grey HxW one, by tiling it within the
    engine's profile. A grey image that stays grey comes back HxW.

    model_input: a tile as the engine gets it, its order of channels aside, for the
    transparency path to reuse tiles (white_pass.py); a grey image runs both passes.

    opaque_is_same: the engine takes FP32 and rounds after its first convolution
    (white_pass.py).

    clip: the result is clip_image's, frozen=True, made in place in the one array the
    image already takes, instead of in a second array. For a runner whose `collect`
    gives a fresh array that nothing else refers to.

    balanced: an image of several tiles uses `balanced_tiles`.

    before_clip: sees the result as the model made it: the clip turns infinity to white or
    black."""
    if img.ndim == 2:
        out = tiled_upscale(
            img[:, :, np.newaxis],
            scale,
            bounds,
            runner,
            progress,
            overlap,
            clip=clip,
            balanced=balanced,
            before_clip=before_clip,
        )
        return out[:, :, 0] if out.shape[2] == 1 else out
    h, w = img.shape[:2]
    tile_h = _tile_extent(bounds.max_h, bounds.min_h, bounds.alignment)
    tile_w = _tile_extent(bounds.max_w, bounds.min_w, bounds.alignment)
    if balanced:
        tile_h, ys = balanced_tiles(h, tile_h, bounds.min_h, bounds.alignment, overlap)
        tile_w, xs = balanced_tiles(w, tile_w, bounds.min_w, bounds.alignment, overlap)
    else:
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
                _padded_extent(h, bounds.min_h, tile_h, bounds.alignment),
                _padded_extent(w, bounds.min_w, tile_w, bounds.alignment),
            )
        )
        single = runner.collect(h * scale, w * scale)
        if progress is not None:
            progress.set_progress(1)
        if before_clip is not None:
            before_clip(single)
        return freeze_normalized(single, clamp=True) if clip else single

    wys = axis_weights(ys, tile_h, h, scale, overlap)
    wxs = axis_weights(xs, tile_w, w, scale, overlap)
    jobs = [(iy, ix) for iy in range(len(ys)) for ix in range(len(xs))]
    tiles = current_white_pass() if model_input is not None else None

    result: np.ndarray | None = None
    pending: deque[tuple[int, int, np.ndarray, np.ndarray | None]] = deque()
    for done, (iy, ix) in enumerate(jobs):
        if progress is not None:
            progress.check_aborted()
        y, x = ys[iy], xs[ix]
        th, tw = min(tile_h, h - y), min(tile_w, w - x)
        tile = img[y : y + th, x : x + tw]
        reused = tiles.reuse(x, y, tile) if tiles is not None else None
        if reused is None:
            runner.submit(
                _pad_tile(
                    tile,
                    _padded_extent(th, bounds.min_h, tile_h, bounds.alignment),
                    _padded_extent(tw, bounds.min_w, tile_w, bounds.alignment),
                )
            )
        pending.append((iy, ix, tile, reused))
        while len(pending) >= 2 or (pending and done == len(jobs) - 1):
            py, px, tile, tile_out = pending.popleft()
            if tile_out is None:
                tile_out = runner.collect(wys[py].size, wxs[px].size)
                if tiles is not None and model_input is not None:
                    tiles.keep(
                        xs[px], ys[py], tile, tile_out, model_input, opaque_is_same
                    )
            result = _blend(
                result,
                tile_out,
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
    if before_clip is not None:
        before_clip(result)
    return freeze_normalized(result, clamp=True) if clip else result
