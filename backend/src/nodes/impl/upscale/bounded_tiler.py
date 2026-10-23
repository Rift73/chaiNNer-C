from __future__ import annotations

from ...utils.utils import Size
from .tiler import Tiler


class BoundedTiler(Tiler):
    """`tiler`, with a bound on the splits that cannot make the tiles smaller, for
    auto_split. A tiler at its minimum (16 px for the max tile size, or a multiple that
    rounds back up) returns the size it was given, and the auto split retried that size
    for ever, without a message, when a model needs more memory than is free even for
    the smallest tiles (or is damaged). A retry at the same size can still pass once
    memory is freed, so it gets as many tries as the exact split gets splits.

    Not in auto_split.py or tiler.py: generate_tiling_cpp.py writes those from upstream."""

    RETRIES = 20

    def __init__(self, tiler: Tiler) -> None:
        self.tiler = tiler
        self.retries = 0

    def allow_smaller_tile_size(self) -> bool:
        return self.tiler.allow_smaller_tile_size()

    def starting_tile_size(self, width: int, height: int, channels: int) -> Size:
        return self.tiler.starting_tile_size(width, height, channels)

    def split(self, tile_size: Size) -> Size:
        smaller = self.tiler.split(tile_size)
        if tuple(smaller) == tuple(tile_size):
            self.retries += 1
            if self.retries > self.RETRIES:
                raise ValueError(
                    f"The image cannot be split into tiles smaller than {smaller[0]}x"
                    f"{smaller[1]} px, which still failed {self.RETRIES} more times."
                    " The model needs more memory than is free, or it is damaged."
                )
        return smaller
