"""BoundedTiler, which the PyTorch, ONNX and NCNN auto splits use, stops the split at
the tiler's smallest tiles: a tile that keeps asking for a split there raises an error
after 20 more tries, where the split before retried the same size for ever (a model
that needs more memory than is free even for 16 px tiles, or a damaged one). A split
that makes the tiles smaller, and a retry at the smallest size that passes, still work
as before."""

from __future__ import annotations

import numpy as np
import pytest

# The tiling is native, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from nodes.impl.ncnn.auto_split import _MultipleTiler
from nodes.impl.upscale.auto_split import Split, auto_split
from nodes.impl.upscale.bounded_tiler import BoundedTiler
from nodes.impl.upscale.tiler import MaxTileSize, Tiler
from nodes.utils.utils import Region


class Endless(Exception):
    """The split asked for more tiles than a bounded split needs."""


def splitting(calls: list[Region], until: int | None = None):
    """An upscale that asks for a split on every call (`until` calls, if given), then
    doubles the tile."""

    def upscale(tile: np.ndarray, region: Region) -> np.ndarray | Split:
        calls.append(region)
        if len(calls) > 1000:
            raise Endless
        if until is None or len(calls) <= until:
            return Split()
        return np.repeat(np.repeat(tile, 2, axis=0), 2, axis=1)

    return upscale


@pytest.mark.parametrize(
    ("size", "tiler", "minimum"),
    [
        (16, MaxTileSize(), "16x16"),
        (40, MaxTileSize(), "16x16"),
        (40, MaxTileSize(32), "16x16"),
        # rounding up to the multiple gives the size back above the 16 px minimum
        (64, _MultipleTiler(MaxTileSize(), 32), "32x32"),
    ],
)
def test_a_split_at_the_smallest_tiles_ends(size: int, tiler: Tiler, minimum: str):
    calls: list[Region] = []
    image = np.zeros((size, size, 3), np.float32)
    with pytest.raises(
        ValueError, match=f"tiles smaller than {minimum} px, which still failed 20"
    ):
        auto_split(image, splitting(calls), BoundedTiler(tiler), overlap=0)
    # the halvings from the starting size, then the first try and 20 more at the minimum
    assert len(calls) < 60


# 64 px tiles: a split to 32, one to 16, then `until - 2` tries again at 16 (22: all 20)
@pytest.mark.parametrize("until", [1, 5, 22])
def test_splits_that_pass_in_time_still_upscale(until: int):
    calls: list[Region] = []
    image = np.random.default_rng(0).random((40, 40, 3), dtype=np.float32)
    result = auto_split(
        image, splitting(calls, until), BoundedTiler(MaxTileSize(64)), overlap=0
    )
    np.testing.assert_array_equal(
        result, np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
    )
