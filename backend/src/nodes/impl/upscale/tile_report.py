"""The tile size an Upscale Image node used, for its Tile Size output.

Not in auto_split.py or tiler.py: generate_tiling_cpp.py writes those from upstream."""

from __future__ import annotations

import numpy as np

from ...utils.utils import Region
from .auto_split import Split, SplitImageOp


class TileReport:
    """The tiles of a node's upscale. An image can take several passes (transparency,
    Separate Alpha, Custom Scale): `text` tells of the last."""

    def __init__(self) -> None:
        self.text = ""

    def record(self, height: int, width: int, *, whole: bool) -> None:
        self.text = "Whole image" if whole else f"{width}x{height} px"

    def recording(self, img: np.ndarray, upscale: SplitImageOp) -> SplitImageOp:
        """`upscale`, for auto_split's pass over `img`: records the largest tile the model
        took, with its overlap, since the last split."""
        image_size = img.shape[:2]
        largest = (0, 0)

        def recorded(tile: np.ndarray, region: Region) -> np.ndarray | Split:
            nonlocal largest
            result = upscale(tile, region)
            if isinstance(result, Split):
                largest = (0, 0)
            else:
                largest = (
                    max(largest[0], tile.shape[0]),
                    max(largest[1], tile.shape[1]),
                )
                self.record(*largest, whole=largest == image_size)
            return result

        return recorded
