from __future__ import annotations

# `x as x` imports: the native mirror reads these names in native/include/tiling_control.hpp
import math as math
from typing import Callable, Union

import numpy as np
from sanic.log import logger as logger

from ...utils.utils import Region as Region
from ...utils.utils import Size
from ...utils.utils import get_h_w_c as get_h_w_c
from ..native_graph import graph
from .exact_split import exact_split as exact_split
from .tile_blending import BlendDirection as BlendDirection
from .tile_blending import TileBlender as TileBlender
from .tile_blending import TileOverlap as TileOverlap
from .tile_blending import half_sin_blend_fn as half_sin_blend_fn
from .tiler import Tiler


class Split:
    pass


SplitImageOp = Callable[[np.ndarray, Region], Union[np.ndarray, Split]]


def auto_split(
    img: np.ndarray, upscale: SplitImageOp, tiler: Tiler, overlap: int = 16
) -> np.ndarray:
    return graph().tiling_installed_auto_split_auto_split(
        globals(), img, upscale, tiler, overlap
    )


class _SplitEx(Exception):
    pass


def _exact_split(
    img: np.ndarray,
    upscale: SplitImageOp,
    starting_tile_size: Size,
    split_tile_size: Callable[[Size], Size],
    overlap: int,
) -> np.ndarray:
    return graph().tiling_installed_auto_split_exact_split(
        globals(), img, upscale, starting_tile_size, split_tile_size, overlap
    )


def _max_split(
    img: np.ndarray,
    upscale: SplitImageOp,
    starting_tile_size: Size,
    split_tile_size: Callable[[Size], Size],
    overlap: int,
) -> np.ndarray:
    return graph().tiling_installed_auto_split_max_split(
        globals(), img, upscale, starting_tile_size, split_tile_size, overlap
    )
