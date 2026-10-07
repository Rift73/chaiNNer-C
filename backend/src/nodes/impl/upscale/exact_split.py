from __future__ import annotations

# `x as x` imports: the native mirror reads these names in native/include/tiling_control.hpp
import math as math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sanic.log import logger as logger

from ...utils.utils import Padding as Padding
from ...utils.utils import Region as Region
from ...utils.utils import Size
from ...utils.utils import get_h_w_c as get_h_w_c
from ..image_utils import BorderType as BorderType
from ..image_utils import create_border as create_border
from ..native_graph import graph
from .tile_blending import BlendDirection as BlendDirection
from .tile_blending import TileBlender as TileBlender
from .tile_blending import TileOverlap as TileOverlap
from .tile_blending import half_sin_blend_fn as half_sin_blend_fn


def _pad_image(img: np.ndarray, min_size: Size):
    return graph().tiling_installed_exact_split_pad_image(globals(), img, min_size)


@dataclass
class _Segment:
    start: int
    end: int
    start_padding: int
    end_padding: int

    @property
    def length(self) -> int:
        return graph().tiling_installed_exact_split__Segment_length(globals(), self)

    @property
    def padded_length(self) -> int:
        return graph().tiling_installed_exact_split__Segment_padded_length(
            globals(), self
        )


def _exact_split_into_segments(length: int, exact: int, overlap: int) -> list[_Segment]:
    return graph().tiling_installed_exact_split_exact_split_into_segments(
        globals(), length, exact, overlap
    )


def _exact_split_into_regions(
    w: int, h: int, exact_w: int, exact_h: int, overlap: int
) -> list[list[tuple[Region, Padding]]]:
    return graph().tiling_installed_exact_split_exact_split_into_regions(
        globals(), w, h, exact_w, exact_h, overlap
    )


def _exact_split_without_padding(
    img: np.ndarray,
    exact_size: Size,
    upscale: Callable[[np.ndarray, Region], np.ndarray],
    overlap: int,
) -> np.ndarray:
    return graph().tiling_installed_exact_split_exact_split_without_padding(
        globals(), img, exact_size, upscale, overlap
    )


def exact_split(
    img: np.ndarray,
    exact_size: Size,
    upscale: Callable[[np.ndarray, Region], np.ndarray],
    overlap: int = 16,
) -> np.ndarray:
    return graph().tiling_installed_exact_split_exact_split(
        globals(), img, exact_size, upscale, overlap
    )
