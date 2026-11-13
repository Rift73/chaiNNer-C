from __future__ import annotations

import gc

import numpy as np
import torch

from api import Progress

from ....utils.utils import Region, Size, get_h_w_c
from ...image_op import to_op
from ...upscale.auto_split import Split, auto_split
from ...upscale.grayscale import SplitMode, grayscale_split
from ...upscale.passthrough import passthrough_single_color
from ...upscale.tiler import Tiler
from ..utils import is_out_of_memory, safe_cuda_cache_empty
from .pix_transform import Params, pix_transform


class _PixTiler(Tiler):
    def __init__(self, max_tile_size: int) -> None:
        self.max_tile_size: int = max_tile_size

    def allow_smaller_tile_size(self) -> bool:
        return False

    def starting_tile_size(self, width: int, height: int, channels: int) -> Size:
        square = min(width, height, self.max_tile_size)
        return square, square

    def split(self, tile_size: Size) -> Size:
        # half the tile size plus a bit extra to account for overlap
        size = tile_size[0] // 2 + tile_size[0] // 8
        if size < 16:
            raise ValueError("Cannot split any further.")
        return size, size


def _as_3d(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return img
    return np.expand_dims(img, axis=2)


def pix_transform_auto_split(
    source: np.ndarray,
    guide: np.ndarray,
    device: torch.device,
    params: Params,
    split_mode: SplitMode,
    tile_size: int,
    progress: Progress,
) -> np.ndarray:
    """Runs PixTransform on the source in tiles, with the matching tiles of the guide.

    The source and guide may have any number of channels and any size, as long as the
    guide's size is a whole multiple (greater than 1) of the source's."""

    s_h, s_w, _ = get_h_w_c(source)
    g_h, g_w, _ = get_h_w_c(guide)

    assert g_h > s_h and g_w > s_w, (
        "The guide image must be larger than the source image."
    )
    assert g_w / s_w == g_w // s_w and g_w / s_w == g_h / s_h, (
        "The size of the guide image must be an integer multiple of the size of the source image (e.g. 2x, 3x, 4x, ...)."
    )

    tiler = _PixTiler(tile_size)
    scale = g_w // s_w

    def upscale(tile: np.ndarray, region: Region):
        try:
            tile_guide = region.scale(scale).read_from(guide)
            pix_op = to_op(pix_transform)(
                guide_img=np.transpose(_as_3d(tile_guide), (2, 0, 1)),
                device=device,
                params=params,
                progress=progress,
            )
            # passthrough single colors to speed up alpha channels
            pass_op = to_op(passthrough_single_color)(scale, pix_op)

            return _as_3d(grayscale_split(tile, pass_op, split_mode))
        except (RuntimeError, MemoryError) as e:
            if not is_out_of_memory(e):
                raise
            gc.collect()
            safe_cuda_cache_empty()
            return Split()

    try:
        return auto_split(source, upscale, tiler)
    finally:
        del device
        gc.collect()
        safe_cuda_cache_empty()
