from __future__ import annotations

import gc
import math
import threading

import numpy as np
from ncnn import ncnn
from sanic.log import logger

from ...utils.utils import Size, get_h_w_c
from ..image_utils import to_uint8
from ..native_framework_images import ncnn_input, planar_to_interleaved
from ..upscale.auto_split import Split, Tiler, auto_split

use_gpu = ncnn.get_gpu_count() > 0
# PyPI ncnn's Extractor has no Vulkan allocator setters, so an extractor takes them
# from the Option it copies at creation. Nets are cached and shared, and node work
# runs on a thread pool, so one upscale's allocators stay on net.opt only while this
# lock is held.
_net_opt_lock = threading.Lock()


class _MultipleTiler(Tiler):
    """Keeps every tile's size and position a multiple of `multiple`.

    The tiler cuts an axis of `length` pixels into ceil(length / max) tiles of
    ceil(length / count) pixels each, padded by the overlap. For an image whose size
    is a multiple, and an even overlap, tiles stay a multiple when that tile size
    is one, so this returns the largest such tile size within the wrapped tiler's.
    """

    def __init__(self, tiler: Tiler, multiple: int) -> None:
        self.tiler = tiler
        self.multiple = multiple
        self.size: Size = (multiple, multiple)

    def allow_smaller_tile_size(self) -> bool:
        return self.tiler.allow_smaller_tile_size()

    def starting_tile_size(self, width: int, height: int, channels: int) -> Size:
        self.size = (width, height)
        return self._fit(self.tiler.starting_tile_size(width, height, channels))

    def split(self, tile_size: Size) -> Size:
        return self._fit(self.tiler.split(tile_size))

    def _fit(self, tile_size: Size) -> Size:
        return (
            self._fit_axis(self.size[0], tile_size[0]),
            self._fit_axis(self.size[1], tile_size[1]),
        )

    def _fit_axis(self, length: int, max_size: int) -> int:
        if max_size >= length:
            return max_size
        count = math.ceil(length / max(max_size, self.multiple))
        while math.ceil(length / count) % self.multiple:
            count += 1
        return math.ceil(length / count)


def ncnn_auto_split(
    img: np.ndarray,
    net: ncnn.Net,
    input_name: str,
    output_name: str,
    blob_vkallocator: ncnn.VkBlobAllocator | None,
    staging_vkallocator: ncnn.VkStagingAllocator | None,
    tiler: Tiler,
    collect_after: bool = True,
    size_multiple: int = 1,
    swap_red_blue: bool = False,
) -> np.ndarray:
    # A model that starts with a Reorg needs a height and width that are a multiple
    # of its stride (NcnnModelWrapper.size_multiple), so the image is padded with
    # its edge pixels, every tile is kept a multiple, and the result is cropped
    # back. Other models are tiled as before. swap_red_blue: a result with the
    # image's 3 or 4 channels comes back with red and blue swapped (RGB(A) <->
    # BGR(A)), as it is interleaved.
    h, w = img.shape[:2]
    pad_h, pad_w = -h % size_multiple, -w % size_multiple
    if pad_h or pad_w:
        logger.debug(
            f"Padding the {w}x{h} image to a multiple of {size_multiple} for NCNN."
        )
        img = np.pad(
            img,
            ((0, pad_h), (0, pad_w)) + ((0, 0),) * (img.ndim - 2),
            mode="edge",
        )
    if size_multiple > 1 and tiler.allow_smaller_tile_size():
        tiler = _MultipleTiler(tiler, size_multiple)

    def clear_vkallocators() -> None:
        if blob_vkallocator is not None:
            blob_vkallocator.clear()
        if staging_vkallocator is not None:
            staging_vkallocator.clear()

    whole_image = img

    def upscale(img: np.ndarray, _: object):
        if use_gpu:
            with _net_opt_lock:
                net.opt.blob_vkallocator = blob_vkallocator
                net.opt.workspace_vkallocator = blob_vkallocator
                net.opt.staging_vkallocator = staging_vkallocator
                try:
                    ex = net.create_extractor()
                finally:
                    net.opt.blob_vkallocator = None
                    net.opt.workspace_vkallocator = None
                    net.opt.staging_vkallocator = None
        else:
            ex = net.create_extractor()
        try:
            lr_c = get_h_w_c(img)[2]
            pixel_image = img
            lr_img_fix = to_uint8(pixel_image)
            if lr_c in (1, 3, 4):
                lr_planar = ncnn_input(lr_img_fix)
                mat_in = ncnn.Mat(lr_planar).clone()
            else:
                pixel_type = ncnn.Mat.PixelType.PIXEL_RGBA
                mat_in = ncnn.Mat.from_pixels(
                    lr_img_fix, pixel_type, lr_img_fix.shape[1], lr_img_fix.shape[0]
                )
                mat_in.substract_mean_normalize([], [1 / 255.0] * lr_c)
            # PyPI ncnn reports a failure only by a non-zero return code; it prints
            # the reason to stderr, which the log keeps.
            if ex.input(input_name, mat_in):
                raise ValueError(f"The NCNN model has no blob named {input_name}.")
            ret, mat_out = ex.extract(output_name)
        except Exception as e:
            if "vkQueueSubmit" in str(e):
                ex = None
                del ex
                gc.collect()
                clear_vkallocators()
                raise RuntimeError(
                    "A critical error has occurred. You may need to restart chaiNNer in order for NCNN upscaling to start working again."
                ) from e
            if "failed" in str(e):
                logger.debug("NCNN out of VRAM, clearing VRAM and splitting.")
                ex = None
                del ex
                gc.collect()
                clear_vkallocators()
                return Split()
            else:
                raise
        if ret != 0:
            # The output is empty; reading it would crash the process.
            del ex, mat_in, mat_out
            clear_vkallocators()
            if ret == -100:  # an allocation failed
                logger.debug("NCNN out of memory, clearing memory and splitting.")
                return Split()
            message = (
                f"NCNN failed with error code {ret}. Its reason is in chaiNNer's log."
            )
            if use_gpu:
                message += " You may need to restart chaiNNer in order for NCNN upscaling to start working again."
            raise RuntimeError(message)
        planes = np.asarray(mat_out)
        result = planar_to_interleaved(
            planes,
            swap_red_blue=swap_red_blue and lr_c in (3, 4) and planes.shape[0] == lr_c,
        )
        del ex, mat_in, mat_out
        clear_vkallocators()
        # The tiler needs every tile upscaled by the same whole multiple on both axes,
        # and fails a bare assert otherwise. The whole image (passed as itself, a tile
        # is a view) is returned as it is; upscale_impl checks it against the model's
        # scale.
        h, w = img.shape[:2]
        out_h, out_w = result.shape[:2]
        if img is not whole_image and (
            out_h % h or out_w % w or out_h // h != out_w // w
        ):
            raise ValueError(
                f"The NCNN model returned a {out_w}x{out_h} image for a {w}x{h} tile,"
                " but tiling needs a whole multiple of the tile's size."
                " Models that expect a padded input, such as waifu2x-ncnn-vulkan's,"
                " are not supported."
            )
        return result

    try:
        result = auto_split(img, upscale, tiler)
        if pad_h or pad_w:
            scale = result.shape[0] // img.shape[0]
            result = result[: h * scale, : w * scale]
        return result
    finally:
        if collect_after:
            gc.collect()
