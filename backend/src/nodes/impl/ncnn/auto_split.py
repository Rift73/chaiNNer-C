from __future__ import annotations

import gc
import threading

import numpy as np
from ncnn import ncnn
from sanic.log import logger

from ...utils.utils import get_h_w_c
from ..image_utils import to_uint8
from ..native_framework_images import ncnn_input
from ..native_tensors import cast_numpy
from ..upscale.auto_split import Split, Tiler, auto_split

use_gpu = ncnn.get_gpu_count() > 0
# PyPI ncnn's Extractor has no Vulkan allocator setters, so an extractor takes them
# from the Option it copies at creation. Nets are cached and shared, and node work
# runs on a thread pool, so one upscale's allocators stay on net.opt only while this
# lock is held.
_net_opt_lock = threading.Lock()


def ncnn_auto_split(
    img: np.ndarray,
    net: ncnn.Net,
    input_name: str,
    output_name: str,
    blob_vkallocator: ncnn.VkBlobAllocator | None,
    staging_vkallocator: ncnn.VkStagingAllocator | None,
    tiler: Tiler,
    collect_after: bool = True,
) -> np.ndarray:

    def clear_vkallocators() -> None:
        if blob_vkallocator is not None:
            blob_vkallocator.clear()
        if staging_vkallocator is not None:
            staging_vkallocator.clear()

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
        result = cast_numpy(
            np.asarray(mat_out).transpose(1, 2, 0), np.dtype(np.float32)
        )
        del ex, mat_in, mat_out
        clear_vkallocators()
        return result

    try:
        return auto_split(img, upscale, tiler)
    finally:
        if collect_after:
            gc.collect()
