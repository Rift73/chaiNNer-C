"""The one image-size limit of every decode route (2026-10-19): 40,000 x 40,000
pixels in all. An image up to that area loads without a warning; a larger one is
refused, and the decoder chain falls through as it does for any refusal.

The worker (server.py) calls apply() at its start, before any module imports cv2, so
OpenCV reads its limit at import and the worker's child processes inherit it; a user's
own OPENCV_IO_MAX_IMAGE_PIXELS wins. Pillow's part is set when a decoding module first
takes Pillow through pil_image(): the worker's start imports no third-party module (a
fresh runtime has no Pillow until the dependency manager installs it, and the system
C++ runtime must load before Pillow's DLLs). The worker is the one process that decodes
image files (the VapourSynth and TensorRT children import neither cv2 nor Pillow), and
the host never imports this module. The native decoders hold the same value
(CN_MAX_IMAGE_PIXELS, native/include/image_codecs.h).
"""

import os
import warnings
from types import ModuleType

MAX_IMAGE_PIXELS = 40_000 * 40_000

_pillow_limited = False


def apply() -> None:
    """OpenCV's part of the limit; it counts only before cv2 is imported."""
    os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(MAX_IMAGE_PIXELS))


def pil_image() -> ModuleType:
    """PIL.Image with the limit set, the first time in this process: Image.MAX_IMAGE_PIXELS,
    and its DecompressionBombWarning an error, as Pillow warns above its limit and refuses
    above twice it. Every module that decodes with Pillow takes Image from here."""
    global _pillow_limited
    from PIL import Image

    if not _pillow_limited:
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        _pillow_limited = True
    return Image
