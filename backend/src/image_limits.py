"""The one image-size limit of every decode route (owner, 2026-10-19): 40,000 x 40,000
pixels in all. An image up to that area loads without a warning; a larger one is
refused, and the decoder chain falls through as it does for any refusal.

Applied by the worker (server.py, apply()) before any module imports cv2, so OpenCV reads
its limit at import and the worker's child processes inherit it; a user's own
OPENCV_IO_MAX_IMAGE_PIXELS wins. The worker is the one process that decodes image
files (the VapourSynth and TensorRT children import neither cv2 nor Pillow), and the
host never imports this module (no Pillow at its start). The native decoders hold the
same value (CN_MAX_IMAGE_PIXELS, native/include/image_codecs.h).
"""

import os
import warnings

from PIL import Image

MAX_IMAGE_PIXELS = 40_000 * 40_000


def apply() -> None:
    """Sets the limit in this process; OpenCV's part only counts before cv2 is imported."""
    os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(MAX_IMAGE_PIXELS))
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    # Pillow warns above its limit and refuses above twice it: one limit refuses above it.
    warnings.simplefilter("error", Image.DecompressionBombWarning)
