"""Image codec adapters; application policy and file ownership are native."""

from __future__ import annotations

from pathlib import Path

# `x as x` imports: the native mirror reads these names in native/src/image_io.cpp
import cv2 as cv2
import numpy as np

from nodes.utils.utils import split_file_path as split_file_path

from .native_graph import graph


def cv_save_image(
    path: Path | str, img: np.ndarray, params: list[int], png_effort: int = 0
):
    """png_effort: the 8-bit PNG encoder's effort for a PNG without parameters (0
    the default PNG; Save Image's Compression preset minus 1)."""
    return graph().image_io_cv_save(globals(), path, img, params, png_effort)
