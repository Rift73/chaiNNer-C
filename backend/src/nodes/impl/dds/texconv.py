from __future__ import annotations

import os

# `x as x` imports: the native mirror reads these names in native/src/image_io.cpp
import platform as platform
import shutil as shutil
import subprocess as subprocess
import sys
import uuid as uuid
from pathlib import Path
from tempfile import mkdtemp as mkdtemp

import numpy as np
from sanic.log import logger as logger

from nodes.impl.native_graph import graph

from ...utils.utils import split_file_path as split_file_path
from ..native_image_io import cv_save_image as cv_save_image
from .format import SRGB_FORMATS as SRGB_FORMATS
from .format import DxgiFormat

__TEXCONV_DIR = os.path.join(
    os.path.dirname(sys.modules["__main__"].__file__),  # type: ignore
    "texconv",  # type: ignore
)
__TEXCONV_EXE = os.path.join(__TEXCONV_DIR, "texconv.exe")


def __decode(b: bytes) -> str:
    return graph().image_io_decode_bytes(b)


def __run_texconv(args: list[str], error_message: str):
    return graph().image_io_run_texconv(globals(), args, error_message)


def dds_to_png_texconv(path: Path) -> Path:
    """
    Converts the given DDS file to PNG by creating a temporary PNG file.
    """
    return graph().image_io_dds_png(globals(), path)


def save_as_dds(
    path: Path,
    image: np.ndarray,
    dds_format: DxgiFormat,
    mipmap_levels: int = 0,
    uniform_weighting: bool = False,
    dithering: bool = False,
    minimal_compression: bool = False,
    maximum_compression: bool = False,
    dx9: bool = False,
    separate_alpha: bool = False,
):
    """
    Saves an image as DDS using texconv.
    See the following page for more information on save options:
    https://github.com/Microsoft/DirectXTex/wiki/Texconv
    """
    return graph().image_io_save_dds(
        globals(),
        (
            path,
            image,
            dds_format,
            mipmap_levels,
            uniform_weighting,
            dithering,
            minimal_compression,
            maximum_compression,
            dx9,
            separate_alpha,
        ),
    )
