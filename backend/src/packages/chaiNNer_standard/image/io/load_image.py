from __future__ import annotations

# `x as x` imports: the native mirror reads these names in native/src/image_io.cpp
import os as os
import platform as platform
from pathlib import Path
from typing import Callable, Iterable, Union

import cv2 as cv2
import numpy as np
import pillow_avif  # type: ignore # noqa: F401
from PIL import Image
from sanic.log import logger as logger

from nodes.impl.dds.texconv import dds_to_png_texconv as dds_to_png_texconv
from nodes.impl.image_formats import get_available_image_formats
from nodes.impl.image_formats import get_opencv_formats as get_opencv_formats
from nodes.impl.image_formats import get_pil_formats as get_pil_formats
from nodes.impl.native_graph import graph
from nodes.properties.inputs import ImageFileInput
from nodes.properties.outputs import DirectoryOutput, FileNameOutput, LargeImageOutput
from nodes.utils.utils import get_h_w_c as get_h_w_c
from nodes.utils.utils import split_file_path as split_file_path

from .. import io_group

# Pillow builds its plugin registry on first use, and an open() racing another thread's
# init() can fail to identify a valid file. Load Images decodes items in parallel
# (SP3b U1), so the registry is built here, once, in the order a first open() builds it.
Image.preinit()
Image.init()

_Decoder = Callable[[Path], Union[np.ndarray, None]]
"""
An image decoder.

Of the given image is naturally not supported, the decoder may return `None`
instead of raising an exception. E.g. when the file extension indicates an
unsupported format.
"""


def get_ext(path: Path | str) -> str:
    return graph().image_io_ext(globals(), path)


def remove_unnecessary_alpha(img: np.ndarray) -> np.ndarray:
    """
    Removes the alpha channel from an image if it is not used.
    """
    return graph().image_io_alpha(globals(), img)


def _read_cv(path: Path) -> np.ndarray | None:
    return graph().image_io_read_cv(globals(), path)


def _read_pil(path: Path) -> np.ndarray | None:
    return graph().image_io_read_pil(globals(), path)


def _read_dds(path: Path) -> np.ndarray | None:
    return graph().image_io_read_dds(globals(), path)


def _for_ext(ext: str | Iterable[str], decoder: _Decoder) -> _Decoder:
    return graph().image_io_for_ext(ext, decoder, get_ext)


_decoders: list[tuple[str, _Decoder]] = [
    ("pil-jpeg", _for_ext([".jpg", ".jpeg"], _read_pil)),
    ("cv", _read_cv),
    ("texconv-dds", _read_dds),
    ("pil", _read_pil),
]

valid_formats = get_available_image_formats()


@io_group.register(
    schema_id="chainner:image:load",
    name="Load Image",
    description=(
        "Load image from specified file. This node will output the loaded image, the"
        " directory of the image file, and the name of the image file (without file"
        " extension)."
    ),
    icon="BsFillImageFill",
    inputs=[
        ImageFileInput(primary_input=True).with_docs(
            "Select the path of an image file."
        )
    ],
    outputs=[
        LargeImageOutput()
        .with_docs(
            "The node will display a preview of the selected image as well as type"
            " information for it. Connect this output to the input of another node to"
            " pass the image to it."
        )
        .suggest(),
        DirectoryOutput("Directory", of_input=0),
        FileNameOutput("Name", of_input=0),
    ],
    side_effects=True,
)
def load_image_node(path: Path) -> tuple[np.ndarray, Path, str]:
    return graph().image_io_load(globals(), path)
