from __future__ import annotations

# `x as x` imports: the native mirror reads these names in native/src/image_io.cpp
import io
import platform as platform
import struct
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Union

import cv2 as cv2
import numpy as np
import pillow_avif  # type: ignore # noqa: F401
from PIL import TiffImagePlugin
from sanic.log import logger as logger

import image_limits
from nodes.impl.colour_profile import convert_to_srgb
from nodes.impl.dds.texconv import read_dds_texconv as read_dds_texconv
from nodes.impl.image_formats import get_opencv_formats as get_opencv_formats
from nodes.impl.image_formats import get_pil_formats as get_pil_formats
from nodes.impl.image_tags import ImageMetadata, read_image_tags
from nodes.impl.native_graph import graph
from nodes.properties.inputs import BaseInput, BoolInput, ImageFileInput
from nodes.properties.outputs import (
    DirectoryOutput,
    FileNameOutput,
    ImageMetadataOutput,
    LargeImageOutput,
)
from nodes.utils.utils import get_h_w_c as get_h_w_c
from nodes.utils.utils import split_file_path as split_file_path

from .. import io_group

# Pillow with the image-size limit, which read_pil and the native Pillow routes read live.
Image = image_limits.pil_image()

# Pillow builds its plugin registry on first use, and an open() racing another thread's
# init() can fail to identify a valid file. Load Images decodes items in parallel, so
# the registry is built here, once, in the order a first open() builds it.
Image.preinit()
Image.init()

_Decoder = Callable[[Path], Union[np.ndarray, None]]
"""
An image decoder.

If the given image is naturally not supported, the decoder may return `None`
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


def _read_tiff_straight_alpha(path: Path) -> np.ndarray | None:
    """
    OpenCV reads 8-bit TIFFs through libtiff's RGBA interface, which premultiplies
    unassociated (straight) alpha (upstream chaiNNer #409), and drops the alpha of
    grey+alpha TIFFs, so Pillow decodes 8-bit RGBA and LA TIFFs that declare it. LA
    loads as BGRA with the grey in B, G and R, as OpenCV loads a grey+alpha PNG. The
    stored pixels are turned once, by the shared orient(), as the Orientation tag says;
    every other TIFF stays with OpenCV. The plainest of these files decode natively to
    the same array (image_io.cpp read_tiff_alpha); None runs Pillow. Pillow cannot open
    16-bit grey+alpha: its plainest files decode natively too, as uint16 with the
    samples as stored (as OpenCV gives 16-bit RGBA); the others load as OpenCV decodes
    them, as 8-bit grey without the alpha, and a warning says so.
    """
    native = graph().image_io_read_tiff_alpha(globals(), path)
    if native is not None:
        return native
    with open(path, "rb") as file:
        header = file.read(8)
        if header[2:3] == b"\x2b":  # BigTIFF, as Pillow reads its header
            header += file.read(8)
        tags = TiffImagePlugin.ImageFileDirectory_v2(header)
        file.seek(tags.next)
        tags.load(file)
    if (
        tags.get(277) == 2  # SamplesPerPixel
        and set(tags.get(258, ())) == {16}
        and tags.get(338) in ((1,), (2,))  # alpha, associated or not
    ):
        logger.warning(
            "16-bit grey+alpha TIFF: alpha dropped and depth reduced to 8 bit;"
            f" save it as RGBA or PNG: {path}"
        )
        return None
    # Other depths and alpha kinds stay with OpenCV, decided from the tags before Pillow
    # opens a file it may not parse (an RGB float TIFF failed here with a logged
    # "Decoder pil-tiff-alpha failed" on every load).
    if (
        set(tags.get(258, ())) != {8}  # BitsPerSample
        or tags.get(338) != (2,)  # ExtraSamples: unassociated alpha
    ):
        return None
    # The plugin class reads only the header; Image.open's size check would also warn
    # on large TIFFs that this decoder leaves to OpenCV.
    with TiffImagePlugin.TiffImageFile(path) as im:
        if im.mode not in ("RGBA", "LA"):
            return None
    orientation = tags.get(274, 1)
    source: Path | io.BytesIO = path
    if orientation != 1:
        # Pillow turns a TIFF itself, but decodes an uncompressed one with Orientation
        # 5-8 at the turned size (12.3). Decode a copy whose tag says 1 instead.
        data = bytearray(Path(path).read_bytes())
        endian = "<" if tags.prefix == b"II" else ">"
        count, value_at, size = (
            ("Q", 12, 20) if header[2:3] == b"\x2b" else ("H", 8, 12)
        )
        first = tags.offset + struct.calcsize(count)
        entries = struct.unpack_from(endian + count, data, tags.offset)[0]
        for entry in range(first, first + size * entries, size):
            tag, kind = struct.unpack_from(endian + "HH", data, entry)
            if tag == 274:
                value = endian + ("H" if kind == 3 else "I")
                struct.pack_into(value, data, entry + value_at, 1)
        source = io.BytesIO(data)
    with TiffImagePlugin.TiffImageFile(source) as im:
        rgba = im.convert("RGBA") if im.mode == "LA" else im
        pixels = cv2.cvtColor(np.array(rgba), cv2.COLOR_RGBA2BGRA)
    if not isinstance(orientation, int):
        orientation = 1  # a malformed tag, as Pillow ignores it
    return graph().image_io_orient(globals(), pixels, orientation)


def _for_ext(ext: str | Iterable[str], decoder: _Decoder) -> _Decoder:
    return graph().image_io_for_ext(ext, decoder, get_ext)


_decoders: list[tuple[str, _Decoder]] = [
    ("pil-jpeg", _for_ext([".jpg", ".jpeg"], _read_pil)),
    ("pil-tiff-alpha", _for_ext([".tif", ".tiff"], _read_tiff_straight_alpha)),
    ("cv", _read_cv),
    ("texconv-dds", _read_dds),
    ("pil", _read_pil),
]


def convert_to_srgb_input() -> BaseInput:
    """Load Image's and Load Images' Convert To sRGB."""
    return BoolInput("Convert To sRGB", default=False).with_docs(
        "Converts an image with an embedded colour profile (Adobe RGB, Display P3, ProPhoto, ...) to sRGB, so that it shows the colours the file means. Images without a profile and images already in sRGB load unchanged.",
        "When off, the pixel values stay as the file stores them: connect Metadata to Save Image to keep the profile that gives them their colours.",
    )


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
        ),
        convert_to_srgb_input().with_id(1),
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
        ImageMetadataOutput().with_id(3),
    ],
    side_effects=True,
)
def load_image_node(
    path: Path, convert_colours: bool = False
) -> tuple[np.ndarray, Path, str, ImageMetadata]:
    image, directory, name = graph().image_io_load(globals(), path)
    if not convert_colours:
        return image, directory, name, path
    tags = read_image_tags(path)
    if tags.icc_profile is None:
        return image, directory, name, tags
    converted = convert_to_srgb(image, tags.icc_profile, path)
    if converted is not image:
        tags = replace(tags, icc_profile=None)  # the pixels are sRGB's now
    return converted, directory, name, tags
