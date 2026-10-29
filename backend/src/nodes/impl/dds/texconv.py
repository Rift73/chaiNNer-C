from __future__ import annotations

import os

# `x as x` imports: the native mirror reads these names in native/src/image_io.cpp
import platform as platform
import shutil as shutil
import subprocess as subprocess
import sys
from pathlib import Path
from tempfile import mkdtemp as mkdtemp
from typing import Literal, NamedTuple

import numpy as np
from sanic.log import logger as logger

from nodes.impl.native_graph import graph

from ...utils.utils import split_file_path as split_file_path
from ..image_utils import normalize as normalize
from ..image_utils import to_uint8 as to_uint8
from ..image_utils import to_uint16 as to_uint16
from .format import DXGI_CODES as DXGI_CODES
from .format import LEGACY_TO_DXGI, DxgiFormat
from .format import SRGB_FORMATS as SRGB_FORMATS

__TEXCONV_DIR = os.path.join(
    os.path.dirname(sys.modules["__main__"].__file__),  # type: ignore
    "texconv",  # type: ignore
)
# Microsoft's texconv.exe on Windows; on Linux chaiNNer-C's own (native/texconv)
__TEXCONV_EXE = os.path.join(
    __TEXCONV_DIR, "texconv.exe" if platform.system() == "Windows" else "texconv"
)

ColourConversion = Literal["keep", "srgb_to_linear", "linear_to_srgb"]
"""What happens to the colour values written: an sRGB or linear format is only a tag."""

MipFilter = Literal["default", "POINT", "LINEAR", "CUBIC", "FANT", "BOX", "TRIANGLE"]
"""texconv's -if for generated mips; "default" leaves texconv its own choice."""

TextureKind = Literal["array", "cubemap", "volume"]
"""A DDS of several images: array elements (one image: a 2D texture), cubemap faces
(6 per cube: +X, -X, +Y, -Y, +Z, -Z) or a volume's depth slices."""


def __decode(b: bytes) -> str:
    return graph().image_io_decode_bytes(b)


def __run_texconv(args: list[str], error_message: str):
    return graph().image_io_run_texconv(globals(), args, error_message)


def read_dds_texconv(
    path: Path, mip_level: int = 0, layer: int = 0, depth_slice: int = 0
) -> np.ndarray:
    """
    Reads one image of the given DDS file, by default the first at full size: texconv
    converts the file to an uncompressed temporary DDS (8-bit, 16-bit or float, as its
    format needs). A layer is an array element, a cubemap's faces counting as 6 layers
    per cube (+X, -X, +Y, -Y, +Z, -Z); a slice is a volume texture's depth slice.
    """
    return graph().image_io_dds_read(globals(), path, mip_level, layer, depth_slice)


def read_dds_images(path: Path, mip_level: int = 0) -> list[np.ndarray]:
    """
    Reads every image of a mip level of the given DDS file in one texconv conversion:
    each layer (cubemap faces), or each depth slice of a volume texture.
    """
    return graph().image_io_dds_read_all(globals(), path, mip_level)


class DdsInfo(NamedTuple):
    format: str
    """The DXGI format, the FourCC of a legacy one (DXT1-DXT5, ATI1, ATI2), or the name
    of a legacy mask format DXGI lacks ("R8G8B8 (legacy)")."""
    width: int
    height: int
    depth: int
    mip_levels: int
    layers: int
    """Array elements, 6 per cube for a cubemap."""
    cubemap: bool


_DXGI_NAMES = {code: name for name, code in DXGI_CODES.items()}
_LEGACY_FOURCCS = {*LEGACY_TO_DXGI, "DXT2", "DXT4"}


def read_dds_info(path: Path) -> DdsInfo:
    """The header of the given DDS file."""
    code, fourcc, masks, width, height, depth, mips, array, cubemap = (
        graph().image_io_dds_info(path)
    )
    if fourcc in _LEGACY_FOURCCS:
        name = fourcc
    elif masks:  # a mask format DXGI lacks
        name = masks
    else:
        name = _DXGI_NAMES.get(code, "UNKNOWN")
    return DdsInfo(
        name, width, height, depth, mips, array * 6 if cubemap else array, cubemap
    )


def save_as_dds(
    path: Path,
    image: np.ndarray | list[np.ndarray],
    dds_format: DxgiFormat,
    mipmap_levels: int = 0,
    uniform_weighting: bool = False,
    dithering: bool = False,
    minimal_compression: bool = False,
    maximum_compression: bool = False,
    dx9: bool = False,
    separate_alpha: bool = False,
    colour_conversion: ColourConversion = "keep",
    premultiply_alpha: bool = False,
    gpu: bool = True,
    mip_filter: MipFilter = "default",
    kind: TextureKind = "array",
):
    """
    Saves an image, or a list of images of one size as a texture of the given kind,
    as DDS using texconv: chaiNNer writes it as an uncompressed DDS of the format's
    depth (8-bit, 16-bit or float) and texconv converts that.
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
            colour_conversion,
            premultiply_alpha,
            gpu,
            mip_filter,
            kind,
        ),
    )
