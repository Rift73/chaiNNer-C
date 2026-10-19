from __future__ import annotations

from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import pillow_avif  # type: ignore # noqa: F401

# `x as x` imports: the native mirror reads these names in native/src/image_io.cpp
from PIL import Image as Image
from sanic.log import logger as logger

from api import KeyInfo, Lazy, NestedGroup
from nodes.groups import Condition, if_enum_group, if_group
from nodes.impl.dds.format import (
    BC6H_FORMATS,
    BC7_FORMATS,
    BC123_FORMATS,
    WITH_ALPHA,
    DDSFormat,
)
from nodes.impl.dds.format import LEGACY_TO_DXGI as LEGACY_TO_DXGI
from nodes.impl.dds.format import PREFER_DX9 as PREFER_DX9
from nodes.impl.dds.format import to_dxgi as to_dxgi
from nodes.impl.dds.texconv import save_as_dds as save_as_dds
from nodes.impl.image_utils import to_uint8 as to_uint8
from nodes.impl.image_utils import to_uint16 as to_uint16
from nodes.impl.item_window import Prepared, register_phases
from nodes.impl.native_graph import graph
from nodes.impl.native_image_io import cv_save_image as cv_save_image
from nodes.properties.inputs import (
    BaseInput,
    BoolInput,
    DirectoryInput,
    DropDownGroup,
    DropDownInput,
    EnumInput,
    ImageInput,
    NumberInput,
    RelativePathInput,
    SliderInput,
)
from nodes.utils.utils import get_h_w_c as get_h_w_c

from .. import io_group


class ImageFormat(Enum):
    PNG = "png"
    JPG = "jpg"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"
    WEBP = "webp"
    TGA = "tga"
    DDS = "dds"
    AVIF = "avif"

    @property
    def extension(self) -> str:
        return self.value


IMAGE_FORMAT_LABELS: dict[ImageFormat, str] = {
    ImageFormat.PNG: "PNG",
    ImageFormat.JPG: "JPG",
    ImageFormat.GIF: "GIF",
    ImageFormat.BMP: "BMP",
    ImageFormat.TIFF: "TIFF",
    ImageFormat.WEBP: "WEBP",
    ImageFormat.TGA: "TGA",
    ImageFormat.DDS: "DDS",
    ImageFormat.AVIF: "AVIF",
}


class JpegSubsampling(Enum):
    FACTOR_444 = int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)
    FACTOR_440 = int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_440)
    FACTOR_422 = int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422)
    FACTOR_420 = int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420)


class AvifSubsampling(Enum):
    FACTOR_444 = "4:4:4"
    FACTOR_422 = "4:2:2"
    FACTOR_420 = "4:2:0"
    FACTOR_400 = "4:0:0"


class PngColorDepth(Enum):
    U8 = "u8"
    U16 = "u16"


class TiffColorDepth(Enum):
    U8 = "u8"
    U16 = "u16"
    F32 = "f32"


class TiffCompression(Enum):
    NONE = 1
    LZW = 5
    ZIP = 8

    @property
    def cv2_code(self) -> int:
        # OpenCV is a beautiful piece of software, so surely they won't forget to define the TIFF compression constants, right?
        return self.value


# Every format texconv writes but typeless, depth/stencil, YUV/video, palettized,
# R1, XR_BIAS and sampler feedback ones. Formats of 8 bits or less are written from
# 8-bit values, 16-bit UNORM and SNORM from 16-bit values (SNORM maps 0..1 to -1..1 at
# every depth) and every other from the float values as they are: float formats keep
# negative values and values above 1, integer formats store the values themselves.
SUPPORTED_DDS_FORMATS: list[tuple[DDSFormat, str]] = [
    ("BC1_UNORM_SRGB", "BC1 (4bpp, sRGB, 1-bit Alpha)"),
    ("BC1_UNORM", "BC1 (4bpp, Linear, 1-bit Alpha)"),
    ("BC2_UNORM_SRGB", "BC2 (8bpp, sRGB, 4-bit Alpha)"),
    ("BC2_UNORM", "BC2 (8bpp, Linear, 4-bit Alpha)"),
    ("BC3_UNORM_SRGB", "BC3 (8bpp, sRGB, 8-bit Alpha)"),
    ("BC3_UNORM", "BC3 (8bpp, Linear, 8-bit Alpha)"),
    ("BC4_UNORM", "BC4 (4bpp, Grayscale)"),
    ("BC4_SNORM", "BC4 (4bpp, Signed, Grayscale)"),
    ("BC5_UNORM", "BC5 (8bpp, Unsigned, 2-channel normal)"),
    ("BC5_SNORM", "BC5 (8bpp, Signed, 2-channel normal)"),
    ("BC6H_UF16", "BC6H (8bpp, Unsigned HDR)"),
    ("BC6H_SF16", "BC6H (8bpp, Signed HDR)"),
    ("BC7_UNORM_SRGB", "BC7 (8bpp, sRGB, 8-bit Alpha)"),
    ("BC7_UNORM", "BC7 (8bpp, Linear, 8-bit Alpha)"),
    ("R8G8B8A8_UNORM_SRGB", "RGBA (32bpp, sRGB, 8-bit Alpha)"),
    ("R8G8B8A8_UNORM", "RGBA (32bpp, Linear, 8-bit Alpha)"),
    ("R8G8B8A8_SNORM", "RGBA (32bpp, Signed, 8-bit Alpha)"),
    ("B8G8R8A8_UNORM_SRGB", "BGRA (32bpp, sRGB, 8-bit Alpha)"),
    ("B8G8R8A8_UNORM", "BGRA (32bpp, Linear, 8-bit Alpha)"),
    ("B5G5R5A1_UNORM", "BGRA (16bpp, Linear, 1-bit Alpha)"),
    ("B4G4R4A4_UNORM", "BGRA (16bpp, Linear, 4-bit Alpha)"),
    ("A4B4G4R4_UNORM", "ABGR (16bpp, Linear, 4-bit Alpha)"),
    ("B5G6R5_UNORM", "BGR (16bpp, Linear)"),
    ("B8G8R8X8_UNORM_SRGB", "BGRX (32bpp, sRGB)"),
    ("B8G8R8X8_UNORM", "BGRX (32bpp, Linear)"),
    ("R8G8_B8G8_UNORM", "RGBG (16bpp, Linear, even width)"),
    ("G8R8_G8B8_UNORM", "GRGB (16bpp, Linear, even width)"),
    ("R8G8_UNORM", "RG (16bpp, Linear)"),
    ("R8G8_SNORM", "RG (16bpp, Signed)"),
    ("R8_UNORM", "R (8bpp, Linear)"),
    ("R8_SNORM", "R (8bpp, Signed)"),
    ("A8_UNORM", "A (8bpp, Alpha only)"),
    ("R16G16B16A16_UNORM", "RGBA (64bpp, Linear, 16-bit)"),
    ("R16G16B16A16_SNORM", "RGBA (64bpp, Signed, 16-bit)"),
    ("R10G10B10A2_UNORM", "RGBA (32bpp, Linear, 10-bit, 2-bit Alpha)"),
    ("R16G16_UNORM", "RG (32bpp, Linear, 16-bit)"),
    ("R16G16_SNORM", "RG (32bpp, Signed, 16-bit)"),
    ("R16_UNORM", "R (16bpp, Linear)"),
    ("R16_SNORM", "R (16bpp, Signed)"),
    ("R32G32B32A32_FLOAT", "RGBA (128bpp, 32-bit Float)"),
    ("R32G32B32_FLOAT", "RGB (96bpp, 32-bit Float)"),
    ("R16G16B16A16_FLOAT", "RGBA (64bpp, 16-bit Float)"),
    ("R11G11B10_FLOAT", "RGB (32bpp, 11/11/10-bit Unsigned Float)"),
    ("R9G9B9E5_SHAREDEXP", "RGB (32bpp, Shared Exponent)"),
    ("R32G32_FLOAT", "RG (64bpp, 32-bit Float)"),
    ("R16G16_FLOAT", "RG (32bpp, 16-bit Float)"),
    ("R32_FLOAT", "R (32bpp, 32-bit Float)"),
    ("R16_FLOAT", "R (16bpp, 16-bit Float)"),
    ("R32G32B32A32_UINT", "RGBA (128bpp, 32-bit Unsigned Integer)"),
    ("R32G32B32A32_SINT", "RGBA (128bpp, 32-bit Signed Integer)"),
    ("R32G32B32_UINT", "RGB (96bpp, 32-bit Unsigned Integer)"),
    ("R32G32B32_SINT", "RGB (96bpp, 32-bit Signed Integer)"),
    ("R16G16B16A16_UINT", "RGBA (64bpp, 16-bit Unsigned Integer)"),
    ("R16G16B16A16_SINT", "RGBA (64bpp, 16-bit Signed Integer)"),
    ("R10G10B10A2_UINT", "RGBA (32bpp, 10-bit Unsigned Integer, 2-bit Alpha)"),
    ("R8G8B8A8_UINT", "RGBA (32bpp, 8-bit Unsigned Integer)"),
    ("R8G8B8A8_SINT", "RGBA (32bpp, 8-bit Signed Integer)"),
    ("R32G32_UINT", "RG (64bpp, 32-bit Unsigned Integer)"),
    ("R32G32_SINT", "RG (64bpp, 32-bit Signed Integer)"),
    ("R16G16_UINT", "RG (32bpp, 16-bit Unsigned Integer)"),
    ("R16G16_SINT", "RG (32bpp, 16-bit Signed Integer)"),
    ("R8G8_UINT", "RG (16bpp, 8-bit Unsigned Integer)"),
    ("R8G8_SINT", "RG (16bpp, 8-bit Signed Integer)"),
    ("R32_UINT", "R (32bpp, 32-bit Unsigned Integer)"),
    ("R32_SINT", "R (32bpp, 32-bit Signed Integer)"),
    ("R16_UINT", "R (16bpp, 16-bit Unsigned Integer)"),
    ("R16_SINT", "R (16bpp, 16-bit Signed Integer)"),
    ("R8_UINT", "R (8bpp, 8-bit Unsigned Integer)"),
    ("R8_SINT", "R (8bpp, 8-bit Signed Integer)"),
    ("DXT1", "DXT1 (4bpp, Linear, 1-bit Alpha)"),
    ("DXT3", "DXT3 (8bpp, Linear, 4-bit Alpha)"),
    ("DXT5", "DXT5 (8bpp, Linear, 8-bit Alpha)"),
    ("ATI1", "ATI1 (4bpp, Grayscale)"),
    ("ATI2", "ATI2 (8bpp, 2-channel normal)"),
]


def DdsFormatDropdown() -> DropDownInput:
    return DropDownInput(
        input_type="DdsFormat",
        label="DDS Format",
        options=[{"option": title, "value": f} for f, title in SUPPORTED_DDS_FORMATS],
        associated_type=DDSFormat,
        groups=[
            DropDownGroup("Compressed", start_at="BC1_UNORM_SRGB"),
            DropDownGroup("Uncompressed", start_at="R8G8B8A8_UNORM_SRGB"),
            DropDownGroup("Uncompressed 10/16-bit", start_at="R16G16B16A16_UNORM"),
            DropDownGroup("Floating Point", start_at="R32G32B32A32_FLOAT"),
            DropDownGroup("Integer", start_at="R32G32B32A32_UINT"),
            DropDownGroup("Legacy Compressed", start_at="DXT1"),
        ],
    ).with_docs(
        "How the image's values are stored: 8-bit and 16-bit formats store 0..1 (values outside are clipped); SNORM (signed) formats map 0..1 to -1..1 at every depth; float formats store the values as they are, negative values and values above 1 included (unsigned ones clip negative values); integer formats store the values themselves, so a 0..1 image becomes 0 and 1."
    )


# In the dropdown's order. A set's order changes from one start to the next, and so
# would the node list, which the app caches between starts.
SUPPORTED_BC7_FORMATS = [f for f, _ in SUPPORTED_DDS_FORMATS if f in BC7_FORMATS]
SUPPORTED_BC123_FORMATS = [f for f, _ in SUPPORTED_DDS_FORMATS if f in BC123_FORMATS]
SUPPORTED_WITH_ALPHA = [f for f, _ in SUPPORTED_DDS_FORMATS if f in WITH_ALPHA]
SUPPORTED_GPU_FORMATS = [
    f for f, _ in SUPPORTED_DDS_FORMATS if f in BC6H_FORMATS | BC7_FORMATS
]


class DDSErrorMetric(Enum):
    PERCEPTUAL = 0
    UNIFORM = 1


class BC7Compression(Enum):
    BEST_SPEED = 1
    DEFAULT = 0
    BEST_QUALITY = 2


class DDSMipFilter(Enum):
    DEFAULT = "default"
    POINT = "POINT"
    LINEAR = "LINEAR"
    CUBIC = "CUBIC"
    FANT = "FANT"
    BOX = "BOX"
    TRIANGLE = "TRIANGLE"


class DDSColourConversion(Enum):
    KEEP = "keep"
    SRGB_TO_LINEAR = "srgb_to_linear"
    LINEAR_TO_SRGB = "linear_to_srgb"


def DdsMipMapsDropdown() -> DropDownInput:
    return DropDownInput(
        input_type="DdsMipMaps",
        label="Generate Mip Maps",
        preferred_style="checkbox",
        options=[
            # these are not boolean values, see dds.py for more info
            {"option": "Yes", "value": 0},
            {"option": "No", "value": 1},
        ],
    )


def dds_inputs() -> list[BaseInput | NestedGroup]:
    """The DDS inputs of Save Image and Save DDS Texture (input 0 the image, 6 the
    format, 10 Generate Mip Maps)."""
    return [
        DdsFormatDropdown().with_id(6),
        if_enum_group(6, SUPPORTED_BC7_FORMATS)(
            EnumInput(
                BC7Compression,
                label="BC7 Compression",
                default=BC7Compression.DEFAULT,
            ).with_id(7),
        ),
        if_enum_group(6, SUPPORTED_BC123_FORMATS)(
            EnumInput(DDSErrorMetric, label="Error Metric").with_id(9),
            BoolInput("Dithering", default=False).with_id(8),
        ),
        DdsMipMapsDropdown()
        .with_id(10)
        .with_docs(
            "Whether [mipmaps](https://en.wikipedia.org/wiki/Mipmap) will be generated."
            " Mipmaps vastly improve the quality of the image when it is viewed at a smaller size, but they also increase the file size by 33%."
        ),
        if_enum_group(10, 0)(
            NumberInput("Mip Levels", default=0, min=0, max=16)
            .with_id(22)
            .with_docs(
                "How many levels to write, the full image included; 0: all of them, down to 1x1."
            ),
            EnumInput(
                DDSMipFilter,
                label="Mip Filter",
                default=DDSMipFilter.DEFAULT,
                option_labels={
                    DDSMipFilter.DEFAULT: "Default (Fant)",
                    DDSMipFilter.POINT: "Point",
                    DDSMipFilter.LINEAR: "Linear",
                    DDSMipFilter.CUBIC: "Cubic",
                    DDSMipFilter.FANT: "Fant",
                    DDSMipFilter.BOX: "Box",
                    DDSMipFilter.TRIANGLE: "Triangle",
                },
            )
            .with_id(23)
            .with_docs(
                "The filter texconv downscales the mips with. Default is texconv's own choice: Fant, except for sRGB formats, which it filters in linear light with filters of its own."
            ),
        ),
        if_group(
            Condition.enum(6, SUPPORTED_WITH_ALPHA)
            & Condition.enum(10, 0)
            & Condition.type(0, "Image { channels: 4 }")
        )(
            BoolInput("Separate Alpha for MipMaps", default=False)
            .with_id(13)
            .with_docs(
                "Enable this option when the alpha channel of an image is **not** transparency.",
                "The normal method for downscaling images with an alpha channel will remove the color information of transparent pixels (setting them to black). This is a problem if the alpha channel isn't transparency. E.g. games commonly store extra material information in the alpha channel of normal maps. Downscaling color and alpha separately fixes this problem.",
                "Note: Do not enable this option if the alpha channel is transparency. Otherwise, dark artifacts may appear around transparency edges.",
            ),
        ),
        EnumInput(
            DDSColourConversion,
            label="Colour Values",
            default=DDSColourConversion.KEEP,
            option_labels={
                DDSColourConversion.KEEP: "Keep",
                DDSColourConversion.SRGB_TO_LINEAR: "sRGB to linear",
                DDSColourConversion.LINEAR_TO_SRGB: "Linear to sRGB",
            },
        )
        .with_id(19)
        .with_docs(
            "An sRGB or a linear format is only a tag: the values are written as they are unless converted here (red, green and blue; alpha stays)."
        ),
        if_group(
            Condition.enum(6, SUPPORTED_WITH_ALPHA)
            & Condition.type(0, "Image { channels: 4 }")
        )(
            BoolInput("Premultiply Alpha", default=False)
            .with_id(20)
            .with_docs(
                "Writes premultiplied alpha (straight alpha otherwise). DXT3 and DXT5 are then written as DXT2 and DXT4."
            ),
        ),
        if_enum_group(6, SUPPORTED_GPU_FORMATS)(
            BoolInput("GPU Compression", default=True)
            .with_id(21)
            .with_docs(
                "Compresses BC6H and BC7 on the GPU (DirectCompute) when one is available, else on the CPU."
            ),
        ),
    ]


@io_group.register(
    schema_id="chainner:image:save",
    name="Save Image",
    description="Save image to file at a specified directory.",
    icon="MdSave",
    inputs=[
        ImageInput().make_lazy(),
        DirectoryInput(must_exist=False),
        RelativePathInput("Subdirectory Path")
        .make_optional()
        .with_docs(
            "An optional subdirectory path. Use this to save the image to a subdirectory of the specified directory. If the subdirectory does not exist, it will be created. Multiple subdirectories can be specified by separating them with a forward slash (`/`).",
            "Example: `foo/bar`",
        ),
        RelativePathInput("Image Name").with_docs(
            "The name of the image file **without** the file extension. If the file already exists, it will be overwritten.",
            "Example: `my-image`",
        ),
        EnumInput(
            ImageFormat,
            "Image Format",
            default=ImageFormat.PNG,
            option_labels=IMAGE_FORMAT_LABELS,
        ).with_id(4),
        if_enum_group(4, ImageFormat.PNG)(
            EnumInput(
                PngColorDepth,
                "Color Depth",
                default=PngColorDepth.U8,
                option_labels={
                    PngColorDepth.U8: "8 Bits/Channel",
                    PngColorDepth.U16: "16 Bits/Channel",
                },
            ).with_id(15),
        ),
        if_enum_group(4, ImageFormat.WEBP)(
            BoolInput("Lossless", default=False).with_id(14),
        ),
        if_group(
            Condition.enum(4, ImageFormat.JPG)
            | Condition.enum(4, ImageFormat.AVIF)
            | (Condition.enum(4, ImageFormat.WEBP) & Condition.enum(14, 0))
        )(
            SliderInput("Quality", min=0, max=100, default=95).with_id(5),
        ),
        if_enum_group(4, ImageFormat.JPG)(
            EnumInput(
                JpegSubsampling,
                label="Chroma Subsampling",
                default=JpegSubsampling.FACTOR_422,
                option_labels={
                    JpegSubsampling.FACTOR_444: "4:4:4 (Best Quality)",
                    JpegSubsampling.FACTOR_440: "4:4:0",
                    JpegSubsampling.FACTOR_422: "4:2:2",
                    JpegSubsampling.FACTOR_420: "4:2:0 (Best Compression)",
                },
            ).with_id(11),
            BoolInput("Progressive", default=False).with_id(12),
        ),
        if_enum_group(4, ImageFormat.TIFF)(
            EnumInput(
                TiffColorDepth,
                "Color Depth",
                default=TiffColorDepth.U8,
                option_labels={
                    TiffColorDepth.U8: "8 Bits/Channel",
                    TiffColorDepth.U16: "16 Bits/Channel",
                    TiffColorDepth.F32: "32 Bits/Channel (Float)",
                },
            ).with_id(16),
            if_enum_group(16, (TiffColorDepth.U8, TiffColorDepth.U16))(
                EnumInput(
                    TiffCompression,
                    "Compression",
                    default=TiffCompression.LZW,
                    option_labels={
                        TiffCompression.NONE: "None",
                        TiffCompression.LZW: "LZW (lossless)",
                        TiffCompression.ZIP: "ZIP (lossless)",
                    },
                ).with_id(18),
            ),
        ),
        if_enum_group(4, ImageFormat.DDS)(*dds_inputs()),
        if_enum_group(4, ImageFormat.AVIF)(
            EnumInput(
                AvifSubsampling,
                label="Chroma Subsampling",
                default=AvifSubsampling.FACTOR_420,
                option_labels={
                    AvifSubsampling.FACTOR_444: "4:4:4 (Best Quality)",
                    AvifSubsampling.FACTOR_422: "4:2:2",
                    AvifSubsampling.FACTOR_420: "4:2:0",
                    AvifSubsampling.FACTOR_400: "4:0:0 (Best Compression)",
                },
            ).with_id(17)
        ),
        BoolInput("Skip existing files", default=False)
        .with_id(1000)
        .with_docs(
            "If enabled, the node will not overwrite existing files.",
        ),
    ],
    outputs=[],
    key_info=KeyInfo.enum(4),
    side_effects=True,
    limited_to_8bpc="Image will be saved with 8 bits/channel by default. Some formats support higher bit depths.",
)
def save_image_node(
    lazy_image: Lazy[np.ndarray],
    base_directory: Path,
    relative_path: str | None,
    filename: str,
    image_format: ImageFormat,
    png_color_depth: PngColorDepth,
    webp_lossless: bool,
    quality: int,
    jpeg_chroma_subsampling: JpegSubsampling,
    jpeg_progressive: bool,
    tiff_color_depth: TiffColorDepth,
    tiff_compression: TiffCompression,
    dds_format: DDSFormat,
    dds_bc7_compression: BC7Compression,
    dds_error_metric: DDSErrorMetric,
    dds_dithering: bool,
    dds_mipmap_levels: int,
    dds_mip_levels: int,
    dds_mip_filter: DDSMipFilter,
    dds_separate_alpha: bool,
    dds_colour_conversion: DDSColourConversion,
    dds_premultiply_alpha: bool,
    dds_gpu: bool,
    avif_chroma_subsampling: AvifSubsampling,
    skip_existing_files: bool,
) -> None:
    return graph().image_io_save(
        globals(),
        (
            lazy_image,
            base_directory,
            relative_path,
            filename,
            image_format,
            png_color_depth,
            webp_lossless,
            quality,
            jpeg_chroma_subsampling,
            jpeg_progressive,
            tiff_color_depth,
            tiff_compression,
            dds_format,
            dds_bc7_compression,
            dds_error_metric,
            dds_dithering,
            dds_mipmap_levels,
            dds_mip_levels,
            dds_mip_filter,
            dds_separate_alpha,
            dds_colour_conversion,
            dds_premultiply_alpha,
            dds_gpu,
            avif_chroma_subsampling,
            skip_existing_files,
        ),
        None,
    )


def get_full_path(
    base_directory: Path,
    relative_path: str | None,
    filename: str,
    image_format: ImageFormat,
) -> Path:
    return graph().image_io_full_path(
        base_directory, relative_path, filename, image_format
    )


def prepare_save_image(inputs: list[object]) -> bytes | None:
    """The bytes Save Image would write for these inputs, input 0 being the image;
    None for DDS, which texconv writes itself (SP3-P9)."""
    return graph().image_io_save_prepare(globals(), tuple(inputs))


def commit_save_image(inputs: list[object], prepared: Prepared) -> None:
    """Save Image with its prepared bytes in place of the conversion and encoder."""
    return graph().image_io_save(globals(), tuple(inputs), prepared)


register_phases("chainner:image:save", prepare_save_image, commit_save_image)
