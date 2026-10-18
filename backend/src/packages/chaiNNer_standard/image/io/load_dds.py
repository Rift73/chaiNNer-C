from __future__ import annotations

from pathlib import Path

import numpy as np

from nodes.impl.dds.texconv import read_dds_info, read_dds_texconv
from nodes.impl.image_utils import normalize
from nodes.properties.inputs import FileInput, NumberInput
from nodes.properties.outputs import (
    BoolOutput,
    LargeImageOutput,
    NumberOutput,
    TextOutput,
)

from .. import io_group
from .load_image import remove_unnecessary_alpha


def dds_image(image: np.ndarray) -> np.ndarray:
    """A DDS read as the Load DDS nodes output it: float32 with values outside 0-1 kept
    (an 8-bit read divided by 255; 16-bit and float reads are float32), without an
    unused alpha."""
    return remove_unnecessary_alpha(
        normalize(image) if image.dtype == np.uint8 else image
    )


@io_group.register(
    schema_id="chainner:image:load_dds",
    name="Load DDS",
    description=(
        "Load one image of a DDS texture: any mip level, array layer or cubemap face,"
        " and volume slice, with the texture's format, size, mip levels, layers, depth"
        " and whether it is a cubemap. 16-bit and float formats load at their depth;"
        " values outside 0-1 are kept."
    ),
    icon="BsFillImageFill",
    inputs=[
        FileInput(
            label="DDS File", file_kind="image", filetypes=[".dds"], primary_input=True
        ),
        NumberInput("Mip Level", default=0, min=0).with_docs(
            "0 is the full-size image, each next level half its size."
        ),
        NumberInput("Layer", default=0, min=0).with_docs(
            "An array element. A cubemap has 6 layers per cube, its faces in the order"
            " +X, -X, +Y, -Y, +Z, -Z."
        ),
        NumberInput("Slice", default=0, min=0).with_docs(
            "A volume texture's depth slice at that mip level."
        ),
    ],
    outputs=[
        LargeImageOutput(assume_normalized=True),
        TextOutput("Format").with_docs(
            "The DXGI format, the FourCC of a legacy one (DXT1-DXT5, ATI1, ATI2), or for"
            ' a legacy format DXGI lacks its channels and bits, such as "R8G8B8 (legacy)".'
        ),
        NumberOutput("Width", output_type="uint"),
        NumberOutput("Height", output_type="uint"),
        NumberOutput("Mip Levels", output_type="uint"),
        NumberOutput("Layers", output_type="uint").with_docs(
            "Array elements, 6 per cube for a cubemap."
        ),
        NumberOutput("Depth", output_type="uint").with_docs(
            "A volume texture's depth slices at full size; 1 otherwise."
        ),
        BoolOutput("Cubemap"),
    ],
    side_effects=True,
)
def load_dds_node(
    path: Path, mip_level: int, layer: int, depth_slice: int
) -> tuple[np.ndarray, str, int, int, int, int, int, bool]:
    info = read_dds_info(path)
    image = read_dds_texconv(path, mip_level, layer, depth_slice)
    return (
        dds_image(image),
        info.format,
        info.width,
        info.height,
        info.mip_levels,
        info.layers,
        info.depth,
        info.cubemap,
    )
