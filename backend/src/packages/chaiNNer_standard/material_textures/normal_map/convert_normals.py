from __future__ import annotations

import numpy as np

import navi
from nodes.impl import native_geometry
from nodes.impl.image_utils import NormalMapType
from nodes.properties.inputs import EnumInput, ImageInput
from nodes.properties.outputs import ImageOutput

from .. import normal_map_group


@normal_map_group.register(
    schema_id="chainner:image:convert_normal_map",
    name="Convert Normals",
    description=[
        """Convert between different normal map formats. Only the R and G
            channels of the input image will be used. For DirectX and OpenGL, the output normal map
            is guaranteed to be normalized.""",
        "Also supports converting to and from the Octahedral format used by RTX Remix.",
    ],
    icon="BsBoxArrowUpRight",
    inputs=[
        ImageInput("Normal Map", channels=[3, 4]),
        EnumInput(
            NormalMapType,
            label="From",
            label_style="inline",
            default=NormalMapType.DIRECTX,
            option_labels={
                NormalMapType.DIRECTX: "DirectX",
                NormalMapType.OPENGL: "OpenGL",
            },
        ),
        EnumInput(
            NormalMapType,
            label="To",
            label_style="inline",
            default=NormalMapType.OPENGL,
            option_labels={
                NormalMapType.DIRECTX: "DirectX",
                NormalMapType.OPENGL: "OpenGL",
                NormalMapType.OCTAHEDRAL: "Octahedral (RTX Remix)",
            },
        ),
    ],
    outputs=[
        ImageOutput(
            "Normal Map",
            image_type=navi.Image(
                width="Input0.width",
                height="Input0.height",
            ),
            channels=3,
        ),
    ],
)
def convert_normals_node(
    img: np.ndarray,
    from_type: NormalMapType,
    to_type: NormalMapType,
) -> np.ndarray:
    formats = (
        NormalMapType.DIRECTX,
        NormalMapType.OPENGL,
        NormalMapType.OCTAHEDRAL,
    )
    return native_geometry.normal_map(
        img, 2, formats.index(from_type), formats.index(to_type)
    )
