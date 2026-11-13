from __future__ import annotations

import numpy as np

from nodes.impl.image_utils import NormalMapType
from nodes.impl.normals.height import integrate_slopes
from nodes.impl.normals.util import gr_to_xyz, octahedral_gr_to_xyz
from nodes.properties.inputs import BoolInput, EnumInput, ImageInput
from nodes.properties.outputs import ImageOutput, NumberOutput

from .. import normal_map_group

# A normal lying flat in the image plane has no finite slope. This Z is a slope of 10.
MIN_Z = 0.1
# R and G as (x, y) reach 1 at most, and rounding adds up to 0.006 at 8 bits per channel and
# 0.036 at the 5 and 6 bits of BC1. A longer (x, y) is no normal, like UV padding.
MAX_XY_LENGTH = 1.04
# Every octahedral code decodes to a normal. Those on the edge of the code square, black among
# them, have no Z; the next 8-bit code has a Z of 0.008.
OCTAHEDRAL_EDGE_Z = 1e-3
# A height range below this many pixels is the rounding of a flat normal map.
MIN_HEIGHT_RANGE = 1e-3


@normal_map_group.register(
    schema_id="chainner:image:normal_to_height",
    name="Normal to Height",
    description=[
        "Generate a height map from a normal map: the surface whose slopes fit the normals best. Only the R and G channels of the input image will be used.",
        "The image is stretched to the full range of 0 to 1, so each image shows the shape of its surface at its own scale. *Height Range* gives the height from the lowest to the highest point in pixels, the scale that the stretch removes: use it to give the tiles of one material a common scale, or as a displacement strength. A flat normal map gives middle gray and a range of 0. An overall tilt of the surface is not part of the output: a normal map stores it mostly as a small bias of its 8-bit values, which would drown the details.",
        "Pixels that are no normals, such as the black padding of game textures, count as flat. Slopes are limited to 10 (about 84 degrees).",
    ],
    icon="MdTerrain",
    inputs=[
        ImageInput("Normal Map", channels=[3, 4]),
        EnumInput(
            NormalMapType,
            label="Format",
            label_style="inline",
            default=NormalMapType.DIRECTX,
            option_labels={
                NormalMapType.DIRECTX: "DirectX",
                NormalMapType.OPENGL: "OpenGL",
                NormalMapType.OCTAHEDRAL: "Octahedral (RTX Remix)",
            },
        )
        .with_docs(
            "The format of the normal map. DirectX and OpenGL differ in the direction of G. With the wrong one, the height map has no clear shape and looks smeared along diagonals.",
            "Normal Map Generator gives DirectX unless its *Invert G* is on.",
        )
        .with_id(1),
        BoolInput("Tileable", default=False)
        .with_docs(
            "If enabled, the normal map will be treated as tileable and a tileable height map will be created.",
            hint=True,
        )
        .with_id(2),
    ],
    outputs=[
        ImageOutput("Height Map", size_as=0, channels=1),
        NumberOutput("Height Range", output_type="0.."),
    ],
)
def normal_to_height_node(
    img: np.ndarray,
    normal_format: NormalMapType,
    tileable: bool,
) -> tuple[np.ndarray, float]:
    if normal_format is NormalMapType.OCTAHEDRAL:
        x, y, z = octahedral_gr_to_xyz(img)
        no_normal = z < OCTAHEDRAL_EDGE_Z
    else:
        x, y, z = gr_to_xyz(img)
        # The decoder shortens a longer (x, y) to length 1, so this reads the channels.
        # In place: each fresh full-size array costs more than the arithmetic on it.
        r, g = img[:, :, 2] * 2, img[:, :, 1] * 2
        for channel in (r, g):
            np.subtract(channel, 1, out=channel)
            np.square(channel, out=channel)
        r += g
        no_normal = r > MAX_XY_LENGTH**2

    # A normal leans away from the rising side. Rows grow downwards, and G points down in
    # DirectX and up in OpenGL.
    z = np.maximum(z, MIN_Z)
    slope_x = -x / z
    slope_y = y / z if normal_format is NormalMapType.OPENGL else -y / z

    # The tilt is the normals' alone: with the padding in its mean, the bias of a flat 8-bit
    # map would stay on the normals as a ramp.
    slope_x[no_normal] = 0
    slope_y[no_normal] = 0
    normals = ~no_normal
    if np.any(normals):
        for slope in (slope_x, slope_y):
            mean = float(np.mean(slope[normals], dtype=np.float64))
            np.subtract(slope, mean, out=slope, where=normals)

    height = integrate_slopes(slope_x, slope_y, tileable)

    low = np.min(height)
    height_range = np.max(height) - low
    if height_range < MIN_HEIGHT_RANGE:
        return np.full(height.shape, 0.5, dtype=np.float32), 0.0
    return (height - low) / height_range, round(float(height_range), 4)
