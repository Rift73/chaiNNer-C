from __future__ import annotations

import numpy as np

import navi
from nodes.impl.native_analysis import apply_palette
from nodes.properties.inputs import ImageInput
from nodes.properties.outputs import ImageOutput

from .. import miscellaneous_group


@miscellaneous_group.register(
    schema_id="chainner:image:lut",
    name="Apply Palette",
    description=(
        "Apply a color palette to a grayscale image."
        " Only the top row of pixels (y=0) of the palette will be used to do the look up."
    ),
    see_also="chainner:image:palette_from_image",
    icon="MdGradient",
    inputs=[
        ImageInput(channels=1),
        ImageInput("Palette"),
    ],
    outputs=[
        ImageOutput(image_type=navi.Image(size_as="Input0", channels_as="Input1"))
    ],
)
def apply_palette_node(
    img: np.ndarray,
    lut: np.ndarray,
) -> np.ndarray:
    return apply_palette(img, lut)
