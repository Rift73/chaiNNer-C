from __future__ import annotations

import numpy as np

from nodes.impl.color.color import Color
from nodes.impl.image_utils import as_target_channels
from nodes.impl.native_generation import image_fill
from nodes.properties.inputs import ColorInput, NumberInput
from nodes.properties.outputs import ImageOutput

from .. import create_images_group


@create_images_group.register(
    schema_id="chainner:image:create_checkerboard",
    name="Create Checkerboard",
    description="Create a Checkerboard of specified dimensions filled with the given colors for the squares.",
    icon="MdBorderAll",
    inputs=[
        NumberInput("Width", min=1, unit="px", default=1024),
        NumberInput("Height", min=1, unit="px", default=1024),
        ColorInput(
            "Color 1", channels=[1, 3, 4], default=Color.bgr((0.75, 0.75, 0.75))
        ).with_id(2),
        ColorInput(
            "Color 2", channels=[1, 3, 4], default=Color.bgr((0.35, 0.35, 0.35))
        ).with_id(3),
        NumberInput("Square Size", min=1, default=32),
    ],
    outputs=[
        ImageOutput(
            image_type="""
                Image {
                    width: Input0,
                    height: Input1,
                    channels: max(Input2.channels, Input3.channels),
                }""",
        )
    ],
)
def create_checkerboard_node(
    width: int,
    height: int,
    color_1: Color,
    color_2: Color,
    square_size: int,
) -> np.ndarray:
    max_channels = max(color_1.channels, color_2.channels)

    color_a = Color.from_1x1_image(
        as_target_channels(color_1.to_1x1_image(), max_channels)
    )
    color_b = Color.from_1x1_image(
        as_target_channels(color_2.to_1x1_image(), max_channels)
    )

    return image_fill(
        width,
        height,
        color_a.value,
        second=color_b.value,
        square=square_size,
        keep_channel=True,
    )
