from __future__ import annotations

import numpy as np

from nodes.groups import icon_set_group
from nodes.impl.native_adjustments import levels
from nodes.properties.inputs import BoolInput, ImageInput, SliderInput
from nodes.properties.outputs import ImageOutput

from .. import adjustments_group


@adjustments_group.register(
    schema_id="chainner:image:color_levels",
    name="Color Levels",
    description="Color Levels can be used to make an image lighter or darker, to change contrast or to correct a predominant color cast.",
    icon="MdOutlineColorLens",
    inputs=[
        ImageInput(channels=[1, 3, 4]),
        icon_set_group("Channels")(
            BoolInput("Red", default=True),
            BoolInput("Green", default=True),
            BoolInput("Blue", default=True),
            BoolInput("Alpha", default=False),
        ),
        SliderInput(
            "In Black",
            min=0,
            max=1,
            default=0,
            precision=3,
            step=0.01,
        ),
        SliderInput(
            "In White",
            min=0,
            max=1,
            default=1,
            precision=3,
            step=0.01,
        ),
        SliderInput(
            "Gamma",
            min=0,
            max=10,
            default=1,
            precision=3,
            step=0.01,
            scale="log",
        ),
        SliderInput(
            "Out Black",
            min=0,
            max=1,
            default=0,
            precision=3,
            step=0.01,
        ),
        SliderInput(
            "Out White",
            min=0,
            max=1,
            default=1,
            precision=3,
            step=0.01,
        ),
    ],
    outputs=[
        ImageOutput(shape_as=0),
    ],
)
def color_levels_node(
    img: np.ndarray,
    red: bool,
    green: bool,
    blue: bool,
    alpha: bool,
    in_black: float,
    in_white: float,
    in_gamma: float,
    out_black: float,
    out_white: float,
) -> np.ndarray:
    mask = int(blue) | (int(green) << 1) | (int(red) << 2) | (int(alpha) << 3)
    return levels(img, mask, in_black, in_white, in_gamma, out_black, out_white)
