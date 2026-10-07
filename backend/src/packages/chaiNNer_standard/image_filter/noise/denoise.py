from __future__ import annotations

import numpy as np

from nodes.groups import Condition, if_group
from nodes.impl.image_utils import to_uint8
from nodes.impl.native_denoise import denoise
from nodes.properties.inputs import ImageInput, NumberInput, SliderInput
from nodes.properties.outputs import ImageOutput

from .. import noise_group


@noise_group.register(
    schema_id="chainner:image:fast_nlmeans",
    name="Denoise",
    description="Use the fast Non-Local Means algorithm to denoise an image.",
    icon="CgEditNoise",
    inputs=[
        ImageInput("Image", channels=[1, 3, 4]),
        SliderInput(
            "Luminance strength",
            min=0,
            max=50,
            default=3.0,
            precision=1,
            step=0.1,
            slider_step=0.1,
        ),
        if_group(Condition.type(0, "Image { channels: 3 | 4 }"))(
            SliderInput(
                "Color strength",
                min=0.0,
                max=50.0,
                default=3.0,
                precision=1,
                step=0.1,
                slider_step=0.1,
            )
        ),
        NumberInput("Patch radius", min=1, default=3, max=30),
        NumberInput("Search radius", min=1, default=10, max=30),
    ],
    outputs=[ImageOutput(shape_as=0)],
    limited_to_8bpc=True,
)
def denoise_node(
    img: np.ndarray,
    h: float,
    h_color: float,
    patch_radius: int,
    search_radius: int,
) -> np.ndarray:
    image_array = to_uint8(img)
    return denoise(image_array, h, h_color, patch_radius, search_radius)
