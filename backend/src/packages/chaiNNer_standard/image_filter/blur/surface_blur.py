from __future__ import annotations

import numpy as np

from nodes.impl.native_bilateral import surface
from nodes.properties.inputs import ImageInput, SliderInput
from nodes.properties.outputs import ImageOutput

from .. import blur_group


@blur_group.register(
    schema_id="chainner:image:bilateral_blur",
    name="Surface Blur",
    description="Apply surface/bilateral blur to an image.",
    icon="MdBlurOn",
    inputs=[
        ImageInput(),
        SliderInput("Radius", min=0, max=100, default=4, scale="sqrt"),
        SliderInput("Color Sigma", default=25, scale="log", min=0, max=1000),
        SliderInput("Space Sigma", default=25, scale="log", min=0, max=1000),
    ],
    outputs=[ImageOutput(shape_as=0)],
)
def surface_blur_node(
    img: np.ndarray,
    radius: int,
    sigma_color: int,
    sigma_space: int,
) -> np.ndarray:
    if radius == 0 or sigma_color == 0 or sigma_space == 0:
        return img

    return surface(img, radius, sigma_color, sigma_space)
