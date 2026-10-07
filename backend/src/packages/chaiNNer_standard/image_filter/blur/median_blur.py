from __future__ import annotations

import numpy as np

from nodes.impl.native_neighborhood import median
from nodes.properties.inputs import ImageInput, SliderInput
from nodes.properties.outputs import ImageOutput

from .. import blur_group


@blur_group.register(
    schema_id="chainner:image:median_blur",
    name="Median Blur",
    description="Apply median blur to an image.",
    icon="MdBlurOn",
    inputs=[
        ImageInput(),
        SliderInput("Radius", min=0, max=1000, default=1, scale="log"),
    ],
    outputs=[ImageOutput(shape_as=0)],
    limited_to_8bpc=True,
)
def median_blur_node(
    img: np.ndarray,
    radius: int,
) -> np.ndarray:
    if radius == 0:
        return img
    return median(img, radius)
