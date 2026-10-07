from __future__ import annotations

import numpy as np

from nodes.impl import native_channels
from nodes.properties.inputs import ImageInput, SliderInput
from nodes.properties.outputs import NumberOutput

from .. import utility_group


@utility_group.register(
    schema_id="chainner:image:get_bbox",
    name="Get Bounding Box",
    description="Gets a bounding box (X, Y, Height, and Width) of the white area of a mask.",
    icon="BsBoundingBox",
    inputs=[
        ImageInput(channels=1),
        SliderInput("Threshold", precision=1, min=0, max=100, step=1, default=0),
    ],
    outputs=[
        NumberOutput("X", output_type="min(uint, Input0.width - 1) & 0.."),
        NumberOutput("Y", output_type="min(uint, Input0.height - 1) & 0.."),
        NumberOutput("Width", output_type="min(uint, Input0.width) & 1.."),
        NumberOutput("Height", output_type="min(uint, Input0.height) & 1.."),
    ],
)
def get_bounding_box_node(
    img: np.ndarray,
    thresh_val: float,
) -> tuple[int, int, int, int]:
    # Threshold value 100 guarantees an empty image, so make sure the max
    # is just below that.
    return native_channels.bounding_box(img, thresh_val)
