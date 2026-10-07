from __future__ import annotations

import numpy as np

from nodes.groups import linked_inputs_group
from nodes.impl import native_layout
from nodes.properties.inputs import ImageInput, SliderInput
from nodes.properties.outputs import ImageOutput

from .. import miscellaneous_group


@miscellaneous_group.register(
    schema_id="chainner:image:pixelate",
    name="Pixelate",
    description="Pixelate an image.",
    icon="MdOutlineAutoFixHigh",
    inputs=[
        ImageInput(),
        linked_inputs_group(
            SliderInput("Size X", min=1, max=1024, default=10, scale="log"),
            SliderInput("Size Y", min=1, max=1024, default=10, scale="log"),
        ),
    ],
    outputs=[ImageOutput(shape_as=0)],
)
def pixelate_node(
    img: np.ndarray,
    size_x: int,
    size_y: int,
) -> np.ndarray:
    return native_layout.pixelate(img, size_x, size_y)
