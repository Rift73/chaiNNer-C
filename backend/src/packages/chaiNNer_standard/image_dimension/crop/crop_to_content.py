from __future__ import annotations

import numpy as np

from nodes.impl import native_channels
from nodes.properties.inputs import ImageInput, SliderInput
from nodes.properties.outputs import ImageOutput

from .. import crop_group


@crop_group.register(
    schema_id="chainner:image:crop_content",
    name="Crop to Content",
    description=(
        "Crop an image to the boundaries of the visible image content, "
        "removing borders at or below the given opacity threshold."
    ),
    icon="MdCrop",
    inputs=[
        ImageInput(),
        SliderInput("Threshold", precision=1, step=1, default=0),
    ],
    outputs=[
        ImageOutput(
            image_type="""
                match Input0.channels {
                    ..3 => Input0,
                    _ => Image {
                        width: min(uint, Input0.width) & 1..,
                        height: min(uint, Input0.height) & 1..,
                        channels: Input0.channels
                    }
                }
                """,
            assume_normalized=True,
        )
    ],
)
def crop_to_content_node(img: np.ndarray, thresh_val: float) -> np.ndarray:
    return native_channels.crop_to_content(img, thresh_val)
