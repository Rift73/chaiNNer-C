from __future__ import annotations

import numpy as np

from nodes.impl.native_adjustments import adjust
from nodes.properties.inputs import ImageInput, SliderInput
from nodes.properties.outputs import ImageOutput

from .. import adjustments_group


@adjustments_group.register(
    schema_id="chainner:image:brightness_and_contrast",
    description="Adjust the brightness and contrast of an image.",
    name="Brightness & Contrast",
    icon="ImBrightnessContrast",
    inputs=[
        ImageInput(),
        SliderInput(
            "Brightness",
            min=-100,
            max=100,
            default=0,
            precision=1,
            step=1,
        ),
        SliderInput(
            "Contrast",
            min=-100,
            max=100,
            default=0,
            precision=1,
            step=1,
        ),
    ],
    outputs=[
        ImageOutput(shape_as=0, assume_normalized=True),
    ],
)
def brightness_and_contrast_node(
    img: np.ndarray, brightness: float, contrast: float
) -> np.ndarray:
    brightness /= 100
    contrast /= 100

    if brightness == 0 and contrast == 0:
        return img

    # Contrast correction factor
    max_c = 259 / 255
    factor: float = (max_c * (contrast + 1)) / (max_c - contrast)
    add: float = factor * brightness + 0.5 * (1 - factor)

    return adjust(img, 3, factor, add, clip=add < 0 or factor + add > 1)
