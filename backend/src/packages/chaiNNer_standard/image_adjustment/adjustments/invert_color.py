from __future__ import annotations

import numpy as np

from nodes.impl.native_adjustments import adjust
from nodes.properties.inputs import ImageInput
from nodes.properties.outputs import ImageOutput
from nodes.utils.utils import get_h_w_c

from .. import adjustments_group


@adjustments_group.register(
    schema_id="chainner:image:invert",
    name="Invert Color",
    description="Inverts all colors in an image.",
    icon="MdInvertColors",
    inputs=[ImageInput()],
    outputs=[ImageOutput(shape_as=0, assume_normalized=True)],
)
def invert_color_node(img: np.ndarray) -> np.ndarray:
    # Upstream returns 1 - img (np.clip's order K) up to three channels, and inverts
    # a copy (C order) beyond.
    return adjust(img, 2, c_order=get_h_w_c(img)[2] > 3)
