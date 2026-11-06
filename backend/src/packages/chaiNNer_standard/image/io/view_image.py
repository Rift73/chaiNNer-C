from __future__ import annotations

import numpy as np

from nodes.properties.inputs import ImageInput
from nodes.properties.outputs import LargeImageOutput
from nodes.utils.utils import get_h_w_c

from .. import io_group


@io_group.register(
    schema_id="chainner:image:view",
    name="View Image",
    description="See an inline preview of the image in the editor.",
    icon="BsEyeFill",
    inputs=[ImageInput()],
    outputs=[
        LargeImageOutput(
            "Preview",
            image_type="Input0",
            has_handle=False,
            assume_normalized=True,
        ),
    ],
    side_effects=True,
)
def view_image_node(img: np.ndarray):
    c = get_h_w_c(img)[2]
    if c not in (1, 3, 4):
        raise ValueError(f"View Image shows images of 1, 3 or 4 channels, not {c}.")
    return img
