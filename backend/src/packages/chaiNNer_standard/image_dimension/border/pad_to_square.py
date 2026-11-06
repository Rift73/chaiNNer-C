from __future__ import annotations

import numpy as np

from api import KeyInfo
from nodes.groups import if_enum_group
from nodes.impl import native_layout
from nodes.impl.color.color import Color
from nodes.impl.image_utils import BorderType
from nodes.properties.inputs import (
    Anchor,
    AnchorInput,
    BorderInput,
    ColorInput,
    ImageInput,
)
from nodes.properties.outputs import ImageOutput
from nodes.utils.utils import get_h_w_c

from .. import padding_group


@padding_group.register(
    schema_id="chainner:image:pad_to_square",
    name="Pad to Square",
    description=[
        "Adds padding to the shorter side of an image, so the image becomes a square with the size of its longer side. The content keeps its aspect ratio.",
        "The anchor places the image inside the square. When it is centered and the padding is an odd number of pixels, the right or bottom edge gets the extra pixel. An image that is already a square stays as it is.",
    ],
    icon="BsBorderOuter",
    inputs=[
        ImageInput(),
        AnchorInput().with_id(1),
        BorderInput().with_id(2),
        if_enum_group(2, BorderType.CUSTOM_COLOR)(
            ColorInput().with_id(3),
        ),
    ],
    outputs=[
        ImageOutput(
            image_type="""
                let i = Input0;
                let side = max(i.width, i.height);

                Image {
                    width: side,
                    height: side,
                    channels: if i.width == i.height {
                        i.channels
                    } else {
                        BorderType::getOutputChannels(Input2, i.channels, Input3)
                    },
                }
            """,
            assume_normalized=True,
        )
    ],
    key_info=KeyInfo.enum(1),
)
def pad_to_square_node(
    img: np.ndarray,
    anchor: Anchor,
    border_type: BorderType,
    color: Color,
) -> np.ndarray:
    h, w, _ = get_h_w_c(img)
    side = max(h, w)
    left, top = anchor.offsets(side - w, side - h)
    return native_layout.pad(
        img,
        border_type.value,
        color,
        top,
        side - w - left,
        side - h - top,
        left,
    )
