from __future__ import annotations

from enum import Enum

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
    EnumInput,
    ImageInput,
    NumberInput,
)
from nodes.properties.outputs import ImageOutput
from nodes.utils.utils import get_h_w_c

from .. import crop_group


class CropIfSmaller(Enum):
    KEEP = 0
    PAD = 1


@crop_group.register(
    schema_id="chainner:image:crop_to_size",
    name="Crop to Size",
    description=[
        "Crop an image to a fixed size, whatever size it has. The anchor selects the part of the image that stays.",
        "When the anchor is centered and an odd number of pixels has to go, the right or bottom edge loses the extra pixel.",
    ],
    icon="MdCrop",
    inputs=[
        ImageInput(),
        NumberInput("Width", unit="px", min=1, default=100).with_id(1),
        NumberInput("Height", unit="px", min=1, default=100).with_id(2),
        AnchorInput().with_id(3),
        EnumInput(
            CropIfSmaller,
            label="If Smaller",
            default=CropIfSmaller.KEEP,
            label_style="inline",
        )
        .with_docs(
            "What happens to a side of the image that is already shorter than the requested size. *Keep* leaves it as it is, so the output can be smaller than the requested size. *Pad* adds a border up to the requested size, placed by the same anchor, so every output has the requested size."
        )
        .with_id(4),
        if_enum_group(4, CropIfSmaller.PAD)(
            BorderInput().with_id(5),
            if_enum_group(5, BorderType.CUSTOM_COLOR)(
                ColorInput().with_id(6),
            ),
        ),
    ],
    outputs=[
        ImageOutput(
            image_type="""
                let i = Input0;
                let width = Input1;
                let height = Input2;

                match Input4 {
                    CropIfSmaller::Keep => Image {
                        width: min(width, i.width),
                        height: min(height, i.height),
                        channels: i.channels,
                    },
                    CropIfSmaller::Pad => Image {
                        width,
                        height,
                        channels: if i.width < width or i.height < height {
                            BorderType::getOutputChannels(Input5, i.channels, Input6)
                        } else { i.channels },
                    },
                }
            """,
            assume_normalized=True,
        )
    ],
    key_info=KeyInfo.enum(3),
)
def crop_to_size_node(
    img: np.ndarray,
    width: int,
    height: int,
    anchor: Anchor,
    if_smaller: CropIfSmaller,
    border_type: BorderType,
    color: Color,
) -> np.ndarray:
    h, w, _ = get_h_w_c(img)

    left, top = anchor.offsets(max(0, w - width), max(0, h - height))
    img = img[top : top + height, left : left + width]

    if if_smaller is CropIfSmaller.KEEP:
        return img

    missing_width = max(0, width - w)
    missing_height = max(0, height - h)
    left, top = anchor.offsets(missing_width, missing_height)
    return native_layout.pad(
        img,
        border_type.value,
        color,
        top,
        missing_width - left,
        missing_height - top,
        left,
    )
