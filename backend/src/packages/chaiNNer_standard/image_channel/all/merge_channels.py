from __future__ import annotations

import numpy as np

import navi
from nodes.impl import native_channels
from nodes.properties.inputs import ImageInput
from nodes.properties.outputs import ImageOutput

from .. import all_group


@all_group.register(
    schema_id="chainner:image:merge_channels",
    name="Merge Channels",
    description=(
        "Merge image channels together into a ≤4 channel image. "
        "Typically used for combining an image with an alpha layer."
    ),
    icon="MdCallMerge",
    inputs=[
        ImageInput("Channel(s) A"),
        ImageInput("Channel(s) B").make_optional(),
        ImageInput("Channel(s) C").make_optional(),
        ImageInput("Channel(s) D").make_optional(),
    ],
    outputs=[
        ImageOutput(
            size_as=0,
            image_type=navi.Image(
                channels="""
                    match (
                        Input0.channels
                        + match Input1 { Image as i => i.channels, _ => 0 }
                        + match Input2 { Image as i => i.channels, _ => 0 }
                        + match Input3 { Image as i => i.channels, _ => 0 }
                    ) {
                        1 => 1,
                        2 | 3 => 3,
                        int(4..) => 4
                    }
                    """,
            ),
        )
    ],
    deprecated=True,
)
def merge_channels_node(
    im1: np.ndarray,
    im2: np.ndarray | None,
    im3: np.ndarray | None,
    im4: np.ndarray | None,
) -> np.ndarray:
    return native_channels.merge_channels(im1, im2, im3, im4)
