from __future__ import annotations

import numpy as np

from nodes.impl import native_channels
from nodes.impl.color.color import Color
from nodes.properties.inputs import ImageInput
from nodes.properties.outputs import ImageOutput

from .. import transparency_group


@transparency_group.register(
    schema_id="chainner:image:merge_transparency",
    name="Merge Transparency",
    description="Merge RGB and Alpha (transparency) image channels into 4-channel RGBA channels.",
    icon="MdCallMerge",
    inputs=[
        ImageInput("RGB", allow_colors=True),
        ImageInput("Alpha", allow_colors=True, channels=1),
    ],
    outputs=[
        ImageOutput(
            image_type="""
                def isImage(i: any) = match i { Image => true, _ => false };
                let anyImages = isImage(Input0) or isImage(Input1);

                if not anyImages {
                    error("At least one input must be an image.")
                } else {
                    def getWidth(i: any) = match i { Image => i.width, _ => Image.width };
                    def getHeight(i: any) = match i { Image => i.height, _ => Image.height };

                    Image {
                        width: getWidth(Input0) & getWidth(Input1),
                        height: getHeight(Input0) & getHeight(Input1),
                    }
                }
            """,
            channels=4,
            assume_normalized=True,
        ).with_never_reason("RGB and Alpha must have the same size.")
    ],
)
def merge_transparency_node(
    rgb: np.ndarray | Color,
    a: np.ndarray | Color,
) -> np.ndarray:
    return native_channels.combine_rgb_alpha(rgb, a)
