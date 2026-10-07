from __future__ import annotations

import numpy as np

from nodes.impl.image_utils import to_uint8
from nodes.impl.native_filters import binary_sdf as native_binary_sdf
from nodes.impl.native_filters import subpixel_sdf
from nodes.properties.inputs import BoolInput, ImageInput, NumberInput
from nodes.properties.outputs import ImageOutput

from .. import miscellaneous_group


def binary_sdf(img: np.ndarray, spread: float) -> np.ndarray:
    return native_binary_sdf(to_uint8(img, normalized=True), spread)


@miscellaneous_group.register(
    schema_id="chainner:image:distance_transform",
    name="Distance Transform",
    description="Perform a distance transform on a monochrome bitmap image, producing a signed distance field.",
    icon="MdBlurOff",
    inputs=[
        ImageInput(channels=1),
        NumberInput("Spread", min=1, default=4),
        BoolInput("Sub-pixel precision", default=False).with_docs(
            "If enabled, then anti-aliasing will be accounted for. If not enabled, then the image will be converted to binary (either black or white) before processing.",
            "Enabling this option will significantly improve the results of anti-aliased shapes, but it cannot be used on anything else. It assumes strictly binary shapes (with optional anti-aliasing), and will return incorrect results for e.g. blurry images. If you cannot guarantee binary image, use the `chainner:image:threshold` node with *Anti-aliasing* enabled.",
            "Sub-pixel distance transform is implemented using the excellent [ESDF algorithm](https://acko.net/blog/subpixel-distance-transform/) by Steven Wittens.",
        ),
    ],
    outputs=[ImageOutput(shape_as=0)],
)
def distance_transform_node(
    img: np.ndarray,
    spread: int,
    use_esdf: bool,
) -> np.ndarray:
    if use_esdf:
        return subpixel_sdf(img, spread * 2)
    return binary_sdf(img, spread)
