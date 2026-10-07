from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from api import KeyInfo
from nodes.groups import if_enum_group
from nodes.impl.image_utils import as_2d_grayscale
from nodes.impl.native_color_ops import threshold as native_threshold
from nodes.impl.native_threshold import binary_threshold_aa
from nodes.properties.inputs import BoolInput, EnumInput, ImageInput, SliderInput
from nodes.properties.outputs import ImageOutput
from nodes.utils.utils import get_h_w_c

from .. import threshold_group


class ThresholdType(Enum):
    BINARY = cv2.THRESH_BINARY
    BINARY_INV = cv2.THRESH_BINARY_INV
    TRUNC = cv2.THRESH_TRUNC
    TO_ZERO = cv2.THRESH_TOZERO
    TO_ZERO_INV = cv2.THRESH_TOZERO_INV


_THRESHOLD_TYPE_LABELS: dict[ThresholdType, str] = {
    ThresholdType.BINARY: "Binary",
    ThresholdType.BINARY_INV: "Binary (Inverted)",
    ThresholdType.TRUNC: "Truncated",
    ThresholdType.TO_ZERO: "To Zero",
    ThresholdType.TO_ZERO_INV: "To Zero (Inverted)",
}


@threshold_group.register(
    schema_id="chainner:image:threshold",
    name="Threshold",
    description="Replaces pixels based on the threshold value. If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to the maximum value.",
    icon="MdShowChart",
    inputs=[
        ImageInput(),
        SliderInput("Threshold", max=100, default=50, precision=1, step=1),
        EnumInput(
            ThresholdType,
            "Threshold Type",
            default=ThresholdType.BINARY,
            option_labels=_THRESHOLD_TYPE_LABELS,
        ).with_id(3),
        if_enum_group(3, (ThresholdType.BINARY, ThresholdType.BINARY_INV))(
            SliderInput(
                "Maximum Value",
                max=100,
                default=100,
                precision=1,
                step=1,
            ).with_id(2),
        ),
        BoolInput("Anti-aliasing", default=False)
        .with_docs(
            "Enables sub-pixel precision. Bilinear interpolation is used to fill in values in between pixels.",
            "Conceptually, the option is equivalent to first upscaling the image by a factor of X (with linear interpolation), thresholding it, and then downscaling it by a factor of X (where X is 20 or more).",
        )
        .with_id(4),
        if_enum_group(4, 1)(
            SliderInput("Softness", default=0, min=0, max=10)
            .with_docs(
                "The strength of a sub-pixel blur applied to be anti-aliased image. This can be be used to make the anti-aliasing even softer.",
                "The blur is very small and higher-quality than a simple Gaussian blur. 0 means that no additional blur will be applied. 10 means that the anti-aliasing will be very soft.",
            )
            .with_id(5),
        ),
    ],
    outputs=[
        ImageOutput(shape_as=0),
    ],
    key_info=KeyInfo.number(1),
    see_also=[
        "chainner:image:generate_threshold",
        "chainner:image:threshold_adaptive",
    ],
)
def threshold_node(
    img: np.ndarray,
    threshold: float,
    thresh_type: ThresholdType,
    max_value: float,
    anti_aliasing: bool,
    extra_smoothness: int,
) -> np.ndarray:
    threshold /= 100
    max_value /= 100
    smoothness = extra_smoothness / 10

    if not anti_aliasing:
        return native_threshold(img, threshold, max_value, thresh_type.value)

    binary = binary_threshold_aa(img, threshold, smoothness)
    if get_h_w_c(binary)[2] == 1:
        binary = as_2d_grayscale(binary)

    return native_threshold(img, threshold, max_value, thresh_type.value, binary)
