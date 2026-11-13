from __future__ import annotations

from enum import Enum

import numpy as np

from api import KeyInfo
from nodes.groups import if_enum_group
from nodes.properties.inputs import EnumInput, ImageInput, SliderInput
from nodes.properties.outputs import ImageOutput, NumberOutput
from nodes.utils.utils import get_h_w_c

from .. import miscellaneous_group


class DiffMode(Enum):
    ABSOLUTE = 0
    SIGNED = 1
    MASK = 2


@miscellaneous_group.register(
    schema_id="chainner:image:image_diff",
    name="Image Diff",
    description=[
        "Shows where two images of the same size and channel count differ, and by how much.",
        "The numbers are in 8-bit levels (0-255) and count every channel, alpha included: *Max Difference* is the largest difference of any sample, *Mean Difference* the mean of all samples, and *Different Pixels* the number of pixels with a sample that differs by more than *Tolerance*.",
    ],
    icon="MdCompare",
    inputs=[
        ImageInput("Image A", channels=[1, 3, 4]),
        ImageInput("Image B", channels=[1, 3, 4]),
        EnumInput(
            DiffMode,
            label="Mode",
            default=DiffMode.ABSOLUTE,
            preferred_style="tabs",
        )
        .with_docs(
            "- Absolute: the size of each channel's difference, black where the images are equal. With alpha, the output has RGB channels and a difference of alpha shows in all three.",
            "- Signed: A minus B around middle gray, brighter where A is brighter. Alpha is not shown.",
            "- Mask: a grayscale image, white where a pixel differs by more than *Tolerance*.",
        )
        .with_id(2),
        if_enum_group(2, (DiffMode.ABSOLUTE, DiffMode.SIGNED))(
            SliderInput(
                "Amplify",
                min=1,
                max=1000,
                default=1,
                precision=1,
                scale="log",
                has_handle=False,
            )
            .with_docs("A factor that makes small differences visible.", hint=True)
            .with_id(3),
        ),
        SliderInput(
            "Tolerance",
            min=0,
            max=255,
            default=0,
            precision=2,
            step=1,
            scale="log-offset",
            has_handle=False,
        )
        .with_docs(
            "The largest difference, in 8-bit levels, that still counts as equal. At 0, any difference counts.",
            hint=True,
        )
        .with_id(4),
    ],
    outputs=[
        ImageOutput(
            "Difference",
            image_type="""
                let a = Input0;
                let b = Input1;
                let channels = a.channels & b.channels;

                Image {
                    width: a.width & b.width,
                    height: a.height & b.height,
                    channels: match Input2 {
                        DiffMode::Mask => min(channels, 1),
                        _ => min(channels, 3),
                    },
                }
            """,
        ).with_never_reason(
            "The given images have different sizes or channel counts but must have the same."
        ),
        NumberOutput("Max Difference", output_type="0.."),
        NumberOutput("Mean Difference", output_type="0.."),
        NumberOutput("Different Pixels", output_type="uint"),
    ],
    key_info=KeyInfo.enum(2),
)
def image_diff_node(
    a: np.ndarray,
    b: np.ndarray,
    mode: DiffMode,
    amplify: float,
    tolerance: float,
) -> tuple[np.ndarray, float, float, int]:
    assert a.shape == b.shape, "Images must have same dimensions and color depth"

    # Each pass writes into an array of its own that nothing reads again, where it can:
    # a fresh full-size array costs about as much as the arithmetic on it.
    signed = a - b
    if mode is DiffMode.SIGNED:
        difference = np.abs(signed)
    else:
        difference = np.abs(signed, out=signed)
    largest = difference
    if difference.ndim == 3:
        # Plane by plane: a reduce over the last axis runs NumPy's loop once per pixel.
        largest = np.maximum(difference[:, :, 0], difference[:, :, 1])
        for channel in range(2, difference.shape[2]):
            np.maximum(largest, difference[:, :, channel], out=largest)

    # A difference of n levels between 8-bit images is n within float32's rounding, so
    # it is compared after the rounding that the Max Difference output gets too.
    if tolerance == 0:
        different = largest > 0
    else:
        levels = largest * 255
        different = np.round(levels, 4, out=levels) > tolerance

    max_difference = round(float(np.max(largest)) * 255, 4)
    mean_difference = round(float(np.mean(difference, dtype=np.float64)) * 255, 4)

    c = get_h_w_c(a)[2]
    if mode is DiffMode.MASK:
        image = different.astype(np.float32)
    elif mode is DiffMode.SIGNED:
        image = (signed[:, :, :3] if c == 4 else signed) * amplify
        image /= 2
        image += 0.5
    elif c == 4:
        image = np.maximum(difference[:, :, :3], difference[:, :, 3:4])
        image *= amplify
    else:
        image = np.multiply(difference, amplify, out=difference)

    return image, max_difference, mean_difference, int(np.count_nonzero(different))
