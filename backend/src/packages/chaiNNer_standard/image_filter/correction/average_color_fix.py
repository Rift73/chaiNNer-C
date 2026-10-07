from __future__ import annotations

from math import ceil

import cv2
import numpy as np

from nodes.impl.native_analysis import correction
from nodes.impl.native_channels import combine_rgb_alpha
from nodes.impl.native_cv_resize import resize as cv_resize
from nodes.impl.resize import ResizeFilter, resize
from nodes.properties.inputs import ImageInput, NumberInput
from nodes.properties.outputs import ImageOutput
from nodes.utils.utils import get_h_w_c

from .. import correction_group


@correction_group.register(
    schema_id="chainner:image:average_color_fix",
    name="Average Color Fix",
    description="""Correct for upscaling model color shift by matching
         the average color of the Input Image to that of a smaller Reference Image.
         Using significant downscaling increases generalization of averaging effect
         and can reduce artifacts in the output.""",
    icon="MdAutoFixHigh",
    inputs=[
        ImageInput("Image", channels=[3, 4]),
        ImageInput("Reference Image", channels=[3, 4]),
        NumberInput(
            "Reference Image Scale Factor",
            precision=4,
            step=12.5,
            max=100.0,
            default=12.5,
            unit="%",
        ),
    ],
    outputs=[ImageOutput(shape_as=0)],
)
def average_color_fix_node(
    input_img: np.ndarray, ref_img: np.ndarray, scale_factor: float
) -> np.ndarray:
    if scale_factor != 100.0:
        # Make sure reference image dims are not resized to 0
        h, w, _ = get_h_w_c(ref_img)
        out_dims = (
            max(ceil(w * (scale_factor / 100)), 1),
            max(ceil(h * (scale_factor / 100)), 1),
        )

        ref_img = resize(ref_img, out_dims, filter=ResizeFilter.BOX)

    input_h, input_w, input_c = get_h_w_c(input_img)
    ref_h, ref_w, ref_c = get_h_w_c(ref_img)

    assert ref_w < input_w and ref_h < input_h, (
        "Image must be larger than Reference Image"
    )

    # Find the diff of both images

    # Downscale the input image
    downscaled_input = resize(input_img, (ref_w, ref_h), filter=ResizeFilter.BOX)

    # adjust channels
    alpha = None
    downscaled_alpha = None
    ref_alpha = None
    if input_c > 3:
        alpha = input_img[:, :, 3:4]
        input_img = input_img[:, :, :3]
        downscaled_alpha = downscaled_input[:, :, 3:4]
        downscaled_input = downscaled_input[:, :, :3]
    if ref_c > 3:
        ref_alpha = ref_img[:, :, 3:4]
        ref_img = ref_img[:, :, :3]

    # Get difference between the reference image and downscaled input
    downscaled_diff, downscaled_alpha_diff = correction(
        downscaled_input, ref_img, downscaled_alpha, ref_alpha, add=False
    )

    # Upsample the difference
    diff = cv_resize(
        downscaled_diff,
        (input_w, input_h),
        interpolation=cv2.INTER_CUBIC,
    )

    alpha_diff = None
    if downscaled_alpha_diff is not None:
        alpha_diff = cv_resize(
            downscaled_alpha_diff,
            (input_w, input_h),
            interpolation=cv2.INTER_CUBIC,
        )
        alpha_diff = np.expand_dims(alpha_diff, 2)

    result, corrected_alpha = correction(input_img, diff, alpha, alpha_diff, add=True)
    if corrected_alpha is not None:
        alpha = corrected_alpha

    # add alpha back in
    if alpha is not None:
        result = combine_rgb_alpha(result, alpha)

    return result
