from __future__ import annotations

import math
from enum import Enum

import numpy as np

from nodes.groups import if_enum_group
from nodes.impl.native_channels import combine_rgb_alpha
from nodes.impl.native_transfer import contrast_bounds, stack_stretched, stretch_range
from nodes.properties.inputs import BoolInput, EnumInput, ImageInput, SliderInput
from nodes.properties.outputs import ImageOutput
from nodes.utils.utils import get_h_w_c

from .. import adjustments_group


def _stretch(img: np.ndarray, range_min: float, range_max: float) -> np.ndarray:
    return stretch_range(img, range_min, range_max)


def _bounds(values: np.ndarray, percentile: float | None) -> tuple[float, float]:
    low, high = contrast_bounds(values, percentile)
    if not (math.isnan(low) or math.isnan(high)):
        return low, high
    # One NaN value must not turn the whole image into NaN: the others give the range.
    valid = values[~np.isnan(values)]
    return contrast_bounds(valid, percentile) if valid.size else (low, high)


class StretchMode(Enum):
    AUTO = 0
    PERCENTILE = 1
    MANUAL = 2


@adjustments_group.register(
    schema_id="chainner:image:stretch_contrast",
    description="Automatically stretches the histogram values in the given image. This is similar to auto levels.",
    name="Stretch Contrast",
    icon="ImContrast",
    inputs=[
        ImageInput(channels=[1, 3, 4]),
        EnumInput(StretchMode, preferred_style="tabs").with_id(1),
        if_enum_group(1, [StretchMode.AUTO, StretchMode.PERCENTILE])(
            BoolInput("Keep Colors", default=True),
        ),
        if_enum_group(1, StretchMode.PERCENTILE)(
            SliderInput(
                "Percentile", min=0, max=50, default=1, precision=2, scale="log"
            ),
        ),
        if_enum_group(1, StretchMode.MANUAL)(
            SliderInput("Minimum", min=0, max=255, default=0, precision=1),
            SliderInput("Maximum", min=0, max=255, default=255, precision=1),
        ),
    ],
    outputs=[
        ImageOutput(
            image_type="""
                let minMaxRangeValid: bool = match Input1 {
                    StretchMode::Manual => Input4 < Input5,
                    _ => true,
                };

                if minMaxRangeValid {
                    Input0
                } else {
                    error("Minimum must be less than Maximum.")
                }
            """,
        ),
    ],
)
def stretch_contrast_node(
    img: np.ndarray,
    mode: StretchMode,
    keep_colors: bool,
    percentile: float,
    manual_min: float,
    manual_max: float,
) -> np.ndarray:
    def get_range_of(i: np.ndarray) -> tuple[float, float]:
        if mode == StretchMode.AUTO:
            return _bounds(i, None)
        elif mode == StretchMode.PERCENTILE:
            return _bounds(i, percentile)
        elif mode == StretchMode.MANUAL:
            if manual_min > manual_max:
                raise ValueError("Minimum must be less than Maximum")
            return manual_min / 255, manual_max / 255

    def stretched(i: np.ndarray) -> np.ndarray:
        range_min, range_max = get_range_of(i)
        # A flat image or channel has no range to stretch and stays as it is, as in
        # Pillow's autocontrast and scikit-image's rescale_intensity (upstream turned
        # it black).
        if range_min == range_max and mode != StretchMode.MANUAL:
            return i
        return _stretch(i, range_min, range_max)

    _, _, c = get_h_w_c(img)

    alpha = None
    if c == 4:
        alpha = img[:, :, 3]
        img = img[:, :, :3]

    if keep_colors or mode == StretchMode.MANUAL or c == 1:
        # stretch all channels together

        img = stretched(img)
    else:
        # stretch all channels individually

        channels: list[np.ndarray] = []
        for i in range(get_h_w_c(img)[2]):
            channels.append(stretched(img[:, :, i]))

        img = stack_stretched(channels)

    if alpha is not None:
        img = (
            combine_rgb_alpha(img, alpha)
            if img.dtype == np.float32
            else stack_stretched([img[:, :, c] for c in range(3)] + [alpha])
        )

    return img
