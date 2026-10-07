from __future__ import annotations

from typing import SupportsFloat

import numpy as np

from nodes.impl.native_analysis import statistics
from nodes.properties.inputs import ImageInput, SliderInput
from nodes.properties.outputs import NumberOutput

from .. import miscellaneous_group


@miscellaneous_group.register(
    schema_id="chainner:image:image_statistics",
    name="Image Statistics",
    description="Returns statistics of the given image.",
    icon="MdOutlineAssessment",
    inputs=[
        ImageInput(channels=1),
        SliderInput(
            "Percentile",
            precision=2,
            min=0,
            max=100,
            default=50,
            step=1,
            hide_trailing_zeros=True,
        ),
    ],
    outputs=[
        NumberOutput("Minimum", output_type="0..255"),
        NumberOutput("Maximum", output_type="0..255"),
        NumberOutput("Arithmetic Mean", output_type="0..255"),
        NumberOutput("Percentile", output_type="0..255"),
    ],
)
def image_statistics_node(
    img: np.ndarray,
    percentile: float,
) -> tuple[float, float, float, float]:
    def to_float(n: SupportsFloat) -> float:
        # float32 has ~8 digits of precision.
        # So by rounding to 4 digits, we have 1 digit left over to contain rounding errors
        return round(float(n) * 255, 4)

    minimum, maximum, mean, quantile = statistics(img, percentile)
    return to_float(minimum), to_float(maximum), to_float(mean), to_float(quantile)
