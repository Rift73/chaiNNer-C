from __future__ import annotations

from enum import Enum

import numpy as np

from nodes.impl.native_color_ops import auto_threshold
from nodes.properties.inputs import EnumInput, ImageInput
from nodes.properties.outputs import NumberOutput

from .. import threshold_group


class AutoThreshold(Enum):
    OTSU = 0
    TRIANGLE = 1


_AUTO_THRESHOLD_LABELS: dict[AutoThreshold, str] = {
    AutoThreshold.OTSU: "Otsu's Method",
    AutoThreshold.TRIANGLE: "Triangle Method",
}


@threshold_group.register(
    schema_id="chainner:image:generate_threshold",
    name="Generate Threshold",
    description="Automatically determines an optimal threshold value for the given image.",
    icon="MdShowChart",
    inputs=[
        ImageInput(),
        EnumInput(AutoThreshold, "Method", option_labels=_AUTO_THRESHOLD_LABELS),
    ],
    outputs=[
        NumberOutput("Threshold", output_type="0..100"),
    ],
    see_also=[
        "chainner:image:threshold",
    ],
)
def generate_threshold_node(img: np.ndarray, method: AutoThreshold) -> float:
    return auto_threshold(img, method.value)
