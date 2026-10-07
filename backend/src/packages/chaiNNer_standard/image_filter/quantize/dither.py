from __future__ import annotations

from enum import Enum

import numpy as np

from nodes.groups import if_enum_group
from nodes.impl.dithering.constants import (
    ERROR_PROPAGATION_MAP_LABELS,
    THRESHOLD_MAP_LABELS,
    ErrorDiffusionMap,
    ThresholdMap,
)
from nodes.impl.native_neighborhood import DIFFUSION_IDS, uniform_dither
from nodes.properties.inputs import EnumInput, ImageInput, NumberInput
from nodes.properties.outputs import ImageOutput

from .. import quantize_group

_THRESHOLD_MAP: dict[ThresholdMap, int] = {
    ThresholdMap.BAYER_2: 2,
    ThresholdMap.BAYER_4: 4,
    ThresholdMap.BAYER_8: 8,
    ThresholdMap.BAYER_16: 16,
}


class UniformDitherAlgorithm(Enum):
    NONE = "None"
    ORDERED = "Ordered"
    DIFFUSION = "Diffusion"
    RIEMERSMA = "Riemersma"


UNIFORM_DITHER_ALGORITHM_LABELS = {
    UniformDitherAlgorithm.NONE: "No dithering",
    UniformDitherAlgorithm.ORDERED: "Ordered Dithering",
    UniformDitherAlgorithm.DIFFUSION: "Error Diffusion",
    UniformDitherAlgorithm.RIEMERSMA: "Riemersma Dithering",
}


@quantize_group.register(
    schema_id="chainner:image:dither",
    name="Dither",
    description="Apply one of a variety of dithering algorithms with a uniform (evenly-spaced) palette.",
    icon="MdShowChart",
    inputs=[
        ImageInput(channels=[1, 3, 4]),
        NumberInput("Colors per channel", min=2, default=8),
        EnumInput(
            UniformDitherAlgorithm,
            option_labels=UNIFORM_DITHER_ALGORITHM_LABELS,
            default=UniformDitherAlgorithm.DIFFUSION,
        ).with_id(2),
        if_enum_group(2, UniformDitherAlgorithm.ORDERED)(
            EnumInput(
                ThresholdMap,
                option_labels=THRESHOLD_MAP_LABELS,
                default=ThresholdMap.BAYER_16,
            ).with_id(3),
        ),
        if_enum_group(2, UniformDitherAlgorithm.DIFFUSION)(
            EnumInput(
                ErrorDiffusionMap,
                option_labels=ERROR_PROPAGATION_MAP_LABELS,
                default=ErrorDiffusionMap.FLOYD_STEINBERG,
            ).with_id(4),
        ),
        if_enum_group(2, UniformDitherAlgorithm.RIEMERSMA)(
            NumberInput("History Length", min=2, default=16).with_id(5),
        ),
    ],
    outputs=[ImageOutput(shape_as=0)],
)
def dither_node(
    img: np.ndarray,
    num_colors: int,
    dither_algorithm: UniformDitherAlgorithm,
    threshold_map: ThresholdMap,
    error_diffusion_map: ErrorDiffusionMap,
    history_length: int,
) -> np.ndarray:
    if dither_algorithm == UniformDitherAlgorithm.NONE:
        return uniform_dither(img, num_colors, 0)
    elif dither_algorithm == UniformDitherAlgorithm.ORDERED:
        map_size = _THRESHOLD_MAP[threshold_map]
        return uniform_dither(img, num_colors, 1, map_size=map_size)
    elif dither_algorithm == UniformDitherAlgorithm.DIFFUSION:
        algorithm = DIFFUSION_IDS[error_diffusion_map.value]
        return uniform_dither(img, num_colors, 2, algorithm=algorithm)
    elif dither_algorithm == UniformDitherAlgorithm.RIEMERSMA:
        return uniform_dither(img, num_colors, 3, history_length=history_length)
