from __future__ import annotations

import numpy as np

import navi
from nodes.impl.normals.addition import AdditionMethod, add_normals
from nodes.impl.normals.util import xyz_to_bgr
from nodes.properties.inputs import EnumInput, ImageInput, SliderInput
from nodes.properties.outputs import ImageOutput

from .. import normal_map_group


@normal_map_group.register(
    schema_id="chainner:image:add_normals",
    name="Add Normals",
    description="""Add 2 normal maps together. Only the R and G
            channels of the input image will be used. The output normal map
            is guaranteed to be normalized.""",
    icon="MdAddCircleOutline",
    inputs=[
        ImageInput("Normal Map 1", channels=[3, 4]),
        SliderInput("Strength 1", max=200, default=100),
        ImageInput("Normal Map 2", channels=[3, 4]),
        SliderInput("Strength 2", max=200, default=100),
        EnumInput(
            AdditionMethod,
            label="Method",
            default=AdditionMethod.PARTIAL_DERIVATIVES,
        ),
    ],
    outputs=[
        ImageOutput(
            "Normal Map",
            image_type=navi.Image(
                width="Input0.width & Input2.width",
                height="Input0.height & Input2.height",
            ),
            channels=3,
        ).with_never_reason(
            "The given normal maps have different sizes but must be the same size."
        ),
    ],
)
def add_normals_node(
    n1: np.ndarray,
    strength1: int,
    n2: np.ndarray,
    strength2: int,
    method: AdditionMethod,
) -> np.ndarray:
    # A map of one pixel, row or column is broadcast over the other.
    (h1, w1), (h2, w2) = n1.shape[:2], n2.shape[:2]
    if (h1 != h2 and 1 not in (h1, h2)) or (w1 != w2 and 1 not in (w1, w2)):
        raise ValueError(
            f"The normal maps must be the same size, not {n1.shape[1]}x{n1.shape[0]}"
            f" and {n2.shape[1]}x{n2.shape[0]}."
        )
    return xyz_to_bgr(
        add_normals(
            method,
            n1,
            n2,
            f1=strength1 / 100,
            f2=strength2 / 100,
        )
    )
