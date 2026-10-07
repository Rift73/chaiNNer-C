from __future__ import annotations

from enum import Enum

import numpy as np

from nodes.groups import optional_list_group
from nodes.impl import native_layout
from nodes.properties.inputs import EnumInput, ImageInput
from nodes.properties.outputs import ImageOutput
from nodes.utils.utils import ALPHABET

from .. import compositing_group


class Expression(Enum):
    MEDIAN = "median"
    MEAN = "mean"
    MIN = "minimum"
    MAX = "maximum"


@compositing_group.register(
    schema_id="chainner:image:z_stack",
    name="Z-Stack Images",
    description="""Aligns multiple images and evaluates them in relation to each other to create a merged image result.""",
    icon="BsLayersHalf",
    inputs=[
        EnumInput(Expression),
        ImageInput("Image A"),
        ImageInput("Image B"),
        optional_list_group(
            *[
                ImageInput(f"Image {letter}").make_optional()
                for letter in ALPHABET[2:14]
            ],
        ),
    ],
    outputs=[
        ImageOutput(
            image_type="""
                def conv(i: Image | null) = match i { Image => i, _ => any };

                Input1 & Input2
                    & conv(Input3)
                    & conv(Input4)
                    & conv(Input5)
                    & conv(Input6)
                    & conv(Input7)
                    & conv(Input8)
                    & conv(Input9)
                    & conv(Input10)
                    & conv(Input11)
                    & conv(Input12)
                    & conv(Input13)
                    & conv(Input14)
            """
        ).with_never_reason(
            "All input images much have the same size and number of channels."
        ),
    ],
)
def z_stack_images_node(
    expression: Expression,
    *inputs: np.ndarray | None,
) -> np.ndarray:
    return native_layout.z_stack(inputs, expression.value)
