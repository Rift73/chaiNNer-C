from __future__ import annotations

import numpy as np

from nodes.groups import optional_list_group
from nodes.properties.inputs import BaseInput, ImageInput
from nodes.properties.outputs import LargeImageOutput, TextOutput
from nodes.utils.utils import get_h_w_c

from .. import io_group


@io_group.register(
    schema_id="chainner:image:view",
    name="View Image",
    description=(
        "See an inline preview of the image in the editor."
        " With two or more images connected, it compares two of them with a slider."
    ),
    icon="BsEyeFill",
    inputs=[
        ImageInput("Image 1"),
        optional_list_group(
            *[ImageInput(f"Image {i}").make_optional() for i in range(2, 11)],
        ),
        optional_list_group(
            *[
                BaseInput(
                    kind="generic", label=f"Name {i}", input_type="string"
                ).make_optional()
                for i in range(1, 11)
            ],
        ),
    ],
    outputs=[
        LargeImageOutput(
            "Preview 1",
            image_type="Input0",
            has_handle=False,
            assume_normalized=True,
            full_size=True,
        ),
        *[
            LargeImageOutput(
                f"Preview {i + 1}",
                image_type=f"Input{i}",
                has_handle=False,
                assume_normalized=True,
                full_size=True,
            ).null_unless(i, "Image")
            for i in range(1, 10)
        ],
        # the name of image k: its Name input's text, which the frontend shows in the viewer
        *[
            TextOutput(
                f"Name {i}", output_type=f"Input{i + 9}", has_handle=False
            ).null_unless(i + 9, "string")
            for i in range(1, 11)
        ],
    ],
    side_effects=True,
)
def view_image_node(
    image_1: np.ndarray,
    image_2: np.ndarray | None = None,
    image_3: np.ndarray | None = None,
    image_4: np.ndarray | None = None,
    image_5: np.ndarray | None = None,
    image_6: np.ndarray | None = None,
    image_7: np.ndarray | None = None,
    image_8: np.ndarray | None = None,
    image_9: np.ndarray | None = None,
    image_10: np.ndarray | None = None,
    name_1: str | None = None,
    name_2: str | None = None,
    name_3: str | None = None,
    name_4: str | None = None,
    name_5: str | None = None,
    name_6: str | None = None,
    name_7: str | None = None,
    name_8: str | None = None,
    name_9: str | None = None,
    name_10: str | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
]:
    images = (
        image_1,
        image_2,
        image_3,
        image_4,
        image_5,
        image_6,
        image_7,
        image_8,
        image_9,
        image_10,
    )
    for img in images:
        if img is not None:
            c = get_h_w_c(img)[2]
            if c not in (1, 3, 4):
                raise ValueError(
                    f"View Image shows images of 1, 3 or 4 channels, not {c}."
                )
    return (
        *images,
        name_1,
        name_2,
        name_3,
        name_4,
        name_5,
        name_6,
        name_7,
        name_8,
        name_9,
        name_10,
    )
