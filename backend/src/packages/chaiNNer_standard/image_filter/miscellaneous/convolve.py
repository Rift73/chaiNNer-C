from __future__ import annotations

import numpy as np

from nodes.impl.native_convolution import convolve
from nodes.properties.inputs import ImageInput, NumberInput, TextInput
from nodes.properties.outputs import ImageOutput

from .. import miscellaneous_group


@miscellaneous_group.register(
    schema_id="chainner:image:image_convolve",
    name="Convolve",
    description="Convolves input image with input kernel",
    icon="MdAutoFixHigh",
    inputs=[
        ImageInput("Image"),
        TextInput(
            "Kernel String",
            placeholder="0 0 0\n0 1 0\n0 0 0",
            multiline=True,
            has_handle=False,
            min_length=1,
        ),
        NumberInput("Padding", min=0, default=0),
    ],
    outputs=[
        ImageOutput(
            image_type="""
                    let w = Input0.width;
                    let h = Input0.height;

                    let kernel = Input1;

                    let padding = Input2;

                    Image {
                        width: w + padding * 2,
                        height: h + padding * 2,
                    }
                """,
        )
    ],
)
def convolve_node(
    img: np.ndarray,
    kernel_in: str,
    padding: int,
) -> np.ndarray:
    kernel = np.stack([l.split() for l in kernel_in.splitlines()], axis=0).astype(float)

    kernel = np.flipud(np.fliplr(kernel))

    return convolve(img, kernel, padding)
