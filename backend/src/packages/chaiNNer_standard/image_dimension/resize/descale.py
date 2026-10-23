from __future__ import annotations

from enum import Enum

import numpy as np

from api import KeyInfo
from nodes.groups import Condition, if_enum_group, if_group
from nodes.impl.linear_light import LightTransfer
from nodes.impl.native_descale import descale
from nodes.properties.inputs import (
    BoolInput,
    EnumInput,
    ImageInput,
    LightTransferInput,
    NumberInput,
)
from nodes.properties.outputs import ImageOutput

from .. import resize_group


class DescaleKernel(Enum):
    BILINEAR = 1
    BICUBIC = 2
    LANCZOS = 3
    SPLINE16 = 4
    SPLINE36 = 5
    SPLINE64 = 6
    POINT = 7


class DescaleBorder(Enum):
    MIRROR = 0
    ZERO = 1
    EXTEND = 2


@resize_group.register(
    schema_id="chainner:image:descale",
    name="Descale",
    description=[
        "Undo an upscale: compute the image that, resized to this image's size with the given kernel, gives this image back.",
        "For video or artwork upscaled from a lower native resolution (common in anime), with the kernel and size it was upscaled with. A wrong kernel or size leaves ringing or blur, so compare candidates.",
        "Uses [vapoursynth-descale](https://github.com/Jaded-Encoding-Thaumaturgy/vapoursynth-descale) (MIT).",
    ],
    icon="MdOutlinePhotoSizeSelectSmall",
    inputs=[
        ImageInput(),
        NumberInput("Width", min=1, default=1280, unit="px").with_id(1),
        NumberInput("Height", min=1, default=720, unit="px").with_id(2),
        EnumInput(
            DescaleKernel,
            label="Kernel",
            default=DescaleKernel.BILINEAR,
            option_labels={
                DescaleKernel.SPLINE16: "Spline16",
                DescaleKernel.SPLINE36: "Spline36",
                DescaleKernel.SPLINE64: "Spline64",
            },
        ).with_id(3),
        if_enum_group(3, DescaleKernel.BICUBIC)(
            NumberInput("B", precision=4, min=-2, max=2, default=0, step=0.1).with_id(
                4
            ),
            NumberInput("C", precision=4, min=-2, max=2, default=0.5, step=0.1).with_id(
                5
            ),
        ),
        if_enum_group(3, DescaleKernel.LANCZOS)(
            NumberInput("Taps", min=1, max=32, default=3).with_id(6),
        ),
        NumberInput("Blur", precision=4, min=0.01, max=10, default=1, step=0.1)
        .with_docs("Widens (above 1) or narrows (below 1) the kernel.")
        .with_id(7),
        NumberInput("Source Left", precision=4, min=-1000, max=1000, default=0)
        .with_docs("Shift of the source window, in target pixels.")
        .with_id(8),
        NumberInput("Source Top", precision=4, min=-1000, max=1000, default=0).with_id(
            9
        ),
        NumberInput("Source Width", precision=4, min=0, default=0)
        .with_docs(
            "Width of the source window in target pixels, for a fractional native size (for example 1279.5). 0 uses Width."
        )
        .with_id(10),
        NumberInput("Source Height", precision=4, min=0, default=0)
        .with_docs("Height of the source window in target pixels. 0 uses Height.")
        .with_id(11),
        EnumInput(
            DescaleBorder,
            label="Border",
            default=DescaleBorder.MIRROR,
            option_labels={DescaleBorder.EXTEND: "Extend"},
        )
        .with_docs("The padding the upscale used at the image edges.")
        .with_id(12),
        BoolInput("Linear Light", default=False)
        .with_docs(
            "For an upscale done in linear light: the colour channels go through the Transfer curve to linear light before the descale and back after. Alpha is not converted.",
            "A one-channel image is converted as one plane, which is approximate for a Y (luma) plane: linear light belongs to R, G and B.",
        )
        .with_id(13),
        if_group(Condition.bool(13, True))(
            LightTransferInput(LightTransfer.GAMMA_24).with_id(14),
        ),
    ],
    outputs=[
        ImageOutput(
            image_type="""
                Image {
                    width: Input1,
                    height: Input2,
                    channels: Input0.channels,
                }
            """
        )
    ],
    key_info=KeyInfo.type(
        """
        let size = string::concat(toString(Input1), "x", toString(Input2));

        if Input13 { string::concat(size, " · ", LightTransfer::label(Input14)) } else { size }
        """
    ),
)
def descale_node(
    img: np.ndarray,
    width: int,
    height: int,
    kernel: DescaleKernel,
    b: float,
    c: float,
    taps: int,
    blur: float,
    src_left: float,
    src_top: float,
    src_width: float,
    src_height: float,
    border: DescaleBorder,
    linear_light: bool,
    transfer: LightTransfer,
) -> np.ndarray:
    return descale(
        img,
        width,
        height,
        kernel.value,
        taps,
        b,
        c,
        blur,
        (src_left, src_top),
        (src_width or width, src_height or height),
        border.value,
        transfer if linear_light else None,
    )
