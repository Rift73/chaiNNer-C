from __future__ import annotations

import numpy as np

from nodes.impl.colour_info import (
    ChromaLocation,
    ColourInfo,
    Matrix,
    Primaries,
    Range,
    Transfer,
)
from nodes.impl.plane_resample import KERNEL_LABELS, Kernel
from nodes.impl.yuv_rgb import planes_to_rgb, resolve_colour_info
from nodes.properties.inputs import (
    ColourInfoInput,
    EnumInput,
    ImageInput,
    colour_override_inputs,
)
from nodes.properties.outputs import ColourInfoOutput, ImageOutput, TextOutput
from nodes.utils.utils import get_h_w_c

from .. import all_group


@all_group.register(
    schema_id="chainner:image:planes_to_rgb",
    name="Planes to RGB",
    description=[
        "Convert video planes (Y, U and V, as Load Video's Planes mode gives them) to an RGB image at Y's size.",
        "U and V are upsampled to Y's size from where their chroma siting places them, and the matrix is applied in 32-bit float without clipping, as VapourSynth's resize does. Transfer, primaries and range are not converted: they are passed on in the Colour Info.",
        "Without a Colour Info, Auto guesses as Load Video does for an untagged video: the BT.709 matrix for planes 1280 or more wide or more than 576 high, else BT.601; limited range; left-sited chroma.",
    ],
    icon="MdCallMerge",
    inputs=[
        ImageInput("Y", channels=1).with_docs("The luma plane."),
        ImageInput("U", channels=1).with_docs(
            "The blue-difference chroma plane: Y's size, half Y's width, or half Y's"
            " width and height (half sizes round up)."
        ),
        ImageInput("V", channels=1).with_docs(
            "The red-difference chroma plane, the size of U."
        ),
        ColourInfoInput()
        .make_optional()
        .with_docs("The planes' colour information, from Load Video or Clip Frames."),
        *colour_override_inputs(4),
        EnumInput(
            Kernel,
            "Chroma Kernel",
            default=Kernel.SPLINE36,
            type_name="ChromaKernel",
            option_labels=KERNEL_LABELS,
        )
        .with_id(9)
        .with_docs("The kernel that upsamples U and V to Y's size."),
    ],
    outputs=[
        ImageOutput(
            image_type="""
                def fits(luma: uint, chroma: uint) =
                    chroma == luma or chroma == ceil(luma / 2) or chroma == ceil(luma / 4);

                if Input1.width != Input2.width or Input1.height != Input2.height {
                    error("The U and V planes must have one size.")
                } else if not (fits(Input0.width, Input1.width) and fits(Input0.height, Input1.height)) {
                    error("The U and V planes must have Y's size, half of it or a quarter of it.")
                } else {
                    match Input4 {
                        Matrix::Bt2020Cl | Matrix::Ictcp => {
                            error("This matrix is not a plain matrix: planes with it cannot be converted to RGB here.")
                        },
                        Matrix::Rgb => {
                            if Input1.width == Input0.width and Input1.height == Input0.height {
                                Image
                            } else {
                                error("RGB (identity) planes cannot be subsampled: use 4:4:4.")
                            }
                        },
                        _ => Image,
                    }
                }
            """,
            size_as=0,
            channels=3,
            assume_normalized=True,
        ),
        ColourInfoOutput().with_docs(
            "The colour information, re-issued for the RGB image (its chroma at full"
            " size, so Save Video does not correct the siting again)."
        ),
        TextOutput("Colour"),
    ],
)
def planes_to_rgb_node(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    info: ColourInfo | None,
    matrix: Matrix,
    transfer: Transfer,
    primaries: Primaries,
    range_: Range,
    chroma_location: ChromaLocation,
    kernel: Kernel,
) -> tuple[np.ndarray, ColourInfo, str]:
    h, w, _ = get_h_w_c(y)
    colour = resolve_colour_info(
        info, (w, h), (w, h), matrix, transfer, primaries, range_, chroma_location
    )
    img = planes_to_rgb(y, u, v, colour.matrix, colour.chroma_location, kernel)
    return img, colour, colour.summary()
