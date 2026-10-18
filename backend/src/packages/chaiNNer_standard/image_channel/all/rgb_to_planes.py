from __future__ import annotations

from enum import Enum

import numpy as np

from nodes.impl.colour_info import (
    ChromaLocation,
    ColourInfo,
    Matrix,
    Primaries,
    Range,
    Transfer,
    chroma_size_for,
)
from nodes.impl.plane_resample import KERNEL_LABELS, Kernel
from nodes.impl.yuv_rgb import resolve_colour_info, rgb_to_planes
from nodes.properties.inputs import (
    ColourInfoInput,
    EnumInput,
    ImageInput,
    colour_override_inputs,
)
from nodes.properties.outputs import ColourInfoOutput, ImageOutput, TextOutput
from nodes.utils.utils import get_h_w_c

from .. import all_group


class ChromaSubsampling(Enum):
    YUV444 = "444"
    YUV422 = "422"
    YUV420 = "420"


# log2 of the chroma subsampling (width, height)
SUBSAMPLING_SHIFTS = {
    ChromaSubsampling.YUV444: (0, 0),
    ChromaSubsampling.YUV422: (1, 0),
    ChromaSubsampling.YUV420: (1, 1),
}

CHROMA_IMAGE = """
    let w = Input0.width;
    let h = Input0.height;
    match Input7 {
        ChromaSubsampling::Yuv444 => Image { width: w, height: h },
        ChromaSubsampling::Yuv422 => Image { width: int & ceil(w / 2), height: h },
        ChromaSubsampling::Yuv420 => Image {
            width: int & ceil(w / 2),
            height: int & ceil(h / 2),
        },
    }
"""


@all_group.register(
    schema_id="chainner:image:rgb_to_planes",
    name="RGB to Planes",
    description=[
        "Convert an RGB image to video planes (Y, U and V), for Save Video's Planes mode or plane-wise processing.",
        "The matrix is applied in 32-bit float without clipping, and U and V are downsampled to the subsampling's size where the chroma siting places them, as VapourSynth's resize does. Transfer, primaries and range are not converted: they are passed on in the Colour Info.",
        "Without a Colour Info, Auto guesses as Load Video does for an untagged video: the BT.709 matrix for images 1280 or more wide or more than 576 high, else BT.601; limited range; left-sited chroma.",
    ],
    icon="MdCallSplit",
    inputs=[
        ImageInput(channels=3),
        ColourInfoInput()
        .make_optional()
        .with_docs(
            "The colour information to convert with, for example the source video's."
        ),
        *colour_override_inputs(2),
        EnumInput(
            ChromaSubsampling,
            "Subsampling",
            default=ChromaSubsampling.YUV420,
            option_labels={
                ChromaSubsampling.YUV444: "4:4:4",
                ChromaSubsampling.YUV422: "4:2:2",
                ChromaSubsampling.YUV420: "4:2:0",
            },
        )
        .with_id(7)
        .with_docs(
            "The size of U and V: the image's (4:4:4), half its width (4:2:2), or half"
            " its width and height (4:2:0). Half sizes round up."
        ),
        EnumInput(
            Kernel,
            "Chroma Kernel",
            default=Kernel.SPLINE36,
            type_name="ChromaKernel",
            option_labels=KERNEL_LABELS,
        )
        .with_id(8)
        .with_docs("The kernel that downsamples U and V."),
    ],
    outputs=[
        ImageOutput("Y", size_as=0, channels=1, assume_normalized=True),
        ImageOutput("U", image_type=CHROMA_IMAGE, channels=1, assume_normalized=True),
        ImageOutput("V", image_type=CHROMA_IMAGE, channels=1, assume_normalized=True),
        ColourInfoOutput().with_docs(
            "The colour information, re-issued for these planes."
        ),
        TextOutput("Colour"),
    ],
)
def rgb_to_planes_node(
    img: np.ndarray,
    info: ColourInfo | None,
    matrix: Matrix,
    transfer: Transfer,
    primaries: Primaries,
    range_: Range,
    chroma_location: ChromaLocation,
    subsampling: ChromaSubsampling,
    kernel: Kernel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ColourInfo, str]:
    h, w, _ = get_h_w_c(img)
    shifts = SUBSAMPLING_SHIFTS[subsampling]
    colour = resolve_colour_info(
        info,
        (w, h),
        chroma_size_for((w, h), shifts),
        matrix,
        transfer,
        primaries,
        range_,
        chroma_location,
    )
    y, u, v = rgb_to_planes(img, colour.matrix, colour.chroma_location, shifts, kernel)
    return y, u, v, colour, colour.summary()
