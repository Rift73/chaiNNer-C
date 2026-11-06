from __future__ import annotations

import numpy as np

from api import NodeContext
from nodes.impl.pytorch.pix_transform.auto_split import pix_transform_auto_split
from nodes.impl.pytorch.pix_transform.pix_transform import Params
from nodes.impl.pytorch.utils import safe_cuda_cache_empty
from nodes.impl.upscale.grayscale import SplitMode
from nodes.properties.inputs import EnumInput, ImageInput, NumberInput, SliderInput
from nodes.properties.outputs import ImageOutput

from ...settings import get_settings
from .. import processing_group


@processing_group.register(
    schema_id="chainner:pytorch:guided_upscale",
    name="Guided Upscale",
    description=(
        "Upscales a source image using a guide."
        " This is most useful for very small source images."
        "\n\nUnder the hood, PixTransform is used which trains a NN to convert the guide image into the source image."
        " Note that this operation is very expensive, because it needs to train a NN."
        " Try a small number of iterations before going up to around 30k."
    ),
    icon="PyTorch",
    inputs=[
        ImageInput("Source"),
        ImageInput("Guide"),
        SliderInput(
            "Iterations",
            min=0.1,
            max=100,
            default=1,
            precision=1,
            scale="log",
            unit="k",
        ),
        EnumInput(
            SplitMode,
            "Channel split mode",
            default=SplitMode.LAB,
            option_labels={SplitMode.RGB: "RGB", SplitMode.LAB: "L*a*b"},
        ),
        NumberInput("Tile Size", min=16, max=2048, default=80, unit="px")
        .with_id(4)
        .with_docs(
            "The largest part of the source, in source pixels, that one network trains on."
            " Each tile trains for all the Iterations, so the time grows with the number"
            " of tiles, which grows with the source's area.",
            "Measured on the CPU at 4x and 500 iterations with a 128x128 RGB source, on one"
            " image: one tile 40.85 dB in 22.6 s, tiles of 80 px 46.15 dB in 33.6 s, tiles"
            " of 48 px 47.18 dB in 158.8 s.",
            "The gain varies by image: one tile against tiles of 80 px gave 43.47 against"
            " 42.58 dB on a second image and 43.37 against 46.45 dB on a third. On a 192x192"
            " source (one channel), tiles of 80 px took 37.3 s against 4.8 s for one tile.",
            hint=True,
        ),
    ],
    outputs=[
        ImageOutput(
            image_type="""
                let source = Input0;
                let guide = Input1;

                let kScale = (
                    // guide image's size must be `k * source.size` for `k>1`
                    guide.width / source.width == int
                    and guide.width / source.width == guide.height / source.height
                );

                if guide.width <= source.width {
                    error("The guide image must be larger than the source image.")
                } else if not kScale {
                    error("The size of the guide image must be an integer multiple of the size of the source image (e.g. 2x, 3x, 4x, ...).")
                } else {
                    Image {
                        width: guide.width,
                        height: guide.height,
                        channels: source.channels,
                    }
                }
                """
        ),
    ],
    node_context=True,
)
def guided_upscale_node(
    context: NodeContext,
    source: np.ndarray,
    guide: np.ndarray,
    iterations: float,
    split_mode: SplitMode,
    tile_size: int,
) -> np.ndarray:
    exec_options = get_settings(context)
    context.add_cleanup(
        safe_cuda_cache_empty,
        after="node" if exec_options.force_cache_wipe else "chain",
    )
    return pix_transform_auto_split(
        source=source,
        guide=guide,
        device=exec_options.device,
        params=Params(iteration=int(iterations * 1000)),
        split_mode=split_mode,
        tile_size=tile_size,
        progress=context,
    )
