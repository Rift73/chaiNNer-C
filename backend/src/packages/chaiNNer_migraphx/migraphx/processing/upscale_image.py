from __future__ import annotations

import numpy as np
from sanic.log import logger

from api import NodeContext
from nodes.groups import Condition, if_group
from nodes.impl.migraphx.model import MIGraphXProgram
from nodes.impl.migraphx.session import use_runner_session
from nodes.impl.upscale.convenient_upscale import convenient_upscale
from nodes.properties.inputs import BoolInput, ImageInput, MIGraphXProgramInput
from nodes.properties.outputs import ImageOutput
from nodes.utils.utils import get_h_w_c
from util import held

from ...settings import get_settings
from .. import processing_group

if processing_group is not None:

    @processing_group.register(
        schema_id="chainner:migraphx:upscale_image",
        description=(
            "Upscales an image using a MIGraphX program. Images of any size are split into"
            " overlapping tiles of the program's size, and the tiles are blended back"
            " together without seams."
        ),
        inputs=[
            ImageInput().with_id(1),
            MIGraphXProgramInput().with_id(0),
            if_group(Condition.type(1, "Image { channels: 4 } "))(
                BoolInput("Separate Alpha", default=False)
                .with_id(4)
                .with_docs(
                    "Upscale alpha separately from color. Enabling this option will cause the alpha of"
                    " the upscaled image to be less noisy and more accurate to the alpha of the original"
                    " image, but the image may suffer from dark borders near transparency edges"
                    " (transition from fully transparent to fully opaque).",
                    "Whether enabling this option will improve the upscaled image depends on the original"
                    " image. We generally recommend this option for images with smooth transitions between"
                    " transparent and opaque regions.",
                )
            ),
        ],
        outputs=[
            ImageOutput(
                "Image",
                image_type="convenientUpscale(Input0, Input1)",
            )
        ],
        name="Upscale Image",
        icon="BsAmd",
        node_context=True,
    )
    def upscale_image_node(
        context: NodeContext,
        img: np.ndarray,
        program: MIGraphXProgram,
        separate_alpha: bool,
    ) -> np.ndarray:
        settings = get_settings(context)
        h, w, c = get_h_w_c(img)
        logger.debug("Upscaling a %dx%dx%d image with MIGraphX", h, w, c)

        # The session outlives the run; clearing this node frees it. Its lock lets items
        # computed ahead share it; their wait for it is not their time.
        with use_runner_session(
            program, settings.gpu_index, context.node_id
        ) as runner, held(runner.lock):
            return convenient_upscale(
                img,
                program.input_channels,
                program.output_channels,
                runner.upscale,
                separate_alpha,
            )
