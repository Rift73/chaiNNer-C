from __future__ import annotations

import numpy as np
from sanic.log import logger

from api import NodeContext
from nodes.groups import Condition, if_group
from nodes.impl.tensorrt.inference import use_tensorrt_session
from nodes.impl.tensorrt.model import TensorRTEngine
from nodes.impl.tensorrt.tiling import tiled_upscale
from nodes.impl.upscale.convenient_upscale import convenient_upscale
from nodes.properties.inputs import BoolInput, ImageInput, TensorRTEngineInput
from nodes.properties.outputs import ImageOutput
from nodes.utils.utils import get_h_w_c

from ...settings import get_settings
from .. import processing_group

if processing_group is not None:

    @processing_group.register(
        schema_id="chainner:tensorrt:upscale_image",
        description=(
            "Upscales an image using a TensorRT engine. Images of any size are split into"
            " overlapping tiles that fit the engine's optimization profile, and the tiles"
            " are blended back together without seams."
        ),
        inputs=[
            ImageInput().with_id(1),
            TensorRTEngineInput().with_id(0),
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
                image_type="convenientUpscaleTrt(Input0, Input1)",
            )
        ],
        name="Upscale Image",
        icon="BsNvidia",
        node_context=True,
    )
    def upscale_image_node(
        context: NodeContext,
        img: np.ndarray,
        engine: TensorRTEngine,
        separate_alpha: bool,
    ) -> np.ndarray:
        settings = get_settings(context)
        scale = engine.scale
        if scale is None:
            raise ValueError("The TensorRT engine's upscale factor is unknown.")

        h, w, c = get_h_w_c(img)
        logger.debug("Upscaling a %dx%dx%d image with TensorRT", h, w, c)

        # The session outlives the run; clearing this node frees it.
        with use_tensorrt_session(
            engine, settings.gpu_index, context.node_id
        ) as session:
            return convenient_upscale(
                img,
                engine.input_channels,
                engine.output_channels,
                lambda i: tiled_upscale(i, scale, session.bounds, session, context),
                separate_alpha,
            )
