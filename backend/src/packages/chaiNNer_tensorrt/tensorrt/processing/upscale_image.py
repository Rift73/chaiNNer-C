from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sanic.log import logger

from api import NodeContext
from nodes.groups import Condition, if_group
from nodes.impl.tensorrt.inference import use_tensorrt_session
from nodes.impl.tensorrt.model import TensorRTEngine
from nodes.impl.tensorrt.runner import (
    runner_enabled,
    runner_takes,
    use_runner_session,
)
from nodes.impl.upscale.convenient_upscale import convenient_upscale
from nodes.impl.upscale.temporal_window import LOOK_AHEAD, WARM_UP
from nodes.impl.upscale.tile_report import TileReport
from nodes.properties.inputs import BoolInput, ImageInput, TensorRTEngineInput
from nodes.properties.outputs import ImageOutput, TextOutput
from nodes.utils.utils import get_h_w_c
from util import held

from ...settings import get_settings
from .. import processing_group


def context_applies(engine: object) -> bool:
    """Upscale Image's gate (`temporal_if` on its engine input): the context items apply
    to an engine set, and to a value that is no engine, which it cannot judge."""
    return not isinstance(engine, TensorRTEngine) or engine.info.temporal is not None


def engine_upscale(
    context: NodeContext,
    img: np.ndarray,
    engine: TensorRTEngine,
    separate_alpha: bool,
    balanced_tiles: bool,
    before_clip: Callable[[np.ndarray], None] | None = None,
) -> tuple[np.ndarray, str]:
    """Upscale Image's image and its tiles' size, also for PyTorch's Upscale Image with Use
    TensorRT Engine. before_clip: sees each result before its clip (tiled_upscale)."""
    settings = get_settings(context)
    if engine.info.temporal is not None:
        # The session imports PyTorch and spandrel, which image engines do not need.
        from nodes.impl.tensorrt.rdual_session import temporal_engine_upscale

        report = TileReport()
        image = temporal_engine_upscale(
            context, img, engine, settings.gpu_index, report
        )
        return image, report.text
    scale = engine.scale
    if scale is None:
        raise ValueError("The TensorRT engine's upscale factor is unknown.")

    h, w, c = get_h_w_c(img)
    logger.debug("Upscaling a %dx%dx%d image with TensorRT", h, w, c)
    report = TileReport()

    if runner_enabled() and runner_takes(engine) and not balanced_tiles:
        # The native runner (opt-in: CHAINNER_C_TRT_RUNNER=1), whose tiles are the
        # default's: kept and locked as the session below.
        with use_runner_session(
            engine, settings.gpu_index, context.node_id
        ) as runner, held(runner.lock):

            def on_runner(i: np.ndarray) -> np.ndarray:
                output = runner.upscale(i, report, context)
                if before_clip is not None:
                    before_clip(output)
                return output

            image = convenient_upscale(
                img,
                engine.input_channels,
                engine.output_channels,
                on_runner,
                separate_alpha,
            )
            return image, report.text

    # The session outlives the run; clearing this node, or Load Engine letting go
    # of its engine, frees it. Items computed ahead share it; their waits for its
    # turns are not their time.
    with use_tensorrt_session(engine, settings.gpu_index, context.node_id) as session:
        context.add_cleanup(session.use_largest_tiles, after="chain")
        image = convenient_upscale(
            img,
            engine.input_channels,
            engine.output_channels,
            lambda i: session.upscale(
                i,
                scale,
                context,
                report,
                balanced=balanced_tiles,
                before_clip=before_clip,
            ),
            separate_alpha,
            clip=False,
        )
        return image, report.text


if processing_group is not None:

    @processing_group.register(
        schema_id="chainner:tensorrt:upscale_image",
        description=[
            "Upscales an image using a TensorRT engine. Images of any size are split into overlapping tiles that fit the engine's optimization profile, and the tiles are blended back together without seams. When the GPU's memory runs out, a Dynamic engine goes on with smaller tiles.",
            "An RDUAL engine set (a video model's engines, from Build Engine or Load Engine) runs inside a loop with the frames around each one, as PyTorch's Upscale Image runs the model: the frame must have the set's size, Separate Alpha and Balanced Tiles do not apply, and when the GPU's memory runs out it reads fewer frames ahead.",
        ],
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
            BoolInput("Balanced Tiles", default=False)
            .with_id(5)
            .with_docs(
                "For an image larger than the engine: the same number of tiles, spread evenly and,"
                " on a Dynamic engine, each only as large as the image needs instead of the maximum"
                " size, so the GPU computes fewer pixels. A 480x270 image on engines of up to 256 px,"
                " RTX 5090: 4x-UltraSharp 89 to 61 ms, DUAL Light 30 to 20 ms. Tile sizes stay"
                " multiples of 64 px on all engines but DUAL's and DRFT's, so some images gain less"
                " there: with 4x-UltraSharp, 512x288 takes 94 to 85 ms, and 512x512 stays the same.",
                "The output changes a little (off keeps it bit-identical): seams still overlap by at"
                " least 48 px, and a tile sees less of the image, which costs DUAL about 1 to 6 dB"
                " against the whole image in PyTorch and ESRGAN up to about 2 dB.",
                "A Fixed engine has one size, so its speed stays the same. An image that fits the"
                " engine runs as one tile either way, which is faster still: a Dynamic engine large"
                " enough for 512x512 takes 61 ms against 141 ms in 256 px tiles, for four times the"
                " GPU memory (1.3 GiB against 0.3 GiB of activations).",
            ),
        ],
        outputs=[
            ImageOutput(
                "Image",
                image_type="convenientUpscaleTrt(Input0, Input1)",
            ),
            TextOutput("Tile Size").with_docs(
                "The size of the tiles the engine ran, or Whole image. For an image that takes"
                " several passes (transparency, Separate Alpha), the last pass's."
            ),
        ],
        name="Upscale Image",
        icon="BsNvidia",
        node_context=True,
        temporal_context=(WARM_UP, LOOK_AHEAD),
        temporal_if=(0, context_applies),
    )
    def upscale_image_node(
        context: NodeContext,
        img: np.ndarray,
        engine: TensorRTEngine,
        separate_alpha: bool,
        balanced_tiles: bool,
    ) -> tuple[np.ndarray, str]:
        return engine_upscale(context, img, engine, separate_alpha, balanced_tiles)
