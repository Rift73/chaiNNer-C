from __future__ import annotations

import math
import weakref

import numpy as np
import psutil
import torch
from sanic.log import logger
from spandrel import ImageModelDescriptor, ModelTiling

from api import KeyInfo, NodeContext, Progress
from gpu import nvidia
from nodes.groups import Condition, if_enum_group, if_group
from nodes.impl.image_op import ImageOp
from nodes.impl.pytorch.auto_engine import (
    ENGINE_MODEL_NAVI,
    migraphx_here,
    release_engine,
    upscale_through_engine,
)
from nodes.impl.pytorch.auto_split import pytorch_auto_split
from nodes.impl.pytorch.utils import safe_cuda_cache_empty
from nodes.impl.upscale.auto_split_tiles import (
    CUSTOM,
    NO_TILING,
    TILE_SIZE_256,
    TileSize,
    estimate_tile_size,
    parse_tile_size_input,
)
from nodes.impl.upscale.basic_upscale import PaddingType, UpscaleInfo, basic_upscale
from nodes.impl.upscale.tile_report import TileReport
from nodes.impl.upscale.tiler import MaxTileSize
from nodes.properties.inputs import (
    BoolInput,
    ImageInput,
    NumberInput,
    PaddingTypeInput,
    SrModelInput,
    TileSizeDropdown,
)
from nodes.properties.outputs import ImageOutput, TextOutput

from ...settings import PyTorchSettings, get_settings
from .. import processing_group

MODEL_BYTES_CACHE = weakref.WeakKeyDictionary()
# Auto's GPU memory per pixel of the image, in FP16, for the architectures measured whole,
# with the width it was measured at: a model of another width takes it in proportion, FP32
# twice. Upstream's estimate goes by the parameters alone, and split a 4x HAT-L's 1200x864
# image in two of a 19.1 GiB budget, though the whole image takes 10.9 GiB; DAT, HAT and
# DUAL pool and normalize over the whole tile, so the split read 48.0 dB against strict
# FP32 (142 codes off at the image's edge) where the whole image reads 70.6 dB, in 2.01 s
# instead of 2.12 s on an RTX 5090. Measured peaks: HAT-L 11.0 KiB, DAT2 5.3 KiB (4x, 180
# channels), DUAL Light 2.4 KiB (2x, 128 channels).
WHOLE_IMAGE_KIB_PER_PIXEL = {"HAT": (11.0, 180), "DAT": (5.3, 180), "DUAL": (2.4, 128)}
# The same input (id 8) on a PC where MIGraphX runs, so a saved chain keeps its value on
# either kind of PC; only a PC with neither has none.
ON_MIGRAPHX = migraphx_here()


def whole_image_tiler(
    model: ImageModelDescriptor, img: np.ndarray, budget: int, use_fp16: bool
) -> MaxTileSize:
    """Auto's tiles for a model WHOLE_IMAGE_KIB_PER_PIXEL lists, on a GPU with `budget`
    bytes: the whole image if it fits, else the largest square that does."""
    kib, width = WHOLE_IMAGE_KIB_PER_PIXEL[model.architecture.name]
    per_pixel = 1024 * kib * model.model.embed_dim / width * (1 if use_fp16 else 2)
    h, w = img.shape[:2]
    if h * w * per_pixel <= budget:
        return MaxTileSize(max(h, w))
    return MaxTileSize(max(1, math.isqrt(int(budget / per_pixel))))


def upscale(
    img: np.ndarray,
    model: ImageModelDescriptor,
    tile_size: TileSize,
    options: PyTorchSettings,
    progress: Progress,
    report: TileReport | None = None,
):
    with torch.no_grad():
        # Borrowed from iNNfer
        logger.debug("Upscaling image")

        # TODO: use bfloat16 if RTX
        use_fp16 = options.use_fp16 and model.supports_half
        device = options.device

        if model.tiling == ModelTiling.INTERNAL:
            # disable tiling if the model already does it internally
            tile_size = NO_TILING

        def estimate():
            model_bytes = MODEL_BYTES_CACHE.get(model)
            if model_bytes is None:
                model_bytes = sum(p.numel() * 4 for p in model.model.parameters())
                MODEL_BYTES_CACHE[model] = model_bytes

            if "cuda" in device.type:
                if options.use_fp16:
                    model_bytes = model_bytes // 2
                mem_info: tuple[int, int] = torch.cuda.mem_get_info(device)  # type: ignore
                _free, total = mem_info
                # only use 75% of the total memory
                total = int(total * 0.75)
                if options.budget_limit > 0:
                    total = min(options.budget_limit * 1024**3, total)
                # Estimate using 80% of the value to be more conservative
                budget = int(total * 0.8)

                if model.architecture.name in WHOLE_IMAGE_KIB_PER_PIXEL:
                    return whole_image_tiler(model, img, budget, use_fp16)

                return MaxTileSize(
                    estimate_tile_size(
                        budget,
                        model_bytes,
                        img,
                        2 if use_fp16 else 4,
                    )
                )
            elif device.type == "cpu":
                free = psutil.virtual_memory().available
                if options.budget_limit > 0:
                    free = min(options.budget_limit * 1024**3, free)
                budget = int(free * 0.8)
                return MaxTileSize(
                    estimate_tile_size(
                        budget,
                        model_bytes,
                        img,
                        4,
                    )
                )
            return MaxTileSize()

        img_out = pytorch_auto_split(
            img,
            model=model,
            device=device,
            use_fp16=use_fp16,
            tiler=parse_tile_size_input(tile_size, estimate),
            progress=progress,
            use_bf16=options.use_cpu_bf16 and model.supports_bfloat16,
            report=report,
        )
        logger.debug("Done upscaling")

        return img_out


@processing_group.register(
    schema_id="chainner:pytorch:upscale_image",
    name="Upscale Image",
    description=(
        "Upscales an image using a PyTorch Super-Resolution model. Select a"
        " manual number of tiles if you are having issues with the automatic mode. "
    ),
    icon="PyTorch",
    inputs=[
        ImageInput().with_id(1),
        SrModelInput().with_id(0),
        if_group(
            Condition.type(0, "PyTorchModel { scale: int(2..) }", if_not_connected=True)
            & (
                Condition.type(
                    0,
                    "PyTorchModel { inputChannels: 1, outputChannels: 1 }",
                    if_not_connected=True,
                )
                | Condition.type(
                    0, "PyTorchModel { inputChannels: 3, outputChannels: 3 }"
                )
                | Condition.type(
                    0, "PyTorchModel { inputChannels: 4, outputChannels: 4 }"
                )
            )
        )(
            BoolInput("Custom Scale", default=False)
            .with_id(4)
            .with_docs(
                "If enabled, the scale factor can be manually set. This makes it possible to e.g. upscale 4x with a 2x model.",
                "Custom scales are **not** supported for 1x models and colorization models.",
                "Under the hood, this will repeatedly apply the model to the image, effectively upscaling by the given factor."
                " E.g. if the model is 2x and the desired scale is 4x, the model will be applied 2 times."
                " If the desired scale cannot be reached exactly, the image will be downscaled to the desired scale after upscaling."
                " E.g. if the model is 2x and the desired scale is 6x, the model will be applied 3 times (8x) and the image will be downscaled to 6x.",
                "If the desired scale is less than the model's scale, the image will be downscaled to the desired scale after upscaling.",
                hint=True,
            ),
            if_group(Condition.bool(4, True))(
                NumberInput(
                    "Scale", default=4, min=1, max=32, label_style="hidden"
                ).with_id(5),
            ),
        ),
        if_group(
            Condition.type(
                0,
                "PyTorchModel { tiling: ModelTiling::Supported | ModelTiling::Discouraged } ",
                if_not_connected=True,
            )
        )(
            TileSizeDropdown()
            .with_id(2)
            .with_docs(
                "Tiled upscaling is used to allow large images to be upscaled without"
                " hitting memory limits.",
                "This works by splitting the image into tiles (with overlap), upscaling"
                " each tile individually, and seamlessly recombining them.",
                "Generally it's recommended to use the largest tile size possible for"
                " best performance (with the ideal scenario being no tiling at all),"
                " but depending on the model and image size, this may not be possible.",
                "If you are having issues with the automatic mode, you can manually"
                " select a tile size. Sometimes, a manually selected tile size may be"
                " faster than what the automatic mode picks.",
                hint=True,
            ),
            if_enum_group(2, CUSTOM)(
                NumberInput(
                    "Custom Tile Size",
                    min=1,
                    max=None,
                    default=TILE_SIZE_256,
                    unit="px",
                ).with_id(6),
            ),
        ),
        PaddingTypeInput().with_id(7),
        if_group(
            Condition.type(1, "Image { channels: 4 } ")
            & (
                Condition.type(
                    0, "PyTorchModel { inputChannels: 1, outputChannels: 1 }"
                )
                | Condition.type(
                    0, "PyTorchModel { inputChannels: 3, outputChannels: 3 }"
                )
            )
        )(
            BoolInput("Separate Alpha", default=False)
            .with_id(3)
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
        if_group(
            Condition.const(nvidia.is_available or ON_MIGRAPHX)
            & Condition.type(0, ENGINE_MODEL_NAVI, if_not_connected=True)
        )(
            BoolInput("Use MIGraphX Program", default=False)
            .with_id(8)
            .with_docs(
                "Runs this model through a MIGraphX program instead of PyTorch, on AMD GPUs.",
                "The first run for a model at a Tile Size exports the model and compiles its"
                " program, which can take several minutes; later runs read it, also after a"
                " restart. Stop ends a running compile.",
                "While this is on, programs take disk space, whether MIGraphX's Program Cache"
                " setting is on or off: in that cache's folder and within its limit of 16 GiB,"
                " the least recently used removed first. While this is off, this node keeps"
                " nothing on disk.",
                "A program takes one input size, so each Tile Size has its own program: that"
                " size, aligned up to the multiple the model needs. Auto, Maximum and No Tiling"
                " take 512 px, and RGT stops at 256 px. A larger image runs in overlapping tiles"
                " of that size. A smaller one is padded to it by reflection and runs through the"
                " program, and the result is cropped to the image's own size.",
                "The output differs slightly from PyTorch's: the program runs in the model's"
                " automatic precision, whose FP16 list was measured on TensorRT, and its tiling"
                " and padding are its own. The node's Clear frees its GPU memory.",
                "When the program cannot be built or fails, the image runs on PyTorch and a"
                " warning says why.",
                hint=True,
            )
            if ON_MIGRAPHX
            else BoolInput("Use TensorRT Engine", default=False)
            .with_id(8)
            .with_docs(
                "Runs this model through a TensorRT engine instead of PyTorch, on NVIDIA GPUs:"
                " on a 4x ESRGAN at 512 px, about twice as fast.",
                "The first run for a model at a Tile Size exports and builds its engine (about a"
                " minute for a 4x ESRGAN, longer for larger models); later runs read it, also after"
                " a restart. Stop waits for a running build to end.",
                "While this is on, engines take disk space, whether TensorRT's Engine Cache"
                " setting is on or off: about 43 MB for a 4x ESRGAN, in that cache's folder and"
                " within its limit of 16 GiB, the least recently used removed first. While this"
                " is off, this node keeps nothing on disk.",
                "The output differs slightly from PyTorch's: the engine runs in the model's"
                " automatic precision, in overlapping tiles up to the Tile Size. Auto, Maximum"
                " and No Tiling take 512 px; on DAT, HAT, DUAL and FIGSR, whose results change"
                " with what a tile sees, they take 1408 px (DUAL up to 1920 px, less on wider"
                " models) on a GPU of 16 GB or more, so that an image up to that size runs whole,"
                " and one that does not fit the GPU's memory runs in smaller tiles. RGT stops at"
                " 256 px. The node's Clear frees its GPU memory.",
                "When the engine cannot be built or fails, the image runs on PyTorch and a warning"
                " says why.",
                hint=True,
            ),
        ),
    ],
    outputs=[
        ImageOutput(
            "Image",
            image_type="""
                let img = Input1;
                let model = Input0;
                let useCustomScale = Input4;
                let customScale = Input5;

                let singleUpscale = convenientUpscale(model, img);

                if useCustomScale and model.scale >= 2 and model.inputChannels == model.outputChannels {
                    Image {
                        width: img.width * customScale,
                        height: img.height * customScale,
                        channels: singleUpscale.channels,
                    }
                } else {
                    singleUpscale
                }
            """,
            assume_normalized=True,  # pytorch_auto_split already does clipping internally
        ),
        TextOutput("Tile Size").with_docs(
            "The size of the tiles the model ran, with their overlap and after any cut for"
            " lack of memory, or Whole image. For an image that takes several passes"
            " (transparency, Separate Alpha, Custom Scale), the last pass's."
        ),
    ],
    key_info=KeyInfo.type(
        """
        let model = Input0;
        let useCustomScale = Input4;
        let customScale = Input5;

        let singleUpscale = convenientUpscale(model, img);

        let scale = if useCustomScale and model.scale >= 2 and model.inputChannels == model.outputChannels {
            customScale
        } else {
            model.scale
        };

        string::concat(toString(scale), "x")
        """
    ),
    node_context=True,
)
def upscale_image_node(
    context: NodeContext,
    img: np.ndarray,
    model: ImageModelDescriptor,
    use_custom_scale: bool,
    custom_scale: int,
    tile_size: TileSize,
    custom_tile_size: int,
    padding: PaddingType,
    separate_alpha: bool,
    use_engine: bool,
) -> tuple[np.ndarray, str]:
    exec_options = get_settings(context)

    context.add_cleanup(
        safe_cuda_cache_empty,
        after="node" if exec_options.force_cache_wipe else "chain",
    )

    info = UpscaleInfo(
        in_nc=model.input_channels, out_nc=model.output_channels, scale=model.scale
    )
    if not use_custom_scale or not info.supports_custom_scale:
        custom_scale = model.scale

    report = TileReport()

    def upscale_with(op: ImageOp) -> np.ndarray:
        return basic_upscale(
            img,
            op,
            upscale_info=info,
            scale=custom_scale,
            separate_alpha=separate_alpha,
            padding=padding,
            clip=False,  # pytorch_auto_split and the engine's node already clip
        )

    size = TileSize(custom_tile_size) if tile_size == CUSTOM else tile_size

    def pytorch(i: np.ndarray) -> np.ndarray:
        return upscale(
            i,
            model,
            size,
            exec_options,
            context,
            report,
        )

    if use_engine:
        image = upscale_through_engine(
            context,
            model,
            exec_options.device,
            exec_options.gpu_index,
            size,
            upscale_with,
            pytorch,
            report,
        )
    else:
        release_engine(context.node_id)
        image = upscale_with(pytorch)
    return image, report.text
