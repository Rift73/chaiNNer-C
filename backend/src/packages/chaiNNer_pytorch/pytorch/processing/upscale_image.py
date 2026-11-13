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
from nodes.impl.pytorch import hat_export, offload
from nodes.impl.pytorch.auto_engine import (
    ENGINE_MODEL_NAVI,
    migraphx_here,
    release_engine,
    upscale_through_engine,
)
from nodes.impl.pytorch.auto_split import pytorch_auto_split
from nodes.impl.pytorch.rdual_engine import temporal_upscale_through_engine
from nodes.impl.pytorch.rdual_window import (
    TILE_OVERLAP,
    RDUALWindows,
    context_applies,
    is_temporal,
    lighter_modes,
    release_memory,
    takes_tiles,
)
from nodes.impl.pytorch.utils import is_out_of_memory, safe_cuda_cache_empty
from nodes.impl.upscale.auto_split_tiles import (
    CUSTOM,
    ESTIMATE,
    NO_TILING,
    TILE_SIZE_256,
    TileSize,
    estimate_tile_size,
    parse_tile_size_input,
)
from nodes.impl.upscale.basic_upscale import PaddingType, UpscaleInfo, basic_upscale
from nodes.impl.upscale.temporal_window import (
    LOOK_AHEAD,
    WARM_UP,
    Fit,
    temporal_upscale,
)
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
# image in two of a 19.1 GiB budget, though the whole image took 10.9 GiB; DAT, HAT and
# DUAL pool and normalize over the whole tile, so the split read 48.0 dB against strict
# FP32 (142 codes off at the image's edge) where the whole image reads 70.6 dB, in 2.01 s
# instead of 2.12 s on an RTX 5090. Measured peaks: DAT2 5.3 KiB (4x, 180 channels), DUAL
# Light 2.4 KiB (2x, 128 channels). HAT-L (4x, 180 channels) peaks at 6.5 KiB at 512x512
# and 5.8-6.0 KiB at 768x768, 1024x1024 and 1344x756 (HAT 6.1 KiB at most; FP32 11.4 KiB,
# under twice FP16) with a spandrel whose HAT frees its overlapping attention's window
# copies before the MLP (11.7 KiB with one that keeps them), and runs only with room
# above its peak for its attention windows:
# with 6.8 KiB, the largest image Auto runs whole on an 8, 12, 16 or 24 GB card, square
# or 16:9, FP16 or FP32, ran under a cap of the card less 1.5 GiB at most 8.3 % slower
# than uncapped. A 4x ESRGAN (64 filters) peaks at 1.15-1.26 KiB from 512x512 to 1920x1080
# with a spandrel whose ESRGAN runs its high-resolution tail in row bands (4.0 KiB with
# one that runs it whole); pixel-unshuffle models (x2plus) take less per input pixel. 4x
# FIGSR (48 channels) 1.64-1.66 KiB at 512x512 and 1024x1024. Their entries are the
# highest figure plus 5 % (rounded up to 0.1), as HAT's. 4x HEART (180 channels), through
# this node in FP16 with a spandrel whose HEART runs in the token layout with its window
# and i-LN kernels and frees qkv before its output projection: 3.21-3.22 KiB from 512x512
# to 1344x1344 (its 184-wide copy), and 5.03-5.29 KiB without Triton (5.29 at 512x512;
# FP32 10.2-10.6 KiB, under twice FP16); its entry is the higher figure plus 5 %, as HAT's,
# so it holds on both paths. A trained 2x GaterV3 (32
# channels) peaks at 1.24 KiB from 256x256 to 1024x1024 with a spandrel that runs its
# upsampler whole, and at 1.12 KiB (up to 512x512) and 1.00 KiB (768x768 to 1344x1344) with
# one that runs it in row bands; FP32 2.0-2.4 KiB. Its entry is the higher figure plus 5 %,
# so it holds with either.
WHOLE_IMAGE_KIB_PER_PIXEL = {
    "HAT": (6.8, 180),
    "DAT": (5.3, 180),
    "DUAL": (2.4, 128),
    "ESRGAN": (1.4, 64),
    "FIGSR": (1.8, 48),
    "HEART": (5.6, 180),
    "GaterV3": (1.4, 32),
}
# The hyperparameter that sets an architecture's width, where it is not embed_dim.
WHOLE_IMAGE_WIDTH = {"ESRGAN": "num_filters", "FIGSR": "dim", "GaterV3": "dim"}
# The same input (id 8) on a PC where MIGraphX runs, so a saved chain keeps its value on
# either kind of PC; only a PC with neither has none.
ON_MIGRAPHX = migraphx_here()
# Auto's error for a temporal model whose lightest whole-frame window does not fit.
AUTO_EXHAUSTED = (
    "The temporal model (RDUAL) does not fit the GPU's memory at this frame size, even in its"
    " lighter memory modes with no frames read ahead and in its smallest tiles. Turn on Use"
    " TensorRT Engine, or use a smaller frame."
)
# A temporal model runs one model pass per frame on RGB: Custom Scale and Separate Alpha
# do not apply to it.
TEMPORAL = Condition.type(0, 'PyTorchModel { arch: "RDUAL" }')
NOT_TEMPORAL = ~TEMPORAL


def whole_image_bytes_per_pixel(model: ImageModelDescriptor, use_fp16: bool) -> float:
    """The peak GPU memory per input pixel WHOLE_IMAGE_KIB_PER_PIXEL gives a model."""
    name = model.architecture.name
    kib, width = WHOLE_IMAGE_KIB_PER_PIXEL[name]
    if name in WHOLE_IMAGE_WIDTH:
        channels = model.model.hyperparameters[WHOLE_IMAGE_WIDTH[name]]
    else:
        channels = model.model.embed_dim
    return 1024 * kib * channels / width * (1 if use_fp16 else 2)


def whole_image_tiler(
    model: ImageModelDescriptor, img: np.ndarray, budget: int, use_fp16: bool
) -> MaxTileSize:
    """Auto's tiles for a model WHOLE_IMAGE_KIB_PER_PIXEL lists, on a GPU with `budget`
    bytes: the whole image if it fits, else the largest square that does."""
    per_pixel = whole_image_bytes_per_pixel(model, use_fp16)
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
        logger.debug("Upscaling image")

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
                mem_info: tuple[int, int] = torch.cuda.mem_get_info(device)
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

        # Auto's tiles of a model with a whole-image estimate fit by that estimate: the
        # activation offload starts from it, so a forward that fits runs as is, with
        # spandrel's kernels, from the first image on.
        h, w = img.shape[:2]
        expected = None
        plain = h * w < offload.MIN_PIXELS
        if (
            tile_size == ESTIMATE
            and device.type == "cuda"
            and model.architecture.name in WHOLE_IMAGE_KIB_PER_PIXEL
        ):
            expected = whole_image_bytes_per_pixel(model, use_fp16)
            tw, th = estimate().starting_tile_size(
                w, h, img.shape[2] if img.ndim == 3 else 1
            )
            plain = offload.runs_plain(device, min(tw, w) * min(th, h), expected)

        img_out = pytorch_auto_split(
            img,
            # HAT's 184-wide eager copy where its kernel runs; the estimate above keeps
            # the loaded model's width.
            model=hat_export.for_eager(model, device, use_fp16, plain),
            device=device,
            use_fp16=use_fp16,
            tiler=parse_tile_size_input(tile_size, estimate),
            progress=progress,
            use_bf16=options.use_cpu_bf16 and model.supports_bfloat16,
            report=report,
            expected_bytes_per_pixel=expected,
        )
        logger.debug("Done upscaling")

        return img_out


@processing_group.register(
    schema_id="chainner:pytorch:upscale_image",
    name="Upscale Image",
    description=[
        (
            "Upscales an image using a PyTorch Super-Resolution model. Select a"
            " manual number of tiles if you are having issues with the automatic mode. "
        ),
        (
            "A temporal model (RDUAL) reads the frames around each one through optical flow"
            " and carries what it saw from frame to frame. Inside a loop such as Load Video or"
            " Load Images, it upscales the frames it can read ahead as one window and keeps the"
            f" outputs of the frames followed by at least {LOOK_AHEAD} more in that window for"
            " the items that follow; at the end of a video it keeps all of them. When no frames"
            " ahead can be read, each frame is upscaled with the frames before it only. When the"
            f" loop starts inside a video (Run this frame, a Range), it also runs the {WARM_UP}"
            " frames before it. Outside a loop, the image is upscaled on its own."
        ),
        (
            "A temporal model upscales the color of an image with transparency and resizes its"
            " alpha (Lanczos); a grayscale image runs as color and comes back as grayscale."
            " Padding pads every frame it reads."
        ),
        (
            "For a temporal model, Tile Size decides how a frame fits the GPU's memory. No"
            " Tiling runs the whole frame, and fails when it does not fit. Maximum runs the"
            " whole frame, and a size runs tiles of that size, each halving when it does not"
            " fit. Auto chooses from an estimate of the window's memory before the first"
            " window: the whole frame with every frame it can read ahead, else the model's"
            " lighter memory modes (a few percent slower), else"
            f" {LOOK_AHEAD + 1} frames ahead and then none (each frame then has only the frames"
            " before it), and only then tiles, halving them. Running out of memory still moves"
            " to the next of these, and a choice is kept for the rest of the loop. In tiles,"
            " the optical flow still sees the whole frame and only the model's trunk runs per"
            f" tile, with {TILE_OVERLAP} px of overlap and its own state per tile; the result"
            " changes slightly, measured against the whole frame in 2x2 tiles at an RMS of"
            " 0.0007 (1488x837) to 0.00095 (1920x1080) on values from 0 to 1."
        ),
    ],
    icon="PyTorch",
    inputs=[
        ImageInput().with_id(1),
        SrModelInput().with_id(0),
        if_group(
            Condition.type(0, "PyTorchModel { scale: int(2..) }", if_not_connected=True)
            & NOT_TEMPORAL
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
            & NOT_TEMPORAL
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
            & (
                Condition.type(0, ENGINE_MODEL_NAVI, if_not_connected=True)
                # RDUAL's engines are TensorRT's only: it stays on PyTorch on AMD GPUs.
                | (Condition.const(not ON_MIGRAPHX) & TEMPORAL)
            )
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
                "For RDUAL (a video model), the engine is a set of four engines for the frame's"
                " size, Padding included. The first frame of a new size exports the model and"
                " builds the set: about two minutes at 1488x837 on an RTX 5090. The set takes"
                " about 51 MB in the Engine Cache's folder, as above, and later runs read it."
                " Frames run whole: Tile Size does not apply to the set, and when the GPU's"
                " memory runs out it reads fewer frames ahead. The output differs slightly"
                " from PyTorch's: the set runs its body in BF16 and its optical flow in mixed"
                " FP16 and FP32. When the set cannot be built or fails, the video goes on with"
                " PyTorch from that frame, and a warning says why.",
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
                let useCustomScale = match model.arch { "RDUAL" => false, _ => Input4 };
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
        let useCustomScale = match model.arch { "RDUAL" => false, _ => Input4 };
        let customScale = Input5;

        let scale = if useCustomScale and model.scale >= 2 and model.inputChannels == model.outputChannels {
            customScale
        } else {
            model.scale
        };

        string::concat(toString(scale), "x")
        """
    ),
    node_context=True,
    temporal_context=(WARM_UP, LOOK_AHEAD),
    temporal_if=(0, context_applies),
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
    if is_temporal(model):
        report = TileReport()
        settings = get_settings(context)
        budget = max(0, settings.budget_limit) * 1024**3

        def windows() -> RDUALWindows:
            # Made only where it runs: it moves the model to the device.
            return RDUALWindows(model, settings.device, budget)

        frame_tiles = TileSize(custom_tile_size) if tile_size == CUSTOM else tile_size
        auto = frame_tiles == ESTIMATE
        fit = Fit(
            parse_tile_size_input(frame_tiles, MaxTileSize),
            shorten=auto,
            out_of_memory=is_out_of_memory,
            release=release_memory,
            lighter=lighter_modes(),
            tiles=takes_tiles(),
            exhausted=AUTO_EXHAUSTED if auto else "",
        )
        if use_engine and not ON_MIGRAPHX:
            image = temporal_upscale_through_engine(
                context,
                img,
                model,
                settings.device,
                windows,
                settings.gpu_index,
                report,
                padding,
                fit,
            )
        else:
            release_engine(context.node_id)
            image = temporal_upscale(context, img, windows(), report, padding, fit)
        return image, report.text

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
