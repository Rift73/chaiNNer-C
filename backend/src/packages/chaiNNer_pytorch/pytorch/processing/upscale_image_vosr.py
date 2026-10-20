from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np
import torch
from sanic.log import logger

from api import NodeContext
from nodes.groups import Condition, if_group
from nodes.impl.pytorch.resource_cache import allow_model_reuse
from nodes.impl.pytorch.utils import safe_cuda_cache_empty
from nodes.impl.pytorch.vosr2 import (
    BundlePaths,
    VosrBackend,
    VosrColorCorrection,
    VosrMemoryMode,
    prepare_image,
)
from nodes.impl.pytorch.vosr2_cache import lease_bundle, release_bundle
from nodes.impl.pytorch.vosr2_context import upscale_scene_focus
from nodes.impl.pytorch.vosr2_download import ensure_face_model, ensure_model
from nodes.impl.pytorch.vosr2_engine_cache import ensure_full_engine
from nodes.impl.pytorch.vosr2_fp8 import ensure_fp8_engine
from nodes.impl.pytorch.vosr2_memory import vram_budget
from nodes.impl.pytorch.vosr2_scene import WindowPlan
from nodes.impl.pytorch.vosr2_temporal import TemporalMode
from nodes.impl.pytorch.vosr2_temporal_step import TemporalStep, frames_ahead
from nodes.impl.pytorch.vosr2_texture import Grain, restore_with_texture
from nodes.impl.pytorch.vosr2_trt_pipeline import (
    lease_full_session,
    release_full_session,
)
from nodes.impl.pytorch.vosr2_trt_restore import LOW_MEMORY, LatentStatistics, restore
from nodes.properties.inputs import (
    BoolInput,
    EnumInput,
    ImageInput,
    NumberInput,
    SeedInput,
)
from nodes.properties.outputs import ImageOutput
from nodes.utils.seed import Seed

from ...settings import get_settings
from .. import processing_group

LOAD_VIDEO = "chainner:image:load_video"
TEMPORAL_METHODS = {
    TemporalMode.THREE_FRAMES: "Three Frames",
    TemporalMode.PERSISTENT_DETAIL: "Persistent Detail",
}


@dataclass
class _VideoState:
    """The node's state in a Load Video loop run: the temporal step, the last frame (its
    index and input), with which the grain changes only where the picture does, and the
    shot's window layout, so that seams stay put within a shot."""

    windows: WindowPlan
    step: TemporalStep | None = None
    frame: tuple[int, np.ndarray] | None = None


@processing_group.register(
    schema_id="chainner:pytorch:upscale_image_vosr",
    name="Upscale Image (VOSR)",
    description=[
        "Upscales an RGB image using a VOSR 2.0 generative super-resolution model.",
        "The image is resized to the chosen scale and restored in overlapping windows that share scene context, so focus and fine texture stay consistent; the seed selects the generated detail.",
        "The first use downloads the official VOSR 2.0 model (about 7 GB) from Hugging Face and OpenCV's YuNet face detector (230 KB, which gives a face split between windows a window of its own) from GitHub into the backend's storage folder; later runs reuse them. The TensorRT backends compile their engines on first use and reuse them afterwards.",
        "Inside a Load Video loop, each frame is also stabilized over time: the restorations of neighbouring frames are aligned with optical flow and combined only where they match reliably, and a detected cut starts afresh.",
        "When the frames ahead cannot be read early (for example, through a node between Load Video and this one that is not computed ahead), only the current and earlier frames are used: Three Frames combines the previous frame alone and Persistent Detail starts each scene from its first frame alone.",
    ],
    icon="PyTorch",
    inputs=[
        ImageInput(channels=3),
        NumberInput("Scale", min=1, max=8, default=4, unit="x"),
        SeedInput(),
        EnumInput(
            VosrColorCorrection,
            "Color Correction",
            default=VosrColorCorrection.WAVELET,
            option_labels={VosrColorCorrection.ADAIN: "AdaIN"},
        ),
        EnumInput(
            VosrMemoryMode,
            "GPU Memory",
            default=VosrMemoryMode.DEFAULT,
            option_labels={VosrMemoryMode.LOW: "Low (under 7 GB)"},
        ).with_docs(
            "Low keeps total GPU memory use, including other programs, under 7 GB, and is slower: PyTorch streams the model weights; TensorRT streams the part of its weights that does not fit and restores images larger than 512 pixels a side in overlapping tiles.",
            hint=True,
        ),
        EnumInput(
            VosrBackend,
            "Backend",
            default=VosrBackend.PYTORCH,
            option_labels={
                VosrBackend.PYTORCH: "PyTorch",
                VosrBackend.TENSORRT: "TensorRT",
                VosrBackend.TENSORRT_FP8: "TensorRT FP8",
            },
        ).with_docs(
            "TensorRT needs an NVIDIA GPU and the TensorRT package. TensorRT FP8 is faster and can slightly change fine detail; it needs an RTX 40 series or newer GPU and the official VOSR 2.0 weights, and downloads its calibrated graphs (about 3.5 GB) on first use.",
            hint=True,
        ),
        if_group(Condition.iterated(0, LOAD_VIDEO))(
            EnumInput(
                TemporalMode,
                "Temporal Method",
                default=TemporalMode.THREE_FRAMES,
                option_labels=TEMPORAL_METHODS,
            )
            .with_id(6)
            .with_docs(
                "Three Frames combines each frame with the previous and the next one. Persistent Detail also carries stable fine texture from earlier frames and starts each scene from up to five frames, so the first output of a scene takes longer.",
                hint=True,
            ),
            NumberInput("Temporal Strength", min=0, max=100, default=65, unit="%")
            .with_id(7)
            .with_docs(
                "0 % upscales every frame on its own. Keep the same seed for a whole video.",
                hint=True,
            ),
        ),
        BoolInput("Texture Fix", default=True)
        .with_id(8)
        .with_docs(
            "Gives flat grainy areas one fine grain across the restoration windows, as strong as the input's grain and in its colours; in a video, still areas keep one grain. Off keeps the model's own output there.",
            hint=True,
        ),
    ],
    outputs=[
        ImageOutput(
            image_type="Image { width: Input0.width * Input1, height: Input0.height * Input1 }",
            channels=3,
        )
    ],
    limited_to_8bpc=True,
    node_context=True,
    # Three Frames needs the frame before; Persistent Detail starts a scene from up to
    # five frames ahead.
    temporal_context=(1, max(frames_ahead(method) for method in TemporalMode)),
)
def upscale_image_vosr_node(
    context: NodeContext,
    img: np.ndarray,
    scale: int,
    seed: Seed,
    color: VosrColorCorrection,
    memory: VosrMemoryMode,
    backend: VosrBackend,
    method: TemporalMode,
    strength: int,
    texture: bool,
) -> np.ndarray:
    device = get_settings(context).device
    if backend != VosrBackend.PYTORCH:
        if device.type != "cuda":
            raise ValueError(
                "VOSR TensorRT requires an NVIDIA GPU. Turn off CPU mode in the PyTorch settings or select the PyTorch backend."
            )
        if find_spec("tensorrt") is None:
            raise ValueError(
                "VOSR TensorRT requires the TensorRT package. Install TensorRT in the dependency manager or select the PyTorch backend."
            )
    if device.type == "cuda":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError(
                    "VOSR 2.0 requires a GPU with BF16 support. Turn on CPU mode in the PyTorch settings to run it on the CPU."
                )
    model = ensure_model(context)
    faces = ensure_face_model(context)
    view = context.iteration
    # Without Texture Fix there is no grain pass.
    grain = Grain(seed.to_u32()) if texture else None
    state = None
    if view is not None and view.iterated_by(0) == LOAD_VIDEO:
        state = view.state
        if not isinstance(state, _VideoState):
            state = _VideoState(windows=WindowPlan(faces))
            view.state = state
        last = state.frame
        if (
            grain is not None
            and last is not None
            and last[0] == view.index - 1
            and last[1].shape == img.shape
        ):
            grain = Grain(seed.to_u32(), view.index, last[1])
        state.frame = (view.index, img)

    # A video keeps one plan (one layout per shot); an image gets its own.
    windows = WindowPlan(faces) if state is None else state.windows

    def upscale(
        image: np.ndarray, restore_context: NodeContext, grain: Grain | None = None
    ) -> np.ndarray:
        # A video's restorations run one at a time, in frame order.
        windows.begin_frame(image)
        return _upscale(
            restore_context,
            image,
            model,
            scale,
            seed,
            color,
            memory,
            backend,
            grain,
            windows,
        )

    if strength == 0 or view is None or state is None:
        return upscale(img, context, grain)
    inputs = (model, scale, seed, color, memory, backend, method, strength)
    step = state.step
    if step is None:
        lookahead = view.lookahead(0)
        step = TemporalStep(
            method, strength / 100, lookahead, backend != VosrBackend.PYTORCH, inputs
        )
        state.step = step
        if not step.ahead:
            logger.info(
                "VOSR temporal: %s needs %d frames ahead, the item window serves %d; using past frames only",
                TEMPORAL_METHODS[method],
                frames_ahead(method),
                lookahead,
            )
    # The grain comes last: stabilization would blend each frame's grain with its
    # neighbours'.
    return restore_with_texture(
        lambda: step.frame(
            context,
            view.index,
            img,
            lambda offset: view.peek(0, offset),
            upscale,
            inputs,
        ),
        img,
        scale,
        grain,
        context.suspend,
        device,
    )


def _upscale(
    context: NodeContext,
    img: np.ndarray,
    model: BundlePaths,
    scale: int,
    seed: Seed,
    color: VosrColorCorrection,
    memory: VosrMemoryMode,
    backend: VosrBackend,
    grain: Grain | None,
    windows: WindowPlan,
) -> np.ndarray:
    """One image's restoration, the device and backend already checked; with `grain`,
    also its grain. `windows` plans the window layout (a video's keeps each shot's)."""
    settings = get_settings(context)
    device = settings.device
    # BF16 with FP32 VAE (and the engines' FP32 exceptions); the CPU uses FP32.
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    reuse = allow_model_reuse(
        device.type, settings.budget_limit, settings.force_cache_wipe
    )
    engine = None
    if backend == VosrBackend.TENSORRT_FP8:
        engine = ensure_fp8_engine(context, model, device)
    elif backend == VosrBackend.TENSORRT:
        engine = ensure_full_engine(context, model, device)
    else:
        release_full_session(context.node_id)
    # After a possible engine build (minutes): the resized image is not held meanwhile.
    image, reference = prepare_image(img, scale)
    context.add_cleanup(
        safe_cuda_cache_empty, after="node" if settings.force_cache_wipe else "chain"
    )
    if engine is not None:
        low = LOW_MEMORY if memory == VosrMemoryMode.LOW else None
        with lease_full_session(context, engine, device, reuse=reuse) as session:
            release_bundle(context.node_id)
            statistics = LatentStatistics.load(model)
            with vram_budget(
                device,
                memory,
                context,
                settings.budget_limit,
                prepare=lambda: session.prepare(None if low is None else low.reserve),
            ) as check:
                return restore_with_texture(
                    lambda: restore(
                        statistics,
                        image,
                        reference,
                        seed.to_u32(),
                        color,
                        context,
                        check,
                        session,
                        windows=windows,
                        low=low,
                    ),
                    img,
                    scale,
                    grain,
                    check,
                    device,
                )
    with lease_bundle(context, model, dtype, reuse=reuse) as bundle:
        with vram_budget(device, memory, context, settings.budget_limit) as check:
            return restore_with_texture(
                lambda: upscale_scene_focus(
                    bundle,
                    image,
                    reference,
                    device,
                    seed.to_u32(),
                    memory,
                    color,
                    context,
                    check,
                    optimized=device.type == "cuda",
                    windows=windows,
                ),
                img,
                scale,
                grain,
                check,
                device,
            )
