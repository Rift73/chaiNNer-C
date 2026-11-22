from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import numpy as np
import torch
from sanic.log import logger

from api import NodeContext
from gpu import nvidia
from nodes.groups import Condition, if_group
from nodes.impl.pytorch.auto_engine import migraphx_here, release_engine
from nodes.impl.pytorch.resource_cache import allow_model_reuse, lease_resource
from nodes.impl.pytorch.temporal_fix import (
    RADIUS,
    FrameWindow,
    build_model,
    even_size,
    load_state,
    uses_fp16,
)
from nodes.impl.pytorch.temporal_fix_engine import Window, fix_through_engine
from nodes.impl.pytorch.utils import Output, host_image, to_host
from nodes.properties.inputs import BoolInput, ImageInput, NumberInput
from nodes.properties.outputs import ImageOutput

from ...settings import get_settings
from .. import processing_group

# The engine sets are TensorRT's only: on AMD GPUs the node stays on PyTorch.
ON_MIGRAPHX = migraphx_here()


@processing_group.register(
    schema_id="chainner:pytorch:temporal_fix",
    name="Temporal Fix",
    description=[
        "Reduces flicker in a video by stabilizing each frame against the three frames before and after it, using the AI model of vs_temporalfix by pifroggi (Apache-2.0); its weights are downloaded on first use.",
        "It works inside a loop such as Load Video or Load Images; when the frames ahead cannot be read early (for example, after an upscale that is not computed ahead), only the current and earlier frames are used.",
        "Outside a loop, the image is returned unchanged.",
    ],
    icon="PyTorch",
    inputs=[
        ImageInput(channels=3),
        NumberInput(
            "Strength", min=0, max=3, default=2, precision=1, step=0.1
        ).with_docs("0 returns the frames unchanged.", hint=True),
        if_group(Condition.const(nvidia.is_available and not ON_MIGRAPHX))(
            BoolInput("Use TensorRT Engine", default=False)
            .with_id(2)
            .with_docs(
                "Runs the model through a TensorRT engine instead of PyTorch, on NVIDIA"
                " GPUs: at 1920x1080 on an RTX 5090, about twice as fast.",
                "The first frame of a new size and Strength exports the model and builds"
                " its engine: about 40 seconds at 1920x1080 on an RTX 5090. Later runs read"
                " it, also after a restart. Stop waits for a running build to end.",
                "While this is on, engines take disk space, whether TensorRT's Engine Cache"
                " setting is on or off: about 1.3 MB per frame size and Strength, in"
                " that cache's folder and within its limit of 16 GiB, the least recently"
                " used removed first. While this is off, this node keeps nothing on disk.",
                "The output differs slightly from PyTorch's: the engine runs its"
                " convolutions in FP16 and the flow, the warps and the temporal average in"
                " FP32. The node's Clear frees its GPU memory.",
                "When the engine cannot be built or fails, the video goes on with PyTorch"
                " from that frame, and a warning says why.",
                hint=True,
            )
        ),
    ],
    outputs=[ImageOutput(image_type="Input0", channels=3)],
    node_context=True,
    # One frame further than the window: on a GPU each route computes the next item's
    # window while this one's result downloads (FrameWindow.gather_ahead)
    temporal_context=(RADIUS, RADIUS + 1),
)
def temporal_fix_node(
    context: NodeContext, img: np.ndarray, strength: float, use_engine: bool
) -> np.ndarray:
    view = context.iteration
    if strength == 0 or view is None:
        return img
    settings = get_settings(context)
    device = settings.device
    dtype = torch.float16 if uses_fp16(device) else torch.float32
    h, w = img.shape[:2]
    hw = even_size(h, w)

    def convert(frame: np.ndarray) -> torch.Tensor:
        # Clamp to 0..1 and cast like upstream's input conversion; BGR HWC -> RGB CHW.
        tensor = torch.from_numpy(frame).to(device).clamp(0, 1).to(dtype)
        tensor = tensor.permute(2, 0, 1).flip(0)
        if hw != (h, w):
            # Upstream needs even sizes: repeat the last row/column, cropped again below.
            tensor = torch.nn.functional.pad(
                tensor[None], (0, hw[1] - w, 0, hw[0] - h), mode="replicate"
            )[0]
        return tensor

    def peek(offset: int) -> torch.Tensor | None:
        frame = view.peek(0, offset)
        if isinstance(frame, np.ndarray) and frame.shape == img.shape:
            return convert(frame)
        return None

    on_engine = use_engine and not ON_MIGRAPHX
    lookahead = view.lookahead(0)
    # On a GPU each route computes the next item's window while this one's result
    # downloads; it reads one frame further (temporal_context)
    further = device.type == "cuda" and lookahead > RADIUS
    state = view.state
    if (
        not isinstance(state, FrameWindow)
        or state.shape != img.shape
        or state.further != further
    ):
        state = FrameWindow[torch.Tensor](lookahead >= RADIUS, img.shape, further)
        view.state = state
        if not state.ahead:
            logger.info(
                "Temporal Fix needs %d frames ahead, the item window serves %d; "
                "the full window is unavailable in this chain, so the future frames "
                "mirror the past ones",
                RADIUS,
                lookahead,
            )
    # Before any torch mode or lease, and outside the engine route's error handling, on
    # every path: reading ahead can run the nodes before this one on this thread (Guided
    # Upscale trains, which inference mode would break), and an error there is that
    # node's. The window of an item the engine route submitted ahead is the one it kept.
    frames = state.gather(view.index, lambda: convert(img), peek)
    folder = context.storage_dir / "temporalfix"

    def pytorch() -> Output:
        # The split network's parts load with the first window, not with the package
        from nodes.impl.pytorch.temporal_fix_eager import fix_pipelined, fix_window

        reuse = allow_model_reuse(
            device.type, settings.budget_limit, settings.force_cache_wipe
        )
        weights = (str(device), round(float(strength), 2), hw)
        with lease_resource(
            context,
            lambda: weights,
            lambda: (build_model(load_state(folder, strength), hw, device), True),
            reuse=reuse,
        ) as model:

            def window(window_frames: Sequence[torch.Tensor]) -> torch.Tensor:
                # Each frame's encoder and pyramid levels once per frame and weights
                # (LevelCache); CUDA graphs only for a model the lease keeps for the
                # next item
                result = fix_window(model, window_frames, state.levels, reuse, weights)
                return result[0, :, :h, :w]

            if state.further and reuse:
                # The next item's window while this one's result downloads: the model
                # stays leased between items (reuse), and the run's pipeline keeps it
                # until the run's end has waited for the window in flight. Reading the
                # next window runs outside inference mode; the lease holds this node's
                # own entry only, never the nodes a read runs.
                return fix_pipelined(
                    context,
                    state,
                    view.index,
                    frames,
                    partial(state.gather_ahead, view.index, peek),
                    model,
                    window,
                )
            with torch.inference_mode():
                # RGB CHW -> BGR HWC float32 on the device, its values as they are
                # (the output normalizes them), then into page-locked memory on a GPU.
                return to_host(window(frames), clamp=False)

    if on_engine:
        ahead = partial(state.gather_ahead, view.index, peek) if state.further else None
        output = fix_through_engine(
            context,
            Window(state, view.index, frames, ahead),
            strength,
            (h, w),
            lambda: load_state(folder, strength),
            device,
            settings.gpu_index,
            pytorch,
        )
    else:
        release_engine(context.node_id)
        output = pytorch()
    return host_image(output)
