from __future__ import annotations

import numpy as np
import torch
from sanic.log import logger

from api import NodeContext
from nodes.impl.pytorch.resource_cache import allow_model_reuse, lease_resource
from nodes.impl.pytorch.temporal_fix import (
    RADIUS,
    FrameWindow,
    build_model,
    load_state,
    uses_fp16,
)
from nodes.properties.inputs import ImageInput, NumberInput
from nodes.properties.outputs import ImageOutput

from ...settings import get_settings
from .. import processing_group


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
    ],
    outputs=[ImageOutput(image_type="Input0", channels=3)],
    node_context=True,
)
def temporal_fix_node(
    context: NodeContext, img: np.ndarray, strength: float
) -> np.ndarray:
    view = context.iteration
    if strength == 0 or view is None:
        return img
    settings = get_settings(context)
    device = settings.device
    dtype = torch.float16 if uses_fp16(device) else torch.float32
    h, w = img.shape[:2]
    hw = (h + h % 2, w + w % 2)

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

    state = view.state
    if not isinstance(state, FrameWindow) or state.shape != img.shape:
        lookahead = view.lookahead(0)
        state = FrameWindow[torch.Tensor](lookahead >= RADIUS, img.shape)
        view.state = state
        if not state.ahead:
            logger.info(
                "Temporal Fix needs %d frames ahead, the item window serves %d; "
                "the full window is unavailable in this chain, so the future frames "
                "mirror the past ones",
                RADIUS,
                lookahead,
            )
    folder = context.storage_dir / "temporalfix"
    with lease_resource(
        context,
        lambda: (str(device), round(float(strength), 2), hw),
        lambda: (build_model(load_state(folder, strength), hw, device), True),
        reuse=allow_model_reuse(
            device.type, settings.budget_limit, settings.force_cache_wipe
        ),
    ) as model, torch.inference_mode():
        frames = state.gather(view.index, lambda: convert(img), peek)
        result = model(torch.stack(frames)[None])[0, :, :h, :w]
        return np.ascontiguousarray(
            result.float().flip(0).permute(1, 2, 0).cpu().numpy()
        )
