"""Readies the GPU for a model in the background after its Load Model preview, so that the
first run does not wait for it.

The GPU driver readies each kernel the first time a process uses it: from its cache on disk,
or, when the cache lacks it, in seconds. On an RTX 5090, a HAT's first upscale takes 16 s more
than its next one with nothing cached, and 1 s more with everything cached. A tiny image uses
the same kernels as any other, whatever its size and the model's weights.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from sanic.log import logger
from spandrel import ImageModelDescriptor, ModelDescriptor

from api import Progress

from .. import background_work
from ..upscale.tiler import NoTiling
from .auto_split import pytorch_auto_split
from .utils import safe_cuda_cache_empty

# Upscale Image's models
_PURPOSES = frozenset({"SR", "Restoration"})
_SIZE = 64


class _NeverStopped(Progress):
    @property
    def aborted(self) -> bool:
        return False

    @property
    def paused(self) -> bool:
        return False

    def set_progress(self, progress: float) -> None:
        pass


def warm_up_in_background(
    model: ModelDescriptor, device: torch.device, use_fp16: bool
) -> None:
    """On a GPU, upscales a tiny image with `model` in the background. `use_fp16` is the
    setting: Upscale Image's precision follows from it and the model."""
    if (
        device.type == "cuda"
        and isinstance(model, ImageModelDescriptor)
        and model.purpose in _PURPOSES
    ):
        background_work.start(
            "GPU warm-up",
            lambda: tiny_upscale(model, device, use_fp16 and model.supports_half),
        )


def tiny_upscale(
    model: ImageModelDescriptor[torch.nn.Module], device: torch.device, use_fp16: bool
) -> None:
    start = time.perf_counter()
    try:
        pytorch_auto_split(
            np.full((_SIZE, _SIZE, model.input_channels), 0.5, np.float32),
            model=model,
            device=device,
            use_fp16=use_fp16,
            tiler=NoTiling(),
            progress=_NeverStopped(),
        )
    finally:
        safe_cuda_cache_empty()
    logger.info(
        f"GPU warm-up of the {model.architecture.name} model finished in"
        f" {(time.perf_counter() - start) * 1000:.0f} ms"
    )
