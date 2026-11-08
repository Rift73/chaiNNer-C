"""A process's first VOSR restoration: the GPU kernels of its small steps are readied while
its models load.

The GPU driver readies each kernel the first time a process uses it, which took the grain
pass's few dozen kernels 0.45 s against 0.05 s afterwards (RTX 5090, cache warm).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import torch
from torch.nn import functional

from .. import background_work
from .vosr2_texture import Grain, TextureGuide, _add_grain
from .vosr2_trt_pipeline import FullSession
from .vosr2_trt_restore import gpu_colour

_readied: set[torch.device] = set()
_SIZE = 32


def ready_small_steps(device: torch.device, session: bool) -> None:
    """Once per device, on a thread. `session`: the run also uses a FullSession's steps."""
    if device.type == "cuda" and device not in _readied:
        _readied.add(device)
        background_work.start("VOSR GPU warm-up", lambda: _ready(device, session))


@torch.inference_mode()
def _ready(device: torch.device, session: bool) -> None:
    field = np.full((_SIZE, _SIZE), 0.5, np.float32)
    image = np.full((_SIZE, _SIZE, 3), 0.5, np.float32)
    guide = TextureGuide(field, field, np.eye(3, dtype=np.float32), field)
    _add_grain(image, 1, guide, Grain(0, 1), lambda: None, device)
    if not session:
        return
    tensor = torch.from_numpy(image).to(device).permute(2, 0, 1)[None]
    constants: dict[str, torch.Tensor] = {}

    def constant(key: str, value: torch.Tensor) -> torch.Tensor:
        return constants.setdefault(key, value.to(device))

    gpu_colour(
        cast(
            FullSession,
            SimpleNamespace(device=device, constants=constants, constant=constant),
        ),
        tensor * 2 - 1,
        image,
    )
    conditioning = functional.interpolate(
        tensor, size=(448, 448), mode="bicubic", align_corners=False
    ).clamp_(0, 1)
    torch.isfinite(conditioning).all()
    torch.cuda.synchronize(device)
