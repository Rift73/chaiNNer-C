"""The PyTorch executor of Upscale Image's temporal path (`temporal_window.Executor`) for
TFDAT and TFDATv2: each output is one call of the model on the clip of `clip_size`
frames centred on its frame, eager, in FP16 or FP32 as the node runs any model.

The model carries nothing from frame to frame: the state a window hands the next is the
frames before that window's first, which its first clips read. Where a clip reaches past
the frames there are (a video's first and last frames, and an image outside a loop,
which is a window of one frame), the missing frames repeat the nearest one: the trainer
trains on whole clips only and defines no edge, and a repeated frame adds no motion and
no content the clip does not have."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import torch
from spandrel import ImageModelDescriptor

from .utils import Output, host_image, on_device, to_host, upload_frame


class TFDATWindows:
    """Frames are RGB CHW tensors on the device in the model's dtype, outputs `Output`s,
    each on its way to the host as soon as its clip ran (in page-locked memory on a GPU),
    and the state is the up to `look_ahead` frames that end at the window's last kept
    one. The model runs as a copy on the device in FP16 or FP32 (`on_device`): the
    loaded model, which other nodes share, is not changed."""

    def __init__(
        self,
        model: ImageModelDescriptor,
        device: torch.device,
        use_fp16: bool = False,
    ):
        self.model = on_device(
            model, device, torch.float16 if use_fp16 else torch.float32
        )
        self.network = self.model.model
        self.device = device
        # The frames on each side of a clip's centre: the frames after a kept one that
        # its output reads, for the planner, and the frames before it, which the state
        # carries.
        self.look_ahead: int = self.network.clip_size // 2

    @property
    def key(self) -> Hashable:
        return ("pytorch", id(self.network), str(self.device))

    def convert(self, frame: np.ndarray) -> torch.Tensor:
        # BGR HWC -> RGB CHW in 0..1 on the model's device, in the model's dtype.
        return upload_frame(frame, self.device, self.model.dtype)

    def run(
        self,
        frames: Sequence[torch.Tensor],
        keep: int,
        past: tuple[tuple[torch.Tensor, ...], torch.Tensor] | None,
        _first: int,
    ) -> tuple[list[Output], tuple[torch.Tensor, ...]]:
        # The carried frames end at the frame just before this window's first, which
        # `past` also names: the index is not needed.
        before = () if past is None else past[0]
        sequence = [*before, *frames]
        last = len(sequence) - 1
        side = self.look_ahead
        outputs: list[Output] = []
        with torch.inference_mode():
            for centre in range(len(before), len(before) + keep):
                clip = torch.stack(
                    [
                        sequence[min(max(centre + k, 0), last)]
                        for k in range(-side, side + 1)
                    ]
                )
                # Each kept frame's copy to the host starts now, behind the next clip.
                outputs.append(to_host(self.network(clip[None])[0]))
        end = len(before) + keep
        return outputs, tuple(sequence[max(0, end - side) : end])

    @staticmethod
    def to_image(output: Output) -> np.ndarray:
        return host_image(output)
