"""The PyTorch executor of Upscale Image's temporal path (`temporal_window.Executor`) for
RSR: its windows through spandrel's `run_window`, eager, in FP16 or FP32 as the node runs
any model."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from spandrel import ImageModelDescriptor

from ..upscale.temporal_window import LOOK_AHEAD
from .utils import (
    Output,
    host_image,
    on_device,
    to_host,
    upload_frame,
    window_memory,
)

MIB = 2**20
# The memory one window adds, in MiB per megapixel of its frames: RSR Compact (2x) in
# FP16, measured in steady state (the second window on) at 960x540, 1280x720 and
# 1920x1080, peak allocated over what was allocated before the window (MiB):
#   frames   960x540   1280x720   1920x1080
#   1            663       1194        2648
#   3           1394       2505        5568
#   7           1831       3287        7325
# that is 1277-1296, 2685-2718 and 3532-3567 MiB per megapixel. A window with frames
# ahead takes SWEEP, plus FRAME (its encoding and reverse state, in the model's dtype)
# and INPUT (its FP32 frame and the window's FP32 stack) for each frame, plus each kept
# frame's FP32 output (OUTPUT) and the head's FP16 output (HEAD), both times scale^2;
# one with none (no reverse sweep) PAST_ONLY in place of SWEEP and the frames. Fitted
# on 1280x720, the largest per megapixel: the estimate is 0.2-2.2 % above every
# measurement. OUTPUT (12 bytes per output pixel) and HEAD (three FP16 tensors of the
# head's 3 x scale^2 channels) are derived, not measured: they set the scales 1 and 4.
# FP32 doubles the model-dtype terms (derived, not measured). A first window reserves
# up to twice what it allocates while the allocator's cache is cold (measured 906 to
# 9698 MiB reserved); later windows reuse the cache.
SWEEP, FRAME, INPUT, PAST_ONLY = 2040, 151, 23, 1190
OUTPUT, HEAD = 11.5, 17.2
# Another RSR shape scales FRAME by the channels per pixel a frame holds (its encoding:
# width at full size and two of context_width at a quarter of the size each way; its
# reverse state: width) and SWEEP and PAST_ONLY by its widest activation (hidden), each
# against Compact's (32, 64, 64): derived, not measured. Compact's factors are 1.
COMPACT_FRAME_CHANNELS, COMPACT_HIDDEN = 2 * 32 + 2 * 64 / 16, 64
# The aligned read (optical flow, NeuFlow v2) peaks in the larger of two phases, as its
# flow runs before the sweeps. Its sweeps take ALIGNED_SWEEP in place of SWEEP (no
# temporal tokens), with the same per-frame and output terms. Its flow runs on the
# frames reduced by spandrel's flow_working_factor and padded to 16 (at least 32), and
# takes per pair the all-pairs volume on the grid of an eighth (two FP32 flows) times
# VOLUME_MARGIN plus FLOW_PER_MEGAPIXEL per megapixel of that grid; a steady window of
# n frames flows n pairs (the last kept frame of the window before and its own n - 1),
# in one call up to the pairs spandrel puts in a call. RSR Compact (2x) in FP16,
# measured in steady state, peak allocated over what was allocated before the window
# (MiB):
#   frames   640x360   720x480   960x540   1920x1080
#   6                             1169
#   7          1463      2763      1282        5042
#   8                             1395
# and 2285 at 960x540 for 7 frames in FP32; the flow is the larger phase at 640x360 and
# 720x480. Fitted to these points (911 and 500), then raised by the block read's own
# margin over its measurements (about 7 %): the estimate is 6-14 % above each.
ALIGNED_SWEEP, FLOW_PER_MEGAPIXEL, VOLUME_MARGIN = 1080, 540, 1.07
# What the estimate leaves free, on top of it, as RDUAL's.
MARGIN, MARGIN_MIB = 0.20, 256


def shape_factors(hyperparameters: Mapping[str, Any]) -> tuple[float, float]:
    """FRAME's and SWEEP's (and PAST_ONLY's) factors for an RSR of these constructor
    values, against Compact's."""
    frame_channels = (
        2 * hyperparameters["width"] + 2 * hyperparameters["context_width"] / 16
    )
    return (
        frame_channels / COMPACT_FRAME_CHANNELS,
        hyperparameters["hidden"] / COMPACT_HIDDEN,
    )


def is_aligned(network: torch.nn.Module) -> bool:
    """Whether this RSR reads its neighbours through optical flow (temporal_read
    "aligned"); spandrel before its aligned read has the block read only."""
    return getattr(network, "temporal_read", "block") == "aligned"


def window_peak(
    height: int,
    width: int,
    ahead: int,
    scale: int,
    dtype_bytes: int,
    factors: tuple[float, float] = (1, 1),
    flow_factor: int | None = None,
) -> int:
    """The bytes a window with `ahead` frames ahead adds to what is allocated, for
    frames of height x width, a model of `scale` and weights of `dtype_bytes`, of a
    shape with `factors` (`shape_factors`; Compact's by default), and for the aligned
    read the flow on frames reduced by `flow_factor` (None: the block read)."""
    frames = ahead + 1
    keep = max(1, frames - LOOK_AHEAD)
    dtype = dtype_bytes / 2
    squared = scale * scale
    frame, sweep = factors
    if ahead == 0:
        body = PAST_ONLY * sweep * dtype
    else:
        sweeps = SWEEP if flow_factor is None else ALIGNED_SWEEP
        body = sweeps * sweep * dtype + (FRAME * frame * dtype + INPUT) * frames
    per_megapixel = body + HEAD * dtype * squared + OUTPUT * squared * keep
    peak = height * width / 1e6 * per_megapixel
    if flow_factor is not None:
        # spandrel's RDUAL copy, which RSR's aligned read takes its flow from
        from spandrel.architectures.RDUAL.__arch.rdual_arch import _flow_pairs_per_call

        # As RSR.guidance pads the reduced frames
        flow_h = max(32, (max(1, round(height / flow_factor)) + 15) // 16 * 16)
        flow_w = max(32, (max(1, round(width / flow_factor)) + 15) // 16 * 16)
        cells = (flow_h // 8) * (flow_w // 8)
        pairs = min(frames, _flow_pairs_per_call(flow_h, flow_w))
        volume = 2 * cells * cells * 4 / MIB * VOLUME_MARGIN
        peak = max(peak, pairs * (volume + FLOW_PER_MEGAPIXEL * flow_h * flow_w / 1e6))
    return int(peak * MIB)


class RSRWindows:
    """Frames are RGB CHW tensors on the device (in the model's dtype; FP32 for the
    aligned read), outputs `Output`s, each on its way to the host as soon as its window
    ran (in page-locked memory on a GPU), and the state is run_window's carry
    (the forward state of the last kept frame and the encodings of the frames after
    it). The model runs as a copy on the device in FP16 or FP32 (`on_device`): the
    loaded model, which other nodes share, is not changed."""

    def __init__(
        self,
        model: ImageModelDescriptor,
        device: torch.device,
        budget: int = 0,
        use_fp16: bool = False,
    ):
        self.model = on_device(
            model, device, torch.float16 if use_fp16 else torch.float32
        )
        self.network = self.model.model
        self.device = device
        # The PyTorch settings' memory budget in bytes; 0: none.
        self.budget = budget

    def fitting_ahead(self, shape: tuple[int, int], ahead: int) -> int:
        """The frames ahead (at most `ahead`) whose window fits the device's free memory
        by `window_peak`, with its margin, or -1 when not even a window of one frame
        fits; on a CPU, `ahead`."""
        if self.device.type != "cuda":
            return ahead
        available = window_memory(self.device, self.budget)
        dtype_bytes = torch.finfo(self.model.dtype).bits // 8
        factors = shape_factors(self.network.hyperparameters)
        flow_factor = None
        if is_aligned(self.network):
            from spandrel.architectures.RSR import flow_working_factor

            flow_factor = flow_working_factor(*shape)
        for frames in range(ahead, -1, -1):
            peak = window_peak(
                *shape, frames, self.model.scale, dtype_bytes, factors, flow_factor
            )
            if peak * (1 + MARGIN) + MARGIN_MIB * MIB <= available:
                return frames
        return -1

    @property
    def key(self) -> Hashable:
        return ("pytorch", id(self.network), str(self.device))

    def convert(self, frame: np.ndarray) -> torch.Tensor:
        # BGR HWC -> RGB CHW in 0..1 on the model's device, already in the model's
        # dtype (run_window's cast then has nothing to do), but in FP32 for the aligned
        # read: its flow reads the FP32 frames.
        dtype = torch.float32 if is_aligned(self.network) else self.model.dtype
        return upload_frame(frame, self.device, dtype)

    def run(
        self,
        frames: Sequence[torch.Tensor],
        keep: int,
        past: tuple[Any, torch.Tensor] | None,
        _first: int,
    ) -> tuple[list[Output], Any]:
        # The carry (run_window's, whose type spandrel does not export: opaque here)
        # holds the encodings of the frames the last window read past its kept ones,
        # which are this window's first: the index is not needed.
        # spandrel before 0.4.2+c23 has no RSR: imported here, so that Upscale Image
        # loads with it.
        from spandrel.architectures.RSR import run_window

        # On a GPU, the window's frames in channels-last order, as Upscale Image hands
        # the CPU its input (auto_split); the weights stay as they are. Measured 2x in
        # FP16 on an RTX 5090, windows of 7 keeping 6, at 960x540: 11.94 against
        # 14.32 ms per kept frame in NCHW, the same peak memory, and the same RMSE
        # against FP32 (7.3e-4; NCHW 7.4e-4). The stack is its one copy; each of its
        # frames is a channels-last view (convert made them in the model's dtype; the
        # aligned read's FP32 frames are cast by run_window, which keeps the layout).
        # The CPU, not measured, keeps NCHW and its outputs bit for bit.
        if self.device.type == "cuda":
            clip = torch.stack([f.permute(1, 2, 0) for f in frames]).permute(0, 3, 1, 2)
        else:
            clip = torch.stack(list(frames))
        with torch.inference_mode():
            outputs, carry = run_window(
                self.network,
                clip[None],
                keep,
                None if past is None else past[0],
            )
            # Each kept frame's copy to the host starts now, behind the window's work.
            return [to_host(output) for output in outputs[0]], carry

    @staticmethod
    def to_image(output: Output) -> np.ndarray:
        return host_image(output)
