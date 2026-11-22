"""The PyTorch executor of Upscale Image's temporal path (`temporal_window.Executor`) for
SMoVSR: its windows through spandrel's `run_window`, eager, in FP16 or FP32 as the node
runs any model.

SMoVSR sweeps a clip forward, carrying a state from frame to frame, and then backward
from the clip's end, so each output reads every later frame. A window keeps the frames
with spandrel's LOOK_AHEAD frames after them in it: its backward sweep starts cold at
the window's last frame, and the outputs stay within a measured bound of a run of the
whole clip (exact at the video's end, where the last window keeps every frame). The
forward sweep is carried exactly from one window into the next, whatever their sizes:
the windows grow after a cold start (`grows`), up to MAX_KEEP kept frames."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import torch
from sanic.log import logger
from spandrel import ImageModelDescriptor

from ..item_window import memory_budget
from .utils import Output, host_image, on_device, to_host, upload_frame, window_memory

MIB = 2**20
# The most frames a window keeps: each window's backward sweep also runs over its
# LOOK_AHEAD frames ahead, so that work is (MAX_KEEP + LOOK_AHEAD) / MAX_KEEP of the
# kept frames' (1.5 at 32 and 16).
MAX_KEEP = 32
# The memory a window adds, in MiB per megapixel of its frames, by the bytes of the
# model's dtype: FIXED, plus FRAME for each frame it encodes and sweeps forward, plus
# KEPT for each kept frame (its head and FP32 output), 2x. Measured on an RTX 5090
# through this executor (its frames' upload included) in steady state, after a window
# that carries 16 frames, peak allocated over what was allocated before (MiB):
#   frames, kept        17, 1   32, 1   32, 16   48, 32   40, 8   24, 1   24, 8
#   FP16  960x540        516    1565     2068     3947    2290
#   FP16  1920x1080     2055                                      3999    4663
#   FP32  960x540       1024    3101     3457     6331    4375
#   FP32  1920x1080     4104                                      7979    8644
# and 8242 (FP16) and 13833 (FP32) at 1920x1080 for 32 frames keeping 16; fitted per
# megapixel over the frames encoded (each window's less the 16 carried), within
# -7.5..+6.6 % of every point. A window's estimate counts every one of its frames as
# encoded, the carry it starts from as well as the frames it adds: its whole peak, which
# fits the memory a window may add (`window_memory`) plus the carry it starts from,
# which is allocated already.
FIXED = {2: 709.0, 4: 1585.0}
FRAME = {2: 133.7, 4: 268.7}
KEPT = {2: 77.1, 4: 58.3}
# What the estimate leaves free, on top of it, as RSR's.
MARGIN, MARGIN_MIB = 0.20, 256
# The frame shapes, frames read ahead and limits already logged as cutting the frames
# read ahead below LOOK_AHEAD: each is logged once.
_logged: set[tuple[tuple[int, int], int, str]] = set()


def window_peak(
    height: int, width: int, frames: int, keep: int, scale: int, dtype_bytes: int
) -> int:
    """The bytes a window of `frames` frames keeping `keep` takes at its peak, its carry
    included, for frames of height x width, a model of `scale` (the kept outputs,
    measured at 2x) and weights of `dtype_bytes` (2 or 4)."""
    kept = KEPT[dtype_bytes] * scale * scale / 4
    per_megapixel = FIXED[dtype_bytes] + FRAME[dtype_bytes] * frames + kept * keep
    return int(height * width / 1e6 * per_megapixel * MIB)


def carry_bytes(carry: Any) -> int:
    """The bytes of run_window's carry (its state, encodings and forward outputs)."""
    state, feats, outs = carry
    return sum(t.nbytes for t in (state, *feats, *outs))


def fits(peak: int, available: int) -> bool:
    """Whether a window whose estimate is `peak` bytes fits `available` bytes, with the
    margin."""
    return peak * (1 + MARGIN) + MARGIN_MIB * MIB <= available


def ram_ahead(
    shape: tuple[int, int],
    ahead: int,
    scale: int,
    look_ahead: int,
    copy_bytes: int = 0,
) -> int:
    """The frames ahead (at most `ahead`) whose window's host memory fits the item
    window's RAM budget (`memory_budget`): the loop's frames held for it (one BGR
    float32 frame each), the executor's own copy of each of the window's frames
    (`copy_bytes` each; 0: none) and its kept outputs, which wait on the host until
    their items take them (one BGR float32 image at the output size each); fewer kept
    frames first. Both SMoVSR executors (this one and the TensorRT session) plan by
    it."""
    height, width = shape
    frame_bytes = height * width * 3 * 4
    output_bytes = frame_bytes * scale * scale
    budget = memory_budget()

    def held(frames_ahead: int) -> int:
        keep = max(1, frames_ahead + 1 - look_ahead)
        copies = (frames_ahead + 1) * copy_bytes
        return frames_ahead * frame_bytes + copies + keep * output_bytes

    return next((a for a in range(ahead, -1, -1) if held(a) <= budget), 0)


def log_fewer_ahead(
    shape: tuple[int, int], fitting: int, ahead: int, look_ahead: int, limit: str
) -> None:
    """Logs once per frame size, frames ahead and `limit` (a memory) when it cut the
    frames ahead of an SMoVSR window below its `look_ahead`."""
    if fitting >= min(ahead, look_ahead):
        return
    seen = (shape, fitting, limit)
    if seen in _logged:
        return
    _logged.add(seen)
    logger.info(
        "Upscale Image: SMoVSR reads %d frames ahead of %d wanted at"
        " %dx%d, limited by %s; its outputs are less exact",
        fitting,
        look_ahead,
        shape[1],
        shape[0],
        limit,
    )


class SMoVSRWindows:
    """Frames are RGB CHW tensors on the device in the model's dtype, outputs `Output`s,
    each on its way to the host as soon as its window ran (in page-locked memory on a
    GPU), and the state is run_window's carry (the forward state after the window's last
    frame, and the encodings and forward outputs of the frames after its kept ones). The
    model runs as a copy on the device in FP16 or FP32 (`on_device`): the loaded model,
    which other nodes share, is not changed."""

    grows = True

    def __init__(
        self,
        model: ImageModelDescriptor,
        device: torch.device,
        budget: int = 0,
        use_fp16: bool = False,
    ):
        # spandrel before 0.4.2+c25 has no SMoVSR: imported here, so that Upscale Image
        # loads with it.
        from spandrel.architectures.SMoVSR import LOOK_AHEAD

        self.model = on_device(
            model, device, torch.float16 if use_fp16 else torch.float32
        )
        self.network = self.model.model
        self.device = device
        # The PyTorch settings' memory budget in bytes; 0: none.
        self.budget = budget
        self.look_ahead: int = LOOK_AHEAD

    def fitting_ahead(
        self, shape: tuple[int, int], ahead: int, carry: Any = None
    ) -> int:
        """The frames ahead (at most `ahead`) whose window fits: on a GPU, `window_peak`
        with its margin within the memory a window may add now (`window_memory`) plus
        the bytes of `carry`, the state the planner carries into it (allocated already,
        and counted in the estimate; None: a cold start), and in the item window's RAM
        budget (`memory_budget`) the loop's frames held for it (one BGR float32 frame
        each) and its kept outputs, which wait on the host until their items take them
        (one BGR float32 image at the output size each); fewer kept frames first, as
        for the GPU. A memory limit that cuts the frames ahead below LOOK_AHEAD is
        logged once (the temporal path logs a loop that offers fewer): the outputs are
        then further from a run of the whole clip, with no error. A frame that does not
        fit even with no frame ahead raises before anything is allocated for it."""
        height, width = shape
        fitting, limit = ahead, ""
        ram = ram_ahead(shape, fitting, self.model.scale, self.look_ahead)
        if ram < fitting:
            fitting, limit = ram, "the RAM budget of the loop's frames and outputs"
        if self.device.type == "cuda":
            carried = 0 if carry is None else carry_bytes(carry)
            available = window_memory(self.device, self.budget) + carried
            dtype_bytes = torch.finfo(self.model.dtype).bits // 8

            def peak(frames_ahead: int) -> int:
                frames = frames_ahead + 1
                keep = max(1, frames - self.look_ahead)
                return window_peak(
                    height, width, frames, keep, self.model.scale, dtype_bytes
                )

            gpu = next(
                (a for a in range(fitting, -1, -1) if fits(peak(a), available)),
                -1,
            )
            if gpu < 0:
                raise torch.OutOfMemoryError(
                    f"SMoVSR does not fit the GPU's memory at {width}x{height}, even"
                    f" with no frames read ahead: a frame takes about"
                    f" {peak(0) // MIB} MiB, and {available // MIB} MiB are free. Use a"
                    " smaller frame, or turn on FP16."
                )
            if gpu < fitting:
                fitting, limit = gpu, "the GPU's memory"
        log_fewer_ahead(shape, fitting, ahead, self.look_ahead, limit)
        return fitting

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
        past: tuple[Any, torch.Tensor] | None,
        _first: int,
    ) -> tuple[list[Output], Any]:
        # The carry holds the encodings of the frames the last window read past its
        # kept ones, which are this window's first: the index is not needed.
        from spandrel.architectures.SMoVSR import run_window

        carry = None if past is None else past[0]
        rest: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]] = ((), ())
        if carry is not None and len(carry[1]) > len(frames):
            # A window shorter than the last one's look-ahead (fewer frames fit now):
            # every frame of it is carried, and the carry's state follows the last
            # carried frame, so the frames past this window stay in the carry for the
            # next one.
            state, feats, outs = carry
            n = len(frames)
            carry, rest = (state, feats[:n], outs[:n]), (feats[n:], outs[n:])
        with torch.inference_mode():
            outputs, (state, feats, outs) = run_window(
                self.network, torch.stack(list(frames))[None], keep, carry
            )
            # Each kept frame's copy to the host starts now, behind the window's work.
            return [to_host(output) for output in outputs[0]], (
                state,
                (*feats, *rest[0]),
                (*outs, *rest[1]),
            )

    @staticmethod
    def to_image(output: Output) -> np.ndarray:
        return host_image(output)
