"""The run-time session of a TFDAT or TFDATv2 engine set (tfdat.py): the executor of
Upscale Image's temporal path (temporal_window.Executor), with the PyTorch executor's
clips (pytorch/tfdat_window.py): each kept frame's output is one engine call on the clip
of `clip_size` frames centred on it, frames past the video's ends repeating the nearest
one, and the state a window hands the next is the up to `look_ahead` frames before it.

The engine takes the clip stacked on the channels in one FP32 binding, which each call
fills from the frames; the call is a CUDA graph captured at its first run and replayed
after (temporal_session.Part). `to_host` converts the output into an image of its own on
the stream before the copy, so the next call never writes over an image on its way.

This module must not import TensorRT at its top (the PyTorch package imports it)."""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from api import NodeContext, NodeId

from ..pytorch.utils import Output, host_image, to_host, upload_frame
from ..upscale.basic_upscale import PaddingType
from ..upscale.temporal_window import temporal_upscale
from ..upscale.tile_report import TileReport
from .cache import SharedCache, engines
from .model import TensorRTEngine
from .temporal_session import Part, close_parts, open_set, set_fit
from .tfdat_export import PART

# A set takes whole frames of its one size, and each frame is one call whatever the
# window, so a frame that does not fit has no further step on this route.
SET_EXHAUSTED = (
    "This TFDAT engine set does not fit the GPU's memory at its frame size. Close other"
    " programs that use the GPU, or run the model on PyTorch (Use TensorRT Engine off)."
)


class TFDATSession:
    """The executor of a TFDAT or TFDATv2 engine set on a GPU. Frames are RGB CHW
    float32 tensors on the device, copied there on the session's stream; outputs are
    `Output`s on their way to the host."""

    def __init__(
        self,
        parts: dict[str, Part],
        runtime: Any,
        resident: int,
        stream: Any,
        device: torch.device,
        size: tuple[int, int, int],
        key: Hashable,
    ) -> None:
        """`parts` holds the set's one engine (open_set), which its runtime outlives;
        `size` is the frames' height, width and the set's scale."""
        del resident  # each frame is one call: there is no window to plan by memory
        self.part = parts[PART]
        self.parts = parts
        self._runtime = runtime
        self.stream = stream
        self.device = device
        self._key = key
        self.height, self.width, self.scale = size
        self._workspace = torch.empty(
            max(1, self.part.workspace), dtype=torch.uint8, device=device
        )
        self.part.use_memory(self._workspace)
        self._clip = self.part.tensor("x", device)
        self._output = self.part.tensor("output", device)
        self.clip_size = self._clip.shape[1] // 3
        self.look_ahead: int = self.clip_size // 2

    @property
    def key(self) -> Hashable:
        return self._key

    def convert(self, frame: np.ndarray) -> torch.Tensor:
        """A BGR HWC float32 frame of the set's size -> RGB CHW in 0..1 on the GPU."""
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"This TFDAT engine set takes {self.width}x{self.height} frames, not"
                f" {frame.shape[1]}x{frame.shape[0]}: build one for this size."
            )
        with torch.cuda.stream(self.stream):
            return upload_frame(frame, self.device)

    @staticmethod
    def to_image(output: Output) -> np.ndarray:
        return host_image(output)

    def run(
        self,
        frames: Sequence[torch.Tensor],
        keep: int,
        past: tuple[tuple[torch.Tensor, ...], torch.Tensor] | None,
        first: int,
    ) -> tuple[list[Output], tuple[torch.Tensor, ...]]:
        """One window (temporal_window.Run), as TFDATWindows.run: the carried frames
        end just before this window's first, so the index is not needed."""
        del first
        if not 1 <= keep <= len(frames):
            raise ValueError(f"A window of {len(frames)} frames cannot keep {keep}.")
        before = () if past is None else past[0]
        sequence = [*before, *frames]
        last = len(sequence) - 1
        side = self.look_ahead
        height, width = self.height * self.scale, self.width * self.scale
        outputs: list[Output] = []
        torch.cuda.set_device(self.device)
        stream = self.stream
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.no_grad(), torch.cuda.stream(stream):
            for centre in range(len(before), len(before) + keep):
                for k in range(-side, side + 1):
                    frame = sequence[min(max(centre + k, 0), last)]
                    self._clip[0, (k + side) * 3 : (k + side + 1) * 3].copy_(frame)
                self.part.call(PART, {"x": self._clip, "output": self._output}, stream)
                outputs.append(to_host(self._output[0, :, :height, :width]))
        end = len(before) + keep
        return outputs, tuple(sequence[max(0, end - side) : end])

    def close(self) -> None:
        """Frees the graph and context before the engine, the engine before its
        runtime."""
        self.stream.synchronize()
        close_parts(self.parts)
        self._runtime = None


def open_session(engine: TensorRTEngine, gpu_index: int) -> TFDATSession:
    """The session of a TFDAT or TFDATv2 engine set on a GPU: its engine deserialized,
    with an execution context."""
    temporal = engine.info.temporal
    assert temporal is not None
    return open_set(engine, gpu_index, temporal.architecture, TFDATSession)


_sessions: SharedCache[TFDATSession] = SharedCache(TFDATSession.close, built_on=engines)


@contextmanager
def use_tfdat_session(
    engine: TensorRTEngine, gpu_index: int, node_id: NodeId
) -> Iterator[TFDATSession]:
    """The session of a TFDAT engine set on a GPU for `node_id`, kept across runs, one
    per node, as rsr_session.use_rsr_session keeps RSR's."""
    with _sessions.use(
        (engine, gpu_index, node_id), node_id, lambda: open_session(engine, gpu_index)
    ) as session:
        yield session


def temporal_engine_upscale(
    context: NodeContext,
    img: np.ndarray,
    engine: TensorRTEngine,
    gpu_index: int,
    report: TileReport,
    padding: PaddingType = PaddingType.NONE,
) -> np.ndarray:
    """`img` upscaled by a TFDAT engine set on the temporal path (temporal_window), its
    session the executor, kept for the node; past running out of memory SET_EXHAUSTED
    is raised."""
    with use_tfdat_session(engine, gpu_index, context.node_id) as session:
        return temporal_upscale(
            context, img, session, report, padding, set_fit(SET_EXHAUSTED)
        )


__all__ = [
    "TFDATSession",
    "open_session",
    "temporal_engine_upscale",
    "use_tfdat_session",
]
