"""The run-time session of an RDUAL engine set (rdual_engine_set.py): the executor of
Upscale Image's temporal path (temporal_window.Executor), whichever route made the set.

A window is frames t0..t0+n-1 in order; the first `keep` get outputs and the rest are the
reverse sweep's look-ahead. Per window, as RDUAL.infer_window: the flow of each adjacent
pair (flow engine) with its alignment cues (the model's own `pack_guidance`, eager PyTorch
FP32, no weights), the prefix engine once per frame, the reverse sweep (revcore) from the
window's last frame down with no future there, then the forward sweep (fwdcore) over the
kept frames, its first frame linked to the carried state by the pair of the frame before.

Every frame index owns a slot of buffers (a ring of `slots` frames, frame i in slot
i mod slots), so a window that continues the last one (its `past`, starting where that
one's kept frames ended) reuses the prefix outputs, flows and cues of the frames both
read: only the reverse sweep repeats. The ring grows to the longest window run, never
beyond. Every engine call is a CUDA graph captured at its first run and replayed after;
the four engines share one working memory. Outputs and the returned state are copies, so
a later window never writes over what the planner still holds. What is not RDUAL's own
is temporal_session.py's, which RSR's session shares.

This module must not import TensorRT at its top (the PyTorch package imports it)."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from torch.nn import functional

from api import NodeContext, NodeId

from ..upscale.basic_upscale import PaddingType
from ..upscale.temporal_window import temporal_upscale
from ..upscale.tile_report import TileReport
from .cache import SharedCache, engines
from .model import TensorRTEngine
from .temporal_session import (
    ALLOCATOR_SHARE,
    OUTSIDE_ALLOCATOR_MARGIN,
    Output,
    Part,
    SetSession,
    WindowMemory,
    open_set,
    set_fit,
)

# A set takes whole frames of its one size, so a frame that does not fit with no frames
# read ahead has no further step on this route.
SET_EXHAUSTED = (
    "This RDUAL engine set does not fit the GPU's memory at its frame size, even with no"
    " frames read ahead. Close other programs that use the GPU, or run the model on"
    " PyTorch (Use TensorRT Engine off), which can tile the frame."
)


class _Slot:
    """One frame's buffers: its padded input, the prefix's outputs, its reverse and
    forward states, and the flow and packed guidance of the pair (it, the next frame)."""

    def __init__(self, parts: dict[str, Part], device: torch.device) -> None:
        prefix, revcore = parts["prefix"], parts["revcore"]
        self.input = prefix.tensor("input", device)
        self.stem = prefix.tensor("stem", device)
        self.fine = prefix.tensor("fine", device)
        self.center = prefix.tensor("center", device)
        self.key = prefix.tensor("key", device)
        self.future = revcore.tensor("state_out", device)
        self.forward = revcore.tensor("state_out", device)
        self.flow = parts["flow"].tensor("guidance", device)
        (_, *cue), _ = revcore.spec("guidance_future")
        self.packed = torch.empty((2, *cue), device=device)

    def tensors(self) -> list[torch.Tensor]:
        return list(vars(self).values())


def pad_frame(frame: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """A 1,3,H,W frame padded to height x width as RDUAL._pad pads it."""
    ph, pw = height - frame.shape[-2], width - frame.shape[-1]
    if not (ph or pw):
        return frame
    small = frame.shape[-2] <= ph or frame.shape[-1] <= pw
    return functional.pad(
        frame, (0, pw, 0, ph), mode="replicate" if small else "reflect"
    )


def model_pack_guidance() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """The model's own `pack_guidance`, of an RDUAL without weights (it reads none)."""
    from spandrel.architectures.RDUAL.__arch.rdual_arch import RDUAL

    with torch.device("meta"):
        model = RDUAL(use_checkpoint=False)
    return model.pack_guidance


class RDUALSession(SetSession[_Slot]):
    """The executor of an RDUAL engine set on a GPU (temporal_session.SetSession); the
    state is the BF16 trunk state."""

    ARCHITECTURE = "RDUAL"

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
        """The buffers every window shares, sized by the parts' bindings; the ring of
        slots grows with the windows (`_ensure_slots`). `parts` are the set's engines
        (open_session), which their runtime outlives, and `resident` the device bytes
        they and their contexts took; `size` is the frames' height, width and the set's
        scale."""
        self._pair = parts["flow"].tensor("frames", device)
        self._zero_state = torch.zeros_like(
            parts["revcore"].tensor("state_out", device)
        )
        self._zero_guidance = torch.zeros_like(
            parts["revcore"].tensor("guidance_future", device)
        )
        self._on = torch.ones(1, dtype=torch.bool, device=device)
        self._off = torch.zeros(1, dtype=torch.bool, device=device)
        self._pack_guidance = model_pack_guidance()
        super().__init__(
            parts,
            runtime,
            resident,
            stream,
            device,
            size,
            key,
            [self._pair, self._zero_state, self._zero_guidance],
            self._zero_state,
        )

    def new_slot(self) -> _Slot:
        return _Slot(self.parts, self.device)

    def _run(
        self,
        frames: Sequence[torch.Tensor],
        keep: int,
        past: tuple[torch.Tensor, torch.Tensor] | None,
        first: int,
    ) -> tuple[list[Output], torch.Tensor]:
        count = len(frames)
        start = first - (past is not None)
        by_index = {first + k: frame for k, frame in enumerate(frames)}
        if past is not None:
            by_index[first - 1] = past[1]
        if past is None or self._next != first:
            self._done = {}
        # Only the window's indices keep their slots' work.
        self._done = {
            i: done for i, done in self._done.items() if start <= i < first + count
        }
        stream = self.stream
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            if past is not None:
                self._slot(first - 1).forward.copy_(past[0])
            for i in range(first, first + count):
                self._prefix(i, by_index[i])
            for i in range(start, first + count - 1):
                self._flow(i, by_index[i], by_index[i + 1])
            for i in range(first + count - 1, first, -1):
                self._reverse(i, i + 1 < first + count)
            outputs = []
            height, width = self.height * self.scale, self.width * self.scale
            for i in range(first, first + keep):
                self._forward(i, i > first or past is not None, i + 1 < first + count)
                outputs.append(self._to_host(height, width))
            state = self._slot(first + keep - 1).forward.clone()
        self._next = first + keep
        return outputs, state

    def _prefix(self, index: int, frame: torch.Tensor) -> None:
        done = self._done.setdefault(index, set())
        if "prefix" in done:
            return
        slot = self._slot(index)
        _, _, height, width = slot.input.shape
        slot.input.copy_(pad_frame(frame[None], height, width))
        self.parts["prefix"].call(
            ("prefix", index % len(self.slots)),
            {
                "input": slot.input,
                "stem": slot.stem,
                "fine": slot.fine,
                "center": slot.center,
                "key": slot.key,
            },
            self.stream,
        )
        done.add("prefix")

    def _flow(self, index: int, earlier: torch.Tensor, later: torch.Tensor) -> None:
        """The pair (index, index + 1): its flow in both directions and their cues."""
        done = self._done.setdefault(index, set())
        if "pair" in done:
            return
        slot = self._slot(index)
        self._pair[0].copy_(earlier)
        self._pair[1].copy_(later)
        self.parts["flow"].call(
            ("flow", index % len(self.slots)),
            {"frames": self._pair, "guidance": slot.flow},
            self.stream,
        )
        packed = self._pack_guidance(self._pair[None], slot.flow[None, None])
        slot.packed.copy_(packed[0, 0])
        done.add("pair")

    def _reverse(self, index: int, has_future: bool) -> None:
        slot, ring = self._slot(index), len(self.slots)
        future = self._slot(index + 1) if has_future else None
        self.parts["revcore"].call(
            ("revcore", index % ring, has_future),
            {
                "state_future": self._zero_state if future is None else future.future,
                "guidance_future": self._zero_guidance
                if future is None
                else slot.packed[1:2],
                "valid_future": self._off if future is None else self._on,
                "fine": slot.fine,
                "center": slot.center,
                "key": slot.key,
                "state_out": slot.future,
            },
            self.stream,
        )

    def _forward(self, index: int, has_past: bool, has_future: bool) -> None:
        slot, ring = self._slot(index), len(self.slots)
        before = self._slot(index - 1) if has_past else None
        after = self._slot(index + 1) if has_future else None
        self.parts["fwdcore"].call(
            ("fwdcore", index % ring, has_past, has_future),
            {
                "input": slot.input,
                "state_past": self._zero_state if before is None else before.forward,
                "guidance_past": self._zero_guidance
                if before is None
                else before.packed[0:1],
                "valid_past": self._off if before is None else self._on,
                "state_future": self._zero_state if after is None else after.future,
                "guidance_future": self._zero_guidance
                if after is None
                else slot.packed[1:2],
                "valid_future": self._off if after is None else self._on,
                "stem": slot.stem,
                "fine": slot.fine,
                "center": slot.center,
                "key": slot.key,
                "output": self._output,
                "state_out": slot.forward,
            },
            self.stream,
        )


def open_session(engine: TensorRTEngine, gpu_index: int) -> RDUALSession:
    """The session of an engine set on a GPU: its four engines deserialized, each with
    an execution context."""
    return open_set(engine, gpu_index, RDUALSession.ARCHITECTURE, RDUALSession)


_sessions: SharedCache[RDUALSession] = SharedCache(RDUALSession.close, built_on=engines)


@contextmanager
def use_rdual_session(
    engine: TensorRTEngine, gpu_index: int, node_id: NodeId
) -> Iterator[RDUALSession]:
    """The session of an engine set on a GPU for `node_id`, kept across runs (see
    `cache`; the node's Clear frees it). Each node has its own: a session keeps the work
    of its last window's frames by their index, which another node's frames of the same
    indices would take for its own (and a node holds its session while it peeks at the
    frames ahead, which a node before it on the same set would wait for)."""
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
    """`img` upscaled by an engine set on the temporal path (temporal_window), its
    session the executor, kept for the node. The set takes whole frames of its one size,
    whatever the node's Tile Size; when the GPU's memory runs out, the window is
    shortened (Fit's steps), and past the last step SET_EXHAUSTED is raised."""
    with use_rdual_session(engine, gpu_index, context.node_id) as session:
        return temporal_upscale(
            context, img, session, report, padding, set_fit(SET_EXHAUSTED)
        )


__all__ = [
    "ALLOCATOR_SHARE",
    "OUTSIDE_ALLOCATOR_MARGIN",
    "Output",
    "RDUALSession",
    "WindowMemory",
    "model_pack_guidance",
    "open_session",
    "pad_frame",
    "temporal_engine_upscale",
    "use_rdual_session",
]
