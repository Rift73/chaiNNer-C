"""The run-time session of an RSR engine set (rsr.py): the executor of Upscale Image's
temporal path (temporal_window.Executor), on temporal_session.SetSession.

A window is frames t0..t0+n-1 in order; the first `keep` get outputs and the rest are the
reverse sweep's look-ahead. Per window, as the library's RSR run_window: the prefix
engine once per frame, the reverse sweep (revcore) from the window's last frame (no
future there) down to its second, then the forward sweep (fwdcore) over the kept
frames, its first frame reading the carried state (none on a cold start). An absent
neighbour is a zero state with its validity flag false.

Every frame index owns a slot (frame i in slot i mod slots), so a window that continues
the last one reuses the prefix outputs of the frames both read: only the reverse sweep
repeats. Every engine call is a CUDA graph captured at its first run (one per part, slot
and validity) and replayed after; the three engines share one working memory. Outputs
and the returned state are copies, so a later window never writes over what the planner
still holds. The states and prefix outputs are FP16 in the engines' channel-last layout:
only the engines read them.

This module must not import TensorRT at its top (the PyTorch package imports it)."""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from api import NodeContext, NodeId

from ..upscale.basic_upscale import PaddingType
from ..upscale.temporal_window import temporal_upscale
from ..upscale.tile_report import TileReport
from .cache import SharedCache, engines
from .model import TensorRTEngine
from .temporal_session import (
    Output,
    Part,
    SetSession,
    open_set,
    set_fit,
)

# A set takes whole frames of its one size, so a frame that does not fit with no frames
# read ahead has no further step on this route.
SET_EXHAUSTED = (
    "This RSR engine set does not fit the GPU's memory at its frame size, even with no"
    " frames read ahead. Close other programs that use the GPU, or run the model on"
    " PyTorch (Use TensorRT Engine off)."
)


class _Slot:
    """One frame's buffers: its input, the prefix's outputs, and its reverse and forward
    states."""

    def __init__(self, parts: dict[str, Part], device: torch.device) -> None:
        prefix, revcore = parts["prefix"], parts["revcore"]
        self.x = prefix.tensor("x", device)
        self.current6 = prefix.tensor("current6", device)
        self.c3 = prefix.tensor("c3", device)
        self.c4 = prefix.tensor("c4", device)
        self.future = revcore.tensor("state", device)
        self.forward = revcore.tensor("state", device)

    def tensors(self) -> list[torch.Tensor]:
        return list(vars(self).values())


class RSRSession(SetSession[_Slot]):
    """The executor of an RSR engine set on a GPU; the state is the forward sweep's
    FP16 state of the last kept frame."""

    ARCHITECTURE = "RSR"

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
        slots grows with the windows. `resident` is the device bytes the engines and
        their contexts took."""
        self._zero_state = torch.zeros_like(parts["revcore"].tensor("state", device))
        self._on = torch.ones(1, dtype=torch.bool, device=device)
        self._off = torch.zeros(1, dtype=torch.bool, device=device)
        super().__init__(
            parts,
            runtime,
            resident,
            stream,
            device,
            size,
            key,
            [self._zero_state],
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
        if past is None or self._next != first:
            self._done = {}
        # Only the window's indices keep their slots' work.
        self._done = {
            i: done for i, done in self._done.items() if first <= i < first + count
        }
        stream = self.stream
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            if past is not None:
                self._slot(first - 1).forward.copy_(past[0])
            for k, frame in enumerate(frames):
                self._prefix(first + k, frame)
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
        slot.x.copy_(frame[None])
        self.parts["prefix"].call(
            ("prefix", index % len(self.slots)),
            {"x": slot.x, "current6": slot.current6, "c3": slot.c3, "c4": slot.c4},
            self.stream,
        )
        done.add("prefix")

    def _reverse(self, index: int, has_future: bool) -> None:
        slot, ring = self._slot(index), len(self.slots)
        future = self._slot(index + 1) if has_future else None
        self.parts["revcore"].call(
            ("revcore", index % ring, has_future),
            {
                "current6": slot.current6,
                "c3": slot.c3,
                "c4": slot.c4,
                "state_future": self._zero_state if future is None else future.future,
                "valid_future": self._off if future is None else self._on,
                "state": slot.future,
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
                "x": slot.x,
                "current6": slot.current6,
                "c3": slot.c3,
                "c4": slot.c4,
                "state_past": self._zero_state if before is None else before.forward,
                "valid_past": self._off if before is None else self._on,
                "state_future": self._zero_state if after is None else after.future,
                "valid_future": self._off if after is None else self._on,
                "output": self._output,
                "state": slot.forward,
            },
            self.stream,
        )


def open_session(engine: TensorRTEngine, gpu_index: int) -> RSRSession:
    """The session of an RSR engine set on a GPU: its three engines deserialized, each
    with an execution context."""
    return open_set(engine, gpu_index, RSRSession.ARCHITECTURE, RSRSession)


_sessions: SharedCache[RSRSession] = SharedCache(RSRSession.close, built_on=engines)


@contextmanager
def use_rsr_session(
    engine: TensorRTEngine, gpu_index: int, node_id: NodeId
) -> Iterator[RSRSession]:
    """The session of an RSR engine set on a GPU for `node_id`, kept across runs, one
    per node, as rdual_session.use_rdual_session keeps RDUAL's."""
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
    """`img` upscaled by an RSR engine set on the temporal path (temporal_window), its
    session the executor, kept for the node; when the GPU's memory runs out, the window
    is shortened, and past the last step SET_EXHAUSTED is raised."""
    with use_rsr_session(engine, gpu_index, context.node_id) as session:
        return temporal_upscale(
            context, img, session, report, padding, set_fit(SET_EXHAUSTED)
        )


__all__ = [
    "RSRSession",
    "open_session",
    "temporal_engine_upscale",
    "use_rsr_session",
]
