"""The run-time session of an SMoVSR engine set (smovsr.py): the executor of Upscale
Image's temporal path (temporal_window.Executor), on temporal_session.SetSession.

A window is frames t0..t0+n-1 in order; the first `keep` get outputs and the rest are the
backward sweep's look-ahead. Per window, as spandrel's SMoVSR run_window: the encfwd
engine once per frame (its encoding and the forward sweep's step, its first frame
reading the carried state: none on a cold start), then the backward sweep (bwd) from the
window's last frame (no future there) down to its first, with the head engine on each
kept frame right after its backward step. An absent neighbour is a zero state with its
validity flag false, which selects the model's cold step.

Every frame index owns a slot (frame i in slot i mod slots) holding its input (RGB, the
form the engines read), its encoding and its forward output, so a window that continues
the last one reuses the forward sweep over the frames both read: no frame is encoded or
swept forward twice, only the backward sweep repeats. A frame stays on the host
(page-locked) until its slot takes it, through one staging buffer. The forward states
alternate between two buffers by the frame's parity: the sweep reads only the state
after the last frame it swept. The backward states alternate the same way, and the
backward output is one buffer the head reads at once. A larger ring keeps the slots that
hold work (they are only placed anew), and a window that ran out of memory leaves the
work it finished in place, so the next window always continues the sweep: no state is
ever restored from the one the planner carries. A carried window the session did not run
last (a planner that outlived the session it ran with) starts cold, as after a seek; a
rebuilt session has a key of its own, so the planner itself starts cold for it.

Every engine call is a CUDA graph captured at its first run (one per part, slot, parity
and validity: a step's own graph, so windows of any length and kept count share them)
and replayed after; the three engines share one working memory.
Outputs are copies, so a later window never writes over what the planner still holds.
A ring slot holds 140 bytes per input pixel (the FP32 input 12, the FP16 encoding and
forward output 64 each, both 32 channels the backward sweep and the head read). The
states and handoffs are FP16 in the engines' channel-last layout: only the engines read
them.

This module must not import TensorRT at its top (the PyTorch package imports it)."""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from api import NodeContext, NodeId

from ..pytorch.smovsr_window import log_fewer_ahead, ram_ahead
from ..pytorch.utils import is_out_of_memory, note_pinned
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

# Each session's own part of its key (SMoVSRSession.key)
_tokens = itertools.count()

# A set takes whole frames of its one size, so a frame that does not fit with no frames
# read ahead has no further step on this route.
SET_EXHAUSTED = (
    "This SMoVSR engine set does not fit the GPU's memory at its frame size, even with"
    " no frames read ahead. Close other programs that use the GPU, or run the model on"
    " PyTorch (Use TensorRT Engine off)."
)


class _Slot:
    """One frame's buffers: its input, its encoding and its forward output."""

    def __init__(self, parts: dict[str, Part], device: torch.device) -> None:
        encfwd = parts["encfwd"]
        self.x = encfwd.tensor("x", device)
        self.feat = encfwd.tensor("feat", device)
        self.fwd_out = encfwd.tensor("fwd_out", device)

    def tensors(self) -> list[torch.Tensor]:
        return list(vars(self).values())


class SMoVSRSession(SetSession[_Slot]):
    """The executor of an SMoVSR engine set on a GPU; the state it returns is a token,
    the frame its forward sweep reached (a CPU tensor), which only links the planner's
    windows: the session continues from its own forward state, so its ring holds no slot
    for a carried one. `look_ahead` is spandrel's LOOK_AHEAD, the
    frames after a kept frame that its output reads; its windows grow after a cold start
    (temporal_window.Executor's `grows`), since the state carries exactly whatever their
    sizes."""

    ARCHITECTURE = "SMoVSR"
    OUTPUT_PART = "head"
    FRAMES_ON_DEVICE = False
    CARRY_SLOTS = 0
    grows = True

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
        from spandrel.architectures.SMoVSR import LOOK_AHEAD

        self.look_ahead = LOOK_AHEAD
        self._token = next(_tokens)
        bwd = parts["bwd"]
        self._zero_state = torch.zeros_like(bwd.tensor("state", device))
        self._backward = (bwd.tensor("state", device), bwd.tensor("state", device))
        self._bwd_out = bwd.tensor("bwd_out", device)
        state = parts["encfwd"].tensor("state", device)
        self._forward = (state, torch.empty_like(state))
        # A frame on its way from the host, BGR HWC as it comes
        self._staging = torch.empty(
            (size[0], size[1], 3), dtype=torch.float32, device=device
        )
        # The last frame the forward sweep swept (its state in _forward[index % 2])
        self._swept = -1
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
            [
                self._zero_state,
                *self._backward,
                self._bwd_out,
                *self._forward,
                self._staging,
            ],
            # The token it carries holds no device memory
            torch.empty(0),
        )

    def new_slot(self) -> _Slot:
        return _Slot(self.parts, self.device)

    @property
    def key(self) -> Hashable:
        """The set's key and this session's own: its forward sweep lives in it, so a
        planner that ran with another session (one dropped and rebuilt for the same
        set) starts cold (temporal_window.Executor's `key`). The token is never reused,
        unlike an object's id once the object is freed."""
        return (super().key, self._token)

    def convert(self, frame: np.ndarray) -> torch.Tensor:
        """A BGR HWC float32 frame of the set's size, kept on the host until its slot
        takes it: a copy of its own (never the frame's memory, which its owner may
        free), page-locked by PyTorch's host allocator when the device is a GPU, as
        upload_frame copies a frame."""
        self.check_frame(frame)
        host = torch.from_numpy(np.ascontiguousarray(frame))
        if self.device.type == "cuda":
            note_pinned()
        return torch.empty_like(host, pin_memory=self.device.type == "cuda").copy_(host)

    def _ensure_slots(self, count: int) -> None:
        """A ring of at least `count` slots. The slots of the frames whose work is kept
        move to their places in the larger ring (no copy: they hold fewer consecutive
        frames than the old ring has slots, so their places stay distinct), the old
        slots without work fill free places, and only the missing ones are new: the
        growth allocates `count` less the old ring's slots. Every graph is dropped.
        Nothing changes if a new slot does not fit."""
        if len(self.slots) >= count:
            return
        self.stream.synchronize()
        placed = {i % count: self._slot(i) for i in self._done} if self.slots else {}
        kept = {id(slot) for slot in placed.values()}
        spare = [slot for slot in self.slots if id(slot) not in kept]
        slots = [
            placed.get(k) or (spare.pop() if spare else self.new_slot())
            for k in range(count)
        ]
        for part in self.parts.values():
            part.graphs.clear()
        self.slots = slots

    def _keeps_work(self, error: BaseException) -> bool:
        """Running out of memory leaves the finished work valid: only allocations fail
        (a larger ring, an output on its way to the host), never an engine's writes."""
        return isinstance(error, Exception) and is_out_of_memory(error)

    def fitting_ahead(
        self, shape: tuple[int, int], ahead: int, carry: object = None
    ) -> int:
        """SetSession.fitting_ahead within the item window's RAM budget, planned as the
        model's PyTorch windows plan it (smovsr_window.ram_ahead), and a memory limit
        that cuts the frames ahead below `look_ahead` logged as they log it (the
        temporal path logs a loop that offers fewer). `carry`, the state the planner
        carries into the window (None: a cold start), is one of the states the
        session's memory counts once it has run (`WindowMemory.carried`), so it adds
        nothing here."""
        fitting, limit = ahead, ""
        # Each frame of the window is held twice on the host: the loop's copy and the
        # page-locked one `convert` made, until its slot takes it, in a block of PyTorch's
        # host allocator, which rounds it up to a power of two.
        frame_bytes = shape[0] * shape[1] * 3 * 4
        ram = ram_ahead(
            shape,
            fitting,
            self.scale,
            self.look_ahead,
            copy_bytes=1 << (frame_bytes - 1).bit_length(),
        )
        if ram < fitting:
            fitting, limit = ram, "the RAM budget of the loop's frames and outputs"
        on_gpu = super().fitting_ahead(shape, fitting)
        if on_gpu < fitting:
            fitting, limit = on_gpu, "the GPU's memory"
        log_fewer_ahead(shape, fitting, ahead, self.look_ahead, limit)
        return fitting

    def _run(
        self,
        frames: Sequence[torch.Tensor],
        keep: int,
        past: tuple[torch.Tensor, torch.Tensor] | None,
        first: int,
    ) -> tuple[list[Output], torch.Tensor]:
        count = len(frames)
        # Only the window this session ran last continues: any other starts cold.
        continuing = past is not None and self._next == first
        if not continuing:
            self._done = {}
            self._swept = first - 1
        # The work of the frames the ring still holds is kept, also past a window
        # shorter than the last one's look-ahead.
        self._done = {
            i: done
            for i, done in self._done.items()
            if first <= i < first + len(self.slots)
        }
        stream = self.stream
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for k, frame in enumerate(frames):
                self._encode_forward(first + k, frame, k > 0 or continuing)
            outputs = []
            height, width = self.height * self.scale, self.width * self.scale
            last = first + count - 1
            for i in range(last, first - 1, -1):
                self._backward_step(i, i < last)
                if i < first + keep:
                    self._head(i)
                    outputs.append(self._to_host(height, width))
            outputs.reverse()
            state = torch.tensor([self._swept])
        self._next = first + keep
        return outputs, state

    def _encode_forward(self, index: int, frame: torch.Tensor, has_past: bool) -> None:
        """Frame `index` (a converted host frame) into its slot, then its encoding and
        forward step from the state after the frame before, the last one swept."""
        done = self._done.setdefault(index, set())
        if "encfwd" in done:
            return
        if index != self._swept + 1:
            raise RuntimeError(
                f"SMoVSR's forward sweep is at frame {self._swept}, not before {index}."
            )
        slot, parity = self._slot(index), index % 2
        self._staging.copy_(frame, non_blocking=True)
        # BGR HWC -> RGB CHW in 0..1, as upload_frame converts a frame
        for channel in range(3):
            torch.clamp(self._staging[..., 2 - channel], 0, 1, out=slot.x[0, channel])
        self.parts["encfwd"].call(
            ("encfwd", index % len(self.slots), parity, has_past),
            {
                "x": slot.x,
                "state_past": self._forward[1 - parity]
                if has_past
                else self._zero_state,
                "valid_past": self._on if has_past else self._off,
                "feat": slot.feat,
                "fwd_out": slot.fwd_out,
                "state": self._forward[parity],
            },
            self.stream,
        )
        self._swept = index
        done.add("encfwd")

    def _backward_step(self, index: int, has_future: bool) -> None:
        slot, parity = self._slot(index), index % 2
        self.parts["bwd"].call(
            ("bwd", index % len(self.slots), parity, has_future),
            {
                "feat": slot.feat,
                "fwd_out": slot.fwd_out,
                "state_future": (
                    self._backward[1 - parity] if has_future else self._zero_state
                ),
                "valid_future": self._on if has_future else self._off,
                "bwd_out": self._bwd_out,
                "state": self._backward[parity],
            },
            self.stream,
        )

    def _head(self, index: int) -> None:
        slot = self._slot(index)
        self.parts["head"].call(
            ("head", index % len(self.slots)),
            {
                "x": slot.x,
                "fwd_out": slot.fwd_out,
                "bwd_out": self._bwd_out,
                "output": self._output,
            },
            self.stream,
        )


def open_session(engine: TensorRTEngine, gpu_index: int) -> SMoVSRSession:
    """The session of an SMoVSR engine set on a GPU: its three engines deserialized,
    each with an execution context."""
    return open_set(engine, gpu_index, SMoVSRSession.ARCHITECTURE, SMoVSRSession)


_sessions: SharedCache[SMoVSRSession] = SharedCache(
    SMoVSRSession.close, built_on=engines
)


@contextmanager
def use_smovsr_session(
    engine: TensorRTEngine, gpu_index: int, node_id: NodeId
) -> Iterator[SMoVSRSession]:
    """The session of an SMoVSR engine set on a GPU for `node_id`, kept across runs,
    one per node, as rsr_session.use_rsr_session keeps RSR's."""
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
    """`img` upscaled by an SMoVSR engine set on the temporal path (temporal_window),
    its session the executor, kept for the node; when the GPU's memory runs out, the
    window is shortened, and past the last step SET_EXHAUSTED is raised."""
    with use_smovsr_session(engine, gpu_index, context.node_id) as session:
        return temporal_upscale(
            context, img, session, report, padding, set_fit(SET_EXHAUSTED)
        )


__all__ = [
    "SMoVSRSession",
    "open_session",
    "temporal_engine_upscale",
    "use_smovsr_session",
]
