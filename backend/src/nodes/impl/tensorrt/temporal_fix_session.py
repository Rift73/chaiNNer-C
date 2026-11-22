"""The run-time session of a Temporal Fix engine set (temporal_fix.py) for the Temporal
Fix node: each output is one call of the set on the window of 7 frames centred on it,
which the node gathers (pytorch/temporal_fix.FrameWindow).

The set is two engines (temporal_fix_split.py), or the frame engine and the window cut
into pieces with kernels between them (tfix_cut.WindowPieces, one call like the window
engine's). The session keeps 7 slots, each one frame's copy and its six pyramid levels.
A frame of the window that no slot holds is copied into a free slot (one no frame of
this window holds) and the frame engine fills that slot's levels; the window engine
then reads each window position's frame and levels from its slot (one binding each),
so a sliding window computes one new frame's levels per output and copies one frame.
A frame is known by its object (a weak reference):
FrameWindow hands on the same tensor for a frame while it keeps it, and a window that
repeats a frame at the clip's ends binds that frame's slot twice.

Both engines' calls are CUDA graphs captured at the first run of a set of bound buffers
and replayed after (temporal_session.Part). A call's graph is looked up by the bound
buffers' addresses themselves, in binding order, so a replay always reads the slots its
window binds: in steady state the 7 rotations of the slots, plus the windows of a
clip's ends and of restarts. The window engine's output is already the node's BGR HWC
FP32 image, copied into an image of its own in page-locked memory before the next call
can write over it (host_copy).

The node's loop runs the session pipelined (temporal_fix_pipeline.Pipelined): it
submits the next item's window before it collects this one's result, so the next window
computes while this result downloads (on a stream of its own, from one of two output
buffers) and while the loop reads and uploads the frame after it; the node's loop reads
one frame further for that (FrameWindow.gather_ahead).

This module must not import TensorRT at its top (the PyTorch package imports it)."""

from __future__ import annotations

import weakref
from collections.abc import Hashable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch

from api import NodeId

from ..pytorch.temporal_fix_pipeline import Pipelined, Ticket, device_streams
from ..pytorch.utils import Output, host_copy
from .cache import SharedCache, engines
from .model import TensorRTEngine
from .rdual_engine_set import unpack
from .temporal_fix_split import FRAME_INPUTS, FRAME_OUTPUTS, FRAMES, window_inputs
from .temporal_session import Part, close_parts, open_set
from .tfix_cut import LAYOUT, WindowPieces


def binding_key(tensors: dict[str, torch.Tensor]) -> tuple[tuple[str, int], ...]:
    """The key of a call's CUDA graph: every binding's name and address."""
    return tuple((name, tensors[name].data_ptr()) for name in sorted(tensors))


class Slot:
    """One frame's copy and its levels, and the frame it holds."""

    def __init__(self, frame_part: Part, device: torch.device) -> None:
        (name,) = FRAME_INPUTS
        self.frame = frame_part.tensor(name, device)
        self.levels = {n: frame_part.tensor(n, device) for n in FRAME_OUTPUTS}
        self.holds: weakref.ref[torch.Tensor] | None = None

    def held(self) -> torch.Tensor | None:
        return None if self.holds is None else self.holds()

    def bindings(self) -> dict[str, torch.Tensor]:
        (name,) = FRAME_INPUTS
        return {name: self.frame, **self.levels}


class TemporalFixSession(Pipelined):
    """The executor of a Temporal Fix engine set on a GPU. Frames are RGB CHW FP16
    tensors in 0..1 on the device, of the size the network runs at."""

    def __init__(
        self,
        parts: dict[str, Part],
        runtime: Any,
        resident: int,
        stream: Any,
        device: torch.device,
        size: tuple[int, int, int],
        key: Hashable,
        data: dict[str, bytes] | None = None,
    ) -> None:
        """`parts` holds the set's engines (open_set), which its runtime outlives: the
        frame and window parts, or the frame part and the window's pieces, whose layout
        and kernels are among the set's `data` parts (tfix_cut.py); `size` is the
        frames' height, width and the set's scale (1)."""
        del resident, size  # one call per frame: there is no window to plan by memory
        self.frame_part = parts["frame"]
        self.window_part: Part | WindowPieces = (
            parts["window"]
            if data is None or LAYOUT not in data
            else WindowPieces(parts, data, device)
        )
        self.parts = parts
        self._runtime = runtime
        self.stream = stream
        self.device = device
        self._key = key
        # A failure past here frees the window's kernels (open_set closes the engines)
        try:
            # The engines run one after the other on the session's stream: one workspace
            self._workspace = torch.empty(
                max(1, self.frame_part.workspace, self.window_part.workspace),
                dtype=torch.uint8,
                device=device,
            )
            self.frame_part.use_memory(self._workspace)
            self.window_part.use_memory(self._workspace)
            self.slots = [Slot(self.frame_part, device) for _ in range(FRAMES)]
            self._output = self.window_part.tensor("output", device)
            # pipelined(): two output buffers, alternating, each with the event of its last
            # download; the download stream; the submitted window of the next item
            self._outputs = [
                self.window_part.tensor("output", device) for _ in range(2)
            ]
            self._downloaded: list[torch.cuda.Event | None] = [None, None]
            self._next = 0
            self.download = device_streams(device)[3]
            self.pending: Ticket | None = None
            # The slots of the last window, by position, and the key of its call
            self.window: list[Slot] = []
            self.window_key: Hashable = ()
        except BaseException:
            if isinstance(self.window_part, WindowPieces):
                self.window_part.close()
            raise

    @property
    def key(self) -> Hashable:
        return self._key

    def _check(self, frames: Sequence[torch.Tensor]) -> None:
        shape = self.slots[0].frame.shape[1:]
        if len(frames) != FRAMES or tuple(frames[0].shape) != tuple(shape):
            raise ValueError(
                f"This Temporal Fix engine set takes {FRAMES} frames of"
                f" {shape[2]}x{shape[1]}, not {len(frames)} of"
                f" {frames[0].shape[2]}x{frames[0].shape[1]}: build one for this size."
            )

    def _slots_of(self, frames: Sequence[torch.Tensor]) -> list[Slot]:
        """Each frame's slot; frames no slot holds go to slots no frame of this window
        holds, the lowest first, and get their levels there (on the session's stream).
        A window that brings more than one new frame (a clip's first window, a restart
        after a seek) starts the slots afresh, filled in the window's order, so the
        slots of a window depend only on where it lies from its restart: the windows'
        bindings (and graphs) are the same few after every seek."""
        found: list[Slot | None] = []
        for frame in frames:
            found.append(next((s for s in self.slots if s.held() is frame), None))
        new = {id(f) for f, slot in zip(frames, found, strict=True) if slot is None}
        if len(new) > 1:
            for slot in self.slots:
                slot.holds = None
            found = [None] * len(frames)
        free = [s for s in self.slots if not any(s is f for f in found)]
        for position, frame in enumerate(frames):
            if found[position] is not None:
                continue
            slot = free.pop(0)
            slot.holds = weakref.ref(frame)
            slot.frame[0].copy_(frame)
            # The frame was made on the other stream: keep its memory until this copy
            # has read it.
            frame.record_stream(self.stream)
            tensors = slot.bindings()
            self.frame_part.call(binding_key(tensors), tensors, self.stream)
            for later in range(position, len(frames)):
                if frames[later] is frame:
                    found[later] = slot
        return [s for s in found if s is not None]

    def run(self, frames: Sequence[torch.Tensor]) -> Output:
        """The centre frame of `frames` stabilized, on its way to the host. The frames
        were made on the current stream, which the session's waits for."""
        with torch.no_grad(), torch.cuda.stream(self._enqueue(frames, self._output)):
            return host_copy(self._output)

    def _enqueue(self, frames: Sequence[torch.Tensor], output: torch.Tensor) -> Any:
        """The window's call into `output`, enqueued on the session's stream after the
        current stream's work (the frames were made there); returns that stream."""
        self._check(frames)
        torch.cuda.set_device(self.device)
        stream = self.stream
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.no_grad(), torch.cuda.stream(stream):
            self.window = self._slots_of(frames)
            names = iter(window_inputs())
            tensors: dict[str, torch.Tensor] = {}
            for slot in self.window:
                tensors[next(names)] = slot.frame
            for slot in self.window:
                for level in FRAME_OUTPUTS:
                    tensors[next(names)] = slot.levels[level]
            tensors["output"] = output
            self.window_key = binding_key(tensors)
            self.window_part.call(self.window_key, tensors, stream)
        return stream

    def submit(
        self, owner: object, index: int, frames: Sequence[torch.Tensor]
    ) -> Ticket:
        """Item `index`'s window enqueued (as run) into the next of two output buffers,
        without waiting for it; collect() takes its result.

        Ordering: every write of the session's device memory (slot copies, both engines)
        is on the session's stream, so a slot or output buffer a window reads or writes
        is overwritten only by work enqueued after it. The one other reader is the
        download stream, which copies an output buffer to the host: the session's stream
        waits for that copy's event before a window writes the same buffer again."""
        output = self._outputs[self._next]
        downloaded = self._downloaded[self._next]
        self._next = 1 - self._next
        if downloaded is not None:
            self.stream.wait_event(downloaded)
        stream = self._enqueue(frames, output)
        done = torch.cuda.Event()
        done.record(stream)
        return Ticket(weakref.ref(owner), index, output, done)

    def collect(self, ticket: Ticket) -> Output:
        """A submitted window's result on its way to the host, copied on the download
        stream once its window is done, while the session's stream goes on with the
        windows submitted after it."""
        download = self.download
        download.wait_event(ticket.done)
        with torch.no_grad(), torch.cuda.stream(download):
            output = host_copy(ticket.output)
        assert output.ready is not None
        buffer = next(k for k, o in enumerate(self._outputs) if o is ticket.output)
        self._downloaded[buffer] = output.ready
        return output

    def close(self) -> None:
        """Frees the graphs and contexts before the engines, the engines before their
        runtime."""
        try:
            self.stream.synchronize()
            self.download.synchronize()
            self.pending = None
            if isinstance(self.window_part, WindowPieces):
                self.window_part.close()
        finally:
            close_parts(self.parts)
            self._runtime = None


def open_session(engine: TensorRTEngine, gpu_index: int) -> TemporalFixSession:
    """The session of a Temporal Fix engine set on a GPU: its engines deserialized, each
    with an execution context."""
    _, data = unpack(engine.bytes)
    extra = {name: part for name, part in data.items() if "." in name}
    return open_set(
        engine, gpu_index, "TemporalFix", partial(TemporalFixSession, data=extra)
    )


_sessions: SharedCache[TemporalFixSession] = SharedCache(
    TemporalFixSession.close, built_on=engines
)


@contextmanager
def use_session(
    engine: TensorRTEngine, gpu_index: int, node_id: NodeId
) -> Iterator[TemporalFixSession]:
    """The session of a Temporal Fix engine set on a GPU for `node_id`, kept across
    runs, one per node, as tfdat_session.use_tfdat_session keeps TFDAT's."""
    with _sessions.use(
        (engine, gpu_index, node_id), node_id, lambda: open_session(engine, gpu_index)
    ) as session:
        yield session


__all__ = ["TemporalFixSession", "Ticket", "binding_key", "open_session", "use_session"]
