"""The order in which Temporal Fix's two routes overlap a window with the transfers around
it (tensorrt/temporal_fix_session.TemporalFixSession and temporal_fix_eager.EagerPipeline):
the node's loop submits the next item's window before it collects this one's result, so
the next window computes while this result downloads and while the loop reads and uploads
the frame after it; the loop reads one frame further for that
(temporal_fix.FrameWindow.gather_ahead).

This module imports neither TensorRT nor ONNX: the PyTorch route runs without them."""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch

from .utils import Output

# device_streams: window, download and capture (PyTorch route), download (engine route)
STREAMS = 4


@cache
def device_streams(device: torch.device) -> tuple[torch.cuda.Stream, ...]:
    """The streams of both routes on `device`: the PyTorch route's window, download and
    graph capture streams (temporal_fix_eager), and the engine sessions' download
    stream. Made once: PyTorch's caching allocator keeps the blocks a stream allocated
    for that stream, so streams made per loop run would each keep a window's
    activations reserved (measured at 1920x1080: about 300 MiB more per run).

    Streams of our own (non-blocking, as the TensorRT sessions' capture streams), never
    destroyed: torch.cuda.Stream() hands out the device's pooled streams in turn, so a
    pooled stream can be another user's at the same time, and a capture on a stream
    that another thread enqueues on takes that work into its graph. Without CUDA
    Python, which comes with the TensorRT package (so never on the engine route), the
    pooled streams are the ones there are."""
    try:
        from cuda.bindings import runtime as cudart
    except ModuleNotFoundError:
        return tuple(torch.cuda.Stream(device=device) for _ in range(STREAMS))
    from ..tensorrt.memory import check_cuda

    def own() -> torch.cuda.Stream:
        with torch.cuda.device(device):
            handle = check_cuda(
                cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking)
            )
        return torch.cuda.ExternalStream(int(handle), device=device)

    return tuple(own() for _ in range(STREAMS))


@dataclass(frozen=True)
class Ticket:
    """A submitted window: the loop run it belongs to (`owner`, a weak reference: a
    stopped run's window does not keep that run's frames) and its item, the output
    buffer it writes and the event of its call's end on the route's stream."""

    owner: weakref.ref[Any]
    index: int
    output: torch.Tensor
    done: torch.cuda.Event


class Pipelined(ABC):
    """A route that submits a window without waiting for it and collects its result
    later; `pending` is the window submitted for the next item."""

    pending: Ticket | None

    @abstractmethod
    def submit(
        self, owner: object, index: int, frames: Sequence[torch.Tensor]
    ) -> Ticket:
        """Item `index`'s window enqueued, without waiting for it."""

    @abstractmethod
    def collect(self, ticket: Ticket) -> Output:
        """A submitted window's result on its way to the host."""

    def pipelined(
        self,
        owner: object,
        index: int,
        frames: Callable[[], Sequence[torch.Tensor]],
        ahead: Callable[[], Sequence[torch.Tensor] | None],
    ) -> Output:
        """Item `index`'s result in the loop run `owner` (the node's FrameWindow: a
        window submitted for another run is never taken), with the next item's window
        submitted before it is collected, so that the next window computes while this
        one's result downloads and the loop reads and uploads the frame after it.
        `frames()` gives this item's window when the last call did not submit it (the
        first item, an item after a seek); `ahead()` gives the next item's window, None
        when there is none (the clip's end, or a loop that cannot read that far ahead).
        A submitted window that is not the next one asked for (a seek) is dropped: its
        call still runs, and its output buffer is written only in the route stream's
        order."""
        ticket = self.pending
        self.pending = None
        if ticket is None or ticket.owner() is not owner or ticket.index != index:
            ticket = self.submit(owner, index, frames())
        following = ahead()
        if following is not None:
            self.pending = self.submit(owner, index + 1, following)
        return self.collect(ticket)


__all__ = ["Pipelined", "Ticket"]
