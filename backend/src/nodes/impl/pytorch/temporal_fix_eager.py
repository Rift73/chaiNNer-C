"""Temporal Fix's PyTorch route: the window computed from each frame's cached pyramid
levels (temporal_fix.LevelCache) by the split network's frame part and window part
(temporal_fix_parts.py) in eager PyTorch, with this route's resizes
(stencil_resize), replayed as CUDA graphs on a GPU (GraphedParts); in a loop on a GPU
the next item's window computes while this item's result downloads (EagerPipeline)."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable, Hashable, Iterator, Sequence
from contextlib import contextmanager
from functools import cache, partial
from threading import Lock
from typing import Any, ClassVar

import torch
from sanic.log import logger

from api import Aborted, NodeContext
from util import held
from vendor.temporalfix import temporalfix_arch as arch_module
from vendor.temporalfix.temporalfix_arch import temporalfix_arch

from ..gpu_setup import gpu_setup_lock
from .temporal_fix import RADIUS, LevelCache
from .temporal_fix_parts import FramePart, WindowPart
from .temporal_fix_pipeline import Pipelined, Ticket, device_streams
from .utils import Output, to_host

# The window part swaps functions of the vendored module while it runs
# (temporal_fix_parts.cached_pyramids, stencil_resizes): one window at a time.
_window_lock = Lock()
# The vendored module's resize: a 5x5 binomial low-pass (zero padding 2) before a
# bilinear reduction, two passes over the full-size input
RESIZE = "_resize_like"
# Per axis: the binomial 5 taps convolved with the 2-tap mean that a bilinear reduction
# by an even integer factor takes at the centre of each block (align_corners=False)
STENCIL = (1.0, 5.0, 10.0, 10.0, 5.0, 1.0)


@cache
def stencil_weight(
    channels: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """stencil_resize's weights, made once per channel count, type and device (a CUDA
    graph's capture allows no copy from pageable memory)."""
    taps = torch.tensor(STENCIL, dtype=torch.float32) / 32
    weight = (taps[:, None] * taps[None, :]).to(dtype).repeat(channels, 1, 1, 1)
    return weight.to(device)


def stencil_resize(
    x: torch.Tensor, size: tuple[int, int], resize: Callable[..., torch.Tensor]
) -> torch.Tensor:
    """The network's resize (`resize`); a reduction by 2, 4 or 8 on both axes computed
    as one 6x6 stencil with that stride on each channel, which reads the input once and
    computes only the output pixels: the low-pass taps of the two centre pixels of each
    block, from the input zero-padded as the low-pass pads it (in FP32 within 4e-7 of
    the two passes, closer to them in FP16). Other sizes keep the two passes."""
    _, channels, h, w = x.shape
    factor = h // size[0]
    if factor not in (2, 4, 8) or factor * size[0] != h or factor * size[1] != w:
        return resize(x, size)
    weight = stencil_weight(channels, x.dtype, x.device)
    # The first output pixel's taps start factor / 2 - 3 pixels into the input.
    pad = 3 - factor // 2
    if pad < 0:
        x = x[..., -pad : h + pad, -pad : w + pad]
        pad = 0
    return torch.nn.functional.conv2d(
        x, weight, stride=factor, padding=pad, groups=channels
    )


@contextmanager
def stencil_resizes() -> Iterator[None]:
    """The vendored module's reductions by 2, 4 and 8 as one stencil each
    (stencil_resize) while the context lasts."""
    resize = getattr(arch_module, RESIZE)
    setattr(arch_module, RESIZE, partial(stencil_resize, resize=resize))
    try:
        yield
    finally:
        setattr(arch_module, RESIZE, resize)


class EagerParts:
    """The network's frame part and window part (temporal_fix_parts) run as eager
    operations, with this route's resizes (stencil_resizes). The model is held weakly:
    the parts are kept by the model (_parts) and must not keep it alive."""

    def __init__(self, model: temporalfix_arch) -> None:
        self._model = weakref.ref(model)

    @property
    def model(self) -> temporalfix_arch:
        model = self._model()
        if model is None:
            raise RuntimeError("Temporal Fix: the parts' model was freed")
        return model

    def frame(self, frame: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """One frame (3, H, W) -> its six pyramid levels, each (1, C, h, w)."""
        with stencil_resizes():
            return FramePart(self.model)(frame[None])

    def window(
        self, frames: torch.Tensor, stacks: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """The window (1, 7, 3, H, W) and its six level stacks (7, C, h, w) -> the
        network's output (1, 3, H, W)."""
        with stencil_resizes():
            return WindowPart(self.model)(frames, *stacks)


class DeviceIndexed(torch.Tensor):
    """A tensor whose indexing by a sequence of positions (the network's supports,
    `frames[:, (0, 1, 2, 4, 5, 6)]`) uses an index tensor kept on its device: indexing
    by a Python sequence copies the positions from pageable memory, which a CUDA graph's
    capture refuses. The results are DeviceIndexed too, so every such indexing on the way
    is served; the index tensors are made in the warm-up before the capture."""

    _device_indices: ClassVar[
        dict[tuple[tuple[int, ...], torch.device], torch.Tensor]
    ] = {}

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if func is torch.Tensor.__getitem__:
            tensor, index = args
            args = (tensor, cls._on_device(index, tensor.device))
        return super().__torch_function__(func, types, args, kwargs or {})

    @classmethod
    def _on_device(cls, index: Any, device: torch.device) -> Any:
        """The network's two indexings by positions served from the device: a list of
        them (`level[supports]`), and a sequence of them inside a tuple index
        (`frames[:, (0, 1, 2, 4, 5, 6)]`). Every other index without such a sequence
        (integers, slices, a tuple of these: `x[0, 1]` is one element) is left as it is;
        a sequence in any other form raises, never giving a different result."""
        if isinstance(index, list):
            return cls._positions(index, device)
        if isinstance(index, tuple):
            return tuple(
                cls._positions(i, device) if isinstance(i, (list, tuple)) else i
                for i in index
            )
        return index

    @classmethod
    def _positions(
        cls, positions: list[Any] | tuple[Any, ...], device: torch.device
    ) -> torch.Tensor:
        if not positions or not all(type(i) is int for i in positions):
            raise TypeError(
                "Temporal Fix's capture indexes by a flat sequence of integer positions"
                f" only, not {positions!r}"
            )
        key = (tuple(positions), device)
        if key not in cls._device_indices:
            cls._device_indices[key] = torch.tensor(list(positions), device=device)
        return cls._device_indices[key]


class GraphedParts:
    """EagerParts' two calls captured once each as a CUDA graph on inputs of their own
    and replayed after: each call copies its inputs in and launches one graph, so the
    host is free while the GPU runs it. The window's output is the graph's own tensor,
    written again by the next replay on the same stream; each frame's levels are copied
    out, since the cache keeps them."""

    def __init__(self, eager: EagerParts, frame: torch.Tensor) -> None:
        self.frame_in = torch.empty_like(frame)
        levels = eager.frame(self.frame_in)
        self.window_in = torch.empty(
            (1, 2 * RADIUS + 1, *frame.shape), dtype=frame.dtype, device=frame.device
        )
        self.stacks_in = [
            torch.empty(
                (2 * RADIUS + 1, *level.shape[1:]),
                dtype=level.dtype,
                device=level.device,
            )
            for level in levels
        ]
        # Lazy initialisation (cuDNN's handles, algorithm choice) outside the capture,
        # on a side stream as the capture's
        window_in = self.window_in.as_subclass(DeviceIndexed)
        stacks_in = [stack.as_subclass(DeviceIndexed) for stack in self.stacks_in]
        side = device_streams(frame.device)[2]
        side.wait_stream(torch.cuda.current_stream(frame.device))
        with torch.cuda.stream(side):
            for _ in range(2):
                eager.frame(self.frame_in)
                eager.window(window_in, stacks_in)
        torch.cuda.current_stream(frame.device).wait_stream(side)
        # What torch.cuda.graph does before a capture, without its garbage collection
        torch.cuda.synchronize(frame.device)
        torch.cuda.empty_cache()
        self.frame_graph = torch.cuda.CUDAGraph()
        self.window_graph = torch.cuda.CUDAGraph()
        # As tensorrt/temporal_session.py captures: this thread's calls only, not while
        # TensorRT makes or frees an engine or a context (gpu_setup_lock), and no garbage
        # collection meanwhile (it can free CUDA objects whose release ends a capture).
        # The stream context is left on every path, a failed capture's included.
        with held(gpu_setup_lock), torch.cuda.stream(side):
            collecting = gc.isenabled()
            gc.disable()
            try:
                self.frame_graph.capture_begin(capture_error_mode="thread_local")
                try:
                    self.frame_out = eager.frame(self.frame_in)
                finally:
                    self.frame_graph.capture_end()
                self.window_graph.capture_begin(
                    self.frame_graph.pool(), capture_error_mode="thread_local"
                )
                try:
                    self.window_out = eager.window(window_in, stacks_in).as_subclass(
                        torch.Tensor
                    )
                finally:
                    self.window_graph.capture_end()
            finally:
                if collecting:
                    gc.enable()

    def frame(self, frame: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self.frame_in.copy_(frame)
        self.frame_graph.replay()
        return tuple(level.clone() for level in self.frame_out)

    def window(
        self, frames: torch.Tensor, stacks: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        self.window_in.copy_(frames)
        for target, stack in zip(self.stacks_in, stacks, strict=True):
            target.copy_(stack)
        self.window_graph.replay()
        return self.window_out


# Each model's parts for the latest frame shape and type it ran: the graphs, or the
# eager parts where a capture failed. Neither holds the model, so an entry goes with its model.
_parts: weakref.WeakKeyDictionary[
    temporalfix_arch, dict[tuple[object, ...], EagerParts | GraphedParts]
] = weakref.WeakKeyDictionary()


def parts_of(
    model: temporalfix_arch, frame: torch.Tensor, graphs: bool
) -> EagerParts | GraphedParts:
    """The parts that run `model` on frames like `frame`: with `graphs` on a GPU,
    replayed CUDA graphs captured at the first window of that shape (eager when the
    capture fails, with the reason logged); otherwise the eager parts, kept nowhere."""
    if not graphs or frame.device.type != "cuda":
        return EagerParts(model)
    key = (tuple(frame.shape), frame.dtype, frame.device)
    known = _parts.setdefault(model, {})
    found = known.get(key)
    if found is not None:
        return found
    # The graphs' pool holds a window's activations: one shape's graphs per model, the
    # latest, so the pool of the shape before goes before this one is captured
    known.clear()
    eager = EagerParts(model)
    parts: EagerParts | GraphedParts = eager
    try:
        parts = GraphedParts(eager, frame)
    except RuntimeError:
        logger.exception(
            "Temporal Fix: the CUDA graphs could not be captured; running eager"
        )
        torch.cuda.synchronize(frame.device)
    known[key] = parts
    return parts


def fix_window(
    model: temporalfix_arch,
    frames: Sequence[torch.Tensor],
    cache: LevelCache,
    graphs: bool = True,
    weights: Hashable = None,
) -> torch.Tensor:
    """The network's output (1, 3, H, W) on the window `frames`, its frames' encoder and
    pyramid levels taken from `cache` (computed once per frame), through parts_of.
    `graphs`: the model is kept from window to window (the node's lease retains it), so
    capturing its graphs pays; a model made for one window runs eager. `weights` names
    the model's weights (the lease's key), so cached levels are used only with the
    weights that computed them; None: they are this model object's."""
    source = weakref.ref(model) if weights is None else weights
    with _window_lock:
        parts = parts_of(model, frames[0], graphs)
        per_frame = [cache.levels(frame, source, parts.frame) for frame in frames]
        stacks = [
            torch.cat([levels[i] for levels in per_frame])
            for i in range(len(per_frame[0]))
        ]
        return parts.window(torch.stack(list(frames))[None], stacks)


class EagerPipeline(Pipelined):
    """Per loop run on a GPU: the PyTorch route computing the next item's window while
    this item's result downloads and the loop reads and uploads the frame after it, in
    the engine session's order (temporal_fix_pipeline.Pipelined). A window runs on a
    stream of its own (`window(frames)`, which the node sets per call: its lease's model)
    into an output of its own; the result downloads on another stream.

    Ordering: the window stream waits for the current stream's work (the frames'
    conversion) before each window; each frame a window reads is recorded on the window
    stream, so its memory is not reused before the window has read it; the download
    stream waits for its window's end, and the window's output is recorded on it. The
    replayed graphs' own inputs and outputs are written only on the window stream, in
    its order, and every cached level is made there.

    The two streams are the device's (device_streams), shared by every pipeline:
    two nodes of a chain, or two chains, enqueue their windows on one stream in the
    order their calls run (one window at a time, under the window part's lock), each
    into an output of its own, and each download waits for its own window's end event;
    the GPU setup lock orders only the captures. The streams live as long as the
    process; they hold no model and no frame."""

    def __init__(self, model: temporalfix_arch, device: torch.device) -> None:
        # The model a submitted window reads: kept until drain() has waited for it
        self.model = model
        self.stream, self.download = device_streams(device)[:2]
        self.pending: Ticket | None = None
        self.window: Callable[[Sequence[torch.Tensor]], torch.Tensor] | None = None

    def drain(self) -> None:
        """Wait for the windows in flight (a run that stopped with one submitted)."""
        self.pending = None
        self.stream.synchronize()
        self.download.synchronize()

    def submit(
        self, owner: object, index: int, frames: Sequence[torch.Tensor]
    ) -> Ticket:
        window = self.window
        assert window is not None
        stream = self.stream
        stream.wait_stream(torch.cuda.current_stream(stream.device))
        for frame in frames:
            frame.record_stream(stream)
        with torch.cuda.stream(stream), torch.inference_mode():
            # Its own memory: a replayed window's output is written again by the next
            output = window(frames).clone()
        done = torch.cuda.Event()
        done.record(stream)
        return Ticket(weakref.ref(owner), index, output, done)

    def collect(self, ticket: Ticket) -> Output:
        download = self.download
        download.wait_event(ticket.done)
        ticket.output.record_stream(download)
        with torch.cuda.stream(download), torch.inference_mode():
            # RGB CHW -> BGR HWC float32, into page-locked memory (as the plain route)
            return to_host(ticket.output, clamp=False)


# Each loop run's pipeline (by the node's FrameWindow); the nodes of this chain run whose
# pipelined route failed, which run the plain route for the rest of it
_pipelines: weakref.WeakKeyDictionary[object, EagerPipeline] = (
    weakref.WeakKeyDictionary()
)
_failed: set[str] = set()
_failed_lock = Lock()


def pipeline_of(
    owner: object, model: temporalfix_arch, device: torch.device
) -> tuple[EagerPipeline, bool]:
    """The pipeline of the loop run `owner` for `model`, and whether it is new (its
    end, end_pipeline, is then to be run when the chain run ends). A pipeline of another
    model (a new lease: another strength) is drained first."""
    found = _pipelines.get(owner)
    if found is not None and found.model is model:
        return found, False
    if found is not None:
        found.drain()
    made = EagerPipeline(model, device)
    _pipelines[owner] = made
    return made, True


def end_pipeline(owner: weakref.ref[object]) -> None:
    """The end of a loop run's pipeline: its windows in flight waited for, then the
    pipeline and the model it kept let go."""
    run = owner()
    found = None if run is None else _pipelines.pop(run, None)
    if found is not None:
        found.drain()


def _forget(node_id: str) -> None:
    with _failed_lock:
        _failed.discard(node_id)


def fix_pipelined(
    context: NodeContext,
    owner: object,
    index: int,
    frames: Sequence[torch.Tensor],
    ahead: Callable[[], Sequence[torch.Tensor] | None],
    model: temporalfix_arch,
    window: Callable[[Sequence[torch.Tensor]], torch.Tensor],
) -> Output:
    """Item `index`'s result through the loop run's pipeline: its window (`frames`,
    gathered before the route runs) submitted unless the last item submitted it, then
    the next item's (`ahead()`) before this one is collected. A failure of the pipeline
    (not of the reader) leaves the node on the plain route for the rest of the chain run,
    this item first, on its own window; an error reading ahead is the reader's, raised
    as it is."""
    from .temporal_fix_engine import ReadAheadError

    def plain() -> Output:
        with torch.inference_mode():
            return to_host(window(frames), clamp=False)

    with _failed_lock:
        failed = context.node_id in _failed
    if failed:
        return plain()

    def read_ahead() -> Sequence[torch.Tensor] | None:
        try:
            return ahead()
        except Aborted:
            raise
        except Exception as e:
            raise ReadAheadError from e

    pipeline, new = pipeline_of(owner, model, frames[0].device)
    if new:
        context.add_cleanup(partial(end_pipeline, weakref.ref(owner)), after="chain")
    pipeline.window = window
    try:
        return pipeline.pipelined(owner, index, lambda: frames, read_ahead)
    except Aborted:
        raise
    except ReadAheadError as e:
        upstream = e.__cause__
    except Exception:
        logger.exception("Temporal Fix: the overlapped PyTorch route failed")
        with _failed_lock:
            _failed.add(context.node_id)
        context.add_cleanup(partial(_forget, context.node_id), after="chain")
        pipeline.drain()
        return plain()
    # The upstream error itself, with its own cause and context (raised outside the
    # handler, which would replace them)
    assert upstream is not None
    raise upstream
