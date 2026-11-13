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
a later window never writes over what the planner still holds.

This module must not import TensorRT at its top (the PyTorch package imports it)."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional

from api import NodeContext, NodeId
from util import held

from ..upscale.basic_upscale import PaddingType
from ..upscale.temporal_window import Fit, temporal_upscale
from ..upscale.tile_report import TileReport
from ..upscale.tiler import NoTiling
from .cache import SharedCache, engines
from .memory import gpu_setup_lock
from .model import TensorRTEngine
from .rdual_engine_set import unpack

Tensors = dict[str, torch.Tensor]
# The share of the free device memory a window may plan for: PyTorch's allocator holds
# about 5 % more than it hands out (WindowMemory).
ALLOCATOR_SHARE = 0.95
# Room kept for what the process holds outside PyTorch's allocator and that grows during
# a run (TensorRT's and CUDA's own memory): it grew by 0.2-0.5 GB over a 1920x1080 run,
# and without this margin the process ended 0.2 GB over a 7 GB and an 11 GB card.
OUTSIDE_ALLOCATOR_MARGIN = 512 * 1024**2
# A set takes whole frames of its one size, so a frame that does not fit with no frames
# read ahead has no further step on this route.
SET_EXHAUSTED = (
    "This RDUAL engine set does not fit the GPU's memory at its frame size, even with no"
    " frames read ahead. Close other programs that use the GPU, or run the model on"
    " PyTorch (Use TensorRT Engine off), which can tile the frame."
)


@dataclass(frozen=True)
class WindowMemory:
    """A session's device memory: `fixed` bytes whatever the window (TensorRT's runtime,
    the engines and their contexts as measured when they were made, their one working
    memory, the shared buffers, and two images of an output on its way to the host) and
    `per_frame` bytes per frame of a window (its slot of the ring and the frame `convert`
    made; a window of n frames that carries a state takes n + 1), and `carried` bytes for
    the two states a window has at once outside the ring: the one it was given (the
    planner holds it) and the copy it returns. All from the bindings' shapes. Measured
    beyond the ring at 1280x720, 1488x837 and 1920x1080, a window of 2 or 8 frames
    adds to PyTorch's peak exactly the two output images (in `fixed`) and keeps one state
    (56, 76 and 127 MiB); one pair's alignment cues peak lower (25, 36 and 56 MiB) and so
    add nothing. The outputs the planner holds are in host memory. PyTorch's allocator
    holds about 5 % more in blocks it keeps for reuse (at 1488x837, 8 frames: 6636 MiB
    reserved for 6347 MiB allocated)."""

    fixed: int
    per_frame: int
    carried: int

    def window_bytes(self, frames: int) -> int:
        """The bytes of a session that has run windows of up to `frames` frames."""
        return self.fixed + self.carried + (frames + 1) * self.per_frame

    def longest_window(self, free: int) -> int:
        """The most frames a window may hold within `free` bytes; 0 if none fits."""
        return max(0, (free - self.fixed - self.carried) // self.per_frame - 1)


@dataclass(frozen=True)
class Output:
    """A kept frame's output on its way to the host: the BGR HWC float32 image in 0..1,
    whose copy has ended once `ready` has passed (None: already there)."""

    image: torch.Tensor
    ready: Any


class _Part:
    """One engine of the set with its execution context, and the CUDA graphs of its
    calls, one per set of bound buffers."""

    def __init__(self, runtime: Any, data: bytes, name: str) -> None:
        import tensorrt as trt

        self.name = name
        with gpu_setup_lock:
            engine = runtime.deserialize_cuda_engine(data)
        if engine is None:
            raise RuntimeError(
                f"Failed to deserialize the RDUAL engine set's {name} engine. It may have"
                " been built with another TensorRT version or for another GPU: build it"
                " again with Build Engine."
            )
        self.engine = engine
        with gpu_setup_lock:
            self.context = engine.create_execution_context(
                trt.ExecutionContextAllocationStrategy.USER_MANAGED
            )
        if self.context is None:
            raise RuntimeError(f"Failed to create the {name} engine's context.")
        self.names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.workspace = int(engine.device_memory_size_v2)
        self.graphs: dict[Hashable, torch.cuda.CUDAGraph] = {}

    def use_memory(self, memory: torch.Tensor) -> None:
        """Run in `memory`, the working memory every part shares."""
        self.context.set_device_memory(memory.data_ptr(), memory.numel())

    def spec(self, name: str) -> tuple[tuple[int, ...], torch.dtype]:
        """A binding's shape and torch type."""
        import tensorrt as trt

        types = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.BOOL: torch.bool,
        }
        shape = tuple(int(d) for d in self.engine.get_tensor_shape(name))
        return shape, types[self.engine.get_tensor_dtype(name)]

    def tensor(self, name: str, device: torch.device) -> torch.Tensor:
        shape, dtype = self.spec(name)
        return torch.empty(shape, dtype=dtype, device=device)

    def call(self, key: Hashable, tensors: Tensors, stream: torch.cuda.Stream) -> None:
        """The engine on `tensors` (every binding), enqueued on `stream`: the graph of
        `key` (one set of buffers), else a run, after which that graph is captured."""
        graph = self.graphs.get(key)
        if graph is not None:
            graph.replay()
            return
        if set(tensors) != set(self.names):
            raise ValueError(
                f"RDUAL {self.name} engine bindings {sorted(tensors)} are not its"
                f" {sorted(self.names)}"
            )
        for name, tensor in tensors.items():
            shape, dtype = self.spec(name)
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"RDUAL {self.name}/{name}: {tuple(tensor.shape)} {tensor.dtype},"
                    f" the engine takes {shape} {dtype}"
                )
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"TensorRT rejected {self.name}/{name}'s address.")
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError(f"RDUAL {self.name} engine failed to run.")
        stream.synchronize()
        # As inference.py captures: this thread only, and not while TensorRT makes or
        # frees a context or an engine (which synchronizes every stream).
        graph = torch.cuda.CUDAGraph()
        with held(gpu_setup_lock):
            graph.capture_begin(capture_error_mode="thread_local")
            try:
                captured = self.context.execute_async_v3(stream.cuda_stream)
            finally:
                graph.capture_end()
        if not captured:
            raise RuntimeError(f"RDUAL {self.name} engine capture failed.")
        self.graphs[key] = graph

    def close(self) -> None:
        self.graphs.clear()
        with gpu_setup_lock:
            del self.context
            del self.engine


class _Slot:
    """One frame's buffers: its padded input, the prefix's outputs, its reverse and
    forward states, and the flow and packed guidance of the pair (it, the next frame)."""

    def __init__(self, parts: dict[str, _Part], device: torch.device) -> None:
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


def _bytes(tensors: Sequence[torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


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


class RDUALSession:
    """The executor of an engine set on a GPU. Frames are RGB CHW float32 tensors on the
    device, copied there on the session's stream; outputs are `Output`s, each copied to
    the host as soon as the forward sweep made it, while the sweep goes on; the state is
    the BF16 trunk state. Nothing waits for the GPU but `to_image`, for its own output.
    Host buffers are page-locked when the device is a GPU."""

    def __init__(
        self,
        parts: dict[str, _Part],
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
        self.parts = parts
        self._runtime = runtime
        self.stream = stream
        self._key = key
        self.device = device
        self._pinned = device.type == "cuda"
        self.height, self.width, self.scale = size
        # One working memory for every engine, which runs one at a time.
        self._workspace = torch.empty(
            max(1, *(p.workspace for p in parts.values())),
            dtype=torch.uint8,
            device=device,
        )
        for part in parts.values():
            part.use_memory(self._workspace)
        self._pair = parts["flow"].tensor("frames", device)
        self._output = parts["fwdcore"].tensor("output", device)
        self._zero_state = torch.zeros_like(
            parts["revcore"].tensor("state_out", device)
        )
        self._zero_guidance = torch.zeros_like(
            parts["revcore"].tensor("guidance_future", device)
        )
        self._on = torch.ones(1, dtype=torch.bool, device=device)
        self._off = torch.zeros(1, dtype=torch.bool, device=device)
        self._pack_guidance = model_pack_guidance()
        self.slots: list[_Slot] = []
        # The frame indices whose prefix ("prefix") or pair ("pair") a later window
        # may reuse, and the first frame of the window that would continue the last.
        self._done: dict[int, set[str]] = {}
        self._next: int | None = None
        probe = _Slot(parts, device)
        output = 3 * self.height * self.scale * self.width * self.scale * 4
        self.memory = WindowMemory(
            fixed=resident
            + 2 * output
            + _bytes(
                [
                    self._workspace,
                    self._pair,
                    self._output,
                    self._zero_state,
                    self._zero_guidance,
                ]
            ),
            per_frame=_bytes(probe.tensors()) + 3 * self.height * self.width * 4,
            carried=2 * _bytes([self._zero_state]),
        )
        del probe

    @property
    def key(self) -> Hashable:
        return self._key

    def convert(self, frame: np.ndarray) -> torch.Tensor:
        """A BGR HWC float32 frame of the set's size -> RGB CHW in 0..1 on the GPU."""
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"This RDUAL engine set takes {self.width}x{self.height} frames, not"
                f" {frame.shape[1]}x{frame.shape[0]}: build one for this size."
            )
        host = torch.from_numpy(np.ascontiguousarray(frame))
        if self._pinned:
            host = host.pin_memory()
        with torch.cuda.stream(self.stream):
            tensor = host.to(self.device, non_blocking=True)
            return tensor.float().clamp(0, 1).permute(2, 0, 1).flip(0)

    @staticmethod
    def to_image(output: Output) -> np.ndarray:
        """The output's BGR HWC float32 image in 0..1, once its copy has ended."""
        if output.ready is not None:
            output.ready.synchronize()
        return output.image.numpy()

    def _to_host(self, height: int, width: int) -> Output:
        """The forward sweep's output cropped to `height` x `width`, turned to BGR HWC
        in 0..1 on the GPU and copied to the host without waiting for the copy."""
        output = self._output[0, :, :height, :width].permute(1, 2, 0).flip(2)
        image = torch.empty((height, width, 3), device=self.device)
        torch.clamp(output, 0, 1, out=image)
        if not self._pinned:
            return Output(image, None)
        host = torch.empty(image.shape, pin_memory=True)
        host.copy_(image, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(self.stream)
        return Output(host, ready)

    def fitting_ahead(self, shape: tuple[int, int], ahead: int) -> int:
        """The frames read ahead, at most `ahead`, whose window fits the device now, or
        -1 when not even a window of one frame does
        (temporal_window.Executor): this session's memory as it stands (its fixed
        buffers and its ring), plus the device's free memory and what PyTorch keeps
        cached unused, within PyTorch's per-process cap (set_per_process_memory_fraction,
        which the free memory does not show), less OUTSIDE_ALLOCATOR_MARGIN, by
        `memory.longest_window`. A frame of
        another size is the set's to refuse, so it is not capped here."""
        if tuple(shape) != (self.height, self.width):
            return ahead
        free, total = torch.cuda.mem_get_info(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        cached = torch.cuda.memory_reserved(self.device) - allocated
        capped = total * torch.cuda.get_per_process_memory_fraction(self.device)
        room = min(free + cached, capped - allocated)
        # What the session holds now: its fixed buffers, its ring, and once it has run,
        # the states a window carries.
        held = self.memory.fixed + len(self.slots) * self.memory.per_frame
        if self.slots:
            held += self.memory.carried
        budget = int(ALLOCATOR_SHARE * max(0, room)) - OUTSIDE_ALLOCATOR_MARGIN
        frames = self.memory.longest_window(held + budget)
        return min(ahead, frames - 1)

    def _slot(self, index: int) -> _Slot:
        return self.slots[index % len(self.slots)]

    def _ensure_slots(self, count: int) -> None:
        """A ring of at least `count` slots; a larger ring drops every graph and the
        work kept in the old one."""
        if len(self.slots) >= count:
            return
        self.stream.synchronize()
        for part in self.parts.values():
            part.graphs.clear()
        self.slots = []
        self._done = {}
        self._next = None
        self.slots = [_Slot(self.parts, self.device) for _ in range(count)]

    def run(
        self,
        frames: Sequence[torch.Tensor],
        keep: int,
        past: tuple[torch.Tensor, torch.Tensor] | None,
        first: int,
    ) -> tuple[list[Output], torch.Tensor]:
        """One window (temporal_window.Run): `keep` outputs and the state after frame
        keep - 1. `past` is the state and frame an earlier call returned and read, or
        None to start cold."""
        count = len(frames)
        if not 1 <= keep <= count:
            raise ValueError(f"A window of {count} frames cannot keep {keep}.")
        torch.cuda.set_device(self.device)
        try:
            self._ensure_slots(count + 1)
            with torch.no_grad():
                return self._run(frames, keep, past, first)
        except BaseException:
            self._done = {}
            self._next = None
            raise

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

    def close(self) -> None:
        """Frees the graphs before the buffers they write, the contexts before their
        engines and the engines before their runtime."""
        self.stream.synchronize()
        _close(self.parts)
        self.slots = []
        self._runtime = None


def _close(parts: dict[str, _Part]) -> None:
    for part in parts.values():
        part.close()
    parts.clear()


def open_session(engine: TensorRTEngine, gpu_index: int) -> RDUALSession:
    """The session of an engine set on a GPU: its four engines deserialized, each with
    an execution context."""
    import tensorrt as trt

    from .trt_logger import TRT_LOGGER

    info, data = unpack(engine.bytes)
    temporal = info.temporal
    if temporal is None or info.scale is None:
        raise ValueError("An RDUAL session runs an RDUAL engine set.")
    torch.cuda.set_device(gpu_index)
    # TensorRT's runtime and plugins load their kernels into the device as well.
    free = torch.cuda.mem_get_info(gpu_index)[0]
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    runtime = trt.Runtime(TRT_LOGGER)
    parts: dict[str, _Part] = {}
    try:
        for name in temporal.parts:
            parts[name] = _Part(runtime, data[name], name)
        resident = free - torch.cuda.mem_get_info(gpu_index)[0]
        return RDUALSession(
            parts,
            runtime,
            max(0, resident),
            torch.cuda.Stream(device=gpu_index),
            torch.device("cuda", gpu_index),
            (temporal.height, temporal.width, info.scale),
            ("tensorrt", id(engine), gpu_index),
        )
    except BaseException:
        _close(parts)
        raise


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
    from ..pytorch.rdual_window import release_memory
    from ..pytorch.utils import is_out_of_memory

    fit = Fit(
        NoTiling(),
        shorten=True,
        out_of_memory=is_out_of_memory,
        release=release_memory,
        tiles=False,
        exhausted=SET_EXHAUSTED,
    )
    with use_rdual_session(engine, gpu_index, context.node_id) as session:
        return temporal_upscale(context, img, session, report, padding, fit)


__all__ = [
    "Output",
    "RDUALSession",
    "WindowMemory",
    "model_pack_guidance",
    "open_session",
    "pad_frame",
    "temporal_engine_upscale",
    "use_rdual_session",
]
