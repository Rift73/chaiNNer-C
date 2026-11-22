"""What the run-time sessions of the temporal engine sets (rdual_session.py,
rsr_session.py, smovsr_session.py) share: an engine of a set with its context and CUDA
graphs (`Part`), the session's memory arithmetic (`WindowMemory`), its outputs on their
way to the host, and `SetSession`, the executor of Upscale Image's temporal path
(temporal_window.Executor) minus what an architecture's engines compute per window.

Every frame index owns a slot of buffers (a ring of `slots` frames, frame i in slot
i mod slots), so a window that continues the last one (its `past`, starting where that
one's kept frames ended) reuses the work of the frames both read. The ring grows to the
longest window run, never beyond; a larger ring drops every graph. The engines share one
working memory.

This module must not import TensorRT at its top (the PyTorch package imports it)."""

from __future__ import annotations

import gc
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Protocol, TypeVar

import numpy as np
import torch

from util import held

from ..gpu_setup import gpu_setup_lock
from ..pytorch.utils import Output, host_image, to_host, upload_frame
from ..upscale.temporal_window import Fit
from ..upscale.tiler import NoTiling
from .memory import check_cuda
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


@dataclass(frozen=True)
class WindowMemory:
    """A session's device memory: `fixed` bytes whatever the window (TensorRT's runtime,
    the engines and their contexts as measured when they were made, their one working
    memory, the shared buffers, and two images of an output on its way to the host) and
    `per_frame` bytes per frame of a window (its slot of the ring and the frame `convert`
    made; a window of n frames that carries a state takes n + 1), and `carried` bytes for
    the two states a window has at once outside the ring: the one it was given (the
    planner holds it) and the copy it returns. All from the bindings' shapes. Measured
    for RDUAL beyond the ring at 1280x720, 1488x837 and 1920x1080, a window of 2 or 8
    frames adds to PyTorch's peak exactly the two output images (in `fixed`) and keeps
    one state (56, 76 and 127 MiB); one pair's alignment cues peak lower (25, 36 and 56
    MiB) and so add nothing. The outputs the planner holds are in host memory.
    PyTorch's allocator holds about 5 % more in blocks it keeps for reuse (at 1488x837,
    8 frames: 6636 MiB reserved for 6347 MiB allocated)."""

    fixed: int
    per_frame: int
    carried: int
    # The ring's slots beyond a window's frames (SetSession.CARRY_SLOTS)
    extra_slots: int = 1

    def window_bytes(self, frames: int) -> int:
        """The bytes of a session that has run windows of up to `frames` frames."""
        return self.fixed + self.carried + (frames + self.extra_slots) * self.per_frame

    def longest_window(self, free: int) -> int:
        """The most frames a window may hold within `free` bytes; 0 if none fits."""
        return max(
            0, (free - self.fixed - self.carried) // self.per_frame - self.extra_slots
        )


class Part:
    """One engine of a set with its execution context, and the CUDA graphs of its
    calls, one per set of bound buffers, captured on a stream of the part's own.
    `architecture` names the set in messages."""

    def __init__(self, runtime: Any, data: bytes, name: str, architecture: str) -> None:
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self.name = name
        self.architecture = architecture
        with gpu_setup_lock:
            engine = runtime.deserialize_cuda_engine(data)
        if engine is None:
            raise RuntimeError(
                f"Failed to deserialize the {architecture} engine set's {name} engine."
                " It may have been built with another TensorRT version or for another"
                " GPU: build it again with Build Engine."
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
        # Not the session's stream: a page-locked block that a copy on that stream used
        # (the session's frames and outputs), freed during a capture by any thread or by
        # a garbage collection, records an event on it, which would join the graph and
        # fail the event's next query ("CUDA error: invalid argument").
        self._capture_handle = check_cuda(
            cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking)
        )
        self.capture_stream = torch.cuda.ExternalStream(
            int(self._capture_handle), device=torch.cuda.current_device()
        )

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
        label = f"{self.architecture} {self.name}"
        self.bind(tensors)
        if not self.enqueue(stream.cuda_stream):
            raise RuntimeError(f"{label} engine failed to run.")
        stream.synchronize()
        self.graphs[key] = capture_graph(self.enqueue, self, f"{label} engine")

    def bind(self, tensors: Tensors) -> None:
        """Set every binding's address to its tensor's, checked against the engine."""
        label = f"{self.architecture} {self.name}"
        if set(tensors) != set(self.names):
            raise ValueError(
                f"{label} engine bindings {sorted(tensors)} are not its"
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
                    f"{self.architecture} {self.name}/{name}: {tuple(tensor.shape)}"
                    f" {tensor.dtype}, the engine takes {shape} {dtype}"
                )
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"TensorRT rejected {self.name}/{name}'s address.")

    def enqueue(self, stream: int) -> bool:
        """The engine on its bound addresses, enqueued on the stream of handle `stream`."""
        return bool(self.context.execute_async_v3(stream))

    def close(self) -> None:
        from cuda.bindings import runtime as cudart

        self.graphs.clear()
        with gpu_setup_lock:
            del self.context
            del self.engine
        check_cuda(cudart.cudaStreamDestroy(self._capture_handle))


def capture_graph(
    enqueue: Callable[[int], bool], part: Part, label: str
) -> torch.cuda.CUDAGraph:
    """The CUDA graph of `enqueue` (its work enqueued on a stream handle), captured on
    `part`'s capture stream, after a run of the same work has finished. As inference.py
    captures: this thread only, and not while TensorRT makes or frees a context or an
    engine (which synchronizes every stream). The capture records and runs nothing; the
    call's outputs are the run before it, and every replay is launched on the caller's
    stream, in its order. No garbage collection meanwhile: it can free CUDA objects
    (graphs, events, page-locked blocks) whose release ends a capture."""
    graph = torch.cuda.CUDAGraph()
    capture = part.capture_stream
    with held(gpu_setup_lock), torch.cuda.stream(capture):
        collecting = gc.isenabled()
        gc.disable()
        try:
            graph.capture_begin(capture_error_mode="thread_local")
            try:
                captured = enqueue(capture.cuda_stream)
            finally:
                graph.capture_end()
        finally:
            if collecting:
                gc.enable()
    if not captured:
        raise RuntimeError(f"{label} capture failed.")
    return graph


def tensor_bytes(tensors: Sequence[torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


class Slot(Protocol):
    def tensors(self) -> list[torch.Tensor]: ...


SlotT = TypeVar("SlotT", bound=Slot)


class SetSession(Generic[SlotT]):
    """The executor of an engine set on a GPU, whatever its architecture. Frames are RGB
    CHW float32 tensors on the device, copied there on the session's stream; outputs are
    `Output`s, each copied to the host as soon as the forward sweep made it, while the
    sweep goes on. Nothing waits for the GPU but `to_image`, for its own output. Host
    buffers are page-locked when the device is a GPU.

    A subclass names its ARCHITECTURE, makes a frame's slot (`new_slot`) and runs a
    window in `_run`; the "output" binding of its OUTPUT_PART part makes each output
    (its forward sweep's "fwdcore" unless it names another)."""

    ARCHITECTURE: ClassVar[str]
    OUTPUT_PART: ClassVar[str] = "fwdcore"
    # Whether `convert` puts each frame on the device (else it stays on the host until
    # a slot takes it), for the memory a window's frames take there
    FRAMES_ON_DEVICE: ClassVar[bool] = True
    # The ring's slots beyond a window's frames: one for the frame before the window,
    # whose slot holds the state carried into it
    CARRY_SLOTS: ClassVar[int] = 1

    def __init__(
        self,
        parts: dict[str, Part],
        runtime: Any,
        resident: int,
        stream: Any,
        device: torch.device,
        size: tuple[int, int, int],
        key: Hashable,
        shared: Sequence[torch.Tensor],
        state: torch.Tensor,
    ) -> None:
        """`parts` are the set's engines (open_set), which their runtime outlives, and
        `resident` the device bytes they and their contexts took; `size` is the frames'
        height, width and the set's scale. `shared` are the subclass's buffers every
        window shares and `state` one carried state, for the memory arithmetic. The
        engines share one working memory, as they run one at a time."""
        self.parts = parts
        self._runtime = runtime
        self.stream = stream
        self._key = key
        self.device = device
        self.height, self.width, self.scale = size
        self._workspace = torch.empty(
            max(1, *(p.workspace for p in parts.values())),
            dtype=torch.uint8,
            device=device,
        )
        for part in parts.values():
            part.use_memory(self._workspace)
        self._output = parts[self.OUTPUT_PART].tensor("output", device)
        self.slots: list[SlotT] = []
        # The frame indices whose work a later window may reuse (by kind), and the first
        # frame of the window that would continue the last.
        self._done: dict[int, set[str]] = {}
        self._next: int | None = None
        probe = self.new_slot()
        output = 3 * self.height * self.scale * self.width * self.scale * 4
        self.memory = WindowMemory(
            fixed=resident
            + 2 * output
            + tensor_bytes([self._workspace, self._output, *shared]),
            per_frame=tensor_bytes(probe.tensors())
            + (3 * self.height * self.width * 4 if self.FRAMES_ON_DEVICE else 0),
            carried=2 * tensor_bytes([state]),
            extra_slots=self.CARRY_SLOTS,
        )
        del probe

    def new_slot(self) -> SlotT:
        raise NotImplementedError

    @property
    def key(self) -> Hashable:
        return self._key

    def check_frame(self, frame: np.ndarray) -> None:
        """Raises for a frame that is not BGR HWC of the set's size."""
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"This {self.ARCHITECTURE} engine set takes {self.width}x{self.height}"
                f" frames, not {frame.shape[1]}x{frame.shape[0]}: build one for this"
                " size."
            )

    def convert(self, frame: np.ndarray) -> torch.Tensor:
        """A BGR HWC float32 frame of the set's size -> RGB CHW in 0..1 on the GPU."""
        self.check_frame(frame)
        with torch.cuda.stream(self.stream):
            return upload_frame(frame, self.device)

    @staticmethod
    def to_image(output: Output) -> np.ndarray:
        return host_image(output)

    def _to_host(self, height: int, width: int) -> Output:
        """The forward sweep's output cropped to `height` x `width` on its way to the
        host (`to_host`), on the session's stream."""
        return to_host(self._output[0, :, :height, :width])

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

    def _slot(self, index: int) -> SlotT:
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
        self.slots = [self.new_slot() for _ in range(count)]

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
            self._ensure_slots(count + self.CARRY_SLOTS)
            with torch.no_grad():
                return self._run(frames, keep, past, first)
        except BaseException as error:
            if not self._keeps_work(error):
                self._done = {}
                self._next = None
            raise

    def _keeps_work(self, _error: BaseException) -> bool:
        """Whether the work the ring holds stays valid after a window failed with
        `_error` (a later window may then continue it); by default none does."""
        return False

    def _run(
        self,
        frames: Sequence[torch.Tensor],
        keep: int,
        past: tuple[torch.Tensor, torch.Tensor] | None,
        first: int,
    ) -> tuple[list[Output], torch.Tensor]:
        raise NotImplementedError

    def close(self) -> None:
        """Frees the graphs before the buffers they write, the contexts before their
        engines and the engines before their runtime."""
        self.stream.synchronize()
        close_parts(self.parts)
        self.slots = []
        self._runtime = None


def close_parts(parts: dict[str, Part]) -> None:
    for part in parts.values():
        part.close()
    parts.clear()


# A session: a SetSession, or tfdat_session.TFDATSession, which runs one engine per clip
S = TypeVar("S")
# A session class's constructor: the set's parts, their runtime, the device bytes they
# took, the stream, the device, the frames' size and scale, and the session's key
SessionFactory = Callable[
    [dict[str, Part], Any, int, Any, torch.device, tuple[int, int, int], Hashable], S
]


def open_set(
    engine: TensorRTEngine,
    gpu_index: int,
    architecture: str,
    session: SessionFactory[S],
) -> S:
    """The session (made by `session`) of an engine set of `architecture` on a GPU: its
    engines deserialized, each with an execution context."""
    import tensorrt as trt

    from .trt_logger import TRT_LOGGER

    info, data = unpack(engine.bytes)
    temporal = info.temporal
    if temporal is None or info.scale is None or temporal.architecture != architecture:
        raise ValueError(
            f"An {architecture} session runs an {architecture} engine set."
        )
    torch.cuda.set_device(gpu_index)
    # TensorRT's runtime and plugins load their kernels into the device as well.
    free = torch.cuda.mem_get_info(gpu_index)[0]
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    runtime = trt.Runtime(TRT_LOGGER)
    parts: dict[str, Part] = {}
    try:
        for name in temporal.parts:
            # A part whose name holds a dot is data the session reads (tfix_cut.LAYOUT)
            if "." not in name:
                parts[name] = Part(runtime, data[name], name, architecture)
        resident = free - torch.cuda.mem_get_info(gpu_index)[0]
        return session(
            parts,
            runtime,
            max(0, resident),
            torch.cuda.Stream(device=gpu_index),
            torch.device("cuda", gpu_index),
            (temporal.height, temporal.width, info.scale),
            ("tensorrt", id(engine), gpu_index),
        )
    except BaseException:
        close_parts(parts)
        raise


def set_fit(exhausted: str) -> Fit:
    """The temporal path's steps for a set: it takes whole frames of its one size,
    whatever the node's Tile Size, so when the GPU's memory runs out the window is
    shortened, and past the last step `exhausted` is raised."""
    from ..pytorch.rdual_window import release_memory
    from ..pytorch.utils import is_out_of_memory

    return Fit(
        NoTiling(),
        shorten=True,
        out_of_memory=is_out_of_memory,
        release=release_memory,
        tiles=False,
        exhausted=exhausted,
    )


__all__ = [
    "ALLOCATOR_SHARE",
    "OUTSIDE_ALLOCATOR_MARGIN",
    "Output",
    "Part",
    "SetSession",
    "WindowMemory",
    "capture_graph",
    "close_parts",
    "open_set",
    "set_fit",
]
