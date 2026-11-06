"""TensorRT inference: one engine, one execution context, one CUDA stream.

Device buffers are allocated once (grown only when a larger tile arrives) and
two pinned host slots let the CPU prepare and blend one tile while the GPU runs
the next. All work for the context is enqueued on a single stream, so the
context is never used by two streams at once.

The item window runs the node for several items at once. Items of one tile share a
session: only their enqueues take turns, so the GPU goes straight from one item's tile
to the next while the items' host work runs beside it. An image of several tiles keeps
its two slots' turns to itself (`TensorRTSession.upscale`).

An allocation that finds no memory takes the session to smaller tiles, on an engine that
has smaller sizes, without disturbing the other items' tiles; its context and stream stay
good. A failed enqueue or wait leaves them in an unknown state: nothing retries on them.
Only an engine built with TensorRT's runtime activation resize, as Build Engine's Dynamic
engines are (engine_builder.activations_follow_tiles), needs less activation memory for a
smaller tile; another saves only the tile's input and output buffers.
"""

from __future__ import annotations

import ctypes
import threading
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from sanic.log import logger

from api import NodeId, Progress
from util import held, waiting

from ..native_framework_images import (
    interleaved_to_bfloat_planes,
    planar_to_interleaved,
)
from ..native_tensors import cast_into, cast_numpy
from .cache import SharedCache, engines
from .dual import input_alignment
from .engine_info import deserialize_engine
from .memory import CudaOutOfMemory, check_cuda, gpu_setup_lock
from .model import TensorRTEngine
from .tiling import (
    TILE_ALIGNMENT,
    ShapeBounds,
    one_tile,
    smaller_tiles,
    tiled_upscale,
)

if TYPE_CHECKING:
    # Annotation only: no new import at runtime.
    from ..upscale.tile_report import TileReport

# CUDA graphs kept per session, one per input shape, and input shapes remembered to
# capture on their second run (see TensorRTSession).
MAX_GRAPHS = 4
MAX_SEEN = 32


class _IoDtype:
    def __init__(self, trt_dtype: Any) -> None:
        self.is_bf16 = trt_dtype == trt.DataType.BF16
        self.storage = (
            np.dtype(np.uint16) if self.is_bf16 else np.dtype(trt.nptype(trt_dtype))
        )


class _Pinned:
    """A grow-only pinned host buffer."""

    def __init__(self) -> None:
        self.ptr = 0
        self.nbytes = 0

    def ensure(self, nbytes: int) -> None:
        if nbytes > self.nbytes:
            self.free()
            self.ptr = check_cuda(cudart.cudaMallocHost(nbytes))
            self.nbytes = nbytes

    def array(self, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
        count = int(np.prod(shape))
        raw = (ctypes.c_byte * (count * dtype.itemsize)).from_address(self.ptr)
        return np.frombuffer(raw, dtype=dtype, count=count).reshape(shape)

    def free(self) -> None:
        if self.ptr:
            check_cuda(cudart.cudaFreeHost(self.ptr))
            self.ptr = 0
            self.nbytes = 0


class _Tile(NamedTuple):
    """A tile in flight: its slot, output shape, and the record (slot, count of tiles
    the slot has had) of the tile enqueued just before it, if any."""

    slot: int
    out_shape: tuple[int, ...]
    before: tuple[int, int] | None


class _ThreadTiles(threading.local):
    """A thread's tiles in flight, oldest first (a thread upscales one image at a time),
    and the input (height, width) of the tile it submitted last."""

    def __init__(self) -> None:
        self.queue: deque[_Tile] = deque()
        self.size = (0, 0)


class TensorRTSession:
    """An execution context on a deserialized engine, ready to upscale tiles
    (implements `TileRunner`)."""

    def __init__(self, engine: TensorRTEngine, gpu_index: int) -> None:
        self._gpu_index = gpu_index
        check_cuda(cudart.cudaSetDevice(gpu_index))
        # The engine Load Engine deserialized, kept alive until `close`.
        self._loaded = engine.acquire(gpu_index, deserialize_engine)
        context: Any = None
        stream: Any = None
        events: list[Any] = []
        try:
            cuda_engine = self._loaded.engine
            # Activation memory is what TensorRT asks for the actual tile shape
            # (`_set_shape`), not more.
            with gpu_setup_lock:
                context = cuda_engine.create_execution_context(
                    trt.ExecutionContextAllocationStrategy.USER_MANAGED
                )
            if context is None:
                raise RuntimeError("Failed to create a TensorRT execution context.")

            inputs: list[str] = []
            outputs: list[str] = []
            for i in range(cuda_engine.num_io_tensors):
                name = cuda_engine.get_tensor_name(i)
                if cuda_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    inputs.append(name)
                else:
                    outputs.append(name)
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError(
                    f"Expected one input and one output tensor, got {inputs} and {outputs}."
                )
            self.input_name, self.output_name = inputs[0], outputs[0]
            self._in = _IoDtype(cuda_engine.get_tensor_dtype(self.input_name))
            self._out = _IoDtype(cuda_engine.get_tensor_dtype(self.output_name))

            mins, _, maxs = cuda_engine.get_tensor_profile_shape(self.input_name, 0)
            # A DUAL or DRFT engine's input names the multiple its sizes must be.
            alignment = input_alignment(self.input_name) or TILE_ALIGNMENT
            self._profile = ShapeBounds(mins[2], mins[3], maxs[2], maxs[3], alignment)
            # The profile, until a run's memory runs out (`_use_smaller_tiles`).
            self.bounds = self._profile

            # Non-blocking: nothing on the legacy stream waits for it. A capture on a
            # blocking stream fails when any thread makes the legacy stream depend on it,
            # which creating an execution context does (cudaError 906).
            stream = check_cuda(
                cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking)
            )
            for _ in range(2):
                events.append(
                    check_cuda(
                        cudart.cudaEventCreateWithFlags(cudart.cudaEventDisableTiming)
                    )
                )
        except BaseException:
            # What was created goes, the context before the engine it came from;
            # unchecked, so a teardown error cannot hide this one.
            for event in events:
                cudart.cudaEventDestroy(event)
            if stream is not None:
                cudart.cudaStreamDestroy(stream)
            with gpu_setup_lock:
                del context
            self._loaded.release()
            raise
        self.context = context
        self._stream = stream
        self._events = events
        # An input shape seen again is captured in a CUDA graph after that run and
        # replayed from then on: TensorRT's per-launch host work happens once (DUAL's AOT
        # plugins cost ~0.3 ms of it per launch, 5x the GPU time without a graph). A
        # capture costs one more enqueue of host work, so a shape seen once (images of
        # assorted sizes) is not captured. Tiles mostly share a shape: a few graphs are
        # kept, the least recently used dropped first.
        self._graphs: OrderedDict[
            tuple[int, ...], tuple[cudart.cudaGraph_t, cudart.cudaGraphExec_t]
        ] = OrderedDict()
        self._seen: OrderedDict[tuple[int, ...], None] = OrderedDict()
        self._graph_failed = False

        self._d_in = 0
        self._d_in_nbytes = 0
        self._d_out = 0
        self._d_out_nbytes = 0
        self._d_act = 0
        self._d_act_nbytes = 0
        self._h_in = [_Pinned(), _Pinned()]
        self._h_out = [_Pinned(), _Pinned()]
        # The slots no tile holds: a tile has one from its submit to its collect.
        self._free = [1, 0]
        self._slot_freed = threading.Condition()
        self._second_tile_waiting = False
        self._tiles = _ThreadTiles()
        # The context, the stream and the device buffers take one caller at a time.
        self._enqueue = threading.Lock()
        self._shape: tuple[int, ...] | None = None
        self._out_shape: tuple[int, ...] = ()
        self._out_nbytes = 0
        self._records = [0, 0]
        self._last_record: tuple[int, int] | None = None
        # The context or the stream failed: no more work goes on them, since an enqueue
        # after a failed one can crash inside TensorRT.
        self._failed = False
        # One image of several tiles at a time: it waits for a second slot while it
        # holds one, which only a tile that needs nothing more may do.
        self._tiled = threading.Lock()

    def _wait_for_stream(self) -> None:
        """The tiles on the stream end. A wait that fails, with any error, reports what a
        launch on the stream met: nothing more goes on it."""
        (error,) = cudart.cudaStreamSynchronize(self._stream)
        self._note_wait(error)
        check_cuda((error,))

    def _note_wait(self, error: Any) -> None:
        if error != cudart.cudaError_t.cudaSuccess:
            self._failed = True

    def _ensure_device(self, in_nbytes: int, out_nbytes: int, act_nbytes: int) -> None:
        if act_nbytes > self._d_act_nbytes or (
            in_nbytes > self._d_in_nbytes or out_nbytes > self._d_out_nbytes
        ):
            # in-flight work may still read or write the old buffers
            self._wait_for_stream()
            # captured graphs hold the old addresses
            self._drop_graphs()
        # A growth that runs out of memory fails its item alone: a buffer it freed reads
        # as none, and the next tile of any size allocates what it needs.
        if act_nbytes > self._d_act_nbytes:
            if self._d_act:
                ptr, self._d_act, self._d_act_nbytes = self._d_act, 0, 0
                check_cuda(cudart.cudaFree(ptr))
            self._d_act = check_cuda(cudart.cudaMalloc(act_nbytes))
            self._d_act_nbytes = act_nbytes
        if act_nbytes:
            self.context.set_device_memory(self._d_act, self._d_act_nbytes)
        if in_nbytes > self._d_in_nbytes:
            if self._d_in:
                ptr, self._d_in, self._d_in_nbytes = self._d_in, 0, 0
                check_cuda(cudart.cudaFree(ptr))
            self._d_in = check_cuda(cudart.cudaMalloc(in_nbytes))
            self._d_in_nbytes = in_nbytes
        if out_nbytes > self._d_out_nbytes:
            if self._d_out:
                ptr, self._d_out, self._d_out_nbytes = self._d_out, 0, 0
                check_cuda(cudart.cudaFree(ptr))
            self._d_out = check_cuda(cudart.cudaMalloc(out_nbytes))
            self._d_out_nbytes = out_nbytes

    def _set_shape(self, in_shape: tuple[int, ...], in_nbytes: int) -> None:
        """Set the context's input shape, the memory it needs and the tensor addresses;
        a tile of the shape set last needs none of this."""
        if in_shape == self._shape:
            return
        self._shape = None  # until every step below has held
        # Another item's tile of the shape before may be in flight: the context changes
        # with nothing running on it. That tile's time, not this node's.
        with waiting():
            self._wait_for_stream()
        if not self.context.set_input_shape(self.input_name, in_shape):
            raise ValueError(
                f"Tile shape {in_shape} is outside the engine's optimization profile."
            )
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if any(d < 0 for d in out_shape):
            raise RuntimeError(f"Unresolved engine output shape {out_shape}.")
        out_nbytes = int(np.prod(out_shape)) * self._out.storage.itemsize
        act_nbytes = self.context.update_device_memory_size_for_shapes()
        self._ensure_device(in_nbytes, out_nbytes, act_nbytes)
        self.context.set_tensor_address(self.input_name, self._d_in)
        self.context.set_tensor_address(self.output_name, self._d_out)
        self._out_shape, self._out_nbytes = out_shape, out_nbytes
        self._shape = in_shape

    def _take_slot(self) -> int:
        """A free slot. An image's second tile in flight goes first: until it has one the
        image cannot collect its first, and items of one tile could take every slot freed
        for a whole batch."""
        second = bool(self._tiles.queue)
        with self._slot_freed:
            if not self._free or (self._second_tile_waiting and not second):
                # Another tile's turn, not this node's time.
                with waiting():
                    if second:
                        self._second_tile_waiting = True
                    while not self._free or (self._second_tile_waiting and not second):
                        self._slot_freed.wait()
                    if second:
                        self._second_tile_waiting = False
                        self._slot_freed.notify_all()
            return self._free.pop()

    def _give_slot(self, slot: int) -> None:
        with self._slot_freed:
            self._free.append(slot)
            self._slot_freed.notify_all()

    def submit(self, tile: np.ndarray) -> None:
        """Enqueue one HWC float32 BGR tile; a thread may have at most two in flight."""
        tiles = self._tiles.queue
        if not tiles:
            # Direct callers may start the next image on another worker thread.
            check_cuda(cudart.cudaSetDevice(self._gpu_index))
            # What an earlier call left on this thread, such as another image's failed
            # allocation: TensorRT's next enqueue reads it and fails. Every call here is
            # checked, so it is old news.
            cudart.cudaGetLastError()
        if len(tiles) >= 2:
            raise RuntimeError("Collect a tile before submitting a third one.")
        h, w, c = tile.shape
        self._tiles.size = (h, w)
        in_shape = (1, c, h, w)
        in_nbytes = int(np.prod(in_shape)) * self._in.storage.itemsize
        slot = self._take_slot()
        try:
            h_in, h_out = self._h_in[slot], self._h_out[slot]
            h_in.ensure(in_nbytes)
            dst = h_in.array(self._in.storage, in_shape)[0]
            if self._in.is_bf16:
                interleaved_to_bfloat_planes(tile, dst, swap_red_blue=c in (3, 4))
            else:
                chw = tile.transpose(2, 0, 1)
                if c == 3:
                    chw = chw[::-1]  # BGR -> RGB
                elif c == 4:
                    chw = chw[[2, 1, 0, 3]]  # BGRA -> RGBA
                cast_into(chw, dst)

            with held(self._enqueue):
                if self._failed:
                    raise RuntimeError(
                        "This TensorRT session failed on another image: run this one"
                        " again."
                    )
                self._set_shape(in_shape, in_nbytes)
                out_shape, out_nbytes = self._out_shape, self._out_nbytes
                h_out.ensure(out_nbytes)
                try:
                    check_cuda(
                        cudart.cudaMemcpyAsync(
                            self._d_in,
                            h_in.ptr,
                            in_nbytes,
                            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                            self._stream,
                        )
                    )
                    self._execute(in_shape)
                    check_cuda(
                        cudart.cudaMemcpyAsync(
                            h_out.ptr,
                            self._d_out,
                            out_nbytes,
                            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                            self._stream,
                        )
                    )
                    check_cuda(cudart.cudaEventRecord(self._events[slot], self._stream))
                except BaseException:
                    self._failed = True
                    raise
                before = self._last_record
                self._records[slot] += 1
                self._last_record = (slot, self._records[slot])
        except BaseException:
            # Part of this tile may be on the stream, using this slot's buffers. Unchecked,
            # so that an error here cannot hide this one.
            self._note_wait(cudart.cudaStreamSynchronize(self._stream)[0])
            self._give_slot(slot)
            raise
        tiles.append(_Tile(slot, out_shape, before))

    @property
    def takes_fp32_into_fp16(self) -> bool:
        """An FP16 engine that takes FP32 up to its first convolution."""
        return self._in.storage == np.float32 and self._out.storage == np.float16

    def model_input(self, tile: np.ndarray) -> np.ndarray:
        """The HWC float32 tile as `submit` gives it to the engine, its order of channels
        aside."""
        if self._in.is_bf16:
            h, w, c = tile.shape
            bits = np.empty((c, h, w), np.uint16)
            interleaved_to_bfloat_planes(tile, bits)
            return bits
        return (
            tile
            if self._in.storage == tile.dtype
            else cast_numpy(tile, self._in.storage)
        )

    def _execute(self, shape: tuple[int, ...]) -> None:
        """Enqueue the engine on the stream: the shape's captured graph when there is
        one, else TensorRT's enqueue, after which a shape seen before is captured."""
        graph = self._graphs.get(shape)
        if graph is not None:
            self._graphs.move_to_end(shape)
            check_cuda(cudart.cudaGraphLaunch(graph[1], self._stream))
            return
        if not self.context.execute_async_v3(int(self._stream)):
            raise RuntimeError("TensorRT failed to enqueue inference.")
        if shape not in self._seen:
            self._seen[shape] = None
            if len(self._seen) > MAX_SEEN:
                self._seen.popitem(last=False)
        elif not self._graph_failed:
            self._capture(shape)

    def _capture(self, shape: tuple[int, ...]) -> None:
        """Capture one execution (not run: the stream only records it). A capture that
        fails, its enqueue refused or not, leaves the session on TensorRT's enqueue, with a
        warning."""
        check_cuda(cudart.cudaStreamSynchronize(self._stream))
        mode = cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
        enqueued = False
        graph: Any = None
        with held(gpu_setup_lock):
            (error,) = cudart.cudaStreamBeginCapture(self._stream, mode)
            if error == cudart.cudaError_t.cudaSuccess:
                try:
                    enqueued = self.context.execute_async_v3(int(self._stream))
                finally:
                    error, graph = cudart.cudaStreamEndCapture(self._stream)
        if error == cudart.cudaError_t.cudaSuccess:
            if enqueued:
                error, graph_exec = cudart.cudaGraphInstantiate(graph, 0)
                if error == cudart.cudaError_t.cudaSuccess:
                    self._graphs[shape] = (graph, graph_exec)
                    if len(self._graphs) > MAX_GRAPHS:
                        self._destroy(self._graphs.popitem(last=False)[1])
                    return
            check_cuda(cudart.cudaGraphDestroy(graph))
        # Taken, so that this thread's next enqueue does not fail on it (the runner's capture).
        (left,) = cudart.cudaGetLastError()
        self._drop_graphs()
        self._graph_failed = True
        logger.warning(
            "TensorRT engine not captured in a CUDA graph (enqueued %s, %s, %s); it runs"
            " without one",
            enqueued,
            error,
            left,
        )

    @staticmethod
    def _destroy(graph: tuple[cudart.cudaGraph_t, cudart.cudaGraphExec_t]) -> None:
        check_cuda(cudart.cudaGraphExecDestroy(graph[1]))
        check_cuda(cudart.cudaGraphDestroy(graph[0]))

    def _drop_graphs(self) -> None:
        while self._graphs:
            self._destroy(self._graphs.popitem()[1])

    def collect(self, height: int, width: int) -> np.ndarray:
        """Wait for this thread's oldest tile; returns its top-left `height` x `width`
        output pixels as a new HWC float32 BGR array."""
        tile = self._tiles.queue.popleft()
        try:
            try:
                if tile.before is not None:
                    # The tile ahead on the stream is that tile's time, not this node's.
                    slot, record = tile.before
                    if self._records[slot] == record:
                        with waiting():
                            check_cuda(cudart.cudaEventSynchronize(self._events[slot]))
                check_cuda(cudart.cudaEventSynchronize(self._events[tile.slot]))
            except BaseException:
                self._failed = True
                raise
            chw = self._h_out[tile.slot].array(self._out.storage, tile.out_shape)[0]
            chw = chw[:, :height, :width]
            # One pass from the pinned planar buffer into interleaved float32, RGB(A)
            # to BGR(A).
            return planar_to_interleaved(
                chw, swap_red_blue=chw.shape[0] in (3, 4), bfloat16=self._out.is_bf16
            )
        finally:
            self._give_slot(tile.slot)

    def _end_tiles(self) -> None:
        """After an error this thread's tiles in flight end and give their slots back, for
        the images still using this session; unchecked, so that an error here cannot hide
        the first."""
        tiles = self._tiles.queue
        while tiles:
            tile = tiles.popleft()
            self._note_wait(cudart.cudaEventSynchronize(self._events[tile.slot])[0])
            self._give_slot(tile.slot)

    def upscale(
        self,
        img: np.ndarray,
        scale: int,
        progress: Progress | None = None,
        report: TileReport | None = None,
        *,
        balanced: bool = False,
        before_clip: Callable[[np.ndarray], None] | None = None,
    ) -> np.ndarray:
        """tiled_upscale's image, clipped (clip=True), beside other threads' images. An
        image that runs out of memory runs again from its start, in smaller tiles if the
        engine has them, and the session's next images use those too."""
        h, w = img.shape[:2]
        while True:
            bounds = self.bounds
            whole = one_tile(h, w, bounds)
            try:
                if whole:
                    result = tiled_upscale(
                        img,
                        scale,
                        bounds,
                        self,
                        progress,
                        model_input=self.model_input,
                        opaque_is_same=self.takes_fp32_into_fp16,
                        clip=True,
                        before_clip=before_clip,
                    )
                else:
                    with held(self._tiled):
                        result = tiled_upscale(
                            img,
                            scale,
                            bounds,
                            self,
                            progress,
                            model_input=self.model_input,
                            opaque_is_same=self.takes_fp32_into_fp16,
                            clip=True,
                            balanced=balanced,
                            before_clip=before_clip,
                        )
            except CudaOutOfMemory as error:
                self._end_tiles()
                if self._failed:
                    raise
                self._use_smaller_tiles(bounds, error)
            except BaseException:
                self._end_tiles()
                raise
            else:
                if report is not None:
                    report.record(*self._tiles.size, whole=whole)
                return result

    def _use_smaller_tiles(self, bounds: ShapeBounds, error: CudaOutOfMemory) -> None:
        """This thread's tile, in `bounds`, found no memory. The device buffers go, so that
        the allocations find their memory: the tiles on the stream end first, their results
        are in their slots, and the next tile of any item allocates what it needs. Smaller
        tiles from now on, unless another item's tile already chose them."""
        height, width = self._tiles.size
        with held(self._enqueue):
            with waiting():
                self._wait_for_stream()
            self._drop_graphs()
            self._shape = None
            buffers = (self._d_act, self._d_in, self._d_out)
            self._d_act = self._d_in = self._d_out = 0
            self._d_act_nbytes = self._d_in_nbytes = self._d_out_nbytes = 0
            for ptr in buffers:
                if ptr:
                    check_cuda(cudart.cudaFree(ptr))
            if self.bounds is bounds:
                smaller = smaller_tiles(bounds, height, width)
                if smaller is None:
                    raise RuntimeError(self._no_memory(height, width)) from error
                logger.warning(
                    "TensorRT ran out of memory on %dx%d px tiles (%s); using at most"
                    " %dx%d px",
                    width,
                    height,
                    error,
                    smaller.max_w,
                    smaller.max_h,
                )
                self.bounds = smaller

    def _reserves_its_largest_size(self) -> bool:
        """Whether the engine needs the same activation memory at its smallest and largest
        size. With the stream idle: this sets the input shape (`_shape` is None)."""
        profile = self._loaded.engine.get_tensor_profile_shape(self.input_name, 0)
        needed = set()
        for shape in (profile[0], profile[2]):
            self.context.set_input_shape(self.input_name, tuple(int(d) for d in shape))
            needed.add(self.context.update_device_memory_size_for_shapes())
        return len(needed) == 1

    def _no_memory(self, height: int, width: int) -> str:
        profile = self._profile
        message = f"There is not enough memory to run this TensorRT engine at {width}x{height} px"
        if profile.min_h == profile.max_h and profile.min_w == profile.max_w:
            return (
                f"{message}, the one size a Fixed engine has. Free some GPU memory, or build"
                " the engine as Dynamic, so that it can use smaller tiles, or as Fixed at a"
                " smaller size."
            )
        if self._reserves_its_largest_size():
            return (
                f"{message}, and smaller tiles do not help: this Dynamic engine takes the"
                " memory of its largest size at any tile size. Free some GPU memory, or"
                " build the engine again: Build Engine's Dynamic engines now take memory by"
                " tile size, except DUAL's and DRFT's, which need a smaller maximum size."
            )
        return (
            f"{message}, its smallest tile size for this image. Free some GPU memory."
        )

    def use_largest_tiles(self) -> None:
        """Back to the profile's tiles, for the next run: the memory may be free by then."""
        self.bounds = self._profile

    def close(self) -> None:
        """Free the session. After a CUDA error any step may raise: every step still
        runs, the context goes before its memory and the engine reference last, and
        the first error is raised at the end."""
        errors: list[Exception] = []

        def attempt(step: Callable[[], object]) -> None:
            try:
                step()
            except Exception as e:
                errors.append(e)

        attempt(lambda: check_cuda(cudart.cudaSetDevice(self._gpu_index)))
        attempt(lambda: check_cuda(cudart.cudaStreamSynchronize(self._stream)))
        attempt(self._drop_graphs)
        for buf in (*self._h_in, *self._h_out):
            attempt(buf.free)
        with gpu_setup_lock:
            del self.context
        for ptr in (self._d_in, self._d_out, self._d_act):
            if ptr:
                attempt(lambda ptr=ptr: check_cuda(cudart.cudaFree(ptr)))
        self._d_in = self._d_out = self._d_act = 0
        self._d_in_nbytes = self._d_out_nbytes = self._d_act_nbytes = 0
        for event in self._events:
            attempt(lambda event=event: check_cuda(cudart.cudaEventDestroy(event)))
        self._events = []
        attempt(lambda: check_cuda(cudart.cudaStreamDestroy(self._stream)))
        self._loaded.release()  # the engine after its context
        if errors:
            raise errors[0]


_sessions: SharedCache[TensorRTSession] = SharedCache(
    TensorRTSession.close, built_on=engines
)


@contextmanager
def use_tensorrt_session(
    engine: TensorRTEngine, gpu_index: int, node_id: NodeId
) -> Iterator[TensorRTSession]:
    """The session for an engine on a GPU, kept across runs for `node_id` (see `cache`),
    shared by the items the window runs at once (its `upscale`)."""
    with _sessions.use(
        (engine, gpu_index),
        node_id,
        lambda: TensorRTSession(engine, gpu_index),
        exclusive=False,
    ) as session:
        check_cuda(cudart.cudaSetDevice(gpu_index))
        yield session
