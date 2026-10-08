"""TensorRT inference: one engine, one execution context, one CUDA stream.

Device buffers are allocated once (grown only when a larger tile arrives) and
two pinned host slots let the CPU prepare and blend one tile while the GPU runs
the next. All work for the context is enqueued on a single stream, so the
context is never used by two streams at once.
"""

from __future__ import annotations

import ctypes
from collections import deque
from contextlib import AbstractContextManager
from typing import Any

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from api import NodeId

from ..native_framework_images import planar_to_interleaved
from ..native_tensors import cast_into
from .cache import SharedCache
from .engine_info import deserialize_engine
from .memory import check_cuda
from .model import TensorRTEngine
from .tiling import ShapeBounds


def _float32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _bf16_bits_to_float32(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.uint32) << np.uint32(16)).view(np.float32)


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


class TensorRTSession:
    """An execution context on a deserialized engine, ready to upscale tiles
    (implements `TileRunner`)."""

    def __init__(self, engine: TensorRTEngine, gpu_index: int) -> None:
        check_cuda(cudart.cudaSetDevice(gpu_index))
        # The engine Load Engine deserialized, kept alive until `close`.
        self._loaded = engine.acquire(gpu_index, deserialize_engine)
        try:
            cuda_engine = self._loaded.engine
            # Activation memory is sized for the actual tile shape, not the
            # profile's maximum (see `_ensure_device`).
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
            self.bounds = ShapeBounds(mins[2], mins[3], maxs[2], maxs[3])
        except BaseException:
            self._loaded.release()
            raise
        self.context = context

        self._stream = check_cuda(cudart.cudaStreamCreate())
        self._events = [
            check_cuda(cudart.cudaEventCreateWithFlags(cudart.cudaEventDisableTiming))
            for _ in range(2)
        ]
        self._d_in = 0
        self._d_in_nbytes = 0
        self._d_out = 0
        self._d_out_nbytes = 0
        self._d_act = 0
        self._d_act_nbytes = 0
        self._h_in = [_Pinned(), _Pinned()]
        self._h_out = [_Pinned(), _Pinned()]
        self._queue: deque[tuple[int, tuple[int, ...]]] = deque()
        self._submitted = 0

    def _ensure_device(self, in_nbytes: int, out_nbytes: int, act_nbytes: int) -> None:
        if act_nbytes > self._d_act_nbytes or (
            in_nbytes > self._d_in_nbytes or out_nbytes > self._d_out_nbytes
        ):
            # in-flight work may still read or write the old buffers
            check_cuda(cudart.cudaStreamSynchronize(self._stream))
        if act_nbytes > self._d_act_nbytes:
            if self._d_act:
                check_cuda(cudart.cudaFree(self._d_act))
            self._d_act = check_cuda(cudart.cudaMalloc(act_nbytes))
            self._d_act_nbytes = act_nbytes
        if act_nbytes:
            self.context.set_device_memory(self._d_act, self._d_act_nbytes)
        if in_nbytes > self._d_in_nbytes:
            if self._d_in:
                check_cuda(cudart.cudaFree(self._d_in))
            self._d_in = check_cuda(cudart.cudaMalloc(in_nbytes))
            self._d_in_nbytes = in_nbytes
        if out_nbytes > self._d_out_nbytes:
            if self._d_out:
                check_cuda(cudart.cudaFree(self._d_out))
            self._d_out = check_cuda(cudart.cudaMalloc(out_nbytes))
            self._d_out_nbytes = out_nbytes

    def submit(self, tile: np.ndarray) -> None:
        """Enqueue one HWC float32 BGR tile; at most two may be in flight."""
        if len(self._queue) >= 2:
            raise RuntimeError("Collect a tile before submitting a third one.")
        h, w, c = tile.shape
        in_shape = (1, c, h, w)
        if not self.context.set_input_shape(self.input_name, in_shape):
            raise ValueError(
                f"Tile shape {in_shape} is outside the engine's optimization profile."
            )
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if any(d < 0 for d in out_shape):
            raise RuntimeError(f"Unresolved engine output shape {out_shape}.")
        in_nbytes = int(np.prod(in_shape)) * self._in.storage.itemsize
        out_nbytes = int(np.prod(out_shape)) * self._out.storage.itemsize
        act_nbytes = self.context.update_device_memory_size_for_shapes()
        self._ensure_device(in_nbytes, out_nbytes, act_nbytes)

        slot = self._submitted % 2
        h_in, h_out = self._h_in[slot], self._h_out[slot]
        h_in.ensure(in_nbytes)
        h_out.ensure(out_nbytes)

        chw = tile.transpose(2, 0, 1)
        if c == 3:
            chw = chw[::-1]  # BGR -> RGB
        dst = h_in.array(self._in.storage, in_shape)[0]
        if self._in.is_bf16:
            dst[...] = _float32_to_bf16_bits(chw)
        else:
            cast_into(chw, dst)

        check_cuda(
            cudart.cudaMemcpyAsync(
                self._d_in,
                h_in.ptr,
                in_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self._stream,
            )
        )
        self.context.set_tensor_address(self.input_name, self._d_in)
        self.context.set_tensor_address(self.output_name, self._d_out)
        if not self.context.execute_async_v3(int(self._stream)):
            raise RuntimeError("TensorRT failed to enqueue inference.")
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
        self._queue.append((slot, out_shape))
        self._submitted += 1

    def collect(self, height: int, width: int) -> np.ndarray:
        """Wait for the oldest tile; returns its top-left `height` x `width`
        output pixels as a new HWC float32 BGR array."""
        slot, out_shape = self._queue.popleft()
        check_cuda(cudart.cudaEventSynchronize(self._events[slot]))
        chw = self._h_out[slot].array(self._out.storage, out_shape)[0]
        chw = chw[:, :height, :width]
        bgr = chw.shape[0] == 3  # RGB -> BGR
        if self._out.is_bf16:
            if bgr:
                chw = chw[::-1]
            return np.ascontiguousarray(_bf16_bits_to_float32(chw).transpose(1, 2, 0))
        # One pass from the pinned planar buffer into interleaved float32.
        return planar_to_interleaved(chw, reverse=bgr)

    def close(self) -> None:
        check_cuda(cudart.cudaStreamSynchronize(self._stream))
        self._queue.clear()
        for buf in (*self._h_in, *self._h_out):
            buf.free()
        del self.context
        for ptr in (self._d_in, self._d_out, self._d_act):
            if ptr:
                check_cuda(cudart.cudaFree(ptr))
        self._d_in = self._d_out = self._d_act = 0
        self._d_in_nbytes = self._d_out_nbytes = self._d_act_nbytes = 0
        for event in self._events:
            check_cuda(cudart.cudaEventDestroy(event))
        self._events = []
        check_cuda(cudart.cudaStreamDestroy(self._stream))
        self._loaded.release()  # the engine after its context


_sessions: SharedCache[TensorRTSession] = SharedCache(TensorRTSession.close)


def use_tensorrt_session(
    engine: TensorRTEngine, gpu_index: int, node_id: NodeId
) -> AbstractContextManager[TensorRTSession]:
    """The session for an engine on a GPU, kept across runs for `node_id` (see `cache`)."""
    return _sessions.use(
        (engine, gpu_index), node_id, lambda: TensorRTSession(engine, gpu_index)
    )
