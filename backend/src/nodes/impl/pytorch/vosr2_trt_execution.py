"""Shape-bound TensorRT CUDA graphs and buffers owned by one leased VOSR session."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import tensorrt as trt


from .vosr2_plan_file import open_plan
from .vosr2_trt_models import profiles


class Stage:
    def __init__(self, path: Path, logger: trt.ILogger, device: torch.device):
        import tensorrt as trt

        self.name = path.name.split(".", 1)[0]
        self.runtime: trt.Runtime | None = trt.Runtime(logger)
        with open_plan(path) as source:

            class Reader(trt.IStreamReaderV2):
                def __init__(self):
                    super().__init__()

                def read(self, size: int, cuda_stream_ptr: int) -> bytes:
                    return source.read(size)

                def seek(self, offset: int, where: trt.SeekPosition) -> None:
                    if where == trt.SeekPosition.SET:
                        source.seek(offset)
                    elif where == trt.SeekPosition.CUR:
                        source.seek(offset, 1)
                    elif where == trt.SeekPosition.END:
                        source.seek(-offset, 2)
                    else:
                        raise ValueError("Invalid VOSR engine seek origin.")

            engine = self.runtime.deserialize_cuda_engine(Reader())
        if engine is None:
            raise RuntimeError(f"Unable to deserialize VOSR stage {self.name}.")
        self.engine: trt.ICudaEngine | None = engine
        if engine.streamable_weights_size:
            engine.weight_streaming_budget_v2 = 0
        self.context: trt.IExecutionContext | None = None
        # Whether a captured graph has `context`; another shape makes its own.
        self.context_taken = False
        self.device = device
        self.inputs = tuple(profiles(self.name))
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        actual = {
            name
            for name in names
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        }
        if actual != set(self.inputs) or set(names) != {*self.inputs, "result"}:
            raise RuntimeError(f"Invalid VOSR {self.name} engine interface.")
        self.types = {trt.float32: torch.float32, trt.bfloat16: torch.bfloat16}
        if any(engine.get_tensor_dtype(name) not in self.types for name in names):
            raise RuntimeError("VOSR BF16 engines may only use BF16/FP32 buffers.")

    def offload(self) -> None:
        self.context = None
        self.context_taken = False
        if self.engine is not None and self.engine.streamable_weights_size:
            self.engine.weight_streaming_budget_v2 = 0

    def close(self) -> None:
        self.offload()
        self.engine = None
        self.runtime = None


ShapeKey = tuple[str, tuple[tuple[int, ...], ...]]

# These inputs are session-owned outputs/constants with stable addresses until
# the geometry is cleared. Consumers can bind them without intermediate copies.
_RESIDENT_INPUTS = {
    ("encoder_qkv", "x"),
    ("decoder_qkv", "x"),
    ("encoder_post", "x"),
    ("decoder_post", "x"),
    ("dit_kv", "conditioning"),
    ("dit_cached", "kv"),
    ("dit_cached", "cos"),
    ("dit_cached", "sin"),
}


@dataclass
class CapturedStage:
    stage: Stage
    context: trt.IExecutionContext
    values: dict[str, torch.Tensor]
    graph: torch.cuda.CUDAGraph


class BoundedQueue:
    """Keep at most two groups of four operations ahead of cancellation checks."""

    def __init__(self) -> None:
        self.count = 0
        self.previous: torch.cuda.Event | None = None

    def advance(self, check: Callable[[], None]) -> None:
        self.count += 1
        if self.count % 4:
            return
        if self.previous is not None:
            self.previous.synchronize()
            check()
        self.previous = torch.cuda.Event()
        self.previous.record()


class StageExecution:
    """Outputs are borrowed until the next call of the same stage and geometry.

    A session lease serializes callers. Every graph uses its own immutable context
    and the session's stream. Scratch can therefore be shared; growing it invalidates
    every graph before its old address is freed. Lanes after the first (launch's
    `lane`) have a stream, scratch and graphs of their own. The image entry point clears this
    collection on geometry changes, bounding the number of shapes.
    """

    def __init__(self, device: torch.device, stream: torch.cuda.Stream):
        self.device = device
        self.stream = stream
        self.memory: torch.Tensor | None = None
        # Lanes after the first
        self.lane_streams: dict[int, torch.cuda.Stream] = {}
        self.lane_memory: dict[int, torch.Tensor] = {}
        self.graphs: dict[ShapeKey, CapturedStage] = {}
        self.buffers: dict[
            tuple[object, tuple[int, ...], torch.dtype], torch.Tensor
        ] = {}
        self.attention_epoch = 0
        self.copied: dict[tuple[str, int], int] = {}

    def buffer(
        self, key: object, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        identity = (key, shape, dtype)
        value = self.buffers.get(identity)
        if value is None:
            value = torch.empty(shape, device=self.device, dtype=dtype)
            self.buffers[identity] = value
        return value

    def _drop_graphs(self) -> None:
        """Destroy the captures; their stages' contexts are free again."""
        for entry in self.graphs.values():
            entry.stage.context_taken = False
        self.graphs.clear()

    def clear(self) -> None:
        self.stream.synchronize()
        self._drop_graphs()
        self.buffers.clear()
        self.copied.clear()
        self.memory = None
        self.lane_memory.clear()

    def _copy(
        self, name: str, binding: str, target: torch.Tensor, source: torch.Tensor
    ) -> None:
        if (name, binding) in _RESIDENT_INPUTS:
            if target.data_ptr() != source.data_ptr():
                raise RuntimeError(
                    f"VOSR resident {name}/{binding} binding changed within a geometry."
                )
            return
        if name.startswith("attention") and binding in ("k", "v"):
            key = (binding, target.data_ptr())
            if self.copied.get(key) == self.attention_epoch:
                return
            self.copied[key] = self.attention_epoch
        target.copy_(source)

    def run(self, stage: Stage, arguments: tuple[torch.Tensor, ...]) -> torch.Tensor:
        current = torch.cuda.current_stream(self.device)
        if current != self.stream:
            self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            result = self.launch(stage, arguments, self.stream, 0)
        if current != self.stream:
            current.wait_stream(self.stream)
            result.record_stream(current)
        return result

    def lane_stream(self, lane: int) -> torch.cuda.Stream:
        if lane == 0:
            return self.stream
        if lane not in self.lane_streams:
            self.lane_streams[lane] = torch.cuda.Stream(device=self.device)
        return self.lane_streams[lane]

    def launch(
        self,
        stage: Stage,
        arguments: tuple[torch.Tensor, ...],
        stream: torch.cuda.Stream,
        lane: int,
    ) -> torch.Tensor:
        """The call on `stream` (current), with `lane`'s scratch and graphs; the caller
        orders it with other streams."""
        import tensorrt as trt

        name = stage.name
        engine = stage.engine
        if engine is None or stage.context is None:
            raise RuntimeError("VOSR BF16 session is not activated.")
        if len(arguments) != len(stage.inputs):
            raise ValueError(f"Wrong input count for VOSR {name}.")
        key = (
            f"{name}/lane{lane}" if lane else name,
            tuple(tuple(value.shape) for value in arguments),
        )
        fresh = False
        entry = self.graphs.get(key)
        if entry is None:
            from nodes.impl.tensorrt.memory import gpu_setup_lock

            if stage.context_taken:
                with gpu_setup_lock:
                    context = engine.create_execution_context(
                        trt.ExecutionContextAllocationStrategy.USER_MANAGED
                    )
                if context is None:
                    raise RuntimeError(f"VOSR cannot allocate {name} graph context.")
            else:
                context = stage.context
                stage.context_taken = True
            values = {}
            for binding, tensor in zip(stage.inputs, arguments, strict=True):
                dtype = stage.types[engine.get_tensor_dtype(binding)]
                if (name, binding) in _RESIDENT_INPUTS:
                    if (
                        tensor.device != self.device
                        or tensor.dtype != dtype
                        or not tensor.is_contiguous()
                    ):
                        raise ValueError(
                            f"VOSR {name}/{binding} must be a resident buffer with the engine's dtype and layout."
                        )
                    value = tensor
                else:
                    identity = (
                        (name, binding)
                        if name.startswith("attention") and binding in ("k", "v")
                        else (key, binding)
                    )
                    value = self.buffer(identity, tuple(tensor.shape), dtype)
                if not context.set_input_shape(binding, tuple(value.shape)):
                    raise ValueError(
                        f"VOSR {name} does not support shape {tuple(value.shape)}."
                    )
                self._copy(name, binding, value, tensor)
                values[binding] = value
            size = context.update_device_memory_size_for_shapes()
            if size < 0:
                raise RuntimeError(f"Invalid VOSR {name} activation size.")
            memory = self.lane_memory.get(lane) if lane else self.memory
            if memory is None or memory.numel() < size:
                torch.cuda.synchronize(self.device)
                # Destroy captures before freeing their shared scratch address.
                self._drop_graphs()
                memory = None
                if lane:
                    self.lane_memory.pop(lane, None)
                else:
                    self.memory = None
                torch.cuda.empty_cache()
                memory = torch.empty(
                    max(1, size), device=self.device, dtype=torch.uint8
                )
                if lane:
                    self.lane_memory[lane] = memory
                else:
                    self.memory = memory
            context.set_device_memory(memory.data_ptr(), memory.numel())
            shape = tuple(context.get_tensor_shape("result"))
            if any(value <= 0 for value in shape):
                raise RuntimeError(f"Unresolved VOSR {name} output shape.")
            values["result"] = self.buffer(
                (key, "result"),
                shape,
                stage.types[engine.get_tensor_dtype("result")],
            )
            for binding, value in values.items():
                if not context.set_tensor_address(binding, value.data_ptr()):
                    raise RuntimeError(f"TensorRT rejected {name}/{binding} address.")
            # Flush shape-dependent initialization outside stream capture. This
            # run has the call's inputs, so it also gives the call's result.
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError(f"VOSR {name} graph initialization failed.")
            stream.synchronize()
            # What torch.cuda.graph does, without its garbage collection and empty
            # cache; only this thread's calls can spoil the capture.
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin(capture_error_mode="thread_local")
            try:
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError(f"VOSR {name} capture failed.")
            finally:
                graph.capture_end()
            entry = CapturedStage(stage, context, values, graph)
            self.graphs[key] = entry
            fresh = True
        if not fresh:
            for binding, tensor in zip(stage.inputs, arguments, strict=True):
                self._copy(name, binding, entry.values[binding], tensor)
            entry.graph.replay()
        return entry.values["result"]
