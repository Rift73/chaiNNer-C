"""Shape-bound TensorRT CUDA graphs and buffers owned by one leased VOSR session."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import tensorrt as trt


from .vosr2_trt_models import profiles


class Stage:
    def __init__(self, path: Path, logger: trt.ILogger, device: torch.device):
        import tensorrt as trt

        self.name = path.stem
        self.runtime: trt.Runtime | None = trt.Runtime(logger)
        with path.open("rb") as source:

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
    and the session's single stream. Scratch can therefore be shared; growing it
    invalidates every graph before its old address is freed. The image entry point
    clears this collection on geometry changes, bounding the number of shapes.
    """

    def __init__(self, device: torch.device, stream: torch.cuda.Stream):
        self.device = device
        self.stream = stream
        self.memory: torch.Tensor | None = None
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

    def clear(self) -> None:
        self.stream.synchronize()
        self.graphs.clear()
        self.buffers.clear()
        self.copied.clear()
        self.memory = None

    def _copy(
        self, name: str, binding: str, target: torch.Tensor, source: torch.Tensor
    ) -> None:
        if (name, binding) in _RESIDENT_INPUTS:
            if target.data_ptr() != source.data_ptr():
                raise RuntimeError(
                    f"VOSR resident {name}/{binding} binding changed within a geometry."
                )
            return
        if name == "attention" and binding in ("k", "v"):
            key = (binding, target.data_ptr())
            if self.copied.get(key) == self.attention_epoch:
                return
            self.copied[key] = self.attention_epoch
        target.copy_(source)

    def run(self, stage: Stage, arguments: tuple[torch.Tensor, ...]) -> torch.Tensor:
        import tensorrt as trt

        name = stage.name
        engine = stage.engine
        if engine is None or stage.context is None:
            raise RuntimeError("VOSR BF16 session is not activated.")
        if len(arguments) != len(stage.inputs):
            raise ValueError(f"Wrong input count for VOSR {name}.")
        key = (name, tuple(tuple(value.shape) for value in arguments))
        current = torch.cuda.current_stream(self.device)
        if current != self.stream:
            self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            entry = self.graphs.get(key)
            if entry is None:
                context = engine.create_execution_context(
                    trt.ExecutionContextAllocationStrategy.USER_MANAGED
                )
                if context is None:
                    raise RuntimeError(f"VOSR cannot allocate {name} graph context.")
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
                            ("attention_kv", binding)
                            if name == "attention" and binding in ("k", "v")
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
                if self.memory is None or self.memory.numel() < size:
                    self.stream.synchronize()
                    # Destroy captures before freeing their shared scratch address.
                    self.graphs.clear()
                    self.memory = None
                    torch.cuda.empty_cache()
                    self.memory = torch.empty(
                        max(1, size), device=self.device, dtype=torch.uint8
                    )
                context.set_device_memory(self.memory.data_ptr(), self.memory.numel())
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
                        raise RuntimeError(
                            f"TensorRT rejected {name}/{binding} address."
                        )
                # Flush shape-dependent initialization outside stream capture.
                if not context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError(f"VOSR {name} graph initialization failed.")
                self.stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.stream):
                    if not context.execute_async_v3(self.stream.cuda_stream):
                        raise RuntimeError(f"VOSR {name} capture failed.")
                entry = CapturedStage(context, values, graph)
                self.graphs[key] = entry
            for binding, tensor in zip(stage.inputs, arguments, strict=True):
                self._copy(name, binding, entry.values[binding], tensor)
            entry.graph.replay()
            result = entry.values["result"]
        if current != self.stream:
            current.wait_stream(self.stream)
            result.record_stream(current)
        return result
