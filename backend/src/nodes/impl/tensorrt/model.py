# This class defines an interface.
# It is important that it does not contain types that depend on TensorRT.
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal


@dataclass
class TensorRTEngineInfo:
    """Metadata about a TensorRT engine."""

    precision: Literal["fp32", "fp16", "bf16", "int8", "mixed"]
    input_channels: int
    output_channels: int
    scale: int | None
    gpu_architecture: str
    tensorrt_version: str
    has_dynamic_shapes: bool
    min_shape: tuple[int, int, int, int] | None  # (-, channels, width, height)
    opt_shape: tuple[int, int, int, int] | None  # (-, channels, width, height)
    max_shape: tuple[int, int, int, int] | None  # (-, channels, width, height)


class DeserializedEngine:
    """A deserialized engine (`trt.ICudaEngine`) and the runtime it came from.

    Its creator holds the first reference and every session built on it one
    more; the last release frees the engine, then its runtime.
    """

    def __init__(self, runtime: Any, engine: Any, gpu_index: int) -> None:
        self._lock = Lock()
        self._refs = 1
        self.runtime: Any = runtime
        self.engine: Any = engine
        self.gpu_index = gpu_index

    def acquire(self) -> bool:
        """Add a reference; False once the engine is freed."""
        with self._lock:
            if not self._refs:
                return False
            self._refs += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._refs -= 1
            if self._refs:
                return
        self.engine = None  # before the runtime it came from
        self.runtime = None


class TensorRTEngine:
    """
    A wrapper class for TensorRT engine data.

    This class holds the serialized engine bytes and metadata without
    requiring TensorRT to be imported. Load Engine also deserializes it once,
    and sessions create their execution contexts from that engine.
    """

    def __init__(
        self,
        engine_bytes: bytes,
        info: TensorRTEngineInfo,
        deserialized: DeserializedEngine | None = None,
    ):
        self.bytes: bytes = engine_bytes
        self.info: TensorRTEngineInfo = info
        self.deserialized = deserialized

    def acquire(
        self, gpu_index: int, deserialize: Callable[[bytes, int], DeserializedEngine]
    ) -> DeserializedEngine:
        """A reference to this engine deserialized on `gpu_index`; release it when done.

        Only Build Engine's output, or an engine whose Load Engine node was
        cleared, is deserialized here.
        """
        loaded = self.deserialized
        if loaded is not None and loaded.gpu_index == gpu_index and loaded.acquire():
            return loaded
        return deserialize(self.bytes, gpu_index)

    def close(self) -> None:
        """Release the creator's reference to the deserialized engine."""
        if self.deserialized is not None:
            self.deserialized.release()

    @property
    def precision(self) -> Literal["fp32", "fp16", "bf16", "int8", "mixed"]:
        return self.info.precision

    @property
    def scale(self) -> int | None:
        return self.info.scale

    @property
    def input_channels(self) -> int:
        return self.info.input_channels

    @property
    def output_channels(self) -> int:
        return self.info.output_channels

    @property
    def has_dynamic_shapes(self) -> bool:
        return self.info.has_dynamic_shapes
