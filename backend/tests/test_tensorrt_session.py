"""A TensorRT session gives its engine reference back when its setup fails, and its
close frees everything after a CUDA error. TensorRT and cuda-python are stand-ins: no
GPU is used."""

from __future__ import annotations

import ctypes
import importlib
import sys
import weakref
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from nodes.impl.tensorrt.model import (
    DeserializedEngine,
    TensorRTEngine,
    TensorRTEngineInfo,
)

ILLEGAL_ADDRESS = 700


class _Handle(str):
    """A stream or event handle: its name, and an integer as TensorRT takes a stream."""

    def __int__(self) -> int:
        return 0


class _Cudart(ModuleType):
    """cuda.bindings.runtime: every call succeeds and is recorded, until `fail` names
    it (call name -> successful calls before it fails). Device and pinned memory are
    host buffers, and copies between them are memmoves."""

    def __init__(self) -> None:
        super().__init__("cuda.bindings.runtime")
        self.__dict__.update(
            cudaEventDisableTiming=2,
            cudaError_t=SimpleNamespace(cudaSuccess=0),
            cudaMemcpyKind=SimpleNamespace(
                cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2
            ),
        )
        self.fail: dict[str, int] = {}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.memory: list[ctypes.Array[ctypes.c_char]] = []

    def __getattr__(self, name: str) -> Callable[..., tuple[Any, ...]]:
        if not name.startswith("cuda"):
            raise AttributeError(name)
        return lambda *args: self._call(name, args)

    def _call(self, name: str, args: tuple[Any, ...]) -> tuple[Any, ...]:
        count = sum(1 for called, _ in self.calls if called == name)
        self.calls.append((name, args))
        created = {"cudaStreamCreate": "stream", "cudaEventCreateWithFlags": "event"}
        if name in self.fail and count >= self.fail[name]:
            return (ILLEGAL_ADDRESS, None) if name in created else (ILLEGAL_ADDRESS,)
        if name in ("cudaGetErrorName", "cudaGetErrorString"):
            return 0, b"cudaErrorIllegalAddress"
        if name in created:
            return 0, _Handle(f"{created[name]}{count}")
        if name in ("cudaMalloc", "cudaMallocHost"):
            self.memory.append(ctypes.create_string_buffer(args[0]))
            return 0, ctypes.addressof(self.memory[-1])
        if name == "cudaMemcpyAsync":
            ctypes.memmove(args[0], args[1], args[2])
        return (0,)

    def destroyed(self) -> list[str]:
        return [
            args[0]
            for name, args in self.calls
            if name in ("cudaEventDestroy", "cudaStreamDestroy")
        ]


class _Context:
    pass


class _CudaEngine:
    num_io_tensors = 2

    def __init__(self) -> None:
        self.contexts: list[weakref.ref[_Context]] = []

    def get_tensor_name(self, index: int) -> str:
        return ("input", "output")[index]

    def get_tensor_mode(self, name: str) -> str:
        return "INPUT" if name == "input" else "OUTPUT"

    def get_tensor_dtype(self, _name: str) -> str:
        return "FLOAT"

    def get_tensor_profile_shape(self, _name: str, _profile: int) -> tuple:
        return (1, 3, 64, 64), (1, 3, 128, 128), (1, 3, 256, 256)

    def create_execution_context(self, _strategy: object) -> _Context:
        context = _Context()
        self.contexts.append(weakref.ref(context))
        return context


class _Loaded(DeserializedEngine):
    """Records, at the last release, whether every context was gone by then."""

    def __init__(self, cuda_engine: _CudaEngine) -> None:
        super().__init__(object(), cuda_engine, 0)
        self.contexts_gone_first: bool | None = None

    def release(self) -> None:
        if self._refs == 1:
            self.contexts_gone_first = all(
                ref() is None for ref in self.engine.contexts
            )
        super().release()


def _fake_tensorrt() -> ModuleType:
    trt = ModuleType("tensorrt")
    trt.DataType = SimpleNamespace(FLOAT="FLOAT", HALF="HALF", BF16="BF16")
    trt.nptype = lambda _dtype: np.float32
    trt.ExecutionContextAllocationStrategy = SimpleNamespace(USER_MANAGED=1)
    trt.TensorIOMode = SimpleNamespace(INPUT="INPUT", OUTPUT="OUTPUT")
    severity = SimpleNamespace(INTERNAL_ERROR=0, ERROR=1, WARNING=2, INFO=3, VERBOSE=4)
    trt.ILogger = type("ILogger", (), {"Severity": severity})
    return trt


@pytest.fixture
def cudart() -> _Cudart:
    return _Cudart()


@pytest.fixture
def inference(monkeypatch: pytest.MonkeyPatch, cudart: _Cudart) -> ModuleType:
    cuda = ModuleType("cuda")
    bindings = ModuleType("cuda.bindings")
    cuda.bindings = bindings
    bindings.runtime = cudart
    for name, module in (
        ("tensorrt", _fake_tensorrt()),
        ("cuda", cuda),
        ("cuda.bindings", bindings),
        ("cuda.bindings.runtime", cudart),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    # Import against the stand-ins; undoing the patches drops these imports again.
    package = importlib.import_module("nodes.impl.tensorrt")
    for name in ("inference", "engine_info", "memory", "trt_logger"):
        monkeypatch.setitem(sys.modules, f"{package.__name__}.{name}", None)
        del sys.modules[f"{package.__name__}.{name}"]
        monkeypatch.setattr(package, name, None, raising=False)
    return importlib.import_module("nodes.impl.tensorrt.inference")


def _engine() -> tuple[TensorRTEngine, _Loaded]:
    loaded = _Loaded(_CudaEngine())
    shape = (1, 3, 64, 64)
    info = TensorRTEngineInfo(
        "fp32", 3, 3, 2, "sm_120", "11", True, shape, shape, shape
    )
    return TensorRTEngine(b"engine", info, loaded), loaded


@pytest.mark.parametrize(
    ("failing", "destroyed"),
    [
        ({"cudaStreamCreate": 0}, []),
        # the second event fails: the first event and the stream go
        ({"cudaEventCreateWithFlags": 1}, ["event0", "stream0"]),
    ],
)
def test_a_failed_setup_gives_the_engine_back(
    inference: ModuleType,
    cudart: _Cudart,
    failing: dict[str, int],
    destroyed: list[str],
):
    engine, loaded = _engine()
    cudart.fail = failing

    with pytest.raises(RuntimeError, match="cudaErrorIllegalAddress"):
        inference.TensorRTSession(engine, 0)

    assert cudart.destroyed() == destroyed
    engine.close()  # Load Engine's own reference: the last one
    assert loaded.engine is None
    assert loaded.contexts_gone_first is True


def test_close_after_a_cuda_error_still_frees_everything(
    inference: ModuleType, cudart: _Cudart
):
    engine, loaded = _engine()
    session = inference.TensorRTSession(engine, 0)
    # a sticky error: every later call fails too
    cudart.fail = {
        "cudaStreamSynchronize": 0,
        "cudaEventDestroy": 0,
        "cudaStreamDestroy": 0,
    }

    with pytest.raises(RuntimeError, match="cudaErrorIllegalAddress"):
        session.close()

    assert cudart.destroyed() == ["event0", "event1", "stream0"]
    # the first error is the one raised: its name was read right after the sync
    first = next(
        i for i, (name, _) in enumerate(cudart.calls) if name == "cudaGetErrorName"
    )
    assert cudart.calls[first - 1][0] == "cudaStreamSynchronize"
    engine.close()
    assert loaded.engine is None
    assert loaded.contexts_gone_first is True


class _Weighting(_Context):
    """An engine's execution: 2x nearest, its output channel k times k + 1, so the
    order the planes arrive in shows in the result."""

    def __init__(self) -> None:
        self.addresses: dict[str, int] = {}
        self.shape = (1, 1, 1, 1)

    def set_input_shape(self, _name: str, shape: tuple[int, ...]) -> bool:
        self.shape = shape
        return True

    def get_tensor_shape(self, _name: str) -> tuple[int, ...]:
        n, c, h, w = self.shape
        return n, c, 2 * h, 2 * w

    def update_device_memory_size_for_shapes(self) -> int:
        return 0

    def set_tensor_address(self, name: str, address: int) -> None:
        self.addresses[name] = address

    def execute_async_v3(self, _stream: int) -> bool:
        _, c, h, w = self.shape
        source = (ctypes.c_float * (c * h * w)).from_address(self.addresses["input"])
        planes = np.frombuffer(source, np.float32).reshape(c, h, w)
        weights = np.arange(1, c + 1, dtype=np.float32)[:, None, None]
        result = planes.repeat(2, 1).repeat(2, 2) * weights
        target = (ctypes.c_float * result.size).from_address(self.addresses["output"])
        np.frombuffer(target, np.float32)[:] = result.reshape(-1)
        return True


class _WeightingEngine(_CudaEngine):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def get_tensor_profile_shape(self, _name: str, _profile: int) -> tuple:
        c = self.channels
        return (1, c, 64, 64), (1, c, 128, 128), (1, c, 256, 256)

    def create_execution_context(self, _strategy: object) -> _Context:
        context = _Weighting()
        self.contexts.append(weakref.ref(context))
        return context


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_planes_reach_the_engine_in_rgb_order_and_return_bgr(
    inference: ModuleType, channels: int
):
    # The engine takes and gives RGB(A) planes, as the ONNX it was built from, and as
    # ONNX's and PyTorch's nodes feed it: an RGBA engine's red and blue were swapped.
    from nodes.impl.tensorrt.tiling import tiled_upscale

    loaded = DeserializedEngine(object(), _WeightingEngine(channels), 0)
    shape = (1, channels, 64, 64)
    info = TensorRTEngineInfo(
        "fp32", channels, channels, 2, "sm_120", "11", True, shape, shape, shape
    )
    session = inference.TensorRTSession(TensorRTEngine(b"engine", info, loaded), 0)
    image = np.random.default_rng(31).random((40, 50, channels), np.float32)
    out = tiled_upscale(image, 2, session.bounds, session)
    session.close()

    # BGR(A) image -> RGB(A) planes -> channel k times k + 1 -> BGR(A) result
    order = [2, 1, 0, 3][:channels] if channels > 1 else [0]
    weights = np.empty(channels, np.float32)
    weights[order] = np.arange(1, channels + 1, dtype=np.float32)
    expected = image.repeat(2, 0).repeat(2, 1) * weights
    np.testing.assert_array_equal(out, expected)
