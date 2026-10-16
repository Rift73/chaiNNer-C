"""A TensorRT session gives its engine reference back when its setup fails. TensorRT
and cuda-python are stand-ins: no GPU is used."""

from __future__ import annotations

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


class _Cudart(ModuleType):
    """cuda.bindings.runtime: every call succeeds and is recorded, until `fail` names
    it (call name -> successful calls before it fails)."""

    def __init__(self) -> None:
        super().__init__("cuda.bindings.runtime")
        self.__dict__.update(
            cudaEventDisableTiming=2, cudaError_t=SimpleNamespace(cudaSuccess=0)
        )
        self.fail: dict[str, int] = {}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

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
            return 0, f"{created[name]}{count}"
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
    for name in ("inference", "engine_info", "memory"):
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
