"""Build Engine's timing cache (the "Timing Cache" setting) is created, reused and
replaced. TensorRT and cuda-python are stand-ins that follow TensorRT's documented
behaviour: `get_timing_cache` returns None unless a cache was set, and
`set_timing_cache` returns False for a cache from another GPU."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

FOREIGN = b"cache from another GPU"


class _TimingCache:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def serialize(self) -> bytes:
        return self.data + b"+timings"


class _Config:
    def __init__(self) -> None:
        self.cache: _TimingCache | None = None

    def set_memory_pool_limit(self, *_: object) -> None:
        pass

    def set_flag(self, _: object) -> None:
        pass

    def clear_flag(self, _: object) -> None:
        pass

    def add_optimization_profile(self, _: object) -> None:
        pass

    def create_timing_cache(self, data: bytes) -> _TimingCache:
        return _TimingCache(bytes(data))

    def set_timing_cache(self, cache: _TimingCache, ignore_mismatch: bool) -> bool:
        if cache.data.startswith(FOREIGN) and not ignore_mismatch:
            return False
        self.cache = cache
        return True

    def get_timing_cache(self) -> _TimingCache | None:
        return self.cache


class _Builder:
    def __init__(self, _logger: object) -> None:
        self.used: list[_TimingCache | None] = []

    def create_network(self, _flags: int) -> object:
        tensor = SimpleNamespace(name="input", dtype="float", shape=(1, 3, 64, 64))
        output = SimpleNamespace(name="output", dtype="float", shape=(1, 3, 128, 128))
        return SimpleNamespace(get_input=lambda _: tensor, get_output=lambda _: output)

    def create_builder_config(self) -> _Config:
        return _Config()

    def build_serialized_network(self, _network: object, config: _Config) -> bytes:
        _BUILDS.append(config.cache)
        return b"engine"


_BUILDS: list[_TimingCache | None] = []


def _fake_tensorrt() -> ModuleType:
    trt = ModuleType("tensorrt")
    trt.__version__ = "10.15.1"
    trt.DataType = SimpleNamespace(FLOAT="float", HALF="half", BF16="bf16")
    trt.BuilderFlag = SimpleNamespace(FP16="fp16", BF16="bf16", TF32="tf32")
    trt.NetworkDefinitionCreationFlag = SimpleNamespace(EXPLICIT_BATCH=0)
    trt.MemoryPoolType = SimpleNamespace(WORKSPACE="workspace")
    trt.Logger = type("Logger", (), {"WARNING": 2, "__init__": lambda self, _: None})
    trt.Builder = _Builder
    trt.OnnxParser = lambda _network, _logger: SimpleNamespace(parse=lambda _: True)
    return trt


@pytest.fixture
def engine_builder(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    cuda = ModuleType("cuda")
    bindings = ModuleType("cuda.bindings")
    runtime = ModuleType("cuda.bindings.runtime")
    runtime.cudaSetDevice = lambda _: (0,)
    cuda.bindings = bindings
    bindings.runtime = runtime
    for name, module in (
        ("tensorrt", _fake_tensorrt()),
        ("cuda", cuda),
        ("cuda.bindings", bindings),
        ("cuda.bindings.runtime", runtime),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    # Import against the stand-ins; undoing the patches drops these imports again.
    package = importlib.import_module("nodes.impl.tensorrt")
    for name in ("engine_builder", "engine_info", "memory"):
        monkeypatch.setitem(sys.modules, f"{package.__name__}.{name}", None)
        del sys.modules[f"{package.__name__}.{name}"]
        monkeypatch.setattr(package, name, None, raising=False)
    module = importlib.import_module("nodes.impl.tensorrt.engine_builder")
    monkeypatch.setattr(module, "check_cuda", lambda _: None)
    monkeypatch.setattr(module, "get_cuda_compute_capability", lambda _: (12, 0))
    monkeypatch.setattr(module, "get_cuda_device_name", lambda _: "Test GPU")
    _BUILDS.clear()
    return module


def _build(engine_builder: ModuleType, cache: Path) -> None:
    config = engine_builder.BuildConfig(
        precision="fp32",
        workspace_size_gb=1.0,
        min_shape=(64, 64),
        opt_shape=(64, 64),
        max_shape=(64, 64),
        use_dynamic_shapes=False,
        allow_tf32=False,
    )
    engine = engine_builder.build_engine_from_onnx(
        b"onnx", config, timing_cache_path=str(cache)
    )
    assert engine.scale == 2


def test_first_build_creates_the_cache(engine_builder: ModuleType, tmp_path: Path):
    cache = tmp_path / "timing.cache"

    _build(engine_builder, cache)

    assert _BUILDS[0] is not None
    assert cache.read_bytes() == b"+timings"


def test_later_builds_reuse_the_cache(engine_builder: ModuleType, tmp_path: Path):
    cache = tmp_path / "timing.cache"
    cache.write_bytes(b"earlier")

    _build(engine_builder, cache)

    assert _BUILDS[0] is not None and _BUILDS[0].data == b"earlier"
    assert cache.read_bytes() == b"earlier+timings"


def test_a_cache_from_another_gpu_is_replaced(
    engine_builder: ModuleType, tmp_path: Path
):
    cache = tmp_path / "timing.cache"
    cache.write_bytes(FOREIGN)

    _build(engine_builder, cache)

    assert _BUILDS[0] is not None and _BUILDS[0].data == b""
    assert cache.read_bytes() == b"+timings"
