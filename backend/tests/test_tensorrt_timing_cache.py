"""Build Engine's timing cache (the "Timing Cache" setting) is created, reused and
replaced. TensorRT and cuda-python are stand-ins that follow TensorRT's documented
behaviour: `get_timing_cache` returns None unless a cache was set, and
`set_timing_cache` returns False for a cache from another GPU."""

from __future__ import annotations

import hashlib
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
        self.flags = {"tf32"}  # TensorRT's default
        self.workspace = 0
        self.builder_optimization_level = 3

    def set_memory_pool_limit(self, _pool: object, size: int) -> None:
        self.workspace = size

    def set_flag(self, flag: str) -> None:
        self.flags.add(flag)

    def clear_flag(self, flag: str) -> None:
        self.flags.discard(flag)

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
        _CONFIGS.append(config)
        return b"engine"


_BUILDS: list[_TimingCache | None] = []
_CONFIGS: list[_Config] = []


def _fake_tensorrt() -> ModuleType:
    trt = ModuleType("tensorrt")
    trt.__version__ = "10.15.1"
    trt.DataType = SimpleNamespace(FLOAT="float", HALF="half", BF16="bf16")
    trt.BuilderFlag = SimpleNamespace(
        FP16="fp16", BF16="bf16", TF32="tf32", SPARSE_WEIGHTS="sparse"
    )
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
    _CONFIGS.clear()
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


@pytest.mark.parametrize(
    ("tf32", "sparse", "level", "flags"),
    [(False, False, 3, set()), (True, True, 5, {"tf32", "sparse"})],
)
def test_the_build_settings_reach_tensorrt(
    engine_builder: ModuleType,
    tf32: bool,
    sparse: bool,
    level: int,
    flags: set[str],
):
    config = engine_builder.BuildConfig(
        precision="fp32",
        workspace_size_gb=12.5,
        min_shape=(64, 64),
        opt_shape=(64, 64),
        max_shape=(64, 64),
        use_dynamic_shapes=False,
        allow_tf32=tf32,
        optimization_level=level,
        sparse_weights=sparse,
    )
    engine_builder.build_engine_from_onnx(b"onnx", config)
    (built,) = _CONFIGS
    assert built.flags == flags
    assert built.builder_optimization_level == level
    assert built.workspace == int(12.5 * 1024**3)


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


def test_the_cache_file_is_named_by_model_gpu_and_tensorrt_version(
    engine_builder: ModuleType,
):
    name = engine_builder.timing_cache_name
    onnx = b"onnx" * 300
    model = hashlib.md5(onnx[:1024]).hexdigest()[:8]

    assert (
        name(onnx, "NVIDIA GeForce RTX 5090", (12, 0), "11.2.1.2")
        == f"timing_{model}_NVIDIA-GeForce-RTX-5090_sm120_trt11.2.1.2.cache"
    )
    names = {
        name(onnx, "NVIDIA GeForce RTX 5090", (12, 0), "11.2.1.2"),
        name(onnx, "NVIDIA GeForce RTX 4090", (8, 9), "11.2.1.2"),
        name(onnx, "NVIDIA GeForce RTX 5090", (12, 0), "10.15.1.29"),
        name(
            onnx[:1000] + b"other model", "NVIDIA GeForce RTX 5090", (12, 0), "11.2.1.2"
        ),
    }
    assert len(names) == 4
    # Characters a file name cannot hold become dashes.
    assert name(onnx, 'Quadro "A/B": 6000', (8, 6), "11.2.1.2").endswith(
        "_Quadro-A-B-6000_sm86_trt11.2.1.2.cache"
    )
