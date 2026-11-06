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
from typing import Any

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
        self.workspace: int | None = None  # TensorRT's default: the GPU's memory
        self.builder_optimization_level = 3
        self.tiling_optimization_level = "NONE"  # TensorRT's default
        self.features: set[str] = set()

    def set_preview_feature(self, feature: str, enabled: bool) -> None:
        assert enabled
        self.features.add(feature)

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


class _Profile:
    def set_shape(self, *_args: object) -> None:
        pass


class _Builder:
    def __init__(self, logger: Any) -> None:
        self.logger = logger
        self.used: list[_TimingCache | None] = []

    def create_network(self, _flags: int) -> object:
        tensor = SimpleNamespace(name="input", dtype="float", shape=(1, 3, 64, 64))
        output = SimpleNamespace(name="output", dtype="float", shape=(1, 3, 128, 128))
        return SimpleNamespace(get_input=lambda _: tensor, get_output=lambda _: output)

    def create_builder_config(self) -> _Config:
        return _Config()

    def build_serialized_network(
        self, _network: object, config: _Config
    ) -> bytes | None:
        _BUILDS.append(config.cache)
        _CONFIGS.append(config)
        if _FAILURE:
            for severity, message in _FAILURE:
                self.logger.log(severity, message)
            return None
        return b"engine"


_BUILDS: list[_TimingCache | None] = []
_CONFIGS: list[_Config] = []
# (severity, message) pairs a failing build logs, as TensorRT does
_FAILURE: list[tuple[int, str]] = []


_REGISTRATIONS: list[str] = []


def _fake_tensorrt() -> ModuleType:
    # stand-ins take attributes ModuleType does not declare
    trt: Any = ModuleType("tensorrt")
    trt.__version__ = "10.15.1"
    trt.DataType = SimpleNamespace(FLOAT="float", HALF="half", BF16="bf16")
    trt.BuilderFlag = SimpleNamespace(
        FP16="fp16", BF16="bf16", TF32="tf32", SPARSE_WEIGHTS="sparse"
    )
    trt.PreviewFeature = SimpleNamespace(RUNTIME_ACTIVATION_RESIZE_10_10="resize")
    trt.NetworkDefinitionCreationFlag = SimpleNamespace(EXPLICIT_BATCH=0)
    trt.MemoryPoolType = SimpleNamespace(WORKSPACE="workspace")
    trt.TilingOptimizationLevel = SimpleNamespace(
        NONE="NONE", FAST="FAST", MODERATE="MODERATE", FULL="FULL"
    )
    trt.Logger = type("Logger", (), {"WARNING": 2, "__init__": lambda self, _: None})
    severity = SimpleNamespace(INTERNAL_ERROR=0, ERROR=1, WARNING=2, INFO=3, VERBOSE=4)
    trt.ILogger = type("ILogger", (), {"Severity": severity})
    trt.Builder = _Builder
    trt.OnnxParser = lambda _network, _logger: SimpleNamespace(
        parse=lambda _: _REGISTRATIONS.append("parse") or True
    )
    trt.init_libnvinfer_plugins = lambda _logger, _namespace: _REGISTRATIONS.append(
        "plugins"
    )
    trt.get_plugin_registry = lambda: SimpleNamespace(load_library=lambda _: object())
    return trt


@pytest.fixture
def engine_builder(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    cuda: Any = ModuleType("cuda")
    bindings: Any = ModuleType("cuda.bindings")
    runtime: Any = ModuleType("cuda.bindings.runtime")
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
    for name in ("engine_builder", "engine_info", "memory", "trt_logger"):
        monkeypatch.setitem(sys.modules, f"{package.__name__}.{name}", None)
        del sys.modules[f"{package.__name__}.{name}"]
        monkeypatch.setattr(package, name, None, raising=False)
    module = importlib.import_module("nodes.impl.tensorrt.engine_builder")
    monkeypatch.setattr(module, "check_cuda", lambda _: None)
    monkeypatch.setattr(module, "get_cuda_compute_capability", lambda _: (12, 0))
    monkeypatch.setattr(module, "get_cuda_device_name", lambda _: "Test GPU")
    _BUILDS.clear()
    _CONFIGS.clear()
    _FAILURE.clear()
    return module


def test_a_build_registers_tensorrts_plugins_before_it_parses(
    engine_builder: ModuleType, tmp_path: Path
):
    # TensorRT's parser takes a DFT through FFTPlugin, in TensorRT's own plugin library
    _REGISTRATIONS.clear()
    _build(engine_builder, tmp_path / "cache")
    assert _REGISTRATIONS[:2] == ["plugins", "parse"]


def test_the_runner_registers_tensorrts_plugins(
    engine_builder: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    # The runner deserializes an engine in native code, with no Python session before it
    monkeypatch.setattr(sys.modules["tensorrt"], "__version__", "11.4.0")
    # Importing again also rebinds the package's attribute, which `from nodes.impl.tensorrt
    # import runner` reads; undoing the patches restores both.
    package = importlib.import_module("nodes.impl.tensorrt")
    for name in ("runner", "drft"):
        monkeypatch.delitem(sys.modules, f"{package.__name__}.{name}", raising=False)
        monkeypatch.setattr(package, name, None, raising=False)
    runner = importlib.import_module("nodes.impl.tensorrt.runner")
    _REGISTRATIONS.clear()
    runner.load_tensorrt()
    assert _REGISTRATIONS == ["plugins"]


@pytest.mark.parametrize(
    ("smallest", "largest", "dynamic", "features"),
    [
        ((64, 64), (512, 512), True, {"resize"}),
        ((64, 256), (64, 512), True, {"resize"}),
        # One size has nothing to resize, whichever shape mode gave it
        ((256, 256), (256, 256), True, set()),
        ((256, 256), (256, 256), False, set()),
    ],
)
def test_an_engine_of_several_sizes_takes_activation_memory_by_the_tile(
    engine_builder: ModuleType,
    smallest: tuple[int, int],
    largest: tuple[int, int],
    dynamic: bool,
    features: set[str],
    monkeypatch: pytest.MonkeyPatch,
):
    # The build sets a profile for a Dynamic engine, and for an ONNX of dynamic axes.
    monkeypatch.setattr(
        engine_builder.trt,
        "Builder",
        type(
            "Builder",
            (_Builder,),
            {"create_optimization_profile": lambda _self: _Profile()},
        ),
    )
    config = engine_builder.BuildConfig(
        precision="fp32",
        workspace_size_gb=1.0,
        min_shape=smallest,
        opt_shape=largest,
        max_shape=largest,
        use_dynamic_shapes=dynamic,
        allow_tf32=False,
    )
    engine_builder.build_engine_from_onnx(b"onnx", config)
    assert _CONFIGS[-1].features == features if dynamic else not _CONFIGS[-1].features
    assert engine_builder.activations_follow_tiles(config) == bool(features)


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
    ("tf32", "sparse", "level", "tiling", "flags"),
    [
        (False, False, 3, "none", set()),
        (True, True, 5, "full", {"tf32", "sparse"}),
        (False, False, 3, "moderate", set()),
    ],
)
def test_the_build_settings_reach_tensorrt(
    engine_builder: ModuleType,
    tf32: bool,
    sparse: bool,
    level: int,
    tiling: str,
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
        tiling_optimization=tiling,
    )
    engine_builder.build_engine_from_onnx(b"onnx", config)
    (built,) = _CONFIGS
    assert built.flags == flags
    assert built.builder_optimization_level == level
    assert built.tiling_optimization_level == tiling.upper()
    assert built.workspace == int(12.5 * 1024**3)


def test_an_automatic_workspace_keeps_tensorrts_default(engine_builder: ModuleType):
    config = engine_builder.BuildConfig(
        precision="fp32",
        workspace_size_gb=0,
        min_shape=(64, 64),
        opt_shape=(64, 64),
        max_shape=(64, 64),
        use_dynamic_shapes=False,
        allow_tf32=False,
    )
    engine_builder.build_engine_from_onnx(b"onnx", config)
    (built,) = _CONFIGS
    assert built.workspace is None


def test_a_failed_build_says_why(engine_builder: ModuleType):
    # TensorRT 11.2's messages for an FDAT model at 2048x2048 with a 4 GiB workspace
    _FAILURE.extend(
        [
            (
                2,
                (
                    "UNSUPPORTED_STATE: Skipping tactic 0 due to insufficient memory on"
                    " requested size of 4302958702 detected for tactic 0x0000000000000000."
                ),
            ),
            (
                1,
                (
                    "IBuilder::buildSerializedNetwork: Error Code 10: Internal Error (Could"
                    " not find any implementation for node {ForeignNode[input...output]}.)"
                ),
            ),
        ]
    )
    config = engine_builder.BuildConfig(
        precision="fp32",
        workspace_size_gb=4.0,
        min_shape=(64, 64),
        opt_shape=(64, 64),
        max_shape=(64, 64),
        use_dynamic_shapes=False,
        allow_tf32=False,
    )
    with pytest.raises(RuntimeError) as failure:
        engine_builder.build_engine_from_onnx(b"onnx", config)
    message = str(failure.value)
    assert "Could not find any implementation" in message
    assert "4.01 GiB" in message
    assert "Workspace (GiB) of 4" in message


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


def test_the_cache_is_replaced_through_a_temporary_file(
    engine_builder: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A build that ends midway, or another build of the same model on the same GPU,
    # must never leave a cut-off cache: the new content arrives by one rename.
    cache = tmp_path / "timing.cache"
    cache.write_bytes(b"earlier")
    replaced: list[tuple[str, str]] = []
    replace = engine_builder.os.replace

    def recording_replace(source: str, target: str) -> None:
        assert Path(source).read_bytes() == b"earlier+timings"
        assert cache.read_bytes() == b"earlier"
        replaced.append((source, target))
        replace(source, target)

    monkeypatch.setattr(engine_builder.os, "replace", recording_replace)
    _build(engine_builder, cache)

    assert [target for _, target in replaced] == [str(cache)]
    assert cache.read_bytes() == b"earlier+timings"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["timing.cache"]


def test_a_failed_cache_write_keeps_the_old_cache(
    engine_builder: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache = tmp_path / "timing.cache"
    cache.write_bytes(b"earlier")

    def failing_replace(_source: str, _target: str) -> None:
        raise PermissionError("[WinError 5] Access is denied")

    monkeypatch.setattr(engine_builder.os, "replace", failing_replace)
    _build(engine_builder, cache)

    assert cache.read_bytes() == b"earlier"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["timing.cache"]


def test_an_unchanged_cache_is_not_written_again(
    engine_builder: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache = tmp_path / "timing.cache"
    write = engine_builder.write_timing_cache

    assert write(b"timings", str(cache)) is True
    assert cache.read_bytes() == b"timings"

    def unexpected_replace(_source: str, _target: str) -> None:
        raise AssertionError("an equal cache was written again")

    monkeypatch.setattr(engine_builder.os, "replace", unexpected_replace)
    assert write(b"timings", str(cache)) is False
    assert sorted(p.name for p in tmp_path.iterdir()) == ["timing.cache"]
