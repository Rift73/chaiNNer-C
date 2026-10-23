"""Build Engine's legacy DRFT route: an ONNX with the DRFT exporter's manifest builds in
a child process (drft_engine_worker.py) that loads chaiNNer-C's DRFT plugin library (the
engine does not carry it; loading processes register it once). Nothing here needs a GPU: the child, the node's GPU
queries and TensorRT itself (for the worker's settings) are stand-ins."""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import cuda.bindings
import pytest
from onnx import TensorProto, helper

from api import NodeContext
from nodes.impl.onnx.model import OnnxGeneric, OnnxInfo
from nodes.impl.tensorrt import build_cache, drft, drft_engine_worker, dual
from packages.chaiNNer_tensorrt.settings import TensorRTSettings
from packages.chaiNNer_tensorrt.tensorrt.utility import build_engine

# Build Engine reads no member of its context once its settings are stubbed.
CONTEXT = cast(NodeContext, SimpleNamespace())
SETTINGS = TensorRTSettings(gpu_index=0, timing_cache_path=None, engine_cache_path=None)
MANIFEST = {
    "format": 1,
    "plugins": "legacy-v1",
    "scale": 4,
    "window_size": 16,
    "min_size": 64,
    "alignment": 16,
}


def _onnx(metadata: dict[str, dict[str, Any]] | None = None) -> bytes:
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph(
        [node],
        "plain",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])],
    )
    model = helper.make_model(graph)
    for key, value in (metadata or {}).items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, json.dumps(value)
    return model.SerializeToString()


def test_only_a_drft_export_has_a_drft_manifest():
    assert drft.export_manifest(_onnx()) is None
    assert drft.export_manifest(_onnx({drft.METADATA_KEY: MANIFEST})) == MANIFEST


@pytest.mark.parametrize(
    ("low", "opt", "high", "message"),
    [
        ((64, 64), (512, 512), (1152, 1152), None),
        ((64, 96), (512, 512), (1088, 1920), None),
        ((256, 256), (256, 256), (256, 256), None),
        ((64, 64), (512, 500), (1152, 1152), "dynamic DRFT engine's optimal size must"),
        ((48, 64), (512, 512), (1152, 1152), r"minimum size must be at least 64 px"),
        ((64, 64), (2048, 512), (1152, 1152), "minimum <= optimal <= maximum"),
        # Shape Mode Fixed: one size, a fixed engine
        ((256, 250), (256, 250), (256, 250), r"^A fixed DRFT engine's size must be"),
        ((32, 32), (32, 32), (32, 32), r"^A fixed DRFT engine's size must be at least"),
    ],
)
def test_a_profile_keeps_to_the_onnxs_alignment_and_minimum(
    low: tuple[int, int],
    opt: tuple[int, int],
    high: tuple[int, int],
    message: str | None,
):
    if message is None:
        drft.check_profile(MANIFEST, low, opt, high)
    else:
        with pytest.raises(ValueError, match=message):
            drft.check_profile(MANIFEST, low, opt, high)


def _library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    library = tmp_path / "drft_trt_plugins.dll"
    library.write_bytes(b"dll")
    monkeypatch.setattr(drft, "PLUGIN_LIBRARY", library)
    return library


def test_the_build_settings_reach_the_engine_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    library = _library(tmp_path, monkeypatch)
    calls: list[tuple[list[str], str]] = []

    def child(args: list[str], label: str, *_: object) -> str:
        calls.append((args, label))
        Path(args[3]).write_bytes(b"engine")
        return ""

    monkeypatch.setattr(dual, "run_child", child)
    engine = drft.build_engine(
        b"onnx",
        MANIFEST,
        1,
        ((64, 64), (512, 512), (1152, 1152)),
        workspace_gib=0,
        optimization_level=5,
        tf32=True,
        sparse_weights=True,
        tiling_optimization="moderate",
    )
    assert engine == b"engine"
    ((args, label),) = calls
    assert label == "DRFT engine build"
    assert args[:2] == ["-m", "nodes.impl.tensorrt.drft_engine_worker"]
    assert args[4:] == [
        "1",
        str(library),
        "0",
        "5",
        "1",
        "1",
        "moderate",
        "64x64",
        "512x512",
        "1152x1152",
        "16",
    ]


def test_a_build_without_the_plugin_library_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(drft, "PLUGIN_LIBRARY", tmp_path / "drft_trt_plugins.dll")
    monkeypatch.setattr(dual, "run_child", pytest.fail)
    with pytest.raises(RuntimeError, match="this build was made without it"):
        drft.build_engine(b"onnx", MANIFEST, 0, ((64, 64), (64, 64), (64, 64)))


class _Config:
    """A builder config that records what the worker sets on it."""

    def __init__(self):
        self.flags: list[str] = []
        self.cleared: list[str] = []
        self.pools: dict[str, int] = {}
        self.profiles: list[_Profile] = []
        self.timing_cache: object = None

    def set_flag(self, flag: str):
        self.flags.append(flag)

    def clear_flag(self, flag: str):
        self.cleared.append(flag)

    def set_memory_pool_limit(self, pool: str, size: int):
        self.pools[pool] = size

    def create_timing_cache(self, data: bytes):
        return ("cache", data)

    def set_timing_cache(self, cache: object, ignore_mismatch: bool):
        self.timing_cache = (cache, ignore_mismatch)

    def add_optimization_profile(self, profile: _Profile):
        self.profiles.append(profile)


class _Profile:
    def __init__(self):
        self.shapes: dict[str, tuple[tuple[int, ...], ...]] = {}

    def set_shape(self, name: str, *shapes: tuple[int, ...]):
        self.shapes[name] = shapes


def _fake_tensorrt(loaded: object) -> tuple[SimpleNamespace, SimpleNamespace]:
    """A TensorRT module whose builder records the worker's settings, and that record;
    its plugin registry's load_library returns `loaded`."""
    record = SimpleNamespace(
        libraries=[],
        config=_Config(),
        image=SimpleNamespace(name="input"),
        parsed=[],
        engines=0,
    )

    def load_library(path: str):
        record.libraries.append(path)
        return loaded

    class Logger:
        WARNING = "warning"

        def __init__(self, _severity: str):
            pass

    class Builder:
        def __init__(self, _logger: Logger):
            pass

        def create_network(self, flags: int):
            record.network_flags = flags
            return SimpleNamespace(get_input=lambda _: record.image)

        def create_builder_config(self):
            return record.config

        def create_optimization_profile(self):
            return _Profile()

        def build_serialized_network(self, _network: object, _config: _Config):
            record.engines += 1
            return b"engine"

    class OnnxParser:
        num_errors = 0

        def __init__(self, _network: object, _logger: Logger):
            pass

        def parse_from_file(self, path: str) -> bool:
            record.parsed.append(path)
            return True

    trt = SimpleNamespace(
        get_plugin_registry=lambda: SimpleNamespace(load_library=load_library),
        Logger=Logger,
        Builder=Builder,
        OnnxParser=OnnxParser,
        NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=3),
        BuilderFlag=SimpleNamespace(TF32="tf32", SPARSE_WEIGHTS="sparse"),
        TilingOptimizationLevel=SimpleNamespace(NONE="tiling-none", FULL="tiling-full"),
        MemoryPoolType=SimpleNamespace(WORKSPACE="workspace"),
        ProfilingVerbosity=SimpleNamespace(DETAILED="detailed"),
    )
    return trt, record


def _fake_cuda(monkeypatch: pytest.MonkeyPatch):
    success = object()
    runtime = SimpleNamespace(
        cudaSetDevice=lambda _: (success,),
        cudaError_t=SimpleNamespace(cudaSuccess=success),
    )
    monkeypatch.setattr(cuda.bindings, "runtime", runtime, raising=False)


def _worker_arguments(
    tmp_path: Path, workspace: str, tf32: str, sparse: str, tiling: str = "none"
):
    return [
        str(tmp_path / "model.onnx"),
        str(tmp_path / "model.engine"),
        "0",
        "C:/chaiNNer/drft_trt_plugins.dll",
        workspace,
        "3",
        tf32,
        sparse,
        tiling,
        "64x64",
        "512x512",
        "1152x1152",
        "16",
    ]


def test_the_worker_loads_the_plugin_library_with_reference_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    trt, record = _fake_tensorrt(loaded=object())
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    _fake_cuda(monkeypatch)
    drft_engine_worker.main(*_worker_arguments(tmp_path, "0", "0", "1"))

    library = "C:/chaiNNer/drft_trt_plugins.dll"
    config = record.config
    assert record.libraries == [library]
    # Not serialized: a second load of such an engine in a process would fail
    assert not hasattr(config, "plugins_to_serialize")
    assert record.network_flags == 1 << 3  # strongly typed
    assert record.parsed == [str(tmp_path / "model.onnx")]
    assert (config.cleared, config.flags) == (["tf32"], ["sparse"])
    assert config.builder_optimization_level == 3
    assert config.tiling_optimization_level == "tiling-none"
    assert config.max_aux_streams == 0
    assert config.pools == {}  # workspace 0: TensorRT's default
    assert config.profiling_verbosity == "detailed"
    assert config.timing_cache == (("cache", b""), False)
    # The input's name marks the engine as mixed precision and carries the alignment
    assert record.image.name == "input_aligned_16"
    assert dual.input_alignment(record.image.name) == 16
    ((profile),) = config.profiles
    assert profile.shapes == {
        "input_aligned_16": ((1, 3, 64, 64), (1, 3, 512, 512), (1, 3, 1152, 1152))
    }
    assert (tmp_path / "model.engine").read_bytes() == b"engine"


def test_the_workers_workspace_tf32_density_and_tiling_follow_the_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    trt, record = _fake_tensorrt(loaded=object())
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    _fake_cuda(monkeypatch)
    drft_engine_worker.main(*_worker_arguments(tmp_path, "2.5", "1", "0", "full"))
    assert (record.config.cleared, record.config.flags) == ([], [])
    assert record.config.pools == {"workspace": int(2.5 * (1 << 30))}
    assert record.config.tiling_optimization_level == "tiling-full"


def test_the_worker_stops_when_tensorrt_cannot_load_the_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    trt, record = _fake_tensorrt(loaded=None)
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    _fake_cuda(monkeypatch)
    with pytest.raises(RuntimeError, match="could not load the DRFT plugin library"):
        drft_engine_worker.main(*_worker_arguments(tmp_path, "0", "0", "1"))
    assert (record.parsed, record.engines) == ([], 0)


@pytest.mark.parametrize("present", [True, False])
def test_processes_register_the_plugin_library_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, present: bool
):
    """Engines do not carry the library: a loading process registers it once (a
    build without it registers nothing, and only DRFT engines then fail)."""
    library = tmp_path / "drft_trt_plugins.dll"
    if present:
        library.write_bytes(b"dll")
    monkeypatch.setattr(drft, "PLUGIN_LIBRARY", library)
    trt, record = _fake_tensorrt(loaded=object())
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    drft.load_plugins.cache_clear()
    try:
        drft.load_plugins()
        drft.load_plugins()
    finally:
        drft.load_plugins.cache_clear()
    assert record.libraries == ([str(library)] if present else [])


def _run_node(monkeypatch: pytest.MonkeyPatch, onnx_bytes: bytes, shape_mode: Any):
    """Build Engine on an ONNX with every builder stubbed; the route it took."""
    routes: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def route(name: str):
        def stub(*args: object, **kwargs: object) -> str:
            routes.append((name, args, kwargs))
            return name

        return stub

    monkeypatch.setattr(
        build_engine,
        "get_settings",
        lambda _: SETTINGS,
    )
    _stub_gpu(monkeypatch)
    # The generic route's build runs at once (its engine is the route's name)
    monkeypatch.setattr(
        build_engine, "cached_engine", lambda _key, build, _path: build()
    )
    monkeypatch.setattr(build_engine, "build_drft_engine", route("drft"))
    monkeypatch.setattr(build_engine, "build_dual_engine", route("dual"))
    monkeypatch.setattr(build_engine, "build_engine_from_onnx", route("generic"))
    model = OnnxGeneric(onnx_bytes, OnnxInfo(opset=20, dtype="fp32"))
    build_engine.build_engine_node(
        CONTEXT,
        model,
        build_engine.Precision.FP16,
        shape_mode,
        64,
        64,
        512,
        512,
        1152,
        1152,
        256,
        256,
        0.0,
        False,
        True,
        3,
        build_engine.TilingOptimization.NONE,
    )
    return routes


def test_build_engine_routes_a_drft_onnx_to_the_drft_route(
    monkeypatch: pytest.MonkeyPatch,
):
    routes = _run_node(
        monkeypatch,
        _onnx({drft.METADATA_KEY: MANIFEST}),
        build_engine.ShapeMode.DYNAMIC,
    )
    ((name, args, kwargs),) = routes
    assert name == "drft"
    assert args[2] == MANIFEST
    assert args[4] == ((64, 64), (512, 512), (1152, 1152))
    assert kwargs == {
        "workspace_gib": 0.0,
        "optimization_level": 3,
        "tf32": False,
        "sparse_weights": True,
        "tiling_optimization": "none",
    }


def test_build_engine_gives_a_fixed_drft_engine_one_size(
    monkeypatch: pytest.MonkeyPatch,
):
    routes = _run_node(
        monkeypatch, _onnx({drft.METADATA_KEY: MANIFEST}), build_engine.ShapeMode.FIXED
    )
    ((name, args, _),) = routes
    assert (name, args[4]) == ("drft", ((256, 256), (256, 256), (256, 256)))


def test_build_engine_keeps_other_onnxs_on_their_routes(
    monkeypatch: pytest.MonkeyPatch,
):
    plain = _run_node(monkeypatch, _onnx(), build_engine.ShapeMode.DYNAMIC)
    assert [name for name, _, _ in plain] == ["generic"]
    dual_onnx = _onnx({dual.METADATA_KEY: {"specialization": {"dynamic": True}}})
    routed = _run_node(monkeypatch, dual_onnx, build_engine.ShapeMode.DYNAMIC)
    assert [name for name, _, _ in routed] == ["dual"]


def _stub_gpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(build_engine, "get_cuda_compute_capability", lambda _: (12, 0))
    monkeypatch.setattr(build_engine, "get_cuda_device_name", lambda _: "RTX 5090")


def _build_drft(profile: Any, tiling: str = "none"):
    return build_engine.build_drft_engine(
        CONTEXT,
        OnnxGeneric(b"onnx", OnnxInfo(opset=20, dtype="mixed")),
        MANIFEST,
        SETTINGS,
        profile,
        workspace_gib=0.0,
        optimization_level=3,
        tf32=False,
        sparse_weights=True,
        tiling_optimization=tiling,
    )


Calls = list[tuple[Any, dict[str, object]]]


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> Calls:
    """drft.build_engine's calls, behind an empty engine cache (in memory only)."""
    monkeypatch.setattr(build_cache, "_memory", OrderedDict())
    monkeypatch.setattr(build_cache, "_building", {})
    _stub_gpu(monkeypatch)
    calls: Calls = []

    def build(*args: object, **kwargs: object) -> bytes:
        calls.append((args, kwargs))
        return b"engine"

    monkeypatch.setattr(drft, "build_engine", build)
    return calls


def test_a_drft_engine_is_mixed_precision_at_the_manifests_scale(builds: Calls):
    profile = ((64, 64), (512, 512), (1152, 1152))
    engine = _build_drft(profile)
    ((args, kwargs),) = builds
    assert args == (b"onnx", MANIFEST, 0, profile)
    assert kwargs["sparse_weights"] is True
    assert kwargs["tiling_optimization"] == "none"
    assert engine.bytes == b"engine"
    info = engine.info
    assert (info.precision, info.scale, info.gpu_architecture) == ("mixed", 4, "sm_120")
    assert (info.min_shape, info.max_shape) == ((1, 3, 64, 64), (1, 3, 1152, 1152))
    assert info.has_dynamic_shapes


def test_a_drft_build_of_the_same_inputs_runs_once(builds: Calls):
    profile = ((64, 64), (512, 512), (1152, 1152))
    first = _build_drft(profile)
    assert _build_drft(profile) is first
    _build_drft(profile, tiling="full")
    _build_drft(((256, 256),) * 3)
    assert [kwargs["tiling_optimization"] for _, kwargs in builds] == [
        "none",
        "full",
        "none",
    ]


def test_build_engine_refuses_a_drft_profile_before_building(
    monkeypatch: pytest.MonkeyPatch,
):
    _stub_gpu(monkeypatch)
    monkeypatch.setattr(drft, "build_engine", pytest.fail)
    with pytest.raises(ValueError, match="multiple of 16 px"):
        _build_drft(((64, 64), (500, 500), (1152, 1152)))
