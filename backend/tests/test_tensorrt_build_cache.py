"""Build Engine's engine cache: a build of the same inputs runs once, kept in memory
within limits and, with the Engine Cache setting on, in its folder. The builds here
are stand-ins that count themselves (no GPU builds an engine)."""

from __future__ import annotations

import dataclasses
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import test_tensorrt_timing_cache

from nodes.impl.onnx.model import OnnxGeneric, OnnxInfo
from nodes.impl.tensorrt import build_cache
from nodes.impl.tensorrt.build_cache import SUFFIX, cached_engine, engine_key
from nodes.impl.tensorrt.model import TensorRTEngine, TensorRTEngineInfo

INFO = TensorRTEngineInfo(
    precision="fp16",
    input_channels=3,
    output_channels=3,
    scale=2,
    gpu_architecture="sm_120",
    tensorrt_version="11.4.0",
    has_dynamic_shapes=True,
    min_shape=(1, 3, 64, 64),
    opt_shape=(1, 3, 256, 256),
    max_shape=(1, 3, 1024, 1024),
)


@pytest.fixture(autouse=True)
def empty_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_cache, "_memory", OrderedDict())
    monkeypatch.setattr(build_cache, "_building", {})


class Builds:
    """A stand-in build that counts its runs; engine `i` is `size` bytes."""

    def __init__(self, size: int = 64) -> None:
        self.runs = 0
        self.size = size

    def __call__(self) -> TensorRTEngine:
        self.runs += 1
        return TensorRTEngine(bytes([self.runs % 256]) * self.size, INFO)


def model(data: bytes = b"onnx model") -> OnnxGeneric:
    return OnnxGeneric(data, OnnxInfo(17, "fp32"))


GPU = ("NVIDIA GeForce RTX 5090", (12, 0), "11.4.0", None)
# engine_builder.py imported against stand-ins for TensorRT and cuda-python
engine_builder = test_tensorrt_timing_cache.engine_builder


def key(name: str = "a", settings: tuple[object, ...] = ("onnx",)) -> str:
    return engine_key(model(name.encode()), settings, *GPU)


def test_a_build_of_the_same_inputs_runs_once():
    build = Builds()
    first = cached_engine(key(), build, None)
    assert cached_engine(key(), build, None) is first
    assert build.runs == 1


def full_key(
    data: bytes = b"onnx model",
    settings: tuple[object, ...] = ("onnx", "fp16"),
    gpu_name: str = GPU[0],
    compute_capability: tuple[int, int] = GPU[1],
    tensorrt_version: str = GPU[2],
    plugins: str | None = GPU[3],
) -> str:
    return engine_key(
        model(data), settings, gpu_name, compute_capability, tensorrt_version, plugins
    )


def test_each_key_input_changes_the_key():
    base = full_key()
    assert full_key() == base  # another model object with the same bytes
    changed = [
        full_key(data=b"other model"),
        full_key(settings=("onnx", "fp32")),
        full_key(gpu_name="NVIDIA GeForce RTX 4090"),
        full_key(compute_capability=(8, 9)),
        full_key(tensorrt_version="11.5.0"),
        full_key(plugins="dual plugins"),
    ]
    assert len({base, *changed}) == len(changed) + 1


def test_every_build_setting_is_in_the_key(engine_builder: ModuleType):
    config = engine_builder.BuildConfig(
        precision="fp16",
        workspace_size_gb=0,
        min_shape=(64, 64),
        opt_shape=(256, 256),
        max_shape=(1024, 1024),
        use_dynamic_shapes=True,
        allow_tf32=False,
    )
    changes = {
        "precision": "bf16",
        "workspace_size_gb": 8.0,
        "min_shape": (32, 32),
        "opt_shape": (512, 512),
        "max_shape": (2048, 2048),
        "use_dynamic_shapes": False,
        "allow_tf32": True,
        "optimization_level": 5,
        "sparse_weights": True,
        "tiling_optimization": "full",
    }
    assert changes.keys() == {f.name for f in dataclasses.fields(config)}

    def settings_key(built: Any) -> str:
        return engine_key(model(), ("onnx", *dataclasses.astuple(built)), *GPU)

    keys = {settings_key(config)}
    for name, value in changes.items():
        keys.add(settings_key(dataclasses.replace(config, **{name: value})))
    assert len(keys) == len(changes) + 1


def test_at_most_four_engines_are_kept_in_memory():
    build = Builds()
    for name in "abcde":
        cached_engine(key(name), build, None)
    cached_engine(key("e"), build, None)
    assert build.runs == 5
    cached_engine(key("a"), build, None)  # the least recently used, dropped
    assert build.runs == 6


def test_the_memory_budget_drops_engines(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(build_cache, "memory_budget", lambda: 100)
    build = Builds(size=60)
    cached_engine(key("a"), build, None)
    cached_engine(key("b"), build, None)  # 120 bytes: "a" goes
    cached_engine(key("b"), build, None)
    cached_engine(key("a"), build, None)
    assert build.runs == 3
    large = Builds(size=101)  # above the budget: not kept
    cached_engine(key("c"), large, None)
    cached_engine(key("c"), large, None)
    assert large.runs == 2


def test_the_budget_is_two_gib_or_a_tenth_of_the_ram():
    import psutil

    assert build_cache.memory_budget() == min(
        2 * 1024**3, psutil.virtual_memory().total // 10
    )


def test_concurrent_builds_of_one_key_build_once():
    started, release = threading.Event(), threading.Event()
    runs: list[int] = []

    def slow() -> TensorRTEngine:
        runs.append(1)
        started.set()
        assert release.wait(timeout=60)
        return TensorRTEngine(b"engine", INFO)

    results: list[TensorRTEngine] = []
    threads = [
        threading.Thread(
            target=lambda: results.append(cached_engine(key(), slow, None))
        )
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    assert started.wait(timeout=60)
    release.set()
    for thread in threads:
        thread.join(timeout=60)
    assert len(runs) == 1 and len(results) == 3
    assert all(result is results[0] for result in results)


def test_a_failed_build_is_not_kept():
    def failing() -> TensorRTEngine:
        raise RuntimeError("TensorRT could not build it")

    with pytest.raises(RuntimeError, match="could not build"):
        cached_engine(key(), failing, None)
    build = Builds()
    cached_engine(key(), build, None)
    assert build.runs == 1


def test_the_engine_cache_folder_keeps_engines_across_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    build = Builds()
    built = cached_engine(key(), build, str(tmp_path))
    assert [p.name for p in tmp_path.iterdir()] == [f"{key()}{SUFFIX}"]
    monkeypatch.setattr(build_cache, "_memory", OrderedDict())  # a restart
    read = cached_engine(key(), build, str(tmp_path))
    assert build.runs == 1
    assert read.bytes == built.bytes and read.info == built.info == INFO
    assert cached_engine(key(), build, str(tmp_path)) is read  # now in memory


@pytest.mark.parametrize("damage", ["truncated", "flipped", "header"])
def test_a_damaged_file_is_removed_and_built_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
):
    build = Builds()
    cached_engine(key(), build, str(tmp_path))
    path = tmp_path / f"{key()}{SUFFIX}"
    data = bytearray(path.read_bytes())
    if damage == "truncated":
        del data[-10:]
    elif damage == "flipped":
        data[-1] ^= 0xFF
    else:
        data[40] ^= 0xFF
    path.write_bytes(bytes(data))
    monkeypatch.setattr(build_cache, "_memory", OrderedDict())
    logged: list[str] = []
    monkeypatch.setattr(
        build_cache.logger,
        "warning",
        lambda message, *args, **kwargs: logged.append(message % args),
    )
    rebuilt = cached_engine(key(), build, str(tmp_path))
    assert build.runs == 2 and len(logged) == 1 and "damaged" in logged[0]
    # The file written again reads back.
    monkeypatch.setattr(build_cache, "_memory", OrderedDict())
    assert cached_engine(key(), build, str(tmp_path)).bytes == rebuilt.bytes
    assert build.runs == 2


def test_the_folder_keeps_the_most_recently_used_within_its_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    foreign = [tmp_path / "mine.engine", tmp_path / f"notes{SUFFIX}.txt"]
    for path in foreign:
        path.write_bytes(b"x" * 10000)
    build = Builds(size=300)
    now = time.time_ns()
    for age, name in zip((3, 2, 1), "abc", strict=True):
        cached_engine(key(name), build, str(tmp_path))
        # Seconds apart, in the past: the file system's time resolution may be coarse.
        os.utime(tmp_path / f"{key(name)}{SUFFIX}", ns=(now, now - age * 10**9))
    size = (tmp_path / f"{key('a')}{SUFFIX}").stat().st_size
    monkeypatch.setattr(build_cache, "DISK_BYTES", 3 * size + size // 2)
    # "a" is read again (a hit marks it used now), then "d" is written: "b" goes.
    monkeypatch.setattr(build_cache, "_memory", OrderedDict())
    cached_engine(key("a"), build, str(tmp_path))
    cached_engine(key("d"), build, str(tmp_path))
    kept = {p.name for p in tmp_path.glob(f"*{SUFFIX}")}
    assert kept == {f"{key(n)}{SUFFIX}" for n in "acd"}
    assert all(path.exists() for path in foreign)


def test_an_engine_above_the_limit_is_still_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(build_cache, "DISK_BYTES", 100)
    cached_engine(key(), Builds(size=300), str(tmp_path))
    assert (tmp_path / f"{key()}{SUFFIX}").exists()


def test_a_folder_that_cannot_be_written_only_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    logged: list[str] = []
    monkeypatch.setattr(
        build_cache.logger,
        "warning",
        lambda message, *args, **kwargs: logged.append(message % args),
    )
    missing = tmp_path / "missing"
    engine = cached_engine(key(), Builds(), str(missing))
    assert engine.bytes and len(logged) == 1 and "could not keep" in logged[0]


def test_the_dual_plugin_identity_follows_the_plugin_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "kernels.py"
    source.write_text("one", encoding="utf-8")
    monkeypatch.setattr(build_cache, "_DUAL_PLUGIN_SOURCES", (source,))
    build_cache.dual_plugin_identity.cache_clear()
    first = build_cache.dual_plugin_identity()
    build_cache.dual_plugin_identity.cache_clear()
    source.write_text("two", encoding="utf-8")
    assert build_cache.dual_plugin_identity() != first
    build_cache.dual_plugin_identity.cache_clear()


def test_the_shipped_dual_plugin_sources_are_read():
    build_cache.dual_plugin_identity.cache_clear()
    assert len(build_cache.dual_plugin_identity()) == 64
