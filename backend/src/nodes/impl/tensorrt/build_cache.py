"""Build Engine's engines, kept so that a build of the same inputs is not run again
(a build takes minutes).

An engine is keyed by the SHA-256 of everything its build reads: the ONNX model's
bytes, every build setting, the GPU's name and compute capability, the TensorRT
version and, for a DUAL engine, its plugins' code; and by FORMAT_VERSION, the version
of the record a file holds (a record of another version is never read). Up to
MEMORY_ENGINES engines and `memory_budget()` bytes are kept in memory, the least
recently used dropped first; an engine above the budget is not kept. With the TensorRT "Engine Cache" setting on, they
are also kept as files named `<key>.chainner-c.engine` in its folder, written
atomically, up to DISK_BYTES in total, the least recently used removed first (only
files of that name are removed). A file holds the info the build returned (JSON) and
the SHA-256 of the engine before the engine itself, so a hit returns what the build
did; a damaged file is removed and the engine built again.

This module must not import TensorRT (the tests have none).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import tempfile
from collections import OrderedDict
from collections.abc import Callable
from functools import cache
from pathlib import Path
from threading import Lock
from weakref import WeakKeyDictionary

import psutil
from sanic.log import logger

from ..onnx.model import OnnxModel
from . import dual
from .model import TensorRTEngine, TensorRTEngineInfo

MEMORY_ENGINES = 4
DISK_BYTES = 16 * 1024**3
SUFFIX = ".chainner-c.engine"
# The record a file holds (_encode): a new layout takes the next version.
FORMAT_VERSION = 1
# The code a DUAL engine's plugins are built from (dual.build_engine's worker)
_DUAL_PLUGIN_SOURCES = (
    Path(__file__).with_name("dual_aot.py"),
    Path(__file__).with_name("dual_dynamic_kernels.py"),
    Path(__file__).with_name("dual_engine_worker.py"),
    dual.VENDOR / "scripts" / "dual_tensorrt" / "kernels.py",
)


def memory_budget() -> int:
    """The bytes of engines kept in memory: 2 GiB, or a tenth of the RAM if less."""
    return min(2 * 1024**3, psutil.virtual_memory().total // 10)


_onnx_digests: WeakKeyDictionary[OnnxModel, bytes] = WeakKeyDictionary()


def _onnx_digest(model: OnnxModel) -> bytes:
    digest = _onnx_digests.get(model)
    if digest is None:
        digest = hashlib.sha256(model.bytes).digest()
        _onnx_digests[model] = digest
    return digest


@cache
def dual_plugin_identity() -> str:
    """The SHA-256 of DUAL's plugin code and the Triton that compiles its kernels."""
    digest = hashlib.sha256()
    for source in _DUAL_PLUGIN_SOURCES:
        digest.update(source.read_bytes())
    try:
        digest.update(importlib.metadata.version("triton").encode())
    except importlib.metadata.PackageNotFoundError:
        # The DUAL build reports the missing Triton itself.
        pass
    return digest.hexdigest()


def engine_key(
    model: OnnxModel,
    settings: tuple[object, ...],
    gpu_name: str,
    compute_capability: tuple[int, int],
    tensorrt_version: str,
    plugins: str | None,
) -> str:
    """The key of an engine of `model` built with `settings` (every build input) on a
    GPU, by a TensorRT version, with DUAL's `plugins` identity (None otherwise)."""
    digest = hashlib.sha256(_onnx_digest(model))
    digest.update(
        repr(
            (
                FORMAT_VERSION,
                settings,
                gpu_name,
                tuple(compute_capability),
                tensorrt_version,
                plugins,
            )
        ).encode()
    )
    return digest.hexdigest()


_lock = Lock()
_memory: OrderedDict[str, TensorRTEngine] = OrderedDict()
# One lock per key: concurrent builds of one key build it once.
_building: dict[str, Lock] = {}


def _remember(key: str, engine: TensorRTEngine) -> None:
    """Keep `engine` in memory, dropping the least recently used beyond the limits."""
    budget = memory_budget()
    with _lock:
        _memory.pop(key, None)
        if len(engine.bytes) > budget:
            return
        _memory[key] = engine
        used = sum(len(e.bytes) for e in _memory.values())
        while len(_memory) > MEMORY_ENGINES or used > budget:
            _, dropped = _memory.popitem(last=False)
            used -= len(dropped.bytes)


def _recalled(key: str) -> TensorRTEngine | None:
    with _lock:
        engine = _memory.get(key)
        if engine is not None:
            _memory.move_to_end(key)
        return engine


def cached_engine(
    key: str, build: Callable[[], TensorRTEngine], directory: str | None
) -> TensorRTEngine:
    """The engine of `key`: kept in memory, read from the Engine Cache `directory`
    (None: off), or built (then kept in both)."""
    with _lock:
        building = _building.setdefault(key, Lock())
    with building:
        engine = _recalled(key)
        if engine is not None:
            logger.info("Build Engine: the same build ran before; reusing its engine.")
            return engine
        if directory is not None:
            engine = _read(Path(directory), key)
            if engine is not None:
                logger.info("Build Engine: reusing the engine from the Engine Cache.")
                _remember(key, engine)
                return engine
        engine = build()
        _remember(key, engine)
        if directory is not None:
            _store(Path(directory), key, engine)
        return engine


def _magic() -> bytes:
    return f"chaiNNer-C engine cache {FORMAT_VERSION}\n".encode()


def _encode(engine: TensorRTEngine) -> bytes:
    header = json.dumps(
        {
            "info": dataclasses.asdict(engine.info),
            "sha256": hashlib.sha256(engine.bytes).hexdigest(),
        }
    ).encode()
    return _magic() + len(header).to_bytes(8, "little") + header + engine.bytes


def _decode(data: bytes) -> TensorRTEngine | None:
    """The engine `_encode` wrote, or None for a damaged file."""
    magic = _magic()
    if not data.startswith(magic):
        return None
    start = len(magic) + 8
    end = start + int.from_bytes(data[len(magic) : start], "little")
    try:
        header = json.loads(data[start:end])
        info = dict(header["info"])
        for name in ("min_shape", "opt_shape", "max_shape"):
            if info[name] is not None:
                info[name] = tuple(info[name])
        engine = TensorRTEngine(data[end:], TensorRTEngineInfo(**info))
        digest = header["sha256"]
    except (ValueError, KeyError, TypeError):
        return None
    if hashlib.sha256(engine.bytes).hexdigest() != digest:
        return None
    return engine


def _read(directory: Path, key: str) -> TensorRTEngine | None:
    path = directory / f"{key}{SUFFIX}"
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    engine = _decode(data)
    if engine is None:
        logger.warning("Engine Cache: %s is damaged; building it again.", path)
        path.unlink(missing_ok=True)
        return None
    os.utime(path)  # its last use, for the eviction
    return engine


def _store(directory: Path, key: str, engine: TensorRTEngine) -> None:
    """Write `engine` atomically, then remove the least recently used files beyond
    DISK_BYTES. A failure is logged: the build itself succeeded."""
    path = directory / f"{key}{SUFFIX}"
    try:
        with tempfile.NamedTemporaryFile(
            dir=directory, prefix=f"{key}.", suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(_encode(engine))
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    except OSError:
        logger.warning("Engine Cache: could not keep %s.", path, exc_info=True)
        return
    try:
        _evict(directory, keep=path)
    except OSError:
        logger.warning("Engine Cache: could not trim %s.", directory, exc_info=True)


def _evict(directory: Path, keep: Path) -> None:
    files = sorted(
        ((p.stat(), p) for p in directory.glob(f"*{SUFFIX}") if p.is_file()),
        key=lambda item: item[0].st_mtime_ns,
    )
    used = sum(stat.st_size for stat, _ in files)
    for stat, path in files:
        if used <= DISK_BYTES:
            break
        if path != keep:
            path.unlink(missing_ok=True)
            used -= stat.st_size
