"""Build Program's programs, kept so that a build of the same inputs is not run again (a
compile takes minutes): the MIGraphX counterpart of tensorrt/build_cache.py, with the same
rules.

A program is keyed by the SHA-256 of everything its build reads: the ONNX model's bytes,
every build setting, the GPU's name and gfx target, the MIGraphX runner (its version and
the SHA-256 of its module) and, for a DUAL or DRFT program, its route's build identity;
and by FORMAT_VERSION, the version of the record a file holds. Up to MEMORY_PROGRAMS
programs and `memory_budget()` bytes are kept in memory, the least recently used dropped
first; a program above the budget is not kept. With the "Program Cache" setting on, they
are also kept as files named `<key>.chainner-c.mxr` in its folder, written atomically, up
to DISK_BYTES in total, the least recently used removed first (only files of that name).
A file holds the info the build returned (JSON) and the SHA-256 of the program before the
program itself; a damaged file is removed and the program built again.

This module must not import MIGraphX (the tests have none).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from weakref import WeakKeyDictionary

from sanic.log import logger

from ..onnx.model import OnnxModel
from ..tensorrt.build_cache import memory_budget
from .model import MIGraphXProgram, MIGraphXProgramInfo

MEMORY_PROGRAMS = 4
DISK_BYTES = 16 * 1024**3
SUFFIX = ".chainner-c.mxr"
# The record a file holds (_encode): a new layout takes the next version.
FORMAT_VERSION = 1

_onnx_digests: WeakKeyDictionary[OnnxModel, bytes] = WeakKeyDictionary()


def _onnx_digest(model: OnnxModel) -> bytes:
    digest = _onnx_digests.get(model)
    if digest is None:
        digest = hashlib.sha256(model.bytes).digest()
        _onnx_digests[model] = digest
    return digest


def program_key(
    model: OnnxModel,
    settings: tuple[object, ...],
    gpu_name: str,
    gfx: str,
    runner: str,
    route: str | None,
) -> str:
    """The key of a program of `model` built with `settings` (every build input) on a GPU,
    by a MIGraphX runner (runner_identity), with a DUAL or DRFT route's build identity
    (None otherwise)."""
    digest = hashlib.sha256(_onnx_digest(model))
    digest.update(
        repr((FORMAT_VERSION, settings, gpu_name, gfx, runner, route)).encode()
    )
    return digest.hexdigest()


_lock = Lock()
_memory: OrderedDict[str, MIGraphXProgram] = OrderedDict()


class _Building:
    """A key's lock, while a call for the key runs or waits: concurrent builds of one key
    build it once."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.calls = 0


_building: dict[str, _Building] = {}


def _remember(key: str, program: MIGraphXProgram) -> None:
    """Keep `program` in memory, dropping the least recently used beyond the limits."""
    budget = memory_budget()
    with _lock:
        _memory.pop(key, None)
        if len(program.bytes) > budget:
            return
        _memory[key] = program
        used = sum(len(p.bytes) for p in _memory.values())
        while len(_memory) > MEMORY_PROGRAMS or used > budget:
            _, dropped = _memory.popitem(last=False)
            used -= len(dropped.bytes)


def _recalled(key: str) -> MIGraphXProgram | None:
    with _lock:
        program = _memory.get(key)
        if program is not None:
            _memory.move_to_end(key)
        return program


def cached_program(
    key: str, build: Callable[[], MIGraphXProgram], directory: str | None
) -> MIGraphXProgram:
    """The program of `key`: kept in memory, read from the Program Cache `directory`
    (None: off), or built (then kept in both)."""
    with _lock:
        building = _building.setdefault(key, _Building())
        building.calls += 1
    try:
        with building.lock:
            return _cached_program(key, build, directory)
    finally:
        with _lock:
            building.calls -= 1
            if building.calls == 0:
                del _building[key]


def _cached_program(
    key: str, build: Callable[[], MIGraphXProgram], directory: str | None
) -> MIGraphXProgram:
    program = _recalled(key)
    if program is not None:
        logger.info("Build Program: the same build ran before; reusing its program.")
        return program
    if directory is not None:
        program = _read(Path(directory), key)
        if program is not None:
            logger.info("Build Program: reusing the program from the Program Cache.")
            _remember(key, program)
            return program
    program = build()
    _remember(key, program)
    if directory is not None:
        _store(Path(directory), key, program)
    return program


def _magic() -> bytes:
    return f"chaiNNer-C MIGraphX program cache {FORMAT_VERSION}\n".encode()


def _encode(program: MIGraphXProgram) -> bytes:
    header = json.dumps(
        {
            "info": dataclasses.asdict(program.info),
            "sha256": hashlib.sha256(program.bytes).hexdigest(),
        }
    ).encode()
    return _magic() + len(header).to_bytes(8, "little") + header + program.bytes


def _decode(data: bytes) -> MIGraphXProgram | None:
    """The program `_encode` wrote, or None for a damaged file."""
    magic = _magic()
    if not data.startswith(magic):
        return None
    start = len(magic) + 8
    end = start + int.from_bytes(data[len(magic) : start], "little")
    try:
        header = json.loads(data[start:end])
        program = MIGraphXProgram(
            data[end:], MIGraphXProgramInfo(**dict(header["info"]))
        )
        digest = header["sha256"]
    except (ValueError, KeyError, TypeError):
        return None
    if hashlib.sha256(program.bytes).hexdigest() != digest:
        return None
    return program


def _read(directory: Path, key: str) -> MIGraphXProgram | None:
    path = directory / f"{key}{SUFFIX}"
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        # Another program may hold it; the program written after the build replaces it.
        logger.warning(
            "Program Cache: could not read %s; building it again.", path, exc_info=True
        )
        return None
    program = _decode(data)
    if program is None:
        logger.warning("Program Cache: %s is damaged; building it again.", path)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Program Cache: could not remove %s.", path, exc_info=True)
        return None
    try:
        os.utime(path)  # its last use, for the eviction
    except OSError:
        logger.warning(
            "Program Cache: could not mark %s used; it may be removed first.",
            path,
            exc_info=True,
        )
    return program


def _store(directory: Path, key: str, program: MIGraphXProgram) -> None:
    """Write `program` atomically, then remove the least recently used files beyond
    DISK_BYTES. A failure is logged: the build itself succeeded."""
    path = directory / f"{key}{SUFFIX}"
    # A process that crashed while writing this key left its temporary file.
    for stray in directory.glob(f"{key}.*.tmp"):
        try:
            stray.unlink()
        except OSError:
            logger.warning("Program Cache: could not remove %s.", stray, exc_info=True)
    try:
        output = tempfile.NamedTemporaryFile(
            dir=directory, prefix=f"{key}.", suffix=".tmp", delete=False
        )
        temporary = Path(output.name)
        try:
            with output:
                output.write(_encode(program))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    except OSError:
        logger.warning("Program Cache: could not keep %s.", path, exc_info=True)
        return
    try:
        _evict(directory, keep=path)
    except OSError:
        logger.warning("Program Cache: could not trim %s.", directory, exc_info=True)


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
