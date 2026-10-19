"""Atomic file writes (native/include/atomic_file.hpp): a temporary file in the target
folder renamed onto the final name, an equal file left untouched, only this process's own
temporary files ever deleted, and errors naming the final path as Python's own open()
and os.replace() name it."""

from __future__ import annotations

import ctypes
import gc
import os
import re
import stat
from pathlib import Path

import pytest

from nodes.impl.native_graph import graph

OLD = 1_000_000_000  # An mtime (2001) no write of this test can produce.
TEMP = re.compile(r"\.chainner-c-(\d+)-(\d+)\.tmp$")


def aged(path: Path, data: bytes) -> Path:
    """path holding data, with an old modification time."""
    path.write_bytes(data)
    os.utime(path, (OLD, OLD))
    return path


def foreign(folder: Path, name: str) -> dict[Path, bytes]:
    """Files a writer of `name` must never touch: another process's temporary file of
    it, this process's name with a sequence number far ahead, and an unrelated .tmp."""
    files = {
        folder / f"{name}.chainner-c-1-0.tmp": b"other process",
        folder / f"{name}.chainner-c-{os.getpid()}-999999999.tmp": b"leftover",
        folder / "unrelated.tmp": b"unrelated",
    }
    for path, data in files.items():
        path.write_bytes(data)
    return files


def contents(folder: Path) -> dict[Path, bytes | None]:
    """Each file's bytes; None for a folder."""
    return {p: None if p.is_dir() else p.read_bytes() for p in folder.iterdir()}


def test_write_creates_replaces_and_leaves_an_equal_file(tmp_path: Path):
    final = tmp_path / "a.bin"
    assert graph().atomic_write(final, b"first") is True
    assert final.read_bytes() == b"first"
    aged(final, b"same")
    assert graph().atomic_write(final, b"same") is False
    assert final.stat().st_mtime == OLD  # untouched
    assert graph().atomic_write(final, b"other") is True
    assert final.read_bytes() == b"other" and final.stat().st_mtime != OLD
    # Equal size, different bytes (in the second 1 MiB block), and an empty file.
    big = bytes(3 << 20)
    aged(final, big)
    changed = bytearray(big)
    changed[(1 << 20) + 5] = 1
    assert graph().atomic_write(final, changed) is True
    assert final.read_bytes() == changed
    aged(final, b"")
    assert graph().atomic_write(final, b"") is False
    assert final.stat().st_mtime == OLD
    assert graph().atomic_write(str(final), memoryview(b"str path")) is True
    assert sorted(tmp_path.iterdir()) == [final]


def test_foreign_temporary_files_are_never_touched(tmp_path: Path):
    others = foreign(tmp_path, "a.bin")
    final = tmp_path / "a.bin"
    graph().atomic_write(final, b"x")
    graph().atomic_write(final, b"x")
    with graph().AtomicTarget(final) as target:
        assert Path(target.temp) not in others
        Path(target.temp).write_bytes(b"y")
    target = graph().AtomicTarget(final)
    Path(target.temp).write_bytes(b"z")
    target.abandon()
    graph().AtomicTarget(final).abandon()  # nothing written: nothing deleted
    with pytest.raises(FileNotFoundError):
        graph().AtomicTarget(final).commit()
    assert contents(tmp_path) == {**others, final: b"y"}


def test_temporary_name_is_in_the_folder_with_pid_and_sequence(tmp_path: Path):
    final = tmp_path / "v.mkv"
    first, second = graph().AtomicTarget(final), graph().AtomicTarget(final)
    for target in (first, second):
        assert Path(target.temp).parent == tmp_path
        assert target.temp.startswith(str(final) + ".chainner-c-")
        match = TEMP.search(target.temp)
        assert match and int(match.group(1)) == os.getpid()
    assert first.temp != second.temp
    assert not list(tmp_path.iterdir())  # a target creates nothing itself


def test_errors_name_the_final_path_as_open_does(tmp_path: Path):
    missing = tmp_path / "missing" / "a.bin"
    with pytest.raises(OSError) as before:
        open(missing, "wb")
    with pytest.raises(OSError) as after:
        graph().atomic_write(missing, b"x")
    assert type(after.value) is type(before.value) is FileNotFoundError
    assert str(after.value) == str(before.value)
    assert after.value.filename == str(missing) and after.value.winerror is None


KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
KERNEL32.CreateFileW.restype = ctypes.c_void_p
KERNEL32.CreateFileW.argtypes = [
    ctypes.c_wchar_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_void_p,
]
KERNEL32.CloseHandle.argtypes = [ctypes.c_void_p]


class Exclusive:
    """A handle on a file that shares nothing, as another program can hold one."""

    def __init__(self, path: Path):
        generic_read, open_existing = 0x80000000, 3
        self.handle = KERNEL32.CreateFileW(
            str(path), generic_read, 0, None, open_existing, 0, None
        )
        assert self.handle not in (None, ctypes.c_void_p(-1).value)

    def close(self):
        KERNEL32.CloseHandle(self.handle)


def test_failures_on_the_final_path_raise_what_open_raises(tmp_path: Path):
    # A folder, a read-only file and a file another program holds: open(final, "wb")
    # raises PermissionError [Errno 13]; the write raises the same, whether the CRT
    # refuses its temporary file or the rename, and leaves no temporary file.
    read_only = aged(tmp_path / "ro.bin", b"keep")
    os.chmod(read_only, stat.S_IREAD)
    held = aged(tmp_path / "held.bin", b"held")
    folder = tmp_path / "folder"
    folder.mkdir()
    holder = Exclusive(held)
    try:
        for final in (folder, read_only, held):
            with pytest.raises(OSError) as before:
                open(final, "wb")
            for write in (graph().atomic_write, _commit):
                with pytest.raises(OSError) as after:
                    write(final, b"new")
                assert type(after.value) is type(before.value) is PermissionError
                assert str(after.value) == str(before.value)
                assert after.value.filename == str(final)
                assert after.value.winerror is None
    finally:
        holder.close()
        os.chmod(read_only, stat.S_IWRITE)
    assert contents(tmp_path) == {folder: None, read_only: b"keep", held: b"held"}
    assert read_only.stat().st_mtime == held.stat().st_mtime == OLD


def _commit(final: Path, data: bytes) -> bool:
    target = graph().AtomicTarget(final)
    Path(target.temp).write_bytes(data)
    return target.commit()


def test_target_commit_and_equal_bytes(tmp_path: Path):
    final = tmp_path / "m.pth"
    assert _commit(final, b"model") is True
    aged(final, b"model")
    assert _commit(final, b"model") is False  # the temporary file is deleted
    assert final.stat().st_mtime == OLD
    assert _commit(final, b"model2") is True
    assert final.read_bytes() == b"model2"
    assert sorted(tmp_path.iterdir()) == [final]


def test_failed_library_write_leaves_no_partial_file(tmp_path: Path):
    # A library writing the temporary file fails half way: the context manager deletes
    # it, and the file at the path is the one before.
    final = aged(tmp_path / "engine.engine", b"previous")
    with pytest.raises(RuntimeError, match="half way"):
        with graph().AtomicTarget(final) as target:
            with open(target.temp, "wb") as file:
                file.write(b"partial")
            raise RuntimeError("half way")
    assert contents(tmp_path) == {final: b"previous"}
    assert final.stat().st_mtime == OLD


def test_abandon_repeats_commit_once_and_the_destructor_abandons(tmp_path: Path):
    final = tmp_path / "s.vpy"
    target = graph().AtomicTarget(final)
    held = open(target.temp, "wb")  # another holder: the first abandon cannot delete
    try:
        target.abandon()
        assert Path(target.temp).exists()
    finally:
        held.close()
    target.abandon()
    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="already committed or abandoned"):
        target.commit()
    committed = graph().AtomicTarget(final)
    Path(committed.temp).write_bytes(b"a")
    committed.commit()
    committed.abandon()  # a committed target keeps its file
    with pytest.raises(RuntimeError, match="already committed or abandoned"):
        committed.commit()
    dropped = graph().AtomicTarget(final)
    Path(dropped.temp).write_bytes(b"b")
    temp = Path(dropped.temp)
    del dropped
    gc.collect()
    assert not temp.exists()
    assert contents(tmp_path) == {final: b"a"}


def test_move_from_another_folder(tmp_path: Path):
    source_folder, target_folder = tmp_path / "chaiNNer-x", tmp_path / "out"
    source_folder.mkdir()
    target_folder.mkdir()
    others = foreign(target_folder, "t.dds")
    source, final = source_folder / "t.dds", target_folder / "t.dds"
    source.write_bytes(b"texture")
    assert graph().atomic_move(source, final) is True
    assert final.read_bytes() == b"texture" and not source.exists()
    source.write_bytes(b"texture")
    os.utime(final, (OLD, OLD))
    assert graph().atomic_move(source, final) is False
    assert source.exists() and final.stat().st_mtime == OLD  # the owner deletes it
    assert contents(target_folder) == {**others, final: b"texture"}
    with pytest.raises(FileNotFoundError) as raised:
        graph().atomic_move(source_folder / "none.dds", final)
    assert str(raised.value) == f"[Errno 2] No such file or directory: {str(final)!r}"
    assert contents(target_folder) == {**others, final: b"texture"}
