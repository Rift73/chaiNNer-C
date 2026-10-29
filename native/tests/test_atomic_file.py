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
import sys
import threading
from pathlib import Path

import pytest

from nodes.impl.native_graph import graph

OLD = 1_000_000_000  # An mtime (2001) no write of this test can produce.
TEMP = re.compile(r"\.chainner-c-(\d+)-(\d+)\.tmp$")


def windows(reason: str):
    """A test of a Windows rule that POSIX file systems do not have."""
    return pytest.mark.skipif(sys.platform != "win32", reason=reason)


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


@pytest.mark.parametrize("length", [236, 244, 250, 255])
def test_long_names_write_as_open_does(tmp_path: Path, length: int):
    # The ruled temporary name would pass the 255-character name limit: the writer
    # takes the short .cnc-<pid>-<seq>.tmp in the same folder.
    final = tmp_path / ("n" * (length - 4) + ".bin")
    with open(final, "wb"):
        pass
    assert graph().atomic_write(final, b"long") is True
    assert final.read_bytes() == b"long"
    assert _commit(final, b"longer") is True
    assert final.read_bytes() == b"longer"
    temp = Path(graph().AtomicTarget(final).temp)
    assert temp.parent == tmp_path
    if len(f"{final.name}.chainner-c-{os.getpid()}-0.tmp") > 255:
        assert re.fullmatch(rf"\.cnc-{os.getpid()}-\d+\.tmp", temp.name)
    else:
        assert TEMP.search(temp.name)
    assert sorted(tmp_path.iterdir()) == [final]


@windows('MAX_PATH and the \\\\?\\ form are Windows path rules')
def test_a_full_path_past_max_path_goes_through_the_long_form(tmp_path: Path):
    # Past MAX_PATH both names of the rename take the \\?\ form; errors name the
    # path as given.
    folder = tmp_path / ("d" * 120) / ("e" * 120)
    folder.mkdir(parents=True)
    final = folder / "f.bin"
    assert len(str(final)) > 260
    assert graph().atomic_write(final, b"deep") is True
    assert _commit(final, b"deeper") is True
    assert final.read_bytes() == b"deeper"
    assert graph().AtomicTarget(final).temp.startswith("\\\\?\\")
    assert sorted(folder.iterdir()) == [final]
    missing = folder / "missing" / "f.bin"
    with pytest.raises(FileNotFoundError) as raised:
        graph().atomic_write(missing, b"x")
    assert raised.value.filename == str(missing)


@pytest.mark.parametrize("wildcard", "?*<>")
@windows('? * < > are valid in POSIX names')
def test_a_wildcard_in_a_long_name_fails_as_open_does(tmp_path: Path, wildcard: str):
    # open() refuses the name with [Errno 22] and writes nothing; so does the writer,
    # which never takes a sibling the name would match as a search pattern.
    victim = aged(tmp_path / ("b" * 240 + "X.png"), b"victim")
    final = tmp_path / ("b" * 240 + wildcard + ".png")
    with pytest.raises(OSError) as before:
        open(final, "wb")
    with pytest.raises(OSError) as after:
        graph().atomic_write(final, b"new")
    assert type(after.value) is type(before.value)
    assert str(after.value) == str(before.value) and after.value.errno == 22
    with pytest.raises(OSError) as raised:
        _commit(final, b"new")
    assert raised.value.errno == 22
    assert contents(tmp_path) == {victim: b"victim"}
    assert victim.stat().st_mtime == OLD


@pytest.mark.parametrize(
    "name",
    [
        "n" * 256 + ".bin",
        pytest.param('a".bin', marks=windows("a quote is valid in POSIX names")),
        pytest.param(
            "c" * 240 + '".bin', marks=windows("a quote is valid in POSIX names")
        ),
    ],
)
def test_an_invalid_name_fails_up_front_as_open_does(tmp_path: Path, name: str):
    # A name past 255 characters or with a quote: open() raises before writing, and so
    # do the writer and the check Save Video runs before encoding.
    final = tmp_path / name
    with pytest.raises(OSError) as before:
        open(final, "wb")
    with pytest.raises(OSError) as after:
        graph().atomic_write(final, b"payload")
    assert type(after.value) is type(before.value)
    assert str(after.value) == str(before.value)
    assert getattr(after.value, "winerror", None) is None
    with pytest.raises(OSError) as checked:
        graph().AtomicTarget(final).check()
    assert checked.value.errno == before.value.errno
    assert getattr(checked.value, "winerror", None) is None
    assert not list(tmp_path.iterdir())


def test_errors_name_the_final_path_as_open_does(tmp_path: Path):
    missing = tmp_path / "missing" / "a.bin"
    with pytest.raises(OSError) as before:
        open(missing, "wb")
    with pytest.raises(OSError) as after:
        graph().atomic_write(missing, b"x")
    assert type(after.value) is type(before.value) is FileNotFoundError
    assert str(after.value) == str(before.value)
    assert after.value.filename == str(missing)
    assert getattr(after.value, "winerror", None) is None


# Windows file handles: attributes, and sharing modes that can block a rename or delete
if sys.platform == "win32":
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
    KERNEL32.SetFileAttributesW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32]
    KERNEL32.ReadFile.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    INVALID = (None, ctypes.c_void_p(-1).value)
    SHARE_READ_WRITE, FULL_SHARING = 1 | 2, 1 | 2 | 4  # FILE_SHARE_READ/WRITE/DELETE


def hold(path: Path, share: int) -> int | None:
    """A read handle on path with this sharing, as another program holds one."""
    generic_read, open_existing = 0x80000000, 3
    return KERNEL32.CreateFileW(
        str(path), generic_read, share, None, open_existing, 0, None
    )


class Holder:
    """A handle on a file with the given sharing (none by default)."""

    def __init__(self, path: Path, share: int = 0):
        self.handle = hold(path, share)
        assert self.handle not in INVALID

    def read(self, size: int) -> bytes:
        buffer, got = ctypes.create_string_buffer(size), ctypes.c_uint32()
        assert KERNEL32.ReadFile(self.handle, buffer, size, ctypes.byref(got), None)
        return buffer.raw[: got.value]

    def close(self):
        KERNEL32.CloseHandle(self.handle)


@windows('hidden and system attributes and sharing modes are Windows file rules')
def test_failures_on_the_final_path_raise_what_open_raises(tmp_path: Path):
    # A folder, a read-only, hidden or system file and a file another program holds:
    # open(final, "wb") raises PermissionError [Errno 13]; the write raises the same,
    # whether the CRT refuses its temporary file or the rename, and leaves no
    # temporary file.
    read_only = aged(tmp_path / "ro.bin", b"keep")
    os.chmod(read_only, stat.S_IREAD)
    hidden, system = aged(tmp_path / "h.bin", b"h"), aged(tmp_path / "s.bin", b"s")
    for path, attribute in ((hidden, 2), (system, 4)):  # FILE_ATTRIBUTE_HIDDEN/SYSTEM
        assert KERNEL32.SetFileAttributesW(str(path), attribute)
    held = aged(tmp_path / "held.bin", b"held")
    folder = tmp_path / "folder"
    folder.mkdir()
    holder = Holder(held)
    try:
        for final in (folder, read_only, hidden, system, held):
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
    assert contents(tmp_path) == {
        folder: None,
        read_only: b"keep",
        hidden: b"h",
        system: b"s",
        held: b"held",
    }
    for path in (read_only, hidden, system, held):
        assert path.stat().st_mtime == OLD


@windows('names differing in case alone are one file on Windows only')
def test_a_name_of_another_case_keeps_the_spelling_on_disk(tmp_path: Path):
    # open("CASE.BIN", "wb") over case.bin writes case.bin: so do both writers.
    final = tmp_path / "case.bin"
    final.write_bytes(b"before")
    for write, data in ((graph().atomic_write, b"one"), (_commit, b"two")):
        assert write(tmp_path / "CASE.BIN", data) is True
        assert os.listdir(tmp_path) == ["case.bin"]
        assert final.read_bytes() == data


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="test_failures_on_the_final_path_raise_what_open_raises holds there",
)
def test_a_folder_or_read_only_final_path_raises_what_open_raises(tmp_path: Path):
    # The POSIX part of the test above: the error open(final, "wb") raises, and the
    # final path and the folder left as they were.
    read_only = aged(tmp_path / "ro.bin", b"keep")
    os.chmod(read_only, stat.S_IREAD)
    folder = tmp_path / "folder"
    folder.mkdir()
    for final in (folder, read_only):
        with pytest.raises(OSError) as before:
            open(final, "wb")
        for write in (graph().atomic_write, _commit):
            with pytest.raises(OSError) as after:
                write(final, b"new")
            assert type(after.value) is type(before.value)
            assert str(after.value) == str(before.value)
            assert after.value.filename == str(final)
    assert read_only.read_bytes() == b"keep"
    assert sorted(tmp_path.iterdir()) == [folder, read_only]


@windows('sharing modes are Windows file rules')
def test_a_target_held_with_delete_sharing_is_replaced(tmp_path: Path):
    # An antivirus, the indexer or a thumbnailer holds the file sharing read, write and
    # delete: open(final, "wb") succeeds there, so do both writers, and the holder
    # keeps reading the bytes it opened.
    final = aged(tmp_path / "shared.bin", b"old")
    holder = Holder(final, FULL_SHARING)
    try:
        assert graph().atomic_write(final, b"new") is True
        assert final.read_bytes() == b"new"
        assert _commit(final, b"newer") is True
        assert final.read_bytes() == b"newer"
        assert holder.read(16) == b"old"
    finally:
        holder.close()
    assert contents(tmp_path) == {final: b"newer"}


@windows('sharing modes are Windows file rules')
def test_a_target_held_without_delete_sharing_fails_as_before(tmp_path: Path):
    # A holder that shares read and write but not delete keeps the file: both writers
    # raise PermissionError [Errno 13] naming the path and leave no temporary file.
    final = aged(tmp_path / "held.bin", b"held")
    holder = Holder(final, SHARE_READ_WRITE)
    try:
        for write in (graph().atomic_write, _commit):
            with pytest.raises(PermissionError) as raised:
                write(final, b"new")
            assert raised.value.errno == 13 and raised.value.filename == str(final)
    finally:
        holder.close()
    assert contents(tmp_path) == {final: b"held"}
    assert final.stat().st_mtime == OLD


def test_concurrent_writers_and_a_reader_on_one_target(tmp_path: Path):
    # Eight writers replace one file while another thread opens and closes it with full
    # sharing, as a thumbnailer does: every write succeeds and the file is one whole
    # payload.
    final = tmp_path / "race.bin"
    payloads = [bytes([i]) * (64 << 10) for i in range(8)]
    final.write_bytes(payloads[0])
    stop, errors = threading.Event(), []

    def reader():
        while not stop.is_set():
            handle = hold(final, FULL_SHARING)
            if handle not in INVALID:
                KERNEL32.CloseHandle(handle)

    def writer(data: bytes):
        for i in range(10):
            try:
                (graph().atomic_write if i % 2 else _commit)(final, data + bytes(i))
            except OSError as error:
                errors.append(repr(error))

    threads = [threading.Thread(target=writer, args=(p,)) for p in payloads]
    watcher = threading.Thread(target=reader)
    watcher.start()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stop.set()
    watcher.join()
    assert errors == []
    data = final.read_bytes()
    assert any(data == p + bytes(i) for p in payloads for i in range(10))
    assert sorted(tmp_path.iterdir()) == [final]


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


@windows('POSIX deletes a file another program holds open')
def test_an_equal_commit_whose_temporary_file_stays_deletes_it_later(tmp_path: Path):
    # The final file holds the bytes already, and another program holds the temporary
    # file, so it cannot go now: the commit reports the equal file, and abandon() (or
    # the destructor) deletes the temporary file once let go.
    final = aged(tmp_path / "m.pth", b"model")
    target = graph().AtomicTarget(final)
    Path(target.temp).write_bytes(b"model")
    with open(target.temp, "rb"):  # shares read and write, not delete
        assert target.commit() is False
        assert Path(target.temp).exists()
    target.abandon()
    assert contents(tmp_path) == {final: b"model"}
    assert final.stat().st_mtime == OLD


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
        # POSIX unlinks a file another holder has open at once
        assert Path(target.temp).exists() == (sys.platform == "win32")
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
