"""Packaging file primitives: guarded walks, verified copies and writes, lock probe."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import package_files
import pytest


def write(root, relative, data=b"x"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_walk_prunes_installed_backend_logs_updater_and_bytecode(tmp_path):
    for name in (
        "chaiNNer.exe",
        "squirrel.exe",
        "debug.log",
        "Squirrel-UpdateSelf.LOG",
        "resources/app/package.json",
        "resources/src/run.py",
        "resources/srcmap/keep.txt",
        "lib/a.py",
        "lib/squirrel.exe",
        "lib/__pycache__/a.cpython-311.pyc",
    ):
        write(tmp_path, name)
    assert package_files.walk(tmp_path, package_files.shell_skip) == [
        "chaiNNer.exe",
        "lib/__pycache__/a.cpython-311.pyc",
        "lib/a.py",
        "lib/squirrel.exe",
        "resources/app/package.json",
        "resources/srcmap/keep.txt",
    ]
    assert package_files.walk(tmp_path, package_files.cache_skip) == [
        "Squirrel-UpdateSelf.LOG",
        "chaiNNer.exe",
        "debug.log",
        "lib/a.py",
        "lib/squirrel.exe",
        "resources/app/package.json",
        "resources/src/run.py",
        "resources/srcmap/keep.txt",
        "squirrel.exe",
    ]


@pytest.mark.skipif(os.name != "nt", reason="Windows directory junctions")
def test_walk_refuses_a_junction_even_where_it_would_be_pruned(tmp_path):
    import _winapi

    outside = tmp_path / "outside"
    write(outside, "run.py")
    root = tmp_path / "app"
    write(root, "chaiNNer.exe")
    junction = root / "resources/src"
    junction.parent.mkdir()
    _winapi.CreateJunction(str(outside), str(junction))
    try:
        with pytest.raises(ValueError, match="Refusing reparse point"):
            package_files.walk(root, package_files.shell_skip)
    finally:
        junction.rmdir()


def test_walk_raises_when_a_directory_cannot_be_listed(tmp_path, monkeypatch):
    write(tmp_path, "a/x.txt")
    write(tmp_path, "b/y.txt")
    refused = tmp_path / "b"
    scandir = os.scandir

    def guarded(path):
        if Path(path) == refused:
            raise PermissionError("listing refused")
        return scandir(path)

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", guarded)
        with pytest.raises(PermissionError, match="listing refused"):
            package_files.walk(tmp_path)


def test_copy_verified_refuses_a_target_that_is_the_source(tmp_path):
    source = write(tmp_path, "source/file.bin", b"payload")
    target = tmp_path / "target/file.bin"
    target.parent.mkdir()
    os.link(source, target)
    with pytest.raises(ValueError, match="shares a source file"):
        package_files.copy_verified((source, target, True))


def test_copy_verified_reuses_only_a_hash_matched_target(tmp_path, monkeypatch):
    source = write(tmp_path, "source/file.bin", b"payload")
    target = write(tmp_path, "target/file.bin", b"payload")
    expected = (target, package_files.sha(b"payload"), len(b"payload"))

    def forbid_copy(*_arguments):
        pytest.fail("A hash-matched target must be reused")

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copy2", forbid_copy)
        assert package_files.copy_verified((source, target, True)) == expected
    target.write_bytes(b"partial")
    assert package_files.copy_verified((source, target, True)) == expected
    assert target.read_bytes() == b"payload"


def test_replace_verified_swaps_the_entry_instead_of_writing_through(tmp_path):
    outside = write(tmp_path, "outside.bin", b"shared bytes")
    target = tmp_path / "package/file.bin"
    target.parent.mkdir()
    os.link(outside, target)
    package_files.replace_verified(
        target, b"new bytes", package_files.sha(b"new bytes")
    )
    assert target.read_bytes() == b"new bytes"
    assert outside.read_bytes() == b"shared bytes"
    assert [path.name for path in target.parent.iterdir()] == ["file.bin"]


def test_replace_verified_removes_its_temporary_when_the_swap_fails(
    tmp_path, monkeypatch
):
    target = write(tmp_path, "file.bin", b"old")

    def mapped(*_arguments):
        raise PermissionError("target is mapped")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", mapped)
        with pytest.raises(PermissionError, match="mapped"):
            package_files.replace_verified(target, b"new", package_files.sha(b"new"))
    assert target.read_bytes() == b"old"
    assert [path.name for path in tmp_path.iterdir()] == ["file.bin"]


def test_replace_verified_rejects_bytes_that_differ_from_the_record(tmp_path):
    with pytest.raises(OSError, match="differ from the planned record"):
        package_files.replace_verified(
            tmp_path / "file.bin", b"new", package_files.sha(b"other")
        )


def test_probe_unlocked_reports_every_unwritable_target(tmp_path):
    free = write(tmp_path, "free.py")
    first = write(tmp_path, "first.dll")
    second = write(tmp_path, "second.pyd")
    absent = tmp_path / "new.py"
    for path in (first, second):
        path.chmod(stat.S_IREAD)
    try:
        with pytest.raises(RuntimeError, match="locked or read-only") as raised:
            package_files.probe_unlocked([free, first, absent, second])
    finally:
        for path in (first, second):
            path.chmod(stat.S_IREAD | stat.S_IWRITE)
    message = str(raised.value)
    assert str(first) in message
    assert str(second) in message
    assert str(free) not in message
    package_files.probe_unlocked([free, first, absent, second])


def test_git_snapshot_reads_tracked_backend_paths_and_provenance(tmp_path):
    def git(*arguments):
        subprocess.run(
            ["git", "-C", str(tmp_path), *arguments], check=True, capture_output=True
        )

    git("init", "-q")
    write(tmp_path, "backend/src/run.py", b"print()\n")
    write(tmp_path, "backend/src/nodes/ünïcode.py", b"value = 1\n")
    write(tmp_path, "src/frontend.ts", b"export {};\n")
    git("add", "-A")
    identity = ("-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid")
    git(*identity, "commit", "-q", "-m", "fixture")
    tracked, provenance = package_files.git_snapshot(tmp_path)
    assert tracked == {"run.py", "nodes/ünïcode.py"}
    assert len(str(provenance["git_head"])) == 40
    assert provenance["git_branch"] not in ("", "HEAD")
    assert provenance["backend_dirty"] is False
    write(tmp_path, "src/frontend.ts", b"export const changed = 1;\n")
    assert package_files.git_snapshot(tmp_path)[1]["backend_dirty"] is False
    write(tmp_path, "backend/src/untracked.py", b"")
    tracked, provenance = package_files.git_snapshot(tmp_path)
    assert "untracked.py" not in tracked
    assert provenance["backend_dirty"] is True


def test_git_snapshot_reports_git_errors_with_stderr(tmp_path):
    with pytest.raises(RuntimeError, match="not a git repository"):
        package_files.git_snapshot(tmp_path)
