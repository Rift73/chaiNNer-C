"""The tests keep to their own backend storage folder (native/tools/scratch_storage.py).

They named %APPDATA%\\chaiNNer\\backend-storage, the installed chaiNNer's folder, for
the FFmpeg it downloads, and gave Save Video that folder, whose FFmpeg setup updates an
older pinned build in place: a test run could replace the user's FFmpeg.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import run_tests
import scratch_storage

from nodes.impl import ffmpeg

ROOT = Path(__file__).resolve().parents[2]
# The user's folder as the tests built it: APPDATA, then chaiNNer and backend-storage
USERS_STORAGE = re.compile(r'"chaiNNer"\s*/\s*"backend-storage"')


def test_no_test_builds_the_path_of_the_users_storage():
    named = [
        path.relative_to(ROOT).as_posix()
        for folder in ("backend/tests", "native/tests")
        for path in sorted((ROOT / folder).glob("*.py"))
        if USERS_STORAGE.search(path.read_text(encoding="utf-8"))
    ]
    assert named == []


def test_the_storage_is_in_the_checkouts_ignored_local_folder():
    assert scratch_storage.STORAGE.parent == ROOT / "native" / "local"
    ignored = (ROOT / "native" / ".gitignore").read_text(encoding="utf-8")
    assert "local/" in ignored.splitlines()


def never_download(_folder: Path) -> None:
    raise AssertionError("The tests copy FFmpeg, and never download it.")


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A local copy of the pinned FFmpeg, and a test storage with none in it."""
    monkeypatch.setattr(scratch_storage, "STORAGE", tmp_path / "storage")
    monkeypatch.setattr(ffmpeg, "setup_integrated_ffmpeg", never_download)
    monkeypatch.setattr(
        ffmpeg, "is_set_up", lambda folder: (folder / "ffmpeg").exists()
    )
    monkeypatch.setattr(ffmpeg, "needs_update", lambda _folder: False)
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "ffmpeg").write_bytes(b"the pinned build")
    return copy


def test_ffmpeg_is_copied_from_the_named_folder(
    monkeypatch: pytest.MonkeyPatch, source: Path
):
    monkeypatch.setenv("CHAINNER_C_TEST_FFMPEG", str(source))
    folder = scratch_storage.set_up_ffmpeg()
    assert folder == scratch_storage.STORAGE / "ffmpeg"
    assert (folder / "ffmpeg").read_bytes() == b"the pinned build"


def test_ffmpeg_already_there_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, source: Path
):
    monkeypatch.setenv("CHAINNER_C_TEST_FFMPEG", str(source))
    folder = scratch_storage.STORAGE / "ffmpeg"
    folder.mkdir(parents=True)
    (folder / "ffmpeg").write_bytes(b"already here")
    scratch_storage.set_up_ffmpeg()
    assert (folder / "ffmpeg").read_bytes() == b"already here"


def test_an_outdated_ffmpeg_is_replaced_by_the_copy(
    monkeypatch: pytest.MonkeyPatch, source: Path
):
    monkeypatch.setenv("CHAINNER_C_TEST_FFMPEG", str(source))
    folder = scratch_storage.STORAGE / "ffmpeg"
    folder.mkdir(parents=True)
    (folder / "ffmpeg").write_bytes(b"an older build")
    monkeypatch.setattr(ffmpeg, "needs_update", lambda path: path == folder)
    scratch_storage.set_up_ffmpeg()
    assert (folder / "ffmpeg").read_bytes() == b"the pinned build"


def test_no_local_copy_fails_without_a_download(
    monkeypatch: pytest.MonkeyPatch, source: Path
):
    monkeypatch.setenv("CHAINNER_C_TEST_FFMPEG", str(source / "missing"))
    monkeypatch.setattr(scratch_storage, "local_copies", lambda: [source / "missing"])
    with pytest.raises(FileNotFoundError, match="No local copy"):
        scratch_storage.set_up_ffmpeg()


def test_a_failed_ffmpeg_setup_lets_the_suite_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    # No local copy: the run goes on, and the tests that run FFmpeg skip
    monkeypatch.setattr(
        run_tests.subprocess,
        "run",
        lambda command, **_options: subprocess.CompletedProcess(command, 1),
    )
    run_tests.set_up_ffmpeg()
    assert "The pinned FFmpeg could not be set up (exit 1" in capsys.readouterr().out
