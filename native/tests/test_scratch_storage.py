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


@pytest.mark.parametrize(
    ("set_up", "outdated", "downloads"),
    [(False, False, True), (True, True, True), (True, False, False)],
)
def test_ffmpeg_is_set_up_unless_the_pinned_build_is_there(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    set_up: bool,
    outdated: bool,
    downloads: bool,
):
    monkeypatch.setattr(scratch_storage, "STORAGE", tmp_path)
    monkeypatch.setattr(ffmpeg, "is_set_up", lambda _folder: set_up)
    monkeypatch.setattr(ffmpeg, "needs_update", lambda _folder: outdated)
    downloaded: list[Path] = []
    monkeypatch.setattr(ffmpeg, "setup_integrated_ffmpeg", downloaded.append)
    assert scratch_storage.set_up_ffmpeg() == tmp_path / "ffmpeg"
    assert downloaded == ([tmp_path / "ffmpeg"] if downloads else [])


def test_a_failed_ffmpeg_setup_lets_the_suite_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    # Offline, the download fails: the run goes on, and the tests that run it skip
    monkeypatch.setattr(
        run_tests.subprocess,
        "run",
        lambda command, **_options: subprocess.CompletedProcess(command, 1),
    )
    run_tests.set_up_ffmpeg()
    assert "The pinned FFmpeg could not be set up (exit 1" in capsys.readouterr().out
