"""The leftovers of interrupted installs and downloads, removed at the backend's start
(leftovers.py): pip's "~" folders in the integrated Python's site-packages and the
.part/.tmp of FFmpeg setups in the storage folder, nothing else."""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import psutil
import pytest

import leftovers

# A process ID no process has: Windows' IDs are multiples of 4, Linux's stop at 2^22.
ENDED = 4_194_307


def files(folder: Path) -> set[str]:
    return {p.name for p in folder.iterdir()}


def test_ffmpeg_setups_of_ended_processes_are_removed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    (tmp_path / f"ffmpeg.{ENDED}-k2j_8x0a.part").write_bytes(b"half an archive")
    extracted = tmp_path / f"ffmpeg.staged.{ENDED}-0abc1234.tmp"
    extracted.mkdir()
    (extracted / "ffmpeg.exe").write_bytes(b"half an executable")
    kept = {
        # a setup of a process that runs (this one)
        f"ffmpeg.{os.getpid()}-k2j_8x0a.part",
        f"ffmpeg.staged.{os.getpid()}-0abc1234.tmp",
        # FFmpeg itself, its update, and names no setup makes
        "ffmpeg",
        "ffmpeg.staged",
        f"other.{ENDED}-k2j_8x0a.part",
        f"ffmpeg.{ENDED}-k2j_8x0a.part.bak",
        "temp",
    }
    for name in kept:
        if name.endswith(".part"):
            (tmp_path / name).write_bytes(b"")
        else:
            (tmp_path / name).mkdir()

    leftovers.remove_ffmpeg_leftovers(tmp_path)

    assert files(tmp_path) == kept
    output = capsys.readouterr().out
    assert f"ffmpeg.{ENDED}-k2j_8x0a.part" in output
    assert f"ffmpeg.staged.{ENDED}-0abc1234.tmp" in output


def site_packages(folder: Path) -> Path:
    folder.mkdir()
    for name in ("torch", "torch-2.14.1.dist-info", "numpy"):
        (folder / name).mkdir()
    stash = folder / "~orch"
    stash.mkdir()
    (stash / "__init__.py").write_text("")
    (folder / "~umpy-2.5.3.dist-info").mkdir()
    # a file with the leading "~" is not a folder pip moved
    (folder / "~note.txt").write_text("")
    return folder


def test_pip_folders_of_an_interrupted_uninstall_are_removed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    folder = site_packages(tmp_path / "site-packages")
    # a Python no process runs
    leftovers.remove_pip_leftovers([folder], str(tmp_path / "python.exe"))
    assert files(folder) == {"torch", "torch-2.14.1.dist-info", "numpy", "~note.txt"}
    output = capsys.readouterr().out
    assert "~orch" in output
    assert "~umpy-2.5.3.dist-info" in output


def test_pip_folders_stay_while_another_process_runs_the_python(tmp_path: Path):
    folder = site_packages(tmp_path / "site-packages")
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
    )
    try:
        leftovers.remove_pip_leftovers([folder], psutil.Process(other.pid).exe())
    finally:
        other.kill()
        other.wait()
    assert "~orch" in files(folder)


def purelib(prefix: Path) -> Path:
    """The site-packages folder of a Python installed at prefix."""
    paths = sysconfig.get_paths(vars={"base": str(prefix), "platbase": str(prefix)})
    return Path(paths["purelib"])


def test_only_the_storage_folder_given_and_its_integrated_python_are_swept(
    tmp_path: Path,
):
    storage = tmp_path / "backend-storage"
    storage.mkdir()
    part = storage / f"ffmpeg.{ENDED}-k2j_8x0a.part"
    part.write_bytes(b"")
    unused = str(tmp_path / "python.exe")

    # no storage folder given (a backend run from source): nothing is removed
    elsewhere = tmp_path / "elsewhere"
    other_lib = purelib(elsewhere)
    other_lib.parent.mkdir(parents=True)
    site_packages(other_lib)
    leftovers.remove(None, str(elsewhere), unused)
    assert part.exists()

    # the app's storage folder and a Python that is not the app's: FFmpeg's only
    leftovers.remove(str(storage), str(elsewhere), unused)
    assert not part.exists()
    assert "~orch" in files(other_lib)

    # the app's integrated Python (<root>/python beside <root>/backend-storage): pip's too
    integrated = tmp_path / "python" / "python"
    lib = purelib(integrated)
    lib.parent.mkdir(parents=True)
    site_packages(lib)
    leftovers.remove(str(storage), str(integrated), unused)
    assert "~orch" not in files(lib)
    assert "~orch" in files(other_lib)
