"""The worker's temporary files live in a folder of its own in the storage folder's temp
area; a new worker removes the folders of workers that are gone, and nothing else."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import pytest

from process_temp import folder_name, is_running, sweep, use_process_temp_folder


def own_folder_name() -> str:
    process = psutil.Process()
    return folder_name(process.pid, process.create_time())


def ended_folder_name() -> str:
    """The folder name of a process that ran and ended."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        name = folder_name(child.pid, psutil.Process(child.pid).create_time())
    finally:
        child.kill()
        child.wait()
    return name


def link_folder(target: Path, link: Path) -> None:
    """A junction on Windows (no privilege needed), a symbolic link elsewhere."""
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def test_a_folder_counts_as_running_only_with_its_process_and_start_time():
    process = psutil.Process()
    assert is_running(own_folder_name())
    # a new process that got the ID of an ended one is not the folder's process
    assert not is_running(folder_name(process.pid, process.create_time() + 1))
    assert not is_running(ended_folder_name())


def test_the_sweep_removes_only_the_folders_of_ended_processes(tmp_path: Path):
    area = tmp_path / "temp"
    area.mkdir()
    ended = area / ended_folder_name()
    (ended / "chaiNNer-dual-export-x").mkdir(parents=True)
    (ended / "chaiNNer-dual-export-x" / "model.onnx").write_bytes(b"onnx")
    running = area / own_folder_name()
    (running / "chaiNNer-view").mkdir(parents=True)

    # never anything this scheme did not create
    kept = ["chaiNNer-notes", "chaiNNer-123-xyz", "other", "chaiNNer-1-2-3"]
    for name in kept:
        (area / name).mkdir()
    a_file = area / ended_folder_name()
    a_file.write_bytes(b"a file, not a folder")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_bytes(b"keep")
    link = area / ended_folder_name()
    link_folder(outside, link)

    sweep(area)

    assert not ended.exists()
    assert (running / "chaiNNer-view").is_dir()
    for name in kept:
        assert (area / name).is_dir()
    assert a_file.read_bytes() == b"a file, not a folder"
    assert os.path.lexists(link)
    assert (outside / "keep.txt").read_bytes() == b"keep"


def test_the_worker_uses_its_own_temp_folder_and_sweeps_at_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(tempfile, "tempdir", None)
    ended = tmp_path / "temp" / ended_folder_name()
    ended.mkdir(parents=True)

    folder = use_process_temp_folder(tmp_path)

    assert folder == tmp_path / "temp" / own_folder_name()
    assert tempfile.gettempdir() == str(folder)
    with tempfile.TemporaryDirectory(prefix="chaiNNer-test-") as temporary:
        assert Path(temporary).parent == folder

    # the sweep runs beside the worker's start
    deadline = time.monotonic() + 10
    while ended.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not ended.exists()
    assert folder.is_dir()


def test_the_folder_name_is_short():
    # the folder adds little to the paths of the temporary files in it
    assert folder_name(4321, 1_760_848_000.123) == "chaiNNer-4321-199fab8347b"
