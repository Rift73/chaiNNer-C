"""The worker's temporary files: tempfile's default folder (where every temporary folder
and file of the worker goes, Python's and the native modules' alike) is a folder of the
worker's own in the storage folder's temp area, named by the process ID and the
process's start time. A worker that ends without cleaning up (a crash, a kill) leaves
its folder behind; the next worker removes the folders whose process is gone. Nothing
but folders named this way is ever removed."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
from pathlib import Path

import psutil
from sanic.log import logger

# chaiNNer-<process ID>-<start time in ms, hex>
_FOLDER = re.compile(r"chaiNNer-(\d+)-([0-9a-f]+)")


def folder_name(pid: int, create_time: float) -> str:
    """The temp folder name of the process with this ID and start time (psutil's)."""
    return f"chaiNNer-{pid}-{round(create_time * 1000):x}"


def is_running(name: str) -> bool:
    """Whether the process a temp folder is named after still runs: a process with its
    ID and start time (a new process that got the ID of an ended one does not count)."""
    match = _FOLDER.fullmatch(name)
    assert match is not None
    pid = int(match.group(1))
    try:
        return folder_name(pid, psutil.Process(pid).create_time()) == name
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        # it runs, but its start time is not ours to read: keep its folder
        return True


def sweep(area: Path) -> None:
    """Removes the temp folders of processes that are gone. Anything else in the area
    (other names, files, links) is left alone."""
    for entry in os.scandir(area):
        if (
            not _FOLDER.fullmatch(entry.name)
            or entry.is_symlink()
            or entry.is_junction()
            or not entry.is_dir()
            or is_running(entry.name)
        ):
            continue

        def failed(_function: object, path: str, error: BaseException) -> None:
            logger.warning(f"Unable to remove the temporary file {path}: {error}")

        logger.info(f"Removing the temporary files of an ended worker: {entry.path}")
        shutil.rmtree(entry.path, onexc=failed)


def use_process_temp_folder(storage_dir: Path) -> Path:
    """Makes this process's folder in the storage folder's temp area tempfile's default
    folder and returns it, then removes the folders of ended processes in a background
    thread. Called once at the worker's start, before any node runs."""
    area = storage_dir / "temp"
    process = psutil.Process()
    folder = area / folder_name(process.pid, process.create_time())
    folder.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(folder)
    threading.Thread(target=sweep, args=(area,), daemon=True, name="temp sweep").start()
    return folder
