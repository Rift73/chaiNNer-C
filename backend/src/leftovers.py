"""What an install or a download that a crash or a kill cut short leaves behind in
chaiNNer's own folders, removed when the backend starts (run.py), once it has installed
its server dependencies (psutil among them) and before its worker downloads anything.
Only the names chaiNNer's setups make, only in the storage folder the app gives the
backend and in the app's integrated Python (beside the storage folder: <root>/python),
and nothing another running process may still use. Each item removed is logged (the
app logs the backend's output)."""

from __future__ import annotations

import os
import re
import shutil
import sysconfig
from pathlib import Path

import psutil

# FFmpeg's setup (nodes/impl/ffmpeg.py, setup_integrated_ffmpeg) downloads each archive
# to "<name>.<pid>-<random>.part" and extracts it into the folder
# "<name>.<pid>-<random>.tmp" beside the FFmpeg folder <name> ("ffmpeg", or the
# update's "ffmpeg.staged").
_FFMPEG_SETUP = re.compile(r"ffmpeg(?:\.staged)?\.(\d+)-[a-z0-9_]+\.(?:part|tmp)")


def _remove(path: Path, what: str) -> None:
    print(f"Removing {what}: {path}")

    def failed(_function: object, failed_path: str, error: BaseException) -> None:
        print(f"Unable to remove {failed_path}: {error}")

    if path.is_dir():
        shutil.rmtree(path, onexc=failed)
    else:
        try:
            path.unlink()
        except OSError as error:
            failed(None, str(path), error)


def remove_ffmpeg_leftovers(storage_dir: Path) -> None:
    """Removes the downloads and extractions of FFmpeg setups in storage_dir whose
    process has ended (a running process with the ID, even another one, keeps them)."""
    found = [
        (entry, int(match.group(1)))
        for entry in os.scandir(storage_dir)
        if (match := _FFMPEG_SETUP.fullmatch(entry.name)) is not None
        and not entry.is_symlink()
        and not entry.is_junction()
    ]
    for entry, pid in found:
        if not psutil.pid_exists(pid):
            _remove(Path(entry.path), "a file of an interrupted FFmpeg setup")


def remove_pip_leftovers(site_packages: list[Path], executable: str) -> None:
    """Removes the folders pip moves a package it uninstalls into (to upgrade or replace
    it) and deletes once done: beside the package, named with a leading "~" ("~orch" for
    torch, pip's AdjacentTempDirectory), which no package name has. Left alone while
    another process runs this Python: its pip may be using them."""
    found = [
        entry
        for folder in site_packages
        if folder.is_dir()
        for entry in os.scandir(folder)
        if entry.name.startswith("~")
        and entry.is_dir(follow_symlinks=False)
        and not entry.is_junction()
    ]
    if not found:
        return
    wanted = os.path.normcase(os.path.realpath(executable))
    for process in psutil.process_iter(["pid", "exe"]):
        exe = process.info["exe"]
        if (
            process.info["pid"] != os.getpid()
            and exe
            and os.path.normcase(os.path.realpath(exe)) == wanted
        ):
            print(
                f"Leaving {len(found)} folders of an interrupted pip run: another"
                f" process runs {executable}"
            )
            return
    for entry in found:
        _remove(Path(entry.path), "a folder of an interrupted pip run")


def remove(storage_dir: str | None, prefix: str, executable: str) -> None:
    """Removes the leftovers in the storage folder the app gave (none given: the backend
    runs from source, and nothing is removed), and pip's in the integrated Python's
    site-packages when `prefix` (sys.prefix) is that Python's."""
    if storage_dir is None:
        return
    storage = Path(storage_dir)
    if storage.is_dir():
        remove_ffmpeg_leftovers(storage)
    if Path(prefix).resolve().is_relative_to((storage.parent / "python").resolve()):
        paths = sysconfig.get_paths(vars={"base": prefix, "platbase": prefix})
        remove_pip_leftovers(
            sorted({Path(paths["purelib"]), Path(paths["platlib"])}), executable
        )
