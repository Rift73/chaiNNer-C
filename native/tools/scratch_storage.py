"""The backend storage folder of the tests: the storage_dir of the node contexts they
build and the --storage-dir of the backends they start, with the pinned FFmpeg in it.

The tests used %APPDATA%\\chaiNNer\\backend-storage, the installed chaiNNer's folder:
Save Video updated the FFmpeg there to the pinned build, and a worker started without
--storage-dir kept its temporary files there. This folder is the checkout's own
(native/local is git-ignored), shared by no chaiNNer and no other checkout.

usage: python -B native/tools/scratch_storage.py

copies the pinned FFmpeg there from a local copy (the folder CHAINNER_C_TEST_FFMPEG names,
or the test storage of the checkout this worktree belongs to), and does nothing once it
is there. It never downloads: without a copy the setup fails, and the tests that run that
FFmpeg skip. run_tests.py runs it before its shards.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORAGE = ROOT / "native" / "local" / "test-storage"
NO_FFMPEG = (
    "the pinned FFmpeg is not in the test storage"
    " (python -B native/tools/scratch_storage.py copies it there)"
)


def local_copies() -> list[Path]:
    """The folders that may hold the pinned FFmpeg, in the order they are tried."""
    copies = []
    named = os.environ.get("CHAINNER_C_TEST_FFMPEG")
    if named:
        copies.append(Path(named))
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if common.returncode == 0:
        checkout = Path(common.stdout.strip()).parent
        copies.append(checkout / "native" / "local" / "test-storage" / "ffmpeg")
    return copies


def set_up_ffmpeg() -> Path:
    """Copies the pinned FFmpeg into STORAGE's ffmpeg folder unless it is there, and
    raises FileNotFoundError when no local copy holds it. Returns the folder."""
    from nodes.impl import ffmpeg

    folder = STORAGE / "ffmpeg"
    if ffmpeg.is_set_up(folder) and not ffmpeg.needs_update(folder):
        return folder
    for source in local_copies():
        if (
            source != folder
            and ffmpeg.is_set_up(source)
            and not ffmpeg.needs_update(source)
        ):
            if folder.exists():
                shutil.rmtree(folder)
            shutil.copytree(source, folder)
            return folder
    raise FileNotFoundError(
        "No local copy of the pinned FFmpeg: name one in CHAINNER_C_TEST_FFMPEG."
    )


def copy_ffmpeg_to(executable: Path, storage: Path) -> None:
    """Gives a test's own storage folder the FFmpeg folder of `executable`, so that Load
    Video and Save Video find it there and never download one."""
    shutil.copytree(executable.parent, storage / "ffmpeg")


if __name__ == "__main__":
    sys.path[:0] = [str(ROOT / "backend" / "src")]
    print(f"FFmpeg set up in {set_up_ffmpeg()}")
