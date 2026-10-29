"""The backend storage folder of the tests: the storage_dir of the node contexts they
build and the --storage-dir of the backends they start, with the pinned FFmpeg in it.

The tests used %APPDATA%\\chaiNNer\\backend-storage, the installed chaiNNer's folder:
Save Video updated the FFmpeg there to the pinned build, and a worker started without
--storage-dir kept its temporary files there. This folder is the checkout's own
(native/local is git-ignored), shared by no chaiNNer and no other checkout.

usage: python -B native/tools/scratch_storage.py

puts the pinned FFmpeg there with the backend's own setup (a download the first time,
nothing once it is there); run_tests.py runs it before its shards. Without it, the tests
that run that FFmpeg skip.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORAGE = ROOT / "native" / "local" / "test-storage"
NO_FFMPEG = (
    "the pinned FFmpeg is not in the test storage"
    " (python -B native/tools/scratch_storage.py puts it there)"
)


def set_up_ffmpeg() -> Path:
    """Puts the pinned FFmpeg into STORAGE's ffmpeg folder unless it is there: the
    setup FFMpegEnv.get_integrated runs, without its wait for a later start when an
    older build is in place. Returns the folder."""
    from nodes.impl import ffmpeg

    folder = STORAGE / "ffmpeg"
    if not ffmpeg.is_set_up(folder) or ffmpeg.needs_update(folder):
        ffmpeg.setup_integrated_ffmpeg(folder)
    return folder


if __name__ == "__main__":
    sys.path[:0] = [str(ROOT / "backend" / "src")]
    print(f"FFmpeg set up in {set_up_ffmpeg()}")
