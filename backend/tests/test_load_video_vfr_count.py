"""Load Video's Frame Count of a variable-rate MP4, which the FFmpeg CLI reads at a
constant rate. The track's sample count (50) was the count of the 240 frames read: the
seek bar stopped at a fifth of the video. The estimate is now its duration at that rate."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from scratch_storage import NO_FFMPEG, STORAGE

# The video module loads the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from nodes.impl import video
from nodes.impl.colour_info import (
    ChromaLocation,
    Matrix,
    Primaries,
    Range,
    Transfer,
    VideoMode,
)
from nodes.impl.ffmpeg import FFMpegEnv, get_executable_path

FFMPEG, FFPROBE = map(str, get_executable_path(STORAGE / "ffmpeg"))
ENV = FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE)
AUTO = (Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO, ChromaLocation.AUTO)
SETTINGS = video.ReadSettings(VideoMode.RGB, False, AUTO)

pytestmark = pytest.mark.skipif(not Path(FFMPEG).exists(), reason=NO_FFMPEG)


@pytest.mark.parametrize("suffix", [".mp4", ".mov", ".mkv"])
def test_the_estimate_is_near_the_frames_read(tmp_path: Path, suffix: str):
    path = tmp_path / f"variable{suffix}"
    subprocess.run(
        [
            FFMPEG,
            *("-hide_banner", "-loglevel", "error", "-y"),
            *("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=25", "-frames:v", "50"),
            *("-vf", "setpts=N*(0.03+0.02*mod(N\\,3))/TB", "-fps_mode", "vfr"),
            *("-c:v", "libx264", "-preset", "ultrafast", str(path)),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    loader = video.VideoLoader(path, ENV, SETTINGS)
    assert loader.exact_frame_count() is None
    read = sum(1 for _ in loader.stream_items())
    assert read > 200
    assert loader.frame_count() == pytest.approx(read, rel=0.02)
