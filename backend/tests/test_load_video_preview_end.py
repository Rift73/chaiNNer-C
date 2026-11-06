"""Load Video's Preview past the frames the FFmpeg CLI reads. The Frame Count of a file
the CLI reads is an estimate, which a longer audio track or a cut file puts past the
frames: the seek bar reaches a frame that is not there, and the Preview's error failed
the whole run, where its documentation shows the last frame."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from scratch_storage import NO_FFMPEG, STORAGE

# The node's modules load the native backend, which GitHub's checks do not build.
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
from packages.chaiNNer_standard.image.video_frames.load_video import load_preview

FFMPEG, FFPROBE = map(str, get_executable_path(STORAGE / "ffmpeg"))
ENV = FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE)
AUTO = (Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO, ChromaLocation.AUTO)
SETTINGS = video.ReadSettings(VideoMode.RGB, True, AUTO)

pytestmark = pytest.mark.skipif(not Path(FFMPEG).exists(), reason=NO_FFMPEG)


def run_ffmpeg(*arguments: str) -> None:
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *arguments],
        check=True,
        stdin=subprocess.DEVNULL,
    )


def test_past_an_estimated_count_the_last_frame_is_shown(tmp_path: Path):
    whole = tmp_path / "whole.mp4"
    run_ffmpeg(
        *("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=25", "-frames:v", "50"),
        *("-vf", "setpts=N*(0.03+0.02*mod(N\\,3))/TB", "-fps_mode", "vfr"),
        *("-c:v", "libx264", "-preset", "ultrafast", "-movflags", "faststart"),
        str(whole),
    )
    # An interrupted copy: the header still gives the whole video's length.
    path = tmp_path / "cut.mp4"
    data = whole.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    loader = video.VideoLoader(path, ENV, SETTINGS)
    assert loader.exact_frame_count() is None
    frames = [item[0] for item in loader.stream_items()]
    count = loader.frame_count()
    assert count > len(frames) > 0  # the case this guards
    image = load_preview("past-the-end", loader, count - 1)
    video.close_previews()
    np.testing.assert_array_equal(image, frames[-1])


def test_a_video_without_frames_still_says_so(tmp_path: Path):
    path = tmp_path / "no-frames.avi"
    run_ffmpeg(
        *("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=25", "-frames:v", "0"),
        *("-c:v", "mpeg4", str(path)),
    )
    loader = video.VideoLoader(path, ENV, SETTINGS)
    with pytest.raises((ValueError, RuntimeError)):
        load_preview("no-frames", loader, 0)
    video.close_previews()
