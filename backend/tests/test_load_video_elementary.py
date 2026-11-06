"""Load Video opens the elementary streams its file filter offers (.h264, .hevc), which
give ffprobe neither a frame count nor a duration: their video packets give the Frame
Count, and the FFmpeg CLI reads them. They failed with "Unable to determine video
length. Please report." """

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
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

pytestmark = pytest.mark.skipif(not Path(FFMPEG).exists(), reason=NO_FFMPEG)

STREAMS = {
    "clip.h264": ("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"),
    "clip.hevc": (
        *("-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p"),
        *("-x265-params", "log-level=error"),
    ),
    "clip.m2v": ("-c:v", "mpeg2video", "-f", "mpeg2video"),
}


@pytest.mark.parametrize("name", sorted(STREAMS))
def test_an_elementary_stream_opens_and_reads(tmp_path: Path, name: str):
    path = tmp_path / name
    source = ("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=25", "-frames:v", "30")
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *source,
            *STREAMS[name],
            str(path),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    settings = video.ReadSettings(VideoMode.RGB, False, AUTO)
    loader = video.VideoLoader(path, ENV, settings)
    assert loader.metadata.fps == 25
    assert loader.frame_count() == 30
    read = [item[0] for item in loader.stream_items()]
    expected = np.frombuffer(
        subprocess.run(
            [
                FFMPEG,
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "v:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-sws_flags",
                video.VideoLoader.SWS_FLAGS,
                "-",
            ],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
        ).stdout,
        np.uint8,
    ).reshape(30, 48, 64, 3)
    assert len(read) == 30
    for frame, reference in zip(read, expected, strict=True):
        np.testing.assert_array_equal(frame, reference)
