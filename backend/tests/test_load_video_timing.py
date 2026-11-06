"""Load Video keeps a video's timing in every container. MPEG-TS, MPEG-PS, FLV, NUT and
GIF report a constant rate for a track whose frames are not evenly spaced, as Matroska
and WebM do, so the frames' timestamps decide who reads them: the FFmpeg CLI, whose
constant rate keeps the video's length, or PyAV, whose stored frames would play the
video faster than its audio. Two captures joined by a file copy restart their timestamps,
where a seek by timestamp would find the wrong frame."""

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
SETTINGS = video.ReadSettings(VideoMode.RGB, False, AUTO)

pytestmark = pytest.mark.skipif(not Path(FFMPEG).exists(), reason=NO_FFMPEG)

SOURCE = ("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=25", "-frames:v", "30")
UNEVEN = ("-vf", "setpts=N*(0.03+0.02*mod(N\\,3))/TB", "-fps_mode", "vfr")


def run_ffmpeg(*arguments: str) -> None:
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *arguments],
        check=True,
        stdin=subprocess.DEVNULL,
    )


def duration(path: Path) -> float:
    probe = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
    )
    return float(probe.stdout.strip())


def frames(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli: bool,
    start: int = 0,
    stop: int | None = None,
) -> list[np.ndarray]:
    if cli:
        monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    else:
        monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    loader = video.VideoLoader(path, ENV, SETTINGS)
    read = [item[0] for item in loader.stream_items(start, stop)]
    assert all(frame is not None for frame in read)
    return [frame for frame in read if frame is not None]


@pytest.mark.parametrize(
    ("suffix", "codec"),
    [
        (".ts", ("-c:v", "libx264", "-preset", "ultrafast")),
        (".flv", ("-c:v", "libx264", "-preset", "ultrafast")),
        (".mpg", ("-c:v", "mpeg2video", "-f", "mpeg")),
        (".nut", ("-c:v", "libx264", "-preset", "ultrafast")),
        (".gif", ()),
    ],
)
def test_uneven_frames_keep_the_video_s_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    codec: tuple[str, ...],
):
    path = tmp_path / f"uneven{suffix}"
    run_ffmpeg(*SOURCE, *UNEVEN, *codec, str(path))
    read = frames(path, monkeypatch, cli=False)
    expected = frames(path, monkeypatch, cli=True)
    fps = video.VideoLoader(path, ENV, SETTINGS).metadata.fps
    assert len(read) == len(expected)
    assert len(read) / fps == pytest.approx(duration(path), rel=0.1)
    for frame, reference in zip(read, expected, strict=True):
        np.testing.assert_array_equal(frame, reference)


@pytest.mark.parametrize("suffix", [".ts", ".flv", ".nut"])
def test_even_frames_stay_in_process(tmp_path: Path, suffix: str):
    # The check keeps PyAV's seeks for the files whose frames are on the rate's grid.
    path = tmp_path / f"even{suffix}"
    run_ffmpeg(*SOURCE, "-c:v", "libx264", "-preset", "ultrafast", str(path))
    assert video.VideoLoader(path, ENV, SETTINGS).exact_frame_count() == 30


def test_a_packet_without_a_timestamp_keeps_its_reason_and_estimate(tmp_path: Path):
    # An MPEG-PS times only some packets: the packets give the estimate, as before.
    path = tmp_path / "untimed.mpg"
    run_ffmpeg(*SOURCE, "-c:v", "mpeg2video", "-bf", "2", "-f", "mpeg", str(path))
    loader = video.VideoLoader(path, ENV, SETTINGS)
    assert loader.frame_count() == 30


def test_joined_captures_seek_to_the_full_read_s_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    parts = []
    for name, colour in (("a.ts", "red"), ("b.ts", "blue")):
        part = tmp_path / name
        run_ffmpeg(
            *("-f", "lavfi", "-i", f"color=c={colour}:size=64x48:rate=25"),
            *("-frames:v", "10", "-c:v", "libx264", "-preset", "ultrafast", str(part)),
        )
        parts.append(part.read_bytes())
    path = tmp_path / "joined.ts"
    path.write_bytes(b"".join(parts))
    whole = frames(path, monkeypatch, cli=False)
    assert len(whole) == 20
    for n in (0, 5, 10, 15):
        (frame,) = frames(path, monkeypatch, cli=False, start=n, stop=n + 1)
        np.testing.assert_array_equal(frame, whole[n])
