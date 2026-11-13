"""A video whose cover art comes first. An MP4 whose metadata box stands before its tracks
lists its attached picture as the first video stream, so Load Video read one frame of
the cover at FPS 90000, where FFmpeg's own choice is the video."""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scratch_storage import NO_FFMPEG, STORAGE

# The video module loads the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from nodes.impl import getnative_media, video
from nodes.impl.colour_info import (
    ChromaLocation,
    Matrix,
    Primaries,
    Range,
    Transfer,
    VideoMode,
)
from nodes.impl.ffmpeg import FFMpegEnv, get_executable_path
from nodes.impl.video import _UseCliReader

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


def boxes(data: bytes, start: int, end: int) -> list[tuple[bytes, int, int]]:
    found = []
    at = start
    while at < end:
        size = struct.unpack(">I", data[at : at + 4])[0]
        found.append((data[at + 4 : at + 8], at, size))
        at += size
    return found


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """(the video with its cover first, the video alone)."""
    folder = tmp_path_factory.mktemp("cover")
    plain, cover, last = folder / "plain.mp4", folder / "cover.jpg", folder / "last.mp4"
    run_ffmpeg(
        *("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=25", "-frames:v", "12"),
        *("-c:v", "libx264", "-preset", "ultrafast", str(plain)),
    )
    run_ffmpeg(
        *("-f", "lavfi", "-i", "color=c=red:size=32x24", "-frames:v", "1", str(cover))
    )
    run_ffmpeg(
        *("-i", str(plain), "-i", str(cover), "-map", "0", "-map", "1", "-c", "copy"),
        *("-disposition:v:1", "attached_pic", str(last)),
    )
    data = last.read_bytes()
    _, moov, moov_size = next(b for b in boxes(data, 0, len(data)) if b[0] == b"moov")
    children = boxes(data, moov + 8, moov + moov_size)
    order = sorted(children, key=lambda child: child[0] != b"udta")
    first = folder / "first.mp4"
    first.write_bytes(
        data[: moov + 8]
        + b"".join(data[at : at + size] for _, at, size in order)
        + data[moov + moov_size :]
    )
    return first, plain


@pytest.mark.parametrize("cli", [False, True])
def test_the_video_is_read_not_its_cover(
    files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, cli: bool
):
    first, plain = files
    if cli:
        monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    else:
        monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    loader = video.VideoLoader(first, ENV, SETTINGS)
    assert (loader.metadata.width, loader.metadata.height) == (64, 48)
    assert loader.metadata.fps == 25
    assert loader.frame_count() == 12
    read = [item[0] for item in loader.stream_items()]
    expected = [
        item[0] for item in video.VideoLoader(plain, ENV, SETTINGS).stream_items()
    ]
    assert len(read) == len(expected) == 12
    for frame, reference in zip(read, expected, strict=True):
        np.testing.assert_array_equal(frame, reference)
    preview = video.preview_frame("cover-first", loader, 7)
    video.close_previews()
    np.testing.assert_array_equal(preview, expected[7])


def test_a_lone_picture_is_still_read(tmp_path: Path):
    path = tmp_path / "song.mp3"
    run_ffmpeg(
        *("-f", "lavfi", "-i", "sine=frequency=440:duration=1"),
        *("-f", "lavfi", "-i", "color=c=red:size=32x24", "-frames:v", "1"),
        *("-map", "0", "-map", "1", "-c:v", "mjpeg", "-c:a", "libmp3lame"),
        *("-disposition:v:0", "attached_pic", str(path)),
    )
    frames = list(video.VideoLoader(path, ENV).stream_frames())
    assert len(frames) == 1 and frames[0].shape == (24, 32, 3)


@pytest.mark.parametrize("cli", [False, True])
def test_get_native_resolution_reads_the_video_not_its_cover(
    files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, cli: bool
):
    first, plain = files
    info, reference = getnative_media.probe(first), getnative_media.probe(plain)
    assert (info.width, info.height, info.pix_fmt) == (64, 48, reference.pix_fmt)
    if cli:
        # Without a frame index, the frame comes from the FFmpeg CLI.
        monkeypatch.setattr(getnative_media, "frame_index", unindexed)
    point = getnative_media.SamplePoint(0.21, getnative_media.REQUESTED)
    plane = getnative_media.decode_frame(first, info, point, lambda: ENV, lambda: None)
    expected = getnative_media.decode_frame(
        plain, reference, point, lambda: ENV, lambda: None
    )
    np.testing.assert_array_equal(plane[0], expected[0])


def unindexed(*_args: object) -> object:
    raise _UseCliReader("read through the CLI in this test")
