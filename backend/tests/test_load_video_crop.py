"""A stream that crops its coded frames at the left (H.264 and HEVC SPS cropping, as a
lossless crop writes it). libavcodec keeps a crop aligned unless told otherwise, so it
gave PyAV a frame wider than the file's, which libswscale then squeezed into the file's
width, every frame resampled; the FFmpeg CLI crops exactly. Both readers give FFmpeg's
frames."""

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

CROP = "crop_left=8:crop_right=16:crop_top=6:crop_bottom=10"
FILES = {
    "h264.mp4": (
        ("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"),
        f"h264_metadata={CROP}",
    ),
    "h264-444-odd.mkv": (
        ("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv444p"),
        "h264_metadata=crop_left=3:crop_right=4:crop_top=5:crop_bottom=2",
    ),
    "h264-10.mkv": (
        ("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le"),
        f"h264_metadata={CROP}",
    ),
    "hevc.mkv": (
        (
            *("-c:v", "libx265", "-preset", "ultrafast"),
            *("-pix_fmt", "yuv420p10le", "-x265-params", "log-level=error"),
        ),
        f"hevc_metadata={CROP}",
    ),
}
SETTINGS = {
    "rgb-float": video.ReadSettings(VideoMode.RGB, True, AUTO),
    "rgb-8": video.ReadSettings(VideoMode.RGB, False, AUTO),
    "planes": video.ReadSettings(VideoMode.PLANES, False, AUTO),
}


def run_ffmpeg(*arguments: str) -> None:
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *arguments],
        check=True,
        stdin=subprocess.DEVNULL,
    )


@pytest.fixture(scope="module")
def folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("crop")


def cropped(folder: Path, name: str) -> Path:
    path = folder / name
    if not path.exists():
        codec, bitstream_filter = FILES[name]
        whole = folder / f"whole-{name}"
        run_ffmpeg(
            *("-f", "lavfi", "-i", "testsrc2=size=128x96:rate=25", "-frames:v", "5"),
            *codec,
            str(whole),
        )
        run_ffmpeg(
            *("-i", str(whole), "-c", "copy", "-bsf:v", bitstream_filter, str(path))
        )
    return path


def items(
    path: Path,
    settings: video.ReadSettings,
    monkeypatch: pytest.MonkeyPatch,
    cli: bool,
    start: int = 0,
    stop: int | None = None,
) -> list[list[np.ndarray]]:
    if cli:
        monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    else:
        monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    loader = video.VideoLoader(path, ENV, settings)
    return [
        [value for value in item if value is not None]
        for item in loader.stream_items(start, stop)
    ]


@pytest.mark.parametrize("mode", sorted(SETTINGS))
@pytest.mark.parametrize("name", sorted(FILES))
def test_both_readers_give_ffmpeg_s_cropped_frames(
    folder: Path, monkeypatch: pytest.MonkeyPatch, name: str, mode: str
):
    path = cropped(folder, name)
    settings = SETTINGS[mode]
    reader = vars(video.VideoLoader)["_pyav_reader"]
    by_pyav: list[bool] = []

    def watched(*args: object) -> object:
        frames = reader(*args)
        by_pyav.append(frames is not None)
        return frames

    monkeypatch.setattr(video.VideoLoader, "_pyav_reader", watched)
    read = items(path, settings, monkeypatch, cli=False)
    assert by_pyav
    assert all(by_pyav)
    expected = items(path, settings, monkeypatch, cli=True)
    assert len(read) == len(expected) == 5
    for item, reference in zip(read, expected, strict=True):
        for value, wanted in zip(item, reference, strict=True):
            np.testing.assert_array_equal(value, wanted)


@pytest.mark.parametrize("name", sorted(FILES))
def test_a_seek_and_the_preview_crop_as_the_full_read(
    folder: Path, monkeypatch: pytest.MonkeyPatch, name: str
):
    path = cropped(folder, name)
    settings = SETTINGS["rgb-float"]
    whole = items(path, settings, monkeypatch, cli=False)
    for n in (2, 4):
        ((frame,),) = items(path, settings, monkeypatch, cli=False, start=n, stop=n + 1)
        np.testing.assert_array_equal(frame, whole[n][0])
        loader = video.VideoLoader(path, ENV, settings)
        preview = video.preview_frame(f"crop-{name}", loader, n)
        np.testing.assert_array_equal(preview, whole[n][0])
    video.close_previews()
