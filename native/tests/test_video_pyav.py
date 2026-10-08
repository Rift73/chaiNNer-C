"""Load Video's in-process reader (PyAV): byte-identical frames to the FFmpeg CLI
reader, and the CLI reader for files PyAV would read differently or cannot open."""

from __future__ import annotations

import shutil
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

from nodes.impl import video
from nodes.impl.ffmpeg import FFMpegEnv

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None, reason="needs ffmpeg and ffprobe on PATH"
)


def _clip(path: Path, *arguments: str) -> Path:
    assert FFMPEG is not None
    command = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *arguments,
        str(path),
    ]
    subprocess.run(command, check=True, stdin=subprocess.DEVNULL)
    return path


def _source(size: str, rate: int = 30) -> list[str]:
    return ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}", "-frames:v", "30"]


def _read(path: Path, monkeypatch, cli: bool):
    assert FFMPEG is not None and FFPROBE is not None
    if cli:
        monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    else:
        monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    frames = video.VideoLoader(path, FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE))
    stream = frames.stream_frames()
    return isinstance(stream, types.GeneratorType), list(stream)


CASES = {
    "h264-420": (*_source("1280x720"), "-c:v", "libx264", "-pix_fmt", "yuv420p"),
    "ffv1-444-odd": (*_source("639x361"), "-c:v", "ffv1", "-pix_fmt", "yuv444p"),
    "mjpeg-full-range": (*_source("320x240"), "-c:v", "mjpeg", "-pix_fmt", "yuvj420p"),
    "h264-bt601": (
        *_source("640x480"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "smpte170m",
        "-color_primaries",
        "smpte170m",
        "-color_trc",
        "smpte170m",
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_pyav_frames_equal_the_cli_reader(tmp_path, monkeypatch, case):
    suffix = ".mkv" if case.startswith("ffv1") else ".mp4"
    path = _clip(tmp_path / f"clip-é{suffix}", *CASES[case])
    in_process, frames = _read(path, monkeypatch, cli=False)
    piped, expected = _read(path, monkeypatch, cli=True)
    assert in_process and not piped
    assert len(frames) == len(expected) == 30
    for frame, reference in zip(frames, expected, strict=True):
        assert frame.flags.c_contiguous and not frame.flags.writeable
        np.testing.assert_array_equal(frame, reference)


def test_rotated_and_variable_rate_files_use_the_cli_reader(tmp_path, monkeypatch):
    plain = _clip(tmp_path / "plain.mp4", *CASES["h264-420"])
    rotated = _clip(
        tmp_path / "rotated.mp4",
        "-display_rotation",
        "90",
        "-i",
        str(plain),
        "-c",
        "copy",
    )
    variable = _clip(
        tmp_path / "variable.mp4",
        *_source("640x360"),
        "-vf",
        "setpts='if(lt(N,10),N,N+15)/(30*TB)'",
        "-fps_mode",
        "vfr",
        "-c:v",
        "libx264",
    )
    for path in (rotated, variable):
        in_process, _ = _read(path, monkeypatch, cli=False)
        assert not in_process, path


def test_without_pyav_the_cli_reader_is_used(tmp_path, monkeypatch):
    path = _clip(tmp_path / "plain.mp4", *CASES["h264-420"])
    monkeypatch.setattr(video, "_pyav", lambda: None)
    in_process, frames = _read(path, monkeypatch, cli=False)
    assert not in_process and len(frames) == 30
