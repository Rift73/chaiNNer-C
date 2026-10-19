"""Load Video's in-process reader (PyAV): byte-identical frames to the FFmpeg CLI
reader (rotated files upright in both), and the CLI reader for variable-rate files
or files PyAV cannot open."""

from __future__ import annotations

import gc
import itertools
import os
import shutil
import subprocess
import types
import weakref
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from nodes.impl import video
from nodes.impl.colour_info import (
    ChromaLocation,
    Matrix,
    Primaries,
    Range,
    Transfer,
    VideoMode,
)
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
    # Matroska and WebM pass the timestamp check (B-frames store them out of order).
    "mkv-h264-bframes": (*_source("320x240"), "-c:v", "libx264", "-bf", "3"),
    "webm-vp9": (*_source("320x240"), "-c:v", "libvpx-vp9", "-deadline", "realtime"),
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
    prefix = case.split("-")[0]
    suffix = {"ffv1": ".mkv", "mkv": ".mkv", "webm": ".webm"}.get(prefix, ".mp4")
    path = _clip(tmp_path / f"clip-é{suffix}", *CASES[case])
    in_process, frames = _read(path, monkeypatch, cli=False)
    piped, expected = _read(path, monkeypatch, cli=True)
    assert in_process and not piped
    assert len(frames) == len(expected) == 30
    for frame, reference in zip(frames, expected, strict=True):
        assert frame.flags.c_contiguous and not frame.flags.writeable
        np.testing.assert_array_equal(frame, reference)


@pytest.mark.parametrize(("size", "padded"), [("639x361", True), ("320x240", False)])
def test_float_rgb_frames_are_todays_bytes(tmp_path, size, padded):
    # A float RGB frame goes from PyAV's G, B and R planes (rows padded to line_size)
    # straight into the BGR image: today's image from the CLI's bytes, bit for bit.
    av = video._pyav()
    assert av is not None and FFMPEG is not None and FFPROBE is not None
    path = _clip(
        tmp_path / "clip.mkv", *_source(size), "-c:v", "ffv1", "-pix_fmt", "yuv444p10le"
    )
    auto = (Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO, ChromaLocation.AUTO)
    settings = video.ReadSettings(VideoMode.RGB, True, auto)
    loader = video.VideoLoader(
        path, FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE), settings
    )
    (frame_format,) = loader.formats
    assert frame_format.pix_fmt == "gbrpf32le"
    width, height = frame_format.size
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        convert = video._converter(av, stream, [], frame_format)
        for frame in itertools.islice(container.decode(stream), 3):
            convert.push(frame)
            converted = convert.pull()
            assert (converted.planes[0].line_size > width * 4) == padded
            image = video._frame_values(converted, frame_format)
            data = video._frame_bytes(converted, frame_format)
            g, b, r = data.view(np.float32).reshape(3, height, width)
            today = np.dstack((b, g, r))
            assert isinstance(image, np.ndarray)
            assert (image.shape, image.dtype) == (today.shape, today.dtype)
            assert image.tobytes() == today.tobytes()
            flags = ("C_CONTIGUOUS", "OWNDATA", "WRITEABLE", "ALIGNED")
            assert [image.flags[f] for f in flags] == [today.flags[f] for f in flags]


ROTATIONS = {
    "rotate-90": (["-display_rotation", "90"], lambda frame: np.rot90(frame, 1)),
    "rotate-180": (["-display_rotation", "180"], lambda frame: np.rot90(frame, 2)),
    "rotate-270": (["-display_rotation", "270"], lambda frame: np.rot90(frame, -1)),
    "mirror": (["-display_hflip"], lambda frame: frame[:, ::-1]),
    "rotate-90-mirror": (["-display_rotation", "90", "-display_hflip"], None),
}


@pytest.mark.parametrize("case", sorted(ROTATIONS))
def test_rotated_files_are_upright_in_both_readers(tmp_path, monkeypatch, case):
    options, upright = ROTATIONS[case]
    source = (*_source("320x240"), "-c:v", "libx264", "-pix_fmt", "yuv444p")
    plain = _clip(tmp_path / "plain.mp4", *source)
    rotated = _clip(tmp_path / "rotated.mp4", *options, "-i", str(plain), "-c", "copy")
    in_process, frames = _read(rotated, monkeypatch, cli=False)
    piped, expected = _read(rotated, monkeypatch, cli=True)
    _, stored = _read(plain, monkeypatch, cli=False)
    assert in_process and not piped
    assert len(frames) == len(expected) == 30
    for frame, reference, original in zip(frames, expected, stored, strict=True):
        np.testing.assert_array_equal(frame, reference)
        if upright is not None:
            # 4:4:4 has no chroma to interpolate, so converting and turning commute.
            np.testing.assert_array_equal(frame, upright(original))


VERTICAL_FLIPS = {
    # Stored in the pixel format read, so the vflip filter passes the frames through
    # bottom-up (a negative row stride): 8-bit RGB, which read memory outside the
    # plane, and float RGB, which raised ValueError.
    "bgr24": (".mov", ("-c:v", "rawvideo", "-pix_fmt", "bgr24"), False),
    "gbrpf32le": (
        ".mkv",
        ("-c:v", "ffv1", "-level", "4", "-strict", "experimental"),
        True,
    ),
}


@pytest.mark.parametrize("case", sorted(VERTICAL_FLIPS))
def test_vertically_flipped_frames_use_the_cli_reader(tmp_path, monkeypatch, case):
    assert FFMPEG is not None and FFPROBE is not None
    suffix, codec, float_rgb = VERTICAL_FLIPS[case]
    source = (*_source("320x240"), *codec, "-pix_fmt", case)
    plain = _clip(tmp_path / f"plain{suffix}", *source)
    flipped = _clip(
        tmp_path / f"flipped{suffix}", "-display_vflip", "-i", str(plain), "-c", "copy"
    )
    reasons: list[str] = []
    monkeypatch.setattr(
        video.logger, "warning", lambda message, *args: reasons.append(message % args)
    )
    auto = (Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO, ChromaLocation.AUTO)
    settings = video.ReadSettings(VideoMode.RGB, float_rgb, auto)

    def frames(path: Path, cli: bool) -> list[np.ndarray]:
        if cli:
            monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
        else:
            monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
        loader = video.VideoLoader(
            path, FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE), settings
        )
        return [item[0] for item in loader.stream_items()]

    read = frames(flipped, cli=False)
    reason = "its display matrix flips the frames vertically"
    assert reasons == [f"Reading {flipped} through the FFmpeg CLI: {reason}"]
    expected = frames(flipped, cli=True)
    stored = frames(plain, cli=False)
    assert len(read) == len(expected) == len(stored) == 30
    for frame, reference, original in zip(read, expected, stored, strict=True):
        np.testing.assert_array_equal(frame, reference)
        np.testing.assert_array_equal(frame, original[::-1])


@pytest.mark.parametrize("suffix", [".mp4", ".mkv", ".webm"])
def test_variable_rate_files_use_the_cli_reader(tmp_path, monkeypatch, suffix):
    # Matroska and WebM report the same average and guessed rate for a variable-rate
    # track, where PyAV yielded the 30 stored frames and the CLI 45.
    codec = (
        ("libvpx-vp9", "-deadline", "realtime") if suffix == ".webm" else ("libx264",)
    )
    variable = _clip(
        tmp_path / f"variable{suffix}",
        *_source("640x360"),
        "-vf",
        "setpts='if(lt(N,10),N,N+15)/(30*TB)'",
        "-fps_mode",
        "vfr",
        "-c:v",
        *codec,
    )
    in_process, frames = _read(variable, monkeypatch, cli=False)
    _, expected = _read(variable, monkeypatch, cli=True)
    assert not in_process
    assert len(frames) == len(expected) > 30


def test_without_pyav_the_cli_reader_is_used(tmp_path, monkeypatch):
    path = _clip(tmp_path / "plain.mp4", *CASES["h264-420"])
    monkeypatch.setattr(video, "_pyav", lambda: None)
    in_process, frames = _read(path, monkeypatch, cli=False)
    assert not in_process and len(frames) == 30


@pytest.mark.parametrize("name", ["parts.ffconcat", "parts.txt"])
def test_a_concat_list_is_probed_again_when_its_part_changes(tmp_path, name):
    # ffprobe's answer for a concat list (by its extension or its header) follows the
    # files it names, which the list's own version does not show, so it is not kept.
    assert FFMPEG is not None and FFPROBE is not None
    env = FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE)
    listing = tmp_path / name
    listing.write_bytes(b"ffconcat version 1.0\nfile 'part.mkv'\nduration 1\n")
    sizes = []
    for size in ("320x240", "640x360"):
        _clip(tmp_path / "part.mkv", *_source(size), "-c:v", "ffv1")
        metadata = video.VideoMetadata.from_file(listing, env)
        sizes.append((metadata.width, metadata.height))
    assert sizes == [(320, 240), (640, 360)]


def test_the_filter_graphs_go_when_the_reader_ends(tmp_path, monkeypatch):
    # PyAV's Graph and its FilterContexts refer to each other and a successful run never
    # collects, so the reader frees its graphs itself: before, a 720p read left 31.7 MiB
    # (the graph and its filters' frame pools) until some later collection.
    path = _clip(tmp_path / "clip.mkv", *_source("320x240"), "-c:v", "ffv1")
    made: list[weakref.ref[object]] = []
    converter = video._converter

    def recording(*arguments):
        graph = converter(*arguments)
        made.append(weakref.ref(graph))
        return graph

    monkeypatch.setattr(video, "_converter", recording)
    gc.collect()
    gc.disable()
    try:
        in_process, frames = _read(path, monkeypatch, cli=False)
        assert in_process and len(frames) == 30
        assert made and [ref() for ref in made] == [None] * len(made)
    finally:
        gc.enable()


def test_a_matroska_file_is_scanned_once_per_version(tmp_path, monkeypatch):
    # The timestamp scan demuxes the whole file: a file read again is not scanned
    # again, and one that changed (here its modification time) is.
    av = video._pyav()
    assert av is not None
    path = _clip(tmp_path / "clip.mkv", *CASES["mkv-h264-bframes"])
    opened: list[str] = []
    open_container = av.open

    def counting_open(file: str, *args: object, **kwargs: object):
        opened.append(file)
        return open_container(file, *args, **kwargs)

    monkeypatch.setattr(av, "open", counting_open)
    rate = Fraction(30)
    assert not video._variable_timestamps(path, rate, av)
    assert not video._variable_timestamps(path, rate, av)
    assert opened == [str(path)]
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    assert not video._variable_timestamps(path, rate, av)
    assert opened == [str(path)] * 2
