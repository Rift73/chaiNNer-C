"""Load Video's exact reads: frame n read by a seek (stream_items(n, n + 1)) is item n of
the full read, bit for bit, in RGB float, RGB 8-bit and Planes and in both readers (PyAV
and the FFmpeg CLI); the Preview's open readers step through the same frames; the audio
is cut to a range. Runs on chaiNNer's integrated FFmpeg."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ffmpeg
import numpy as np
import pytest

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

STORAGE = Path(os.environ.get("APPDATA", "")) / "chaiNNer" / "backend-storage"
FFMPEG, FFPROBE = map(str, get_executable_path(STORAGE / "ffmpeg"))
ENV = FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE)

pytestmark = pytest.mark.skipif(
    not Path(FFMPEG).exists(), reason="the integrated FFmpeg is not downloaded"
)
AUTO = (Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO, ChromaLocation.AUTO)
SOURCE = ("-f", "lavfi", "-i", "testsrc2=size=192x108:rate=30", "-frames:v", "150")
H264 = "h264-bframes.mp4"

# The files of the exactness matrix: what each is made from (another file of the matrix,
# or None for SOURCE), FFmpeg's input options for it and its output options.
FILES: dict[str, tuple[str | None, tuple[str, ...], tuple[str, ...]]] = {
    H264: (
        None,
        (),
        ("-c:v", "libx264", "-bf", "3", "-g", "60", "-keyint_min", "60"),
    ),
    "hevc-10-bit.mkv": (
        None,
        (),
        (
            *("-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params"),
            "keyint=60:min-keyint=60:bframes=4:open-gop=1:log-level=error",
        ),
    ),
    # Cut without re-encoding: an edit list hides the frames before the cut, which
    # the demuxer flags as discarded.
    "edit-list.mp4": (H264, ("-ss", "1.3"), ("-c", "copy")),
    # An open GOP's leading frames refer to the GOP before.
    "hevc-open-gop.mp4": (
        None,
        (),
        (
            *("-c:v", "libx265", "-pix_fmt", "yuv420p", "-x265-params"),
            "keyint=48:min-keyint=1:bframes=6:open-gop=1:scenecut=40:log-level=error",
        ),
    ),
    # MPEG-TS seeks by decoding timestamps and lands past the keyframe, so a seek starts
    # again from the keyframe before, and the first GOP decodes from the file's start.
    "h264.ts": (
        None,
        (),
        ("-c:v", "libx264", "-bf", "3", "-g", "60", "-keyint_min", "60"),
    ),
    "hevc-open-gop.ts": (
        None,
        (),
        (
            *("-c:v", "libx265", "-pix_fmt", "yuv420p", "-x265-params"),
            "keyint=48:min-keyint=1:bframes=6:open-gop=1:scenecut=40:log-level=error",
        ),
    ),
    # A display matrix (90 degrees); the integrated FFmpeg 5.1 has no -display_rotation.
    "rotated.mp4": (H264, (), ("-c", "copy", "-metadata:s:v:0", "rotate=90")),
    # Read through the FFmpeg CLI, which renumbers the frames to a constant rate.
    "variable.mkv": (
        None,
        (),
        (
            *("-vf", "setpts='if(lt(N,20),N,N+15)/(30*TB)'", "-fps_mode", "vfr"),
            *("-c:v", "libx264", "-g", "30"),
        ),
    ),
}
SETTINGS = {
    "rgb-float": video.ReadSettings(VideoMode.RGB, True, AUTO),
    "rgb-8-bit": video.ReadSettings(VideoMode.RGB, False, AUTO),
    "planes": video.ReadSettings(VideoMode.PLANES, True, AUTO),
}


def run_ffmpeg(*arguments: str) -> None:
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", *arguments],
        check=True,
        stdin=subprocess.DEVNULL,
    )


@pytest.fixture(scope="module")
def folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("seek")


def made(folder: Path, name: str) -> Path:
    """The matrix's file `name` in `folder`, made on first use."""
    path = folder / name
    if not path.exists():
        source, input_options, options = FILES[name]
        inputs = SOURCE if source is None else ("-i", str(made(folder, source)))
        run_ffmpeg(*input_options, *inputs, *options, str(path))
    return path


def assert_items_equal(item: tuple[object, ...], expected: tuple[object, ...]):
    assert len(item) == len(expected) == 4
    for value, reference in zip(item, expected, strict=True):
        if reference is None:
            assert value is None
            continue
        assert isinstance(value, np.ndarray) and isinstance(reference, np.ndarray)
        assert (value.dtype, value.shape) == (reference.dtype, reference.shape)
        np.testing.assert_array_equal(value, reference)


@pytest.mark.parametrize("mode", sorted(SETTINGS))
@pytest.mark.parametrize("name", sorted(FILES))
def test_seek_equals_full_read(
    folder: Path, monkeypatch: pytest.MonkeyPatch, name: str, mode: str
):
    monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    path = made(folder, name)
    warnings: list[str] = []
    monkeypatch.setattr(
        video.logger, "warning", lambda message, *args: warnings.append(message % args)
    )
    loader = video.VideoLoader(path, ENV, SETTINGS[mode])
    full = list(loader.stream_items())
    count = len(full)
    cli = f"Reading {path} through the FFmpeg CLI: its frame rate is variable"
    if name == "variable.mkv":
        # The CLI duplicates frames into the gap: frame n is not the n-th stored one.
        assert cli in warnings and count > 150
        keyframes: tuple[int, ...] = ()
    else:
        assert cli not in warnings
        index = video.frame_index(path)
        assert loader.frame_count() == len(index.pts) == count
        keyframes = index.keyframes
        assert len(keyframes) > 1
    if name == "rotated.mp4" and mode != "planes":
        # Upright: the stored 192x108 frames turned a quarter.
        assert full[0][0] is not None and full[0][0].shape[:2] == (192, 108)
    sample = {0, 1, count // 2, count - 1}
    sample |= {k + d for k in keyframes for d in range(-6, 2) if 0 <= k + d < count}
    # How many times each seek started decoding (a keyframe, or the file's start).
    starts: list[int] = []
    decode = vars(video)["_decode"]

    def counting(*arguments: Any):
        starts[-1] += 1
        return decode(*arguments)

    monkeypatch.setattr(video, "_decode", counting)
    for n in sorted(sample):
        starts.append(0)
        items = list(loader.stream_items(n, n + 1))
        assert len(items) == 1, n
        assert_items_equal(items[0], full[n])
    if name.endswith(".ts"):
        assert max(starts) > 1
    elif name != "variable.mkv":
        assert set(starts) == {1}
    # A range, a range to the end and one past it (it stops at the end, as a slice).
    for start, stop in ((count // 3, count // 3 + 7), (count - 3, None)):
        items = list(loader.stream_items(start, stop))
        assert len(items) == len(full[start:stop])
        for item, expected in zip(items, full[start:stop], strict=True):
            assert_items_equal(item, expected)
    assert len(list(loader.stream_items(count - 2, count + 5))) == 2
    with pytest.raises(
        ValueError,
        match=rf"^frame {count} is past the end \(the video has {count} frames\)$",
    ):
        list(loader.stream_items(count, count + 1))


def locked(path: Path) -> bool:
    """Whether a reader holds `path` open: Windows renames no file that is open."""
    moved = path.with_name(path.name + ".moved")
    try:
        os.replace(path, moved)
    except PermissionError:
        return True
    os.replace(moved, path)
    return False


def rgb_float_frames(path: Path, choices: tuple[Any, ...] = AUTO) -> list[np.ndarray]:
    settings = video.ReadSettings(VideoMode.RGB, True, choices)
    frames: list[np.ndarray] = []
    for rgb, *_ in video.VideoLoader(path, ENV, settings).stream_items():
        assert rgb is not None
        frames.append(rgb)
    return frames


@pytest.fixture
def previews() -> Iterator[None]:
    """Every preview reader closed after the test, as at a chain's end."""
    yield
    video.close_previews()


@pytest.mark.usefixtures("previews")
@pytest.mark.parametrize("cli", [False, True])
def test_preview_reader_steps(folder: Path, monkeypatch: pytest.MonkeyPatch, cli: bool):
    # The steps' frames equal the RGB float full read's, in every Mode; PyAV decodes a
    # frame in the same or the next GOP forward and seeks to an earlier or farther one.
    path = folder / "steps.mp4"
    shutil.copyfile(made(folder, H264), path)
    full = rgb_float_frames(path)
    if cli:
        monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    else:
        monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    seeks: list[int] = []
    seek = vars(video)["_seek"]

    def counting(*arguments: Any):
        seeks.append(arguments[-1])
        return seek(*arguments)

    monkeypatch.setattr(video, "_seek", counting)
    for mode, settings in SETTINGS.items():
        loader = video.VideoLoader(path, ENV, settings)
        seeks.clear()
        for n in (30, 31, 33, 29, 140, 0, 61, 149):
            frame = video.preview_frame(mode, loader, n)
            assert frame.dtype == np.float32
            np.testing.assert_array_equal(frame, full[n])
        assert seeks == ([] if cli else [30, 29, 140, 0])
        assert locked(path)
    with pytest.raises(ValueError, match=r"^frame 150 is past the end"):
        video.preview_frame("planes", loader, 150)
    with pytest.raises(ValueError, match=r"^frame -1 is before the start"):
        video.preview_frame("planes", loader, -1)
    video.close_previews()
    assert not locked(path) and not vars(video)["_previews"]


@pytest.mark.usefixtures("previews")
def test_preview_reader_follows_the_file_and_settings(
    folder: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    first, second = folder / "first.mp4", folder / "second.mkv"
    shutil.copyfile(made(folder, H264), first)
    shutil.copyfile(made(folder, "hevc-10-bit.mkv"), second)
    settings = SETTINGS["rgb-float"]
    frame = video.preview_frame("node", video.VideoLoader(first, ENV, settings), 5)
    np.testing.assert_array_equal(frame, rgb_float_frames(first)[5])
    # Another file: the node's reader opens it, and lets the first go.
    frame = video.preview_frame("node", video.VideoLoader(second, ENV, settings), 5)
    np.testing.assert_array_equal(frame, rgb_float_frames(second)[5])
    assert not locked(first) and locked(second)
    # Other colour choices: the reader converts by them.
    full_range = (*AUTO[:3], Range.FULL, AUTO[4])
    chosen = video.ReadSettings(VideoMode.PLANES, True, full_range)
    frame = video.preview_frame("node", video.VideoLoader(second, ENV, chosen), 6)
    expected = rgb_float_frames(second, full_range)[6]
    np.testing.assert_array_equal(frame, expected)
    assert not np.array_equal(expected, rgb_float_frames(second)[6])
    assert len(vars(video)["_previews"]) == 1
    video.close_previews()
    assert not locked(second)


@pytest.mark.usefixtures("previews")
def test_preview_readers_are_few_and_close_when_idle(
    folder: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    paths = [folder / f"node-{n}.mp4" for n in range(5)]
    for path in paths:
        shutil.copyfile(made(folder, H264), path)
    settings = SETTINGS["rgb-float"]
    for n, path in enumerate(paths):
        video.preview_frame(f"node-{n}", video.VideoLoader(path, ENV, settings), 3)
    # At most four nodes keep a reader: the least recently used closes.
    assert list(vars(video)["_previews"]) == [f"node-{n}" for n in range(1, 5)]
    assert [locked(path) for path in paths] == [False, True, True, True, True]
    # A reader idle for PREVIEW_IDLE_SECONDS closes, and so do the others in turn.
    monkeypatch.setattr(video, "PREVIEW_IDLE_SECONDS", 0.2)
    video.close_previews()
    video.preview_frame("idle", video.VideoLoader(paths[0], ENV, settings), 3)
    assert locked(paths[0])
    time.sleep(0.1)
    video.preview_frame("later", video.VideoLoader(paths[1], ENV, settings), 3)
    deadline = time.monotonic() + 5
    while vars(video)["_previews"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not vars(video)["_previews"]
    assert not locked(paths[0]) and not locked(paths[1])


def test_audio_range(folder: Path):
    # Save Video's mux reads the audio from the stream's input (its "filename"): from
    # `start` (-ss, only past 0) for `duration` (-t). A ramp's samples tell the time.
    path = folder / "ramp.mkv"
    run_ffmpeg(
        *("-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10:duration=4"),
        *("-f", "lavfi", "-i", "aevalsrc=t/4:s=8000:d=4"),
        *("-c:v", "ffv1", "-c:a", "pcm_s16le", str(path)),
    )
    loader = video.VideoLoader(path, ENV, SETTINGS["rgb-float"])
    assert loader.get_audio_stream().node.kwargs == {"filename": path}
    assert loader.get_audio_stream(1.5).node.kwargs == {"filename": path, "t": 1.5}
    assert loader.get_audio_stream(1.5, 0.0).node.kwargs == {
        "filename": path,
        "t": 1.5,
    }
    assert loader.get_audio_stream(None, 2.0).node.kwargs == {
        "filename": path,
        "ss": 2.0,
    }
    cut = loader.get_audio_stream(1.5, 2.0)
    assert cut.node.kwargs == {"filename": path, "ss": 2.0, "t": 1.5}
    target = folder / "cut.wav"
    ffmpeg.output(cut, str(target)).overwrite_output().run(cmd=FFMPEG, quiet=True)
    with wave.open(str(target)) as cut_file:
        samples = np.frombuffer(cut_file.readframes(cut_file.getnframes()), "<i2")
    seconds = (2.0 + np.arange(samples.size) / 8000) / 4
    assert samples.size == 1.5 * 8000
    np.testing.assert_allclose(samples / 32768, seconds, atol=1e-3)
