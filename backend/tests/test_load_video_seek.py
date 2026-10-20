"""Load Video's exact reads: frame n read by a seek (stream_items(n, n + 1)) is item n of
the full read, bit for bit, in RGB float, RGB 8-bit and Planes and in both readers (PyAV
and the FFmpeg CLI); the Preview's open readers step through the same frames; the audio
is cut to a range. The node: a Range is the slice of a full run (Index, audio, context
frames, a one-frame run's chosen item), its errors, the Preview and the Frame Count.
Runs on chaiNNer's integrated FFmpeg."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import wave
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import ffmpeg
import numpy as np
import pytest

# The video module loads the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import NodeContext
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
from packages.chaiNNer_standard.image.video_frames.load_video import (
    FramesMode,
    RgbPrecision,
    load_video_node,
)

STORAGE = Path(os.environ.get("APPDATA", "")) / "chaiNNer" / "backend-storage"
FFMPEG, FFPROBE = map(str, get_executable_path(STORAGE / "ffmpeg"))
ENV = FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE)

pytestmark = pytest.mark.skipif(
    not Path(FFMPEG).exists(), reason="the integrated FFmpeg is not downloaded"
)
AUTO = (Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO, ChromaLocation.AUTO)
SOURCE = ("-f", "lavfi", "-i", "testsrc2=size=192x108:rate=30", "-frames:v", "150")
H264 = "h264-bframes.mp4"
UNTIMED = "untimed.mpg"
H264_OPEN_GOP = (
    *("-c:v", "libx264", "-bf", "3", "-x264-params"),
    "open-gop=1:keyint=24:min-keyint=24:scenecut=0",
)

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
    # H.264 open GOPs: the keyframes after the first are I-frames a decoder can start
    # at (recovery points), not IDR frames, so the B-frames shown before each refer to
    # the GOP before.
    "h264-open-gop.mp4": (None, (), H264_OPEN_GOP),
    "h264-open-gop.ts": (None, (), H264_OPEN_GOP),
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
    # MPEG-PS with big packets: a PES packet holds several frames and times only the
    # first, so most frames' packets have no timestamp to index them by. Read through
    # the FFmpeg CLI.
    UNTIMED: (
        None,
        (),
        (
            *("-c:v", "libx264", "-bf", "3", "-g", "60", "-keyint_min", "60"),
            *("-b:v", "50k", "-packetsize", "65000", "-f", "mpeg"),
        ),
    ),
}
# The files the FFmpeg CLI reads, and why.
CLI_FILES = {
    "variable.mkv": "its frame rate is variable",
    UNTIMED: "a frame's packet has no timestamp",
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
    cli = f"Reading {path} through the FFmpeg CLI"
    if name in CLI_FILES:
        assert f"{cli}: {CLI_FILES[name]}" in warnings
        # The CLI duplicates frames into a variable-rate file's gap: frame n is not the
        # n-th stored one.
        assert count > 150 if name == "variable.mkv" else count == 150
        # The CLI path counts by the probe's estimate.
        metadata: video.VideoMetadata = vars(loader)["metadata"]
        assert loader.exact_frame_count() is None
        assert loader.frame_count() == metadata.frame_count
        keyframes: tuple[int, ...] = ()
    else:
        assert not any(warning.startswith(cli) for warning in warnings)
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
    if name in CLI_FILES:
        # The CLI decodes and discards the frames before n: nothing decodes in-process.
        assert set(starts) == {0}
    elif name.endswith(".ts"):
        assert max(starts) > 1
    else:
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


def test_the_index_numbers_only_the_frames_it_can_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The index numbers the frames by their packets' timestamps. A frame's packet without
    # one (MPEG-PS times only some), or packets of which none is shown, would give a
    # short or empty index, so such a file is read through the FFmpeg CLI
    # (_UseCliReader, the reason its message). A discarded packet (never shown) needs no
    # timestamp; the empty packet that ends the demux (it flushes the decoder) is none.
    def packet(
        pts: int | None, keyframe: bool = False, discard: bool = False, size: int = 100
    ):
        return SimpleNamespace(
            pts=pts, is_keyframe=keyframe, is_discard=discard, size=size
        )

    class Container:
        def __init__(self, packets: list[SimpleNamespace]):
            self.packets = packets
            self.streams = SimpleNamespace(video=[object()])

        def __enter__(self):
            return self

        def __exit__(self, *_: object):
            return None

        def demux(self, _stream: object):
            return iter(self.packets)

    flush = packet(None, size=0)
    cases: list[tuple[list[SimpleNamespace], video.FrameIndex | str]] = [
        (
            [packet(None, discard=True), packet(3, True), packet(1), packet(2), flush],
            video.FrameIndex((1, 2, 3), (2,)),
        ),
        ([flush], video.FrameIndex((), ())),
        (
            [packet(0, True), packet(None), packet(2), flush],
            "a frame's packet has no timestamp",
        ),
        (
            [packet(0, True, discard=True), packet(1, discard=True), flush],
            "its packets time none of its frames",
        ),
    ]
    for number, (packets, expected) in enumerate(cases):
        path = tmp_path / f"{number}.mpg"
        path.write_bytes(b"")
        # The scan's cache takes PyAV's module as a key: a module, then.
        fake = ModuleType("av")
        vars(fake)["open"] = lambda _name, packets=packets: Container(packets)
        monkeypatch.setattr(video, "_pyav", lambda fake=fake: fake)
        if isinstance(expected, str):
            refusal = vars(video)["_UseCliReader"]
            with pytest.raises(refusal, match=f"^{expected}$"):
                video.frame_index(path)
        else:
            assert video.frame_index(path) == expected


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


def test_closing_readers_logs_every_failure_and_raises_the_first(
    monkeypatch: pytest.MonkeyPatch,
):
    # Every reader closes; each failure is logged with its traceback, and the first is
    # raised once all have closed.
    class Reader:
        def __init__(self, name: str, error: Exception | None = None):
            self.path = Path(name)
            self.error = error
            self.closed = False

        def close(self):
            self.closed = True
            if self.error is not None:
                raise self.error

    logged: list[tuple[str, object]] = []
    monkeypatch.setattr(
        video.logger,
        "warning",
        lambda message, *args, **kwargs: logged.append(
            (message % args, kwargs.get("exc_info"))
        ),
    )
    first, second = RuntimeError("first"), OSError("second")
    readers = [Reader("a.mp4", first), Reader("b.mp4"), Reader("c.mp4", second)]
    with pytest.raises(RuntimeError, match=r"^first$"):
        vars(video)["_close_readers"](readers)
    assert all(reader.closed for reader in readers)
    assert logged == [
        (f"Closing the Preview's reader of {Path('a.mp4')} failed", first),
        (f"Closing the Preview's reader of {Path('c.mp4')} failed", second),
    ]


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


class Context:
    """The NodeContext members Load Video uses."""

    node_id = "load-video"

    def __init__(self):
        self.storage_dir = STORAGE
        self.cleanups: list[Callable[[], None]] = []

    def add_cleanup(self, fn: Callable[[], None]):
        self.cleanups.append(fn)

    def cleanup(self):
        for cleanup in self.cleanups:
            cleanup()
        self.cleanups.clear()


def run_node(
    context: Context,
    path: Path,
    mode: VideoMode = VideoMode.RGB,
    precision: RgbPrecision = RgbPrecision.FLOAT,
    frames: tuple[int, int] | None = None,
    seek: int = 0,
) -> tuple[Any, ...]:
    """Load Video's outputs: the generator, directory, name, FPS, audio, Colour Info,
    Colour, Preview and Frame Count; Frames All, or the Range `frames` (In, Out)."""
    first, last = frames or (0, 0)
    return load_video_node(
        cast(NodeContext, context),
        path,
        seek,
        FramesMode.ALL if frames is None else FramesMode.RANGE,
        first,
        last,
        mode,
        precision,
        *AUTO,
    )


def items_of(iterable: Any) -> list[tuple[Any, ...]]:
    items: list[tuple[Any, ...]] = []
    for item in iterable:
        assert isinstance(item, tuple) and len(item) == 5
        items.append(item)
    return items


def assert_node_items_equal(
    items: list[tuple[Any, ...]], expected: list[tuple[Any, ...]]
):
    """Load Video's items (frame, Index, Y, U, V) are `expected`'s, Index included."""
    assert len(items) == len(expected)
    for (frame, index, y, u, v), (reference, number, *planes) in zip(
        items, expected, strict=True
    ):
        assert index == number
        assert_items_equal((frame, y, u, v), (reference, *planes))


NODE_MODES = [
    (VideoMode.RGB, RgbPrecision.FLOAT),
    (VideoMode.RGB, RgbPrecision.EIGHT_BIT),
    (VideoMode.PLANES, RgbPrecision.FLOAT),
    (VideoMode.BOTH, RgbPrecision.FLOAT),
]


@pytest.mark.usefixtures("previews")
@pytest.mark.parametrize(("mode", "precision"), NODE_MODES)
def test_a_range_is_the_slice_of_a_full_run(
    folder: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: VideoMode,
    precision: RgbPrecision,
):
    monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    path = made(folder, H264)
    context = Context()
    whole = run_node(context, path, mode, precision)
    full = items_of(whole[0].supplier())
    # All, as before: every frame, numbered from 0, and the whole audio. Its length is
    # the Frame Count, exact.
    assert whole[8] == whole[0].expected_length == len(full) == 150
    assert (whole[0].index_base, whole[0].chosen_item) == (0, None)
    assert [item[1] for item in full] == list(range(150))
    assert whole[4].node.kwargs == {"filename": path}
    generator, _, _, fps, audio, *_ = run_node(context, path, mode, precision, (8, 11))
    assert (generator.expected_length, generator.index_base) == (4, 8)
    assert generator.chosen_item is None
    assert_node_items_equal(items_of(generator.supplier()), full[8:12])
    # The audio of frames 8..11: from 8 / fps for 4 / fps.
    assert fps == 30
    assert audio.node.kwargs == {"filename": path, "ss": 8 / fps, "t": 4 / fps}
    # Context frames: 3 before and 2 after the Range, Index counting through them.
    assert generator.context is not None
    around, before, after = generator.context(3, 2)
    assert (before, after) == (3, 2)
    assert_node_items_equal(items_of(around), full[5:14])
    context.cleanup()


@pytest.mark.usefixtures("previews")
def test_context_frames_are_clipped_to_the_video_and_closed_by_the_chain(
    folder: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    path = folder / "context.mp4"
    shutil.copyfile(made(folder, H264), path)
    context = Context()
    full = items_of(run_node(context, path)[0].supplier())
    for frames, wanted, clipped, (start, stop) in [
        ((1, 4), (3, 2), (1, 2), (0, 7)),
        ((145, 148), (3, 5), (3, 1), (142, 150)),
        ((0, 149), (3, 5), (0, 0), (0, 150)),
        # Out past the end: none after.
        ((145, 999), (3, 5), (3, 0), (142, 150)),
    ]:
        generator = run_node(context, path, frames=frames)[0]
        assert generator.context is not None
        around, before, after = generator.context(*wanted)
        assert (before, after) == clipped
        assert_node_items_equal(items_of(around), full[start:stop])
    # The loop may stop before the context frames end: the chain's cleanup closes them.
    # The Preview's reader stays open for the next step, until shutdown (close_previews).
    around, _, _ = run_node(context, path, frames=(60, 70))[0].context(2, 2)
    next(iter(around))
    context.cleanup()
    assert locked(path)
    video.close_previews()
    assert not locked(path)


@pytest.mark.usefixtures("previews")
def test_a_one_frame_range_is_the_chosen_item(
    folder: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    path = made(folder, H264)
    context = Context()
    full = rgb_float_frames(path)
    for seek in (8, 3):
        outputs = run_node(context, path, frames=(8, 8), seek=seek)
        generator, preview = outputs[0], outputs[7]
        assert (generator.expected_length, generator.chosen_item) == (1, 8)
        ((frame, index, *_),) = items_of(generator.supplier())
        assert index == 8
        np.testing.assert_array_equal(frame, full[8])
        # With Seek on the frame, the run's frame is the Preview's: decoded once.
        assert (frame is preview) == (seek == 8)
    # Read another way (Planes), the frame is decoded for the run.
    generator = run_node(context, path, VideoMode.PLANES, frames=(8, 8), seek=8)[0]
    ((frame, index, y, _, _),) = items_of(generator.supplier())
    assert frame is None and y is not None and index == 8
    context.cleanup()


@pytest.mark.usefixtures("previews")
@pytest.mark.parametrize("cli", [False, True])
def test_out_is_an_upper_bound_and_in_past_the_end_fails(
    folder: Path, monkeypatch: pytest.MonkeyPatch, cli: bool
):
    if cli:
        monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    else:
        monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    path = made(folder, H264)
    context = Context()
    full = rgb_float_frames(path)
    # A migrated "Use limit" chain (Range 0..999) on this 150-frame clip reads every
    # frame, as the limit did; so does any Out past the end. The audio is cut to the
    # frames read where the count is exact (PyAV), else at Out.
    for first, last in ((0, 999), (140, 150)):
        generator, _, _, fps, audio, *_ = run_node(context, path, frames=(first, last))
        assert generator.expected_length == 150 - first
        read = items_of(generator.supplier())
        assert [item[1] for item in read] == list(range(first, 150))
        np.testing.assert_array_equal(read[-1][0], full[149])
        cut = {"t": (last - first + 1 if cli else 150 - first) / fps}
        if first > 0:
            cut = {"ss": first / fps, **cut}
        assert audio.node.kwargs == {"filename": path, **cut}
    for frames, message in [
        ((12, 8), r"^In \(frame 12\) is after Out \(frame 8\)\.$"),
        ((150, 155), r"^frame 150 is past the end \(the video has 150 frames\)$"),
        # A one-frame Range past the end: the In error.
        ((150, 150), r"^frame 150 is past the end \(the video has 150 frames\)$"),
    ]:
        outputs = run_node(context, path, frames=frames, seek=500)
        # The node still reports its Preview (of the last frame, Seek being past it) and
        # its Frame Count, by which the seek bar sets the Range again.
        assert outputs[8] == 150
        np.testing.assert_array_equal(outputs[7], full[149])
        with pytest.raises(ValueError, match=message):
            next(iter(outputs[0].supplier()))
    if cli:
        # The CLI path counts by the probe's estimate: an Out past a low one still reads
        # to the stream's end, the length being the estimate's.
        init = video.VideoLoader.__init__

        def low_estimate(loader: video.VideoLoader, *arguments: Any):
            init(loader, *arguments)
            # metadata is set by video_loader_init (native/src/video_io.cpp).
            attributes = vars(loader)
            attributes["metadata"] = replace(attributes["metadata"], frame_count=100)

        monkeypatch.setattr(video.VideoLoader, "__init__", low_estimate)
        for last, end in ((120, 121), (999, 150)):
            outputs = run_node(context, path, frames=(95, last))
            assert (outputs[8], outputs[0].expected_length) == (100, 5)
            read = items_of(outputs[0].supplier())
            assert [item[1] for item in read] == list(range(95, end))
            np.testing.assert_array_equal(read[-1][0], full[end - 1])
    context.cleanup()


@pytest.mark.usefixtures("previews")
@pytest.mark.parametrize(("mode", "precision"), NODE_MODES)
def test_preview_is_the_rgb_float_frame_at_seek(
    folder: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: VideoMode,
    precision: RgbPrecision,
):
    monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    context = Context()
    for name in (H264, "edit-list.mp4"):
        path = made(folder, name)
        full = rgb_float_frames(path)
        for seek, shown in ((37, 37), (36, 36), (len(full) + 40, len(full) - 1)):
            outputs = run_node(context, path, mode, precision, seek=seek)
            assert outputs[8] == len(full)
            assert outputs[7].dtype == np.float32
            np.testing.assert_array_equal(outputs[7], full[shown])
    context.cleanup()


@pytest.mark.usefixtures("previews")
def test_preview_of_planes_rgb_cannot_convert_is_the_y_plane(
    folder: Path, monkeypatch: pytest.MonkeyPatch
):
    # BT.2020 constant luminance: RGB mode refuses it, Planes mode reads it and its
    # Preview shows the Y plane in gray, with a warning; Frame Count and FPS still come.
    monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    path = folder / "constant-luminance.mkv"
    run_ffmpeg(
        *SOURCE[:-2],
        *("-frames:v", "12", "-c:v", "ffv1", "-pix_fmt", "yuv420p10le"),
        *("-colorspace", "bt2020c", str(path)),
    )
    warnings: list[str] = []
    monkeypatch.setattr(
        video.logger, "warning", lambda message, *args: warnings.append(message % args)
    )
    context = Context()
    outputs = run_node(context, path, VideoMode.PLANES, seek=5)
    full = items_of(outputs[0].supplier())
    assert (outputs[3], outputs[8]) == (30, 12)
    y = full[5][2]
    np.testing.assert_array_equal(outputs[7], np.dstack((y, y, y)))
    assert any(
        warning.startswith(f"{path}: the Preview shows the Y plane in gray:")
        and "cannot convert the BT.2020 constant matrix" in warning
        for warning in warnings
    )
    context.cleanup()


@pytest.mark.usefixtures("previews")
def test_a_file_without_timestamps_reads_through_the_cli(
    folder: Path, monkeypatch: pytest.MonkeyPatch
):
    # Most frames of this MPEG-PS file have no timestamp to index them by: the Preview
    # and a Range read it through the FFmpeg CLI, which decodes and discards the frames
    # before the one asked for, as the full read does.
    monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    path = made(folder, UNTIMED)
    full = rgb_float_frames(path)
    warnings: list[str] = []
    monkeypatch.setattr(
        video.logger, "warning", lambda message, *args: warnings.append(message % args)
    )
    loader = video.VideoLoader(path, ENV, SETTINGS["rgb-float"])
    for n in (40, 41, 10):
        np.testing.assert_array_equal(
            video.preview_frame("untimed", loader, n), full[n]
        )
    reason = CLI_FILES[UNTIMED]
    assert f"Previewing {path} through the FFmpeg CLI: {reason}" in warnings
    context = Context()
    outputs = run_node(context, path, frames=(40, 45))
    read = items_of(outputs[0].supplier())
    assert [item[1] for item in read] == list(range(40, 46))
    for (frame, *_), expected in zip(read, full[40:46], strict=True):
        np.testing.assert_array_equal(frame, expected)
    # The Frame Count is the probe's estimate (short here: ffprobe times the file by
    # its few timestamps).
    metadata: video.VideoMetadata = vars(loader)["metadata"]
    assert outputs[8] == metadata.frame_count < 150
    context.cleanup()
