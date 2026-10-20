"""Save Video in a run of one chosen frame (Load Video's "Run this frame", or In = Out;
owner-approved 2026-10-21) saves that frame as `<video name>_f000123.png` next to where
the video would go: in 16 bits when the video would have more than 8, Planes converted
to RGB by their coding, the audio left out. FFmpeg never starts and the video's file is
never created or touched."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest

# The node's modules load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import NodeContext
from nodes.impl.colour_info import (
    ChromaLocation,
    ColourInfo,
    Matrix,
    Primaries,
    Range,
    Transfer,
)
from nodes.impl.image_utils import to_uint8, to_uint16
from nodes.impl.plane_resample import Kernel
from nodes.impl.yuv_rgb import guess_colour_info, planes_to_rgb, resolve_colour_info
from packages.chaiNNer_standard.image.video_frames import save_video as module
from packages.chaiNNer_standard.image.video_frames.save_video import (
    AudioSettings,
    BitDepth,
    FrameCollector,
    SimpleVideoFormat,
    Simplicity,
    SitingCorrection,
    VideoEncoder,
    VideoFormat,
    VideoInputMode,
    VideoPreset,
    prepare_video_frame,
    save_video_node,
)

ITEM = 123
AUTO_CHOICES = (
    Matrix.AUTO,
    Transfer.AUTO,
    Primaries.AUTO,
    Range.AUTO,
    ChromaLocation.AUTO,
)
OLD = 1_000_000_000  # An mtime (2001) no write of this test can produce.


class Context:
    """The NodeContext members Save Video uses, in a run of the one chosen item."""

    chosen_item = ITEM

    def __init__(self):
        self.storage_dir = Path("no-storage")
        self.cleanups: list[Callable[[], None]] = []

    def add_cleanup(self, fn: Callable[[], None], after: str = "chain"):
        self.cleanups.append(fn)


class Forbidden:
    """Stands in for what writes or probes a video: any use fails the test."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args: object, **kwargs: object):
        raise AssertionError(f"a one-frame run used {self.name}")

    def __getattr__(self, attribute: str):
        raise AssertionError(f"a one-frame run used {self.name}.{attribute}")


class GuardedGraph:
    """The native graph without its video writer."""

    def __init__(self, graph: object):
        self.graph = graph

    def __getattr__(self, name: str):
        if name in ("video_save", "video_writer_start", "video_writer_frame"):
            return Forbidden(name)
        return getattr(self.graph, name)


class Logger:
    def __init__(self):
        self.records: list[tuple[str, str]] = []

    def __getattr__(self, level: str):
        return lambda message, *args, **_: self.records.append(
            (level, str(message) % args)
        )


@pytest.fixture(autouse=True)
def no_video_writer(monkeypatch: pytest.MonkeyPatch):
    """FFmpeg, the video's temporary file and the integrated FFmpeg's set-up fail."""
    graph = GuardedGraph(module.graph())
    monkeypatch.setattr(module, "graph", lambda: graph)
    monkeypatch.setattr(module, "ffmpeg", Forbidden("ffmpeg"))
    monkeypatch.setattr(module, "atomic_target", Forbidden("atomic_target"))
    monkeypatch.setattr(
        module.FFMpegEnv, "get_integrated", Forbidden("FFMpegEnv.get_integrated")
    )
    monkeypatch.setattr(subprocess, "Popen", Forbidden("subprocess.Popen"))


def collector(
    folder: Path,
    *,
    name: str = "out",
    mode: VideoInputMode = VideoInputMode.RGB,
    simplicity: Simplicity = Simplicity.ADVANCED,
    container: VideoFormat = VideoFormat.MKV,
    encoder: VideoEncoder = VideoEncoder.FFV1,
    additional: str | None = None,
    simple: SimpleVideoFormat = SimpleVideoFormat.MP4_H264,
    audio: object | None = None,
    colour: ColourInfo | None = None,
    bit_depth: BitDepth = BitDepth.AUTO,
) -> FrameCollector:
    result = save_video_node(
        cast(NodeContext, Context()),
        None,
        folder,
        name,
        mode,
        None,
        None,
        None,
        simplicity,
        container,
        encoder,
        VideoPreset.ULTRA_FAST,
        23,
        additional,
        simple,
        75,
        30,
        audio,
        AudioSettings.AUTO,
        colour,
        *AUTO_CHOICES,
        bit_depth,
        SitingCorrection.AUTO,
    )
    assert isinstance(result, FrameCollector)
    return result


def save(folder: Path, frame: tuple[np.ndarray | None, ...], **options) -> Path:
    """Save Video's one frame (Image Sequence, Y, U, V); the PNG's path."""
    c = collector(folder, **options)
    c.on_iterate(frame)
    c.on_complete()
    return folder / f"{options.get('name', 'out')}_f{ITEM:06d}.png"


def image(width: int = 6, height: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((height, width, 3), np.float32)


def read(path: Path) -> np.ndarray:
    png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert png is not None, path
    return png


def colour(
    bit_depth: int, size: tuple[int, int] = (6, 4), chroma: tuple[int, int] = (3, 2)
) -> ColourInfo:
    return replace(guess_colour_info(size, chroma), bit_depth=bit_depth)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"simplicity": Simplicity.SIMPLE},
        # H.264's Auto writes 8 bits of a 10-bit source.
        {"encoder": VideoEncoder.H264, "colour": colour(10)},
        # GIF's palette.
        {"container": VideoFormat.GIF, "bit_depth": BitDepth.BITS_10},
        # A user's 8-bit -pix_fmt wins over Bit Depth (the writer's colour options).
        {"bit_depth": BitDepth.BITS_10, "additional": "-pix_fmt yuv444p"},
    ],
)
def test_an_8_bit_video_gives_an_8_bit_png(tmp_path: Path, options: dict[str, Any]):
    img = image()
    png = read(save(tmp_path, (img, None, None, None), **options))
    assert png.dtype == np.uint8
    assert np.array_equal(png, to_uint8(img, normalized=True))


@pytest.mark.parametrize(
    "options",
    [
        {"encoder": VideoEncoder.H265, "bit_depth": BitDepth.BITS_10},
        {"bit_depth": BitDepth.BITS_16},
        # FFV1's Auto keeps the source's 10 bits; H.265's caps 12 at 10.
        {"colour": colour(10)},
        {"encoder": VideoEncoder.H265, "colour": colour(12)},
        # The writer's colour options keep a user's -pix_fmt.
        {"encoder": VideoEncoder.H265, "additional": "-pix_fmt yuv420p10le"},
    ],
)
def test_a_deeper_video_gives_a_16_bit_png(tmp_path: Path, options: dict[str, Any]):
    img = image()
    png = read(save(tmp_path, (img, None, None, None), **options))
    assert png.dtype == np.uint16
    assert np.array_equal(png, to_uint16(img, normalized=True))


@pytest.mark.parametrize(
    ("info", "bit_depth", "convert"),
    [
        (None, BitDepth.AUTO, to_uint8),
        (colour(10, (8, 6), (4, 3)), BitDepth.AUTO, to_uint16),
        (colour(8, (8, 6), (4, 3)), BitDepth.BITS_12, to_uint16),
    ],
)
def test_planes_are_converted_to_rgb_by_their_coding(
    tmp_path: Path,
    info: ColourInfo | None,
    bit_depth: BitDepth,
    convert: Callable[..., np.ndarray],
):
    rng = np.random.default_rng(1)
    y = rng.random((6, 8), np.float32)
    u, v = (rng.random((3, 4), np.float32) for _ in range(2))
    path = save(
        tmp_path,
        (None, y, u, v),
        mode=VideoInputMode.PLANES,
        colour=info,
        bit_depth=bit_depth,
    )
    coding = resolve_colour_info(info, (8, 6), (4, 3), *AUTO_CHOICES)
    rgb = planes_to_rgb(y, u, v, coding.matrix, coding.chroma_location, Kernel.SPLINE36)
    # Random chroma leaves the RGB cube: the PNG clips it.
    assert rgb.min() < 0 or rgb.max() > 1
    assert np.array_equal(read(path), convert(np.clip(rgb, 0, 1), normalized=True))


def test_no_video_file_and_no_window_payload(tmp_path: Path):
    c = collector(tmp_path)
    # The item window's prepare leaves the frame to on_iterate (no FFmpeg payload).
    assert prepare_video_frame(c, [image(), None, None, None]) is None
    c.on_iterate((image(), None, None, None))
    c.on_complete()
    assert [p.name for p in tmp_path.iterdir()] == [f"out_f{ITEM:06d}.png"]


def test_an_existing_video_is_untouched(tmp_path: Path):
    video = tmp_path / "out.mkv"
    video.write_bytes(b"an existing video")
    os.utime(video, (OLD, OLD))
    # Held open without delete sharing, as by a player: the video's rename check would fail.
    with video.open("rb"):
        save(tmp_path, (image(), None, None, None))
    assert video.read_bytes() == b"an existing video"
    assert video.stat().st_mtime == OLD
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "out.mkv",
        f"out_f{ITEM:06d}.png",
    ]


def test_the_audio_is_left_out_with_a_log_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    logger = Logger()
    monkeypatch.setattr(module, "logger", logger)
    img = image()
    path = save(tmp_path, (img, None, None, None), audio=Forbidden("the audio"))
    assert np.array_equal(read(path), to_uint8(img, normalized=True))
    assert logger.records == [
        (
            "info",
            f"Save Video saves the one chosen frame {ITEM} as a PNG, without the audio.",
        )
    ]


def test_the_png_goes_where_the_video_would(tmp_path: Path):
    # Video Name's folders and a missing Directory are created, as for the video.
    path = save(tmp_path / "new", (image(), None, None, None), name="clips/a")
    assert path == tmp_path / "new" / "clips" / f"a_f{ITEM:06d}.png"
    assert path.is_file()


def test_a_second_frame_is_an_error(tmp_path: Path):
    first, second = image(seed=2), image(seed=3)
    c = collector(tmp_path)
    c.on_iterate((first, None, None, None))
    with pytest.raises(RuntimeError, match="second frame in a run of the one chosen"):
        c.on_iterate((second, None, None, None))
    png = read(tmp_path / f"out_f{ITEM:06d}.png")
    assert np.array_equal(png, to_uint8(first, normalized=True))
