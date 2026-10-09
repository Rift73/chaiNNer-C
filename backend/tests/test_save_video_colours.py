"""Save Video tags its YUV output with all four colour tags and converts to match
them: BT.601 for SD, BT.709 for HD (width >= 1280 or height > 576). Load Video reads
untagged HD video as BT.709 by the same rule, in both of its readers (upstream
chaiNNer #3053). Runs on chaiNNer's integrated FFmpeg."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest

# The node's modules load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import NodeContext
from nodes.impl.ffmpeg import FFMpegEnv, get_executable_path
from nodes.impl.video import VideoLoader
from packages.chaiNNer_standard.image.video_frames.save_video import (
    AudioSettings,
    SimpleVideoFormat,
    Simplicity,
    VideoEncoder,
    VideoFormat,
    VideoPreset,
    save_video_node,
)

# The FFmpeg the backend downloads to its storage on first use (FFMpegEnv.get_integrated).
STORAGE = Path(os.environ.get("APPDATA", "")) / "chaiNNer" / "backend-storage"
FFMPEG, FFPROBE = map(str, get_executable_path(STORAGE / "ffmpeg"))

pytestmark = pytest.mark.skipif(
    not Path(FFMPEG).exists(), reason="the integrated FFmpeg is not downloaded"
)

# Flat 8-bit RGB bars: primaries, secondaries, skin, grass, greys.
BARS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (224, 172, 105),
    (86, 125, 70),
    (128, 128, 128),
    (235, 235, 235),
    (16, 16, 16),
]
BT709 = ("bt709", "bt709", "bt709", "tv")
BT601 = ("smpte170m", "smpte170m", "smpte170m", "tv")


class Context:
    """The two NodeContext members Save Video uses."""

    def __init__(self):
        self.storage_dir = STORAGE
        self.cleanups = []

    def add_cleanup(self, fn: Callable[[], None]):
        self.cleanups.append(fn)


def bars(width: int, height: int) -> np.ndarray:
    """RGB bars as Save Video takes them: float BGR in 0..1."""
    image = np.zeros((height, width, 3), np.uint8)
    edges = np.linspace(0, width, len(BARS) + 1).astype(int)
    for (r, g, b), x0, x1 in zip(BARS, edges, edges[1:], strict=False):
        image[:, x0:x1] = (b, g, r)
    return image.astype(np.float32) / 255


def centres(frame: np.ndarray) -> np.ndarray:
    """Each bar's centre pixel, as RGB."""
    height, width = frame.shape[:2]
    edges = np.linspace(0, width, len(BARS) + 1).astype(int)
    xs = (edges[:-1] + edges[1:]) // 2
    return frame[height // 2, xs, ::-1].astype(int)


def save(
    tmp_path: Path,
    image: np.ndarray,
    container: VideoFormat,
    encoder: VideoEncoder,
    additional: str | None = None,
) -> Path:
    collector = save_video_node(
        cast(NodeContext, Context()),
        None,
        tmp_path,
        "out",
        Simplicity.ADVANCED,
        container,
        encoder,
        VideoPreset.ULTRA_FAST,
        0,
        additional,
        SimpleVideoFormat.MP4_H264,
        75,
        2,
        None,
        AudioSettings.AUTO,
    )
    for _ in range(2):
        collector.on_iterate(image)
    collector.on_complete()
    return tmp_path / f"out.{container.ext}"


def tags(path: Path) -> tuple[str, ...]:
    probe = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    ).stdout
    stream = json.loads(probe)["streams"][0]
    keys = ("color_space", "color_primaries", "color_transfer", "color_range")
    return tuple(stream.get(key, "-") for key in keys)


def read(path: Path, monkeypatch: pytest.MonkeyPatch, cli: bool) -> np.ndarray:
    if cli:
        monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    else:
        monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    loader = VideoLoader(path, FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE))
    return next(iter(loader.stream_frames()))


FORMATS = [
    (VideoFormat.MP4, VideoEncoder.H264),
    (VideoFormat.MP4, VideoEncoder.H265),
    (VideoFormat.MP4, VideoEncoder.VP9),
    (VideoFormat.MKV, VideoEncoder.H264),
    (VideoFormat.MKV, VideoEncoder.FFV1),
    (VideoFormat.MOV, VideoEncoder.H264),
    (VideoFormat.WEBM, VideoEncoder.VP9),
    (VideoFormat.AVI, VideoEncoder.H264),
]


@pytest.mark.parametrize(("container", "encoder"), FORMATS)
@pytest.mark.parametrize(
    ("size", "expected"), [((720, 480), BT601), ((1280, 720), BT709)]
)
def test_every_format_carries_four_tags_and_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    container: VideoFormat,
    encoder: VideoEncoder,
    size: tuple[int, int],
    expected: tuple[str, ...],
):
    image = bars(*size)
    path = save(tmp_path, image, container, encoder)
    assert tags(path) == expected
    original = centres(np.round(image * 255).astype(np.uint8))
    for cli in (False, True):
        error = np.abs(centres(read(path, monkeypatch, cli)) - original).max()
        assert error <= 2


def test_576_lines_get_pal_primaries(tmp_path: Path):
    path = save(tmp_path, bars(720, 576), VideoFormat.MKV, VideoEncoder.H264)
    assert tags(path) == ("smpte170m", "bt470bg", "smpte170m", "tv")


def test_grey_stays_grey(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # FFmpeg 5.1.2's default conversion turned grey 128 into (125, 128, 125).
    image = np.full((48, 64, 3), 128 / 255, np.float32)
    path = save(tmp_path, image, VideoFormat.MP4, VideoEncoder.H264)
    assert (read(path, monkeypatch, cli=True)[24, 32] == 128).all()


@pytest.mark.parametrize(
    ("size", "additional", "expected"),
    [
        (
            (720, 480),
            "-colorspace bt709 -color_primaries bt709 -color_trc bt709",
            BT709,
        ),
        # A user matrix alone takes its own family's primaries and transfer.
        ((1280, 720), "-colorspace bt2020nc", ("bt2020nc", "bt2020", "bt709", "tv")),
    ],
)
def test_user_tags_win_and_drive_the_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: tuple[int, int],
    additional: str,
    expected: tuple[str, ...],
):
    image = bars(*size)
    path = save(tmp_path, image, VideoFormat.MP4, VideoEncoder.H264, additional)
    assert tags(path) == expected
    original = centres(np.round(image * 255).astype(np.uint8))
    assert np.abs(centres(read(path, monkeypatch, cli=True)) - original).max() <= 2


def test_a_user_filter_runs_before_the_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image = bars(1280, 720)
    path = save(tmp_path, image, VideoFormat.MP4, VideoEncoder.H264, "-vf hflip")
    assert tags(path) == BT709
    flipped = centres(np.round(image[:, ::-1] * 255).astype(np.uint8))
    assert np.abs(centres(read(path, monkeypatch, cli=True)) - flipped).max() <= 2


def untagged_clip(tmp_path: Path, size: tuple[int, int], matrix: str) -> Path:
    """Bars converted with `matrix` and written without colour tags."""
    width, height = size
    image = np.round(bars(width, height) * 255).astype(np.uint8)
    path = tmp_path / f"untagged-{matrix}-{width}x{height}.mp4"
    scale = f"scale=out_color_matrix={matrix}:out_range=tv"
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-i",
            "pipe:",
            "-vf",
            f"{scale}:flags=accurate_rnd+full_chroma_int",
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        input=image.tobytes(),
        capture_output=True,
        check=True,
    )
    assert tags(path)[:3] == ("-", "-", "-")
    return path


@pytest.mark.parametrize(
    ("size", "matrix"), [((1920, 1080), "bt709"), ((640, 480), "bt601")]
)
def test_untagged_video_is_read_by_the_size_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: tuple[int, int],
    matrix: str,
):
    path = untagged_clip(tmp_path, size, matrix)
    original = centres(np.round(bars(*size) * 255).astype(np.uint8))
    frames = [read(path, monkeypatch, cli) for cli in (False, True)]
    np.testing.assert_array_equal(frames[0], frames[1])
    assert np.abs(centres(frames[0]) - original).max() <= 2


def gradient(width: int, height: int) -> np.ndarray:
    """Red across, green down, blue at half: float BGR in 0..1."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    return np.stack([np.full_like(x, 0.5), y / (height - 1), x / (width - 1)], axis=-1)


@pytest.mark.parametrize("pattern", ["gradient", "grey"])
def test_gif_gets_a_palette_per_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pattern: str
):
    # FFmpeg replaced the node's yuv420p with bgr8's fixed 3-3-2 palette: both came back
    # up to 43 levels off, grey 128 as (144, 144, 170). Frames and timing are unchanged.
    if pattern == "grey":
        image = np.full((48, 64, 3), 128 / 255, np.float32)
    else:
        image = gradient(160, 120)
    path = save(tmp_path, image, VideoFormat.GIF, VideoEncoder.H264)
    probe = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "frame=pts",
            "-of",
            "json",
            str(path),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    ).stdout
    assert [frame["pts"] for frame in json.loads(probe)["frames"]] == [0, 50]
    original = np.round(image * 255).astype(int)
    for cli in (False, True):
        error = np.abs(read(path, monkeypatch, cli).astype(int) - original)
        assert error.max() <= (0 if pattern == "grey" else 30)
        assert error.mean() <= 5
