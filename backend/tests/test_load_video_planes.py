"""Load Video's Mode (RGB, Planes, Both), RGB Precision and colour outputs, in both of
its readers (PyAV and the FFmpeg CLI). Runs on chaiNNer's integrated FFmpeg."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import ffmpeg
import numpy as np
import pytest

# The node's modules load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import NodeContext, OutputId
from nodes.impl import video
from nodes.impl.colour_info import (
    ChromaLocation,
    ColourInfo,
    Matrix,
    Primaries,
    Range,
    Transfer,
    VideoMode,
    float_to_code,
)
from nodes.impl.ffmpeg import get_executable_path
from nodes.impl.video import VideoMetadata
from packages.chaiNNer_standard.image.video_frames.load_video import (
    FramesMode,
    LoadedVideo,
    RgbPrecision,
    get_item_types,
    load_video_node,
)

STORAGE = Path(os.environ.get("APPDATA", "")) / "chaiNNer" / "backend-storage"
FFMPEG, FFPROBE = map(str, get_executable_path(STORAGE / "ffmpeg"))
# The local gates' FFmpeg (native/tools/bench_oracle.py), newer than the integrated one.
NEWER_FFMPEG = Path(r"C:\Executables\ffmpeg-8.1-full_build-shared\bin\ffmpeg.exe")

pytestmark = pytest.mark.skipif(
    not Path(FFMPEG).exists(), reason="the integrated FFmpeg is not downloaded"
)
AUTO = (Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO, ChromaLocation.AUTO)


class Context:
    """The NodeContext members Load Video uses."""

    node_id = "load-video"  # the Preview's reader is the node's

    def __init__(self, storage: Path = STORAGE):
        self.storage_dir = storage
        self.cleanups: list[Callable[[], None]] = []

    def add_cleanup(self, fn: Callable[[], None]):
        self.cleanups.append(fn)


def make_video(path: Path, source: str, *arguments: str) -> Path:
    subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            source,
            *arguments,
            str(path),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    return path


def raw_planes(path: Path, pix_fmt: str) -> bytes:
    return subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "pipe:",
        ],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    ).stdout


def load(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli: bool,
    mode: VideoMode,
    precision: RgbPrecision = RgbPrecision.FLOAT,
    choices: tuple[Any, ...] = AUTO,
    storage: Path = STORAGE,
) -> tuple[list[tuple[Any, ...]], ColourInfo, str]:
    if cli:
        monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    else:
        monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    context = Context(storage)
    outputs = load_video_node(
        cast(NodeContext, context),
        path,
        0,
        FramesMode.ALL,
        0,
        0,
        mode,
        precision,
        *choices,
    )
    items: list[tuple[Any, ...]] = []
    for item in outputs[0].supplier():
        assert isinstance(item, tuple)
        items.append(item)
    for cleanup in context.cleanups:
        cleanup()
    video.close_previews()  # the Preview's reader lets the file go
    return items, outputs[5], outputs[6]


SOURCE = "testsrc2=size=96x64:rate=5:duration=0.6"


@pytest.mark.parametrize("cli", [False, True])
@pytest.mark.parametrize(
    ("pix_fmt", "chroma"),
    [("yuv420p", (48, 32)), ("yuv422p10le", (48, 64)), ("yuv444p16le", (96, 64))],
)
def test_planes_are_the_stored_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli: bool,
    pix_fmt: str,
    chroma: tuple[int, int],
):
    path = make_video(
        tmp_path / "planes.mkv", SOURCE, "-c:v", "ffv1", "-pix_fmt", pix_fmt
    )
    items, colour, _ = load(path, monkeypatch, cli, VideoMode.PLANES)
    depth = colour.bit_depth
    assert (colour.luma_size, colour.chroma_size) == ((96, 64), chroma)
    dtype = np.uint8 if depth == 8 else np.dtype("<u2")
    raw = np.frombuffer(raw_planes(path, pix_fmt), dtype)
    sizes = [96 * 64, chroma[0] * chroma[1], chroma[0] * chroma[1]]
    assert len(items) == 3 and raw.size == 3 * sum(sizes)
    offset = 0
    for frame, index, *planes in items:
        assert frame is None and isinstance(index, int)
        for plane_index, (plane, size) in enumerate(zip(planes, sizes)):
            assert plane.dtype == np.float32
            codes = float_to_code(plane, depth, colour.full_range, plane_index > 0)
            np.testing.assert_array_equal(codes.ravel(), raw[offset : offset + size])
            offset += size


@pytest.fixture
def newer_ffmpeg_storage(tmp_path: Path):
    """A storage folder whose ffmpeg/ is the local gates' FFmpeg 8.1 (a junction,
    removed link-only)."""
    if not NEWER_FFMPEG.exists():
        pytest.skip("the local gates' FFmpeg is not installed")
    storage = tmp_path / "storage"
    storage.mkdir()
    import _winapi

    _winapi.CreateJunction(str(NEWER_FFMPEG.parent), str(storage / "ffmpeg"))
    yield storage
    os.rmdir(storage / "ffmpeg")


def test_cli_planes_on_a_newer_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, newer_ffmpeg_storage: Path
):
    """The CLI reader keeps the stored orientation of planes with -noautorotate, which
    FFmpeg 8.1 takes as 5.1.2 does: 8.1's -autorotate takes no value, so `-autorotate
    0` failed to start (its 0 read as the output)."""
    path = make_video(
        tmp_path / "planes.mkv", SOURCE, "-c:v", "ffv1", "-pix_fmt", "yuv420p"
    )
    items, colour, _ = load(
        path, monkeypatch, True, VideoMode.PLANES, storage=newer_ffmpeg_storage
    )
    raw = np.frombuffer(raw_planes(path, "yuv420p"), np.uint8)
    codes = [
        float_to_code(plane, 8, colour.full_range, index > 0).ravel()
        for _, _, *planes in items
        for index, plane in enumerate(planes)
    ]
    assert len(items) == 3
    np.testing.assert_array_equal(np.concatenate(codes), raw)


def test_both_readers_and_modes_agree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = make_video(
        tmp_path / "both.mkv", SOURCE, "-c:v", "ffv1", "-pix_fmt", "yuv420p10le"
    )
    results = {}
    for cli in (False, True):
        for mode in VideoMode:
            items, _, _ = load(path, monkeypatch, cli, mode)
            results[cli, mode] = items
    for mode in VideoMode:
        for a, b in zip(results[False, mode], results[True, mode], strict=True):
            for x, y in zip(a, b, strict=True):
                if isinstance(x, np.ndarray):
                    np.testing.assert_array_equal(x, y)
                else:
                    assert x == y
    for both, rgb, planes in zip(
        results[False, VideoMode.BOTH],
        results[False, VideoMode.RGB],
        results[False, VideoMode.PLANES],
        strict=True,
    ):
        np.testing.assert_array_equal(both[0], rgb[0])
        for x, y in zip(both[2:], planes[2:], strict=True):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("cli", [False, True])
def test_float_rgb_is_within_one_code_of_the_8_bit_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli: bool
):
    # Flat colours: where chroma does not change, its siting does not matter.
    path = make_video(
        tmp_path / "flat.mp4",
        "color=c=0x3080C0:size=96x64:rate=5:duration=0.4",
        *("-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "0"),
    )
    floats, _, _ = load(path, monkeypatch, cli, VideoMode.RGB)
    bytes_, _, _ = load(path, monkeypatch, cli, VideoMode.RGB, RgbPrecision.EIGHT_BIT)
    for (exact, _, *_), (old, _, *_) in zip(floats, bytes_, strict=True):
        assert exact.dtype == np.float32 and old.dtype == np.uint8
        assert np.abs(exact * 255 - old.astype(np.float32)).max() <= 1


@pytest.mark.parametrize("cli", [False, True])
def test_8_bit_reading_ignores_a_transfer_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli: bool
):
    # Sharp chroma edges, left-sited: a chroma siting would change the 8-bit frames.
    path = make_video(
        tmp_path / "left.mkv", SOURCE, "-c:v", "ffv1", "-pix_fmt", "yuv420p"
    )
    auto, colour, _ = load(
        path, monkeypatch, cli, VideoMode.RGB, RgbPrecision.EIGHT_BIT
    )
    transfer = (AUTO[0], Transfer.LINEAR, *AUTO[2:])
    chosen, _, _ = load(
        path, monkeypatch, cli, VideoMode.RGB, RgbPrecision.EIGHT_BIT, transfer
    )
    assert colour.chroma_location is ChromaLocation.LEFT
    for (old, *_), (new, *_) in zip(auto, chosen, strict=True):
        assert old.dtype == new.dtype == np.uint8
        np.testing.assert_array_equal(old, new)


def test_colour_outputs_and_choices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = make_video(
        tmp_path / "tagged.mkv",
        SOURCE,
        *("-vf", "setparams=color_primaries=bt709:color_trc=bt709"),
        *("-c:v", "ffv1", "-pix_fmt", "yuv420p"),
        *("-colorspace", "bt709", "-color_primaries", "bt709"),
        *("-color_trc", "bt709", "-color_range", "tv"),
    )
    items, colour, text = load(path, monkeypatch, False, VideoMode.PLANES)
    assert (colour.matrix, colour.range, colour.origin_of("matrix")) == (
        Matrix.BT709,
        Range.LIMITED,
        "file",
    )
    assert text == colour.summary() and text.startswith("Matrix BT.709")
    # Choosing full range reads the same codes as full-range values.
    full = (*AUTO[:3], Range.FULL, AUTO[4])
    chosen_items, chosen, _ = load(
        path, monkeypatch, False, VideoMode.PLANES, choices=full
    )
    assert chosen.range is Range.FULL and chosen.origin_of("range") == "set"
    y, chosen_y = items[0][2], chosen_items[0][2]
    np.testing.assert_array_equal(
        float_to_code(y, 8, False, False), float_to_code(chosen_y, 8, True, False)
    )


def test_rgb_coded_video_has_no_planes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = make_video(tmp_path / "rgb.mkv", SOURCE, "-c:v", "ffv1", "-pix_fmt", "gbrp")
    with pytest.raises(ValueError, match="RGB-coded"):
        load(path, monkeypatch, False, VideoMode.PLANES)
    items, colour, _ = load(path, monkeypatch, False, VideoMode.RGB)
    assert colour.matrix is Matrix.RGB and items[0][0].shape == (64, 96, 3)


@pytest.mark.parametrize("pix_fmt", ["gray", "gray10le"])
def test_gray_video_has_no_chroma_planes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pix_fmt: str
):
    # U and V are typed as images in Planes and Both, so gray video is refused there.
    path = make_video(
        tmp_path / "gray.mkv", SOURCE, "-c:v", "ffv1", "-pix_fmt", pix_fmt
    )
    for mode in (VideoMode.PLANES, VideoMode.BOTH):
        with pytest.raises(ValueError, match="gray video, so it has no U and V planes"):
            load(path, monkeypatch, False, mode)
    items, _, _ = load(path, monkeypatch, False, VideoMode.RGB)
    assert items[0][0].shape == (64, 96, 3) and items[0][2:] == (None, None, None)


def test_failed_hdr_probe_reads_without_hdr_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A PQ stream without HDR side data has its first frame probed for the metadata;
    # when that probe fails, the video is read without it and one warning says so.
    path = make_video(
        tmp_path / "pq.mkv",
        SOURCE,
        *("-vf", "setparams=color_primaries=bt2020:color_trc=smpte2084"),
        *("-c:v", "ffv1", "-pix_fmt", "yuv420p10le", "-colorspace", "bt2020nc"),
        *("-color_primaries", "bt2020", "-color_trc", "smpte2084"),
    )
    probe, warnings = ffmpeg.probe, []

    def failing_frame_probe(*args: Any, **kwargs: Any) -> Any:
        if "show_frames" in kwargs:
            raise ffmpeg.Error("ffprobe", b"", b"simulated failure")
        return probe(*args, **kwargs)

    monkeypatch.setattr(ffmpeg, "probe", failing_frame_probe)
    monkeypatch.setattr(
        video.logger, "warning", lambda *a: warnings.append(a[0] % a[1:])
    )
    items, colour, _ = load(path, monkeypatch, False, VideoMode.PLANES)
    assert len(items) == 3 and colour.transfer is Transfer.PQ and colour.hdr is None
    [message] = [w for w in warnings if "HDR10" in w]
    assert message == (
        f"{path}: its first frame's HDR10 metadata could not be read, so it is left"
        " out: simulated failure"
    )


def test_rgb_mode_refuses_constant_luminance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # libswscale would convert BT.2020 constant luminance as non-constant.
    path = make_video(
        tmp_path / "cl.mkv",
        SOURCE,
        *("-vf", "format=yuv420p10le,setparams=colorspace=bt2020c"),
        *("-c:v", "ffv1", "-pix_fmt", "yuv420p10le", "-colorspace", "bt2020c"),
    )
    with pytest.raises(
        ValueError, match=r"cannot convert the BT\.2020 constant matrix"
    ):
        load(path, monkeypatch, False, VideoMode.RGB)


def test_item_types_follow_the_mode():
    colour = ColourInfo(
        matrix=Matrix.BT709,
        transfer=Transfer.BT709,
        primaries=Primaries.BT709,
        range=Range.LIMITED,
        chroma_location=ChromaLocation.LEFT,
        bit_depth=10,
        luma_size=(1920, 1080),
        chroma_size=(960, 540),
    )
    metadata = VideoMetadata(width=1920, height=1080, fps=24.0, frame_count=10)
    rgb = get_item_types(LoadedVideo(metadata, VideoMode.RGB, colour))
    planes = get_item_types(LoadedVideo(metadata, VideoMode.PLANES, colour))
    both = get_item_types(LoadedVideo(metadata, VideoMode.BOTH, colour))
    assert set(rgb) == {OutputId(0)}
    assert set(planes) == {OutputId(6), OutputId(7), OutputId(8)}
    assert set(both) == set(rgb) | set(planes)
    assert planes[OutputId(7)] == {
        "type": "named",
        "name": "Image",
        "fields": {"width": 960, "height": 540, "channels": 1},
    }
