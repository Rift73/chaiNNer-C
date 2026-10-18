"""Load Video and Save Video together: planes round trip bit-identically through FFV1,
chroma siting is honoured on load, on save and by the siting correction, and 8-bit
(as before) reads as the old reader did. Runs on chaiNNer's integrated FFmpeg."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

# The node's modules load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import NodeContext
from nodes.impl.colour_info import (
    ChromaLocation,
    ColourInfo,
    HdrInfo,
    Matrix,
    Primaries,
    Range,
    Transfer,
    VideoMode,
    code_to_float,
)
from nodes.impl.ffmpeg import get_executable_path
from nodes.impl.plane_resample import Kernel, resample_plane
from packages.chaiNNer_standard.image.video_frames.load_video import (
    RgbPrecision,
    load_video_node,
)
from packages.chaiNNer_standard.image.video_frames.save_video import (
    AudioSettings,
    BitDepth,
    SimpleVideoFormat,
    Simplicity,
    SitingCorrection,
    VideoEncoder,
    VideoFormat,
    VideoInputMode,
    VideoPreset,
    save_video_node,
)

STORAGE = Path(os.environ.get("APPDATA", "")) / "chaiNNer" / "backend-storage"
FFMPEG, FFPROBE = map(str, get_executable_path(STORAGE / "ffmpeg"))

pytestmark = pytest.mark.skipif(
    not Path(FFMPEG).exists(), reason="the integrated FFmpeg is not downloaded"
)
AUTO = (Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO, ChromaLocation.AUTO)
SWS_FLAGS = "lanczos+accurate_rnd+full_chroma_int+full_chroma_inp+bitexact"


class Context:
    """The two NodeContext members the video nodes use."""

    def __init__(self):
        self.storage_dir = STORAGE
        self.cleanups: list[Callable[[], None]] = []

    def add_cleanup(self, fn: Callable[[], None]):
        self.cleanups.append(fn)


def ffmpeg(*arguments: str, data: bytes | None = None) -> bytes:
    return subprocess.run(
        [FFMPEG, "-v", "error", "-y", *arguments],
        input=data,
        capture_output=True,
        check=True,
        stdin=None if data is not None else subprocess.DEVNULL,
    ).stdout


def load(
    path: Path,
    mode: VideoMode,
    precision: RgbPrecision = RgbPrecision.FLOAT,
    choices: tuple[Any, ...] = AUTO,
) -> tuple[list[tuple[Any, ...]], ColourInfo]:
    context = Context()
    outputs = load_video_node(
        cast(NodeContext, context), path, False, 0, mode, precision, *choices
    )
    items: list[tuple[Any, ...]] = []
    for item in outputs[0].supplier():
        assert isinstance(item, tuple)
        items.append(item)
    for cleanup in context.cleanups:
        cleanup()
    return items, outputs[5]


def save(
    directory: Path,
    frames: list[tuple[Any, ...]],
    mode: VideoInputMode,
    colour: ColourInfo | None,
    choices: tuple[Any, ...] = AUTO,
    bit_depth: BitDepth = BitDepth.AUTO,
    siting: SitingCorrection = SitingCorrection.AUTO,
    encoder: VideoEncoder = VideoEncoder.FFV1,
    additional: str | None = None,
) -> Path:
    matrix, transfer, primaries, range_, chroma_location = choices
    context = Context()
    collector = save_video_node(
        cast(NodeContext, context),
        None,
        directory,
        "out",
        mode,
        None,
        None,
        None,
        Simplicity.ADVANCED,
        VideoFormat.MKV,
        encoder,
        VideoPreset.ULTRA_FAST,
        0,
        additional,
        SimpleVideoFormat.MP4_H264,
        75,
        5,
        None,
        AudioSettings.AUTO,
        colour,
        matrix,
        transfer,
        primaries,
        range_,
        chroma_location,
        bit_depth,
        siting,
    )
    for frame in frames:
        collector.on_iterate(frame)
    collector.on_complete()
    return directory / "out.mkv"


TAG_SETS = {
    "bt709-left": ("bt709", "bt709", "bt709", "left"),
    "bt2020-hlg-topleft": ("bt2020nc", "bt2020", "arib-std-b67", "topleft"),
}


@pytest.mark.parametrize("tags", sorted(TAG_SETS))
@pytest.mark.parametrize("colour_range", ["tv", "pc"])
@pytest.mark.parametrize("depth", ["", "10le", "16le"])
@pytest.mark.parametrize("layout", ["420", "422", "444"])
def test_planes_round_trip_bit_identically(
    tmp_path: Path, layout: str, depth: str, colour_range: str, tags: str
):
    matrix, primaries, transfer, location = TAG_SETS[tags]
    source = tmp_path / "source.mkv"
    ffmpeg(
        *("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=5:duration=0.6"),
        *("-c:v", "ffv1", "-pix_fmt", f"yuv{layout}p{depth}"),
        *("-colorspace", matrix, "-color_primaries", primaries),
        *("-color_trc", transfer, "-color_range", colour_range),
        *("-chroma_sample_location", location),
        str(source),
    )
    items, colour = load(source, VideoMode.PLANES)
    assert all(colour.origin_of(name) == "file" for name, _ in colour.origin)
    assert len(items) == 3
    planes = [(None, *item[2:]) for item in items]
    saved = save(tmp_path, planes, VideoInputMode.PLANES, colour)
    reloaded, reloaded_colour = load(saved, VideoMode.PLANES)
    assert reloaded_colour == colour
    assert len(reloaded) == len(items)
    for item, again in zip(items, reloaded, strict=True):
        for plane, plane_again in zip(item[2:], again[2:], strict=True):
            assert plane.dtype == plane_again.dtype == np.float32
            np.testing.assert_array_equal(plane, plane_again)


def impulse_video(path: Path, location: str, column: int) -> None:
    """A 64x16 4:2:0 8-bit video: grey, with one U sample raised in each chroma row at
    `column`, tagged with the chroma siting `location`."""
    y = np.full((16, 64), 126, np.uint8)
    u = np.full((8, 32), 128, np.uint8)
    v = np.full((8, 32), 128, np.uint8)
    u[:, column] = 200
    data = b"".join(p.tobytes() for p in (y, u, v)) * 2
    ffmpeg(
        *("-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "64x16", "-r", "5"),
        *("-i", "-", "-c:v", "ffv1", "-colorspace", "bt709"),
        *("-color_range", "tv", "-chroma_sample_location", location, str(path)),
        data=data,
    )


def centroid(values: np.ndarray) -> float:
    """The centre of a row's deviation from its edges' level, the deviation (with the
    resampling kernels' negative lobes) as the weight."""
    weights = values.astype(np.float64) - values[0]
    return float((weights * np.arange(values.size)).sum() / weights.sum())


def test_chroma_siting_is_honoured_on_load(tmp_path: Path):
    column = 15
    sites = {}
    for location in ("left", "center"):
        path = tmp_path / f"{location}.mkv"
        impulse_video(path, location, column)
        items, colour = load(path, VideoMode.RGB)
        assert colour.chroma_location.value == location
        blue = items[0][0][8, :, 0]  # U raises blue
        sites[location] = centroid(blue)
    # Left: the chroma sample sits on luma column 2c; centred, half a pixel right of it.
    assert sites["left"] == pytest.approx(2 * column, abs=0.05)
    assert sites["center"] == pytest.approx(2 * column + 0.5, abs=0.05)


def test_chroma_siting_is_honoured_on_save(tmp_path: Path):
    # A blue column at luma x = 30: left-sited chroma puts its response at chroma 15,
    # centred at (30 - 0.5) / 2.
    image = np.full((16, 64, 3), 0.5, np.float32)
    image[:, 30, 0] = 1.0
    frames = [(image, None, None, None)] * 2
    sites = {}
    for location in (ChromaLocation.LEFT, ChromaLocation.CENTER):
        directory = tmp_path / location.value
        directory.mkdir()
        choices = (*AUTO[:4], location)
        saved = save(directory, frames, VideoInputMode.RGB, None, choices)
        items, colour = load(saved, VideoMode.PLANES)
        assert colour.chroma_location is location
        sites[location] = centroid(items[0][3][4])
    assert sites[ChromaLocation.LEFT] == pytest.approx(15, abs=0.05)
    assert sites[ChromaLocation.CENTER] == pytest.approx(14.75, abs=0.05)


def test_siting_correction_realigns_upscaled_left_sited_chroma(tmp_path: Path):
    # A left-sited 4:2:0 source whose U is a ramp of the source luma position x:
    # sample i sits at x = 2i. Upscaled 2x (centre-aligned) its sample m holds x = m -
    # 0.5, where the saved left siting places x = m - 0.25 (new luma 2m).
    width, height = 64, 16
    source_u = np.tile(0.3 + 0.01 * 2 * np.arange(width // 2, dtype=np.float32), (8, 1))
    y = np.full((height * 2, width * 2), 0.5, np.float32)
    up_u = resample_plane(source_u, width, height, Kernel.SPLINE36)
    v = np.full_like(up_u, 0.5)
    colour = ColourInfo(
        matrix=Matrix.BT709,
        transfer=Transfer.BT709,
        primaries=Primaries.BT709,
        range=Range.LIMITED,
        chroma_location=ChromaLocation.LEFT,
        bit_depth=10,
        luma_size=(width, height),
        chroma_size=(width // 2, height // 2),
    )
    frames = [(None, y, up_u, v)] * 2
    results = {}
    for siting in (SitingCorrection.AUTO, SitingCorrection.OFF):
        directory = tmp_path / siting.value
        directory.mkdir()
        saved = save(directory, frames, VideoInputMode.PLANES, colour, siting=siting)
        items, saved_colour = load(saved, VideoMode.PLANES)
        assert saved_colour.chroma_size == (width, height)
        results[siting] = items[0][3][4]
    m = np.arange(width, dtype=np.float32)
    step = 1 / 224 / 4  # one 10-bit chroma code
    interior = slice(4, width - 4)
    np.testing.assert_allclose(
        results[SitingCorrection.AUTO][interior],
        (0.3 + 0.01 * (m - 0.25))[interior],
        atol=step,
    )
    np.testing.assert_allclose(
        results[SitingCorrection.OFF][interior],
        (0.3 + 0.01 * (m - 0.5))[interior],
        atol=step,
    )


def old_reader(path: Path, matrix: str | None) -> list[np.ndarray]:
    """The frames as the CLI reader read them before Load Video's Mode."""
    scale = (
        ["-vf", f"scale=in_color_matrix={matrix}:flags={SWS_FLAGS}"] if matrix else []
    )
    data = ffmpeg(
        *("-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24"),
        *("-sws_flags", SWS_FLAGS, *scale, "pipe:"),
    )
    probe = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
    ).stdout
    width, height = (int(v) for v in probe.split()[0].split(","))
    frames = np.frombuffer(data, np.uint8).reshape(-1, height, width, 3)
    return list(frames)


CASES = {
    "h264-hd-untagged": (
        "1280x720",
        ("-c:v", "libx264", "-pix_fmt", "yuv420p"),
        "bt709",
    ),
    "ffv1-444-odd": ("639x361", ("-c:v", "ffv1", "-pix_fmt", "yuv444p"), None),
    "mjpeg-full-range": ("320x240", ("-c:v", "mjpeg", "-pix_fmt", "yuvj420p"), None),
    "h264-bt601-tagged": (
        "640x480",
        ("-c:v", "libx264", "-pix_fmt", "yuv420p", "-colorspace", "smpte170m"),
        None,
    ),
}


@pytest.mark.parametrize("cli", [False, True])
@pytest.mark.parametrize("case", sorted(CASES))
def test_8_bit_as_before_equals_the_old_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, cli: bool
):
    size, codec, matrix = CASES[case]
    suffix = ".mkv" if "ffv1" in case else ".mp4"
    path = tmp_path / f"clip{suffix}"
    ffmpeg(
        *("-f", "lavfi", "-i", f"testsrc2=size={size}:rate=10", "-frames:v", "4"),
        *codec,
        str(path),
    )
    if cli:
        monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    else:
        monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    items, _ = load(path, VideoMode.RGB, RgbPrecision.EIGHT_BIT)
    expected = old_reader(path, matrix)
    assert len(items) == len(expected) == 4
    for item, frame in zip(items, expected, strict=True):
        np.testing.assert_array_equal(item[0], frame)


def test_16_bit_rgb_save_keeps_more_than_8_bits(tmp_path: Path):
    # A ramp finer than 8 bits saved as 16-bit 4:4:4 keeps its steps.
    ramp = np.linspace(0.25, 0.25 + 64 / 65535 * 40, 64, dtype=np.float32)
    image = np.dstack([np.tile(ramp, (16, 1))] * 3)
    saved = save(
        tmp_path,
        [(image, None, None, None)] * 2,
        VideoInputMode.RGB,
        None,
        bit_depth=BitDepth.BITS_16,
    )
    items, colour = load(saved, VideoMode.PLANES)
    assert colour.bit_depth == 16
    y = items[0][2][8]
    expected = code_to_float(
        np.rint((ramp * 219 + 16) * 256).astype(np.uint16), 16, False, False
    )
    assert np.unique(y).size > 32  # 8 bits would give at most 11 steps
    # Within a quarter of an 8-bit step: libswscale's RGB to YUV is not exact in 16 bits.
    np.testing.assert_allclose(y, expected, atol=1 / 219 / 4)


def test_bit_depth_auto_follows_the_colour_info_within_the_encoders_cap(
    tmp_path: Path,
):
    colour = ColourInfo(
        matrix=Matrix.BT709,
        transfer=Transfer.BT709,
        primaries=Primaries.BT709,
        range=Range.LIMITED,
        chroma_location=ChromaLocation.LEFT,
        bit_depth=12,
        luma_size=(64, 48),
        chroma_size=(32, 24),
    )
    planes = (
        None,
        np.full((48, 64), 0.5, np.float32),
        np.full((24, 32), 0.5, np.float32),
        np.full((24, 32), 0.5, np.float32),
    )
    saved = save(tmp_path, [planes] * 2, VideoInputMode.PLANES, colour)
    _, saved_colour = load(saved, VideoMode.PLANES)
    assert saved_colour.bit_depth == 12  # FFV1 keeps the source's
    assert replace(saved_colour, origin=()) == colour


def test_rgb_coded_planes_are_coded_like_luma(tmp_path: Path):
    # Identity-matrix planes are G, B and R (Planes to RGB's order), none of them
    # 0.5-centred: they are written as gbrp, each coded as Y is.
    rng = np.random.default_rng(3)
    planes = [rng.integers(0, 1024, (16, 32)).astype(np.float32) / 1023 for _ in "GBR"]
    colour = ColourInfo(
        matrix=Matrix.RGB,
        transfer=Transfer.SRGB,
        primaries=Primaries.BT709,
        range=Range.FULL,
        chroma_location=ChromaLocation.CENTER,
        bit_depth=10,
        luma_size=(32, 16),
        chroma_size=(32, 16),
    )
    saved = save(tmp_path, [(None, *planes)] * 2, VideoInputMode.PLANES, colour)
    _, saved_colour = load(saved, VideoMode.RGB)
    assert (saved_colour.matrix, saved_colour.bit_depth) == (Matrix.RGB, 10)
    raw = ffmpeg("-i", str(saved), "-f", "rawvideo", "-pix_fmt", "gbrp10le", "pipe:")
    codes = np.frombuffer(raw, "<u2").reshape(2, 3, 16, 32)
    for frame in codes:
        for plane, code in zip(planes, frame, strict=True):
            np.testing.assert_array_equal(np.rint(plane * 1023), code)


def stream_tags(path: Path) -> tuple[str, ...]:
    keys = ("color_space", "color_primaries", "color_transfer", "color_range")
    probe = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        check=True,
        stdin=subprocess.DEVNULL,
    ).stdout
    stream = json.loads(probe)["streams"][0]
    return tuple(stream.get(key, "-") for key in keys)


def test_rgb_mode_takes_tags_from_the_colour_info(tmp_path: Path):
    image = np.full((16, 64, 3), 0.5, np.float32)
    frames = [(image, None, None, None)] * 2
    hlg = ColourInfo(
        matrix=Matrix.BT2020_NCL,
        transfer=Transfer.HLG,
        primaries=Primaries.BT2020,
        range=Range.FULL,
        chroma_location=ChromaLocation.TOP_LEFT,
        bit_depth=10,
        luma_size=(64, 16),
        chroma_size=(32, 8),
    )
    saved = save(tmp_path, frames, VideoInputMode.RGB, hlg)
    assert stream_tags(saved) == ("bt2020nc", "bt2020", "arib-std-b67", "pc")
    _, colour = load(saved, VideoMode.PLANES)
    assert (colour.bit_depth, colour.chroma_location) == (10, ChromaLocation.TOP_LEFT)
    # An RGB-coded source gives its transfer and primaries; the YUV output's matrix,
    # range and siting are the size guess's.
    rgb = replace(hlg, matrix=Matrix.RGB, chroma_size=(64, 16))
    directory = tmp_path / "rgb"
    directory.mkdir()
    saved = save(directory, frames, VideoInputMode.RGB, rgb)
    assert stream_tags(saved) == ("smpte170m", "bt2020", "arib-std-b67", "tv")


def test_rgb_mode_refuses_a_matrix_ffmpeg_cannot_convert_to(tmp_path: Path):
    image = np.full((16, 64, 3), 0.5, np.float32)
    choices = (Matrix.YCGCO, *AUTO[1:])
    with pytest.raises(ValueError, match="YCgCo"):
        save(tmp_path, [(image, None, None, None)], VideoInputMode.RGB, None, choices)


def test_encoders_refuse_pixel_formats_they_cannot_write(tmp_path: Path):
    planes = (
        None,
        np.full((16, 64), 0.5, np.float32),
        np.full((8, 32), 0.5, np.float32),
        np.full((8, 32), 0.5, np.float32),
    )
    context = Context()
    collector = save_video_node(
        cast(NodeContext, context),
        None,
        tmp_path,
        "out",
        VideoInputMode.PLANES,
        None,
        None,
        None,
        Simplicity.ADVANCED,
        VideoFormat.MKV,
        VideoEncoder.H264,
        VideoPreset.ULTRA_FAST,
        23,
        None,
        SimpleVideoFormat.MP4_H264,
        75,
        5,
        None,
        AudioSettings.AUTO,
        None,
        *AUTO,
        BitDepth.BITS_16,
        SitingCorrection.AUTO,
    )
    with pytest.raises(ValueError, match="libx264 encoder cannot write yuv420p16le"):
        collector.on_iterate(planes)
    for cleanup in context.cleanups:
        cleanup()


def test_hdr10_metadata_round_trips_through_h265(tmp_path: Path):
    hdr = HdrInfo(
        mastering_display=(
            "G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,50)"
        ),
        max_cll=1000,
        max_fall=400,
    )
    colour = ColourInfo(
        matrix=Matrix.BT2020_NCL,
        transfer=Transfer.PQ,
        primaries=Primaries.BT2020,
        range=Range.LIMITED,
        chroma_location=ChromaLocation.TOP_LEFT,
        bit_depth=10,
        luma_size=(64, 48),
        chroma_size=(32, 24),
        hdr=hdr,
    )
    planes = (
        None,
        np.full((48, 64), 0.5, np.float32),
        np.full((24, 32), 0.5, np.float32),
        np.full((24, 32), 0.5, np.float32),
    )
    saved = save(
        tmp_path,
        [planes] * 2,
        VideoInputMode.PLANES,
        colour,
        encoder=VideoEncoder.H265,
        additional="-x265-params pools=none:frame-threads=1:log-level=error",
    )
    _, reloaded = load(saved, VideoMode.PLANES)
    assert reloaded.hdr == hdr
    assert (reloaded.transfer, reloaded.primaries, reloaded.bit_depth) == (
        Transfer.PQ,
        Primaries.BT2020,
        10,
    )
