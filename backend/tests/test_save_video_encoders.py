"""Save Video's AV1, ProRes and NVENC encoders on the pinned FFmpeg: synthetic clips through
the node, then ffprobe and a decode compared with what went in. The NVENC encodes need
an NVIDIA GPU and skip without one; the rest runs anywhere."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from scratch_storage import NO_FFMPEG, STORAGE

import gpu

# The node's modules load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import NodeContext
from nodes.condition import Condition
from nodes.impl.colour_info import (
    ChromaLocation,
    ColourInfo,
    HdrInfo,
    Matrix,
    Primaries,
    Range,
    Transfer,
)
from nodes.impl.ffmpeg import FFMpegEnv, get_executable_path
from nodes.impl.video import ReadSettings, VideoLoader, VideoMode
from packages.chaiNNer_standard.image.video_frames import save_video as node
from packages.chaiNNer_standard.image.video_frames.save_video import (
    AudioSettings,
    BitDepth,
    NvencPreset,
    ProResProfile,
    SimpleVideoFormat,
    Simplicity,
    SitingCorrection,
    VideoEncoder,
    VideoFormat,
    VideoInputMode,
    VideoPreset,
    save_video_node,
)

FFMPEG, FFPROBE = map(str, get_executable_path(STORAGE / "ffmpeg"))

pytestmark = pytest.mark.skipif(not Path(FFMPEG).exists(), reason=NO_FFMPEG)

AUTO = (Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO, ChromaLocation.AUTO)
NVENC = (VideoEncoder.H264_NVENC, VideoEncoder.H265_NVENC, VideoEncoder.AV1_NVENC)
HDR = HdrInfo(
    mastering_display=(
        "G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,50)"
    ),
    max_cll=1000,
    max_fall=400,
)


class _Nvidia:
    def __init__(self, available: bool) -> None:
        self.is_available = available


class Context:
    """The NodeContext members Save Video uses."""

    storage_dir = STORAGE
    chosen_item: int | None = None

    def __init__(self) -> None:
        self.cleanups: list[Any] = []

    def add_cleanup(self, cleanup: Any) -> None:
        self.cleanups.append(cleanup)


def collector(
    folder: Path,
    container: VideoFormat,
    encoder: VideoEncoder,
    mode: VideoInputMode = VideoInputMode.RGB,
    colour: ColourInfo | None = None,
    choices: tuple[Matrix, Transfer, Primaries, Range, ChromaLocation] = AUTO,
    bit_depth: BitDepth = BitDepth.AUTO,
    av1: tuple[int, int] = (11, 35),
    profile: ProResProfile = ProResProfile.HQ,
    nvenc: tuple[NvencPreset, int, int] = (NvencPreset.P4, 23, 30),
    additional: str | None = None,
    audio: object | None = None,
    context: Context | None = None,
):
    return save_video_node(
        cast(NodeContext, context or Context()),
        None,
        folder,
        "out",
        mode,
        None,
        None,
        None,
        Simplicity.ADVANCED,
        container,
        encoder,
        VideoPreset.ULTRA_FAST,
        23,
        *av1,
        profile,
        *nvenc,
        additional,
        SimpleVideoFormat.MP4_H264,
        75,
        10,
        audio,
        AudioSettings.AUTO,
        colour,
        *choices,
        bit_depth,
        SitingCorrection.AUTO,
    )


def rgb_frames(width: int = 256, height: int = 192, count: int = 8) -> list[np.ndarray]:
    """Smooth moving gradients, which a lossy encoder keeps close."""
    y, x = np.indices((height, width), dtype=np.float32)
    return [
        np.stack(
            (
                0.5 + 0.4 * np.sin((x + 3 * i) / 37),
                0.5 + 0.4 * np.cos((y - 2 * i) / 29),
                0.5 + 0.4 * np.sin((x + y + 5 * i) / 53),
            ),
            axis=2,
        )
        for i in range(count)
    ]


def save_rgb(folder: Path, container: VideoFormat, encoder: VideoEncoder, **options):
    frames = rgb_frames()
    c = collector(folder, container, encoder, **options)
    for frame in frames:
        c.on_iterate((frame, None, None, None))
    c.on_complete()
    return folder / f"out.{container.ext}", frames


def planes(
    layout: tuple[int, int], width: int = 128, height: int = 96, count: int = 4
) -> list[tuple[None, np.ndarray, np.ndarray, np.ndarray]]:
    """Y, U and V gradients on the limited-range scale's safe middle."""
    sx, sy = layout
    _y, x = np.indices((height, width), dtype=np.float32)
    v, u = np.indices(
        (-(-height // (1 << sy)), -(-width // (1 << sx))), dtype=np.float32
    )
    return [
        (
            None,
            0.3 + 0.4 * (x + 2 * i) / (width + 8),
            0.4 + 0.2 * u / u.shape[1],
            0.6 - 0.2 * v / v.shape[0],
        )
        for i in range(count)
    ]


def probe(path: Path) -> dict[str, Any]:
    output = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_streams",
            "-count_frames",
            "-of",
            "json",
            "-select_streams",
            "v",
            str(path),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    ).stdout
    return json.loads(output)["streams"][0]


def decode_planar(path: Path, pix_fmt: str) -> np.ndarray:
    return np.frombuffer(
        subprocess.run(
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
                "-",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
        ).stdout,
        dtype=np.uint16 if pix_fmt.endswith("le") else np.uint8,
    )


def read_rgb(path: Path, monkeypatch: pytest.MonkeyPatch, cli: bool) -> np.ndarray:
    if cli:
        monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    else:
        monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    loader = VideoLoader(path, FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE))
    return np.stack(list(loader.stream_frames()))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    error = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if error == 0 else 10 * np.log10(1.0 / error)


def assert_tags(stream: dict[str, Any], *expected: str) -> None:
    keys = ("color_space", "color_primaries", "color_transfer", "color_range")
    assert tuple(stream.get(key) for key in keys) == expected


BT709 = ("bt709", "bt709", "bt709", "tv")
FLAT = (0.2, 0.5, 0.8)
BT601 = ("smpte170m", "smpte170m", "smpte170m", "tv")


@pytest.mark.parametrize(
    "container", [VideoFormat.MKV, VideoFormat.MP4, VideoFormat.WEBM]
)
def test_av1_clip_decodes_to_what_went_in(
    tmp_path: Path, container: VideoFormat, monkeypatch: pytest.MonkeyPatch
):
    path, frames = save_rgb(tmp_path, container, VideoEncoder.AV1, av1=(11, 20))
    stream = probe(path)
    assert stream["codec_name"] == "av1" and stream["pix_fmt"] == "yuv420p"
    assert (stream["width"], stream["height"]) == (256, 192)
    assert stream["nb_read_frames"] == "8"
    assert_tags(stream, *BT601)
    assert stream["chroma_location"] == "left"
    expected = np.stack(frames)
    for cli in (False, True):
        decoded = read_rgb(path, monkeypatch, cli)
        assert decoded.shape == expected.shape
        assert psnr(decoded / 255, expected) > 38


def test_av1_preset_and_crf_reach_the_encoder(tmp_path: Path):
    sizes = {}
    for crf in (1, 63):
        folder = tmp_path / str(crf)
        path, _ = save_rgb(folder, VideoFormat.MKV, VideoEncoder.AV1, av1=(11, crf))
        sizes[crf] = path.stat().st_size
    assert sizes[1] > 4 * sizes[63]


def test_av1_ten_bits_keep_more_than_eight(tmp_path: Path):
    y = (0.3 + 0.0009765625 * np.arange(128, dtype=np.float32)) * np.ones(
        (96, 1), np.float32
    )
    frame = (
        None,
        y,
        np.full((48, 64), 0.5, np.float32),
        np.full((48, 64), 0.5, np.float32),
    )
    c = collector(
        tmp_path,
        VideoFormat.MKV,
        VideoEncoder.AV1,
        mode=VideoInputMode.PLANES,
        bit_depth=BitDepth.BITS_10,
        av1=(11, 1),
    )
    for _ in range(3):
        c.on_iterate(frame)
    c.on_complete()
    stream = probe(tmp_path / "out.mkv")
    assert stream["pix_fmt"] == "yuv420p10le" and stream["profile"] == "Main"
    decoded = decode_planar(tmp_path / "out.mkv", "yuv420p10le")
    row = decoded[: 128 * 96].reshape(96, 128)[48].astype(np.float64)
    # Codes of a one-code-per-pixel ramp survive to within two: 8 bits would make steps of 4.
    ramp = np.round(y[48] * (235 - 16) * 4 + 64)
    assert np.abs(row - ramp).max() <= 2


@pytest.mark.parametrize(
    ("encoder", "depth", "planes_layout", "message"),
    [
        (
            VideoEncoder.AV1,
            BitDepth.BITS_12,
            (1, 1),
            "libsvtav1 encoder cannot write yuv420p12le",
        ),
        (
            VideoEncoder.AV1,
            BitDepth.BITS_10,
            (1, 0),
            "libsvtav1 encoder cannot write yuv422p10le",
        ),
        (
            VideoEncoder.AV1,
            BitDepth.BITS_8,
            (0, 0),
            "libsvtav1 encoder cannot write yuv444p",
        ),
        (
            VideoEncoder.PRORES,
            BitDepth.BITS_12,
            (1, 0),
            "prores_ks encoder cannot write yuv422p12le",
        ),
        (
            VideoEncoder.PRORES,
            BitDepth.BITS_8,
            (1, 0),
            "prores_ks encoder cannot write yuv422p;",
        ),
        (
            VideoEncoder.PRORES,
            BitDepth.AUTO,
            (1, 1),
            "prores_ks encoder cannot write yuv420p10le",
        ),
        (
            VideoEncoder.H265_NVENC,
            BitDepth.BITS_12,
            (1, 1),
            "hevc_nvenc encoder cannot write yuv420p12le",
        ),
        (
            VideoEncoder.AV1_NVENC,
            BitDepth.BITS_10,
            (0, 0),
            "av1_nvenc encoder cannot write yuv444p10le",
        ),
    ],
)
def test_encoders_refuse_what_they_cannot_write(
    tmp_path: Path,
    encoder: VideoEncoder,
    depth: BitDepth,
    planes_layout: tuple[int, int],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(node, "nvidia", _Nvidia(True))
    context = Context()
    c = collector(
        tmp_path,
        VideoFormat.MKV,
        encoder,
        mode=VideoInputMode.PLANES,
        bit_depth=depth,
        context=context,
    )
    with pytest.raises(ValueError, match=message):
        c.on_iterate(planes(planes_layout)[0])
    for cleanup in context.cleanups:
        cleanup()


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        (ChromaLocation.LEFT, "left"),
        (ChromaLocation.TOP_LEFT, "topleft"),
        (ChromaLocation.CENTER, "center"),
    ],
)
def test_av1_chroma_location_in_matroska(
    tmp_path: Path, location: ChromaLocation, expected: str
):
    path, _ = save_rgb(
        tmp_path,
        VideoFormat.MKV,
        VideoEncoder.AV1,
        choices=(Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO, location),
    )
    assert probe(path)["chroma_location"] == expected


def test_av1_writes_full_range(tmp_path: Path):
    white = np.ones((192, 256, 3), np.float32)
    c = collector(
        tmp_path,
        VideoFormat.MKV,
        VideoEncoder.AV1,
        choices=(
            Matrix.BT709,
            Transfer.BT709,
            Primaries.BT709,
            Range.FULL,
            ChromaLocation.AUTO,
        ),
        av1=(11, 10),
    )
    for _ in range(2):
        c.on_iterate((white, None, None, None))
    c.on_complete()
    path = tmp_path / "out.mkv"
    assert_tags(probe(path), "bt709", "bt709", "bt709", "pc")
    y = np.frombuffer(
        subprocess.run(
            [FFMPEG, "-v", "error", "-i", str(path), "-f", "rawvideo", "-"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
        ).stdout,
        dtype=np.uint8,
    )[: 192 * 256]
    assert y.min() >= 253


def test_svt_prints_nothing_that_gets_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    warnings: list[str] = []

    class Log:
        def __getattr__(self, name: str):
            def record(message: object, *args: object, **_: object) -> None:
                if name == "warning":
                    warnings.append(str(message))

            return record

    monkeypatch.delenv("SVT_LOG", raising=False)
    monkeypatch.setattr(node, "logger", Log())
    save_rgb(tmp_path, VideoFormat.MKV, VideoEncoder.AV1)
    assert warnings == []


def test_hdr10_metadata_round_trips_through_av1(tmp_path: Path):
    colour = ColourInfo(
        matrix=Matrix.BT2020_NCL,
        transfer=Transfer.PQ,
        primaries=Primaries.BT2020,
        range=Range.LIMITED,
        chroma_location=ChromaLocation.TOP_LEFT,
        bit_depth=10,
        luma_size=(128, 96),
        chroma_size=(64, 48),
        hdr=HDR,
    )
    c = collector(
        tmp_path,
        VideoFormat.MKV,
        VideoEncoder.AV1,
        mode=VideoInputMode.PLANES,
        colour=colour,
    )
    for frame in planes((1, 1)):
        c.on_iterate(frame)
    c.on_complete()
    path = tmp_path / "out.mkv"
    stream = probe(path)
    assert stream["pix_fmt"] == "yuv420p10le"
    assert_tags(stream, "bt2020nc", "bt2020", "smpte2084", "tv")
    assert stream["chroma_location"] == "topleft"
    reader = VideoLoader(
        path,
        FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE),
        ReadSettings(VideoMode.PLANES, False, AUTO),
    )
    assert reader.colour is not None and reader.colour.hdr == HDR


@pytest.mark.parametrize(
    ("profile", "name", "tag", "pix_fmt"),
    [
        (ProResProfile.PROXY, "Proxy", "apco", "yuv422p10le"),
        (ProResProfile.LT, "LT", "apcs", "yuv422p10le"),
        (ProResProfile.STANDARD, "Standard", "apcn", "yuv422p10le"),
        (ProResProfile.HQ, "HQ", "apch", "yuv422p10le"),
        # FFmpeg's decoder gives 4444 frames in 12 bits, whatever was coded.
        (ProResProfile.P4444, "4444", "ap4h", "yuv444p12le"),
        (ProResProfile.P4444_XQ, "XQ", "ap4x", "yuv444p12le"),
    ],
)
def test_prores_profiles(
    tmp_path: Path, profile: ProResProfile, name: str, tag: str, pix_fmt: str
):
    path, frames = save_rgb(
        tmp_path, VideoFormat.MOV, VideoEncoder.PRORES, profile=profile
    )
    stream = probe(path)
    assert stream["codec_name"] == "prores" and stream["profile"] == name
    assert stream["codec_tag_string"] == tag and stream["pix_fmt"] == pix_fmt
    assert stream["nb_read_frames"] == "8"
    assert_tags(stream, *BT601)
    decoded = decode_planar(path, "gbrp16le").reshape(8, 3, 192, 256) / 65535
    expected = np.stack(frames)[..., [1, 0, 2]].transpose(0, 3, 1, 2)
    assert psnr(decoded, expected) > (36 if profile is ProResProfile.PROXY else 42)


@pytest.mark.parametrize("container", [VideoFormat.MOV, VideoFormat.MKV])
@pytest.mark.parametrize(
    ("layout", "profile", "pix_fmt"),
    [
        ((1, 0), ProResProfile.HQ, "yuv422p10le"),
        ((0, 0), ProResProfile.P4444, "yuv444p10le"),
    ],
)
def test_prores_planes_stay_close_to_their_codes(
    tmp_path: Path,
    container: VideoFormat,
    layout: tuple[int, int],
    profile: ProResProfile,
    pix_fmt: str,
):
    frames = planes(layout)
    c = collector(
        tmp_path,
        container,
        VideoEncoder.PRORES,
        mode=VideoInputMode.PLANES,
        profile=profile,
    )
    for frame in frames:
        c.on_iterate(frame)
    c.on_complete()
    path = tmp_path / f"out.{container.ext}"
    stream = probe(path)
    assert stream["pix_fmt"][:7] == pix_fmt[:7] and stream["color_range"] == "tv"
    assert stream["nb_read_frames"] == "4"
    decoded = decode_planar(path, pix_fmt)
    y = decoded[: 96 * 128].reshape(96, 128)
    expected = frames[0][1] * (235 - 16) * 4 + 64
    assert np.abs(y - expected).max() < 8


@pytest.mark.parametrize(
    ("layout", "profile"),
    [((0, 0), ProResProfile.HQ), ((1, 0), ProResProfile.P4444_XQ)],
)
def test_prores_profile_must_match_the_planes_layout(
    tmp_path: Path, layout: tuple[int, int], profile: ProResProfile
):
    context = Context()
    c = collector(
        tmp_path,
        VideoFormat.MOV,
        VideoEncoder.PRORES,
        mode=VideoInputMode.PLANES,
        profile=profile,
        context=context,
    )
    with pytest.raises(ValueError, match=f"ProRes {profile.value} profile cannot hold"):
        c.on_iterate(planes(layout)[0])
    for cleanup in context.cleanups:
        cleanup()


def source(range_: Range, width: int = 256, height: int = 192) -> ColourInfo:
    return ColourInfo(
        matrix=Matrix.BT709,
        transfer=Transfer.BT709,
        primaries=Primaries.BT709,
        range=range_,
        chroma_location=ChromaLocation.LEFT,
        bit_depth=8,
        luma_size=(width, height),
        chroma_size=(width // 2, height // 2),
    )


def test_prores_refuses_a_full_range_choice_and_full_range_planes(tmp_path: Path):
    for mode, colour, choices, frame in (
        (
            VideoInputMode.RGB,
            None,
            (
                Matrix.AUTO,
                Transfer.AUTO,
                Primaries.AUTO,
                Range.FULL,
                ChromaLocation.AUTO,
            ),
            (rgb_frames()[0], None, None, None),
        ),
        (VideoInputMode.PLANES, source(Range.FULL, 128, 96), AUTO, planes((1, 0))[0]),
    ):
        context = Context()
        c = collector(
            tmp_path,
            VideoFormat.MOV,
            VideoEncoder.PRORES,
            mode=mode,
            colour=colour,
            choices=choices,
            context=context,
        )
        with pytest.raises(ValueError, match="ProRes is limited range only"):
            c.on_iterate(frame)
        for cleanup in context.cleanups:
            cleanup()


def test_prores_converts_rgb_from_a_full_range_source_to_limited(tmp_path: Path):
    white = np.ones((192, 256, 3), np.float32)
    c = collector(
        tmp_path, VideoFormat.MOV, VideoEncoder.PRORES, colour=source(Range.FULL)
    )
    for _ in range(2):
        c.on_iterate((white, None, None, None))
    c.on_complete()
    path = tmp_path / "out.mov"
    assert_tags(probe(path), *BT709)
    y = decode_planar(path, "yuv422p10le")[: 192 * 256]
    assert np.abs(y.astype(int) - 940).max() <= 4  # lossy, but nowhere near 1023


@pytest.mark.parametrize(
    ("encoder", "container"),
    [(VideoEncoder.AV1, VideoFormat.MKV), (VideoEncoder.PRORES, VideoFormat.MOV)],
)
@pytest.mark.parametrize(
    ("matrix", "transfer", "primaries", "expected"),
    [
        (Matrix.BT709, Transfer.BT709, Primaries.BT709, BT709),
        (
            Matrix.BT470BG,
            Transfer.SMPTE170M,
            Primaries.BT470BG,
            ("bt470bg", "bt470bg", "smpte170m", "tv"),
        ),
        (
            Matrix.BT2020_NCL,
            Transfer.PQ,
            Primaries.BT2020,
            ("bt2020nc", "bt2020", "smpte2084", "tv"),
        ),
        (
            Matrix.BT2020_NCL,
            Transfer.HLG,
            Primaries.BT2020,
            ("bt2020nc", "bt2020", "arib-std-b67", "tv"),
        ),
    ],
)
def test_colour_choices_keep_their_tags(
    tmp_path: Path,
    encoder: VideoEncoder,
    container: VideoFormat,
    matrix: Matrix,
    transfer: Transfer,
    primaries: Primaries,
    expected: tuple[str, ...],
):
    path, _ = save_rgb(
        tmp_path,
        container,
        encoder,
        choices=(matrix, transfer, primaries, Range.LIMITED, ChromaLocation.AUTO),
        bit_depth=BitDepth.BITS_10,
    )
    stream = probe(path)
    assert stream["pix_fmt"].endswith("10le")
    assert_tags(stream, *expected)


@pytest.mark.parametrize(
    ("encoder", "container", "warned"),
    [
        (VideoEncoder.PRORES, VideoFormat.MOV, True),
        (VideoEncoder.H265_NVENC, VideoFormat.MKV, True),
        (VideoEncoder.AV1_NVENC, VideoFormat.MKV, True),
        (VideoEncoder.AV1, VideoFormat.MKV, False),
        (VideoEncoder.H265, VideoFormat.MKV, False),
    ],
)
def test_encoders_without_hdr_metadata_say_so(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoder: VideoEncoder,
    container: VideoFormat,
    warned: bool,
):
    warnings: list[str] = []

    class Log:
        def warning(self, message: str, *args: object) -> None:
            warnings.append(message % args)

    monkeypatch.setattr(node, "logger", Log())
    monkeypatch.setattr(node, "nvidia", _Nvidia(True))
    colour = ColourInfo(
        matrix=Matrix.BT2020_NCL,
        transfer=Transfer.PQ,
        primaries=Primaries.BT2020,
        range=Range.LIMITED,
        chroma_location=ChromaLocation.TOP_LEFT,
        bit_depth=10,
        luma_size=(256, 192),
        chroma_size=(128, 96),
        hdr=HDR,
    )
    c = collector(tmp_path, container, encoder, colour=colour)
    writer = cast(node.VideoCollector, c).writer
    writer.first_frame((rgb_frames(count=1)[0], None, None, None))
    assert len(warnings) == int(warned)
    if warned:
        assert "without the HDR10" in warnings[0] and encoder.value in warnings[0]


@pytest.mark.parametrize(
    ("encoder", "additional"),
    [
        (VideoEncoder.PRORES, "-profile:v 2"),
        (VideoEncoder.PRORES, "-profile 2"),
        (VideoEncoder.AV1, "-crf 20"),
        (VideoEncoder.AV1, "-preset 3"),
        (VideoEncoder.H264_NVENC, "-cq 10"),
        (VideoEncoder.H264_NVENC, "-preset p7"),
    ],
)
def test_additional_parameters_cannot_repeat_an_input(
    tmp_path: Path,
    encoder: VideoEncoder,
    additional: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(node, "nvidia", _Nvidia(True))
    with pytest.raises(ValueError, match="Duplicate parameter"):
        collector(tmp_path, VideoFormat.MKV, encoder, additional=additional)


def test_additional_parameters_keep_other_keys(tmp_path: Path):
    """H.264's -profile stays the user's, and -svtav1-params joins the HDR options."""
    path, _ = save_rgb(
        tmp_path,
        VideoFormat.MKV,
        VideoEncoder.AV1,
        additional="-svtav1-params tune=0",
    )
    assert probe(path)["codec_name"] == "av1"
    folder = tmp_path / "h264"
    c = save_video_node(
        cast(NodeContext, Context()),
        None,
        folder,
        "out",
        VideoInputMode.RGB,
        None,
        None,
        None,
        Simplicity.ADVANCED,
        VideoFormat.MKV,
        VideoEncoder.H264,
        VideoPreset.ULTRA_FAST,
        23,
        8,
        35,
        ProResProfile.HQ,
        NvencPreset.P4,
        23,
        30,
        "-profile:v baseline",
        SimpleVideoFormat.MP4_H264,
        75,
        10,
        None,
        AudioSettings.AUTO,
        None,
        *AUTO,
        BitDepth.AUTO,
        SitingCorrection.AUTO,
    )
    for frame in rgb_frames(count=2):
        c.on_iterate((frame, None, None, None))
    c.on_complete()
    assert probe(folder / "out.mkv")["profile"] == "Constrained Baseline"


@pytest.mark.parametrize("encoder", NVENC)
def test_nvenc_without_nvidia_fails_before_any_work(
    tmp_path: Path, encoder: VideoEncoder, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(node, "nvidia", _Nvidia(False))
    with pytest.raises(
        ValueError, match=f"The {encoder.value} encoder needs an NVIDIA GPU"
    ):
        collector(tmp_path, VideoFormat.MKV, encoder)
    assert not list(tmp_path.iterdir())


def test_nvenc_options_show_only_with_nvidia():
    """The schema hides NVENC behind a constant of this computer's NVIDIA check."""
    schema = next(
        n
        for n in node.video_frames_group.nodes
        if n.schema_id == "chainner:image:save_video"
    )
    encoder = next(i for i in schema.inputs if i.id == 3)
    options = {o["value"]: o["condition"] for o in cast(Any, encoder).options}
    constant = Condition.const(gpu.nvidia.is_available).to_json()
    for member in NVENC:
        assert constant in options[member.value]["items"]
    for software in (VideoEncoder.AV1, VideoEncoder.PRORES):
        assert options[software.value]["kind"] == "enum"


def test_nvenc_failure_names_ffmpegs_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An NVIDIA card the encoder cannot use (hidden here) is FFmpeg's message, in the run's error."""
    monkeypatch.setattr(node, "nvidia", _Nvidia(True))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    context = Context()
    c = collector(tmp_path, VideoFormat.MKV, VideoEncoder.H264_NVENC, context=context)
    with pytest.raises(RuntimeError, match=r"(?si)FFmpeg stopped.*(nvenc|nvidia|cuda)"):
        for frame in rgb_frames(count=2):
            c.on_iterate((frame, None, None, None))
        c.on_complete()
    for cleanup in context.cleanups:
        cleanup()
    assert not (tmp_path / "out.mkv").exists()


@pytest.mark.parametrize(
    ("encoder", "location", "expected"),
    [
        (VideoEncoder.H264_NVENC, "topleft", "h264_metadata=chroma_sample_loc_type=2"),
        (VideoEncoder.H265_NVENC, "center", "hevc_metadata=chroma_sample_loc_type=1"),
        (VideoEncoder.H264_NVENC, "left", None),
        (
            VideoEncoder.AV1_NVENC,
            "left",
            "av1_metadata=chroma_sample_position=vertical",
        ),
        (
            VideoEncoder.AV1_NVENC,
            "topleft",
            "av1_metadata=chroma_sample_position=colocated",
        ),
        (VideoEncoder.AV1_NVENC, "center", None),
        (VideoEncoder.AV1, "topleft", None),
        (VideoEncoder.H265, "topleft", None),
    ],
)
def test_nvenc_siting_filter(
    tmp_path: Path,
    encoder: VideoEncoder,
    location: str,
    expected: str | None,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(node, "nvidia", _Nvidia(True))
    c = collector(tmp_path, VideoFormat.MKV, encoder)
    writer = cast(node.VideoCollector, c).writer
    params = {
        "vcodec": encoder.value,
        "pix_fmt": "yuv420p",
        "chroma_sample_location": location,
    }
    writer.colour_options(params, 256, 192)
    assert params.get("bsf:v") == expected
    params = {**params, "bsf:v": "user"}
    writer.colour_options(params, 256, 192)
    assert params["bsf:v"] == "user"


@pytest.mark.parametrize("encoder", NVENC)
def test_nvenc_refuses_odd_sizes(
    tmp_path: Path, encoder: VideoEncoder, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(node, "nvidia", _Nvidia(True))
    context = Context()
    c = collector(tmp_path, VideoFormat.MKV, encoder, context=context)
    with pytest.raises(Exception, match="requires an even-number frame resolution"):
        c.on_iterate((np.zeros((195, 257, 3), np.float32), None, None, None))
    for cleanup in context.cleanups:
        cleanup()


def assert_flat_to_the_edge(path: Path, width: int, height: int) -> None:
    """A flat colour (FLAT, in B, G, R) comes back flat, its last column and row too."""
    decoded = decode_planar(path, "gbrp16le").reshape(-1, 3, height, width) / 65535
    blue, green, red = FLAT
    expected = np.array((green, blue, red)).reshape(1, 3, 1, 1)
    assert np.abs(decoded - expected).max() < 0.03


@pytest.mark.parametrize(
    ("encoder", "profile"),
    [
        (VideoEncoder.AV1, ProResProfile.HQ),
        (VideoEncoder.PRORES, ProResProfile.P4444),
        (VideoEncoder.PRORES, ProResProfile.P4444_XQ),
    ],
)
def test_software_encoders_write_odd_sizes(
    tmp_path: Path, encoder: VideoEncoder, profile: ProResProfile
):
    c = collector(tmp_path, VideoFormat.MKV, encoder, profile=profile)
    for _ in range(2):
        c.on_iterate((np.full((97, 129, 3), FLAT, np.float32), None, None, None))
    c.on_complete()
    stream = probe(tmp_path / "out.mkv")
    assert (stream["width"], stream["height"]) == (129, 97)
    assert_flat_to_the_edge(tmp_path / "out.mkv", 129, 97)


@pytest.mark.parametrize("width", [129, 65, 17])
@pytest.mark.parametrize("profile", [ProResProfile.STANDARD, ProResProfile.HQ])
def test_prores_422_refuses_an_odd_width(
    tmp_path: Path, width: int, profile: ProResProfile
):
    """The encoder leaves the last chroma column at code 4, without a word from FFmpeg."""
    context = Context()
    c = collector(
        tmp_path, VideoFormat.MOV, VideoEncoder.PRORES, profile=profile, context=context
    )
    with pytest.raises(ValueError, match=f"odd width \\({width}\\)"):
        c.on_iterate((np.full((96, width, 3), FLAT, np.float32), None, None, None))
    for cleanup in context.cleanups:
        cleanup()
    assert not (tmp_path / "out.mov").exists()


def test_prores_422_planes_refuse_an_odd_width(tmp_path: Path):
    context = Context()
    c = collector(
        tmp_path,
        VideoFormat.MOV,
        VideoEncoder.PRORES,
        mode=VideoInputMode.PLANES,
        context=context,
    )
    with pytest.raises(ValueError, match=r"odd width \(129\)"):
        c.on_iterate(planes((1, 0), width=129, height=97)[0])
    for cleanup in context.cleanups:
        cleanup()


def test_a_run_of_one_chosen_frame_refuses_an_odd_prores_422_width(tmp_path: Path):
    context = Context()
    context.chosen_item = 3
    c = collector(tmp_path, VideoFormat.MOV, VideoEncoder.PRORES, context=context)
    with pytest.raises(ValueError, match=r"odd width \(129\)"):
        c.on_iterate((np.full((97, 129, 3), FLAT, np.float32), None, None, None))
    assert not (tmp_path / "out_f000003.png").exists()


def test_prores_422_keeps_an_even_width_at_an_odd_height(tmp_path: Path):
    c = collector(tmp_path, VideoFormat.MOV, VideoEncoder.PRORES)
    for _ in range(2):
        c.on_iterate((np.full((97, 128, 3), FLAT, np.float32), None, None, None))
    c.on_complete()
    stream = probe(tmp_path / "out.mov")
    assert (stream["width"], stream["height"]) == (128, 97)
    assert_flat_to_the_edge(tmp_path / "out.mov", 128, 97)


@pytest.mark.parametrize(
    ("width", "height"), [(2, 2), (3, 3), (4, 4), (8, 8), (9, 9), (15, 64), (64, 15)]
)
def test_av1_refuses_tiny_frames_before_ffmpeg_starts(
    tmp_path: Path, width: int, height: int
):
    """SVT-AV1 ends FFmpeg with heap corruption or an access violation at such sizes."""
    c = collector(tmp_path, VideoFormat.MKV, VideoEncoder.AV1)
    writer = cast(node.VideoCollector, c).writer
    frame = (np.zeros((height, width, 3), np.float32), None, None, None)
    with pytest.raises(ValueError, match=f"not {width}x{height}"):
        writer.first_frame(frame)
    assert writer.out is None


def test_a_run_of_one_chosen_frame_refuses_a_tiny_av1_frame(tmp_path: Path):
    context = Context()
    context.chosen_item = 3
    c = collector(tmp_path, VideoFormat.MKV, VideoEncoder.AV1, context=context)
    with pytest.raises(ValueError, match="not 8x8"):
        c.on_iterate((np.zeros((8, 8, 3), np.float32), None, None, None))
    assert not (tmp_path / "out_f000003.png").exists()


@pytest.mark.parametrize("depth", [BitDepth.BITS_8, BitDepth.BITS_10])
def test_av1_writes_noise_at_the_smallest_size_it_takes(
    tmp_path: Path, depth: BitDepth
):
    rng = np.random.default_rng(16)
    c = collector(
        tmp_path, VideoFormat.MKV, VideoEncoder.AV1, bit_depth=depth, av1=(8, 1)
    )
    for _ in range(8):
        c.on_iterate((rng.random((16, 16, 3), dtype=np.float32), None, None, None))
    c.on_complete()
    stream = probe(tmp_path / "out.mkv")
    assert (stream["width"], stream["height"]) == (16, 16)
    assert stream["nb_read_frames"] == "8"


def test_encoder_inputs_line_up_with_the_node_parameters():
    """The engine passes inputs by position: each encoder input reaches its own parameter."""
    schema = next(
        n
        for n in node.video_frames_group.nodes
        if n.schema_id == "chainner:image:save_video"
    )
    names = list(inspect.signature(save_video_node).parameters)
    # The node's first parameter is its NodeContext, which the schema does not list.
    by_id = dict(zip((int(i.id) for i in schema.inputs), names[1:], strict=True))
    assert {key: by_id[key] for key in (8, 9, 31, 32, 33, 34, 35, 36, 13)} == {
        8: "video_preset",
        9: "crf",
        31: "av1_preset",
        32: "av1_crf",
        33: "prores_profile",
        34: "nvenc_preset",
        35: "nvenc_cq",
        36: "av1_nvenc_cq",
        13: "additional_parameters",
    }


def test_container_tables():
    assert VideoEncoder.AV1.formats == (
        VideoFormat.MKV,
        VideoFormat.MP4,
        VideoFormat.WEBM,
    )
    assert VideoEncoder.PRORES.formats == (VideoFormat.MKV, VideoFormat.MOV)
    assert VideoEncoder.H264_NVENC.formats == (
        VideoFormat.MKV,
        VideoFormat.MP4,
        VideoFormat.MOV,
        VideoFormat.AVI,
    )
    assert VideoEncoder.H265_NVENC.formats == (
        VideoFormat.MKV,
        VideoFormat.MP4,
        VideoFormat.MOV,
    )
    assert VideoEncoder.AV1_NVENC.formats == (
        VideoFormat.MKV,
        VideoFormat.MP4,
        VideoFormat.WEBM,
    )


def _audio_source(folder: Path) -> Path:
    path = folder / "audio.mkv"
    subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x48:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "ffv1",
            "-c:a",
            "flac",
            str(path),
        ],
        stdin=subprocess.DEVNULL,
        check=True,
        capture_output=True,
    )
    return path


@pytest.mark.parametrize(
    ("encoder", "container", "codec"),
    [
        (VideoEncoder.PRORES, VideoFormat.MOV, "prores"),
        (VideoEncoder.AV1, VideoFormat.MP4, "av1"),
        (VideoEncoder.AV1, VideoFormat.WEBM, "av1"),
    ],
)
def test_audio_mux_keeps_the_video(
    tmp_path: Path, encoder: VideoEncoder, container: VideoFormat, codec: str
):
    source_path = _audio_source(tmp_path)
    audio = VideoLoader(
        source_path, FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE)
    ).get_audio_stream()
    path, _ = save_rgb(tmp_path, container, encoder, audio=audio)
    output = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    ).stdout
    streams = json.loads(output)["streams"]
    assert {s["codec_type"] for s in streams} == {"video", "audio"}
    assert next(s for s in streams if s["codec_type"] == "video")["codec_name"] == codec


def test_run_of_one_chosen_frame_writes_prores_depth(tmp_path: Path):
    context = Context()
    context.chosen_item = 3
    c = collector(tmp_path, VideoFormat.MOV, VideoEncoder.PRORES, context=context)
    c.on_iterate((rgb_frames(count=1)[0], None, None, None))
    c.on_complete()
    png = (tmp_path / "out_f000003.png").read_bytes()
    assert png[24] == 16  # the video would hold more than 8 bits
    assert not (tmp_path / "out.mov").exists()


@pytest.mark.skipif(
    not gpu.nvidia.is_available or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
    reason="NVENC needs a visible NVIDIA GPU",
)
@pytest.mark.parametrize(
    ("encoder", "container", "depth", "pix_fmt", "codec"),
    [
        (VideoEncoder.H264_NVENC, VideoFormat.MKV, BitDepth.AUTO, "yuv420p", "h264"),
        (VideoEncoder.H264_NVENC, VideoFormat.MOV, BitDepth.AUTO, "yuv420p", "h264"),
        (VideoEncoder.H264_NVENC, VideoFormat.AVI, BitDepth.AUTO, "yuv420p", "h264"),
        (VideoEncoder.H265_NVENC, VideoFormat.MP4, BitDepth.AUTO, "yuv420p", "hevc"),
        (
            VideoEncoder.H265_NVENC,
            VideoFormat.MKV,
            BitDepth.BITS_10,
            "yuv420p10le",
            "hevc",
        ),
        (VideoEncoder.AV1_NVENC, VideoFormat.MP4, BitDepth.AUTO, "yuv420p", "av1"),
        (
            VideoEncoder.AV1_NVENC,
            VideoFormat.WEBM,
            BitDepth.BITS_10,
            "yuv420p10le",
            "av1",
        ),
    ],
)
def test_nvenc_clip_decodes_to_what_went_in(
    tmp_path: Path,
    encoder: VideoEncoder,
    container: VideoFormat,
    depth: BitDepth,
    pix_fmt: str,
    codec: str,
    monkeypatch: pytest.MonkeyPatch,
):
    path, frames = save_rgb(tmp_path, container, encoder, bit_depth=depth)
    stream = probe(path)
    assert stream["codec_name"] == codec and stream["pix_fmt"] == pix_fmt
    assert stream["nb_read_frames"] == "8"
    assert_tags(stream, *BT601)
    decoded = read_rgb(path, monkeypatch, False)
    assert psnr(decoded / 255, np.stack(frames)) > 34


@pytest.mark.skipif(
    not gpu.nvidia.is_available or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
    reason="NVENC needs a visible NVIDIA GPU",
)
@pytest.mark.parametrize(
    ("encoder", "layout", "depth", "pix_fmt", "bits"),
    [
        (VideoEncoder.H265_NVENC, (1, 1), BitDepth.BITS_10, "yuv420p10le", 10),
        (VideoEncoder.AV1_NVENC, (1, 1), BitDepth.BITS_10, "yuv420p10le", 10),
        (VideoEncoder.H265_NVENC, (0, 0), BitDepth.BITS_10, "yuv444p10le", 10),
        (VideoEncoder.H264_NVENC, (1, 1), BitDepth.BITS_8, "yuv420p", 8),
        (VideoEncoder.H265_NVENC, (1, 0), BitDepth.BITS_8, "yuv422p", 8),
        (VideoEncoder.H265_NVENC, (1, 0), BitDepth.BITS_10, "yuv422p10le", 10),
    ],
)
def test_nvenc_planes_keep_their_codes_close(
    tmp_path: Path,
    encoder: VideoEncoder,
    layout: tuple[int, int],
    depth: BitDepth,
    pix_fmt: str,
    bits: int,
):
    frames = planes(layout, width=320, height=180, count=4)
    c = collector(
        tmp_path,
        VideoFormat.MKV,
        encoder,
        mode=VideoInputMode.PLANES,
        bit_depth=depth,
        nvenc=(NvencPreset.P4, 18, 18),
    )
    for frame in frames:
        c.on_iterate(frame)
    c.on_complete()
    path = tmp_path / "out.mkv"
    stream = probe(path)
    assert stream["pix_fmt"] == pix_fmt
    assert stream["color_range"] == "tv"
    decoded = decode_planar(path, pix_fmt)
    y = decoded[: 180 * 320].reshape(180, 320).astype(np.float64)
    unit = 1 << (bits - 8)
    expected = frames[0][1] * (235 - 16) * unit + 16 * unit
    assert np.abs(y - expected).mean() < 2 * unit


@pytest.mark.skipif(
    not gpu.nvidia.is_available or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
    reason="NVENC needs a visible NVIDIA GPU",
)
@pytest.mark.parametrize("encoder", NVENC)
@pytest.mark.parametrize("container", [VideoFormat.MP4, VideoFormat.MKV])
@pytest.mark.parametrize(
    ("location", "expected"),
    [(ChromaLocation.TOP_LEFT, "topleft"), (ChromaLocation.LEFT, "left")],
)
def test_nvenc_chroma_siting_reaches_the_stream(
    tmp_path: Path,
    encoder: VideoEncoder,
    container: VideoFormat,
    location: ChromaLocation,
    expected: str,
):
    path, _ = save_rgb(
        tmp_path,
        container,
        encoder,
        choices=(Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO, location),
    )
    stream = probe(path)
    assert stream["chroma_location"] == expected
    assert stream["nb_read_frames"] == "8"


@pytest.mark.skipif(
    not gpu.nvidia.is_available or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
    reason="NVENC needs a visible NVIDIA GPU",
)
@pytest.mark.parametrize("encoder", NVENC)
def test_nvenc_cq_reaches_the_encoder(tmp_path: Path, encoder: VideoEncoder):
    sizes = {}
    for cq in (1, 51):
        path, _ = save_rgb(
            tmp_path / str(cq),
            VideoFormat.MKV,
            encoder,
            nvenc=(NvencPreset.P4, cq, cq),
        )
        sizes[cq] = path.stat().st_size
    assert sizes[1] > 2 * sizes[51]
