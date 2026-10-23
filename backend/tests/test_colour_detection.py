"""Colour detection from ffprobe stream entries (colour_info.detect_colour_info): tags
first, then HDR side data, cross-field inference and mpv's size guesses; warnings for
implausible tags. Synthetic entries for every rule, and real files FFmpeg tags."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import ffmpeg
import pytest

from nodes.impl.colour_info import (
    ChromaLocation,
    HdrInfo,
    Matrix,
    Primaries,
    Range,
    Transfer,
    detect_colour_info,
    option_value,
    pixel_layout,
)

STORAGE = Path(os.environ.get("APPDATA", "")) / "chaiNNer" / "backend-storage"
FFMPEG = STORAGE / "ffmpeg" / "ffmpeg.exe"
FFPROBE = STORAGE / "ffmpeg" / "ffprobe.exe"


def entry(**fields: object) -> dict[str, object]:
    base: dict[str, object] = {
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": 1920,
        "height": 1080,
    }
    base.update(fields)
    return base


def test_pixel_layouts():
    assert pixel_layout("yuv420p10le").planar == "yuv420p10le"
    assert pixel_layout("yuv420p10be").planar == "yuv420p10le"
    assert pixel_layout("yuvj422p").planar == "yuvj422p"
    assert pixel_layout("yuvj422p").jpeg
    assert pixel_layout("yuva444p12le").planar == "yuv444p12le"
    p010 = pixel_layout("p010le")
    assert (p010.kind, p010.bit_depth, p010.subsampling, p010.planar) == (
        "yuv",
        10,
        (1, 1),
        "yuv420p10le",
    )
    assert pixel_layout("nv12").planar == "yuv420p"
    assert pixel_layout("yuyv422").planar == "yuv422p"
    assert pixel_layout("yuv410p").subsampling == (2, 2)
    assert pixel_layout("gray16be").planar == "gray16le"
    assert pixel_layout("gray").planar == "gray"
    assert pixel_layout("grayf32le").planar is None
    assert (pixel_layout("gbrp10le").kind, pixel_layout("gbrp10le").bit_depth) == (
        "rgb",
        10,
    )
    assert pixel_layout("rgb48le").bit_depth == 16
    assert pixel_layout("bgr24").kind == "rgb"
    assert pixel_layout("bayer_rggb8").kind == "other"


def test_tags_are_trusted_and_marked_file():
    info, warnings = detect_colour_info(
        entry(
            pix_fmt="yuv422p10le",
            width=720,
            height=576,
            color_space="bt709",
            color_transfer="smpte170m",
            color_primaries="bt470bg",
            color_range="pc",
            chroma_location="topleft",
        )
    )
    assert (info.matrix, info.transfer, info.primaries, info.range) == (
        Matrix.BT709,
        Transfer.SMPTE170M,
        Primaries.BT470BG,
        Range.FULL,
    )
    assert info.chroma_location is ChromaLocation.TOP_LEFT
    assert (info.bit_depth, info.luma_size, info.chroma_size) == (
        10,
        (720, 576),
        (360, 576),
    )
    assert all(info.origin_of(name) == "file" for name, _ in info.origin)
    assert warnings == []


@pytest.mark.parametrize(
    ("width", "height", "matrix", "primaries"),
    [
        (1920, 1080, Matrix.BT709, Primaries.BT709),
        (1280, 720, Matrix.BT709, Primaries.BT709),
        (720, 578, Matrix.BT709, Primaries.BT709),
        (720, 576, Matrix.BT470BG, Primaries.BT470BG),
        (720, 480, Matrix.SMPTE170M, Primaries.SMPTE170M),
        (720, 486, Matrix.SMPTE170M, Primaries.SMPTE170M),
        (640, 360, Matrix.SMPTE170M, Primaries.BT709),
    ],
)
def test_untagged_yuv_is_guessed_from_its_size(
    width: int, height: int, matrix: Matrix, primaries: Primaries
):
    info, warnings = detect_colour_info(entry(width=width, height=height))
    assert (info.matrix, info.primaries) == (matrix, primaries)
    assert (info.transfer, info.range, info.chroma_location) == (
        Transfer.BT709,
        Range.LIMITED,
        ChromaLocation.LEFT,
    )
    assert all(info.origin_of(name) == "guessed" for name, _ in info.origin)
    assert warnings == []


def test_untagged_values_count_as_missing():
    info, _ = detect_colour_info(
        entry(
            color_space="unknown", color_range="unknown", chroma_location="unspecified"
        )
    )
    assert info.origin_of("matrix") == "guessed" and info.matrix is Matrix.BT709


def test_rgb_formats_are_identity_coded_full_range_srgb():
    info, _ = detect_colour_info(entry(pix_fmt="gbrp", codec_name="ffv1"))
    assert (info.matrix, info.range, info.primaries, info.transfer) == (
        Matrix.RGB,
        Range.FULL,
        Primaries.BT709,
        Transfer.SRGB,
    )
    assert info.subsampling == (0, 0)


def test_full_range_formats_and_codecs_sit_chroma_in_the_centre():
    jpeg, _ = detect_colour_info(entry(pix_fmt="yuvj420p", codec_name="mjpeg"))
    assert (jpeg.range, jpeg.chroma_location) == (Range.FULL, ChromaLocation.CENTER)
    tagged_full, _ = detect_colour_info(entry(color_range="pc"))
    assert tagged_full.chroma_location is ChromaLocation.CENTER
    mpeg1, _ = detect_colour_info(entry(codec_name="mpeg1video"))
    assert (mpeg1.range, mpeg1.chroma_location) == (
        Range.LIMITED,
        ChromaLocation.CENTER,
    )
    # AV1's top-left default applies only when the file says so.
    av1, _ = detect_colour_info(entry(codec_name="av1"))
    assert av1.chroma_location is ChromaLocation.LEFT
    gray, _ = detect_colour_info(entry(pix_fmt="gray10le"))
    assert (gray.range, gray.bit_depth) == (Range.FULL, 10)


MASTERING = {
    "side_data_type": "Mastering display metadata",
    "red_x": "35400/50000",
    "red_y": "14600/50000",
    "green_x": "8500/50000",
    "green_y": "39850/50000",
    "blue_x": "6550/50000",
    "blue_y": "2300/50000",
    "white_point_x": "15635/50000",
    "white_point_y": "16450/50000",
    "min_luminance": "50/10000",
    "max_luminance": "10000000/10000",
}
LIGHT_LEVEL = {
    "side_data_type": "Content light level metadata",
    "max_content": 1000,
    "max_average": 400,
}


def test_hdr_side_data_implies_pq_and_bt2020():
    info, warnings = detect_colour_info(
        entry(pix_fmt="yuv420p10le", side_data_list=[MASTERING, LIGHT_LEVEL])
    )
    assert (info.transfer, info.primaries, info.matrix) == (
        Transfer.PQ,
        Primaries.BT2020,
        Matrix.BT2020_NCL,
    )
    assert info.hdr == HdrInfo(
        mastering_display=(
            "G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,50)"
        ),
        max_cll=1000,
        max_fall=400,
    )
    assert info.origin_of("transfer") == "guessed"
    assert warnings == []
    # A tagged transfer wins over the side data's PQ.
    hlg, _ = detect_colour_info(
        entry(
            pix_fmt="yuv420p10le",
            color_transfer="arib-std-b67",
            side_data_list=[LIGHT_LEVEL],
        )
    )
    assert (hlg.transfer, hlg.primaries) == (Transfer.HLG, Primaries.BT2020)
    assert hlg.hdr == HdrInfo(max_cll=1000, max_fall=400)


@pytest.mark.parametrize(
    "tags",
    [
        {"color_primaries": "bt2020"},
        {"color_transfer": "smpte2084"},
        {"color_transfer": "arib-std-b67"},
        {"color_space": "bt2020nc"},
    ],
)
def test_bt2020_or_hdr_tags_imply_the_bt2020_matrix_and_primaries(
    tags: dict[str, str],
):
    info, _ = detect_colour_info(entry(pix_fmt="yuv420p10le", **tags))
    assert (info.matrix, info.primaries) == (Matrix.BT2020_NCL, Primaries.BT2020)


def test_implausible_combinations_warn_and_name_the_override():
    _, hdr8 = detect_colour_info(entry(color_transfer="smpte2084"))
    assert len(hdr8) == 1 and "8 bits" in hdr8[0] and "set Transfer" in hdr8[0]
    _, sd2020 = detect_colour_info(
        entry(width=720, height=480, color_primaries="bt2020", pix_fmt="yuv420p10le")
    )
    assert any("set Primaries" in w and "720x480" in w for w in sd2020)
    assert any("set Matrix" in w for w in sd2020)
    _, pal709 = detect_colour_info(
        entry(codec_name="mpeg2video", width=720, height=576, color_space="bt709")
    )
    assert len(pal709) == 1 and "MPEG-2" in pal709[0] and "set Matrix" in pal709[0]
    # A guessed BT.709 is not a tag, so it does not warn.
    _, guessed = detect_colour_info(entry(codec_name="mpeg2video", height=578))
    assert guessed == []


def test_unsupported_tags_are_guessed_with_a_warning():
    info, warnings = detect_colour_info(entry(color_transfer="log100"))
    assert info.transfer is Transfer.BT709 and info.origin_of("transfer") == "guessed"
    assert len(warnings) == 1 and "log100" in warnings[0]


def make_video(path: Path, *arguments: str) -> dict[str, object]:
    subprocess.run(
        [
            str(FFMPEG),
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=5:duration=0.4",
            *arguments,
            str(path),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    probe = ffmpeg.probe(str(path), cmd=str(FFPROBE))
    return next(s for s in probe["streams"] if s["codec_type"] == "video")


@pytest.mark.skipif(
    not FFMPEG.exists(), reason="the integrated FFmpeg is not downloaded"
)
def test_real_files_report_their_tags(tmp_path: Path):
    tagged = make_video(
        tmp_path / "tagged.mkv",
        *("-vf", "setparams=color_primaries=bt2020:color_trc=arib-std-b67"),
        *("-c:v", "ffv1", "-pix_fmt", "yuv422p10le"),
        *("-colorspace", "bt2020nc", "-color_primaries", "bt2020"),
        *("-color_trc", "arib-std-b67", "-color_range", "pc"),
        *("-chroma_sample_location", "topleft"),
    )
    info, warnings = detect_colour_info(tagged)
    assert (info.matrix, info.primaries, info.transfer, info.range) == (
        Matrix.BT2020_NCL,
        Primaries.BT2020,
        Transfer.HLG,
        Range.FULL,
    )
    assert info.chroma_location is ChromaLocation.TOP_LEFT
    assert (info.bit_depth, info.chroma_size) == (10, (160, 240))
    assert all(info.origin_of(name) == "file" for name, _ in info.origin)
    # 320x240 is SD, which BT.2020 rarely is.
    assert len(warnings) == 2

    untagged = make_video(
        tmp_path / "untagged.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p"
    )
    info, warnings = detect_colour_info(untagged)
    assert (info.matrix, info.primaries, info.range) == (
        Matrix.SMPTE170M,
        Primaries.BT709,
        Range.LIMITED,
    )
    assert info.origin_of("matrix") == "guessed" and warnings == []


def test_option_values_follow_ffmpeg_s_option_names():
    assert option_value(Matrix.RGB) == "rgb"
    assert option_value(Transfer.BT470M) == "gamma22"
    assert option_value(Transfer.BT470BG) == "gamma28"
    assert option_value(Matrix.BT2020_NCL) == "bt2020nc"
    assert option_value(ChromaLocation.TOP_LEFT) == "topleft"
