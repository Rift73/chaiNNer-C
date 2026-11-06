"""Save Video's audio where FFmpeg gives no error or the wrong one. libopus takes only
the Vorbis channel layouts, so the 5.1(side) of AC-3 and E-AC-3 (DVD, Blu-ray) failed
WebM's Auto and Transcode. FFmpeg's MOV muxer stores WMA under a tag no reader knows,
without an error, so Auto and Copy left a video whose audio cannot be played."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from scratch_storage import NO_FFMPEG, STORAGE

# The node's modules load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import NodeContext
from nodes.impl.colour_info import ChromaLocation, Matrix, Primaries, Range, Transfer
from nodes.impl.ffmpeg import FFMpegEnv, get_executable_path
from nodes.impl.video import VideoLoader
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
ENV = FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE)

pytestmark = pytest.mark.skipif(not Path(FFMPEG).exists(), reason=NO_FFMPEG)


class Context:
    chosen_item = None
    node_id = "save"
    storage_dir = STORAGE

    def add_cleanup(self, _fn: Any) -> None:
        pass


def source(path: Path, layout: str, *codec: str) -> Path:
    subprocess.run(
        [
            FFMPEG,
            *("-hide_banner", "-loglevel", "error", "-y"),
            *("-f", "lavfi", "-i", "color=c=gray:size=64x48:rate=30"),
            *("-f", "lavfi", "-i"),
            f"sine=frequency=440:sample_rate=48000,aformat=channel_layouts={layout}",
            *("-t", "1", "-c:v", "libx264", "-preset", "ultrafast", *codec, str(path)),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    return path


def save(folder: Path, src: Path, container: VideoFormat, audio: AudioSettings) -> Path:
    encoder = VideoEncoder.VP9 if container == VideoFormat.WEBM else VideoEncoder.H264
    collector = save_video_node(
        cast(NodeContext, Context()),
        None,
        folder,
        "out",
        VideoInputMode.RGB,
        None,
        None,
        None,
        Simplicity.ADVANCED,
        container,
        encoder,
        VideoPreset.ULTRA_FAST,
        23,
        *(8, 35, ProResProfile.HQ, NvencPreset.P4, 23, 30),
        None,
        SimpleVideoFormat.MP4_H264,
        75,
        30.0,
        VideoLoader(src, ENV).get_audio_stream(),
        audio,
        *(None, Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO),
        *(ChromaLocation.AUTO, BitDepth.AUTO, SitingCorrection.AUTO),
    )
    for value in np.linspace(0, 1, 5, dtype=np.float32):
        collector.on_iterate(
            (np.full((48, 64, 3), value, np.float32), None, None, None)
        )
    collector.on_complete()
    return folder / f"out.{container.ext}"


def audio_streams(path: Path) -> list[dict[str, Any]]:
    probe = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    return json.loads(probe.stdout)["streams"]


@pytest.mark.parametrize("audio", [AudioSettings.AUTO, AudioSettings.TRANSCODE])
@pytest.mark.parametrize(
    ("name", "layout", "codec", "channels"),
    [
        ("ac3.mkv", "5.1(side)", ("-c:a", "ac3"), 6),
        ("pcm-7.1-wide.mkv", "7.1(wide)", ("-c:a", "pcm_s16le"), 8),
        ("pcm-2.1.mkv", "2.1", ("-c:a", "pcm_s16le"), 2),
        ("pcm-16.mkv", "hexadecagonal", ("-c:a", "pcm_s16le"), 8),
    ],
)
def test_webm_takes_any_channel_layout(
    tmp_path: Path,
    name: str,
    layout: str,
    codec: tuple[str, ...],
    channels: int,
    audio: AudioSettings,
):
    src = source(tmp_path / name, layout, *codec)
    (stream,) = audio_streams(save(tmp_path, src, VideoFormat.WEBM, audio))
    assert (stream["codec_name"], stream["channels"]) == ("opus", channels)


def test_auto_transcodes_wma_for_mov(tmp_path: Path):
    src = source(tmp_path / "wma.mkv", "stereo", "-c:a", "wmav2")
    (stream,) = audio_streams(save(tmp_path, src, VideoFormat.MOV, AudioSettings.AUTO))
    assert stream["codec_name"] == "aac"


def test_copy_refuses_wma_for_mov(tmp_path: Path):
    src = source(tmp_path / "wma.mkv", "stereo", "-c:a", "wmav2")
    with pytest.raises(
        RuntimeError, match=r"could not copy the wmav2 audio into the \.mov"
    ):
        save(tmp_path, src, VideoFormat.MOV, AudioSettings.COPY)
    assert not (tmp_path / "out.mov").exists()


def test_wma_stays_a_copy_where_it_can_be_read(tmp_path: Path):
    src = source(tmp_path / "wma.mkv", "stereo", "-c:a", "wmav2")
    (stream,) = audio_streams(save(tmp_path, src, VideoFormat.MKV, AudioSettings.AUTO))
    assert stream["codec_name"] == "wmav2"
