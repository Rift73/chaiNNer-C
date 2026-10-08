"""Save Video muxes the source's audio with chaiNNer's own FFmpeg, so a PC without
FFmpeg on PATH still gets it; "Auto" transcodes the audio to AAC when the container
cannot hold a copy of it; and audio the mux cannot carry fails the run instead of
leaving a silent video (upstream chaiNNer #3331)."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest

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


class Context:
    """The two NodeContext members Save Video uses."""

    def __init__(self):
        self.storage_dir = STORAGE
        self.cleanups = []

    def add_cleanup(self, fn: Callable[[], None]):
        self.cleanups.append(fn)


def ffmpeg(*args: str) -> bytes:
    return subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    ).stdout


def source(
    tmp_path: Path, name: str, audio_codec: str | None, channels: int = 2
) -> Path:
    """One second of H.264 video with a tone in `audio_codec`, or no audio for None."""
    path = tmp_path / name
    video = ["-f", "lavfi", "-i", "testsrc2=size=64x48:rate=30"]
    codecs = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio_codec is None:
        ffmpeg(*video, "-t", "1", *codecs, str(path))
        return path
    audio = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    codecs += ["-c:a", audio_codec, "-ac", str(channels)]
    ffmpeg(*video, *audio, "-t", "1", *codecs, str(path))
    return path


def streams(path: Path) -> list[str]:
    probe = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    ).stdout
    return [
        f"{s['codec_type']}:{s['codec_name']}" for s in json.loads(probe)["streams"]
    ]


def audio_packets(path: Path) -> str:
    """MD5 of the first audio stream's packets: equal only for a stream copy."""
    return ffmpeg(
        "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "md5", "-"
    ).decode()


def save(tmp_path: Path, src: Path, container: VideoFormat, audio: AudioSettings):
    encoder = VideoEncoder.VP9 if container == VideoFormat.WEBM else VideoEncoder.H264
    loader = VideoLoader(src, FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE))
    collector = save_video_node(
        cast(NodeContext, Context()),
        None,
        tmp_path,
        "out",
        Simplicity.ADVANCED,
        container,
        encoder,
        VideoPreset.ULTRA_FAST,
        23,
        None,
        SimpleVideoFormat.MP4_H264,
        75,
        30.0,
        loader.get_audio_stream(),
        audio,
    )
    for value in np.linspace(0, 1, 30, dtype=np.float32):
        collector.on_iterate(np.full((48, 64, 3), value, np.float32))
    collector.on_complete()


def saved(tmp_path: Path, container: VideoFormat) -> Path:
    """The saved video, once no temporary file of the audio step is left behind."""
    assert not list(tmp_path.glob("out_av.*"))
    return tmp_path / f"out.{container.ext}"


@pytest.fixture(autouse=True)
def no_ffmpeg_on_path(monkeypatch: pytest.MonkeyPatch):
    # The release zip's normal case: the only FFmpeg is chaiNNer's own.
    kept = [
        d
        for d in os.environ["PATH"].split(os.pathsep)
        if not any(Path(d, n).exists() for n in ("ffmpeg", "ffmpeg.exe"))
    ]
    monkeypatch.setenv("PATH", os.pathsep.join(kept))


@pytest.mark.parametrize("audio", [AudioSettings.AUTO, AudioSettings.COPY])
@pytest.mark.parametrize("container", [VideoFormat.MP4, VideoFormat.MKV])
def test_copyable_audio_is_copied_unchanged(
    tmp_path: Path, container: VideoFormat, audio: AudioSettings
):
    src = source(tmp_path, "src.mp4", "aac")
    save(tmp_path, src, container, audio)
    out = saved(tmp_path, container)
    assert streams(out) == ["video:h264", "audio:aac"]
    assert audio_packets(out) == audio_packets(src)


def test_auto_transcodes_audio_the_container_cannot_copy(tmp_path: Path):
    # MP4 cannot hold PCM on the integrated FFmpeg 5.1.2.
    src = source(tmp_path, "src.mov", "pcm_s16le")
    save(tmp_path, src, VideoFormat.MP4, AudioSettings.AUTO)
    assert streams(saved(tmp_path, VideoFormat.MP4)) == ["video:h264", "audio:aac"]


def test_auto_copies_pcm_where_the_container_holds_it(tmp_path: Path):
    src = source(tmp_path, "src.mov", "pcm_s16le")
    save(tmp_path, src, VideoFormat.MKV, AudioSettings.AUTO)
    out = saved(tmp_path, VideoFormat.MKV)
    assert streams(out) == ["video:h264", "audio:pcm_s16le"]
    assert audio_packets(out) == audio_packets(src)


def test_copy_the_container_cannot_hold_is_an_error(tmp_path: Path):
    # Copy never transcodes; the run fails and keeps the video without audio.
    src = source(tmp_path, "src.mov", "pcm_s16le")
    message = r"copy the pcm_s16le audio into the \.mp4 file\. Set Audio to Auto or"
    with pytest.raises(RuntimeError, match=message):
        save(tmp_path, src, VideoFormat.MP4, AudioSettings.COPY)
    assert streams(saved(tmp_path, VideoFormat.MP4)) == ["video:h264"]


@pytest.mark.parametrize("container", [VideoFormat.MP4, VideoFormat.MKV])
def test_transcode_writes_aac(tmp_path: Path, container: VideoFormat):
    src = source(tmp_path, "src.mp4", "aac")
    save(tmp_path, src, container, AudioSettings.TRANSCODE)
    out = saved(tmp_path, container)
    assert streams(out) == ["video:h264", "audio:aac"]
    assert audio_packets(out) != audio_packets(src)


@pytest.mark.parametrize("audio", [AudioSettings.AUTO, AudioSettings.TRANSCODE])
def test_webm_transcodes_to_opus(tmp_path: Path, audio: AudioSettings):
    src = source(tmp_path, "src.mp4", "aac")
    save(tmp_path, src, VideoFormat.WEBM, audio)
    assert streams(saved(tmp_path, VideoFormat.WEBM)) == ["video:vp9", "audio:opus"]


def test_webm_copy_is_an_error(tmp_path: Path):
    src = source(tmp_path, "src.mp4", "aac")
    with pytest.raises(ValueError, match="WebM does not support"):
        save(tmp_path, src, VideoFormat.WEBM, AudioSettings.COPY)


def test_mono_into_webm_is_an_error(tmp_path: Path):
    # libopus refuses 320 kb/s for one channel; the bitrate is the owner's call, but
    # the failure must not leave a silent video behind.
    src = source(tmp_path, "src.mp4", "aac", channels=1)
    message = r"transcode the aac audio to libopus for the \.webm file"
    with pytest.raises(RuntimeError, match=message):
        save(tmp_path, src, VideoFormat.WEBM, AudioSettings.AUTO)
    assert streams(saved(tmp_path, VideoFormat.WEBM)) == ["video:vp9"]


@pytest.mark.parametrize("audio", list(AudioSettings))
def test_source_without_audio_saves_the_video(tmp_path: Path, audio: AudioSettings):
    # Nothing is lost when Load Video's file has no audio stream.
    src = source(tmp_path, "src.mp4", None)
    save(tmp_path, src, VideoFormat.MP4, audio)
    assert streams(saved(tmp_path, VideoFormat.MP4)) == ["video:h264"]
