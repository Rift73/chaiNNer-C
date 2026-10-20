"""Save Video muxes the source's audio with chaiNNer's own FFmpeg, so a PC without
FFmpeg on PATH still gets it; "Auto" transcodes the audio to AAC when the container
cannot hold a copy of it; and audio the mux cannot carry fails the run instead of
leaving a silent video (upstream chaiNNer #3331). An MP4 or MOV with audio gets its
faststart on the mux, which writes the final file (instant-exec gate R5a), and Load
Video's Range cuts the audio to the frames read (R7)."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

# The node's modules load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import NodeContext
from nodes.impl.colour_info import (
    ChromaLocation,
    Matrix,
    Primaries,
    Range,
    Transfer,
    VideoMode,
)
from nodes.impl.ffmpeg import FFMpegEnv, get_executable_path
from nodes.impl.video import VideoLoader, close_previews
from packages.chaiNNer_standard.image.video_frames.load_video import (
    FramesMode,
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

# No Colour Info, every colour choice Auto, Bit Depth Auto, siting correction Auto.
RGB_AUTO = (
    None,
    Matrix.AUTO,
    Transfer.AUTO,
    Primaries.AUTO,
    Range.AUTO,
    ChromaLocation.AUTO,
    BitDepth.AUTO,
    SitingCorrection.AUTO,
)

# The FFmpeg the backend downloads to its storage on first use (FFMpegEnv.get_integrated).
STORAGE = Path(os.environ.get("APPDATA", "")) / "chaiNNer" / "backend-storage"
FFMPEG, FFPROBE = map(str, get_executable_path(STORAGE / "ffmpeg"))

pytestmark = pytest.mark.skipif(
    not Path(FFMPEG).exists(), reason="the integrated FFmpeg is not downloaded"
)


class Context:
    """The NodeContext members Load Video and Save Video use."""

    chosen_item = None  # every item runs, so the video is written
    node_id = "load-video"  # Load Video's Preview reader is the node's

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
    tmp_path: Path,
    name: str,
    audio_codec: str | None,
    channels: int = 2,
    seconds: int = 1,
) -> Path:
    """`seconds` of H.264 video with a tone in `audio_codec`, or no audio for None."""
    path = tmp_path / name
    video = ["-f", "lavfi", "-i", "testsrc2=size=64x48:rate=30"]
    codecs = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio_codec is None:
        ffmpeg(*video, "-t", str(seconds), *codecs, str(path))
        return path
    audio = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    codecs += ["-c:a", audio_codec, "-ac", str(channels)]
    ffmpeg(*video, *audio, "-t", str(seconds), *codecs, str(path))
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


def video_packets(path: Path) -> str:
    """MD5 of the first video stream's packets."""
    return ffmpeg(
        "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "md5", "-"
    ).decode()


def top_level_boxes(path: Path) -> list[str]:
    """The types of an MP4's or MOV's top-level boxes, in file order."""
    names = []
    with path.open("rb") as file:
        while header := file.read(8):
            size, name = int.from_bytes(header[:4], "big"), header[4:].decode("latin-1")
            names.append(name)
            if size == 0:  # the box runs to the end of the file
                break
            if size == 1:  # a 64-bit size follows the type
                size = int.from_bytes(file.read(8), "big") - 8
            file.seek(size - 8, os.SEEK_CUR)
    return names


def stream_times(path: Path, entry: str) -> dict[str, float]:
    """Each stream's time `entry` ("duration" or "start_time") in seconds, by codec
    type."""
    probe = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    ).stdout
    return {s["codec_type"]: float(s[entry]) for s in json.loads(probe)["streams"]}


def save(
    tmp_path: Path,
    src: Path,
    container: VideoFormat,
    audio: AudioSettings,
    with_audio: bool = True,
    name: str = "out",
):
    encoder = VideoEncoder.VP9 if container == VideoFormat.WEBM else VideoEncoder.H264
    loader = VideoLoader(src, FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE))
    collector = save_video_node(
        cast(NodeContext, Context()),
        None,
        tmp_path,
        name,
        VideoInputMode.RGB,
        None,
        None,
        None,
        Simplicity.ADVANCED,
        container,
        encoder,
        VideoPreset.ULTRA_FAST,
        23,
        None,
        SimpleVideoFormat.MP4_H264,
        75,
        30.0,
        loader.get_audio_stream() if with_audio else None,
        audio,
        *RGB_AUTO,
    )
    for value in np.linspace(0, 1, 30, dtype=np.float32):
        image = np.full((48, 64, 3), value, np.float32)
        collector.on_iterate((image, None, None, None))  # Image Sequence, Y, U, V
    collector.on_complete()


def saved(tmp_path: Path, container: VideoFormat) -> Path:
    """The saved video, once no temporary file of the video or the audio step is left
    behind."""
    assert not list(tmp_path.glob("out.*.chainner-c-*.tmp"))
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
    # Copy never transcodes; the run fails and, as every failed write, leaves no file:
    # the video goes to its path only at a successful close.
    src = source(tmp_path, "src.mov", "pcm_s16le")
    message = r"copy the pcm_s16le audio into the \.mp4 file\. Set Audio to Auto or"
    with pytest.raises(RuntimeError, match=message):
        save(tmp_path, src, VideoFormat.MP4, AudioSettings.COPY)
    assert not saved(tmp_path, VideoFormat.MP4).exists()


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
    assert not saved(tmp_path, VideoFormat.WEBM).exists()


@pytest.mark.parametrize("audio", [AudioSettings.AUTO, AudioSettings.TRANSCODE])
def test_mono_into_webm_transcodes_to_opus(tmp_path: Path, audio: AudioSettings):
    # libopus takes at most 256 kb/s per channel, so mono gets 256k instead of 320k.
    src = source(tmp_path, "src.mp4", "aac", channels=1)
    save(tmp_path, src, VideoFormat.WEBM, audio)
    assert streams(saved(tmp_path, VideoFormat.WEBM)) == ["video:vp9", "audio:opus"]


@pytest.mark.parametrize("audio", list(AudioSettings))
def test_source_without_audio_saves_the_video(tmp_path: Path, audio: AudioSettings):
    # Nothing is lost when Load Video's file has no audio stream: also not the faststart
    # the encode pass left to the mux (R5a).
    src = source(tmp_path, "src.mp4", None)
    save(tmp_path, src, VideoFormat.MP4, audio)
    out = saved(tmp_path, VideoFormat.MP4)
    assert streams(out) == ["video:h264"]
    boxes = top_level_boxes(out)
    assert boxes.index("moov") < boxes.index("mdat")


@pytest.mark.parametrize("container", [VideoFormat.MP4, VideoFormat.MOV])
def test_audio_mux_writes_the_moov_first(tmp_path: Path, container: VideoFormat):
    # R5a: the mux writes the final file with faststart (the moov box before the media
    # data), where the encode pass rewrote its own file for it and the mux then wrote
    # the final file without. Only the box order changes: the video packets are those
    # of the same frames saved without audio (the encode pass as before), the audio
    # packets the source's.
    src = source(tmp_path, "src.mp4", "aac")
    save(tmp_path, src, container, AudioSettings.AUTO)
    save(tmp_path, src, container, AudioSettings.AUTO, with_audio=False, name="mute")
    out = saved(tmp_path, container)
    boxes = top_level_boxes(out)
    assert boxes.index("moov") < boxes.index("mdat")
    assert streams(out) == ["video:h264", "audio:aac"]
    assert video_packets(out) == video_packets(tmp_path / f"mute.{container.ext}")
    assert audio_packets(out) == audio_packets(src)


def save_frames(
    tmp_path: Path,
    src: Path,
    frames: tuple[int, int] | None,
    audio: AudioSettings,
    container: VideoFormat = VideoFormat.MP4,
):
    """Load Video's frames of `src` (the Range `frames` (In, Out), or All) saved by Save
    Video as `container` with its audio, as `audio` says."""
    context = Context()
    auto = (Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO, ChromaLocation.AUTO)
    first, last = frames or (0, 0)
    generator, _, _, fps, audio_stream, _, _, _, _ = load_video_node(
        cast(NodeContext, context),
        src,
        0,
        FramesMode.ALL if frames is None else FramesMode.RANGE,
        first,
        last,
        VideoMode.RGB,
        RgbPrecision.FLOAT,
        *auto,
    )
    collector = save_video_node(
        cast(NodeContext, context),
        None,
        tmp_path,
        "out",
        VideoInputMode.RGB,
        None,
        None,
        None,
        Simplicity.ADVANCED,
        container,
        VideoEncoder.H264,
        VideoPreset.ULTRA_FAST,
        23,
        None,
        SimpleVideoFormat.MP4_H264,
        75,
        fps,
        audio_stream,
        audio,
        *RGB_AUTO,
    )
    for item in generator.supplier():
        assert isinstance(item, tuple)
        collector.on_iterate((item[0], None, None, None))  # Image Sequence, Y, U, V
    collector.on_complete()
    for cleanup in context.cleanups:
        cleanup()
    close_previews()  # the Preview's reader lets the source go


@pytest.mark.parametrize("frames", [(0, 59), (30, 89), (0, 149), None])
def test_a_range_cuts_the_audio_to_the_frames_read(
    tmp_path: Path, frames: tuple[int, int] | None
):
    # R7: Load Video's Range of 60 frames (2 s of a 5 s source, from its start or from
    # 1 s) gives audio that ends within one AAC frame (1024 samples at 48 kHz) of the
    # video, and so does the Range of all 150 frames. Frames All copies the whole audio,
    # as before.
    src = source(tmp_path, "src.mp4", "aac", seconds=5)
    save_frames(tmp_path, src, frames, AudioSettings.AUTO)
    out = saved(tmp_path, VideoFormat.MP4)
    lengths = stream_times(out, "duration")
    if frames is None:
        assert lengths["video"] == pytest.approx(5.0, abs=0.001)
        assert audio_packets(out) == audio_packets(src)
    else:
        seconds = (frames[1] - frames[0] + 1) / 30
        assert lengths["video"] == pytest.approx(seconds, abs=0.001)
        assert abs(lengths["audio"] - lengths["video"]) <= 1024 / 48000


def audio_packet_hashes(path: Path) -> list[str]:
    """Each audio packet's MD5, in order (FFmpeg's framemd5 of a stream copy: stream,
    dts, pts, duration, size, the data's hash, then any side data's)."""
    lines = ffmpeg(
        "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "framemd5", "-"
    ).decode()
    return [
        line.split(",")[5].strip()
        for line in lines.splitlines()
        if line and not line.startswith("#")
    ]


def decoded_audio(path: Path) -> np.ndarray:
    """The first audio stream decoded to mono float32 samples (48 kHz here)."""
    data = ffmpeg("-i", str(path), "-map", "0:a:0", "-ac", "1", "-f", "f32le", "-")
    return np.frombuffer(data, np.float32)


def found_at(part: np.ndarray, whole: np.ndarray, near: int, reach: int) -> int:
    """Where in `whole` the first 0.1 s of `part` lies, within `reach` samples of
    `near`: the best match of the noise."""
    head = part[:4800]
    window = whole[near - reach : near + reach + head.size]
    return near - reach + int(np.argmax(np.correlate(window, head, mode="valid")))


def noise_source(tmp_path: Path) -> Path:
    """5 s of H.264 video with white noise in AAC (48 kHz mono): any stretch of the noise
    tells where it lies."""
    src = tmp_path / "noise.mp4"
    ffmpeg(
        *("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=30"),
        *("-f", "lavfi", "-i", "anoisesrc=color=white:amplitude=0.5:seed=7:r=48000"),
        *("-t", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p"),
        *("-c:a", "aac", "-ac", "1", str(src)),
    )
    return src


@pytest.mark.parametrize("audio", [AudioSettings.AUTO, AudioSettings.TRANSCODE])
@pytest.mark.parametrize("container", [VideoFormat.MP4, VideoFormat.MKV])
def test_a_range_s_audio_starts_with_its_first_frame(
    tmp_path: Path, container: VideoFormat, audio: AudioSettings
):
    # A Range from In > 0 reads its audio from In / fps (the input's -ss). FFmpeg 5.1
    # copies such audio from a packet before In: an MP4's edit list hides the extra
    # audio, a Matroska file plays it (the audio started 1.26 s before the video). Auto
    # transcodes it, as Transcode does, which FFmpeg cuts at In to the sample.
    src = noise_source(tmp_path)
    save_frames(tmp_path, src, (37, 96), audio, container)
    out = saved(tmp_path, container)
    packets, source_packets = audio_packet_hashes(out), audio_packet_hashes(src)
    assert not set(packets) & set(source_packets)
    # The audio that plays with the video's first frame is the source's at frame 37
    # (1.2333 s, sample 59200), and the audio lasts the Range's 2 s, both within what
    # Matroska's millisecond timestamps tell (48 samples), the length within one AAC
    # frame (1024 samples). A Matroska file plays the encoder's 1024 priming samples
    # before the video, which starts 1024 samples later.
    whole, part = decoded_audio(src), decoded_audio(out)
    begins = stream_times(out, "start_time")
    lead = round((begins["video"] - begins["audio"]) * 48000)
    start = 37 * 48000 // 30
    assert abs(found_at(part, whole, start, 2048) + lead - start) <= 48
    assert abs(part.size - lead - 2 * 48000) <= 1024


@pytest.mark.parametrize("container", list(VideoFormat))
def test_copy_refuses_a_range_s_audio_after_frame_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, container: VideoFormat
):
    # Copy cannot cut the audio at a Range's In > 0 (FFmpeg copies from a packet before
    # it), so Save Video refuses it when the node runs, before FFmpeg starts and before
    # anything is written, in every container that carries audio (a GIF has none).
    src = noise_source(tmp_path)
    started: list[str] = []
    popen = subprocess.Popen

    def recording(args: Any, *rest: Any, **kwargs: Any):
        started.append(Path(args[0]).stem)
        return popen(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", recording)
    message = (
        r"^Copy cannot cut the audio exactly at a Range that starts after frame 0;"
        r" choose Auto \(re-encodes it\) or Transcode\.$"
    )
    if container is VideoFormat.GIF:
        save_frames(tmp_path, src, (37, 96), AudioSettings.COPY, container)
        assert streams(saved(tmp_path, container)) == ["video:gif"]
        return
    with pytest.raises(ValueError, match=message):
        save_frames(tmp_path, src, (37, 96), AudioSettings.COPY, container)
    close_previews()  # the Preview's reader lets the source go
    assert "ffmpeg" not in started
    assert sorted(path.name for path in tmp_path.iterdir()) == ["noise.mp4"]


@pytest.mark.parametrize("container", [VideoFormat.MP4, VideoFormat.MKV])
def test_copy_copies_a_range_s_audio_from_frame_0(
    tmp_path: Path, container: VideoFormat
):
    # A Range from frame 0 (an old chain's Use limit among them) cuts only the audio's
    # end: Copy copies it, every packet the source's, in order.
    src = noise_source(tmp_path)
    save_frames(tmp_path, src, (0, 59), AudioSettings.COPY, container)
    packets = audio_packet_hashes(saved(tmp_path, container))
    assert packets == audio_packet_hashes(src)[: len(packets)]
