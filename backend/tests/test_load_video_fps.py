"""Load Video's FPS is the rate its readers give the frames at, FFmpeg's guess. ffprobe's
r_frame_rate is only the timestamps' resolution in a recording whose timestamps jitter
(90000 for a phone's 25 fps video, which Save Video then wrote 3600 times too fast), and
twice the rate of a short H.264 track."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from scratch_storage import NO_FFMPEG, STORAGE

# The video module loads the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from nodes.impl import video
from nodes.impl.ffmpeg import FFMpegEnv, get_executable_path

FFMPEG, FFPROBE = map(str, get_executable_path(STORAGE / "ffmpeg"))
ENV = FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE)

pytestmark = pytest.mark.skipif(not Path(FFMPEG).exists(), reason=NO_FFMPEG)

H264 = ("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p")


def run_ffmpeg(*arguments: str) -> None:
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *arguments],
        check=True,
        stdin=subprocess.DEVNULL,
    )


def rates(path: Path) -> tuple[str, str]:
    probe = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    return stream["r_frame_rate"], stream["avg_frame_rate"]


def jittered(path: Path, time_base: str) -> Path:
    run_ffmpeg(
        *("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=25", "-frames:v", "100"),
        *("-vf", f"settb={time_base},setpts=(N*0.04+0.017*random(1))/TB"),
        *("-fps_mode", "vfr", "-enc_time_base", time_base),
        *H264,
        str(path),
    )
    return path


@pytest.mark.parametrize(
    ("name", "time_base", "reported"),
    [
        ("phone.mp4", "1/90000", "90000/1"),
        ("phone.mov", "1/90000", "90000/1"),
        ("screen.mp4", "1/1000", "2000/1"),
    ],
)
def test_jittered_timestamps_give_the_frames_rate(
    tmp_path: Path, name: str, time_base: str, reported: str
):
    path = jittered(tmp_path / name, time_base)
    assert rates(path)[0] == reported  # the case this guards
    loader = video.VideoLoader(path, ENV)
    assert loader.metadata.fps == pytest.approx(25, rel=0.01)
    frames = sum(1 for _ in loader.stream_frames())
    assert frames == loader.metadata.frame_count == 100


def test_a_short_h264_track_gives_its_own_rate(tmp_path: Path):
    path = tmp_path / "short.mkv"
    run_ffmpeg(
        *("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=5", "-frames:v", "3"),
        *H264,
        str(path),
    )
    assert rates(path) == ("10/1", "5/1")  # the case this guards
    assert video.VideoLoader(path, ENV).metadata.fps == 5


def test_a_count_by_duration_follows_the_rate(tmp_path: Path):
    path = jittered(tmp_path / "screen.mkv", "1/1000")
    loader = video.VideoLoader(path, ENV)
    assert loader.metadata.fps == pytest.approx(25, rel=0.01)
    assert loader.metadata.frame_count == pytest.approx(100, abs=2)


def test_a_constant_rate_opens_no_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "even.mp4"
    run_ffmpeg(
        *("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=25", "-frames:v", "5"),
        *H264,
        str(path),
    )

    def no_pyav():
        raise AssertionError("a constant rate needs no second opinion")

    monkeypatch.setattr(video, "_pyav", no_pyav)
    assert video.VideoLoader(path, ENV).metadata.fps == 25
