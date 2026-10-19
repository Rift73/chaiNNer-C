"""Save Video writes a temporary file in the target folder and renames it onto the video's
path at a successful close (instant-exec A4): FFmpeg is told the muxer the extension picks,
which writes the same bytes; an equal video is left untouched; Stop and failures delete only
the temporary file, so the path keeps what it held."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import cast

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
)
from nodes.impl.ffmpeg import get_executable_path
from packages.chaiNNer_standard.image.video_frames import save_video as module
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
FFMPEG = str(get_executable_path(STORAGE / "ffmpeg")[0])

pytestmark = pytest.mark.skipif(
    not Path(FFMPEG).exists(), reason="the integrated FFmpeg is not downloaded"
)

OLD = 1_000_000_000  # An mtime (2001) no write of this test can produce.
# Random muxer IDs and dates differ run to run; bitexact leaves the bytes the options give.
BITEXACT = ["-fflags", "+bitexact", "-flags", "+bitexact", "-threads", "1"]
# An encoder each container takes, as Save Video passes it.
ENCODERS = {
    VideoFormat.MKV: ["-c:v", "libx264", "-pix_fmt", "yuv420p"],
    VideoFormat.MP4: [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "faststart",
    ],
    VideoFormat.MOV: [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "faststart",
    ],
    VideoFormat.WEBM: ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p"],
    VideoFormat.AVI: ["-c:v", "libx264", "-pix_fmt", "yuv420p"],
    VideoFormat.GIF: [],
}


class Context:
    """The two NodeContext members Save Video uses; cleanup() is the chain's end (Stop)."""

    def __init__(self):
        self.storage_dir = STORAGE
        self.cleanups: list[Callable[[], None]] = []

    def add_cleanup(self, fn: Callable[[], None], after: str = "chain"):
        self.cleanups.append(fn)

    def cleanup(self):
        for fn in self.cleanups:
            fn()


def ffmpeg(*args: str):
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    )


@pytest.mark.parametrize("container", list(VideoFormat))
def test_the_named_muxer_writes_the_bytes_the_extension_picks(
    tmp_path: Path, container: VideoFormat
):
    source = ["-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10", "-frames:v", "3"]
    by_extension = tmp_path / f"out.{container.ext}"
    named = tmp_path / f"out.{container.ext}.chainner-c-1-0.tmp"
    ffmpeg(*source, *ENCODERS[container], *BITEXACT, str(by_extension))
    ffmpeg(*source, *ENCODERS[container], *BITEXACT, "-f", container.muxer, str(named))
    assert by_extension.read_bytes() == named.read_bytes()


def save(
    folder: Path,
    frames: list[np.ndarray],
    additional: str | None = " ".join(BITEXACT),
    context: Context | None = None,
    complete: bool = True,
) -> Context:
    """Save Video (FFV1 in MKV) of the frames to folder/out.mkv."""
    context = context or Context()
    collector = save_video_node(
        cast(NodeContext, context),
        None,
        folder,
        "out",
        VideoInputMode.RGB,
        None,
        None,
        None,
        Simplicity.ADVANCED,
        VideoFormat.MKV,
        VideoEncoder.FFV1,
        VideoPreset.ULTRA_FAST,
        23,
        additional,
        SimpleVideoFormat.MP4_H264,
        75,
        10.0,
        None,
        AudioSettings.AUTO,
        None,
        Matrix.AUTO,
        Transfer.AUTO,
        Primaries.AUTO,
        Range.AUTO,
        ChromaLocation.AUTO,
        BitDepth.AUTO,
        SitingCorrection.AUTO,
    )
    for frame in frames:
        collector.on_iterate((frame, None, None, None))  # Image Sequence, Y, U, V
    if complete:
        collector.on_complete()
    return context


def frames(value: float) -> list[np.ndarray]:
    return [np.full((32, 48, 3), value * i, np.float32) for i in range(1, 4)]


def foreign(folder: Path) -> dict[Path, bytes]:
    files = {
        folder / "out.mkv.chainner-c-1-0.tmp": b"another process",
        folder / "unrelated.tmp": b"unrelated",
    }
    for path, data in files.items():
        path.write_bytes(data)
    return files


def contents(folder: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in folder.iterdir()}


def test_an_equal_video_is_left_untouched_and_a_different_one_replaced(
    tmp_path: Path,
):
    others = foreign(tmp_path)
    path = tmp_path / "out.mkv"
    save(tmp_path, frames(0.1))
    first = path.read_bytes()
    os.utime(path, (OLD, OLD))
    save(tmp_path, frames(0.1))
    assert path.stat().st_mtime == OLD and path.read_bytes() == first
    save(tmp_path, frames(0.2))
    assert path.stat().st_mtime != OLD and path.read_bytes() != first
    assert contents(tmp_path) == {**others, path: path.read_bytes()}


def test_stop_deletes_only_the_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "out.mkv"
    path.write_bytes(b"the video before")
    os.utime(path, (OLD, OLD))
    others = foreign(tmp_path)
    existed: list[bool] = []

    class Spy:
        """The node's target, noting whether FFmpeg's file existed when abandoned."""

        def __init__(self, final: str):
            self.target = module.graph().AtomicTarget(final)
            self.temp = self.target.temp

        def abandon(self):
            existed.append(Path(self.temp).is_file())
            self.target.abandon()

    monkeypatch.setattr(module, "atomic_target", Spy)
    context = save(tmp_path, frames(0.1), complete=False)
    # Stop: the chain's cleanup aborts the writer; FFmpeg finishes the temporary file on
    # the pipe's end, then it is deleted.
    context.cleanup()
    assert existed == [True]
    assert contents(tmp_path) == {**others, path: b"the video before"}
    assert path.stat().st_mtime == OLD


def test_a_failed_encoder_leaves_the_file_before(tmp_path: Path):
    path = tmp_path / "out.mkv"
    path.write_bytes(b"the video before")
    os.utime(path, (OLD, OLD))
    others = foreign(tmp_path)
    context = Context()
    with pytest.raises((RuntimeError, OSError)):
        # FFmpeg refuses the pixel format and exits: a write or the close fails.
        save(tmp_path, frames(0.1) * 30, "-pix_fmt no-such-format", context)
    with suppress(OSError):  # Closing the dead pipe may report it, after the cleanup.
        context.cleanup()
    assert contents(tmp_path) == {**others, path: b"the video before"}
    assert path.stat().st_mtime == OLD
