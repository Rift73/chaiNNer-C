"""Load Video's exact reads: frame n read by a seek (stream_items(n, n + 1)) is item n of
the full read, bit for bit, in RGB float, RGB 8-bit and Planes and in both readers (PyAV
and the FFmpeg CLI). Runs on chaiNNer's integrated FFmpeg."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# The video module loads the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from nodes.impl import video
from nodes.impl.colour_info import (
    ChromaLocation,
    Matrix,
    Primaries,
    Range,
    Transfer,
    VideoMode,
)
from nodes.impl.ffmpeg import FFMpegEnv, get_executable_path

STORAGE = Path(os.environ.get("APPDATA", "")) / "chaiNNer" / "backend-storage"
FFMPEG, FFPROBE = map(str, get_executable_path(STORAGE / "ffmpeg"))
ENV = FFMpegEnv(ffmpeg=FFMPEG, ffprobe=FFPROBE)

pytestmark = pytest.mark.skipif(
    not Path(FFMPEG).exists(), reason="the integrated FFmpeg is not downloaded"
)
AUTO = (Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO, ChromaLocation.AUTO)
SOURCE = ("-f", "lavfi", "-i", "testsrc2=size=192x108:rate=30", "-frames:v", "150")
H264 = "h264-bframes.mp4"

# The files of the exactness matrix: what each is made from (another file of the matrix,
# or None for SOURCE), FFmpeg's input options for it and its output options.
FILES: dict[str, tuple[str | None, tuple[str, ...], tuple[str, ...]]] = {
    H264: (
        None,
        (),
        ("-c:v", "libx264", "-bf", "3", "-g", "60", "-keyint_min", "60"),
    ),
    "hevc-10-bit.mkv": (
        None,
        (),
        (
            *("-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params"),
            "keyint=60:min-keyint=60:bframes=4:open-gop=1:log-level=error",
        ),
    ),
    # Cut without re-encoding: an edit list hides the frames before the cut, which
    # the demuxer flags as discarded.
    "edit-list.mp4": (H264, ("-ss", "1.3"), ("-c", "copy")),
    # An open GOP's leading frames refer to the GOP before.
    "hevc-open-gop.mp4": (
        None,
        (),
        (
            *("-c:v", "libx265", "-pix_fmt", "yuv420p", "-x265-params"),
            "keyint=48:min-keyint=1:bframes=6:open-gop=1:scenecut=40:log-level=error",
        ),
    ),
    # MPEG-TS seeks by decoding timestamps and lands past the keyframe, so a seek starts
    # again from the keyframe before, and the first GOP decodes from the file's start.
    "h264.ts": (
        None,
        (),
        ("-c:v", "libx264", "-bf", "3", "-g", "60", "-keyint_min", "60"),
    ),
    "hevc-open-gop.ts": (
        None,
        (),
        (
            *("-c:v", "libx265", "-pix_fmt", "yuv420p", "-x265-params"),
            "keyint=48:min-keyint=1:bframes=6:open-gop=1:scenecut=40:log-level=error",
        ),
    ),
    # A display matrix (90 degrees); the integrated FFmpeg 5.1 has no -display_rotation.
    "rotated.mp4": (H264, (), ("-c", "copy", "-metadata:s:v:0", "rotate=90")),
    # Read through the FFmpeg CLI, which renumbers the frames to a constant rate.
    "variable.mkv": (
        None,
        (),
        (
            *("-vf", "setpts='if(lt(N,20),N,N+15)/(30*TB)'", "-fps_mode", "vfr"),
            *("-c:v", "libx264", "-g", "30"),
        ),
    ),
}
SETTINGS = {
    "rgb-float": video.ReadSettings(VideoMode.RGB, True, AUTO),
    "rgb-8-bit": video.ReadSettings(VideoMode.RGB, False, AUTO),
    "planes": video.ReadSettings(VideoMode.PLANES, True, AUTO),
}


def ffmpeg(*arguments: str) -> None:
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", *arguments],
        check=True,
        stdin=subprocess.DEVNULL,
    )


@pytest.fixture(scope="module")
def folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("seek")


def made(folder: Path, name: str) -> Path:
    """The matrix's file `name` in `folder`, made on first use."""
    path = folder / name
    if not path.exists():
        source, input_options, options = FILES[name]
        inputs = SOURCE if source is None else ("-i", str(made(folder, source)))
        ffmpeg(*input_options, *inputs, *options, str(path))
    return path


def assert_items_equal(item: tuple[object, ...], expected: tuple[object, ...]):
    assert len(item) == len(expected) == 4
    for value, reference in zip(item, expected, strict=True):
        if reference is None:
            assert value is None
            continue
        assert isinstance(value, np.ndarray) and isinstance(reference, np.ndarray)
        assert (value.dtype, value.shape) == (reference.dtype, reference.shape)
        np.testing.assert_array_equal(value, reference)


@pytest.mark.parametrize("mode", sorted(SETTINGS))
@pytest.mark.parametrize("name", sorted(FILES))
def test_seek_equals_full_read(
    folder: Path, monkeypatch: pytest.MonkeyPatch, name: str, mode: str
):
    monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    path = made(folder, name)
    warnings: list[str] = []
    monkeypatch.setattr(
        video.logger, "warning", lambda message, *args: warnings.append(message % args)
    )
    loader = video.VideoLoader(path, ENV, SETTINGS[mode])
    full = list(loader.stream_items())
    count = len(full)
    cli = f"Reading {path} through the FFmpeg CLI: its frame rate is variable"
    if name == "variable.mkv":
        # The CLI duplicates frames into the gap: frame n is not the n-th stored one.
        assert cli in warnings and count > 150
        keyframes: tuple[int, ...] = ()
    else:
        assert cli not in warnings
        index = video.frame_index(path)
        assert loader.frame_count() == len(index.pts) == count
        keyframes = index.keyframes
        assert len(keyframes) > 1
    if name == "rotated.mp4" and mode != "planes":
        # Upright: the stored 192x108 frames turned a quarter.
        assert full[0][0] is not None and full[0][0].shape[:2] == (192, 108)
    sample = {0, 1, count // 2, count - 1}
    sample |= {k + d for k in keyframes for d in range(-6, 2) if 0 <= k + d < count}
    # How many times each seek started decoding (a keyframe, or the file's start).
    starts: list[int] = []
    decode = vars(video)["_decode"]

    def counting(*arguments: Any):
        starts[-1] += 1
        return decode(*arguments)

    monkeypatch.setattr(video, "_decode", counting)
    for n in sorted(sample):
        starts.append(0)
        items = list(loader.stream_items(n, n + 1))
        assert len(items) == 1, n
        assert_items_equal(items[0], full[n])
    if name.endswith(".ts"):
        assert max(starts) > 1
    elif name != "variable.mkv":
        assert set(starts) == {1}
    # A range, a range to the end and one past it (it stops at the end, as a slice).
    for start, stop in ((count // 3, count // 3 + 7), (count - 3, None)):
        items = list(loader.stream_items(start, stop))
        assert len(items) == len(full[start:stop])
        for item, expected in zip(items, full[start:stop], strict=True):
            assert_items_equal(item, expected)
    assert len(list(loader.stream_items(count - 2, count + 5))) == 2
    with pytest.raises(
        ValueError,
        match=rf"^frame {count} is past the end \(the video has {count} frames\)$",
    ):
        list(loader.stream_items(count, count + 1))
