"""A generator whose items end before its expected length ends the run as a success with
one warning (Generator.shortfall_warning), on both executors: Load Video on a file cut
short (an interrupted copy or recording) read 150 of its 300 frames and said nothing.
An item's deferred error explains the end itself, so it gives no warning."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from engine_differential import KINDS, Result, execute, load_registry_errors
from scratch_storage import NO_FFMPEG, STORAGE
from test_engine_scenarios import Builder, items
from verify_runtime import edge, make_node

import api
from chain.json import JsonNode, parse_json
from chain.optimize import optimize
from nodes.impl.ffmpeg import FFMpegEnv, get_executable_path
from nodes.impl.video import VideoLoader
from packages.chaiNNer_standard.image.video_frames.load_video import shortfall

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"
FFMPEG = get_executable_path(STORAGE / "ffmpeg")[0]


@pytest.fixture
def test_package(monkeypatch: pytest.MonkeyPatch):
    # Test nodes belong to no package: both executors read their settings from a stand-in.
    monkeypatch.setattr(
        api.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )


@pytest.fixture(scope="module")
def schemas():
    main = sys.modules["__main__"]
    previous = getattr(main, "__file__", None)
    main.__file__ = str(BACKEND / "run.py")
    importlib.import_module("packages.chaiNNer_standard")
    errors = load_registry_errors(BACKEND / "server.py")
    assert not errors, [e.module for e in errors]
    yield {
        schema: {"inputs": [i.to_dict() for i in data.inputs], "kind": data.kind}
        for schema, (data, _) in api.registry.nodes.items()
    }
    if previous is None:
        del main.__file__
    else:
        main.__file__ = previous


def warnings(result: Result) -> list[str]:
    return [
        event["data"]["message"]
        for event in result.events
        if event["event"] == "execution-warning"
    ]


def numbers(trace: list[object], values: list[object], length: int, fail_fast=True):
    b = Builder()

    def generator():
        generator = items(values, fail_fast=fail_fast)
        generator.expected_length = length
        return generator

    b.node("numbers", generator, kind="generator", inputs=0)
    b.node("sink", trace.append, outputs=0, side_effects=True)
    b.edge("numbers", 0, "sink", 0)
    return b.chain


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_items_that_end_early_warn_once(tmp_path: Path, kind: str):
    trace: list[object] = []
    result = execute(kind, numbers(trace, [1, 2, 3], 5), storage=tmp_path)
    assert result.error is None
    assert trace == [1, 2, 3]
    assert warnings(result) == [
        "numbers ended after 3 of 5 items; the items read were processed."
    ]


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_items_of_the_expected_length_do_not_warn(tmp_path: Path, kind: str):
    result = execute(kind, numbers([], [1, 2, 3], 3), storage=tmp_path)
    assert result.error is None
    assert warnings(result) == []


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_a_deferred_item_error_explains_the_end(tmp_path: Path, kind: str):
    trace: list[object] = []
    chain = numbers(trace, [1, ValueError("item 2 is bad"), 3], 5, fail_fast=False)
    result = execute(kind, chain, storage=tmp_path)
    assert "item 2 is bad" in str(result.error)
    assert trace == [1, 3]
    assert warnings(result) == []


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_of_two_generators_only_the_one_that_ran_out_warns(tmp_path: Path, kind: str):
    # The loop ends with the shorter one: the other still had items
    trace: list[object] = []
    b = Builder()
    lengths: list[tuple[str, list[object]]] = [
        ("long", [1, 2, 3, 4, 5]),
        ("short", [10, 20]),
    ]
    for name, values in lengths:

        def generator(values: list[object] = values):
            generator = items(values)
            generator.expected_length = 5
            return generator

        b.node(name, generator, kind="generator", inputs=0)
    b.node(
        "sink", lambda *pair: trace.append(pair), inputs=2, outputs=0, side_effects=True
    )
    b.edge("long", 0, "sink", 0)
    b.edge("short", 0, "sink", 1)
    result = execute(kind, b.chain, storage=tmp_path)
    assert result.error is None
    assert trace == [(1, 10), (2, 20)]
    assert warnings(result) == [
        "short ended after 2 of 5 items; the items read were processed."
    ]


@pytest.mark.parametrize(
    ("ranged", "exact", "expected", "header", "read", "warning"),
    [
        (False, True, 300, 300, 150, "150 of 300 frames (the file may"),
        (False, True, 300, 300, 300, None),
        # PyAV counts the frames a cut file has left: its header has the full count
        (False, True, 150, 300, 150, "150 of about 300 frames (an estimate"),
        (False, False, 300, 300, 150, "150 of about 300 frames (an estimate"),
        # an estimate: more than 5 % and at least 2 frames
        (False, False, 300, 300, 285, None),
        (False, False, 300, 300, 284, "284 of about 300 frames"),
        (False, False, 20, 20, 19, None),
        (False, False, 20, 20, 18, "18 of about 20 frames"),
        # a Range: its clamped exact length only; Out past an estimated end reads on
        (True, True, 50, 300, 40, "40 of 50 frames"),
        (True, True, 50, 300, 50, None),
        (True, False, 101, 300, 30, None),
        # an estimate a longer audio track lengthens: the track's own count
        (False, False, 45, 30, 30, None),
        # no header count the reader can trust (an MP4's, read by the CLI)
        (False, False, 210, None, 120, None),
    ],
)
def test_load_video_s_shortfall(
    ranged: bool,
    exact: bool,
    expected: int,
    header: int,
    read: int,
    warning: str | None,
):
    message = shortfall("clip.mkv", ranged, exact, expected, header)(read)
    if warning is None:
        assert message is None
    else:
        assert message is not None and message.startswith(
            f"Load Video: clip.mkv ended after {warning}"
        )
        assert message.endswith("; the frames read were processed.")


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """cut.mkv: the first half of the bytes of a 300-frame FFV1 clip; whole.mkv: 30
    frames; sound.mkv: 30 frames and 1.5 s of audio, which makes the file's duration;
    copied.mp4: seconds 3-7 of a 10 s H.264 clip with 5 s GOPs cut by stream copy (as
    LosslessCut cuts), whose edit list hides the 90 frames before second 3;
    cut.mp4: the first 60 % of the bytes of a 300-frame MP4 with its index in front,
    whose last packets also fail to decode."""
    if not FFMPEG.exists():
        pytest.skip(NO_FFMPEG)
    folder = tmp_path_factory.mktemp("clips")
    audio = ("-f", "lavfi", "-i", "sine=duration=1.5")
    for name, seconds, more in (
        ("full.mkv", 10, ()),
        ("whole.mkv", 1, ()),
        ("sound.mkv", 1, audio),
    ):
        subprocess.run(
            [
                str(FFMPEG),
                *("-nostdin", "-v", "error", "-f", "lavfi"),
                *("-i", f"testsrc2=size=64x48:rate=30:duration={seconds}", *more),
                *("-c:v", "ffv1", "-c:a", "flac", str(folder / name)),
            ],
            check=True,
            timeout=120,
            stdin=subprocess.DEVNULL,
        )
    data = (folder / "full.mkv").read_bytes()
    (folder / "cut.mkv").write_bytes(data[: len(data) // 2])
    h264 = ("-c:v", "libx264", "-g", "150", "-keyint_min", "150", "-sc_threshold", "0")
    for arguments in (
        (*("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=30:duration=10"), *h264),
        ("-ss", "3", "-i", str(folder / "source.mp4"), "-t", "4", "-c", "copy"),
    ):
        name = "copied.mp4" if "copy" in arguments else "source.mp4"
        subprocess.run(
            [str(FFMPEG), "-nostdin", "-v", "error", *arguments, str(folder / name)],
            check=True,
            timeout=120,
            stdin=subprocess.DEVNULL,
        )
    subprocess.run(
        [
            str(FFMPEG),
            *("-nostdin", "-v", "error", "-i", str(folder / "source.mp4")),
            *("-c", "copy", "-movflags", "+faststart", str(folder / "front.mp4")),
        ],
        check=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )
    data = (folder / "front.mp4").read_bytes()
    (folder / "cut.mp4").write_bytes(data[: len(data) * 6 // 10])
    return folder


def load_video_run(
    schemas, kind: str, video: Path, output: Path, frames: dict[int, object]
) -> Result:
    nodes = [
        make_node(
            schemas, "load", "chainner:image:load_video", {0: str(video), **frames}
        ),
        make_node(
            schemas,
            "save",
            "chainner:image:save",
            {0: edge("load"), 1: str(output), 3: edge("load", 1), 4: "png"},
        ),
    ]
    chain = parse_json(cast("list[JsonNode]", nodes))
    optimize(chain)
    return execute(kind, chain, storage=STORAGE)


@pytest.mark.parametrize("reader", ["pyav", "cli"])
@pytest.mark.parametrize("kind", KINDS)
def test_load_video_on_a_cut_file_warns_and_succeeds(
    schemas,
    clips: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    reader: str,
):
    if reader == "cli":
        monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    else:
        monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    result = load_video_run(schemas, kind, clips / "cut.mkv", tmp_path, {})
    assert result.error is None
    read = len(list(tmp_path.glob("*.png")))
    assert 100 < read < 300
    message = (
        f"Load Video: cut.mkv ended after {read} of about 300 frames (an estimate from"
        " the file's header; the file may be truncated or damaged); the frames read"
        " were processed."
    )
    assert warnings(result) == [message]


@pytest.mark.parametrize("reader", ["pyav", "cli"])
@pytest.mark.parametrize("kind", KINDS)
def test_a_range_past_the_end_does_not_warn(
    schemas,
    clips: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    reader: str,
):
    if reader == "cli":
        monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    else:
        monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    result = load_video_run(
        schemas, kind, clips / "whole.mkv", tmp_path, {10: "range", 11: 20, 12: 40}
    )
    assert result.error is None
    assert len(list(tmp_path.glob("*.png"))) == 10
    assert warnings(result) == []


def test_the_header_s_count_is_the_video_track_s(clips: Path):
    # Matroska times the container by its longest track and each track by its own
    # DURATION tag, which a cut file keeps (it sits before the frames)
    env = FFMpegEnv(*map(str, get_executable_path(STORAGE / "ffmpeg")))
    sound = VideoLoader(clips / "sound.mkv", env)
    assert (vars(sound)["metadata"].frame_count, sound.header_frame_count()) == (45, 30)
    assert VideoLoader(clips / "cut.mkv", env).header_frame_count() == 300


@pytest.mark.parametrize("reader", ["pyav", "cli"])
@pytest.mark.parametrize("kind", KINDS)
def test_a_longer_audio_track_does_not_warn(
    schemas,
    clips: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    reader: str,
):
    if reader == "cli":
        monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    else:
        monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    result = load_video_run(schemas, kind, clips / "sound.mkv", tmp_path, {})
    assert result.error is None
    assert len(list(tmp_path.glob("*.png"))) == 30
    assert warnings(result) == []


def test_an_mp4_s_header_count_leaves_out_what_its_edit_list_hides(
    clips: Path, monkeypatch: pytest.MonkeyPatch
):
    # nb_frames counts every stored frame, PyAV's index the shown ones; the CLI cannot
    # tell the hidden ones apart
    env = FFMpegEnv(*map(str, get_executable_path(STORAGE / "ffmpeg")))
    monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    copied = VideoLoader(clips / "copied.mp4", env)
    shown = copied.exact_frame_count()
    assert shown is not None and vars(copied)["metadata"].frame_count > shown * 1.5
    assert copied.header_frame_count() == shown
    monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    assert VideoLoader(clips / "copied.mp4", env).header_frame_count() is None


@pytest.mark.parametrize("reader", ["pyav", "cli"])
@pytest.mark.parametrize("kind", KINDS)
def test_a_stream_copy_cut_does_not_warn(
    schemas,
    clips: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    reader: str,
):
    if reader == "cli":
        monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    else:
        monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    result = load_video_run(schemas, kind, clips / "copied.mp4", tmp_path, {})
    assert result.error is None
    assert len(list(tmp_path.glob("*.png"))) > 100
    assert warnings(result) == []


@pytest.mark.parametrize("kind", KINDS)
def test_a_cut_mp4_warns_where_pyav_reads_it(
    schemas,
    clips: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
):
    monkeypatch.delenv("CHAINNER_C_VIDEO_READER", raising=False)
    result = load_video_run(schemas, kind, clips / "cut.mp4", tmp_path, {})
    assert result.error is None
    read = len(list(tmp_path.glob("*.png")))
    assert 100 < read < 300
    [message] = warnings(result)
    assert message.startswith(f"Load Video: cut.mp4 ended after {read} of ")
    assert message.endswith("damaged); the frames read were processed.")
