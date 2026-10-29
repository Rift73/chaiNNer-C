"""The VapourSynth I/O nodes: Split Planes, Preview Clips (vsview, one window per
node), Clip Frames (vspipe frames as chaiNNer images), Frames to Clip (chaiNNer images
as a lossless clip) and Save Script; a failed vspipe run names the node whose statement
failed. VapourSynth tests run in native\\local\\vapoursynth\\venv; no test opens vsview."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from api import MB, Aborted, NodeAttributedError, NodeContext, SettingsParser
from nodes.impl.colour_info import (
    FIELDS,
    ChromaLocation,
    ColourInfo,
    Matrix,
    Primaries,
    Range,
    Transfer,
    VideoMode,
    code_to_float,
    detect_colour_info,
)
from nodes.impl.ffmpeg import FFMpegEnv, get_executable_path
from nodes.impl.vapoursynth import runner
from nodes.impl.vapoursynth.colour_props import colour_stream
from nodes.impl.vapoursynth.runner import VSScriptError, clip_info, read_frames
from nodes.impl.vapoursynth.script import VideoClip, clip_var, compile_script
from packages.chaiNNer_vapoursynth.vapoursynth import preview_clips
from packages.chaiNNer_vapoursynth.vapoursynth.preview_clips import (
    preview_clips_node,
)
from packages.chaiNNer_vapoursynth.vapoursynth.split_planes import split_planes_node


class Context:
    """The NodeContext members the VapourSynth nodes use."""

    def __init__(self, storage: Path, node_id: str):
        self.storage_dir = storage
        self.node_id = node_id
        self.settings = SettingsParser({"cache_mb": 256})
        self.cleanups: list[Callable[[], None]] = []
        self.aborted = False

    def check_aborted(self):
        if self.aborted:
            raise Aborted

    def add_cleanup(self, fn: Callable[[], None], after: str = "chain"):
        self.cleanups.append(fn)

    def run_cleanups(self):
        for fn in self.cleanups:
            fn()
        self.cleanups.clear()


def context(storage: Path, node_id: str) -> NodeContext:
    return cast(NodeContext, Context(storage, node_id))


def clip(node_id: str, call: str, *deps: VideoClip) -> VideoClip:
    var = clip_var(node_id)
    return VideoClip(
        var=var, stmt=f"{var} = {call}", deps=deps, imports=frozenset(), node_id=node_id
    )


def pairs(*values: VideoClip | str | None) -> list[VideoClip | str | None]:
    """Preview Clips' 32 inputs (16 clips with their names): the given ones, the rest
    unset."""
    return [*values, *[None] * (32 - len(values))]


BLANK = "core.std.BlankClip(width=64, height=48, length=5)"


def test_split_planes_three_outputs_one_statement():
    source = clip("src", "core.std.BlankClip(format=vs.YUV444P8, length=2)")
    y, u, v = split_planes_node(context(Path(), "planes"), source)

    assert [c.var for c in (y, u, v)] == [clip_var("planes", i) for i in range(3)]
    # Module-qualified, as every call: a user file split.py cannot shadow it.
    statement = f"{y.var}, {u.var}, {v.var} = vstools.split({source.var})"
    for plane in (y, u, v):
        assert plane.stmt == statement
        assert plane.deps == (source,)
        assert plane.imports == frozenset({"import vstools"})
        assert plane.node_id == "planes"

    script = compile_script([(y, "Y"), (u, "U"), (v, "V")], cache_mb=64)
    assert script.count(statement) == 1
    assert "import vstools" in script.splitlines()


def test_sixteen_named_clips_keep_the_first_eight_ids(tmp_path: Path):
    from packages.chaiNNer_vapoursynth.vapoursynth import clips_group
    from packages.chaiNNer_vapoursynth.vapoursynth.save_script import save_script_node

    nodes = {node.schema_id: node for node in clips_group.nodes}
    labels = [label for n in range(1, 17) for label in (f"Clip {n}", f"Name {n}")]
    # A saved chain's values of the first 8 pairs keep their ids; pairs 9 to 16 add
    # new ids after them.
    preview = nodes["chainner:vapoursynth:preview_clips"]
    assert [(i.id, i.label) for i in preview.inputs] == list(enumerate(labels))
    save = nodes["chainner:vapoursynth:save_script"]
    assert [(i.id, i.label) for i in save.inputs[2:]] == list(enumerate(labels, 2))

    # 16 connected pairs are 16 outputs, in order.
    clips = [clip(f"c{n}", BLANK) for n in range(16)]
    names = [f"Clip number {n}" for n in range(16)]
    values = [value for pair in zip(clips, names, strict=True) for value in pair]
    save_script_node(context(tmp_path, "save"), tmp_path, "sixteen", *pairs(*values))
    lines = (tmp_path / "sixteen.vpy").read_text(encoding="utf-8").splitlines()
    outputs = [line for line in lines if line.startswith("set_output(")]
    assert outputs == [
        f"set_output({c.var}, {n}, {name!r})"
        for n, (c, name) in enumerate(zip(clips, names, strict=True))
    ]


def test_old_frames_videos_are_deleted_at_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    from packages.chaiNNer_vapoursynth.chains import remove_old_videos

    day_old = time.time() - 25 * 3600
    recent = time.time() - 23 * 3600
    files = {
        # Frames to Clip's video and BestSource's index of a node that ran no more.
        "gone/frames-0123456789ab.mkv": (day_old, 3 * MB // 2),
        "gone/frames-0123456789ab.mkv.0.bsindex": (day_old, MB // 2),
        "gone/frames-locked.mkv": (day_old, 7),
        # A newer video, the scripts and vsview's log stay.
        "kept/frames-ba9876543210.mkv": (recent, 5),
        "gone/frames.vpy": (day_old, 5),
        "gone/frames.ab12cd.frames.vpy": (day_old, 5),
        "gone/frames-x.vpy": (day_old, 5),
        "gone/preview.vpy": (day_old, 5),
        "gone/vsview.log": (day_old, 5),
        "gone/frames-x.log": (day_old, 5),
    }
    for name, (mtime, size) in files.items():
        path = tmp_path / "chains" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * size)
        os.utime(path, (mtime, mtime))

    unlink = Path.unlink

    def locked(path: Path, missing_ok: bool = False) -> None:
        if path.name == "frames-locked.mkv":
            raise PermissionError(13, "in use by another process", str(path))
        unlink(path, missing_ok)

    monkeypatch.setattr(Path, "unlink", locked)
    with caplog.at_level(logging.INFO):
        remove_old_videos(tmp_path)
    left = sorted(
        p.relative_to(tmp_path / "chains").as_posix()
        for p in (tmp_path / "chains").rglob("*")
        if p.is_file()
    )
    deleted = {name for name in files if name.startswith("gone/frames-0123")}
    assert len(deleted) == 2
    assert left == sorted(set(files) - deleted)
    messages = [r.getMessage() for r in caplog.records]
    assert any("frames-locked.mkv" in m and "in use" in m for m in messages)
    assert any("Deleted 2 temporary files" in m and "(2.0 MB)" in m for m in messages)

    # Outside the server, or before any chain ran: nothing to do, and no error.
    remove_old_videos(None)
    remove_old_videos(tmp_path / "missing")


@pytest.mark.parametrize(
    ("width", "height", "matrix"),
    [
        (720, 480, 6),
        (720, 576, 6),
        (1024, 576, 6),
        # HD by Load Video's rule (width >= 1280 or height > 576): scope crops of 720p
        # and 1080p, and tall clips.
        (1280, 536, 1),
        (1920, 544, 1),
        (1280, 720, 1),
        (720, 1280, 1),
    ],
)
def test_untagged_yuv_frames_use_load_videos_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    height: int,
    matrix: int,
):
    # The script vspipe would render: the chain's, with the conversion to RGBS.
    rendered: list[str] = []

    def vspipe_frames(script: Path, *_: object):
        rendered.append(script.read_text(encoding="utf-8"))
        yield from ()

    monkeypatch.setattr(runner, "_vspipe_frames", vspipe_frames)
    script = tmp_path / "frames.vpy"
    script.write_text("# the chain\n", encoding="utf-8")
    info = runner.ClipInfo(width, height, 2, 24, 1, "YUV420P8")
    assert list(read_frames(script, 0, 0, 2, info=info)) == []
    assert len(rendered) == 1
    assert f"format=vapoursynth.RGBS, matrix_in={matrix})" in rendered[0]


@pytest.mark.parametrize("run", ["evaluation", "frame"])
def test_stop_ends_vspipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run: str):
    # A vspipe that does not end: an evaluation that indexes a long video or a hung
    # plugin, a frame that an AI model renders for minutes.
    # The runner's own Popen arguments pass through, its text mode included.
    started: list[subprocess.Popen[Any]] = []
    popen = subprocess.Popen

    def hanging(command: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
        sleep = [sys.executable, "-c", "import time; time.sleep(60)"]
        started.append(popen(sleep, **kwargs))
        return started[-1]

    monkeypatch.setattr(runner, "vspipe", lambda: Path("vspipe"))
    monkeypatch.setattr(runner, "vs_env", lambda: dict(os.environ))
    monkeypatch.setattr(subprocess, "Popen", hanging)
    script = tmp_path / "chain.vpy"
    script.write_text("# the chain\n", encoding="utf-8")
    ctx = Context(tmp_path, "frames")
    ctx.aborted = True

    begin = time.monotonic()
    with pytest.raises(Aborted):
        if run == "evaluation":
            clip_info(script, 0, cast(NodeContext, ctx))
        else:
            info = runner.ClipInfo(8, 6, 2, 24, 1, "RGBS")
            list(
                read_frames(script, 0, 0, 2, info=info, context=cast(NodeContext, ctx))
            )
    assert time.monotonic() - begin < 10
    assert len(started) == 1
    assert started[0].poll() is not None


def test_executor_reports_an_attributed_error_on_its_node():
    from process import NodeExecutionError, on_failing_node

    frames_data = SimpleNamespace(schema_id="chainner:vapoursynth:clip_frames")
    bad_data = SimpleNamespace(schema_id="chainner:vapoursynth:convert_format")
    nodes = cast(
        Any,
        {
            "frames": SimpleNamespace(id="frames", data=frames_data),
            "bad": SimpleNamespace(id="bad", data=bad_data),
        },
    )
    # VapourSynth's errors are NodeAttributedErrors: the executor knows no VapourSynth.
    script_error = VSScriptError("Crop: negative corner coordinates", "bad")
    assert isinstance(script_error, NodeAttributedError)

    # Clip Frames ran the script; the statement of node "bad" failed.
    error = NodeExecutionError(
        cast(Any, "frames"), cast(Any, frames_data), str(script_error), {}
    )
    error.__cause__ = script_error
    moved = on_failing_node(error, nodes)
    assert isinstance(moved, NodeExecutionError)
    assert (moved.node_id, moved.node_data) == ("bad", bad_data)
    assert str(moved) == "Crop: negative corner coordinates"
    assert moved.__cause__ is script_error

    # An error in a generator's iteration (a frame of Clip Frames) arrives unwrapped.
    attributed = NodeAttributedError("frame 2 failed", "bad")
    unwrapped = on_failing_node(attributed, nodes)
    assert isinstance(unwrapped, NodeExecutionError)
    assert (unwrapped.node_id, unwrapped.__cause__) == ("bad", attributed)

    # Errors that name no node in the chain, or no node at all, stay as they are.
    for cause in (
        ValueError("x"),
        NodeAttributedError("x"),
        NodeAttributedError("x", "elsewhere"),
    ):
        other = NodeExecutionError(cast(Any, "frames"), cast(Any, frames_data), "x", {})
        other.__cause__ = cause
        assert on_failing_node(other, nodes) is other
        assert on_failing_node(cause, nodes) is cause


def test_preview_without_vsview_for_the_glibc_says_why_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(preview_clips, "glibc_version", lambda: (2, 31))

    with pytest.raises(VSScriptError, match=r"glibc 2\.34 or newer.* glibc 2\.31\."):
        preview_clips_node(context(tmp_path, "view"), *pairs(clip("src", BLANK)))
    assert not (tmp_path / "vapoursynth").exists()


# VapourSynth from here on: the venv's vspipe renders in child processes.


@pytest.fixture
def vs() -> Any:
    return pytest.importorskip("vapoursynth")


class FakeViewer:
    """A vsview process that runs until `close()`."""

    def __init__(self, script: Path):
        self.script = script
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def close(self):
        self.returncode = 0


@pytest.fixture
def viewers(monkeypatch: pytest.MonkeyPatch, vs: Any) -> list[FakeViewer]:
    """Every vsview the test's runs start; none is a real process, on a system where
    vsview installs whatever this one's glibc. (Preview Clips checks its script in
    vspipe first.)"""
    monkeypatch.setattr(preview_clips, "glibc_version", lambda: None)
    started: list[FakeViewer] = []

    def open_viewer(script: Path) -> FakeViewer:
        started.append(FakeViewer(script))
        return started[-1]

    monkeypatch.setattr(runner, "_open_viewer", open_viewer)
    monkeypatch.setattr(runner, "_viewers", {})
    return started


def test_preview_writes_named_outputs(tmp_path: Path, viewers: list[FakeViewer]):
    source = clip("src", BLANK)
    crop = clip("crop", f"{clip_var('src')}.std.Crop(left=8)", source)

    preview_clips_node(
        context(tmp_path, "view"), *pairs(source, "Source", None, None, crop)
    )

    script = tmp_path / "vapoursynth" / "chains" / "view" / "preview.vpy"
    text = script.read_text(encoding="utf-8")
    # Connected pairs in order: the third pair is output 1 and has its default name.
    assert f"set_output({source.var}, 0, 'Source')" in text
    assert f"set_output({crop.var}, 1, 'Clip 3')" in text
    assert "core.max_cache_size = 256" in text
    assert [v.script for v in viewers] == [script]


def test_preview_reuses_running_viewer(tmp_path: Path, viewers: list[FakeViewer]):
    source = clip("src", BLANK)
    script = tmp_path / "vapoursynth" / "chains" / "view" / "preview.vpy"

    preview_clips_node(context(tmp_path, "view"), *pairs(source, "First"))
    preview_clips_node(context(tmp_path, "view"), *pairs(source, "Second"))
    # The open viewer gets the rewritten file (the user reloads it); no second window.
    assert len(viewers) == 1
    assert "'Second'" in script.read_text(encoding="utf-8")

    # Another Preview Clips node has a viewer of its own.
    preview_clips_node(context(tmp_path, "other"), *pairs(source))
    assert len(viewers) == 2

    # Once the user closes vsview, the next run opens it again.
    viewers[0].close()
    preview_clips_node(context(tmp_path, "view"), *pairs(source, "Third"))
    assert len(viewers) == 3
    assert viewers[2].script == script


def test_preview_error_shows_on_its_node_and_opens_nothing(
    tmp_path: Path, viewers: list[FakeViewer]
):
    source = clip("src", BLANK)
    bad = clip("bad", f"{clip_var('src')}.std.Crop(left=-4)", source)

    with pytest.raises(NodeAttributedError, match="negative corner") as raised:
        preview_clips_node(context(tmp_path, "view"), *pairs(source, None, bad))
    assert raised.value.node_id == "bad"
    assert viewers == []


def test_clip_frames_count_and_values(
    tmp_path: Path, vs: Any, monkeypatch: pytest.MonkeyPatch
):
    from packages.chaiNNer_vapoursynth.vapoursynth.clip_frames import (
        clip_frames_node,
    )

    # Every vspipe --info process (an evaluation of the script before the frames' run).
    info_runs: list[list[str]] = []
    popen = subprocess.Popen

    def counting(command: list[str], **kwargs: Any) -> Any:
        if "--info" in command:
            info_runs.append(command)
        return popen(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", counting)

    # Five 8x6 frames; frame i is RGB (10i, 20i, 30i) at 25 fps.
    frames = clip(
        "src",
        "core.std.Splice([core.std.BlankClip(width=8, height=6, length=1, fpsnum=25,"
        " fpsden=1, color=[10 * i, 20 * i, 30 * i]) for i in range(5)])",
    )
    ctx = context(tmp_path, "frames")
    generator, fps, colour = clip_frames_node(ctx, frames, VideoMode.RGB)

    assert fps == 25
    assert generator.expected_length == 5
    # An RGB clip without colour properties: identity matrix, full range.
    assert (colour.matrix, colour.range) == (Matrix.RGB, Range.FULL)
    assert (colour.luma_size, colour.bit_depth) == ((8, 6), 8)
    items = cast(list[tuple[Any, ...]], list(generator.supplier()))
    cast(Context, ctx).run_cleanups()
    # One --info for the length, FPS and colour; the frames' run reuses it.
    assert len(info_runs) == 1
    assert [item[1] for item in items] == list(range(5))
    for i, (frame, _, *planes) in enumerate(items):
        # Mode RGB: no planes.
        assert planes == [None, None, None]
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (6, 8, 3)
        assert frame.dtype == np.float32
        # zimg scales 8-bit values by a float32 reciprocal: within an ulp of x/255.
        expected = np.array([30 * i, 20 * i, 10 * i], np.float32) / 255
        np.testing.assert_allclose(
            frame, np.broadcast_to(expected, (6, 8, 3)), rtol=0, atol=1e-7
        )

    # A run that stops early (an error downstream, Stop) ends vspipe at its cleanup;
    # the RGB24 clip's float32 copy of the script goes with it.
    folder = tmp_path / "vapoursynth" / "chains" / "frames"
    partial = iter(generator.supplier())
    next(partial)
    assert len(list(folder.glob("*.frames.vpy"))) == 1
    cast(Context, ctx).run_cleanups()
    assert list(folder.glob("*.frames.vpy")) == []


@pytest.mark.parametrize(
    "output",
    [
        "core.std.BlankClip(format=vs.RGB24, width=8, height=6, length=2, color=[9, 99, 199])",
        "core.std.BlankClip(format=vs.RGBS, width=8, height=6, length=2, color=[0.1, 0.5, 0.9]).set_output(0)",
        # RGBS with an alpha clip, which vspipe would write after the colour planes.
        (
            "core.std.BlankClip(format=vs.RGBS, width=8, height=6, length=2, color=[0.1, 0.5, 0.9])"
            ".set_output(0, alpha=core.std.BlankClip(format=vs.GRAYS, width=8, height=6,"
            " length=2, color=0.25))"
        ),
        "core.std.BlankClip(format=vs.GRAYH, width=8, height=6, length=2, color=0.3)",
        # HD YUV without _Matrix: BT.709.
        (
            "core.std.BlankClip(format=vs.YUV420P10, width=1280, height=720, length=2,"
            " color=[300, 400, 700])"
        ),
    ],
)
def test_read_frames_with_info_matches_without(tmp_path: Path, vs: Any, output: str):
    script = tmp_path / "formats.vpy"
    if ".set_output(" not in output:
        output += ".set_output(0)"
    script.write_text(f"import vapoursynth as vs\ncore = vs.core\n{output}\n", "utf-8")

    plain = list(read_frames(script, 0, 0, 2))
    reused = list(read_frames(script, 0, 0, 2, info=clip_info(script, 0)))
    assert len(reused) == 2
    for a, b in zip(plain, reused, strict=True):
        np.testing.assert_array_equal(a, b)


# The FFmpeg the backend downloads to its storage on first use (FFMpegEnv.get_integrated).
FFMPEG_DIR = (
    Path(os.environ.get("APPDATA") or Path.home() / ".config")
    / "chaiNNer"
    / "backend-storage"
)


@pytest.fixture
def integrated_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> FFMpegEnv:
    """The downloaded FFmpeg as the nodes' integrated one (their storage is tmp_path)."""
    ffmpeg, ffprobe = map(str, get_executable_path(FFMPEG_DIR / "ffmpeg"))
    if not Path(ffmpeg).exists():
        pytest.skip("the integrated FFmpeg is not downloaded")
    env = FFMpegEnv(ffmpeg, ffprobe)
    monkeypatch.setattr(FFMpegEnv, "get_integrated", staticmethod(lambda _: env))
    return env


def collect(
    storage: Path,
    items: list[tuple[np.ndarray | None, ...]],
    source: str = "frames",
    precision: str = "16",
    colour: ColourInfo | None = None,
    choices: tuple[Enum, ...] = (),
) -> VideoClip:
    """A run of Frames to Clip, node "collect", at 23.976 fps over `items` (Frames, Y,
    U, V each), with these settings and colour choices (the rest Auto)."""
    from packages.chaiNNer_vapoursynth.vapoursynth.frames_to_clip import (
        ClipPrecision,
        ClipSource,
        frames_to_clip_node,
    )

    chosen: dict[type, Any] = {type(choice): choice for choice in choices}
    ctx = context(storage, "collect")
    collector = frames_to_clip_node(
        ctx,
        ClipSource(source),
        None,
        ClipPrecision(precision),
        None,
        None,
        None,
        colour,
        chosen.get(Matrix, Matrix.AUTO),
        chosen.get(Transfer, Transfer.AUTO),
        chosen.get(Primaries, Primaries.AUTO),
        chosen.get(Range, Range.AUTO),
        chosen.get(ChromaLocation, ChromaLocation.AUTO),
        23.976,
    )
    try:
        for item in items:
            collector.on_iterate(item)
        return collector.on_complete()
    finally:
        cast(Context, ctx).run_cleanups()


@pytest.mark.parametrize(
    ("precision", "channels", "format_name"),
    [
        ("16", 3, "RGB48"),
        ("16", 1, "Gray16"),
        ("32", 3, "RGBS"),
        ("32", 1, "GrayS"),
    ],
)
def test_frames_to_clip_roundtrip(
    tmp_path: Path,
    vs: Any,
    integrated_ffmpeg: FFMpegEnv,
    precision: str,
    channels: int,
    format_name: str,
):
    rng = np.random.default_rng(5)
    # An odd size, values between the 16-bit levels and outside 0-1. Gray frames come
    # as ImageInput gives them: HxW.
    shape = (5, 7, 3) if channels == 3 else (5, 7)
    images = [rng.uniform(-0.25, 1.25, shape).astype(np.float32) for _ in range(3)]
    items = [(image, None, None, None) for image in images]

    first = collect(tmp_path, items, precision=precision)
    assert "core.bs.VideoSource(" in first.stmt
    assert first.var == clip_var("collect")
    assert first.deps == ()
    folder = tmp_path / "vapoursynth" / "chains" / "collect"
    suffix = ".nut" if precision == "32" else ".mkv"
    assert [p.suffix for p in folder.glob("frames-*")] == [suffix]

    script = tmp_path / "roundtrip.vpy"
    script.write_text(compile_script([(first, "Frames")], cache_mb=256), "utf-8")
    info = clip_info(script, 0)
    assert (info.width, info.height, info.num_frames) == (7, 5, 3)
    assert (info.fps_num, info.fps_den) == (24000, 1001)
    assert info.format_name == format_name
    for image, frame in zip(images, read_frames(script, 0, 0, 3), strict=True):
        assert frame.shape == (5, 7, channels)
        values = frame.reshape(image.shape)
        if precision == "32":
            # Every value as it was, also outside 0-1.
            np.testing.assert_array_equal(values, image)
        else:
            # Clipped to 0-1, within half a 16-bit step (and float32's rounding).
            expected = np.clip(image, 0, 1)
            assert np.abs(values - expected).max() <= 0.5 / 65535 + 1e-7

    # The next run of the same node replaces the previous run's file.
    second = collect(tmp_path, items, precision=precision)
    assert second.stmt != first.stmt
    assert len(list(folder.glob("frames-*"))) == 1


def random_planes(
    rng: np.random.Generator,
    size: tuple[int, int],
    subsampling: tuple[int, int],
    full_range: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Every 16-bit code at random in Y, U, V planes of a luma `size` (width, height),
    and the planes' values (colour_info's convention)."""
    width, height = size
    chroma = (height >> subsampling[1], width >> subsampling[0])
    codes = [
        rng.integers(0, 65536, shape, dtype=np.uint16)
        for shape in ((height, width), chroma, chroma)
    ]
    values = [
        code_to_float(c, 16, full_range, chroma=i > 0) for i, c in enumerate(codes)
    ]
    return codes, values


@pytest.mark.parametrize(
    ("subsampling", "colour", "choices", "props"),
    [
        # 4:2:0 HDR10, from a Colour Info: its tags.
        (
            (1, 1),
            ColourInfo(
                Matrix.BT2020_NCL,
                Transfer.PQ,
                Primaries.BT2020,
                Range.LIMITED,
                ChromaLocation.TOP_LEFT,
                10,
                (8, 6),
                (4, 3),
            ),
            (),
            (9, 16, 9, 0, 2),
        ),
        # 4:2:2 without Colour Info: the guesses for an 8x6 picture (BT.601's matrix,
        # BT.709's primaries and transfer), full range and centre siting chosen.
        ((1, 0), None, (Range.FULL, ChromaLocation.CENTER), (6, 1, 1, 1, 1)),
        # 4:4:4 with a choice over the Colour Info's matrix; bottom-left siting, which
        # Matroska cannot store, set by the statement; the transfer that FFmpeg's
        # option calls gamma22.
        (
            (0, 0),
            ColourInfo(
                Matrix.SMPTE170M,
                Transfer.BT470M,
                Primaries.DISPLAY_P3,
                Range.FULL,
                ChromaLocation.BOTTOM_LEFT,
                8,
                (8, 6),
                (8, 6),
            ),
            (Matrix.BT709,),
            (1, 4, 12, 1, 4),
        ),
    ],
)
def test_frames_to_clip_planes(
    tmp_path: Path,
    vs: Any,
    integrated_ffmpeg: FFMpegEnv,
    subsampling: tuple[int, int],
    colour: ColourInfo | None,
    choices: tuple[Enum, ...],
    props: tuple[int, ...],
):
    rng = np.random.default_rng(7)
    full_range = props[3] == 1
    frames = [random_planes(rng, (8, 6), subsampling, full_range) for _ in range(3)]
    clip_record = collect(
        tmp_path,
        [(None, *values) for _, values in frames],
        "planes",
        colour=colour,
        choices=choices,
    )

    # The clip as VapourSynth makes it from the node's statement: the planes' codes
    # exactly, and the colour tags as frame properties.
    scope = {"core": vs.core, "vs": vs}
    exec(clip_record.stmt, scope)
    clip = scope[clip_record.var]
    layout = {(0, 0): "444", (1, 0): "422", (1, 1): "420"}[subsampling]
    assert clip.format.name == f"YUV{layout}P16"
    assert (clip.width, clip.height, clip.num_frames) == (8, 6, 3)
    for n, (codes, _) in enumerate(frames):
        frame = clip.get_frame(n)
        for plane, expected in enumerate(codes):
            np.testing.assert_array_equal(np.asarray(frame[plane]), expected)
        keys = ("_Matrix", "_Transfer", "_Primaries", "_Range", "_ChromaLocation")
        assert tuple(int(frame.props[k]) for k in keys) == props


# The local gates' FFmpeg (native/tools/bench_oracle.py), newer than the integrated one.
NEWER_FFMPEG = Path(r"C:\Executables\ffmpeg-8.1-full_build-shared\bin\ffmpeg.exe")


def test_frames_to_clip_planes_keep_their_codes_on_a_newer_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """FFmpeg 8.1 converts frames whose colour tags differ from the output's, where the
    integrated 5.1.2 leaves them: Frames to Clip gives its input the planes' tags too,
    so the file holds every code of the planes."""
    if not NEWER_FFMPEG.exists():
        pytest.skip("the local gates' FFmpeg is not installed")
    env = FFMpegEnv(str(NEWER_FFMPEG), str(NEWER_FFMPEG.with_name("ffprobe.exe")))
    monkeypatch.setattr(FFMpegEnv, "get_integrated", staticmethod(lambda _: env))
    rng = np.random.default_rng(7)
    frames = [random_planes(rng, (8, 6), (1, 1), False) for _ in range(3)]
    colour = ColourInfo(
        Matrix.BT709,
        Transfer.BT709,
        Primaries.BT709,
        Range.LIMITED,
        ChromaLocation.LEFT,
        10,
        (8, 6),
        (4, 3),
    )
    planes = [(None, *values) for _, values in frames]
    collect(tmp_path, planes, "planes", colour=colour)
    (video,) = (tmp_path / "vapoursynth" / "chains" / "collect").glob("frames-*.mkv")
    command = [str(NEWER_FFMPEG), "-v", "error", "-i", str(video), "-f", "rawvideo"]
    command += ["-pix_fmt", "yuv420p16le", "-"]
    decoded = subprocess.run(
        command,
        capture_output=True,
        check=True,
        stdin=subprocess.DEVNULL,
    ).stdout
    expected = b"".join(c.astype("<u2").tobytes() for codes, _ in frames for c in codes)
    assert decoded == expected


def test_frames_to_clip_errors(
    tmp_path: Path, integrated_ffmpeg: FFMpegEnv, monkeypatch: pytest.MonkeyPatch
):
    from packages.chaiNNer_vapoursynth.vapoursynth import frames_to_clip

    rgb = np.zeros((4, 6, 3), np.float32)
    with pytest.raises(ValueError, match="got no frames"):
        collect(tmp_path, [])
    with pytest.raises(
        ValueError, match=r"were 6x4 with 3 channels, these are 6x4 with 1"
    ):
        collect(tmp_path, [(rgb, None, None, None), (rgb[:, :, 0], None, None, None)])
    with pytest.raises(ValueError, match="needs Frames"):
        collect(tmp_path, [(None, None, None, None)])

    y, u = np.zeros((6, 8), np.float32), np.zeros((3, 4), np.float32)
    with pytest.raises(ValueError, match="needs Y, U and V"):
        collect(tmp_path, [(rgb, y, u, None)], "planes")
    with pytest.raises(ValueError, match="U is 4x3, V is 8x6"):
        collect(tmp_path, [(None, y, u, y)], "planes")
    with pytest.raises(ValueError, match="not a Y plane 8x6 with U and V planes 2x6"):
        collect(tmp_path, [(None, y, y[:, :2], y[:, :2])], "planes")
    with pytest.raises(ValueError, match=r"4:2:0 planes to be a multiple of 2x2.* 7x6"):
        collect(tmp_path, [(None, y[:, :7], u, u)], "planes")
    with pytest.raises(ValueError, match=r"RGB \(identity\)"):
        collect(tmp_path, [(None, y, u, u)], "planes", choices=(Matrix.RGB,))
    with pytest.raises(ValueError, match="were 8x6 with 1 channels, 4x3 with 1"):
        collect(tmp_path, [(None, y, u, u), (None, y, y, y)], "planes")

    # FFmpeg rejects an option (another codec's tag): its exit code (1 from FFmpeg 5.1,
    # the error's own code from FFmpeg 7) and its message.
    monkeypatch.setattr(frames_to_clip, "GBRPF32_NUT_TAG", "FFV1")
    with pytest.raises(RuntimeError, match=r"FFmpeg stopped .*exit code \d+\).*FFV1"):
        collect(tmp_path, [(rgb, None, None, None)] * 3, precision="32")


@pytest.mark.parametrize(
    ("size", "planes", "sliced"),
    [
        ((1920, 1080), 3, True),
        # 7 pixels a side at least: FFmpeg refuses 16 slices below 4 and misplaces
        # slices from 4 to 6.
        ((7, 7), 3, True),
        ((6, 1080), 3, False),
        ((1920, 6), 1, False),
        ((3, 3), 1, False),
        # A slice of at most 2^27 bits at 17 bits a sample: 2,631,720 samples a plane
        # of three planes, 7,895,160 of one. A side not a multiple of 4 rounds up.
        ((4 * 1620, 4 * 1624), 3, True),
        ((4 * 1620 + 1, 4 * 1624), 3, False),
        ((4 * 1620 + 1, 4 * 1624), 1, True),
        ((4 * 2809, 4 * 2810), 1, True),
        ((4 * 2810, 4 * 2810), 1, False),
    ],
)
def test_frames_to_clip_ffv1_slices(size: tuple[int, int], planes: int, sliced: bool):
    from packages.chaiNNer_vapoursynth.vapoursynth.frames_to_clip import ffv1_options

    slices = ["-level", "3", "-slices", "16"] if sliced else []
    assert ffv1_options(*size, planes) == ["-c:v", "ffv1", *slices]


def ffv1_commands(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The FFmpeg commands that encode FFV1 from here on."""
    commands: list[list[str]] = []
    popen = subprocess.Popen

    def recording(command: list[str], **kwargs: Any) -> Any:
        if "ffv1" in command:
            commands.append(command)
        return popen(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", recording)
    return commands


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("channels", [3, 1])
@pytest.mark.parametrize(
    ("size", "sliced"),
    [
        # Below 7 pixels a side: one slice, as before; from 7 (odd sizes too): 16.
        ((6, 9), False),
        ((9, 6), False),
        ((7, 7), True),
        ((33, 17), True),
        ((1920, 1080), True),
    ],
)
def test_frames_to_clip_lossless_around_slice_threshold(
    tmp_path: Path,
    vs: Any,
    integrated_ffmpeg: FFMpegEnv,
    monkeypatch: pytest.MonkeyPatch,
    bits: int,
    channels: int,
    size: tuple[int, int],
    sliced: bool,
):
    commands = ffv1_commands(monkeypatch)
    rng = np.random.default_rng(11)
    width, height = size
    shape = (height, width, channels) if channels == 3 else (height, width)
    # Random codes of the bit depth, as 16-bit codes (8-bit: x257); frames of their
    # values.
    step = 65535 // ((1 << bits) - 1)
    codes = [
        rng.integers(0, 1 << bits, shape, dtype=np.uint16) * step for _ in range(2)
    ]
    record = collect(
        tmp_path, [(c.astype(np.float32) / 65535, None, None, None) for c in codes]
    )
    assert len(commands) == 1
    assert ("-slices" in commands[0]) == sliced

    scope = {"core": vs.core, "vs": vs}
    exec(record.stmt, scope)
    clip = scope[record.var]
    assert clip.format.name == ("RGB48" if channels == 3 else "Gray16")
    for n, code in enumerate(codes):
        frame = clip.get_frame(n)
        # R, G, B planes of chaiNNer's BGR.
        planes = (
            [code[:, :, 2], code[:, :, 1], code[:, :, 0]] if channels == 3 else [code]
        )
        for plane, expected in enumerate(planes):
            np.testing.assert_array_equal(np.asarray(frame[plane]), expected)


@pytest.mark.parametrize(
    ("subsampling", "size", "sliced"),
    [
        ((0, 0), (6, 8), False),
        ((0, 0), (7, 7), True),
        ((1, 0), (8, 6), False),
        # 4:2:2 at 8 wide: 4 chroma samples a row, one a slice.
        ((1, 0), (8, 8), True),
        ((1, 1), (6, 8), False),
        ((1, 1), (8, 8), True),
        ((1, 1), (1920, 1080), True),
    ],
)
def test_frames_to_clip_planes_lossless_around_slice_threshold(
    tmp_path: Path,
    vs: Any,
    integrated_ffmpeg: FFMpegEnv,
    monkeypatch: pytest.MonkeyPatch,
    subsampling: tuple[int, int],
    size: tuple[int, int],
    sliced: bool,
):
    commands = ffv1_commands(monkeypatch)
    rng = np.random.default_rng(13)
    frames = [random_planes(rng, size, subsampling, True) for _ in range(2)]
    record = collect(
        tmp_path,
        [(None, *values) for _, values in frames],
        "planes",
        choices=(Range.FULL,),
    )
    assert len(commands) == 1
    assert ("-slices" in commands[0]) == sliced

    scope = {"core": vs.core, "vs": vs}
    exec(record.stmt, scope)
    clip = scope[record.var]
    assert (clip.width, clip.height, clip.num_frames) == (*size, 2)
    for n, (codes, _) in enumerate(frames):
        frame = clip.get_frame(n)
        for plane, expected in enumerate(codes):
            np.testing.assert_array_equal(np.asarray(frame[plane]), expected)


def test_save_script_runs_in_vspipe(tmp_path: Path, vs: Any):
    from packages.chaiNNer_vapoursynth.vapoursynth.save_script import save_script_node

    source = clip("src", BLANK)
    gray = clip(
        "gray",
        "core.std.BlankClip(width=32, height=16, length=7, format=vs.GRAY16)",
    )
    folder = tmp_path / "it's ü"
    save_script_node(
        context(tmp_path, "save"),
        folder,
        "my script",
        *pairs(source, "Source", gray, None),
    )

    script = folder / "my script.vpy"
    text = script.read_text(encoding="utf-8")
    assert f"set_output({source.var}, 0, 'Source')" in text
    assert f"set_output({gray.var}, 1, 'Clip 2')" in text
    # The first pair is output 0.
    assert clip_info(script, 0).format_name == "RGB24"
    assert clip_info(script, 1).format_name == "Gray16"
    done = subprocess.run(
        [str(runner.vspipe()), "-o", "1", "-c", "y4m", str(script), "-"],
        env=runner.vs_env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert done.stdout.startswith(b"YUV4MPEG2 Cmono16 W32 H16")


def test_vspipe_failure_names_the_node(tmp_path: Path, vs: Any):
    from packages.chaiNNer_vapoursynth.vapoursynth.clip_frames import (
        clip_frames_node,
    )

    source = clip("src", BLANK)
    # An error when vspipe evaluates the script: the node run fails.
    bad = clip("bad", f"{clip_var('src')}.std.Crop(left=-4)", source)
    with pytest.raises(VSScriptError, match="negative corner coordinates") as raised:
        clip_frames_node(context(tmp_path, "frames"), bad, VideoMode.RGB)
    assert raised.value.node_id == "bad"

    # An error at frame 2: the iteration fails (vspipe renders ahead in parallel, so
    # frames 0 and 1 may or may not come first).
    late = clip(
        "late",
        f"{clip_var('src')}.std.ModifyFrame({clip_var('src')},"
        " lambda n, f: f if n < 2 else 1 / 0)",
        source,
    )
    ctx = context(tmp_path, "frames")
    generator, _, _ = clip_frames_node(ctx, late, VideoMode.RGB)
    delivered = []
    with pytest.raises(VSScriptError, match="ZeroDivisionError") as raised:
        for item in generator.supplier():
            delivered.append(item)
    cast(Context, ctx).run_cleanups()
    assert raised.value.node_id == "late"
    assert len(delivered) <= 2


def test_plugin_frame_error_shows_on_clip_frames(tmp_path: Path, vs: Any):
    from packages.chaiNNer_vapoursynth.vapoursynth.clip_frames import (
        clip_frames_node,
    )
    from process import NodeExecutionError, on_failing_node

    # A plugin's error at frame time has no line of the script in its message: YUV
    # without a matrix cannot become RGB, which the resizer finds per frame.
    source = clip("src", "core.std.BlankClip(format=vs.YUV420P8, width=64, height=48)")
    rgb = clip("rgb", f"{clip_var('src')}.resize.Bicubic(format=vs.RGB24)", source)
    ctx = context(tmp_path, "frames")
    generator, _, colour = clip_frames_node(ctx, rgb, VideoMode.RGB)
    # The first frame fails: no colour properties, every value guessed.
    assert {colour.origin_of(name) for name in FIELDS} == {"guessed"}
    with pytest.raises(VSScriptError, match="Failed to retrieve frame") as raised:
        for _ in generator.supplier():
            pass
    cast(Context, ctx).run_cleanups()
    # The iteration's error arrives unwrapped: it goes to the node that ran vspipe.
    assert raised.value.node_id == "frames"
    assert raised.value.__traceback__ is not None
    data = SimpleNamespace(schema_id="chainner:vapoursynth:clip_frames")
    nodes = cast(Any, {"frames": SimpleNamespace(id="frames", data=data)})
    moved = on_failing_node(raised.value, nodes)
    assert isinstance(moved, NodeExecutionError)
    assert (moved.node_id, moved.__cause__) == ("frames", raised.value)


@pytest.mark.parametrize(
    ("format_name", "pix_fmt"),
    [
        ("YUV420P8", "yuv420p"),
        ("YUV422P10", "yuv422p10le"),
        ("YUV444PS", "yuv444p16le"),
        ("YUV410P8", "yuv410p"),
        ("Gray8", "gray"),
        ("GrayH", "gray16le"),
        ("RGB24", "gbrp"),
        ("RGB30", "gbrp10le"),
        ("RGBS", "gbrp16le"),
        ("Variable", ""),
    ],
)
def test_vapoursynth_formats_as_ffmpegs(format_name: str, pix_fmt: str):
    from nodes.impl.vapoursynth.colour_props import vs_pix_fmt

    assert vs_pix_fmt(format_name) == pix_fmt


def test_colour_props_as_an_ffprobe_entry():
    from nodes.impl.vapoursynth.colour_props import colour_stream

    props = {"_Matrix": 9, "_Transfer": 16, "_Primaries": 2, "_ChromaLocation": 2}
    assert colour_stream(props | {"_Range": 0}, "yuv420p10le", 64, 48) == {
        "pix_fmt": "yuv420p10le",
        "width": 64,
        "height": 48,
        "color_space": "bt2020nc",
        "color_transfer": "smpte2084",
        # Unspecified (2) is no tag; for the siting, 2 is top-left.
        "color_range": "tv",
        "chroma_location": "topleft",
    }
    # Codes FFmpeg has no name for keep their number.
    stream = colour_stream({"_Transfer": 99, "_Range": 1}, "gray", 8, 8)
    assert (stream["color_transfer"], stream["color_range"]) == ("H.273 code 99", "pc")


def test_clip_frames_item_types_follow_the_mode():
    from packages.chaiNNer_vapoursynth.vapoursynth.clip_frames import (
        FramesInfo,
        get_item_types,
    )

    clip_record = runner.ClipInfo(64, 48, 5, 24, 1, "YUV420P10")
    colour = ColourInfo(
        Matrix.BT709,
        Transfer.BT709,
        Primaries.BT709,
        Range.LIMITED,
        ChromaLocation.LEFT,
        10,
        (64, 48),
        (32, 24),
    )
    import navi

    image = {"width": 64, "height": 48}
    for mode, outputs in (
        (VideoMode.RGB, {0: 3}),
        (VideoMode.PLANES, {3: 1, 4: 1, 5: 1}),
        (VideoMode.BOTH, {0: 3, 3: 1, 4: 1, 5: 1}),
    ):
        types = get_item_types(FramesInfo(clip_record, mode, colour))
        assert set(types) == set(outputs)
        for output, channels in outputs.items():
            size = image if output in (0, 3) else {"width": 32, "height": 24}
            expected = navi.Image(**size, channels=channels)
            assert types[cast(Any, output)] == expected


def test_clip_frames_item_types_of_a_variable_size_clip():
    import navi
    from packages.chaiNNer_vapoursynth.vapoursynth.clip_frames import (
        FramesInfo,
        get_item_types,
    )

    # vspipe reports a variable size as 0x0: every plane's size is then unknown.
    clip_record = runner.ClipInfo(0, 0, 5, 24, 1, "YUV420P10")
    colour, _ = detect_colour_info(colour_stream({}, "yuv420p10le", 0, 0))
    types = get_item_types(FramesInfo(clip_record, VideoMode.PLANES, colour))
    assert set(types) == {3, 4, 5}
    for output in (3, 4, 5):
        assert types[cast(Any, output)] == navi.Image(channels=1)


def frames_items(
    storage: Path, record: VideoClip, mode: VideoMode, node_id: str = "frames"
) -> tuple[list[tuple[Any, ...]], ColourInfo]:
    """A run of Clip Frames over `record`: its items and its Colour Info."""
    from packages.chaiNNer_vapoursynth.vapoursynth.clip_frames import (
        clip_frames_node,
    )

    ctx = context(storage, node_id)
    generator, _, colour = clip_frames_node(ctx, record, mode)
    try:
        return cast(list[tuple[Any, ...]], list(generator.supplier())), colour
    finally:
        cast(Context, ctx).run_cleanups()


@pytest.mark.parametrize(
    ("subsampling", "full_range"), [((1, 1), False), ((1, 0), True), ((0, 0), False)]
)
def test_planes_roundtrip_through_frames_to_clip_and_clip_frames(
    tmp_path: Path,
    vs: Any,
    integrated_ffmpeg: FFMpegEnv,
    subsampling: tuple[int, int],
    full_range: bool,
):
    rng = np.random.default_rng(11)
    frames = [random_planes(rng, (8, 6), subsampling, full_range) for _ in range(3)]
    chroma_size = (8 >> subsampling[0], 6 >> subsampling[1])
    tags = ColourInfo(
        Matrix.BT709,
        Transfer.HLG,
        Primaries.BT2020,
        Range.FULL if full_range else Range.LIMITED,
        ChromaLocation.TOP,
        10,
        (8, 6),
        chroma_size,
    )
    record = collect(
        tmp_path, [(None, *values) for _, values in frames], "planes", colour=tags
    )

    items, colour = frames_items(tmp_path, record, VideoMode.PLANES)
    # Every plane value exactly as it went in.
    assert [item[1] for item in items] == [0, 1, 2]
    for (frame, _, *planes), (_, values) in zip(items, frames, strict=True):
        assert frame is None
        for plane, expected in zip(planes, values, strict=True):
            assert plane.dtype == np.float32
            np.testing.assert_array_equal(plane, expected)
    # The clip's Colour Info: its tags, from the file, at 16 bits and the clip's sizes.
    assert colour == ColourInfo(
        Matrix.BT709,
        Transfer.HLG,
        Primaries.BT2020,
        tags.range,
        ChromaLocation.TOP,
        16,
        (8, 6),
        chroma_size,
        origin=tuple((name, "file") for name in sorted(FIELDS)),
    )

    # Both: RGB frames as Mode RGB gives them, and the same planes, from one render.
    rgb, _ = frames_items(tmp_path, record, VideoMode.RGB)
    both, _ = frames_items(tmp_path, record, VideoMode.BOTH)
    for rgb_item, both_item, planes_item in zip(rgb, both, items, strict=True):
        assert both_item[0].shape == (6, 8, 3)
        np.testing.assert_array_equal(both_item[0], rgb_item[0])
        for got, expected in zip(both_item[2:], planes_item[2:], strict=True):
            np.testing.assert_array_equal(got, expected)


def test_clip_frames_planes_of_float_and_retagged_clips(
    tmp_path: Path, vs: Any, caplog: pytest.LogCaptureFixture
):
    # Float samples as they are, VapourSynth's zero-centred chroma moved to 0.5;
    # no properties: every value guessed for 64x48.
    float_clip = clip(
        "float",
        "core.std.BlankClip(format=vs.YUV444PS, width=64, height=48, length=2,"
        " color=[0.25, -0.125, 1.5])",
    )
    items, colour = frames_items(tmp_path, float_clip, VideoMode.PLANES)
    for _, _, y, u, v in items:
        assert (y.shape, u.shape, v.shape) == ((48, 64),) * 3
        for plane, value in ((y, 0.25), (u, 0.375), (v, 2.0)):
            np.testing.assert_array_equal(plane, np.full((48, 64), value, np.float32))
    assert (colour.matrix, colour.range, colour.bit_depth) == (
        Matrix.SMPTE170M,
        Range.LIMITED,
        16,
    )
    assert {colour.origin_of(name) for name in FIELDS} == {"guessed"}

    # A matrix code chaiNNer does not convert: guessed (BT.2020 beside a PQ transfer),
    # with a warning; the other properties as set. Limited 8-bit codes.
    tagged = clip(
        "tagged",
        "core.std.BlankClip(format=vs.YUV420P8, width=64, height=48, length=1,"
        " color=[235, 16, 128]).std.SetFrameProps(_Matrix=11, _Transfer=16,"
        " _Primaries=9, _Range=0, _ChromaLocation=2)",
    )
    with caplog.at_level(logging.WARNING):
        items, colour = frames_items(tmp_path, tagged, VideoMode.PLANES)
    assert any("smpte2085" in r.getMessage() for r in caplog.records)
    assert colour.matrix is Matrix.BT2020_NCL
    assert colour.origin_of("matrix") == "guessed"
    assert (colour.transfer, colour.primaries) == (Transfer.PQ, Primaries.BT2020)
    assert (colour.range, colour.chroma_location) == (
        Range.LIMITED,
        ChromaLocation.TOP_LEFT,
    )
    assert (colour.chroma_size, colour.bit_depth) == ((32, 24), 8)
    _, _, y, u, v = items[0]
    np.testing.assert_array_equal(y, np.ones((48, 64), np.float32))
    low = code_to_float(np.full((24, 32), 16, np.uint8), 8, False, chroma=True)
    np.testing.assert_array_equal(u, low)
    np.testing.assert_array_equal(v, np.full((24, 32), 0.5, np.float32))

    # Planes and Both need a YUV clip.
    rgb_clip = clip("rgb", "core.std.BlankClip(format=vs.RGB24, width=8, height=6)")
    for mode in (VideoMode.PLANES, VideoMode.BOTH):
        with pytest.raises(
            ValueError, match=r"needs a YUV clip, and this clip is RGB24"
        ):
            frames_items(tmp_path, rgb_clip, mode)
