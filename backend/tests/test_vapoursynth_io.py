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
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from api import MB, Aborted, NodeAttributedError, NodeContext, SettingsParser
from nodes.impl.vapoursynth import runner
from nodes.impl.vapoursynth.runner import VSScriptError, clip_info, read_frames
from nodes.impl.vapoursynth.script import VideoClip, clip_var, compile_script
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
    started: list[subprocess.Popen[bytes]] = []
    popen = subprocess.Popen

    def hanging(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
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
    """Every vsview the test's runs start; none is a real process. (Preview Clips
    checks its script in vspipe first.)"""
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
    generator, fps = clip_frames_node(ctx, frames)

    assert fps == 25
    assert generator.expected_length == 5
    items = cast(list[tuple[np.ndarray, int]], list(generator.supplier()))
    cast(Context, ctx).run_cleanups()
    # One --info for the length and FPS; the frames' run reuses it.
    assert len(info_runs) == 1
    assert [index for _, index in items] == list(range(5))
    for i, (frame, _) in enumerate(items):
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
FFMPEG_DIR = Path(os.environ.get("APPDATA", "")) / "chaiNNer" / "backend-storage"


def test_frames_to_clip_roundtrip_8bit(
    tmp_path: Path, vs: Any, monkeypatch: pytest.MonkeyPatch
):
    # Save Video's FFmpeg writer is native code, which GitHub's checks do not build.
    pytest.importorskip("nodes.impl._chainner_graph")
    from nodes.impl.ffmpeg import FFMpegEnv, get_executable_path
    from packages.chaiNNer_vapoursynth.vapoursynth.frames_to_clip import (
        frames_to_clip_node,
    )

    ffmpeg, ffprobe = map(str, get_executable_path(FFMPEG_DIR / "ffmpeg"))
    if not Path(ffmpeg).exists():
        pytest.skip("the integrated FFmpeg is not downloaded")
    # The node's storage is tmp_path; its FFmpeg is the downloaded one.
    monkeypatch.setattr(
        FFMpegEnv, "get_integrated", staticmethod(lambda _: FFMpegEnv(ffmpeg, ffprobe))
    )

    rng = np.random.default_rng(5)
    # An odd size, and values between the 8-bit levels.
    images = [rng.random((5, 7, 3), dtype=np.float32) for _ in range(3)]

    def run() -> VideoClip:
        ctx = context(tmp_path, "collect")
        collector = frames_to_clip_node(ctx, None, 23.976)
        for image in images:
            collector.on_iterate(image)
        result = collector.on_complete()
        cast(Context, ctx).run_cleanups()
        return result

    first = run()
    assert "core.bs.VideoSource(" in first.stmt
    assert first.var == clip_var("collect")
    assert first.deps == ()
    folder = tmp_path / "vapoursynth" / "chains" / "collect"
    assert len(list(folder.glob("*.mkv"))) == 1

    script = tmp_path / "roundtrip.vpy"
    script.write_text(compile_script([(first, "Frames")], cache_mb=256), "utf-8")
    info = clip_info(script, 0)
    assert (info.width, info.height, info.num_frames) == (7, 5, 3)
    assert (info.fps_num, info.fps_den) == (24000, 1001)
    assert info.format_name == "RGB24"
    for image, frame in zip(images, read_frames(script, 0, 0, 3), strict=True):
        assert frame.shape == (5, 7, 3)
        assert np.abs(frame - image).max() <= 1 / 255

    # The next run of the same node replaces the previous run's file.
    second = run()
    assert second.stmt != first.stmt
    assert len(list(folder.glob("*.mkv"))) == 1


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
        clip_frames_node(context(tmp_path, "frames"), bad)
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
    generator, _ = clip_frames_node(ctx, late)
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
    generator, _ = clip_frames_node(ctx, rgb)
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
