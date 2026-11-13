"""Load Videos: a folder of videos in one run (folder_videos.py), on both executors. A chain's files give the frames and
files of one Load Video run each; what does not depend on the file runs once; a bad file, a node's error on a frame and
Stop end what they should, with no partial output; the UI sees one run with the whole folder's progress."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import cv2
import numpy as np
import pytest
from engine_differential import KINDS, Result, execute
from scratch_storage import NO_FFMPEG, STORAGE
from test_engine_scenarios import Builder, items
from verify_runtime import edge, make_node

import api
from api import ExecutionOptions, NodeId
from chain.chain import Chain
from chain.json import JsonNode, parse_json
from chain.optimize import optimize
from events import EventQueue
from folder_videos import (
    _colliding_files,
    _Listing,
    _plan_folder,
    _video_items,
    run_executor,
)
from native_executor import NativeExecutor, ending_runs
from nodes.impl.ffmpeg import get_executable_path
from process import Executor, NodeExecutionError
from progress_controller import Aborted

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"
FFMPEG = get_executable_path(STORAGE / "ffmpeg")[0]
LOAD_VIDEOS = "chainner:image:load_videos"
LOAD_VIDEO = "chainner:image:load_video"
SAVE_VIDEO = "chainner:image:save_video"
# Load Videos' and Load Video's outputs, by position.
FRAME, INDEX, NAME, FPS, AUDIO, COLOUR, SUBDIRECTORY = 0, 1, 3, 4, 5, 9, 13
CLIPS = {
    "a.mkv": ("testsrc2=size=64x48:rate=30:duration=1", ()),
    "b.mkv": (
        "testsrc2=size=48x64:rate=25:duration=0.6",
        ("-f", "lavfi", "-i", "sine=duration=0.6"),
    ),
    "c.mkv": ("smptebars=size=32x32:rate=10:duration=1", ()),
}


def make_clip(path: Path, source: str, more: tuple[str, ...] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(FFMPEG),
            *("-nostdin", "-v", "error", "-f", "lavfi", "-i", source, *more),
            *("-c:v", "ffv1", "-c:a", "flac", str(path)),
        ],
        check=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Three clips of different sizes, lengths and frame rates, one with audio, beside a file that is no video."""
    if not FFMPEG.exists():
        pytest.skip(NO_FFMPEG)
    folder = tmp_path_factory.mktemp("clips")
    for name, (source, more) in CLIPS.items():
        make_clip(folder / name, source, more)
    (folder / "notes.txt").write_text("not a video")
    return folder


@pytest.fixture
def test_package(monkeypatch: pytest.MonkeyPatch):
    # Test nodes belong to no package: both executors read their settings from a stand-in.
    monkeypatch.setattr(
        api.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )


def save_video(schemas, output: Path, source: object, **more: Any) -> dict[str, Any]:
    """Save Video writing FFV1 in Matroska with the source's name, FPS, audio and colour."""
    values: dict[int, object] = {
        0: source,
        1: str(output),
        2: edge("load", NAME),
        16: 1,
        4: "mkv",
        3: "ffv1",
        14: edge("load", FPS),
        15: edge("load", AUDIO),
        23: edge("load", COLOUR),
    }
    values.update({int(key[1:]): value for key, value in more.items()})
    return make_node(schemas, "save", SAVE_VIDEO, values)


def real_chain(*nodes: dict[str, Any]) -> Chain:
    return parse_json(cast("list[JsonNode]", nodes))


def folder_node(schemas, folder: Path, **more: Any) -> dict[str, Any]:
    values: dict[int, object] = {0: str(folder)}
    values.update({int(key[1:]): value for key, value in more.items()})
    return make_node(schemas, "load", LOAD_VIDEOS, values)


def run_chain(kind: str, chain: Chain, **more: Any) -> Result:
    optimize(chain)
    return execute(kind, chain, storage=STORAGE, **more)


def frame_lines(video: Path) -> list[str]:
    """The video's frames and audio as FFmpeg decodes them, with their timestamps."""
    framemd5 = subprocess.run(
        [
            str(FFMPEG),
            *("-nostdin", "-v", "error", "-i", str(video)),
            *("-map", "0", "-f", "framemd5", "-"),
        ],
        check=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in framemd5.splitlines() if not line.startswith("#")]


def single_run(schemas, kind: str, video: Path, output: Path) -> Result:
    load = make_node(schemas, "load", LOAD_VIDEO, {0: str(video)})
    chain = real_chain(load, save_video(schemas, output, edge("load")))
    return run_chain(kind, chain)


def events_of(
    result: Result, name: str, node: str | None = None
) -> list[dict[str, Any]]:
    return [
        event["data"]
        for event in result.events
        if event["event"] == name
        and (node is None or event["data"].get("nodeId") == node)
    ]


def only_files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


@pytest.mark.parametrize("kind", KINDS)
def test_a_folder_gives_what_a_run_per_file_gives(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    together, apart = tmp_path / "together", tmp_path / "apart"
    chain = real_chain(
        folder_node(schemas, clips),
        save_video(schemas, together, edge("load")),
    )
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    for name in CLIPS:
        assert single_run(schemas, kind, clips / name, apart).error is None
    assert only_files(together) == sorted(CLIPS)
    for name in CLIPS:
        lines = frame_lines(together / name)
        assert lines
        assert lines == frame_lines(apart / name), name
    assert any(line.startswith("1,") for line in frame_lines(together / "b.mkv"))


@pytest.mark.parametrize("kind", KINDS)
def test_planes_mode_gives_what_a_run_per_file_gives(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    def planes_chain(load: dict[str, Any], output: Path) -> Chain:
        return real_chain(
            load,
            save_video(
                schemas,
                output,
                None,
                i19="planes",
                i20=edge("load", 6),
                i21=edge("load", 7),
                i22=edge("load", 8),
            ),
        )

    together, apart = tmp_path / "together", tmp_path / "apart"
    chain = planes_chain(folder_node(schemas, clips, i3="planes"), together)
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    for name in CLIPS:
        load = make_node(
            schemas, "load", LOAD_VIDEO, {0: str(clips / name), 3: "planes"}
        )
        assert run_chain(kind, planes_chain(load, apart)).error is None
    for name in CLIPS:
        assert frame_lines(together / name) == frame_lines(apart / name), name


@pytest.mark.parametrize("kind", KINDS)
def test_a_real_image_shared_by_the_files(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    picture = tmp_path / "picture.png"
    cv2.imwrite(str(picture), np.full((8, 8, 3), 200, np.uint8))

    def blended(load: dict[str, Any], output: Path) -> Chain:
        return real_chain(
            load,
            make_node(schemas, "picture", "chainner:image:load", {0: str(picture)}),
            make_node(
                schemas,
                "blend",
                "chainner:image:blend",
                {0: edge("load"), 1: edge("picture")},
            ),
            save_video(schemas, output, edge("blend")),
        )

    together, apart = tmp_path / "together", tmp_path / "apart"
    result = run_chain(kind, blended(folder_node(schemas, clips), together))
    assert result.error is None, result.error
    assert len(events_of(result, "node-finish", "picture")) == 1
    for name in CLIPS:
        load = make_node(schemas, "load", LOAD_VIDEO, {0: str(clips / name)})
        assert run_chain(kind, blended(load, apart)).error is None
    for name in CLIPS:
        assert frame_lines(together / name) == frame_lines(apart / name), name


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_a_temporal_node_starts_each_file_anew(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    seen: list[tuple[int, bool]] = []

    def temporal(context: Any, frame: np.ndarray):
        view = context.iteration
        assert view.iterated_by(0) == LOAD_VIDEO
        seen.append((view.index, view.state is None))
        view.state = True
        return frame

    chain = real_chain(folder_node(schemas, clips), save_video(schemas, tmp_path, None))
    b = Builder()
    b.chain = chain
    b.node("temporal", temporal, context=True)
    b.edge("load", FRAME, "temporal", 0)
    b.edge("temporal", 0, "save", 0)
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    assert only_files(tmp_path) == sorted(CLIPS)
    assert [index for index, _ in seen] == [*range(30), *range(15), *range(10)]
    assert all(fresh == (index == 0) for index, fresh in seen)


@pytest.mark.parametrize("kind", KINDS)
def test_the_ui_sees_one_run_with_the_folder_s_progress(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    chain = real_chain(
        folder_node(schemas, clips), save_video(schemas, tmp_path, edge("load"))
    )
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    starts = events_of(result, "chain-start")
    assert len(starts) == 1
    assert sorted(starts[0]["nodes"]) == ["load", "save"]
    assert len(events_of(result, "node-start", "load")) == 1
    assert len(events_of(result, "node-finish", "load")) == 1
    # Each file's Save Video ends: its status, in the UI, follows its file.
    assert len(events_of(result, "node-finish", "save")) == 3
    progress = events_of(result, "node-progress", "load")
    shares = [p["progress"] for p in progress]
    assert shares == sorted(shares)
    assert {p["total"] for p in progress} == {3}
    assert [p["index"] for p in progress] == sorted(p["index"] for p in progress)
    assert (progress[-1]["index"], progress[-1]["progress"], progress[-1]["eta"]) == (
        3,
        1,
        0,
    )
    # Inside its bytes' share the bar moves frame by frame.
    sizes = [os.stat(clips / name).st_size for name in sorted(CLIPS)]
    first = sizes[0] / sum(sizes)
    assert any(0 < share < first for share in shares)
    assert first in shares


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_only_what_depends_on_the_file_runs_per_file(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    calls: Counter[str] = Counter()
    hashes: Counter[tuple[int, str]] = Counter()
    models: set[int] = set()
    lock = threading.Lock()

    def shared_value():
        calls["shared"] += 1
        return SimpleNamespace(value=2)

    def before_shared():
        calls["before shared"] += 1
        return 1

    def named(name: str):
        calls["named"] += 1
        return f"{name}_out"

    def elsewhere():
        calls["elsewhere"] += 1

    def mark(frame: np.ndarray, index: int, model: SimpleNamespace):
        assert model.value == 2
        with lock:
            models.add(id(model))
            hashes[(index, hashlib.sha256(frame.tobytes()).hexdigest())] += 1
        return frame

    def build(load: dict[str, Any]) -> Chain:
        chain = real_chain(load, save_video(schemas, tmp_path, None, i2=None))
        b = Builder()
        b.chain = chain
        b.node("before", before_shared, inputs=0)
        b.node("shared", lambda _: shared_value(), side_effects=True)
        b.edge("before", 0, "shared", 0)
        b.node("named", named)
        b.edge("load", NAME, "named", 0)
        b.node("elsewhere", elsewhere, inputs=0, outputs=0, side_effects=True)
        b.node("mark", mark, inputs=3)
        b.edge("load", FRAME, "mark", 0)
        b.edge("load", INDEX, "mark", 1)
        b.edge("shared", 0, "mark", 2)
        b.edge("mark", 0, "save", 0)
        b.edge("named", 0, "save", 2)
        return chain

    result = run_chain(kind, build(folder_node(schemas, clips)))
    assert result.error is None, result.error
    # A pure name node runs twice for each file: in its run, and for the check before the runs.
    assert calls == {"before shared": 1, "shared": 1, "named": 6, "elsewhere": 1}
    assert only_files(tmp_path) == ["a_out.mkv", "b_out.mkv", "c_out.mkv"]
    assert sum(hashes.values()) == 30 + 15 + 10
    # One object for all the files, as the engines cached by a model need it.
    assert len(models) == 1
    together = hashes.copy()
    hashes.clear()
    for name in CLIPS:
        load = make_node(schemas, "load", LOAD_VIDEO, {0: str(clips / name)})
        assert run_chain(kind, build(load)).error is None
    assert hashes == together


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_a_lazy_input_s_branch_runs_only_when_read(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    calls: Counter[str] = Counter()

    def unread():
        calls["unread"] += 1
        return 1

    def read():
        calls["read"] += 1
        return 2

    def gate(frame: np.ndarray, never: Any, always: Any):
        assert always.value == 2
        return frame

    chain = real_chain(folder_node(schemas, clips), save_video(schemas, tmp_path, None))
    b = Builder()
    b.chain = chain
    b.node("unread", unread, inputs=0)
    b.node("read", read, inputs=0)
    b.node("gate", gate, inputs=3, lazy=(1, 2))
    b.edge("load", FRAME, "gate", 0)
    b.edge("unread", 0, "gate", 1)
    b.edge("read", 0, "gate", 2)
    b.edge("gate", 0, "save", 0)
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    # The native engine may compute a pure source ahead, in a file's run as in any run.
    assert calls["unread"] == (0 if kind == "python" else 3)
    # Once per file: a lazy input's source runs where the chain reads it.
    assert calls["read"] == 3
    assert only_files(tmp_path) == sorted(CLIPS)


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_a_connected_directory(schemas, clips: Path, tmp_path: Path, kind: str):
    chain = real_chain(
        folder_node(schemas, tmp_path / "unused"),
        save_video(schemas, tmp_path / "regular", edge("load")),
    )
    b = Builder()
    b.chain = chain
    b.node("directory", lambda: str(clips), inputs=0)
    b.edge("directory", 0, "load", 0)
    assert run_chain(kind, chain).error is None
    assert only_files(tmp_path / "regular") == sorted(CLIPS)
    # A generator's per-run output: read for the file list, its items never run.
    chain = real_chain(
        folder_node(schemas, tmp_path / "unused"),
        save_video(schemas, tmp_path / "generator", edge("load")),
    )
    b = Builder()
    b.chain = chain
    b.node(
        "generator",
        lambda: (items([1]), str(clips)),
        kind="generator",
        inputs=0,
        outputs=2,
    )
    b.edge("generator", 1, "load", 0)
    assert run_chain(kind, chain).error is None
    assert only_files(tmp_path / "generator") == sorted(CLIPS)


@pytest.mark.parametrize("kind", KINDS)
def test_two_folders(schemas, clips: Path, tmp_path: Path, kind: str):
    other = tmp_path / "other"
    make_clip(other / "d.mkv", "testsrc2=size=32x32:rate=10:duration=0.2")
    save_other = make_node(
        schemas,
        SAVE_VIDEO,
        SAVE_VIDEO,
        {
            0: edge("second"),
            1: str(tmp_path / "out2"),
            2: edge("second", NAME),
            16: 1,
            4: "mkv",
            3: "ffv1",
        },
    )
    second = make_node(schemas, "second", LOAD_VIDEOS, {0: str(other)})
    chain = real_chain(
        folder_node(schemas, clips),
        save_video(schemas, tmp_path / "out1", edge("load")),
        second,
        save_other,
    )
    assert run_chain(kind, chain).error is None
    assert only_files(tmp_path / "out1") == sorted(CLIPS)
    assert only_files(tmp_path / "out2") == ["d.mkv"]
    # Feeding one node, they would have to pair their files.
    chain = real_chain(
        folder_node(schemas, clips),
        save_video(schemas, tmp_path / "out3", edge("second")),
        second,
    )
    result = run_chain(kind, chain)
    assert isinstance(result.error, NodeExecutionError)
    assert "cannot feed one node" in str(result.error)
    assert not (tmp_path / "out3").exists()


@pytest.mark.parametrize("kind", KINDS)
def test_a_bad_file_is_reported_at_the_end(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    folder = tmp_path / "in"
    folder.mkdir()
    for name in ("a.mkv", "c.mkv"):
        (folder / name).write_bytes((clips / name).read_bytes())
    (folder / "b.mp4").write_bytes(b"this is no video" * 100)
    output = tmp_path / "out"
    chain = real_chain(
        folder_node(schemas, folder), save_video(schemas, output, edge("load"))
    )
    result = run_chain(kind, chain)
    assert only_files(output) == ["a.mkv", "c.mkv"]
    assert frame_lines(output / "c.mkv")
    assert isinstance(result.error, Exception)
    message = str(result.error)
    assert message.startswith("Errors occurred during iteration:\n- ")
    assert str(folder / "b.mp4") in message
    assert "a.mkv" not in message and "c.mkv" not in message
    assert events_of(result, "node-progress", "load")[-1]["progress"] == 1


@pytest.mark.parametrize("kind", KINDS)
def test_stop_on_first_error_stops_at_the_bad_file(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    folder = tmp_path / "in"
    folder.mkdir()
    for name in ("a.mkv", "c.mkv"):
        (folder / name).write_bytes((clips / name).read_bytes())
    (folder / "b.mp4").write_bytes(b"this is no video" * 100)
    output = tmp_path / "out"
    chain = real_chain(
        folder_node(schemas, folder, i19=True),
        save_video(schemas, output, edge("load")),
    )
    result = run_chain(kind, chain)
    assert only_files(output) == ["a.mkv"]
    assert isinstance(result.error, NodeExecutionError)
    assert result.error.node_id == "load"
    assert str(result.error).startswith(f"{folder / 'b.mp4'}: ")


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("stop", [False, True])
def test_a_node_s_error_on_a_frame_leaves_no_file(
    schemas, clips: Path, tmp_path: Path, kind: str, stop: bool
):
    def fail(frame: np.ndarray, index: int, name: str):
        if name == "b" and index == 3:
            raise ValueError("frame 3 of b is bad")
        return frame

    chain = real_chain(
        folder_node(schemas, clips, i19=stop),
        save_video(schemas, tmp_path, None),
    )
    b = Builder()
    b.chain = chain
    b.node("fail", fail, inputs=3)
    b.edge("load", FRAME, "fail", 0)
    b.edge("load", INDEX, "fail", 1)
    b.edge("load", NAME, "fail", 2)
    b.edge("fail", 0, "save", 0)
    result = run_chain(kind, chain)
    assert only_files(tmp_path) == (["a.mkv"] if stop else ["a.mkv", "c.mkv"])
    assert "frame 3 of b is bad" in str(result.error)
    assert str(clips / "b.mkv") in str(result.error)


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_stop_ends_the_folder_in_its_file(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    reached, stopped = threading.Event(), threading.Event()

    def mark(frame: np.ndarray, name: str):
        if name == "b":
            reached.set()
            assert stopped.wait(60)
        return frame

    def stop(executor: Any) -> None:
        assert reached.wait(60)
        executor.kill()
        assert executor.progress.aborted
        stopped.set()

    chain = real_chain(folder_node(schemas, clips), save_video(schemas, tmp_path, None))
    b = Builder()
    b.chain = chain
    b.node("mark", mark, inputs=2)
    b.edge("load", FRAME, "mark", 0)
    b.edge("load", NAME, "mark", 1)
    b.edge("mark", 0, "save", 0)
    result = run_chain(kind, chain, during=stop)
    assert isinstance(result.error, Aborted)
    assert only_files(tmp_path) == ["a.mkv"]
    assert frame_lines(tmp_path / "a.mkv")
    assert not [e for e in events_of(result, "node-progress", "load") if e["index"] > 2]


@pytest.mark.usefixtures("test_package")
def test_stop_does_not_wait_for_a_slow_shared_node(
    schemas, clips: Path, tmp_path: Path
):
    """A model's load may take minutes, and the host's /kill allows seconds: the run ends as a stopped run does, with
    the node finishing in the background. (process.Executor waits for its nodes.)"""
    reached, released = threading.Event(), threading.Event()

    trace: list[str] = []

    def slow(context: Any):
        context.add_cleanup(lambda: trace.append("cleanup"), after="chain")
        reached.set()
        assert released.wait(60)
        trace.append("ended")
        return 2

    chain = real_chain(folder_node(schemas, clips), save_video(schemas, tmp_path, None))
    b = Builder()
    b.chain = chain
    b.node("slow", slow, inputs=0, side_effects=True, context=True)
    b.node("mix", lambda frame, _: frame, inputs=2)
    b.edge("load", FRAME, "mix", 0)
    b.edge("slow", 0, "mix", 1)
    b.edge("mix", 0, "save", 0)
    optimize(chain)

    async def main():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(4) as pool:
            run = run_executor(
                NativeExecutor,
                id="stopped",
                chain=chain,
                send_broadcast_data=True,
                options=ExecutionOptions.parse({}),
                loop=loop,
                queue=EventQueue(),
                storage_dir=STORAGE,
                pool=pool,
                pool_size=4,
            )
            task = asyncio.ensure_future(run.run())
            await loop.run_in_executor(None, reached.wait, 60)
            run.kill()
            try:
                with pytest.raises(Aborted):
                    await asyncio.wait_for(task, 10)
                assert not released.is_set()
            finally:
                released.set()
            for ending in ending_runs():
                await ending

    asyncio.run(main())
    assert only_files(tmp_path) == []
    # After the node that Stop left running, as a run's own cleanups are.
    assert trace == ["ended", "cleanup"]


@pytest.mark.usefixtures("test_package")
def test_a_shared_cleanup_waits_for_the_file_s_node_that_stop_left_running(
    schemas, clips: Path, tmp_path: Path
):
    """A stopped run's chain cleanups run after the node Stop left running, a folder's shared nodes' too: the file's node
    may still use what they free. (process.Executor waits for its nodes.)"""
    reached, released = threading.Event(), threading.Event()
    trace: list[str] = []

    def shared(context: Any):
        context.add_cleanup(lambda: trace.append("shared cleanup"), after="chain")
        return 2

    def per_file(frame: np.ndarray, _: int):
        reached.set()
        assert released.wait(60)
        trace.append("file's node ended")
        return frame

    chain = real_chain(folder_node(schemas, clips), save_video(schemas, tmp_path, None))
    b = Builder()
    b.chain = chain
    b.node("shared", shared, inputs=0, side_effects=True, context=True)
    b.node("mix", per_file, inputs=2)
    b.edge("load", FRAME, "mix", 0)
    b.edge("shared", 0, "mix", 1)
    b.edge("mix", 0, "save", 0)
    optimize(chain)

    async def main():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(4) as pool:
            run = run_executor(
                NativeExecutor,
                id="stopped",
                chain=chain,
                send_broadcast_data=True,
                options=ExecutionOptions.parse({}),
                loop=loop,
                queue=EventQueue(),
                storage_dir=STORAGE,
                pool=pool,
                pool_size=4,
            )
            task = asyncio.ensure_future(run.run())
            await loop.run_in_executor(None, reached.wait, 60)
            run.kill()
            try:
                with pytest.raises(Aborted):
                    await asyncio.wait_for(task, 10)
                assert trace == []
            finally:
                released.set()
            for ending in ending_runs():
                await ending

    asyncio.run(main())
    assert trace == ["file's node ended", "shared cleanup"]


@pytest.mark.parametrize("executor_class", [Executor, NativeExecutor])
def test_stop_does_not_wait_for_a_slow_listing(
    schemas,
    clips: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    executor_class: Any,
):
    """A folder on a slow disk: the host's /kill allows seconds, the listing may take minutes. It finishes in the
    background, where its result goes nowhere."""
    module = sys.modules["packages.chaiNNer_standard.image.video_frames.load_videos"]
    original = module.list_videos
    reached, released = threading.Event(), threading.Event()
    listed: list[list[Path]] = []

    def slow(*arguments: Any):
        reached.set()
        assert released.wait(60)
        listed.append(original(*arguments))
        return listed[-1]

    monkeypatch.setattr(module, "list_videos", slow)
    chain = real_chain(
        folder_node(schemas, clips), save_video(schemas, tmp_path, edge("load"))
    )
    optimize(chain)

    async def main():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(4) as pool:
            run = run_executor(
                executor_class,
                id="stopped",
                chain=chain,
                send_broadcast_data=True,
                options=ExecutionOptions.parse({}),
                loop=loop,
                queue=EventQueue(),
                storage_dir=STORAGE,
                pool=pool,
                pool_size=4,
            )
            task = asyncio.ensure_future(run.run())
            await loop.run_in_executor(None, reached.wait, 60)
            run.kill()
            try:
                with pytest.raises(Aborted):
                    await asyncio.wait_for(task, 5)
                assert not released.is_set()
            finally:
                released.set()

    asyncio.run(main())
    assert len(listed) == 1
    assert only_files(tmp_path) == []


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_pause_holds_the_folder(schemas, clips: Path, tmp_path: Path, kind: str):
    reached, go_on = threading.Event(), threading.Event()
    seen: list[str] = []

    def mark(frame: np.ndarray, name: str):
        seen.append(name)
        reached.set()
        assert go_on.wait(60)
        return frame

    def pause_and_resume(executor: Any) -> None:
        assert reached.wait(60)
        executor.pause()
        assert executor.progress.paused
        go_on.set()
        executor.resume()
        assert not executor.progress.paused

    chain = real_chain(folder_node(schemas, clips), save_video(schemas, tmp_path, None))
    b = Builder()
    b.chain = chain
    b.node("mark", mark, inputs=2)
    b.edge("load", FRAME, "mark", 0)
    b.edge("load", NAME, "mark", 1)
    b.edge("mark", 0, "save", 0)
    result = run_chain(kind, chain, during=pause_and_resume)
    assert result.error is None, result.error
    assert only_files(tmp_path) == sorted(CLIPS)
    assert set(seen) == {"a", "b", "c"}


@pytest.mark.parametrize("kind", KINDS)
def test_a_constant_name_would_overwrite_and_fails_before_any_frame(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    chain = real_chain(
        folder_node(schemas, clips),
        save_video(schemas, tmp_path, edge("load"), i2="same"),
    )
    result = run_chain(kind, chain)
    assert isinstance(result.error, NodeExecutionError)
    assert result.error.node_id == "save"
    assert "to one file" in str(result.error)
    assert only_files(tmp_path) == []
    # With one video, it overwrites nothing.
    chain = real_chain(
        folder_node(schemas, clips, i17=True, i18=1),
        save_video(schemas, tmp_path, edge("load"), i2="same"),
    )
    assert run_chain(kind, chain).error is None
    assert only_files(tmp_path) == ["same.mkv"]


@pytest.mark.parametrize("kind", KINDS)
def test_two_files_of_one_name_fail_before_any_frame(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    folder = tmp_path / "in"
    make_clip(folder / "one" / "x.mkv", "testsrc2=size=32x32:rate=10:duration=0.2")
    make_clip(folder / "two" / "x.mkv", "testsrc2=size=48x32:rate=10:duration=0.2")
    output = tmp_path / "out"
    output.mkdir()
    chain = real_chain(
        folder_node(schemas, folder, i15=True),
        save_video(schemas, output, edge("load")),
    )
    result = run_chain(kind, chain)
    assert isinstance(result.error, NodeExecutionError)
    assert result.error.node_id == "save"
    assert str(folder / "one" / "x.mkv") in str(result.error)
    assert only_files(output) == []


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_subdirectory_path_tells_files_of_one_name_apart(
    schemas, tmp_path: Path, kind: str
):
    if not FFMPEG.exists():
        pytest.skip(NO_FFMPEG)
    folder = tmp_path / "in"
    make_clip(folder / "one" / "x.mkv", "testsrc2=size=32x32:rate=10:duration=0.2")
    make_clip(folder / "two" / "x.mkv", "testsrc2=size=48x32:rate=10:duration=0.2")
    output = tmp_path / "out"
    chain = real_chain(
        folder_node(schemas, folder, i15=True),
        save_video(schemas, output, edge("load")),
    )
    b = Builder()
    b.chain = chain
    b.node("into", lambda subdirectory: output / subdirectory)
    b.edge("load", SUBDIRECTORY, "into", 0)
    b.edge("into", 0, "save", 1)
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    assert only_files(output) == ["one", "two"]
    assert only_files(output / "one") == ["x.mkv"]
    assert only_files(output / "two") == ["x.mkv"]
    assert frame_lines(output / "one" / "x.mkv") != frame_lines(
        output / "two" / "x.mkv"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_glob_and_limit_choose_the_files(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    chain = real_chain(
        folder_node(schemas, clips, i14=True, i16="[bc].*"),
        save_video(schemas, tmp_path / "glob", edge("load")),
    )
    assert run_chain(kind, chain).error is None
    assert only_files(tmp_path / "glob") == ["b.mkv", "c.mkv"]
    chain = real_chain(
        folder_node(schemas, clips, i17=True, i18=2),
        save_video(schemas, tmp_path / "limit", edge("load")),
    )
    assert run_chain(kind, chain).error is None
    assert only_files(tmp_path / "limit") == ["a.mkv", "b.mkv"]


@pytest.mark.parametrize("kind", KINDS)
def test_a_folder_without_videos_fails_at_the_node(schemas, tmp_path: Path, kind: str):
    (tmp_path / "notes.txt").write_text("not a video")
    chain = real_chain(
        folder_node(schemas, tmp_path),
        save_video(schemas, tmp_path / "out", edge("load")),
    )
    result = run_chain(kind, chain)
    assert isinstance(result.error, NodeExecutionError)
    assert result.error.node_id == "load"
    assert "has no video files" in str(result.error)


def test_a_folder_input_alone_is_no_collision(schemas, tmp_path: Path):
    # Get Native Resolution has side effects and an optional Export Folder, and no file name.
    chain = real_chain(
        folder_node(schemas, tmp_path),
        make_node(
            schemas, "native", "chainner:image:get_native_resolution", {0: edge("load")}
        ),
    )
    folder = _plan_folder(chain, cast("Any", chain.nodes[NodeId("load")]))
    assert "native" in folder.per_file
    paths = [tmp_path / "a.mkv", tmp_path / "b.mkv"]
    listing = _Listing(folder, paths, [1, 1], tmp_path, False)
    assert _colliding_files(chain, [_video_items(listing)], {}) is None


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_what_feeds_two_shared_nodes_runs_once(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    """A model's Load Model into two nodes the files read."""
    calls: Counter[str] = Counter()

    def source():
        calls["source"] += 1
        return 1

    chain = real_chain(folder_node(schemas, clips), save_video(schemas, tmp_path, None))
    b = Builder()
    b.chain = chain
    b.node("source", source, inputs=0, side_effects=True)
    b.node("left", lambda value: value + 1)
    b.node("right", lambda value: value + 2)
    b.node("use", lambda frame, _, __: frame, inputs=3)
    b.edge("source", 0, "left", 0)
    b.edge("source", 0, "right", 0)
    b.edge("load", FRAME, "use", 0)
    b.edge("left", 0, "use", 1)
    b.edge("right", 0, "use", 2)
    b.edge("use", 0, "save", 0)
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    assert calls == {"source": 1}
    assert len(events_of(result, "node-finish", "source")) == 1
    assert only_files(tmp_path) == sorted(CLIPS)


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_what_feeds_a_shared_node_and_the_rest_runs_once(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    calls: Counter[str] = Counter()

    def source():
        calls["source"] += 1
        return 1

    def sink(_: int):
        calls["sink"] += 1

    chain = real_chain(folder_node(schemas, clips), save_video(schemas, tmp_path, None))
    b = Builder()
    b.chain = chain
    b.node("source", source, inputs=0, side_effects=True)
    b.node("middle", lambda value: value + 1)
    b.node("use", lambda frame, _: frame, inputs=2)
    b.node("sink", sink, outputs=0, side_effects=True)
    b.edge("source", 0, "middle", 0)
    b.edge("source", 0, "sink", 0)
    b.edge("load", FRAME, "use", 0)
    b.edge("middle", 0, "use", 1)
    b.edge("use", 0, "save", 0)
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    assert calls == {"source": 1, "sink": 1}


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_an_unread_lazy_branch_of_a_shared_node_does_not_run_first(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    """The nodes a shared node reads run first, those it may leave unread only if it reads them."""
    calls: Counter[str] = Counter()

    def unread():
        calls["unread"] += 1
        return 1

    def gate(_: Any):
        return 2

    chain = real_chain(folder_node(schemas, clips), save_video(schemas, tmp_path, None))
    b = Builder()
    b.chain = chain
    b.node("unread", unread, inputs=0, side_effects=False)
    # Of no image or utility schema, so not a pure node: no engine computes it ahead either.
    b.chain.nodes[NodeId("unread")].data = replace(
        b.chain.nodes[NodeId("unread")].data, schema_id="chainner:test:unread"
    )
    b.node("gate", gate, lazy=(0,))
    b.node("use", lambda frame, _: frame, inputs=2)
    b.edge("unread", 0, "gate", 0)
    b.edge("load", FRAME, "use", 0)
    b.edge("gate", 0, "use", 1)
    b.edge("use", 0, "save", 0)
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    assert calls == {}


@pytest.mark.parametrize("kind", KINDS)
def test_load_images_into_the_directory_and_its_own_branch_starts_once(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    source = tmp_path / "in"
    make_clip(source / "a.mkv", "testsrc2=size=32x32:rate=10:duration=0.2")
    cv2.imwrite(str(source / "p.png"), np.full((8, 8, 3), 100, np.uint8))
    cv2.imwrite(str(source / "q.png"), np.full((8, 8, 3), 200, np.uint8))
    output, images = tmp_path / "out", tmp_path / "images"
    chain = real_chain(
        make_node(schemas, "images", "chainner:image:load_images", {0: str(source)}),
        make_node(
            schemas,
            "save image",
            "chainner:image:save",
            {0: edge("images"), 1: str(images), 3: edge("images", 3), 4: "png"},
        ),
        folder_node(schemas, tmp_path / "unused", i0=edge("images", 1)),
        save_video(schemas, output, edge("load")),
    )
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    assert only_files(output) == ["a.mkv"]
    assert only_files(images) == ["p.png", "q.png"]
    assert len(events_of(result, "node-start", "images")) == 1


@pytest.mark.parametrize("through_a_node", [False, True])
@pytest.mark.parametrize("kind", KINDS)
def test_a_collector_into_the_file_list_is_refused_by_name(
    schemas, clips: Path, tmp_path: Path, kind: str, through_a_node: bool
):
    """Load Videos lists its files before any loop runs, and a collector's value needs its loop."""
    nodes = [
        make_node(schemas, "range", "chainner:utility:range", {0: 0, 2: 2}),
        make_node(schemas, "sum", "chainner:utility:accumulate", {0: edge("range")}),
    ]
    limit = edge("sum")
    if through_a_node:
        nodes.append(
            make_node(
                schemas,
                "plus",
                "chainner:utility:math",
                {0: edge("sum"), 1: "add", 2: 0},
            )
        )
        limit = edge("plus")
    chain = real_chain(
        *nodes,
        folder_node(schemas, clips, i17=True, i18=limit),
        save_video(schemas, tmp_path, edge("load")),
    )
    result = run_chain(kind, chain)
    assert isinstance(result.error, NodeExecutionError)
    assert result.error.node_id == "load"
    assert "Limit" in str(result.error)
    assert "Accumulate" in str(result.error)
    assert only_files(tmp_path) == []


@pytest.mark.parametrize("kind", KINDS)
def test_a_folder_named_like_a_video_is_no_video(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    source = tmp_path / "in"
    make_clip(source / "a.mkv", "testsrc2=size=32x32:rate=10:duration=0.2")
    make_clip(source / "b.mkv", "testsrc2=size=32x32:rate=10:duration=0.2")
    make_clip(
        source / "0.mkv" / "inner.mkv", "testsrc2=size=48x32:rate=10:duration=0.2"
    )
    chain = real_chain(
        folder_node(schemas, source, i15=False),
        save_video(schemas, tmp_path / "flat", edge("load")),
    )
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    assert only_files(tmp_path / "flat") == ["a.mkv", "b.mkv"]
    chain = real_chain(
        folder_node(schemas, source, i15=True),
        save_video(schemas, tmp_path / "deep", edge("load")),
    )
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    assert only_files(tmp_path / "deep") == ["a.mkv", "b.mkv", "inner.mkv"]
    # Limit counts videos: the folder, first in the order, takes no place.
    chain = real_chain(
        folder_node(schemas, source, i15=False, i17=True, i18=1),
        save_video(schemas, tmp_path / "limit", edge("load")),
    )
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    assert only_files(tmp_path / "limit") == ["a.mkv"]
    only_folders = tmp_path / "only folders"
    (only_folders / "dir.mkv").mkdir(parents=True)
    (only_folders / "notes.txt").write_text("not a video")
    chain = real_chain(
        folder_node(schemas, only_folders),
        save_video(schemas, tmp_path / "none", edge("load")),
    )
    result = run_chain(kind, chain)
    assert isinstance(result.error, NodeExecutionError)
    assert result.error.node_id == "load"
    assert "has no video files" in str(result.error)


def text_node(schemas, node_id: str, source: str, suffix: str) -> dict[str, Any]:
    """Text Append: the Load Videos' Name, then the suffix."""
    return make_node(
        schemas,
        node_id,
        "chainner:utility:text_append",
        {0: "", 1: edge(source, NAME), 2: suffix},
    )


def second_save(
    schemas, output: Path, source: str, name: object, container: str = "mkv"
) -> dict[str, Any]:
    return make_node(
        schemas,
        "save second",
        SAVE_VIDEO,
        {0: edge(source), 1: str(output), 2: name, 16: 1, 4: container, 3: "ffv1"},
    )


def two_folders_of_one_name(tmp_path: Path) -> tuple[Path, Path]:
    one, two = tmp_path / "one", tmp_path / "two"
    make_clip(one / "x.mkv", "testsrc2=size=32x32:rate=10:duration=0.2")
    make_clip(two / "x.mkv", "testsrc2=size=48x32:rate=10:duration=0.2")
    return one, two


@pytest.mark.parametrize("kind", KINDS)
def test_two_folders_of_one_name_into_one_output_folder_fail_before_any_frame(
    schemas, tmp_path: Path, kind: str
):
    if not FFMPEG.exists():
        pytest.skip(NO_FFMPEG)
    one, two = two_folders_of_one_name(tmp_path)
    output = tmp_path / "out"
    chain = real_chain(
        folder_node(schemas, one),
        save_video(schemas, output, edge("load")),
        make_node(schemas, "second", LOAD_VIDEOS, {0: str(two)}),
        second_save(schemas, output, "second", edge("second", NAME)),
    )
    result = run_chain(kind, chain)
    assert isinstance(result.error, NodeExecutionError)
    assert "to one file" in str(result.error)
    assert str(one / "x.mkv") in str(result.error)
    assert str(two / "x.mkv") in str(result.error)
    assert not output.exists() or only_files(output) == []


@pytest.mark.parametrize("kind", KINDS)
def test_two_folders_through_like_text_nodes_fail_before_any_frame(
    schemas, tmp_path: Path, kind: str
):
    if not FFMPEG.exists():
        pytest.skip(NO_FFMPEG)
    one, two = two_folders_of_one_name(tmp_path)
    output = tmp_path / "out"
    chain = real_chain(
        folder_node(schemas, one),
        text_node(schemas, "first text", "load", "_out"),
        save_video(schemas, output, edge("load"), i2=edge("first text")),
        make_node(schemas, "second", LOAD_VIDEOS, {0: str(two)}),
        text_node(schemas, "second text", "second", "_out"),
        second_save(schemas, output, "second", edge("second text")),
    )
    result = run_chain(kind, chain)
    assert isinstance(result.error, NodeExecutionError)
    assert "to one file" in str(result.error)
    assert not output.exists() or only_files(output) == []


@pytest.mark.parametrize("kind", KINDS)
def test_two_folders_with_different_names_folders_or_formats_run(
    schemas, tmp_path: Path, kind: str
):
    if not FFMPEG.exists():
        pytest.skip(NO_FFMPEG)
    one, two = two_folders_of_one_name(tmp_path)
    output = tmp_path / "out"
    chain = real_chain(
        folder_node(schemas, one),
        text_node(schemas, "first text", "load", "_a"),
        save_video(schemas, output, edge("load"), i2=edge("first text")),
        make_node(schemas, "second", LOAD_VIDEOS, {0: str(two)}),
        text_node(schemas, "second text", "second", "_b"),
        second_save(schemas, output, "second", edge("second text")),
    )
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    assert only_files(output) == ["x_a.mkv", "x_b.mkv"]
    chain = real_chain(
        folder_node(schemas, one),
        save_video(schemas, tmp_path / "a", edge("load")),
        make_node(schemas, "second", LOAD_VIDEOS, {0: str(two)}),
        second_save(schemas, tmp_path / "b", "second", edge("second", NAME)),
    )
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    assert only_files(tmp_path / "a") == only_files(tmp_path / "b") == ["x.mkv"]
    chain = real_chain(
        folder_node(schemas, one),
        save_video(schemas, tmp_path / "formats", edge("load")),
        make_node(schemas, "second", LOAD_VIDEOS, {0: str(two)}),
        second_save(
            schemas, tmp_path / "formats", "second", edge("second", NAME), "avi"
        ),
    )
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    assert only_files(tmp_path / "formats") == ["x.avi", "x.mkv"]


@pytest.mark.parametrize("kind", KINDS)
def test_two_files_of_one_name_through_a_text_node_fail_before_any_frame(
    schemas, tmp_path: Path, kind: str
):
    """The common wiring, Name through a text node, is still checked."""
    if not FFMPEG.exists():
        pytest.skip(NO_FFMPEG)
    folder = tmp_path / "in"
    make_clip(folder / "one" / "x.mkv", "testsrc2=size=32x32:rate=10:duration=0.2")
    make_clip(folder / "two" / "x.mkv", "testsrc2=size=48x32:rate=10:duration=0.2")
    output = tmp_path / "out"
    chain = real_chain(
        folder_node(schemas, folder, i15=True),
        text_node(schemas, "text", "load", "_x4"),
        save_video(schemas, output, edge("load"), i2=edge("text")),
    )
    result = run_chain(kind, chain)
    assert isinstance(result.error, NodeExecutionError)
    assert result.error.node_id == "save"
    assert "to one file" in str(result.error)
    assert not output.exists() or only_files(output) == []


@pytest.mark.parametrize("kind", KINDS)
def test_a_text_node_that_folds_two_names_into_one_fails_before_any_frame(
    schemas, tmp_path: Path, kind: str
):
    source = tmp_path / "in"
    make_clip(source / "take1.mkv", "testsrc2=size=32x32:rate=10:duration=0.2")
    make_clip(source / "take2.mkv", "testsrc2=size=48x32:rate=10:duration=0.2")
    output = tmp_path / "out"
    chain = real_chain(
        folder_node(schemas, source),
        make_node(
            schemas,
            "fold",
            "chainner:utility:text_replace",
            {0: edge("load", NAME), 1: "2", 2: "1"},
        ),
        save_video(schemas, output, edge("load"), i2=edge("fold")),
    )
    result = run_chain(kind, chain)
    assert isinstance(result.error, NodeExecutionError)
    assert result.error.node_id == "save"
    assert "to one file" in str(result.error)
    assert not output.exists() or only_files(output) == []


@pytest.mark.parametrize("kind", KINDS)
def test_go_up_that_folds_two_folders_into_one_fails_before_any_frame(
    schemas, tmp_path: Path, kind: str
):
    source = tmp_path / "in"
    make_clip(source / "one" / "x.mkv", "testsrc2=size=32x32:rate=10:duration=0.2")
    make_clip(source / "two" / "x.mkv", "testsrc2=size=48x32:rate=10:duration=0.2")
    chain = real_chain(
        folder_node(schemas, source, i15=True),
        make_node(
            schemas,
            "up",
            "chainner:utility:back_directory",
            {0: edge("load", 2), 1: 1},
        ),
        save_video(schemas, tmp_path / "unused", edge("load"), i1=edge("up")),
    )
    result = run_chain(kind, chain)
    assert isinstance(result.error, NodeExecutionError)
    assert "to one file" in str(result.error)
    assert only_files(source) == ["one", "two"]


@pytest.mark.parametrize("kind", KINDS)
def test_nodes_that_keep_the_names_apart_run(schemas, tmp_path: Path, kind: str):
    source = tmp_path / "in"
    make_clip(source / "take1.mkv", "testsrc2=size=32x32:rate=10:duration=0.2")
    make_clip(source / "take2.mkv", "testsrc2=size=48x32:rate=10:duration=0.2")
    output = tmp_path / "out"
    chain = real_chain(
        folder_node(schemas, source),
        make_node(
            schemas,
            "keep",
            "chainner:utility:text_replace",
            {0: edge("load", NAME), 1: "take", 2: "shot"},
        ),
        save_video(schemas, output, edge("load"), i2=edge("keep")),
    )
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    assert only_files(output) == ["shot1.mkv", "shot2.mkv"]


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_a_name_node_that_fails_on_a_file_is_left_to_its_run(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    """The check does not report it: that file's run does, with the others done."""

    def named(name: str):
        if name == "b":
            raise ValueError("no name for b")
        return "same" if name == "a" else "other"

    chain = real_chain(
        folder_node(schemas, clips),
        save_video(schemas, tmp_path, edge("load"), i2=None),
    )
    b = Builder()
    b.chain = chain
    b.node("named", named)
    b.edge("load", NAME, "named", 0)
    b.edge("named", 0, "save", 2)
    result = run_chain(kind, chain)
    assert "no name for b" in str(result.error)
    assert "to one file" not in str(result.error)
    assert only_files(tmp_path) == ["other.mkv", "same.mkv"]


@pytest.mark.parametrize("kind", KINDS)
def test_frame_numbers_alone_in_one_folder_fail_before_any_frame(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    """Every file has a frame 0."""
    chain = real_chain(
        folder_node(schemas, clips),
        make_node(
            schemas,
            "text",
            "chainner:utility:text_pattern",
            {0: "frame {1}", 1: edge("load", INDEX)},
        ),
        make_node(
            schemas,
            "save",
            "chainner:image:save",
            {
                0: edge("load"),
                1: str(tmp_path),
                2: None,
                3: edge("text"),
                4: "png",
                15: "u8",
            },
        ),
    )
    result = run_chain(kind, chain)
    assert isinstance(result.error, NodeExecutionError)
    assert "to one file" in str(result.error)
    assert only_files(tmp_path) == []


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_a_name_from_the_frames_is_no_collision(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    """Save Image's name from each frame's content: nothing says that two files collide."""

    def digest(frame: np.ndarray):
        return hashlib.sha256(frame.tobytes()).hexdigest()[:16]

    chain = real_chain(
        folder_node(schemas, clips),
        make_node(
            schemas,
            "save",
            "chainner:image:save",
            {0: edge("load"), 1: str(tmp_path), 2: None, 3: None, 4: "png", 15: "u8"},
        ),
    )
    b = Builder()
    b.chain = chain
    b.node("digest", digest)
    b.edge("load", FRAME, "digest", 0)
    b.edge("digest", 0, "save", 3)
    result = run_chain(kind, chain)
    assert result.error is None, result.error
    assert len(only_files(tmp_path)) > 3


@pytest.mark.usefixtures("test_package")
def test_a_name_from_a_zipped_generator_is_no_collision(
    schemas, clips: Path, tmp_path: Path
):
    """A generator zipped with the files gives the name: its value is its own for each file. The check runs above the
    executors. On process.Executor: the native engine fails on a collector reading a zipped generator's per-run output
    with or without a folder."""
    labels: list[str] = []

    def numbers():
        labels.append(f"label{len(labels)}")
        generator = items(list(range(100)))
        generator.length_is_exact = False
        return generator, labels[-1]

    chain = real_chain(
        folder_node(schemas, clips), save_video(schemas, tmp_path, None, i2=None)
    )
    b = Builder()
    b.chain = chain
    b.node("numbers", numbers, kind="generator", inputs=0, outputs=2)
    b.node("mix", lambda frame, _: frame, inputs=2)
    b.edge("load", FRAME, "mix", 0)
    b.edge("numbers", 0, "mix", 1)
    b.edge("mix", 0, "save", 0)
    b.edge("numbers", 1, "save", 2)
    result = run_chain("python", chain)
    assert result.error is None, result.error
    assert only_files(tmp_path) == ["label0.mkv", "label1.mkv", "label2.mkv"]


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_a_collision_fails_before_the_rest_of_the_chain(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    calls: list[str] = []
    output = tmp_path / "out"
    chain = real_chain(
        folder_node(schemas, clips),
        save_video(schemas, output, edge("load"), i2="same"),
    )
    b = Builder()
    b.chain = chain
    b.node(
        "elsewhere",
        lambda: calls.append("elsewhere"),
        inputs=0,
        outputs=0,
        side_effects=True,
    )
    result = run_chain(kind, chain)
    assert isinstance(result.error, NodeExecutionError)
    assert "to one file" in str(result.error)
    assert calls == []
    assert not output.exists() or only_files(output) == []


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_a_folder_without_videos_fails_before_the_rest_of_the_chain(
    schemas, tmp_path: Path, kind: str
):
    calls: list[str] = []
    empty = tmp_path / "empty"
    empty.mkdir()
    chain = real_chain(
        folder_node(schemas, empty), save_video(schemas, tmp_path / "out", edge("load"))
    )
    b = Builder()
    b.chain = chain
    b.node(
        "elsewhere",
        lambda: calls.append("elsewhere"),
        inputs=0,
        outputs=0,
        side_effects=True,
    )
    result = run_chain(kind, chain)
    assert "has no video files" in str(result.error)
    assert calls == []


def cleaned_up_chain(
    schemas, clips: Path, tmp_path: Path, trace: list[str], last: Any = None
) -> Chain:
    """A shared node that registers a cleanup for the run's end; a file's frame leaves a trace."""

    def shared(context: Any):
        context.add_cleanup(lambda: trace.append("cleanup"), after="chain")
        trace.append("shared")
        return 2

    def mark(frame: np.ndarray, _: int):
        trace.append("frame")
        return last(frame) if last is not None else frame

    chain = real_chain(folder_node(schemas, clips), save_video(schemas, tmp_path, None))
    b = Builder()
    b.chain = chain
    b.node("shared", shared, inputs=0, context=True)
    b.node("mark", mark, inputs=2)
    b.edge("load", FRAME, "mark", 0)
    b.edge("shared", 0, "mark", 1)
    b.edge("mark", 0, "save", 0)
    return chain


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_a_shared_node_s_chain_cleanup_runs_once_at_the_run_s_end(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    trace: list[str] = []
    result = run_chain(kind, cleaned_up_chain(schemas, clips, tmp_path, trace))
    assert result.error is None, result.error
    assert trace[0] == "shared"
    assert trace.count("cleanup") == 1
    assert trace[-1] == "cleanup"
    assert trace.count("frame") == 30 + 15 + 10


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_a_shared_node_s_chain_cleanup_runs_when_the_run_fails(
    schemas, tmp_path: Path, kind: str
):
    trace: list[str] = []
    empty = tmp_path / "empty"
    empty.mkdir()
    result = run_chain(kind, cleaned_up_chain(schemas, empty, tmp_path / "out", trace))
    assert "has no video files" in str(result.error)
    assert trace == ["shared", "cleanup"]


@pytest.mark.usefixtures("test_package")
@pytest.mark.parametrize("kind", KINDS)
def test_a_shared_node_s_chain_cleanup_runs_when_the_run_is_stopped(
    schemas, clips: Path, tmp_path: Path, kind: str
):
    trace: list[str] = []
    reached, stopped = threading.Event(), threading.Event()

    def hold(frame: np.ndarray):
        reached.set()
        assert stopped.wait(60)
        return frame

    def stop(executor: Any) -> None:
        assert reached.wait(60)
        executor.kill()
        stopped.set()

    chain = cleaned_up_chain(schemas, clips, tmp_path, trace, hold)
    result = run_chain(kind, chain, during=stop)
    assert isinstance(result.error, Aborted)
    assert trace.count("cleanup") == 1
    assert trace[-1] == "cleanup"


@pytest.mark.usefixtures("test_package")
def test_the_plan_splits_the_chain(schemas, tmp_path: Path):
    chain = parse_json(cast("list[JsonNode]", [folder_node(schemas, tmp_path)]))
    b = Builder()
    b.chain = chain
    b.node("root", lambda: 1, inputs=0)
    b.node("shared", lambda _: 2)
    b.edge("root", 0, "shared", 0)
    b.node("unread root", lambda: 3, inputs=0)
    b.node("unread", lambda _: 4)
    b.edge("unread root", 0, "unread", 0)
    b.node("zipped", lambda: items([1, 2, 3]), kind="generator", inputs=0)
    b.node("gate", lambda *_: 5, inputs=3, lazy=(2,))
    b.edge("load", FRAME, "gate", 0)
    b.edge("zipped", 0, "gate", 0)
    b.edge("shared", 0, "gate", 1)
    b.edge("unread", 0, "gate", 2)
    b.node("sink", lambda _: None, outputs=0, side_effects=True)
    b.edge("gate", 0, "sink", 0)
    b.node("elsewhere", lambda: None, inputs=0, outputs=0, side_effects=True)
    folder = _plan_folder(chain, cast("Any", chain.nodes[NodeId("load")]))
    assert folder.per_file == {"load", "zipped", "gate", "sink"}
    assert folder.shared == {"shared"}
    assert folder.chain_nodes == {*folder.per_file, "shared", "unread", "unread root"}


class FakeExecutor:
    """An executor that runs nothing: each run ends with what the test gives it."""

    def __init__(self, made: list[FakeExecutor], at_end: list[Any], **_: Any) -> None:
        self.at_end = at_end
        self.paused = False
        self.killed = False
        made.append(self)

    async def run(self) -> None:
        self.at_end.pop(0)()

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def kill(self) -> None:
        self.killed = True


def fake_run(
    schemas,
    clips: Path,
    tmp_path: Path,
    at_end: Callable[[Any, list[FakeExecutor]], list[Any]],
) -> Any:
    """A folder's run over FakeExecutors; returns it and them, also when it raises."""
    chain = real_chain(
        folder_node(schemas, clips), save_video(schemas, tmp_path, edge("load"))
    )
    optimize(chain)
    made: list[FakeExecutor] = []

    async def main():
        with ThreadPoolExecutor(1) as pool:
            run: Any = None
            actions = at_end(lambda: run, made)
            run = run_executor(
                functools.partial(FakeExecutor, made, actions),
                id="fake",
                chain=chain,
                queue=EventQueue(),
                loop=asyncio.get_running_loop(),
                pool=pool,
            )
            await run.run()
            return run

    return asyncio.run(main()), made


def test_a_new_file_s_executor_starts_paused_if_the_run_is(
    schemas, clips: Path, tmp_path: Path
):
    def actions(run: Callable[[], Any], made: list[FakeExecutor]) -> list[Any]:
        def pause():
            run().pause()

        def resume():
            assert made[1].paused
            run().resume()

        return [pause, resume, lambda: None]

    run, made = fake_run(schemas, clips, tmp_path, actions)
    assert len(made) == 3
    assert not made[2].paused
    assert not run.progress.paused


def test_stop_between_two_files_starts_no_other(schemas, clips: Path, tmp_path: Path):
    made: list[FakeExecutor] = []

    def actions(run: Callable[[], Any], executors: list[FakeExecutor]) -> list[Any]:
        def stop():
            # The file's executor has done its work: the run's Stop keeps the next from starting.
            run().kill()
            assert executors[0].killed
            made.extend(executors)

        return [stop]

    with pytest.raises(Aborted):
        fake_run(schemas, clips, tmp_path, actions)
    assert len(made) == 1


DRIVER = """
import asyncio, json, os, shutil, sys
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = [os.environ["TOOLS"]]
sys.modules["__main__"].__file__ = os.path.abspath("run.py")
import importlib
import api
importlib.import_module("packages.chaiNNer_standard")
api.registry.load_nodes(os.path.abspath("server.py"))
import server

# The handlers run without a started Sanic app: its loop is the running one.
type(server.app).loop = property(lambda self: asyncio.get_running_loop())
from verify_runtime import edge, make_node

schemas = {s: {"inputs": [i.to_dict() for i in d.inputs], "kind": d.kind} for s, (d, _) in api.registry.nodes.items()}
clips, output = Path(os.environ["CLIPS"]), Path(os.environ["OUTPUT"])
load = make_node(schemas, "load", "chainner:image:load_videos", {0: str(clips)})
save = make_node(schemas, "save", "chainner:image:save_video", {
    0: edge("load", 0), 1: str(output), 2: edge("load", 3), 16: 1, 4: "mkv", 3: "ffv1", 14: edge("load", 4),
    15: edge("load", 5), 23: edge("load", 9)})


class Recorder:
    def __init__(self):
        self.events = []

    def put(self, event):
        self.events.append(json.loads(json.dumps(event)))


async def main():
    ctx = server.app.ctx
    ctx.__dict__["storage_dir"] = Path(os.environ["STORAGE"])
    ctx.cache.clear()
    recorder = Recorder()
    ctx.__dict__["queue"] = recorder
    request = SimpleNamespace(app=server.app)
    request.json = {"id": "load", "schemaId": load["schemaId"], "options": {},
                    "inputs": [i["value"] for i in load["inputs"]]}
    preview = await server.run_individual(request)
    await asyncio.gather(*list(server.individual_releases))
    preview_events = recorder.events
    recorder = Recorder()
    ctx.__dict__["queue"] = recorder
    request.json = {"data": [load, save], "options": {}, "sendBroadcastData": True}
    answer = await server.run(request)
    return {"preview": json.loads(preview.body), "preview_events": preview_events, "status": answer.status,
            "body": json.loads(answer.body), "events": recorder.events, "busy": ctx.executor is not None}


print(json.dumps(asyncio.run(main())))
"""


@pytest.mark.parametrize("kind", KINDS)
def test_through_the_server(clips: Path, tmp_path: Path, kind: str):
    environment = dict(
        os.environ,
        CLIPS=str(clips),
        OUTPUT=str(tmp_path),
        STORAGE=str(STORAGE),
        TOOLS=str(BACKEND.parents[1] / "native" / "tools"),
        CUDA_VISIBLE_DEVICES="-1",
        CHAINNER_C_EXECUTOR=kind,
    )
    done = subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(DRIVER)],
        cwd=BACKEND,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, (done.stdout[-2000:], done.stderr[-4000:])
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["preview"]["success"], result["preview"]
    broadcast = next(
        e["data"] for e in result["preview_events"] if e["event"] == "node-broadcast"
    )
    assert broadcast["nodeId"] == "load"
    assert broadcast["types"]["14"]["value"] == 3
    assert result["status"] == 200, result["body"]
    assert not result["busy"]
    assert only_files(tmp_path) == sorted(CLIPS)
    assert [e["event"] for e in result["events"]].count("chain-start") == 1


def images_to_png(schemas, folder: Path, output: Path, **more: Any) -> Chain:
    """Load Images into Save Image as PNG in `output`, named by the image's Name; Subdirectory Path not connected."""
    values: dict[int, object] = {0: str(folder)}
    values.update({int(key[1:]): value for key, value in more.items()})
    return real_chain(
        make_node(schemas, "load", "chainner:image:load_images", values),
        make_node(
            schemas,
            "save",
            "chainner:image:save",
            {0: edge("load"), 1: str(output), 2: None, 3: edge("load", 3), 4: "png"},
        ),
    )


def write_image(path: Path, colour: tuple[int, int, int]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((8, 8, 3), colour, np.uint8))
    return path.read_bytes()


@pytest.mark.parametrize("kind", KINDS)
def test_two_images_saved_to_one_file_are_refused_before_anything_is_written(
    schemas, tmp_path: Path, kind: str
):
    # a.jpg would write a.png over the user's a.png before a.png is read.
    write_image(tmp_path / "a.jpg", (0, 0, 255))
    original = write_image(tmp_path / "a.png", (255, 0, 0))
    result = run_chain(kind, images_to_png(schemas, tmp_path, tmp_path))
    assert isinstance(result.error, NodeExecutionError)
    assert result.error.node_id == "save"
    assert str(result.error).startswith(
        f"Save Image would write {tmp_path / 'a.jpg'} and {tmp_path / 'a.png'} to one"
        " file:"
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.jpg", "a.png"]
    assert (tmp_path / "a.png").read_bytes() == original


@pytest.mark.parametrize("kind", KINDS)
def test_one_name_in_two_subfolders_is_refused_under_recursive(
    schemas, tmp_path: Path, kind: str
):
    source, output = tmp_path / "source", tmp_path / "out"
    write_image(source / "x" / "a.png", (0, 0, 255))
    write_image(source / "y" / "a.png", (255, 0, 0))
    output.mkdir()
    result = run_chain(kind, images_to_png(schemas, source, output, i2=True))
    assert isinstance(result.error, NodeExecutionError)
    assert "would write" in str(result.error)
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("kind", KINDS)
def test_images_processed_in_place_still_run(schemas, tmp_path: Path, kind: str):
    # Each item writes over its own source: allowed, as is a.png matching no other output.
    originals = {
        name: write_image(tmp_path / name, colour)
        for name, colour in (("a.png", (0, 0, 255)), ("b.png", (255, 0, 0)))
    }
    result = run_chain(kind, images_to_png(schemas, tmp_path, tmp_path))
    assert result.error is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.png", "b.png"]
    assert all((tmp_path / name).exists() for name in originals)
