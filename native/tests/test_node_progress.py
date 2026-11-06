"""A node's NodeContext.set_progress reaches the app as node-progress without items (total 0)
while the node runs, on both executors: at most ten a second, rising within a pass, before the
node's finish. A pass shorter than the rate sends nothing. The comparisons of two runs leave
these events out, since the clock sends them."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from engine_differential import KINDS, execute
from spandrel import ImageModelDescriptor
from spandrel.architectures.ESRGAN import ESRGAN, ESRGANArch
from test_engine_scenarios import Builder
from verify_runtime import sse_contract, sse_record

import api
from api import NodeContext, NodeId, Progress
from chain.chain import Chain
from events import EventConsumer
from nodes.impl.pytorch.auto_split import pytorch_auto_split
from nodes.impl.upscale.tiler import MaxTileSize
from process import NODE_PROGRESS_SECONDS, run_progress


@pytest.fixture(autouse=True)
def test_package(monkeypatch: pytest.MonkeyPatch):
    # Test nodes belong to no package: both executors read their settings from a stand-in.
    monkeypatch.setattr(
        api.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )


def tiling_chain(tiles: int, seconds: float) -> Chain:
    def upscale(context: NodeContext, value: int) -> int:
        for tile in range(tiles):
            time.sleep(seconds)
            context.set_progress((tile + 1) / tiles)
        return value * 3

    b = Builder()
    b.node("upscale", upscale, context=True, side_effects=True)
    b.value("upscale", 0, 7)
    return b.chain


def own_progress(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e["data"] for e in events if e["event"] == "node-progress"]


@pytest.mark.parametrize("kind", KINDS)
def test_a_long_node_reports_its_progress_while_it_runs(tmp_path: Path, kind: str):
    tiles = 40
    result = execute(kind, tiling_chain(tiles, 0.02), storage=tmp_path)
    assert result.error is None

    progress = own_progress(result.events)
    # 0.8 s of updates, at most one per 0.1 s, the first update only starting the clock.
    assert 3 <= len(progress) <= 8
    assert all(p["nodeId"] == "upscale" for p in progress)
    assert all(p["total"] == 0 and p["index"] == 0 and p["eta"] == 0 for p in progress)
    values = [p["progress"] for p in progress]
    assert values == sorted(set(values))
    assert 0 < values[0] and values[-1] < 1

    kinds = [e["event"] for e in result.events if "nodeId" in e["data"]]
    assert kinds.index("node-progress") > kinds.index("node-start")
    assert "node-progress" not in kinds[kinds.index("node-finish") :]


@pytest.mark.parametrize("kind", KINDS)
def test_a_short_node_reports_nothing(tmp_path: Path, kind: str):
    result = execute(kind, tiling_chain(4, 0), storage=tmp_path)
    assert result.error is None
    assert own_progress(result.events) == []


class Clock:
    """run_progress's sends with the time of each update given."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.now = 1000.0
        self.sent: list[float] = []
        monkeypatch.setattr(time, "monotonic", lambda: self.now)
        loop: Any = SimpleNamespace(call_soon_threadsafe=lambda put, e: put(e))
        self.send = run_progress(loop, self, NodeId("n"))

    def put(self, event: Any) -> None:
        self.sent.append(event["data"]["progress"])

    def update(self, seconds: float, progress: float) -> None:
        self.now += seconds
        self.send(progress)


def test_updates_are_at_least_the_rate_apart(monkeypatch: pytest.MonkeyPatch):
    clock = Clock(monkeypatch)
    # The third update after the one before is the first at least the rate later.
    for tile in range(1, 11):
        clock.update(NODE_PROGRESS_SECONDS * 0.4, tile / 20)
    assert clock.sent == [0.2, 0.35, 0.5]


def test_a_pass_starts_over_and_its_end_is_left_to_the_finish(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = Clock(monkeypatch)
    # The colour pass, then the alpha pass: the first update of each only starts the clock.
    for _ in range(2):
        for progress in (0.25, 0.5, 0.75, 1):
            clock.update(1, progress)
    assert clock.sent == [0.5, 0.75, 0.5, 0.75]


def test_a_frame_shorter_than_the_rate_sends_nothing(monkeypatch: pytest.MonkeyPatch):
    clock = Clock(monkeypatch)
    for _ in range(100):
        for progress in (0.25, 0.5, 0.75, 1):
            clock.update(0.01, progress)
    assert clock.sent == []


def test_an_untiled_image_sends_nothing_however_long(monkeypatch: pytest.MonkeyPatch):
    clock = Clock(monkeypatch)
    clock.update(60, 1)
    assert clock.sent == []


class RecordedProgress(Progress):
    def __init__(self) -> None:
        self.values: list[float] = []

    @property
    def aborted(self) -> bool:
        return False

    @property
    def paused(self) -> bool:
        return False

    def set_progress(self, progress: float) -> None:
        self.values.append(progress)


def test_a_pytorch_upscale_reports_each_tile():
    torch.manual_seed(0)
    model = ESRGANArch().load(ESRGAN(num_filters=8, num_blocks=1, scale=2).state_dict())
    assert isinstance(model, ImageModelDescriptor)
    progress = RecordedProgress()
    pytorch_auto_split(
        np.full((96, 96, 3), 0.5, np.float32),
        model,
        torch.device("cpu"),
        False,
        MaxTileSize(64),
        progress,
    )
    assert len(progress.values) > 1
    assert progress.values == sorted(set(progress.values))
    assert progress.values[-1] == 1


def test_comparisons_leave_out_a_nodes_own_progress(tmp_path: Path):
    start = {"event": "node-start", "data": {"nodeId": "n"}}
    own = {
        "event": "node-progress",
        "data": {"nodeId": "n", "progress": 0.5, "index": 0, "total": 0, "eta": 0},
    }
    item = {
        "event": "node-progress",
        "data": {"nodeId": "g", "progress": 0.5, "index": 1, "total": 2, "eta": 1},
    }
    assert sse_contract([start, own], tmp_path) == sse_contract([start], tmp_path)
    assert sse_record([start, own], tmp_path) == sse_record([start], tmp_path)
    assert sse_contract([start, item], tmp_path) != sse_contract([start], tmp_path)
    assert sse_record([start, item], tmp_path) != sse_record([start], tmp_path)


def test_an_individual_run_has_no_progress_events():
    # A preview's node shows no bar: its /run takes the output from the cache.
    assert "node-progress" not in EventConsumer.INDIVIDUAL_RUN_EVENTS
