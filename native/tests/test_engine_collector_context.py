"""Critical shared-collector and composed temporal-context scenarios."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from engine_differential import execute
from test_engine_scenarios import Builder, Frames

import api
from api import Collector, ExecutionOptions, Generator, IteratorInputInfo, NodeContext
from chain.chain import Chain
from native_executor import build_plan
from nodes.impl.item_window import ITEM_WINDOW_VARIABLE


@pytest.fixture(autouse=True)
def node_package(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        api.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )


@pytest.mark.parametrize("kind", ["python", "native"])
@pytest.mark.parametrize("via_stack", [True, False], ids=["stack", "direct"])
def test_shared_collector_runs_once_per_item(
    kind: str, via_stack: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv(ITEM_WINDOW_VARIABLE, "1")
    b = Builder()
    trace: list[object] = []
    b.node(
        "left",
        lambda: Generator(lambda: iter([1, 2, 3]), 3),
        kind="generator",
        inputs=0,
    )
    b.node(
        "right",
        lambda: Generator(lambda: iter([10, 20, 30]), 3),
        kind="generator",
        inputs=0,
    )

    def gather(*_: object):
        trace.append(("create",))

        def iterate(value: object):
            trace.append(("iterate", value))

        def complete():
            trace.append(("complete",))

        return Collector(iterate, complete)

    collector = b.node(
        "collect", gather, kind="collector", inputs=1 if via_stack else 2, outputs=0
    )
    if via_stack:
        b.node("stack", lambda left, right: (left, right), inputs=2)
        b.edge("left", 0, "stack", 0)
        b.edge("right", 0, "stack", 1)
        b.edge("stack", 0, "collect", 0)
    else:
        collector.data.iterable_inputs[:] = [IteratorInputInfo([0, 1])]
        b.edge("left", 0, "collect", 0)
        b.edge("right", 0, "collect", 1)

    result = execute(kind, b.chain, storage=tmp_path / "storage")
    assert result.error is None, (str(result.error), trace)
    expected = [("iterate", (i, i * 10)) for i in (1, 2, 3)]
    assert trace == [("create",), *expected, ("complete",)]


@pytest.mark.parametrize("kind", ["python", "native"])
@pytest.mark.parametrize("exact", [False, True], ids=["upper-bounds", "known-lengths"])
def test_only_exact_generator_lengths_must_match(
    kind: str, exact: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(ITEM_WINDOW_VARIABLE, "1")
    b = Builder()
    b.node(
        "left",
        lambda: Generator(lambda: iter([1, 2, 3]), 3),
        kind="generator",
        inputs=0,
    )
    b.node(
        "right",
        lambda: Generator(lambda: iter([10, 20, 30]), 5, length_is_exact=exact),
        kind="generator",
        inputs=0,
    )
    seen: list[tuple[object, object]] = []
    b.node(
        "sink", lambda a, c: seen.append((a, c)), inputs=2, outputs=0, side_effects=True
    )
    b.edge("left", 0, "sink", 0)
    b.edge("right", 0, "sink", 1)
    result = execute(kind, b.chain, storage=tmp_path / "storage")
    if exact:
        assert result.error is not None
        assert "same length" in str(result.error)
        assert seen == []
    else:
        assert result.error is None, str(result.error)
        assert seen == [(1, 10), (2, 20), (3, 30)]


@pytest.mark.parametrize("kind", ["python", "native"])
@pytest.mark.parametrize("serial", [True, False], ids=["serial", "merge"])
def test_temporal_radii_follow_dependency_paths(
    kind: str, serial: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # K = 1: token histories test warm-up without any look-ahead or model loads.
    monkeypatch.setenv(ITEM_WINDOW_VARIABLE, "1")

    def chain(generator: Callable[[], Generator], trace: list[object]) -> Chain:
        b = Builder()
        b.node("frames", generator, kind="generator", inputs=0)

        def first(context: NodeContext, value: object):
            view = context.iteration
            assert view is not None
            history = (*cast("tuple[object, ...]", view.state or ()), value)[-2:]
            view.state = history
            return history

        def second(context: NodeContext, value: object):
            view = context.iteration
            assert view is not None
            history = (*cast("tuple[object, ...]", view.state or ()), value)[-4:]
            view.state = history
            return history

        b.node("first", first, context=True, temporal=(1, 5))
        b.node("second", second, context=True, temporal=(3, 3))
        b.edge("frames", 0, "first", 0)
        if serial:
            b.node("pass", lambda value: value)
            b.edge("first", 0, "pass", 0)
            b.edge("pass", 0, "second", 0)
            output = "second"
        else:
            b.edge("frames", 0, "second", 0)
            b.node("merge", lambda left, right: (left, right), inputs=2)
            b.edge("first", 0, "merge", 0)
            b.edge("second", 0, "merge", 1)
            output = "merge"
        b.node("sink", trace.append, outputs=0, side_effects=True)
        b.edge(output, 0, "sink", 0)
        return b.chain

    full_trace: list[object] = []
    full = execute(kind, chain(Frames(20).whole, full_trace), storage=tmp_path / "full")
    assert full.error is None, str(full.error)
    frames = Frames(20)
    range_trace: list[object] = []
    ranged_chain = chain(lambda: frames.range(9, 1), range_trace)
    wanted = (4, 8) if serial else (3, 5)
    plan, *_ = build_plan(
        ranged_chain,
        ExecutionOptions.parse({}),
        tmp_path / "range",
        False,
        None,
        lambda: pytest.fail("Planning must not execute nodes"),
    )
    groups = plan["groups"]
    assert isinstance(groups, list)
    assert groups[0]["context"] == wanted
    ranged = execute(
        kind,
        ranged_chain,
        storage=tmp_path / "range",
    )
    assert ranged.error is None, str(ranged.error)
    assert frames.asked == [wanted]
    assert range_trace == [full_trace[9]]
