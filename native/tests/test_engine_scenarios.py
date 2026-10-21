"""The native run engine against process.Executor on executor scenarios (explore-executor.md section 12): chains of
test nodes (Python bodies, no registry) run whole through both executors in this process
(native/tools/engine_differential.py), compared by outcome, SSE events (verify_runtime.sse_event_mismatches), the
engine's A6 order invariants, and the side effects the nodes record (their calls, in order)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from engine_differential import Result, execute, mismatches, order_violations
from verify_runtime import canonical

import api
import native_executor
import process
import util
from api import (
    BaseInput,
    BaseOutput,
    Collector,
    ExecutionOptions,
    Generator,
    InputId,
    IteratorInputInfo,
    IteratorOutputInfo,
    NodeAttributedError,
    NodeData,
    NodeId,
    NodeKind,
    OutputId,
)
from chain.chain import (
    Chain,
    CollectorNode,
    Edge,
    EdgeSource,
    EdgeTarget,
    FunctionNode,
    GeneratorNode,
)
from events import EventConsumer
from native_executor import NativeExecutor
from nodes.impl.item_window import (
    ITEM_WINDOW_VARIABLE,
    SERIAL_PRODUCER_VARIABLE,
    ItemWindow,
    register_phases,
)
from process import ExecutionId, Executor, RegularOutput


class Builder:
    """A chain of test nodes: pure ones are utility nodes without side effects or context (pure_cpu_node)."""

    def __init__(self) -> None:
        self.chain = Chain()

    def node(
        self,
        node_id: str,
        run: Callable[..., object],
        *,
        kind: NodeKind = "regularNode",
        inputs: int = 1,
        outputs: int = 1,
        side_effects: bool = False,
        context: bool = False,
        lazy: tuple[int, ...] = (),
        iterated_outputs: tuple[int, ...] = (0,),
        temporal: tuple[int, int] | None = None,
    ):
        node_class = {
            "generator": GeneratorNode,
            "collector": CollectorNode,
            "regularNode": FunctionNode,
        }[kind]
        data = NodeData(
            schema_id=f"chainner:utility:test_{node_id}",
            description="",
            see_also=[],
            name=node_id,
            icon="",
            kind=kind,
            inputs=[
                (
                    BaseInput("any", f"Input{i}").with_id(i).make_lazy()
                    if i in lazy
                    else BaseInput("any", f"Input{i}").with_id(i)
                )
                for i in range(inputs)
            ],
            outputs=[
                BaseOutput("any", f"Output{i}").with_id(i) for i in range(outputs)
            ],
            group_layout=[],
            iterable_inputs=[IteratorInputInfo(0)] if kind == "collector" else [],
            iterable_outputs=[IteratorOutputInfo(list(iterated_outputs))]
            if kind == "generator"
            else [],
            key_info=None,
            suggestions=[],
            side_effects=side_effects,
            deprecated=False,
            node_context=context,
            features=[],
            run=run,
            temporal_context=temporal,
        )
        node = object.__new__(node_class)
        node.id, node.schema_id, node.data = NodeId(node_id), data.schema_id, data
        self.chain.add_node(node)
        return node

    def edge(self, source: str, output: int, target: str, input_index: int) -> None:
        self.chain.add_edge(
            Edge(
                EdgeSource(NodeId(source), OutputId(output)),
                EdgeTarget(NodeId(target), InputId(input_index)),
            )
        )

    def value(self, target: str, input_index: int, value: object) -> None:
        self.chain.inputs.set(NodeId(target), InputId(input_index), value)


Scenario = Callable[[list[object]], Chain]


def run_both(
    scenario: Scenario, tmp_path: Path, **options
) -> tuple[Chain, Result, Result, list[object], list[object]]:
    traces: list[list[object]] = []
    results: list[Result] = []
    chain = Chain()
    for kind in ("python", "native"):
        trace: list[object] = []
        chain = scenario(trace)
        results.append(execute(kind, chain, storage=tmp_path / "storage", **options))
        traces.append(trace)
    return chain, results[0], results[1], traces[0], traces[1]


def assert_same(scenario: Scenario, tmp_path: Path, **options) -> Result:
    chain, oracle, engine, oracle_trace, engine_trace = run_both(
        scenario, tmp_path, **options
    )
    assert mismatches(scenario.__name__, chain, oracle, engine, tmp_path) == []
    assert engine_trace == oracle_trace
    return engine


@pytest.fixture(autouse=True)
def test_package(monkeypatch):
    # Test nodes belong to no package: both executors read their settings from a stand-in.
    monkeypatch.setattr(
        api.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )


# Regular nodes


def independent_sinks(trace: list[object]) -> Chain:
    b = Builder()
    for name in ("a", "b", "c"):
        b.node(f"double_{name}", lambda x: x * 2)
        b.value(f"double_{name}", 0, {"a": 1, "b": 2, "c": 3}[name])
        b.node(
            f"sink_{name}",
            lambda x, name=name: trace.append((name, x)),
            outputs=0,
            side_effects=True,
        )
        b.edge(f"double_{name}", 0, f"sink_{name}", 0)
    return b.chain


def diamond(trace: list[object]) -> Chain:
    b = Builder()
    calls: Counter[str] = Counter()

    def shared(x):
        calls["shared"] += 1
        return x + 1

    b.node("shared", shared)
    b.value("shared", 0, 10)
    b.node("left", lambda x: x * 2)
    b.node("right", lambda x: x * 3)
    b.node("join", lambda x, y: x + y, inputs=2)
    b.node(
        "sink",
        lambda v: trace.append(("sink", v, calls["shared"])),
        outputs=0,
        side_effects=True,
    )
    b.edge("shared", 0, "left", 0)
    b.edge("shared", 0, "right", 0)
    b.edge("left", 0, "join", 0)
    b.edge("right", 0, "join", 1)
    b.edge("join", 0, "sink", 0)
    return b.chain


def lazy_unread(trace: list[object]) -> Chain:
    b = Builder()

    def expensive(x):
        trace.append("expensive ran")
        return x

    b.node("expensive", expensive)
    b.value("expensive", 0, 5)
    b.node(
        "choose",
        lambda condition, lazy: trace.append(
            ("chose", lazy.value if condition else None)
        ),
        inputs=2,
        outputs=0,
        side_effects=True,
        lazy=(1,),
    )
    b.value("choose", 0, False)
    b.edge("expensive", 0, "choose", 1)
    return b.chain


def lazy_read(trace: list[object]) -> Chain:
    b = Builder()
    b.node("upstream", lambda x: x * 7)
    b.value("upstream", 0, 6)
    b.node(
        "reader",
        lambda lazy: trace.append(("read", lazy.value)),
        outputs=0,
        side_effects=True,
        lazy=(0,),
    )
    b.edge("upstream", 0, "reader", 0)
    return b.chain


def lazy_constant(trace: list[object]) -> Chain:
    b = Builder()
    b.node(
        "reader",
        lambda lazy: trace.append(("read", lazy.value)),
        outputs=0,
        side_effects=True,
        lazy=(0,),
    )
    b.value("reader", 0, "constant")
    return b.chain


def lazy_upstream_error(trace: list[object]) -> Chain:
    b = Builder()

    def fails(x):
        raise ValueError(f"upstream failed on {x}")

    b.node("upstream", fails)
    b.value("upstream", 0, 3)
    b.node(
        "reader",
        lambda lazy: trace.append(("read", lazy.value)),
        outputs=0,
        side_effects=True,
        lazy=(0,),
    )
    b.edge("upstream", 0, "reader", 0)
    return b.chain


def input_order_errors(trace: list[object]) -> Chain:
    b = Builder()

    def slow_fail(x):
        time.sleep(0.05)
        raise RuntimeError("first input failed")

    def fast_fail(x):
        raise KeyError("second input failed")

    b.node("first", slow_fail)
    b.value("first", 0, 1)
    b.node("second", fast_fail)
    b.value("second", 0, 2)
    b.node(
        "sink",
        lambda x, y: trace.append((x, y)),
        inputs=2,
        outputs=0,
        side_effects=True,
    )
    b.edge("first", 0, "sink", 0)
    b.edge("second", 0, "sink", 1)
    return b.chain


def context_cleanups(trace: list[object]) -> Chain:
    b = Builder()

    def body(context, x):
        context.add_cleanup(lambda: trace.append("node cleanup"), after="node")

        def chain_cleanup():
            trace.append("chain cleanup")

        # The same function registered twice runs once.
        context.add_cleanup(chain_cleanup)
        context.add_cleanup(chain_cleanup)
        trace.append(("body", x, context.node_id))
        return x

    b.node("contextual", body, context=True, side_effects=True)
    b.value("contextual", 0, 4)
    return b.chain


def attributed_error(trace: list[object]) -> Chain:
    b = Builder()
    b.node("victim", lambda x: x, side_effects=True)
    b.value("victim", 0, 1)

    def script(x):
        raise NodeAttributedError("the victim's statement failed", "victim")

    b.node("script", script, outputs=0, side_effects=True)
    b.value("script", 0, 2)
    return b.chain


# Generators and collectors


def items(
    values: list[object], *, fail_fast: bool = True, source_paths: tuple | None = ()
) -> Generator:
    def supplier():
        yield from values

    return Generator(
        supplier, len(values), fail_fast=fail_fast, source_paths=source_paths
    )


def loop_chain(
    trace: list[object],
    values: list[object],
    *,
    fail_fast: bool = True,
    middle: Callable[[Any], Any] = lambda x: x * 10,
    collect: bool = True,
) -> Chain:
    b = Builder()
    b.node(
        "numbers",
        lambda: items(values, fail_fast=fail_fast),
        kind="generator",
        inputs=0,
    )
    b.node("scale", middle)
    b.node("sink", lambda x: trace.append(("sink", x)), outputs=0, side_effects=True)
    b.edge("numbers", 0, "scale", 0)
    b.edge("scale", 0, "sink", 0)
    if collect:

        def accumulate(_):
            collected: list[Any] = []
            return Collector(collected.append, lambda: sum(collected))

        b.node("total", accumulate, kind="collector")
        b.edge("scale", 0, "total", 0)
        b.node(
            "report",
            lambda total: trace.append(("total", total)),
            outputs=0,
            side_effects=True,
        )
        b.edge("total", 0, "report", 0)
    return b.chain


def generator_items(trace: list[object]) -> Chain:
    return loop_chain(trace, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


def generator_fail_fast(trace: list[object]) -> Chain:
    return loop_chain(trace, [1, 2, ValueError("item 3 is bad"), 4, 5])


def generator_deferred(trace: list[object]) -> Chain:
    return loop_chain(
        trace,
        [1, ValueError("item 2 is bad"), 3, KeyError("item 4 is bad"), 5],
        fail_fast=False,
    )


def node_error_in_loop(trace: list[object]) -> Chain:
    def middle(x):
        if x == 3:
            raise ZeroDivisionError("item 3 divides by zero")
        return x

    return loop_chain(trace, [1, 2, 3, 4, 5], middle=middle)


def node_error_deferred(trace: list[object]) -> Chain:
    def middle(x):
        if x in (2, 4):
            raise ZeroDivisionError(f"item {x} divides by zero")
        return x

    return loop_chain(trace, [1, 2, 3, 4, 5], fail_fast=False, middle=middle)


def empty_generator(trace: list[object]) -> Chain:
    return loop_chain(trace, [])


def static_into_loop(trace: list[object]) -> Chain:
    b = Builder()
    calls: Counter[str] = Counter()

    def offset(x):
        calls["offset"] += 1
        return x

    b.node("offset", offset)
    b.value("offset", 0, 100)
    b.node(
        "numbers",
        lambda: items([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        kind="generator",
        inputs=0,
    )
    b.node("add", lambda x, y: x + y, inputs=2)
    b.node(
        "sink",
        lambda v: trace.append((v, calls["offset"])),
        outputs=0,
        side_effects=True,
    )
    b.edge("numbers", 0, "add", 0)
    b.edge("offset", 0, "add", 1)
    b.edge("add", 0, "sink", 0)
    return b.chain


def zipped_generators(trace: list[object]) -> Chain:
    b = Builder()
    b.node("left", lambda: items([1, 2, 3, 4, 5]), kind="generator", inputs=0)
    b.node("right", lambda: items([10, 20, 30, 40, 50]), kind="generator", inputs=0)
    b.node("pair", lambda x, y: x + y, inputs=2)
    b.node("sink", trace.append, outputs=0, side_effects=True)
    b.edge("left", 0, "pair", 0)
    b.edge("right", 0, "pair", 1)
    b.edge("pair", 0, "sink", 0)
    return b.chain


def unequal_lengths(trace: list[object]) -> Chain:
    b = Builder()
    b.node("left", lambda: items([1, 2, 3]), kind="generator", inputs=0)
    b.node("right", lambda: items([1, 2]), kind="generator", inputs=0)
    b.node("pair", lambda x, y: x + y, inputs=2)
    b.node("sink", trace.append, outputs=0, side_effects=True)
    b.edge("left", 0, "pair", 0)
    b.edge("right", 0, "pair", 1)
    b.edge("pair", 0, "sink", 0)
    return b.chain


def generator_static_output(trace: list[object]) -> Chain:
    b = Builder()
    b.node(
        "numbers",
        lambda: (items([1, 2, 3, 4, 5, 6, 7]), "folder"),
        kind="generator",
        inputs=0,
        outputs=2,
    )
    b.node(
        "sink",
        lambda x, folder: trace.append((x, folder)),
        inputs=2,
        outputs=0,
        side_effects=True,
    )
    b.edge("numbers", 0, "sink", 0)
    b.edge("numbers", 1, "sink", 1)
    b.node(
        "after",
        lambda folder: trace.append(("after", folder)),
        outputs=0,
        side_effects=True,
    )
    b.edge("numbers", 1, "after", 0)
    return b.chain


def collector_only(trace: list[object]) -> Chain:
    b = Builder()
    b.node(
        "numbers", lambda: items([3, 1, 4, 1, 5, 9, 2, 6]), kind="generator", inputs=0
    )

    def gather(_):
        seen: list[Any] = []
        return Collector(
            lambda v: (seen.append(v), trace.append(("iterate", v)))[0],
            lambda: max(seen),
        )

    b.node("maximum", gather, kind="collector")
    b.edge("numbers", 0, "maximum", 0)
    b.node("report", lambda v: trace.append(("max", v)), outputs=0, side_effects=True)
    b.edge("maximum", 0, "report", 0)
    return b.chain


def look_ahead_chain(trace: list[object], source_paths: tuple | None) -> Chain:
    b = Builder()
    b.node(
        "frames",
        lambda: items(list(range(12)), source_paths=source_paths),
        kind="generator",
        inputs=0,
    )
    b.node("offset", lambda x: x + 1000)

    def temporal(context, frame, offset):
        view = context.iteration
        ahead = view.lookahead(0)
        peeked = [view.peek(0, k) for k in range(1, ahead + 1)]
        peeked_offset = view.peek(1, 1) if view.lookahead(1) else None
        trace.append(
            (
                view.index,
                frame,
                ahead,
                peeked,
                peeked_offset,
                view.iterated_by(0),
                view.source,
            )
        )
        state = view.state or []
        view.state = [*state, frame]
        return frame

    b.node("temporal", temporal, inputs=2, context=True, side_effects=True)
    b.edge("frames", 0, "temporal", 0)
    b.edge("frames", 0, "offset", 0)
    b.edge("offset", 0, "temporal", 1)
    return b.chain


def look_ahead(trace: list[object]) -> Chain:
    return look_ahead_chain(trace, ())


def look_ahead_blocked(trace: list[object]) -> Chain:
    # Unknown sources block reading ahead: K = 1, a warning, and every look-ahead empty.
    return look_ahead_chain(trace, None)


def look_ahead_serial_chain(trace: list[object], fail_fast: bool) -> Chain:
    # A look-ahead through a serial node (a context node, as a GPU upscale): computed ahead on the driver one item at a
    # time, in order, with its iteration view at that item and its state shared (process.py's serial candidates),
    # only when every error ends the loop.
    b = Builder()
    b.node(
        "frames",
        lambda: items(list(range(10)), fail_fast=fail_fast),
        kind="generator",
        inputs=0,
    )

    def upscale(context, x):
        view = context.iteration
        view.state = (view.state or 0) + 1
        trace.append(("upscale", view.index, x, view.state))
        return x * 100

    b.node("upscale", upscale, context=True)

    def temporal(context, frame):
        view = context.iteration
        ahead = view.lookahead(0)
        trace.append(
            (
                "temporal",
                view.index,
                frame,
                [view.peek(0, k) for k in range(1, ahead + 1)],
            )
        )
        return frame

    b.node("temporal", temporal, context=True, side_effects=True)
    b.edge("frames", 0, "upscale", 0)
    b.edge("upscale", 0, "temporal", 0)
    return b.chain


def look_ahead_serial(trace: list[object]) -> Chain:
    return look_ahead_serial_chain(trace, True)


def look_ahead_serial_deferred_errors(trace: list[object]) -> Chain:
    return look_ahead_serial_chain(trace, False)


def look_ahead_nested(trace: list[object]) -> Chain:
    # A serial node that looks ahead itself (as VOSR) feeding a node that looks ahead at it (as Temporal Fix): each of
    # its computations ahead peeks up to its own item + K - 1, past the committing item's window.
    b = Builder()
    b.node("frames", lambda: items(list(range(14))), kind="generator", inputs=0)

    def vosr(context, frame):
        view = context.iteration
        peeked = [view.peek(0, k) for k in range(1, view.lookahead(0) + 1)]
        trace.append(("vosr", view.index, frame, peeked))
        return frame * 100 + sum(p or 0 for p in peeked)

    b.node("vosr", vosr, context=True)

    def temporal(context, frame):
        view = context.iteration
        trace.append(
            (
                "temporal",
                view.index,
                frame,
                [view.peek(0, k) for k in range(1, view.lookahead(0) + 1)],
            )
        )

    b.node("temporal", temporal, context=True, side_effects=True, outputs=0)
    b.edge("frames", 0, "vosr", 0)
    b.edge("vosr", 0, "temporal", 0)
    return b.chain


def failed_prefetched_branch(trace: list[object]) -> Chain:
    # Two independent sinks, the first's branch failing: the second's, computed ahead beside it, sends no events
    # (process.Executor never runs it).
    b = Builder()

    def fails(_):
        time.sleep(0.05)
        raise ValueError("branch a fails")

    b.node("pa", fails)
    b.value("pa", 0, 1)
    b.node("pb", lambda x: x + 1)
    b.value("pb", 0, 2)
    b.node("sink_a", lambda x: trace.append(("a", x)), outputs=0, side_effects=True)
    b.node("sink_b", lambda x: trace.append(("b", x)), outputs=0, side_effects=True)
    b.edge("pa", 0, "sink_a", 0)
    b.edge("pb", 0, "sink_b", 0)
    return b.chain


def side_effect_feeding(trace: list[object]) -> Chain:
    # A side-effect node read by another node: consumed by its reader and by the run (its extra reservation).
    b = Builder()
    b.node(
        "writer", lambda x: (trace.append(("write", x)), x + 1)[1], side_effects=True
    )
    b.value("writer", 0, 1)
    b.node("reader", lambda x: trace.append(("read", x)), outputs=0, side_effects=True)
    b.edge("writer", 0, "reader", 0)
    return b.chain


def collector_iterate_error(trace: list[object]) -> Chain:
    b = Builder()
    b.node("numbers", lambda: items([1, 2, 3, 4, 5, 6]), kind="generator", inputs=0)

    def gather(_):
        def iterate(v):
            if v == 4:
                raise RuntimeError("the collector refuses 4")
            trace.append(("iterate", v))

        return Collector(iterate, lambda: None)

    b.node("refuse", gather, kind="collector", outputs=0)
    b.edge("numbers", 0, "refuse", 0)
    return b.chain


def collector_construction_error(trace: list[object]) -> Chain:
    b = Builder()
    b.node("numbers", lambda: items([1, 2, 3]), kind="generator", inputs=0)

    def broken(_, setting):
        raise ValueError(f"the collector cannot start with {setting}")

    b.node("broken", broken, kind="collector", inputs=2, outputs=0)
    b.edge("numbers", 0, "broken", 0)
    b.value("broken", 1, "bad setting")
    return b.chain


def sink_after_loop_failure(trace: list[object]) -> Chain:
    # A non-iterated sink after a loop that fails fast: it never runs.
    chain = loop_chain(trace, [1, ValueError("stop here"), 3], collect=False)
    b = Builder()
    b.chain = chain
    b.node("later", lambda x: trace.append(("later", x)), outputs=0, side_effects=True)
    b.value("later", 0, 9)
    return b.chain


SCENARIOS = [
    independent_sinks,
    diamond,
    lazy_unread,
    lazy_read,
    lazy_constant,
    lazy_upstream_error,
    input_order_errors,
    context_cleanups,
    attributed_error,
    generator_items,
    generator_fail_fast,
    generator_deferred,
    node_error_in_loop,
    node_error_deferred,
    empty_generator,
    static_into_loop,
    zipped_generators,
    unequal_lengths,
    generator_static_output,
    collector_only,
    look_ahead,
    look_ahead_blocked,
    look_ahead_serial,
    look_ahead_serial_deferred_errors,
    failed_prefetched_branch,
    side_effect_feeding,
    collector_iterate_error,
    collector_construction_error,
    sink_after_loop_failure,
]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.__name__ for s in SCENARIOS])
def test_scenario_matches(scenario, tmp_path):
    assert_same(scenario, tmp_path)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.__name__ for s in SCENARIOS])
def test_scenario_matches_without_broadcasts(scenario, tmp_path):
    assert_same(scenario, tmp_path, send_broadcast_data=False)


def test_nested_look_ahead(tmp_path):
    # A stop after a minute turns a hang into a failure.
    timers: list[threading.Timer] = []

    def watchdog(executor):
        timer = threading.Timer(60, executor.kill)
        timer.daemon = True
        timer.start()
        timers.append(timer)

    try:
        engine = assert_same(look_ahead_nested, tmp_path, during=watchdog)
    finally:
        for timer in timers:
            timer.cancel()
    assert engine.error is None


@pytest.mark.parametrize("kind", ["python", "native"])
def test_a_node_s_time_leaves_out_its_waits_for_other_work(kind, tmp_path):
    # TensorRT Upscale Image's items ahead queue for one session (util.held): a node reports its own work, not its
    # turn in that queue (9-11 s shown for a 1.5 s upscale, 2026-10-14; again in the run engine, 2026-10-21).
    def upscale() -> int:
        with util.waiting():
            time.sleep(0.5)
        return 1

    b = Builder()
    b.node("upscale", upscale, inputs=0, side_effects=True)
    result = execute(kind, b.chain, storage=tmp_path / "storage")
    assert result.error is None, result.error
    times = [e["data"]["executionTime"] for e in result.events if e["event"] == "node-finish"]
    assert len(times) == 1
    assert times[0] < 0.25


def test_pause_and_resume(tmp_path):
    def during(executor):
        time.sleep(0.02)
        executor.pause()
        time.sleep(0.3)
        assert executor.progress.paused
        executor.resume()

    def slow_items(trace: list[object]) -> Chain:
        return loop_chain(
            trace, list(range(20)), middle=lambda x: (time.sleep(0.01), x)[1]
        )

    assert_same(slow_items, tmp_path, during=during)


def test_stop_ends_with_aborted_and_cleanups(tmp_path):
    started = threading.Event()

    def stop_chain(trace: list[object]) -> Chain:
        b = Builder()
        b.node("numbers", lambda: items(list(range(200))), kind="generator", inputs=0)

        def cleanup():
            trace.append("chain cleanup")

        def body(context, x):
            context.add_cleanup(cleanup)
            started.set()
            time.sleep(0.005)
            trace.append(x)

        b.node("sink", body, outputs=0, side_effects=True, context=True)
        b.edge("numbers", 0, "sink", 0)
        return b.chain

    def during(executor):
        started.wait(10)
        time.sleep(0.05)
        executor.kill()

    for kind in ("python", "native"):
        started.clear()
        trace: list[object] = []
        result = execute(
            kind, stop_chain(trace), storage=tmp_path / "storage", during=during
        )
        assert type(result.error).__name__ == "Aborted", (kind, result.error)
        assert trace[-1] == "chain cleanup" and trace.count("chain cleanup") == 1, kind
        assert 0 < len(trace) - 1 < 200, kind


def test_parent_cache_nodes_are_not_run(tmp_path):
    # /run with the server's cache of individually run nodes: those are neither run, started nor listed.
    def cached_chain(trace: list[object]) -> Chain:
        b = Builder()
        b.node(
            "loaded", lambda x: trace.append(("loaded ran", x)) or x, side_effects=True
        )
        b.value("loaded", 0, 5)
        b.node("use", lambda x: trace.append(("use", x)), outputs=0, side_effects=True)
        b.edge("loaded", 0, "use", 0)
        return b.chain

    cache = {NodeId("loaded"): RegularOutput([42])}
    assert_same(cached_chain, tmp_path, parent_cache=cache)


@pytest.mark.parametrize("kind", ["regularNode", "generator"])
def test_individual_runs_match(tmp_path, kind):
    """/run/individual: one node, its events filtered as the server filters them, its output as the server caches
    it (a regular node's)."""

    def single(trace: list[object]) -> Chain:
        b = Builder()
        if kind == "generator":
            b.node(
                "single",
                lambda x: (items([x, x + 1]), "static"),
                kind="generator",
                inputs=1,
                outputs=2,
            )
        else:
            b.node("single", lambda x: (trace.append(x), x * 3)[1], side_effects=True)
        b.value("single", 0, 7)
        return b.chain

    class Sink(EventConsumer):
        def __init__(self, events: list[dict]) -> None:
            self.events = events

        def put(self, event) -> None:
            self.events.append(json.loads(json.dumps(event)))

    async def individual(executor_kind: str, chain: Chain, events: list[dict]):
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(8)
        queue = EventConsumer.filter(
            Sink(events), {"node-finish", "node-broadcast", "execution-error"}
        )
        arguments: dict[str, Any] = {
            "id": ExecutionId("individual-executor single"),
            "chain": chain,
            "send_broadcast_data": True,
            "options": ExecutionOptions.parse({}),
            "loop": loop,
            "queue": queue,
            "storage_dir": tmp_path,
            "pool": pool,
            "pool_size": 8,
        }
        executor = (
            Executor(**arguments)
            if executor_kind == "python"
            else NativeExecutor(**arguments)
        )
        node = chain.nodes[NodeId("single")]
        if kind == "generator":
            assert isinstance(node, GeneratorNode)
            output = await executor.process_generator_node(node)
            value = (output.partial_output, output.generator.expected_length)
        else:
            assert isinstance(node, FunctionNode)
            output = await executor.process_regular_node(node)
            value = output.output
        await executor.flush_broadcasts()
        pool.shutdown()
        return value

    outcomes = []
    for executor_kind in ("python", "native"):
        trace: list[object] = []
        events: list[dict] = []
        value = asyncio.run(individual(executor_kind, single(trace), events))
        outcomes.append((value, [canonical(e, tmp_path) for e in events], trace))
    assert outcomes[1] == outcomes[0]


# Context frames (native/local/video-seek/DESIGN.md): a generator reading a range of its source also reads, for the
# loop's temporal nodes, the neighbours they need (NodeData.temporal_context, Generator.context). Items before the
# range run the temporal nodes and the nodes they need alone; items after it only serve look-aheads. Each test runs on
# both executors (CONTEXT_EXECUTORS), and each must see what it sees in its own full run.

CONTEXT_EXECUTORS = ["python", "native"]


class Frames:
    """A source of `total` items, item i being i * 10 + 1, read whole or as a range with context frames, as Load Video
    reads a video in All or Range mode. `exists`: the items there are, fewer when `total` is an estimate. `pulled`:
    the item numbers its reads yielded, in order; `asked`: each context request (before, after)."""

    def __init__(self, total: int, exists: int | None = None) -> None:
        self.total = total
        self.exists = total if exists is None else exists
        self.pulled: list[int] = []
        self.asked: list[tuple[int, int]] = []

    def read(self, start: int, stop: int):
        for i in range(start, min(stop, self.exists)):
            self.pulled.append(i)
            yield i * 10 + 1

    def whole(self) -> Generator:
        return Generator(lambda: self.read(0, self.total), self.total, source_paths=())

    def range(self, base: int, length: int, chosen: int | None = None) -> Generator:
        def context(before: int, after: int):
            self.asked.append((before, after))
            b, a = min(before, base), min(after, self.total - base - length)
            return self.read(base - b, base + length + a), b, a

        generator = Generator(
            lambda: self.read(base, base + length), length, source_paths=()
        )
        return generator.with_range(base, chosen, context)


def records(trace: list[object], tag: str) -> list[tuple[Any, ...]]:
    """The records one node made, its tag left out."""
    return [r[1:] for r in trace if isinstance(r, tuple) and r[0] == tag]


def progress(result: Result, node: str = "frames") -> list[tuple[int, int]]:
    """The node's progress events as (index, total)."""
    return [
        (e["data"]["index"], e["data"]["total"])
        for e in result.events
        if e["event"] == "node-progress" and e["data"]["nodeId"] == node
    ]


def per_node(result: Result, event: str) -> Counter[str]:
    """How many events of this kind each node sent."""
    return Counter(e["data"]["nodeId"] for e in result.events if e["event"] == event)


def context_chain(
    trace: list[object],
    generator: Callable[[], Generator],
    *,
    temporal: tuple[int, int] | None = (3, 2),
    serial: bool = False,
    on_temporal: Callable[[int], None] = lambda _index: None,
) -> Chain:
    """frames -> offset -> temporal -> sink, temporal -> collect -> report, and frames -> other -> sink.

    temporal declares the neighbours `temporal` and looks ahead two items. It records per call its index, context
    flag, input, state (its last three (index, input) pairs, restarted after a gap), lookahead, peeks and chosen
    item; on_temporal(index) runs first. offset (x + 1000) is pure, or with `serial` a context node (as a GPU
    upscale) recording its own view; other (x * 2) is pure and feeds the sink alone. Every node records its runs."""
    b = Builder()
    b.node("frames", generator, kind="generator", inputs=0)
    if serial:

        def offset(context, x):
            view = context.iteration
            trace.append(("offset", view.index, view.context, x))
            return x + 1000

        b.node("offset", offset, context=True)
    else:
        b.node("offset", lambda x: (trace.append(("offset", x)), x + 1000)[1])

    def temporal_node(context, value):
        view = context.iteration
        on_temporal(view.index)
        state = view.state or ()
        if state and state[-1][0] != view.index - 1:
            state = ()
        peeks = [view.peek(0, k) for k in (1, 2)]
        trace.append(
            (
                "temporal",
                view.index,
                view.context,
                value,
                state,
                view.lookahead(0),
                peeks,
                context.chosen_item,
            )
        )
        view.state = (*state, (view.index, value))[-3:]
        return value

    b.node("temporal", temporal_node, context=True, temporal=temporal)
    b.node("other", lambda x: (trace.append(("other", x)), x * 2)[1])

    def sink(context, value, doubled):
        view = context.iteration
        trace.append(
            ("sink", view.index, view.context, value, doubled, context.chosen_item)
        )

    b.node("sink", sink, inputs=2, outputs=0, side_effects=True, context=True)

    def gather(context, _):
        trace.append(("collector", context.chosen_item))
        collected: list[Any] = []

        def iterate(value):
            trace.append(("collect", value))
            collected.append(value)

        return Collector(iterate, lambda: len(collected))

    b.node("collect", gather, kind="collector", context=True)
    b.node(
        "report",
        lambda context, count: trace.append(("report", count, context.chosen_item)),
        outputs=0,
        side_effects=True,
        context=True,
    )
    b.edge("frames", 0, "offset", 0)
    b.edge("offset", 0, "temporal", 0)
    b.edge("frames", 0, "other", 0)
    b.edge("temporal", 0, "sink", 0)
    b.edge("other", 0, "sink", 1)
    b.edge("temporal", 0, "collect", 0)
    b.edge("collect", 0, "report", 0)
    return b.chain


def run_context(
    kind: str,
    scenario: Scenario,
    tmp_path: Path,
    monkeypatch,
    window: str,
    during: Callable[[object], None] | None = None,
) -> tuple[Result, list[object], Chain]:
    """The scenario run on one executor with the item window K = `window`. A run still going after a minute is
    stopped, so a hang fails the test."""
    monkeypatch.setenv(ITEM_WINDOW_VARIABLE, window)
    trace: list[object] = []
    chain = scenario(trace)
    timers: list[threading.Timer] = []

    def watched(executor) -> None:
        timer = threading.Timer(60, executor.kill)
        timer.daemon = True
        timer.start()
        timers.append(timer)
        if during is not None:
            during(executor)

    try:
        result = execute(kind, chain, storage=tmp_path / "storage", during=watched)
    finally:
        for timer in timers:
            timer.cancel()
    return result, trace, chain


@pytest.mark.parametrize("producer", ["phased", "serial-producer"])
@pytest.mark.parametrize("serial", [False, True], ids=["pure", "serial"])
@pytest.mark.parametrize("kind", CONTEXT_EXECUTORS)
def test_context_frames_give_a_range_its_neighbours(
    kind, serial, producer, monkeypatch, tmp_path
):
    # Items 8..11 of 20, temporal (3, 2), K = 4: temporal runs for 5..11, 5..7 as context items, and at 8..11 sees
    # what it sees there in a full run (state, and peeks reaching 12 and 13 after the range). offset, which it needs,
    # runs for every item read; nothing else runs for a context item, and the sink, the collector, the broadcasts and
    # the progress see the range alone. Served through offset computed ahead, pure or serial (its own view then
    # counts the items as temporal's does), with process.Executor's two-phase or serial producer.
    if producer == "serial-producer":
        monkeypatch.setenv(SERIAL_PRODUCER_VARIABLE, "1")
    else:
        monkeypatch.delenv(SERIAL_PRODUCER_VARIABLE, raising=False)
    whole = Frames(20)
    full, full_trace, _ = run_context(
        kind,
        lambda t: context_chain(t, whole.whole, serial=serial),
        tmp_path,
        monkeypatch,
        "4",
    )
    # The native engine's broadcasts as it sends them, (node, first output value).
    sent: list[tuple[str, object]] = []
    broadcast = native_executor._Hooks.broadcast

    def record_broadcast(hooks: native_executor._Hooks, index, outputs, *rest):
        sent.append((hooks.nodes[index].id, outputs[0]))
        return broadcast(hooks, index, outputs, *rest)

    monkeypatch.setattr(native_executor._Hooks, "broadcast", record_broadcast)
    frames = Frames(20)
    result, trace, chain = run_context(
        kind,
        lambda t: context_chain(t, lambda: frames.range(8, 4), serial=serial),
        tmp_path,
        monkeypatch,
        "4",
    )
    assert full.error is None and result.error is None, (full.error, result.error)
    assert whole.asked == [] and frames.asked == [(3, 2)]
    seen = records(trace, "temporal")
    assert [r[:2] for r in seen] == [(i, i < 8) for i in range(5, 12)]
    assert seen[3:] == records(full_trace, "temporal")[8:12]
    assert seen[3][3] == ((5, 1051), (6, 1061), (7, 1071))  # the context items
    assert seen[-1][4:6] == (3, [1121, 1131])  # the items after the range
    assert records(trace, "sink") == [
        (i, False, i * 10 + 1001, i * 20 + 2, None) for i in range(8, 12)
    ]
    assert sorted(records(trace, "other")) == [(i * 10 + 1,) for i in range(8, 12)]
    if serial:
        assert records(trace, "offset") == [
            (i, i < 8, i * 10 + 1) for i in range(5, 14)
        ]
    else:
        assert sorted(records(trace, "offset")) == [(i * 10 + 1,) for i in range(5, 14)]
    assert records(trace, "collector") == [(None,)]
    assert records(trace, "collect") == [(i * 10 + 1001,) for i in range(8, 12)]
    assert records(trace, "report") == [(4, None)]
    assert frames.pulled == list(range(5, 14))
    assert progress(result) == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (4, 4)]
    broadcasts = per_node(result, "node-broadcast")
    # frames: its sequence, the range's 4 items, its end
    counts = (broadcasts["frames"], broadcasts["offset"], broadcasts["temporal"])
    if kind == "python":
        assert counts == (6, 4, 4)
    else:
        # The engine keeps one pending broadcast per node (LatestBroadcasts; the oracle sends each one), so it may
        # send fewer: each of the range's items, never a context item's.
        assert all(0 < count <= most for count, most in zip(counts, (6, 4, 4)))
        range_values = {i * 10 + 1001 for i in range(8, 12)}
        assert {value for node, value in sent if node == "frames"} <= {
            None,
            *(i * 10 + 1 for i in range(8, 12)),
        }
        assert {v for n, v in sent if n in ("offset", "temporal")} <= range_values
    assert per_node(result, "node-finish")["temporal"] == 7
    assert order_violations(result.events, chain) == []


@pytest.mark.parametrize("kind", CONTEXT_EXECUTORS)
def test_context_frames_without_a_window(kind, monkeypatch, tmp_path):
    # K = 1 serves no look-ahead, so nothing after the range is read and nothing waits for it; the items before it
    # still give temporal at 8..11 what a full run at K = 1 gives it there.
    whole, frames = Frames(20), Frames(20)
    full, full_trace, _ = run_context(
        kind, lambda t: context_chain(t, whole.whole), tmp_path, monkeypatch, "1"
    )
    result, trace, _ = run_context(
        kind,
        lambda t: context_chain(t, lambda: frames.range(8, 4)),
        tmp_path,
        monkeypatch,
        "1",
    )
    assert full.error is None and result.error is None, (full.error, result.error)
    assert frames.asked == [(3, 2)] and frames.pulled == list(range(5, 12))
    seen = records(trace, "temporal")
    assert [r[:2] for r in seen] == [(i, i < 8) for i in range(5, 12)]
    assert all(r[4:6] == (0, [None, None]) for r in seen)
    assert seen[3:] == records(full_trace, "temporal")[8:12]
    assert [r[0] for r in records(trace, "sink")] == list(range(8, 12))
    assert progress(result)[-1] == (4, 4)


@pytest.mark.parametrize("kind", CONTEXT_EXECUTORS)
def test_context_frames_are_clipped_at_the_source_s_start(kind, monkeypatch, tmp_path):
    # Items 1..4 want 3 before, and only item 0 exists: one context item, where temporal starts as a full run does.
    whole, frames = Frames(20), Frames(20)
    full, full_trace, _ = run_context(
        kind, lambda t: context_chain(t, whole.whole), tmp_path, monkeypatch, "4"
    )
    result, trace, _ = run_context(
        kind,
        lambda t: context_chain(t, lambda: frames.range(1, 4)),
        tmp_path,
        monkeypatch,
        "4",
    )
    assert full.error is None and result.error is None, (full.error, result.error)
    assert frames.asked == [(3, 2)] and frames.pulled == list(range(7))
    seen = records(trace, "temporal")
    assert [r[:2] for r in seen] == [(i, i < 1) for i in range(5)]
    assert seen[1:] == records(full_trace, "temporal")[1:5]
    assert [r[0] for r in records(trace, "sink")] == list(range(1, 5))


@pytest.mark.parametrize("exists", [13, 12, 10])
@pytest.mark.parametrize("kind", CONTEXT_EXECUTORS)
def test_a_source_ending_inside_the_context_or_the_range(
    kind, exists, monkeypatch, tmp_path
):
    # The source claims 20 items (an estimate) and has `exists`: 13 and 12 end among the items after the range 8..11,
    # whose last peeks then find nothing; 10 ends inside the range, so the loop ends after item 9. Each run matches a
    # full read of the same source at those items, and nothing hangs.
    whole, frames = Frames(20, exists), Frames(20, exists)
    full, full_trace, _ = run_context(
        kind, lambda t: context_chain(t, whole.whole), tmp_path, monkeypatch, "4"
    )
    result, trace, _ = run_context(
        kind,
        lambda t: context_chain(t, lambda: frames.range(8, 4)),
        tmp_path,
        monkeypatch,
        "4",
    )
    assert full.error is None and result.error is None, (full.error, result.error)
    last = min(11, exists - 1)
    seen = records(trace, "temporal")
    assert [r[:2] for r in seen] == [(i, i < 8) for i in range(5, last + 1)]
    assert seen[3:] == records(full_trace, "temporal")[8 : last + 1]
    assert [r[0] for r in records(trace, "sink")] == list(range(8, last + 1))
    assert frames.pulled == list(range(5, min(exists, 14)))
    assert progress(result)[-1] == (last - 7, last - 7)


@pytest.mark.parametrize("kind", CONTEXT_EXECUTORS)
def test_a_loop_without_temporal_nodes_reads_its_range_alone(
    kind, monkeypatch, tmp_path
):
    # temporal declares no neighbours: the generator's context is never asked for and only the range is read; the
    # index still counts from the range's start.
    frames = Frames(20)
    result, trace, _ = run_context(
        kind,
        lambda t: context_chain(t, lambda: frames.range(8, 4), temporal=None),
        tmp_path,
        monkeypatch,
        "4",
    )
    assert result.error is None, result.error
    assert frames.asked == [] and frames.pulled == [8, 9, 10, 11]
    seen = records(trace, "temporal")
    assert [r[:2] for r in seen] == [(i, False) for i in range(8, 12)]
    assert seen[-1][5] == [None, None]
    assert [r[0] for r in records(trace, "sink")] == list(range(8, 12))
    assert progress(result)[-1] == (4, 4)


@pytest.mark.parametrize("kind", CONTEXT_EXECUTORS)
def test_a_one_item_run_shows_its_chosen_item_to_the_loop(kind, monkeypatch, tmp_path):
    # "Run this frame" on item 8: the loop's nodes, temporal on its context items too, and its collector see
    # chosen_item 8; report, after the loop, does not. temporal still gets 5..7 before it and 9, 10 after it.
    frames = Frames(20)
    result, trace, _ = run_context(
        kind,
        lambda t: context_chain(t, lambda: frames.range(8, 1, chosen=8)),
        tmp_path,
        monkeypatch,
        "4",
    )
    assert result.error is None, result.error
    seen = records(trace, "temporal")
    assert [(r[0], r[1], r[6]) for r in seen] == [(i, i < 8, 8) for i in range(5, 9)]
    assert seen[-1][5] == [1091, 1101]
    assert records(trace, "sink") == [(8, False, 1081, 162, 8)]
    assert records(trace, "collector") == [(8,)]
    assert records(trace, "report") == [(1, None)]
    assert progress(result) == [(0, 1), (1, 1), (1, 1)]


@pytest.mark.parametrize("kind", CONTEXT_EXECUTORS)
def test_stop_during_a_context_item_ends_the_run_cleanly(kind, monkeypatch, tmp_path):
    # Stop while temporal runs for context item 6: its look-ahead raises Aborted, the run ends with it, and nothing
    # reached the sink, the collector or the progress.
    reached, released = threading.Event(), threading.Event()

    def hold(index):
        if index == 6:
            reached.set()
            assert released.wait(10)

    def stop(executor):
        assert reached.wait(10)
        executor.kill()
        released.set()

    frames = Frames(20)
    result, trace, _ = run_context(
        kind,
        lambda t: context_chain(t, lambda: frames.range(8, 4), on_temporal=hold),
        tmp_path,
        monkeypatch,
        "4",
        during=stop,
    )
    assert type(result.error).__name__ == "Aborted", result.error
    assert [r[0] for r in records(trace, "temporal")] == [5]
    assert not any(records(trace, tag) for tag in ("sink", "collect", "report"))
    assert progress(result) == [(0, 4)]


@pytest.mark.parametrize("serial", [False, True], ids=["pure", "serial"])
def test_context_frames_match_across_executors(serial, monkeypatch, tmp_path):
    # The native engine's range with context frames is the Python executor's: the same events (the engine's in A6's
    # order) and reads, and per node the same records, in the driver's order for the nodes it runs one at a time.
    def run(kind: str) -> tuple[Result, list[object], list[int], Chain]:
        frames = Frames(20)
        result, trace, chain = run_context(
            kind,
            lambda t: context_chain(t, lambda: frames.range(8, 4), serial=serial),
            tmp_path,
            monkeypatch,
            "4",
        )
        assert result.error is None, result.error
        return result, trace, frames.pulled, chain

    oracle, oracle_trace, oracle_pulled, _ = run("python")
    engine, engine_trace, engine_pulled, chain = run("native")
    assert mismatches("context frames", chain, oracle, engine, tmp_path) == []
    assert engine_pulled == oracle_pulled
    ordered = ["temporal", "sink", "collector", "collect", "report"]
    for tag in [*ordered, "offset"] if serial else ordered:
        assert records(engine_trace, tag) == records(oracle_trace, tag), tag
    for tag in ["other"] if serial else ["other", "offset"]:
        assert sorted(records(engine_trace, tag)) == sorted(
            records(oracle_trace, tag)
        ), tag


@pytest.mark.parametrize("kind", CONTEXT_EXECUTORS)
def test_context_items_start_no_sink_prepare(kind, monkeypatch, tmp_path):
    # save has prepare and commit phases (as Save Image). The window, decided by temporal's look-ahead during the first
    # context item, may prepare save ahead for the range's items, never for a context item before or after the range,
    # where save never runs.
    monkeypatch.setattr("nodes.impl.item_window._PHASES", {})
    prepared: list[int] = []

    def prepare(enforced):
        prepared.append(enforced[0])
        return enforced[0]

    frames = Frames(20)

    def scenario(trace: list[object]) -> Chain:
        register_phases(
            "chainner:utility:test_save",
            prepare,
            lambda _enforced, value: trace.append(("save", value.result())),
        )
        b = Builder()
        b.node("frames", lambda: frames.range(8, 4), kind="generator", inputs=0)

        def temporal(context, x):
            context.iteration.peek(0, 1)
            return x

        b.node("temporal", temporal, context=True, temporal=(3, 2))
        b.node(
            "save", lambda x: trace.append(("save", x)), outputs=0, side_effects=True
        )
        b.edge("frames", 0, "temporal", 0)
        b.edge("temporal", 0, "save", 0)
        return b.chain

    result, trace, _ = run_context(kind, scenario, tmp_path, monkeypatch, "4")
    assert result.error is None, result.error
    assert trace == [("save", i * 10 + 1) for i in range(8, 12)]
    assert set(prepared) <= {i * 10 + 1 for i in range(8, 12)}


def test_native_window_is_decided_after_the_range_s_first_item(monkeypatch, tmp_path):
    # The native engine: with no look-ahead before it, the window is decided at the range's second item (slot 4, after
    # three context items), and its per-item estimate measures the range's first item: big's 4 KiB, which no context
    # item computes. With a budget of one such item, the producer reads one item ahead of the committing one: item 10 at
    # most while item 9 is saved (K = 4 alone would allow item 12).
    decided: list[int] = []
    window = native_executor._Hooks.window

    def record_window(hooks: native_executor._Hooks, *arguments: Any):
        assert hooks.engine is not None
        decided.append(hooks.engine.commit_slot)
        return window(hooks, *arguments)

    monkeypatch.setattr(native_executor._Hooks, "window", record_window)
    monkeypatch.setattr(native_executor, "memory_budget", lambda: 4096)
    frames = Frames(20)
    # Per item saved: its value, big's bytes and the last item read by then.
    saved: list[tuple[int, int, int]] = []

    def scenario(_trace: list[object]) -> Chain:
        b = Builder()
        b.node("frames", lambda: frames.range(8, 4), kind="generator", inputs=0)
        b.node("temporal", lambda context, x: x, context=True, temporal=(3, 2))
        b.node("big", lambda x: np.zeros(4096, np.uint8))

        def save(x, data):
            if x == 91:
                # The producer reads ahead meanwhile, as far as the budget lets it.
                time.sleep(0.3)
            saved.append((x, data.nbytes, max(frames.pulled)))

        b.node("save", save, inputs=2, outputs=0, side_effects=True)
        b.edge("frames", 0, "temporal", 0)
        b.edge("frames", 0, "big", 0)
        b.edge("temporal", 0, "save", 0)
        b.edge("big", 0, "save", 1)
        return b.chain

    result, _, _ = run_context("native", scenario, tmp_path, monkeypatch, "4")
    assert result.error is None, result.error
    assert [s[:2] for s in saved] == [(i * 10 + 1, 4096) for i in range(8, 12)]
    assert decided == [4]
    assert saved[1][2] <= 10


def test_python_window_is_decided_after_the_range_s_first_item(monkeypatch, tmp_path):
    # process.Executor: with no look-ahead before it, the window is decided at the range's second item (slot 4, after
    # three context items), and its per-item estimate measures the range's first item: big's 4 KiB, which no context
    # item computes. The window learns the context slots: before 3, from 7 on (items 12 and 13), and the nodes they
    # run (temporal).
    opened: list[dict[str, Any]] = []
    begun: list[int] = []
    window_class, begin = process.ItemWindow, ItemWindow.begin

    def capture(**options):
        opened.append(options)
        return window_class(**options)

    def record_begin(window, slot):
        begun.append(slot)
        begin(window, slot)

    monkeypatch.setattr(process, "ItemWindow", capture)
    monkeypatch.setattr(ItemWindow, "begin", record_begin)
    frames = Frames(20)

    def scenario(trace: list[object]) -> Chain:
        b = Builder()
        b.node("frames", lambda: frames.range(8, 4), kind="generator", inputs=0)
        b.node("temporal", lambda context, x: x, context=True, temporal=(3, 2))
        b.node("big", lambda x: np.zeros(4096, np.uint8))
        b.node(
            "sink",
            lambda x, data: trace.append((x, data.nbytes)),
            inputs=2,
            outputs=0,
            side_effects=True,
        )
        b.edge("frames", 0, "temporal", 0)
        b.edge("frames", 0, "big", 0)
        b.edge("temporal", 0, "sink", 0)
        b.edge("big", 0, "sink", 1)
        return b.chain

    result, trace, _ = run_context("python", scenario, tmp_path, monkeypatch, "4")
    assert result.error is None, result.error
    assert trace == [(i * 10 + 1, 4096) for i in range(8, 12)]
    [options] = opened
    assert options["estimate"] == 4096
    assert (options["first"], options["end"], options["context_nodes"]) == (
        3,
        7,
        frozenset({"temporal"}),
    )
    assert begun[0] == 4
