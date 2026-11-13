"""The native run engine against process.Executor on executor scenarios: chains of
test nodes (Python bodies, no registry) run whole through both executors in this process
(native/tools/engine_differential.py), compared by outcome, SSE events (verify_runtime.sse_event_mismatches), the
engine's A6 order invariants, and the side effects the nodes record (their calls, in order)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import weakref
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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
    Lazy,
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
from nodes.impl import execution_scheduler
from nodes.impl.execution_scheduler import ALWAYS_READ_LAZY_INPUTS
from nodes.impl.item_window import (
    ITEM_WINDOW_VARIABLE,
    SERIAL_PRODUCER_VARIABLE,
    ItemWindow,
    register_phases,
)
from nodes.properties.inputs import DirectoryInput
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
        gate: tuple[int, Callable[[object], bool]] | None = None,
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
            temporal_if=None if gate is None else (InputId(gate[0]), gate[1]),
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


def generator_attributed_error(trace: list[object]) -> Chain:
    # VapourSynth's Clip Frames: a frame fails in the statement of another node of the script.
    return loop_chain(trace, [1, NodeAttributedError("frame 2 failed", "sink"), 3])


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


def generator_without_sinks(trace: list[object]) -> Chain:
    # A seek bar's preview run: Clip Frames behind its clip feeds nothing; it runs for
    # its static output (a Preview) and its items are not read.
    b = Builder()
    b.node("clip", lambda: 5, inputs=0)

    def numbers(clip, _seek):
        trace.append(("ran", clip))

        def supplier():
            for value in (1, 2, 3):
                trace.append(("item", value))
                yield value

        return Generator(supplier, 3), "preview"

    frames = b.node(
        "numbers", numbers, kind="generator", inputs=2, outputs=2, side_effects=True
    )
    seek = BaseInput("any", "Seek", kind="video-seek").with_id(1)
    object.__setattr__(
        frames, "data", replace(frames.data, inputs=[frames.data.inputs[0], seek])
    )
    b.edge("clip", 0, "numbers", 0)
    b.value("numbers", 1, 0)

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
    generator_attributed_error,
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
    generator_without_sinks,
]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.__name__ for s in SCENARIOS])
def test_scenario_matches(scenario, tmp_path):
    assert_same(scenario, tmp_path)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.__name__ for s in SCENARIOS])
def test_scenario_matches_without_broadcasts(scenario, tmp_path):
    assert_same(scenario, tmp_path, send_broadcast_data=False)


def stray_generator(trace: list[object]) -> Chain:
    # An unconnected generator without a seek bar (Load Images or Load Video whose file
    # is gone) feeds nothing and stays skipped: it neither runs nor fails the run.
    b = Builder()

    def stray():
        trace.append("stray ran")
        raise FileNotFoundError("the folder is gone")

    b.node("stray", stray, kind="generator", inputs=0, side_effects=True)
    b.node("later", lambda x: trace.append(("later", x)), outputs=0, side_effects=True)
    b.value("later", 0, 9)
    return b.chain


def test_a_stray_generator_without_a_seek_bar_is_skipped(tmp_path):
    _, _, _, oracle_trace, engine_trace = run_both(stray_generator, tmp_path)

    assert oracle_trace == engine_trace == [("later", 9)]


def test_a_generator_without_sinks_runs_without_reading_its_items(tmp_path):
    _, _, _, oracle_trace, engine_trace = run_both(generator_without_sinks, tmp_path)

    assert oracle_trace == engine_trace == [("ran", 5), ("later", 9)]


def test_generator_item_errors_name_their_node(tmp_path):
    # A generator's item error is the generator's (process.loop_node_error), and one that names another node is
    # moved to that node (on_failing_node), on both executors.
    for scenario, node_id in (
        (generator_fail_fast, "numbers"),
        (generator_attributed_error, "sink"),
    ):
        engine = assert_same(scenario, tmp_path)
        assert isinstance(engine.error, process.NodeExecutionError), engine.error
        assert engine.error.node_id == node_id


class Held:
    """An output whose life a test follows."""


def lazy_untaken(trace: list[object], *, read_first: bool) -> Chain:
    """big feeds pick's unread lazy input. read_first: big is also read by `other`, which pick reads, so big runs
    before pick decides; else pick reads a constant and big runs later, for `late`. `after` (and `after_late`)
    record whether big's output is alive once pick (and `late`) ran: None before it ran."""
    b = Builder()
    alive: list[Any] = []

    def big(x):
        held = Held()
        alive.append(weakref.ref(held))
        return held

    def alive_after(name):
        return lambda x: (
            trace.append((name, alive[0]() is not None if alive else None)) or x
        )

    b.node("big", big)
    b.value("big", 0, 1)
    b.node("pick", lambda unread, read: read.value, inputs=2, lazy=(0, 1))
    b.edge("big", 0, "pick", 0)
    b.node("after", alive_after("big alive after pick"))
    b.edge("pick", 0, "after", 0)
    b.node("sink", lambda x: trace.append(("sink", x)), outputs=0, side_effects=True)
    b.edge("after", 0, "sink", 0)
    if read_first:
        b.node("other", lambda held: 5)
        b.edge("big", 0, "other", 0)
        b.edge("other", 0, "pick", 1)
    else:
        b.value("pick", 1, 5)
        b.node("late", lambda held: 5)
        b.edge("big", 0, "late", 0)
        b.node("after_late", alive_after("big alive after late"))
        b.edge("late", 0, "after_late", 0)
        b.node("later_sink", lambda x: None, outputs=0, side_effects=True)
        b.edge("after_late", 0, "later_sink", 0)
    return b.chain


def test_an_unread_lazy_input_frees_its_upstream_after_its_other_reads(tmp_path):
    # A Conditional's untaken branch: its upstream, read by another node too, is freed once pick ran, not at the
    # run's end (before: alive after pick on both executors).
    _, oracle, engine, oracle_trace, engine_trace = run_both(
        lambda trace: lazy_untaken(trace, read_first=True),
        tmp_path,
        send_broadcast_data=False,
    )
    assert oracle.error is None and engine.error is None
    assert (
        oracle_trace == engine_trace == [("big alive after pick", False), ("sink", 5)]
    )


def test_the_engine_counts_an_unread_lazy_input_ahead_of_its_upstreams_run(tmp_path):
    # big may run after pick decided (for `late`): the engine frees it after its last read all the same.
    # process.Executor gives up only the reads of a cached output (its counted cache has no entry before the run).
    trace: list[object] = []
    result = execute(
        "native",
        lazy_untaken(trace, read_first=False),
        storage=tmp_path / "storage",
        send_broadcast_data=False,
    )
    assert result.error is None
    found = dict(entry for entry in trace if isinstance(entry, tuple))
    assert found["sink"] == 5 and found["big alive after late"] is False, trace


def folder_look_ahead(trace: list[object], source: Path, folder: Path) -> Chain:
    """frames (one file per item, in `source`) -> temporal (looks ahead from item 0) -> save, whose directory comes from
    `folder`, a pure node save's gather would run only after temporal decided the window."""
    b = Builder()
    b.node(
        "frames",
        lambda: items(list(range(6)), source_paths=(source / "0.png",)),
        kind="generator",
        inputs=0,
    )
    b.node("folder", lambda: str(folder), inputs=0)

    def temporal(context, frame):
        view = context.iteration
        ahead = view.lookahead(0)
        trace.append(
            ("temporal", view.index, [view.peek(0, k) for k in range(1, ahead + 1)])
        )
        return frame

    b.node("temporal", temporal, context=True)
    save = b.node(
        "save",
        lambda frame, directory: trace.append(("save", frame, str(directory))),
        inputs=2,
        outputs=0,
        side_effects=True,
    )
    object.__setattr__(
        save.data,
        "inputs",
        [save.data.inputs[0], DirectoryInput(must_exist=False).with_id(1)],
    )
    b.edge("frames", 0, "temporal", 0)
    b.edge("temporal", 0, "save", 0)
    b.edge("folder", 0, "save", 1)
    return b.chain


def test_a_folder_node_runs_before_a_look_ahead_decides_at_item_0(
    tmp_path, monkeypatch
):
    # The look-ahead at item 0 decides the window before save's gather would run `folder`: both executors run it
    # first, so the read-ahead check resolves save's folder and the look-ahead is served (before: unresolved, K = 1
    # and a warning on both).
    monkeypatch.setenv(ITEM_WINDOW_VARIABLE, "3")
    source, folder = tmp_path / "in", tmp_path / "out"
    chain, oracle, engine, oracle_trace, trace = run_both(
        lambda trace: folder_look_ahead(trace, source, folder), tmp_path
    )
    assert mismatches("folder_look_ahead", chain, oracle, engine, tmp_path) == []
    assert trace == oracle_trace
    assert oracle.error is None and engine.error is None
    assert [e for e in engine.events if e["event"] == "execution-warning"] == []
    looks = [e for e in trace if isinstance(e, tuple) and e[0] == "temporal"]
    assert looks == [
        ("temporal", index, [index + k if index + k < 6 else None for k in (1, 2)])
        for index in range(6)
    ]
    assert [e for e in trace if isinstance(e, tuple) and e[0] == "save"] == [
        ("save", index, str(folder)) for index in range(6)
    ]


def test_an_engine_fault_outside_a_node_names_the_node_being_scheduled(
    tmp_path, monkeypatch
):
    # An exception escaping the engine's driver outside any node's job (here a hook's malformed answer while the loop
    # measures its first item) ends the run as the error of the node the driver was running, which the app then
    # shows (before: a RuntimeError that named no node).
    monkeypatch.setattr(
        native_executor._Hooks, "output_bytes", lambda self, values: "not a size"
    )
    b = Builder()
    b.node("numbers", lambda: items([1, 2, 3]), kind="generator", inputs=0)
    b.node("sink", lambda x: None, outputs=0, side_effects=True)
    b.edge("numbers", 0, "sink", 0)
    result = execute("native", b.chain, storage=tmp_path / "storage")
    assert isinstance(result.error, process.NodeExecutionError), result.error
    assert result.error.node_id == "numbers"
    assert str(result.error).startswith(
        "The native run engine failed while running numbers: "
    )


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
    # turn in that queue (9-11 s shown for a 1.5 s upscale, in both engines).
    def upscale() -> int:
        with util.waiting():
            time.sleep(0.5)
        return 1

    b = Builder()
    b.node("upscale", upscale, inputs=0, side_effects=True)
    result = execute(kind, b.chain, storage=tmp_path / "storage")
    assert result.error is None, result.error
    times = [
        e["data"]["executionTime"] for e in result.events if e["event"] == "node-finish"
    ]
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


# Context frames: a generator reading a range of its source also reads, for the
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
        assert all(
            0 < count <= most for count, most in zip(counts, (6, 4, 4), strict=True)
        )
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


def gated_chain(
    trace: list[object],
    generator: Callable[[], Generator],
    model: object,
    *,
    fed: str = "outside",
    reader: tuple[int, int] | None = None,
    beside: bool = False,
    applies: Callable[[object], bool] = lambda model: model != "image",
    second: str | None = None,
    collect: bool = False,
    lazy: bool = False,
    source: Callable[[Builder], None] | None = None,
) -> Chain:
    """frames -> gated -> sink, gated's context (3, 2) applying only when `applies` (by default: its model input is
    not "image", as Upscale Image's applies to a temporal model alone) is true for its model. The model comes from
    `fed`: "outside" the loop (load, as Load Model; a model "broken" fails it), a "constant", "none" (the input left
    unconnected), or "inside" it (pick, per item). Then, in this order, toward the sink:
    - `second`: gated feeds gated2, (2, 1), gated the same way by load2's model `second`;
    - `reader`: a temporal node of that context, reader;
    - `beside`: both, which also reads frames -> temporal (3, 2);
    - `collect`: the last node also feeds collect -> report;
    - `lazy`: the sink reads the last node through a lazy input it never takes.
    `source(b)` adds the loop's source "frames" instead of the generator. gated and gated2 record their index,
    context and unread flags and model; reader and temporal their index and context flag; load, load2, pick,
    collect, report and the sink their runs."""
    b = Builder()
    if source is None:
        b.node("frames", generator, kind="generator", inputs=0)
    else:
        source(b)

    def gated(tag: str):
        def run(context, value, model):
            view = context.iteration
            trace.append((tag, view.index, view.context, view.unread, model))
            return value

        return run

    def load(tag: str):
        def run(name):
            trace.append((tag, name))
            if name == "broken":
                raise ValueError("the model file is broken")
            return name

        return run

    node = b.node(
        "gated",
        gated("gated"),
        inputs=2,
        context=True,
        temporal=(3, 2),
        gate=(1, applies),
    )
    b.edge("frames", 0, "gated", 0)
    if fed == "none":
        node.data.inputs[1].make_optional()
    elif fed == "outside":
        b.node("load", load("load"))
        b.value("load", 0, model)
        b.edge("load", 0, "gated", 1)
    elif fed == "constant":
        b.value("gated", 1, model)
    elif fed == "inside":
        b.node("pick", lambda _frame: (trace.append(("pick",)), model)[1])
        b.edge("frames", 0, "pick", 0)
        b.edge("pick", 0, "gated", 1)

    def recorded(tag: str):
        def run(context, value):
            view = context.iteration
            trace.append((tag, view.index, view.context))
            return value

        return run

    last = "gated"
    if second is not None:
        b.node(
            "gated2",
            gated("gated2"),
            inputs=2,
            context=True,
            temporal=(2, 1),
            gate=(1, applies),
        )
        b.node("load2", load("load2"))
        b.value("load2", 0, second)
        b.edge("gated", 0, "gated2", 0)
        b.edge("load2", 0, "gated2", 1)
        last = "gated2"
    if reader is not None:
        b.node("reader", recorded("reader"), context=True, temporal=reader)
        b.edge(last, 0, "reader", 0)
        last = "reader"
    if beside:
        b.node("temporal", recorded("temporal"), context=True, temporal=(3, 2))
        b.node("both", lambda x, _y: x, inputs=2)
        b.edge("frames", 0, "temporal", 0)
        b.edge(last, 0, "both", 0)
        b.edge("temporal", 0, "both", 1)
        last = "both"
    if collect:

        def gather(_):
            trace.append(("collector",))
            collected: list[object] = []

            def iterate(value):
                trace.append(("collect", value))
                collected.append(value)

            return Collector(iterate, lambda: len(collected))

        b.node("collect", gather, kind="collector")
        b.node(
            "report",
            lambda count: trace.append(("report", count)),
            outputs=0,
            side_effects=True,
        )
        b.edge(last, 0, "collect", 0)
        b.edge("collect", 0, "report", 0)
    if lazy:
        b.node(
            "sink",
            lambda _x: trace.append(("sink", "untaken")),
            outputs=0,
            side_effects=True,
            lazy=(0,),
        )
    else:
        b.node(
            "sink", lambda x: trace.append(("sink", x)), outputs=0, side_effects=True
        )
    b.edge(last, 0, "sink", 0)
    return b.chain


def both_engines(
    scenario: Scenario, monkeypatch, tmp_path: Path, label: str, window: str = "4"
) -> tuple[Result, list[object]]:
    """The scenario on both executors with K = `window`: the same events (mismatches) and the same trace. The native
    run's result and trace."""
    oracle, oracle_trace, _ = run_context(
        "python", scenario, tmp_path, monkeypatch, window
    )
    engine, engine_trace, chain = run_context(
        "native", scenario, tmp_path, monkeypatch, window
    )
    assert mismatches(label, chain, oracle, engine, tmp_path) == []
    assert engine_trace == oracle_trace
    return engine, engine_trace


@pytest.mark.parametrize(
    ("model", "fed", "asked"),
    [
        ("image", "outside", []),
        ("temporal", "outside", [(3, 2)]),
        ("image", "constant", []),
        ("temporal", "constant", [(3, 2)]),
        ("image", "inside", [(3, 2)]),
        (None, "none", [(3, 2)]),
    ],
    ids=["off", "on", "constant-off", "constant-on", "fed-inside-the-loop", "unset"],
)
@pytest.mark.parametrize("kind", CONTEXT_EXECUTORS)
def test_a_gate_decides_whether_the_context_applies(
    kind, model, fed, asked, monkeypatch, tmp_path
):
    # Items 8..11 of 20. The gate's value known before the loop (load, run once before the first item, or a
    # constant) decides: "image" asks no context items, so gated runs for the range alone; a temporal model asks
    # (3, 2). A model fed from inside the loop is not known then: the context applies, whatever the model. An
    # unconnected gate's value (None) is the predicate's to judge; this one applies.
    frames = Frames(20)
    result, trace, _ = run_context(
        kind,
        lambda t: gated_chain(t, lambda: frames.range(8, 4), model, fed=fed),
        tmp_path,
        monkeypatch,
        "4",
    )
    assert result.error is None, result.error
    first, last = (5, 14) if asked else (8, 12)
    assert frames.asked == asked and frames.pulled == list(range(first, last))
    assert [r[:2] for r in records(trace, "gated")] == [
        (i, i < 8) for i in range(first, 12)
    ]
    assert records(trace, "load") == ([(model,)] if fed == "outside" else [])
    assert records(trace, "sink") == [(i * 10 + 1,) for i in range(8, 12)]


@pytest.mark.parametrize("kind", CONTEXT_EXECUTORS)
def test_a_gated_off_node_adds_nothing_to_a_reader_s_context(
    kind, monkeypatch, tmp_path
):
    # gated (3, 2), off, feeds reader (2, 1): the loop asks reader's (2, 1), not (5, 3). gated still runs for the
    # two context items, as reader needs its output there (not unread).
    frames = Frames(20)
    result, trace, _ = run_context(
        kind,
        lambda t: gated_chain(t, lambda: frames.range(8, 4), "image", reader=(2, 1)),
        tmp_path,
        monkeypatch,
        "4",
    )
    assert result.error is None, result.error
    assert frames.asked == [(2, 1)]
    assert records(trace, "gated") == [(i, i < 8, False, "image") for i in range(6, 12)]
    assert records(trace, "reader") == [(i, i < 8) for i in range(6, 12)]


@pytest.mark.parametrize("kind", CONTEXT_EXECUTORS)
def test_a_gated_off_node_beside_a_temporal_one_skips_its_context_items(
    kind, monkeypatch, tmp_path
):
    # gated (off) and temporal (3, 2) both feed the sink: the loop asks temporal's (3, 2) and only temporal runs for
    # the context items; gated, whose context does not apply, runs for the range alone.
    frames = Frames(20)
    result, trace, _ = run_context(
        kind,
        lambda t: gated_chain(t, lambda: frames.range(8, 4), "image", beside=True),
        tmp_path,
        monkeypatch,
        "4",
    )
    assert result.error is None, result.error
    assert frames.asked == [(3, 2)]
    assert [r[:2] for r in records(trace, "gated")] == [
        (i, False) for i in range(8, 12)
    ]
    assert records(trace, "temporal") == [(i, i < 8) for i in range(5, 12)]


@pytest.mark.parametrize(
    ("first", "second", "asked"),
    [
        ("image", "image", []),
        ("image", "temporal", [(2, 1)]),
        ("temporal", "image", [(3, 2)]),
        ("temporal", "temporal", [(5, 3)]),
    ],
    ids=["off-off", "off-on", "on-off", "on-on"],
)
@pytest.mark.parametrize("kind", CONTEXT_EXECUTORS)
def test_two_gated_nodes_in_a_chain(kind, first, second, asked, monkeypatch, tmp_path):
    # gated (3, 2) -> gated2 (2, 1), each with its own model: only the contexts that apply add along the path.
    frames = Frames(20)
    result, trace, _ = run_context(
        kind,
        lambda t: gated_chain(t, lambda: frames.range(8, 4), first, second=second),
        tmp_path,
        monkeypatch,
        "4",
    )
    assert result.error is None, result.error
    assert frames.asked == asked
    before = asked[0][0] if asked else 0
    assert [r[:2] for r in records(trace, "gated2")] == [
        (i, i < 8) for i in range(8 - (before if second == "temporal" else 0), 12)
    ]
    assert [r[:2] for r in records(trace, "gated")] == [
        (i, i < 8) for i in range(8 - before, 12)
    ]


@pytest.mark.parametrize(
    "options",
    [
        {"model": "image", "reader": (2, 1)},
        {"model": "temporal", "reader": (2, 1)},
        {"model": "image", "beside": True},
        {"model": "temporal", "collect": True},
        {"model": "image", "collect": True, "beside": True},
        {"model": "temporal", "second": "image", "collect": True},
    ],
    ids=[
        "off-read",
        "on-read",
        "off-beside",
        "on-collect",
        "off-beside-collect",
        "two",
    ],
)
def test_gated_context_matches_across_executors(options, monkeypatch, tmp_path):
    # Both engines resolve the gate at the same point (before the collectors are made) and run the same nodes for
    # the same items: the same events, reads and records.
    pulled: list[list[int]] = []

    def scenario(trace: list[object]) -> Chain:
        frames = Frames(20)
        pulled.append(frames.pulled)
        return gated_chain(trace, lambda: frames.range(8, 4), **options)

    result, trace = both_engines(scenario, monkeypatch, tmp_path, "gated context")
    assert result.error is None, result.error
    assert pulled[0] == pulled[1]
    if options.get("collect"):
        assert records(trace, "report") == [(4,)]
        assert trace.index(("collector",)) > trace.index(("load", options["model"]))


@pytest.mark.parametrize("kind", CONTEXT_EXECUTORS)
def test_a_gate_reached_only_through_an_untaken_lazy_input_is_not_known(
    kind, monkeypatch, tmp_path
):
    # The sink never takes gated's output: gated's gate is not resolved before the loop, so its context applies
    # (the safe side) and load runs when gated first runs, for the first context item. Not in the cross-engine
    # comparison: any temporal node behind an untaken lazy input, gated or not, runs for context items outside the
    # reader's executions, which its lazy-input order check reports.
    frames = Frames(20)
    result, trace, _ = run_context(
        kind,
        lambda t: gated_chain(t, lambda: frames.range(8, 4), "image", lazy=True),
        tmp_path,
        monkeypatch,
        "4",
    )
    assert result.error is None, result.error
    assert frames.asked == [(3, 2)]
    assert trace[:2] == [("load", "image"), ("gated", 5, True, True, "image")]


def test_a_failing_gate_source_fails_before_the_first_item(monkeypatch, tmp_path):
    # load fails while the gate is resolved: the run fails with load's error in both engines, and no item is read.
    pulled: list[list[int]] = []

    def scenario(trace: list[object]) -> Chain:
        frames = Frames(20)
        pulled.append(frames.pulled)
        return gated_chain(trace, lambda: frames.range(8, 4), "broken")

    result, trace = both_engines(scenario, monkeypatch, tmp_path, "failing source")
    error = result.error
    assert isinstance(error, process.NodeExecutionError), error
    assert error.node_id == "load" and "the model file is broken" in str(error)
    assert pulled == [[], []]
    assert trace == [("load", "broken")]


def range_source(stop: int) -> Callable[[Builder], None]:
    """The native Range generator (0 to stop) as the loop's source "frames"; the Python executor runs a body that
    gives the same items."""

    def add(b: Builder) -> None:
        node = b.node(
            "frames",
            lambda start, _first, end, _last: Generator(
                lambda: iter(range(start, end)), end - start, source_paths=()
            ),
            kind="generator",
            inputs=4,
        )
        node.data = replace(node.data, schema_id="chainner:utility:range")
        node.schema_id = node.data.schema_id
        for index, value in enumerate((0, True, stop, False)):
            b.value("frames", index, value)

    return add


def zipped_source(length: int) -> Callable[[Builder], None]:
    """Two generators of `length` items, zipped into the loop's source "frames"."""

    def add(b: Builder) -> None:
        for name in ("left", "right"):
            b.node(
                name,
                lambda: Generator(lambda: iter(range(length)), length, source_paths=()),
                kind="generator",
                inputs=0,
            )
        b.node("frames", lambda x, y: x + y, inputs=2)
        b.edge("left", 0, "frames", 0)
        b.edge("right", 0, "frames", 1)

    return add


@pytest.mark.parametrize(
    "source",
    [
        lambda: Frames(0).whole(),
        lambda: Frames(20).range(8, 0),
        lambda: Frames(20).whole(),
    ],
    ids=["empty", "empty-range", "whole"],
)
def test_a_loop_that_reads_no_context_runs_its_gate_s_source_as_before(
    source, monkeypatch, tmp_path
):
    # A generator without context items (an empty one, one that reads its source whole): no gate is resolved
    # before the loop, so a broken load never runs for an empty loop and fails at the first item otherwise.
    result, trace = both_engines(
        lambda t: gated_chain(t, source, "broken"), monkeypatch, tmp_path, "no gate"
    )
    if source().expected_length == 0:
        assert result.error is None, result.error
        assert trace == []
    else:
        assert isinstance(result.error, process.NodeExecutionError), result.error
        assert result.error.node_id == "load"


@pytest.mark.parametrize(
    "source",
    [range_source(0), range_source(4), zipped_source(0), zipped_source(3)],
    ids=["range-empty", "range", "zipped-empty", "zipped"],
)
def test_a_native_or_zipped_loop_resolves_no_gate(source, monkeypatch, tmp_path):
    # Native Range, and a loop of two generators, read no context items: gated's context never applies to them and
    # load runs as the first item needs it (never, for an empty loop).
    result, trace = both_engines(
        lambda t: gated_chain(t, lambda: Frames(0).whole(), "temporal", source=source),
        monkeypatch,
        tmp_path,
        "native or zipped",
    )
    assert result.error is None, result.error
    seen = records(trace, "gated")
    assert all(not r[1] for r in seen)
    assert records(trace, "load") == ([("temporal",)] if seen else [])


@pytest.mark.parametrize("model", ["image", "temporal"])
def test_run_this_frame_with_a_gate(model, monkeypatch, tmp_path):
    # "Run this frame" on item 8: a temporal model asks (3, 2) and gives gated items 5..7 before it, an image
    # model asks none.
    asked: list[list[tuple[int, int]]] = []

    def scenario(trace: list[object]) -> Chain:
        frames = Frames(20)
        asked.append(frames.asked)
        return gated_chain(trace, lambda: frames.range(8, 1, chosen=8), model)

    result, trace = both_engines(scenario, monkeypatch, tmp_path, "run this frame")
    assert result.error is None, result.error
    first = 5 if model == "temporal" else 8
    assert [r[:2] for r in records(trace, "gated")] == [
        (i, i < 8) for i in range(first, 9)
    ]
    assert asked == [[(3, 2)] if model == "temporal" else []] * 2
    assert records(trace, "sink") == [(81,)]


def test_a_gate_s_failing_predicate_fails_its_node(monkeypatch, tmp_path):
    # The predicate raises for the loaded model: before the first item, both engines fail the run with gated's
    # error, its cause the predicate's, and no item runs.
    def applies(model: object) -> bool:
        raise ValueError(f"cannot judge {model}")

    def run(kind: str) -> tuple[Result, list[object], Chain]:
        frames = Frames(20)
        result, trace, chain = run_context(
            kind,
            lambda t: gated_chain(
                t, lambda: frames.range(8, 4), "odd", applies=applies
            ),
            tmp_path,
            monkeypatch,
            "4",
        )
        assert frames.pulled == []
        return result, trace, chain

    oracle, oracle_trace, _ = run("python")
    engine, engine_trace, chain = run("native")
    for result in (oracle, engine):
        error = result.error
        assert isinstance(error, process.NodeExecutionError), error
        assert error.node_id == "gated" and str(error) == "cannot judge odd"
        assert isinstance(error.__cause__, ValueError)
    assert mismatches("failing gate", chain, oracle, engine, tmp_path) == []
    assert engine_trace == oracle_trace == [("load", "odd")]


def device_chain(
    trace: list[object],
    generator: Callable[[], Generator],
    model: str,
    *,
    fed: str = "outside",
    applies: Callable[[object], bool] = lambda model: model != "image",
) -> Chain:
    """frames -> upscale -> sink, upscale a temporal node (3, 2) on a locked-device schema (TensorRT Upscale Image's,
    so pure_cpu_node holds), its context applying when `applies` (by default: its model is not "image") is true for
    its model, which comes from `fed`: "outside" the loop (load), a "constant", or "inside" it (pick, per item). With
    a model other than "image" it records per item its index, context flag and its view's peek one item ahead;
    with "image", nothing (it may run ahead on a worker, in no fixed order)."""
    b = Builder()
    b.node("frames", generator, kind="generator", inputs=0)

    def upscale(context, value, model):
        view = context.iteration
        if model != "image":
            trace.append(
                (
                    "upscale",
                    None if view is None else view.index,
                    None if view is None else view.context,
                    None if view is None else view.peek(0, 1),
                )
            )
        return value

    node = b.node(
        "upscale",
        upscale,
        inputs=2,
        context=True,
        temporal=(3, 2),
        gate=(1, applies),
    )
    node.data = replace(node.data, schema_id="chainner:tensorrt:upscale_image")
    node.schema_id = node.data.schema_id
    assert execution_scheduler.pure_cpu_node(node.data)
    b.edge("frames", 0, "upscale", 0)
    if fed == "outside":
        b.node("load", lambda name: (trace.append(("load", name)), name)[1])
        b.value("load", 0, model)
        b.edge("load", 0, "upscale", 1)
    elif fed == "constant":
        b.value("upscale", 1, model)
    else:
        b.node("pick", lambda _frame: model)
        b.edge("frames", 0, "pick", 0)
        b.edge("pick", 0, "upscale", 1)
    b.node("sink", lambda x: trace.append(("sink", x)), outputs=0, side_effects=True)
    b.edge("upscale", 0, "sink", 0)
    return b.chain


def spy_pure_in_loop(monkeypatch) -> list[bool]:
    """The decisions both engines make through process.pure_in_loop, in order."""
    decided: list[bool] = []
    original = process.pure_in_loop

    def spy(*args):
        pure = original(*args)
        decided.append(pure)
        return pure

    monkeypatch.setattr(process, "pure_in_loop", spy)
    monkeypatch.setattr(native_executor, "pure_in_loop", spy)
    return decided


@pytest.mark.parametrize("window", ["4", "1"])
@pytest.mark.parametrize(
    ("model", "fed"),
    [
        ("temporal", "outside"),
        ("image", "outside"),
        ("temporal", "constant"),
        ("image", "constant"),
        ("image", "inside"),
    ],
    ids=["on", "off", "constant-on", "constant-off", "fed-inside-the-loop"],
)
@pytest.mark.parametrize("ranged", [True, False], ids=["context", "whole"])
def test_a_gated_node_on_a_pure_schema_gets_its_view_while_its_context_applies(
    model, fed, ranged, window, monkeypatch, tmp_path
):
    # While its context applies (a temporal model, or a model fed from inside the loop, never judged) the node runs
    # on the driver with its view on every item, context items included, and peeks the next frame; with an image
    # model it stays pure for the run (a run-ahead candidate, as without a gate). Both engines decide it once per
    # run, at the node's first run on the commit path (any K, the window or none), and send the same events.
    decided = spy_pure_in_loop(monkeypatch)
    viewed = model != "image" or fed == "inside"

    def scenario(trace: list[object]) -> Chain:
        frames = Frames(20)
        return device_chain(
            trace,
            (lambda: frames.range(8, 4)) if ranged else frames.whole,
            "temporal" if fed == "inside" else model,
            fed=fed,
            applies=(lambda _model: False)
            if fed == "inside"
            else lambda m: m != "image",
        )

    result, trace = both_engines(
        scenario, monkeypatch, tmp_path, "pure schema", window=window
    )
    assert result.error is None, result.error
    assert decided == [not viewed] * 2
    if viewed:
        before = 3 if ranged else 0
        items = range(8 - before, 12) if ranged else range(20)
        # K = 1 builds no window: a look-ahead sees nothing.
        assert records(trace, "upscale") == [
            (
                i,
                ranged and i < 8,
                (i + 1) * 10 + 1 if i < 19 and window != "1" else None,
            )
            for i in items
        ]
    else:
        assert records(trace, "upscale") == []
    assert records(trace, "sink") == [
        (i * 10 + 1,) for i in (range(8, 12) if ranged else range(20))
    ]


@pytest.mark.parametrize("window", ["4", "1"])
@pytest.mark.parametrize("fail_fast", [True, False], ids=["fail-fast", "deferred"])
def test_a_raising_gate_predicate_fails_a_pure_schema_node_once(
    fail_fast, window, monkeypatch, tmp_path
):
    # The source reads no context items, so the gate is first judged at the node's first run: the predicate's error
    # fails that run once, as the node's error with the predicate's as its cause. With a fail-fast source the run
    # ends there; otherwise item 0 is dropped and the node runs every later item with its view.
    decided = spy_pure_in_loop(monkeypatch)

    def applies(model: object) -> bool:
        raise ValueError(f"cannot judge {model}")

    def scenario(trace: list[object]) -> Chain:
        frames = Frames(5)
        return device_chain(
            trace,
            lambda: frames.whole().with_fail_fast(fail_fast),
            "odd",
            applies=applies,
        )

    result, trace = both_engines(
        scenario, monkeypatch, tmp_path, "raising gate", window=window
    )
    error = result.error
    if fail_fast:
        assert isinstance(error, process.NodeExecutionError), error
        assert error.node_id == "upscale" and str(error) == "cannot judge odd"
        assert isinstance(error.__cause__, ValueError)
    else:
        # The loop's deferred errors, item 0's alone.
        assert str(error) == "Errors occurred during iteration:\n- cannot judge odd"
    assert decided == []
    later = [] if fail_fast else list(range(1, 5))
    assert records(trace, "upscale") == [
        (i, False, (i + 1) * 10 + 1 if i < 4 and window != "1" else None) for i in later
    ]
    assert records(trace, "sink") == [(i * 10 + 1,) for i in later]


def test_register_checks_a_gate():
    # temporal_if gates a temporal_context and names one of the node's inputs; the node data keeps it by input id.
    package = api.Package(
        where="test", id="test", name="Test", description="", icon="", color="#000000"
    )
    group = package.add_category("Test", "", "", "#000000").add_node_group("Test")

    def register(**options):
        return group.register(
            "chainner:utility:test_gate",
            name="Gate",
            description="",
            inputs=[BaseInput("any", "Frame").with_id(0)],
            outputs=[],
            generated=True,
            **options,
        )(lambda frame: frame)

    with pytest.raises(AssertionError, match="temporal_if gates a temporal_context"):
        register(temporal_if=(0, bool))
    with pytest.raises(AssertionError, match="names input 1, which it has not"):
        register(temporal_context=(1, 1), temporal_if=(1, bool))
    assert group.nodes == []
    register(temporal_context=(1, 1), temporal_if=(0, bool))
    assert group.nodes[0].temporal_if == (InputId(0), bool)


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


def upscale_node(b: Builder, node_id: str, run: Callable[..., object]) -> None:
    """A GPU upscale that runs ahead (execution_scheduler.LOCKED_DEVICE_NODES)."""
    node = b.node(node_id, run, context=True)
    object.__setattr__(node.data, "schema_id", "chainner:tensorrt:upscale_image")
    node.schema_id = node.data.schema_id


@pytest.mark.parametrize("kind", ["python", "native"])
def test_an_upscale_a_skipping_save_never_reads_is_not_computed_ahead(
    monkeypatch, tmp_path, kind
):
    # Save Image's Skip existing files: the items that exist never read the upscale
    # behind its lazy input, so the window does not compute it for them.
    monkeypatch.setenv(ITEM_WINDOW_VARIABLE, "8")
    upscaled: list[int] = []
    saved: list[object] = []
    b = Builder()
    b.node("numbers", lambda: items(list(range(24))), kind="generator", inputs=0)
    upscale_node(b, "up", lambda _context, x: upscaled.append(x) or x * 10)

    def save(image: Lazy) -> None:
        if not saved:
            saved.append(image.value)

    b.node("save", save, outputs=0, side_effects=True, lazy=(0,))
    b.edge("numbers", 0, "up", 0)
    b.edge("up", 0, "save", 0)
    result = execute(kind, b.chain, storage=tmp_path / "s", pool_size=8)
    assert result.error is None, result.error
    assert (upscaled, saved) == ([0], [0])


@pytest.mark.parametrize("kind", ["python", "native"])
def test_a_conditional_s_other_branch_computes_no_upscale_ahead(
    monkeypatch, tmp_path, kind
):
    # The untaken branch: neither its upscale nor a pure node after it runs ahead; the
    # taken branch's upscale runs for every item, when the Conditional reads it.
    monkeypatch.setenv(ITEM_WINDOW_VARIABLE, "8")
    taken: list[int] = []
    untaken: list[int] = []
    saved: list[object] = []
    b = Builder()
    b.node("numbers", lambda: items(list(range(12))), kind="generator", inputs=0)
    upscale_node(b, "up_taken", lambda _context, x: taken.append(x) or x)
    upscale_node(b, "up_other", lambda _context, x: untaken.append(x) or x)
    b.node("brighten", lambda x: x + 1)
    b.node("pick", lambda other, read: read.value, inputs=2, lazy=(0, 1))
    b.node("save", saved.append, outputs=0, side_effects=True)
    b.edge("numbers", 0, "up_taken", 0)
    b.edge("numbers", 0, "up_other", 0)
    b.edge("up_other", 0, "brighten", 0)
    b.edge("brighten", 0, "pick", 0)
    b.edge("up_taken", 0, "pick", 1)
    b.edge("pick", 0, "save", 0)
    result = execute(kind, b.chain, storage=tmp_path / "s", pool_size=8)
    assert result.error is None, result.error
    assert untaken == []
    assert sorted(taken) == list(range(12)) and saved == list(range(12))


def save_like(
    monkeypatch, b: Builder, node_id: str, run: Callable[..., object]
) -> None:
    """A sink with Save Image's lazy image (input 0, always read unless its boolean input
    1000, Skip existing files, is on or connected: ALWAYS_READ_LAZY_INPUTS)."""
    node = b.node(node_id, run, inputs=2, outputs=0, side_effects=True, lazy=(0,))
    node.data.inputs[1].id = InputId(1000)
    monkeypatch.setitem(ALWAYS_READ_LAZY_INPUTS, node.data.schema_id, {0: 1000})


def ahead_at_item_2(
    monkeypatch, tmp_path, kind: str, build: Callable[[Builder, Callable], None]
) -> list[int]:
    """The upscales done when item 2's sink ends (12 items, K = 8, 30 ms upscales, the
    sink holding item 2 for 1 s): 10 when the window computes them ahead, 3 when not."""
    monkeypatch.setenv(ITEM_WINDOW_VARIABLE, "8")
    upscaled: list[int] = []
    counts: list[int] = []

    def upscale(_context, x):
        time.sleep(0.03)
        upscaled.append(x)
        return x

    def reached(x: object) -> None:
        if x == 2:
            time.sleep(1.0)
            counts.append(len(upscaled))

    b = Builder()
    b.node("numbers", lambda: items(list(range(12))), kind="generator", inputs=0)
    upscale_node(b, "up", upscale)
    b.edge("numbers", 0, "up", 0)
    build(b, reached)
    result = execute(kind, b.chain, storage=tmp_path / "s", pool_size=8)
    assert result.error is None, result.error
    return counts


@pytest.mark.parametrize("kind", ["python", "native"])
@pytest.mark.parametrize("sink", ["eager", "lazy, always read"])
def test_a_sink_that_always_reads_its_lazy_image_keeps_the_upscales_ahead(
    monkeypatch, tmp_path, kind, sink
):
    # Save Image with Skip existing files off: Load Images -> Upscale Image -> Save
    # Image computes the upscales ahead, as for a sink without a lazy input.
    def build(b: Builder, reached: Callable) -> None:
        if sink == "eager":
            b.node("save", reached, outputs=0, side_effects=True)
        else:
            save_like(monkeypatch, b, "save", lambda image, _skip: reached(image.value))
            b.value("save", 1000, False)
        b.edge("up", 0, "save", 0)

    assert ahead_at_item_2(monkeypatch, tmp_path, kind, build) == [10]


@pytest.mark.parametrize("kind", ["python", "native"])
@pytest.mark.parametrize("skip", ["on", "connected"])
def test_skip_existing_files_on_or_connected_computes_no_upscale_ahead(
    monkeypatch, tmp_path, kind, skip
):
    monkeypatch.setenv(ITEM_WINDOW_VARIABLE, "8")
    upscaled: list[int] = []
    saved: list[object] = []
    b = Builder()
    b.node("numbers", lambda: items(list(range(24))), kind="generator", inputs=0)
    upscale_node(b, "up", lambda _context, x: upscaled.append(x) or x)

    def save(image: Lazy, _skip: object) -> None:
        # Every file but the first exists
        if not saved:
            saved.append(image.value)

    save_like(monkeypatch, b, "save", save)
    b.edge("numbers", 0, "up", 0)
    b.edge("up", 0, "save", 0)
    if skip == "on":
        b.value("save", 1000, True)
    else:
        b.edge("numbers", 0, "save", 1000)
    result = execute(kind, b.chain, storage=tmp_path / "s", pool_size=8)
    assert result.error is None, result.error
    assert (upscaled, saved) == ([0], [0])


@pytest.mark.parametrize("kind", ["python", "native"])
def test_an_upscale_one_sink_reads_eagerly_stays_ahead_beside_a_lazy_one(
    monkeypatch, tmp_path, kind
):
    # The lazy sink, walked first, may leave it unread; the eager sink still needs it.
    def build(b: Builder, reached: Callable) -> None:
        b.node("unread", lambda _image: None, outputs=0, side_effects=True, lazy=(0,))
        b.node("save", reached, outputs=0, side_effects=True)
        b.edge("up", 0, "unread", 0)
        b.edge("up", 0, "save", 0)

    assert ahead_at_item_2(monkeypatch, tmp_path, kind, build) == [10]


@pytest.mark.parametrize("kind", ["python", "native"])
def test_an_upscale_behind_a_driver_node_s_unread_lazy_input_is_not_computed_ahead(
    monkeypatch, tmp_path, kind
):
    # A node on the driver (a context) with a lazy input it may leave unread, before an
    # eager sink: the upscale runs only when that node reads it, item by item.
    def build(b: Builder, reached: Callable) -> None:
        b.node("gate", lambda _context, image: image.value, context=True, lazy=(0,))
        b.node("save", reached, outputs=0, side_effects=True)
        b.edge("up", 0, "gate", 0)
        b.edge("gate", 0, "save", 0)

    assert ahead_at_item_2(monkeypatch, tmp_path, kind, build) == [3]


def test_a_device_node_computed_ahead_runs_its_cleanups(monkeypatch, tmp_path):
    # A locked-device node (TensorRT's Upscale Image) is pure, so the item window computes it ahead of the
    # committing item; the cleanups it registers there run as on the commit path: a node's after its job, a chain's
    # at the run's end.
    device = "chainner:utility:test_device"
    for module in (execution_scheduler, process, native_executor):
        monkeypatch.setattr(
            module,
            "LOCKED_DEVICE_NODES",
            execution_scheduler.LOCKED_DEVICE_NODES | {device},
        )

    def scenario(trace: list[object]) -> Chain:
        b = Builder()
        b.node(
            "frames",
            lambda: items(list(range(12))),
            kind="generator",
            inputs=0,
        )

        def cleanup():
            trace.append("chain cleanup")

        def upscale(context, x):
            context.add_cleanup(cleanup)
            context.add_cleanup(lambda: trace.append(("node cleanup", x)), after="node")
            return x * 100

        b.node("device", upscale, context=True)
        b.node(
            "sink", lambda x: trace.append(("sink", x)), outputs=0, side_effects=True
        )
        b.edge("frames", 0, "device", 0)
        b.edge("device", 0, "sink", 0)
        return b.chain

    chain = scenario([])
    assert execution_scheduler.pure_cpu_node(chain.nodes[NodeId("device")].data)
    chain, oracle, engine, oracle_trace, engine_trace = run_both(scenario, tmp_path)
    assert mismatches("device", chain, oracle, engine, tmp_path) == []
    assert engine.error is None
    # The engine runs the jobs ahead in parallel: their node cleanups come in any order.
    assert Counter(map(str, engine_trace)) == Counter(map(str, oracle_trace))
    assert sum(str(entry).startswith("('node cleanup") for entry in oracle_trace) == 12


def clip_loop_chain(trace: list[object], name_from_first: bool) -> Chain:
    """Load Video into Frames To Clip, a loop over the clip (Clip Frames) into a sink: the sink's
    name from the first loop's per-item Index (through Text Append), or, as must still run, from the second loop."""
    b = Builder()
    b.node("frames", lambda: items([1, 2, 3]), kind="generator", inputs=0)

    def gather(_):
        collected: list[Any] = []
        return Collector(collected.append, lambda: list(collected))

    b.node("clip", gather, kind="collector")
    b.edge("frames", 0, "clip", 0)
    b.node("clip_frames", lambda clip: items([x * 10 for x in clip]), kind="generator")
    b.edge("clip", 0, "clip_frames", 0)
    b.node("upscale", lambda x: x + 1)
    b.edge("clip_frames", 0, "upscale", 0)
    b.node("append", lambda index: f"frame {index}")
    b.edge("frames" if name_from_first else "clip_frames", 0, "append", 0)
    b.node(
        "save",
        lambda image, name: trace.append(("save", image, name)),
        inputs=2,
        outputs=0,
        side_effects=True,
    )
    b.edge("upscale", 0, "save", 0)
    b.edge("append", 0, "save", 1)
    return b.chain


def test_a_loop_reading_another_loop_s_items_and_its_collector_is_refused_before_it_runs(
    tmp_path,
):
    _chain, oracle, engine, oracle_trace, engine_trace = run_both(
        lambda trace: clip_loop_chain(trace, True), tmp_path
    )
    for result in (oracle, engine):
        assert isinstance(result.error, process.NodeExecutionError), result.error
        assert result.error.node_id == "save"
        assert str(result.error) == (
            "Input1 of save comes from the items of frames, one value per item, but save"
            " also depends on clip, which has its result only after the loop of frames"
            " has ended. Take Input1 from the outputs of clip_frames instead, or connect a"
            " value that does not change per item."
        )
    assert oracle_trace == engine_trace == []


def zipped_loops(trace: list[object]) -> Chain:
    b = Builder()
    b.node("a", lambda: items([1, 2]), kind="generator", inputs=0)
    b.node("b", lambda: items([3, 4]), kind="generator", inputs=0)
    b.node(
        "sink",
        lambda x, y: trace.append((x, y)),
        inputs=2,
        outputs=0,
        side_effects=True,
    )
    b.edge("a", 0, "sink", 0)
    b.edge("b", 0, "sink", 1)
    return b.chain


def second_loop_reads_a_static_output(trace: list[object]) -> Chain:
    b = Builder()
    b.node(
        "frames",
        lambda: (items([1, 2, 3]), "video"),
        kind="generator",
        inputs=0,
        outputs=2,
    )

    def gather(_):
        collected: list[Any] = []
        return Collector(collected.append, lambda: list(collected))

    b.node("clip", gather, kind="collector")
    b.edge("frames", 0, "clip", 0)
    b.node("clip_frames", items, kind="generator")
    b.edge("clip", 0, "clip_frames", 0)
    b.node(
        "save",
        lambda x, name: trace.append((x, name)),
        inputs=2,
        outputs=0,
        side_effects=True,
    )
    b.edge("clip_frames", 0, "save", 0)
    b.edge("frames", 1, "save", 1)
    return b.chain


@pytest.mark.parametrize(
    "scenario",
    [
        lambda trace: clip_loop_chain(trace, False),
        zipped_loops,
        lambda trace: loop_chain(trace, [1, 2, 3]),
        second_loop_reads_a_static_output,
    ],
    ids=["second-loop-name", "zipped", "collector-after-loop", "static-output"],
)
def test_loops_that_read_only_what_they_can_still_run(scenario, tmp_path):
    engine = assert_same(scenario, tmp_path)
    assert engine.error is None


def refusal(chain: Chain, tmp_path) -> list[tuple[object, str]]:
    """Each engine's (node, message) for a chain refused before it runs; nothing ran."""
    _, oracle, engine, oracle_trace, engine_trace = run_both(lambda _: chain, tmp_path)
    assert oracle_trace == engine_trace == []
    assert all("comes from the items of" in str(r.error) for r in (oracle, engine))
    return [
        (getattr(result.error, "node_id", None), str(result.error))
        for result in (oracle, engine)
    ]


def gather(_: object, *_read: object) -> Collector:
    collected: list[Any] = []
    return Collector(collected.append, lambda: list(collected))


def test_the_refusal_names_the_input_the_item_arrives_on(tmp_path):
    # Save Image's Directory from Load Video's Video Directory (one value per run) and its
    # name from Load Video's Index (one per item): the name is the input to change.
    b = Builder()
    b.node(
        "frames",
        lambda: (items([1, 2]), "folder"),
        kind="generator",
        inputs=0,
        outputs=2,
    )
    b.node("clip", gather, kind="collector")
    b.edge("frames", 0, "clip", 0)
    b.node("clip_frames", items, kind="generator")
    b.edge("clip", 0, "clip_frames", 0)
    b.node("append", lambda index: f"frame {index}")
    b.edge("frames", 0, "append", 0)
    b.node("save", lambda *_: None, inputs=3, outputs=0, side_effects=True)
    b.edge("clip_frames", 0, "save", 0)
    b.edge("frames", 1, "save", 1)
    b.edge("append", 0, "save", 2)
    expected = (
        "save",
        (
            "Input2 of save comes from the items of frames, one value per item, but save"
            " also depends on clip, which has its result only after the loop of frames"
            " has ended. Take Input2 from the outputs of clip_frames instead, or connect"
            " a value that does not change per item."
        ),
    )
    assert refusal(b.chain, tmp_path) == [expected, expected]


def test_the_refusal_names_the_node_where_the_item_meets_the_collector(tmp_path):
    b = Builder()
    b.node("numbers", lambda: items([1, 2]), kind="generator", inputs=0)
    b.node("total", gather, kind="collector")
    b.edge("numbers", 0, "total", 0)
    b.node("mix", lambda x, total: x, inputs=2)
    b.edge("numbers", 0, "mix", 0)
    b.edge("total", 0, "mix", 1)
    b.node("scale", lambda x: x * 2)
    b.edge("mix", 0, "scale", 0)
    b.node("sink", lambda _: None, outputs=0, side_effects=True)
    b.edge("scale", 0, "sink", 0)
    assert [node for node, _ in refusal(b.chain, tmp_path)] == ["mix", "mix"]


def test_a_collector_reading_another_collector_of_its_loop_is_refused(tmp_path):
    b = Builder()
    b.node("numbers", lambda: items([1, 2]), kind="generator", inputs=0)
    b.node("total", gather, kind="collector")
    b.edge("numbers", 0, "total", 0)
    b.node("second", gather, kind="collector", inputs=2)
    b.edge("numbers", 0, "second", 0)
    b.edge("total", 0, "second", 1)
    b.node("report", lambda _: None, outputs=0, side_effects=True)
    b.edge("second", 0, "report", 0)
    assert [node for node, _ in refusal(b.chain, tmp_path)] == ["second", "second"]


def test_a_node_reading_its_zipped_partner_s_collector_is_refused(tmp_path):
    trace: list[object] = []
    b = Builder()
    b.node("a", lambda: items([1, 2]), kind="generator", inputs=0)
    b.node("b", lambda: items([3, 4]), kind="generator", inputs=0)
    b.node(
        "pair",
        lambda x, y: trace.append(("pair", x, y)),
        inputs=2,
        outputs=0,
        side_effects=True,
    )
    b.edge("a", 0, "pair", 0)
    b.edge("b", 0, "pair", 1)
    b.node("total", gather, kind="collector")
    b.edge("a", 0, "total", 0)
    b.node("mix", lambda x, total: x, inputs=2)
    b.edge("b", 0, "mix", 0)
    b.edge("total", 0, "mix", 1)
    b.node("late", lambda _: None, outputs=0, side_effects=True)
    b.edge("mix", 0, "late", 0)
    assert [node for node, _ in refusal(b.chain, tmp_path)] == ["mix", "mix"]
    assert trace == []
