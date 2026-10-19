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

import pytest

import api
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
from chain.chain import Chain, CollectorNode, Edge, EdgeSource, EdgeTarget, FunctionNode, GeneratorNode
from engine_differential import Result, execute, mismatches
from events import EventConsumer
from native_executor import NativeExecutor
from process import ExecutionId, Executor, RegularOutput
from verify_runtime import canonical


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
    ):
        node_class = {"generator": GeneratorNode, "collector": CollectorNode, "regularNode": FunctionNode}[kind]
        data = NodeData(
            schema_id=f"chainner:utility:test_{node_id}",
            description="",
            see_also=[],
            name=node_id,
            icon="",
            kind=kind,
            inputs=[
                (BaseInput("any", f"Input{i}").with_id(i).make_lazy() if i in lazy else BaseInput("any", f"Input{i}").with_id(i))
                for i in range(inputs)
            ],
            outputs=[BaseOutput("any", f"Output{i}").with_id(i) for i in range(outputs)],
            group_layout=[],
            iterable_inputs=[IteratorInputInfo(0)] if kind == "collector" else [],
            iterable_outputs=[IteratorOutputInfo(list(iterated_outputs))] if kind == "generator" else [],
            key_info=None,
            suggestions=[],
            side_effects=side_effects,
            deprecated=False,
            node_context=context,
            features=[],
            run=run,
        )
        node = object.__new__(node_class)
        node.id, node.schema_id, node.data = NodeId(node_id), data.schema_id, data
        self.chain.add_node(node)
        return node

    def edge(self, source: str, output: int, target: str, input_index: int) -> None:
        self.chain.add_edge(
            Edge(EdgeSource(NodeId(source), OutputId(output)), EdgeTarget(NodeId(target), InputId(input_index)))
        )

    def value(self, target: str, input_index: int, value: object) -> None:
        self.chain.inputs.set(NodeId(target), InputId(input_index), value)


Scenario = Callable[[list[object]], Chain]


def run_both(scenario: Scenario, tmp_path: Path, **options) -> tuple[Chain, Result, Result, list[object], list[object]]:
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
    chain, oracle, engine, oracle_trace, engine_trace = run_both(scenario, tmp_path, **options)
    assert mismatches(scenario.__name__, chain, oracle, engine, tmp_path) == []
    assert engine_trace == oracle_trace
    return engine


@pytest.fixture(autouse=True)
def test_package(monkeypatch):
    # Test nodes belong to no package: both executors read their settings from a stand-in.
    monkeypatch.setattr(api.registry, "get_package", lambda _: SimpleNamespace(id="test"))


# Regular nodes


def independent_sinks(trace: list[object]) -> Chain:
    b = Builder()
    for name in ("a", "b", "c"):
        b.node(f"double_{name}", lambda x: x * 2)
        b.value(f"double_{name}", 0, {"a": 1, "b": 2, "c": 3}[name])
        b.node(f"sink_{name}", lambda x, name=name: trace.append((name, x)), outputs=0, side_effects=True)
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
    b.node("sink", lambda v: trace.append(("sink", v, calls["shared"])), outputs=0, side_effects=True)
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
    b.node("choose", lambda condition, lazy: trace.append(("chose", lazy.value if condition else None)), inputs=2,
           outputs=0, side_effects=True, lazy=(1,))
    b.value("choose", 0, False)
    b.edge("expensive", 0, "choose", 1)
    return b.chain


def lazy_read(trace: list[object]) -> Chain:
    b = Builder()
    b.node("upstream", lambda x: x * 7)
    b.value("upstream", 0, 6)
    b.node("reader", lambda lazy: trace.append(("read", lazy.value)), outputs=0, side_effects=True, lazy=(0,))
    b.edge("upstream", 0, "reader", 0)
    return b.chain


def lazy_constant(trace: list[object]) -> Chain:
    b = Builder()
    b.node("reader", lambda lazy: trace.append(("read", lazy.value)), outputs=0, side_effects=True, lazy=(0,))
    b.value("reader", 0, "constant")
    return b.chain


def lazy_upstream_error(trace: list[object]) -> Chain:
    b = Builder()

    def fails(x):
        raise ValueError(f"upstream failed on {x}")

    b.node("upstream", fails)
    b.value("upstream", 0, 3)
    b.node("reader", lambda lazy: trace.append(("read", lazy.value)), outputs=0, side_effects=True, lazy=(0,))
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
    b.node("sink", lambda x, y: trace.append((x, y)), inputs=2, outputs=0, side_effects=True)
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


def items(values: list[object], *, fail_fast: bool = True, source_paths: tuple | None = ()) -> Generator:
    def supplier():
        yield from values

    return Generator(supplier, len(values), fail_fast=fail_fast, source_paths=source_paths)


def loop_chain(
    trace: list[object],
    values: list[object],
    *,
    fail_fast: bool = True,
    middle: Callable[[Any], Any] = lambda x: x * 10,
    collect: bool = True,
) -> Chain:
    b = Builder()
    b.node("numbers", lambda: items(values, fail_fast=fail_fast), kind="generator", inputs=0)
    b.node("scale", middle)
    b.node("sink", lambda x: trace.append(("sink", x)), outputs=0, side_effects=True)
    b.edge("numbers", 0, "scale", 0)
    b.edge("scale", 0, "sink", 0)
    if collect:
        def accumulate(_):
            collected: list[Any] = []
            return Collector(lambda v: collected.append(v), lambda: sum(collected))

        b.node("total", accumulate, kind="collector")
        b.edge("scale", 0, "total", 0)
        b.node("report", lambda total: trace.append(("total", total)), outputs=0, side_effects=True)
        b.edge("total", 0, "report", 0)
    return b.chain


def generator_items(trace: list[object]) -> Chain:
    return loop_chain(trace, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


def generator_fail_fast(trace: list[object]) -> Chain:
    return loop_chain(trace, [1, 2, ValueError("item 3 is bad"), 4, 5])


def generator_deferred(trace: list[object]) -> Chain:
    return loop_chain(trace, [1, ValueError("item 2 is bad"), 3, KeyError("item 4 is bad"), 5], fail_fast=False)


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
    b.node("numbers", lambda: items([1, 2, 3, 4, 5, 6, 7, 8, 9]), kind="generator", inputs=0)
    b.node("add", lambda x, y: x + y, inputs=2)
    b.node("sink", lambda v: trace.append((v, calls["offset"])), outputs=0, side_effects=True)
    b.edge("numbers", 0, "add", 0)
    b.edge("offset", 0, "add", 1)
    b.edge("add", 0, "sink", 0)
    return b.chain


def zipped_generators(trace: list[object]) -> Chain:
    b = Builder()
    b.node("left", lambda: items([1, 2, 3, 4, 5]), kind="generator", inputs=0)
    b.node("right", lambda: items([10, 20, 30, 40, 50]), kind="generator", inputs=0)
    b.node("pair", lambda x, y: x + y, inputs=2)
    b.node("sink", lambda v: trace.append(v), outputs=0, side_effects=True)
    b.edge("left", 0, "pair", 0)
    b.edge("right", 0, "pair", 1)
    b.edge("pair", 0, "sink", 0)
    return b.chain


def unequal_lengths(trace: list[object]) -> Chain:
    b = Builder()
    b.node("left", lambda: items([1, 2, 3]), kind="generator", inputs=0)
    b.node("right", lambda: items([1, 2]), kind="generator", inputs=0)
    b.node("pair", lambda x, y: x + y, inputs=2)
    b.node("sink", lambda v: trace.append(v), outputs=0, side_effects=True)
    b.edge("left", 0, "pair", 0)
    b.edge("right", 0, "pair", 1)
    b.edge("pair", 0, "sink", 0)
    return b.chain


def generator_static_output(trace: list[object]) -> Chain:
    b = Builder()
    b.node("numbers", lambda: (items([1, 2, 3, 4, 5, 6, 7]), "folder"), kind="generator", inputs=0, outputs=2)
    b.node("sink", lambda x, folder: trace.append((x, folder)), inputs=2, outputs=0, side_effects=True)
    b.edge("numbers", 0, "sink", 0)
    b.edge("numbers", 1, "sink", 1)
    b.node("after", lambda folder: trace.append(("after", folder)), outputs=0, side_effects=True)
    b.edge("numbers", 1, "after", 0)
    return b.chain


def collector_only(trace: list[object]) -> Chain:
    b = Builder()
    b.node("numbers", lambda: items([3, 1, 4, 1, 5, 9, 2, 6]), kind="generator", inputs=0)

    def gather(_):
        seen: list[Any] = []
        return Collector(lambda v: (seen.append(v), trace.append(("iterate", v)))[0], lambda: max(seen))

    b.node("maximum", gather, kind="collector")
    b.edge("numbers", 0, "maximum", 0)
    b.node("report", lambda v: trace.append(("max", v)), outputs=0, side_effects=True)
    b.edge("maximum", 0, "report", 0)
    return b.chain


def look_ahead_chain(trace: list[object], source_paths: tuple | None) -> Chain:
    b = Builder()
    b.node("frames", lambda: items(list(range(12)), source_paths=source_paths), kind="generator", inputs=0)
    b.node("offset", lambda x: x + 1000)

    def temporal(context, frame, offset):
        view = context.iteration
        ahead = view.lookahead(0)
        peeked = [view.peek(0, k) for k in range(1, ahead + 1)]
        peeked_offset = view.peek(1, 1) if view.lookahead(1) else None
        trace.append((view.index, frame, ahead, peeked, peeked_offset, view.iterated_by(0), view.source))
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


def side_effect_feeding(trace: list[object]) -> Chain:
    # A side-effect node read by another node: consumed by its reader and by the run (its extra reservation).
    b = Builder()
    b.node("writer", lambda x: (trace.append(("write", x)), x + 1)[1], side_effects=True)
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


def test_pause_and_resume(tmp_path):
    def during(executor):
        time.sleep(0.02)
        executor.pause()
        time.sleep(0.3)
        assert executor.progress.paused
        executor.resume()

    def slow_items(trace: list[object]) -> Chain:
        return loop_chain(trace, list(range(20)), middle=lambda x: (time.sleep(0.01), x)[1])

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
        result = execute(kind, stop_chain(trace), storage=tmp_path / "storage", during=during)
        assert type(result.error).__name__ == "Aborted", (kind, result.error)
        assert trace[-1] == "chain cleanup" and trace.count("chain cleanup") == 1, kind
        assert 0 < len(trace) - 1 < 200, kind


def test_parent_cache_nodes_are_not_run(tmp_path):
    # /run with the server's cache of individually run nodes: those are neither run, started nor listed.
    def cached_chain(trace: list[object]) -> Chain:
        b = Builder()
        b.node("loaded", lambda x: trace.append(("loaded ran", x)) or x, side_effects=True)
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
            b.node("single", lambda x: (items([x, x + 1]), "static"), kind="generator", inputs=1, outputs=2)
        else:
            b.node("single", lambda x: (trace.append(x), x * 3)[1], side_effects=True)
        b.value("single", 0, 7)
        return b.chain

    outcomes = []
    for executor_kind in ("python", "native"):
        trace: list[object] = []
        chain = single(trace)
        events: list[dict] = []

        class Sink(EventConsumer):
            def put(self, event) -> None:
                events.append(json.loads(json.dumps(event)))

        async def main():
            loop = asyncio.get_running_loop()
            pool = ThreadPoolExecutor(8)
            queue = EventConsumer.filter(Sink(), {"node-finish", "node-broadcast", "execution-error"})
            arguments: dict[str, Any] = dict(
                id=ExecutionId("individual-executor single"),
                chain=chain,
                send_broadcast_data=True,
                options=ExecutionOptions.parse({}),
                loop=loop,
                queue=queue,
                storage_dir=tmp_path,
                pool=pool,
                pool_size=8,
            )
            executor = Executor(**arguments) if executor_kind == "python" else NativeExecutor(**arguments)
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

        outcomes.append((asyncio.run(main()), [canonical(e, tmp_path) for e in events], trace))
    assert outcomes[1] == outcomes[0]
