"""Real executor regressions for one generator construction per sequence run."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import process
from api import (
    BaseInput,
    BaseOutput,
    Collector,
    ExecutionOptions,
    Generator,
    InputId,
    IteratorInputInfo,
    IteratorOutputInfo,
    NodeData,
    NodeId,
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
from nodes.impl import native_profile
from nodes.impl.item_window import SERIAL_PRODUCER_VARIABLE
from process import ExecutionId, Executor
from progress_controller import Aborted


def make_node(node_id, kind, run, input_count, output_count, *, iterator=False):
    node_class = {
        "generator": GeneratorNode,
        "collector": CollectorNode,
        "regularNode": FunctionNode,
    }[kind]
    data = NodeData(
        schema_id=f"test:generator-lifecycle:{node_id}",
        description="",
        see_also=[],
        name=node_id,
        icon="",
        kind=kind,
        inputs=[BaseInput("any", f"Input{i}").with_id(i) for i in range(input_count)],
        outputs=[
            BaseOutput("any", "Directory" if i == 1 else f"Output{i}").with_id(i)
            for i in range(output_count)
        ],
        group_layout=[],
        iterable_inputs=[IteratorInputInfo(0)] if kind == "collector" else [],
        iterable_outputs=[IteratorOutputInfo(0)] if iterator else [],
        key_info=None,
        suggestions=[],
        side_effects=kind == "regularNode",
        deprecated=False,
        node_context=kind in {"generator", "collector"},
        features=[],
        run=run,
    )
    # Avoid registering fixture schemas globally; these are ordinary production
    # node objects with actual NodeData and actual executor/cache/edge traversal.
    node = object.__new__(node_class)
    node.id, node.schema_id, node.data = NodeId(node_id), data.schema_id, data
    return node


class EventCallback(EventConsumer):
    """An event queue that hands every event the executor sends to a callback."""

    def __init__(self, callback):
        self.callback = callback

    def put(self, event):
        self.callback(event)


class CountedIterator:
    """Counts every next() call on the iterator, StopIteration included."""

    def __init__(self, iterator, counter, key):
        self.iterator, self.counter, self.key = iterator, counter, key

    def __iter__(self):
        return self

    def __next__(self):
        self.counter[self.key] += 1
        return next(self.iterator)


class SplitIterator(CountedIterator):
    """A CountedIterator that also names its next item (describe) and gives it
    (materialize), as Load Images' native iterator does (SP3b); describe() is counted
    as next() and raises StopIteration at the end."""

    def describe(self):
        return (next(self),)

    def materialize(self, token):
        (value,) = token
        return value


class Scenario:
    def __init__(
        self,
        monkeypatch,
        tmp_path,
        sequences,
        *,
        fail_fast=True,
        collect=False,
        intermediate=False,
        source_paths=None,
        expected_lengths=None,
        split=False,
        lazy=False,
    ):
        # split: each generator's iterator is a SplitIterator (Load Images' two phases).
        # lazy: the sink's input 0 is lazy, and the sink reads it, as Save its image.
        self.monkeypatch = monkeypatch
        self.directory = tmp_path
        self.sequences = sequences
        self.fail_fast = fail_fast
        self.source_paths = source_paths
        self.expected_lengths = expected_lengths
        self.created = Counter()
        self.supplied = Counter()
        self.advances = Counter()  # every next() call, StopIteration included
        self.yielded = defaultdict(list)
        self.inside = 0  # threads inside on_yield
        self.inside_at_cleanup = []
        self.outputs = []
        self.static_outputs = []
        self.cleanups = []
        self.broadcasts = []
        self.events = []
        self.collector_calls = Counter()
        self.collected = []
        self.mapped = []
        self.on_sink: Callable[[tuple[object, ...]], object] | None = None
        self.on_yield: Callable[[str, object], object] | None = None
        self.on_event: Callable[[dict], object] | None = None
        self.collector_directory: Path | None = None
        self._executor: Executor | None = None
        self.loop_thread = None  # the thread whose event loop ran the executor
        self.pool_calls = PoolCalls()
        self.chain = Chain()
        monkeypatch.setattr(
            process.registry, "get_package", lambda _: SimpleNamespace(id="test")
        )

        for number in range(len(sequences)):
            node_id = f"g{number}"

            def factory(context, node_id=node_id, number=number):
                self.created[node_id] += 1
                generation = self.created[node_id]
                # Simulates a directory scan: construction captures the list once.
                captured = list(self.sequences[number])
                directory = self.directory / f"{node_id}-scan-{generation}"

                def cleanup():
                    self.cleanups.append((node_id, generation))
                    self.inside_at_cleanup.append(self.inside)

                context.add_cleanup(cleanup)

                def items():
                    for value in captured:
                        self.yielded[node_id].append(value)
                        if self.on_yield is not None:
                            self.inside += 1
                            try:
                                self.on_yield(node_id, value)
                            finally:
                                self.inside -= 1
                        yield value

                def supplier():
                    self.supplied[node_id] += 1
                    iterator = SplitIterator if split else CountedIterator
                    return iterator(items(), self.advances, node_id)

                length = (
                    len(captured)
                    if self.expected_lengths is None
                    else self.expected_lengths[number]
                )
                generator = Generator(supplier, length, self.fail_fast)
                if self.source_paths is not None:
                    generator.source_paths = self.source_paths
                return generator, directory

            self.chain.add_node(
                make_node(node_id, "generator", factory, 0, 2, iterator=True)
            )

        def consume(*values):
            if lazy:
                values = (values[0].value, *values[1:])
            self.outputs.append(values)
            if self.on_sink is not None:
                self.on_sink(values)

        sink = make_node("sink", "regularNode", consume, len(sequences) * 2, 0)
        if lazy:
            sink.data.inputs[0].make_lazy()
        self.chain.add_node(sink)
        for number in range(len(sequences)):
            source = f"g{number}"
            self.connect(source, 0, "sink", number * 2)
            self.connect(source, 1, "sink", number * 2 + 1)

        if intermediate:
            assert len(sequences) == 1
            edge = self.chain.edge_to(NodeId("sink"), InputId(0))
            assert edge is not None
            self.chain.remove_edge(edge)

            def map_value(value):
                self.mapped.append(value)
                return value * 10 + 7

            node = make_node("map", "regularNode", map_value, 1, 1)
            object.__setattr__(node.data, "side_effects", False)
            self.chain.add_node(node)
            self.connect("g0", 0, "map", 0)
            self.connect("map", 0, "sink", 0)

        # A non-iterated consumer exercises final restoration of the original
        # static Directory output after the item cache has been cleared.
        self.chain.add_node(
            make_node("directory", "regularNode", self.static_outputs.append, 1, 0)
        )
        self.connect("g0", 1, "directory", 0)

        if collect:
            assert len(sequences) == 1

            def collector_factory(_context, _item, directory):
                self.collector_calls["create"] += 1
                self.collector_directory = directory

                def complete():
                    self.collector_calls["complete"] += 1
                    return len(self.collected)

                return Collector(self.collected.append, complete)

            self.chain.add_node(
                make_node("collector", "collector", collector_factory, 2, 1)
            )
            self.connect("g0", 0, "collector", 0)
            self.connect("g0", 1, "collector", 1)

    @property
    def executor(self) -> Executor:
        """The executor new_executor() made."""
        assert self._executor is not None, "new_executor() has not run"
        return self._executor

    def connect(self, source, output, target, input_id):
        self.chain.add_edge(
            Edge(
                EdgeSource(NodeId(source), output), EdgeTarget(NodeId(target), input_id)
            )
        )

    def new_executor(self, pool, pool_size=None):
        def event(value):
            self.events.append(value)
            if self.on_event is not None:
                self.on_event(value)

        sized = {} if pool_size is None else {"pool_size": pool_size}
        self.loop_thread = threading.get_ident()
        self.pool_calls.install()
        executor = Executor(
            ExecutionId("generator-lifecycle"),
            self.chain,
            False,
            ExecutionOptions({}),
            asyncio.get_running_loop(),
            EventCallback(event),
            pool,
            self.directory,
            **sized,
        )

        async def capture(node, values, generators=None):
            self.broadcasts.append((node.id, list(values), generators is not None))

        # Capture callback payloads, leaving real node execution, input/output
        # enforcement, cache, progress and event scheduling untouched.
        self.monkeypatch.setattr(executor, "_Executor__send_node_broadcast", capture)
        self._executor = executor
        return executor


async def spin(steps=100):
    """Run the loop `steps` times. With no pool call running, whatever the window
    would start or finish by itself has happened by then (one describe's completion
    takes three steps): run before asserting that something did not start."""
    for _ in range(steps):
        await asyncio.sleep(0)


class PoolCalls:
    """The executor's pool calls in flight, so a node can wait for the engine to
    settle instead of sleeping to give it a lead.

    install() wraps the running loop's run_in_executor, through which the executor
    and its item window make every pool call; a call is in flight until its future
    is done. settle(own), from a pool thread, returns once at most `own` calls (the
    caller's own, which wait) are in flight and spin()'s steps saw no call start or
    finish: the window has started and finished everything it would by itself.
    """

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.in_flight = self.started = self.finished = 0
        self.changed: asyncio.Event | None = None

    def install(self):
        """Count the running loop's pool calls (once per loop)."""
        loop = asyncio.get_running_loop()
        if self.loop is loop:
            return
        self.loop, self.changed = loop, asyncio.Event()
        run_in_executor = loop.run_in_executor

        def counted[T](
            executor: concurrent.futures.Executor | None,
            func: Callable[..., T],
            *args: object,
        ) -> asyncio.Future[T]:
            future = run_in_executor(executor, func, *args)
            self.in_flight += 1
            self.started += 1
            future.add_done_callback(self.done)
            return future

        loop.run_in_executor = counted

    def done(self, _future):
        assert self.changed is not None
        self.in_flight -= 1
        self.finished += 1
        self.changed.set()

    async def settled(self, own):
        changed = self.changed
        assert changed is not None
        while True:
            if own is not None and self.in_flight > own:
                changed.clear()
                await changed.wait()
                continue
            before = (self.started, self.finished)
            await spin()
            if (self.started, self.finished) == before:
                return

    def settle(self, own: int | None = 1, timeout=5):
        """Block this pool thread until settled(own) on the loop; None `own` waits
        until every call in flight waits too."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:  # no loop runs on this thread
            running = None
        assert running is not self.loop, "settle() blocks: call it on a pool thread"
        assert self.loop is not None, "install() has not run"
        future = asyncio.run_coroutine_threadsafe(self.settled(own), self.loop)
        try:
            future.result(timeout)
        except TimeoutError:
            future.cancel()
            raise AssertionError(f"the engine did not settle in {timeout} s") from None


DEFAULT = object()


def drive(body, timeout=10):
    """Run body() on its own loop in a daemon thread: a hang fails the test, not the suite.

    body's exception, if any, is raised here.
    """
    outcome = []

    def target():
        try:
            asyncio.run(body())
            outcome.append(None)
        except BaseException as error:
            outcome.append(error)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), f"the run hung for {timeout} s"
    if outcome[0] is not None:
        raise outcome[0]


def run(scenario, workers=2, pool_size=DEFAULT, timeout=30):
    """Run the scenario on a pool of `workers` threads, bounded by `timeout` seconds.

    The executor learns the pool size (DEFAULT: `workers`); None withholds it. The
    loop runs in its own thread (scenario.loop_thread), so a hang fails the test.
    """
    if pool_size is DEFAULT:
        pool_size = workers

    async def body():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            await scenario.new_executor(pool, pool_size).run()

    drive(body, timeout)


def profiled_run(scenario, workers=2, timeout=30):
    """run(scenario), wrapped as the worker's /run wraps its executor (server.py): the
    profile is reset and, if it is on, the loop thread's CPU time started before it;
    after it, on the loop thread, the profile's log line is taken and returned."""
    lines = []

    async def body():
        native_profile.reset()
        if native_profile.enabled():
            native_profile.start_loop_thread_cpu()
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                await scenario.new_executor(pool, workers).run()
        finally:
            lines.append(native_profile.log_line())

    drive(body, timeout)
    return lines[0]


def set_window(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CHAINNER_C_ITEM_WINDOW", raising=False)
    else:
        monkeypatch.setenv("CHAINNER_C_ITEM_WINDOW", value)


def decisions(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if re.match(r"item window K=\d+ \(", r.getMessage())
    ]


def stripped(events):
    return [
        (e["event"], e["data"].get("nodeId"), e["data"].get("index")) for e in events
    ]


def window_summaries(caplog):
    """One dict per `item window K=<k>: <n> <name>, ...` line: "k" and each count."""
    summaries = []
    for record in caplog.records:
        if match := re.fullmatch(r"item window K=(\d+): (.*)", record.getMessage()):
            summary = {"k": int(match[1])}
            for pair in match[2].split(", "):
                count, name = pair.split(" ", 1)
                summary[name] = int(count)
            summaries.append(summary)
    return summaries


def as_range(scenario, node_id):
    """Give the generator Range's schema id: the loop and the producer advance it inline."""
    node = scenario.chain.nodes[NodeId(node_id)]
    node.schema_id = "chainner:utility:range"
    object.__setattr__(node.data, "schema_id", "chainner:utility:range")


def add_group(scenario, name, values):
    """A second, separate group: generator `name` over values and its sink `<name>-sink`."""
    sunk = []

    def factory(_context):
        generator = Generator(lambda: iter(values), len(values))
        generator.source_paths = ()
        return generator

    scenario.chain.add_node(make_node(name, "generator", factory, 0, 1, iterator=True))
    scenario.chain.add_node(make_node(f"{name}-sink", "regularNode", sunk.append, 1, 0))
    scenario.connect(name, 0, f"{name}-sink", 0)
    return sunk


@pytest.mark.parametrize("length", [0, 1, 2, 9])
@pytest.mark.parametrize("collect", [False, True])
def test_generator_and_supplier_created_once_with_stable_directory(
    monkeypatch, tmp_path, length, collect
):
    scenario = Scenario(monkeypatch, tmp_path, [list(range(length))], collect=collect)
    run(scenario)
    directory = tmp_path / "g0-scan-1"
    assert scenario.created == {"g0": 1}
    assert scenario.supplied == {"g0": 1}
    assert scenario.outputs == [(i, directory) for i in range(length)]
    assert scenario.static_outputs == [directory]
    assert scenario.cleanups == [("g0", 1)]
    starts = [
        x
        for x in scenario.events
        if x["event"] == "node-start" and x["data"]["nodeId"] == "g0"
    ]
    assert len(starts) == 1
    broadcasts = [x for x in scenario.broadcasts if x[0] == "g0"]
    assert broadcasts[0] == ("g0", [None, directory], True)
    assert broadcasts[-1] == ("g0", [None, directory], False)
    if collect:
        assert scenario.collector_calls == {"create": 1, "complete": 1}
        assert scenario.collected == list(range(length))
        assert scenario.collector_directory == directory


@pytest.mark.parametrize("length", [0, 1, 5])
def test_zipped_generators_each_construct_once(monkeypatch, tmp_path, length):
    scenario = Scenario(
        monkeypatch, tmp_path, [list(range(length)), list(range(10, 10 + length))]
    )
    run(scenario)
    assert scenario.created == scenario.supplied == {"g0": 1, "g1": 1}
    assert scenario.outputs == [
        (i, tmp_path / "g0-scan-1", i + 10, tmp_path / "g1-scan-1")
        for i in range(length)
    ]
    assert sorted(scenario.cleanups) == [("g0", 1), ("g1", 1)]


def test_zip_length_error_does_not_recreate_or_consume(monkeypatch, tmp_path):
    scenario = Scenario(monkeypatch, tmp_path, [[1], [10, 20]])
    with pytest.raises(AssertionError, match="same length"):
        run(scenario)
    assert scenario.created == scenario.supplied == {"g0": 1, "g1": 1}
    assert not scenario.outputs and not scenario.yielded
    assert sorted(scenario.cleanups) == [("g0", 1), ("g1", 1)]


@pytest.mark.parametrize("fail_fast", [False, True])
def test_yielded_errors_keep_policy_and_constructor_identity(
    monkeypatch, tmp_path, fail_fast
):
    failure = ValueError("item failed")
    scenario = Scenario(
        monkeypatch, tmp_path, [[0, failure, 2]], fail_fast=fail_fast, collect=True
    )
    with pytest.raises(
        ValueError if fail_fast else Exception, match="item failed"
    ) as captured:
        run(scenario)
    if fail_fast:
        assert captured.value is failure
    else:
        assert str(captured.value) == "Errors occurred during iteration:\n- item failed"
    assert scenario.created == scenario.supplied == {"g0": 1}
    assert [x[0] for x in scenario.outputs] == ([0] if fail_fast else [0, 2])
    assert all(x[1] == tmp_path / "g0-scan-1" for x in scenario.outputs)
    assert scenario.collected == ([0] if fail_fast else [0, 2])
    assert scenario.collector_calls["complete"] == (0 if fail_fast else 1)
    assert scenario.cleanups == [("g0", 1)]


def test_continued_downstream_failure_clears_partial_iteration_cache(
    monkeypatch, tmp_path
):
    scenario = Scenario(
        monkeypatch, tmp_path, [[0, 1, 2]], fail_fast=False, intermediate=True
    )

    def fail_once(_values):
        if len(scenario.outputs) == 1:
            raise ValueError("transient sink failure")

    scenario.on_sink = fail_once
    with pytest.raises(Exception, match="transient sink failure"):
        run(scenario)
    assert scenario.created == scenario.supplied == {"g0": 1}
    assert scenario.mapped == [0, 1, 2]
    assert [x[0] for x in scenario.outputs] == [7, 17, 27]
    assert all(x[1] == tmp_path / "g0-scan-1" for x in scenario.outputs)


@pytest.mark.parametrize("window", ["1", "3"])
@pytest.mark.parametrize("mode", ["abort", "cancel"])
@pytest.mark.parametrize("all_errors", [False, True])
def test_abort_and_cancel_do_not_recreate_or_drain_sequence(
    monkeypatch, tmp_path, caplog, mode, all_errors, window
):
    set_window(monkeypatch, window)
    items = (
        [ValueError(f"bad{i}") for i in range(12)] if all_errors else list(range(12))
    )
    scenario = Scenario(
        monkeypatch, tmp_path, [items], fail_fast=not all_errors, source_paths=()
    )
    # K = 3 stops between items, after the window opened (spec 5.2); all_errors stops
    # inside item 0, before slot 1, at either K.
    stop_at = 1 if window == "1" or all_errors else 2
    if window == "3" and not all_errors:
        # the producer reads ahead while item 1 commits
        scenario.on_sink = lambda _values: scenario.pool_calls.settle()

    async def body():
        with ThreadPoolExecutor(max_workers=2) as pool:
            executor = scenario.new_executor(pool, pool_size=2)
            task = None
            loop = asyncio.get_running_loop()

            def stop():
                if mode == "abort":
                    executor.kill()
                else:
                    assert task is not None
                    task.cancel()

            if all_errors:

                def on_yield(_node, _value):
                    if len(scenario.yielded["g0"]) == 1:
                        loop.call_soon_threadsafe(stop)

                scenario.on_yield = on_yield
            else:

                def on_event(event):
                    if (
                        event["event"] == "node-progress"
                        and event["data"]["nodeId"] == "g0"
                        and event["data"]["index"] == stop_at
                    ):
                        stop()

                scenario.on_event = on_event
            task = asyncio.create_task(executor.run())
            with pytest.raises(Aborted if mode == "abort" else asyncio.CancelledError):
                await asyncio.wait_for(task, 5)

    with caplog.at_level(logging.INFO, logger="sanic.root"):
        asyncio.run(body())
    if window == "3":
        committed = 0 if all_errors else stop_at
        # all_errors stops in item 0, before the decision at slot 1
        assert decisions(caplog) == (
            [] if all_errors else ["item window K=3 (override)"]
        )
        assert scenario.created == scenario.supplied == {"g0": 1}
        # at most K - 1 items beyond the last committed one
        assert len(scenario.yielded["g0"]) <= committed + 2
        if not all_errors:
            # and it did read ahead
            assert len(scenario.yielded["g0"]) >= committed + 1
            # logged when the window closed on Stop
            assert window_summaries(caplog)[0]["ahead"] >= 1
        # outputs and events only for committed items
        assert len(scenario.outputs) == committed
        progress = [
            e["data"]["index"] for e in scenario.events if e["event"] == "node-progress"
        ]
        assert max(progress) == committed
        # cleanup once, after every engine job finished
        assert scenario.cleanups == [("g0", 1)] and scenario.inside_at_cleanup == [0]
    else:
        assert scenario.created == scenario.supplied == {"g0": 1}
        assert len(scenario.yielded["g0"]) == 1
        assert len(scenario.outputs) == (0 if all_errors else 1)
        assert scenario.cleanups == [("g0", 1)]


@pytest.mark.parametrize("window", ["1", "3"])
def test_pause_resume_uses_same_supplier_without_extra_construction(
    monkeypatch, tmp_path, caplog, window
):
    set_window(monkeypatch, window)
    # K = 3 pauses between items, after the window opened (spec 5.2)
    pause_at, items = (0, [0, 1, 2]) if window == "1" else (1, list(range(6)))
    scenario = Scenario(monkeypatch, tmp_path, [items], source_paths=())

    async def body():
        loop = asyncio.get_running_loop()
        paused = asyncio.Event()
        with ThreadPoolExecutor(max_workers=2) as pool:
            executor = scenario.new_executor(pool, pool_size=2)

            def pause_once(values):
                if values[0] == pause_at:
                    if window == "3":
                        scenario.pool_calls.settle()  # the producer reads ahead

                    def pause():
                        executor.pause()
                        paused.set()

                    loop.call_soon_threadsafe(pause)

            scenario.on_sink = pause_once
            task = asyncio.create_task(executor.run())
            try:
                await asyncio.wait_for(paused.wait(), 5)
                await asyncio.sleep(0)
                assert not task.done() and executor.progress.paused
                assert scenario.created == scenario.supplied == {"g0": 1}
                if window == "3":
                    yielded = scenario.yielded["g0"]
                    assert decisions(caplog) == ["item window K=3 (override)"]
                    # read ahead, at most K - 1 items beyond item 1
                    assert 3 <= len(yielded) <= 4
                    assert yielded[:2] == [0, 1]
                else:
                    assert scenario.yielded["g0"] == [0]
            finally:
                executor.resume()
            await asyncio.wait_for(task, 5)

    with caplog.at_level(logging.INFO, logger="sanic.root"):
        asyncio.run(body())
    assert scenario.created == scenario.supplied == {"g0": 1}
    assert [x[0] for x in scenario.outputs] == items
    if window == "3":
        assert window_summaries(caplog)[0]["ahead"] >= 1


@pytest.mark.parametrize("fresh_executor", [False, True])
def test_subsequent_run_captures_fresh_input_list(
    monkeypatch, tmp_path, fresh_executor
):
    scenario = Scenario(monkeypatch, tmp_path, [[0, 1]])

    async def body():
        with ThreadPoolExecutor(max_workers=2) as pool:
            executor = scenario.new_executor(pool)
            await executor.run()
            scenario.sequences[0] = [7, 8, 9]
            if fresh_executor:
                executor = scenario.new_executor(pool)
            await executor.run()

    asyncio.run(body())
    assert scenario.created == scenario.supplied == {"g0": 2}
    assert scenario.outputs == [(i, tmp_path / "g0-scan-1") for i in [0, 1]] + [
        (i, tmp_path / "g0-scan-2") for i in [7, 8, 9]
    ]
    assert scenario.static_outputs == [tmp_path / "g0-scan-1", tmp_path / "g0-scan-2"]
    assert scenario.cleanups == [("g0", 1), ("g0", 2)]


def test_producer_reads_ahead_at_most_k_minus_1_items(monkeypatch, tmp_path, caplog):
    set_window(monkeypatch, "3")
    scenario = Scenario(monkeypatch, tmp_path, [list(range(8))], source_paths=())
    seen, threads = [], []

    def slow_sink(_values):
        scenario.pool_calls.settle()  # the producer reads all that K admits
        seen.append(len(scenario.yielded["g0"]))

    scenario.on_sink = slow_sink
    scenario.on_yield = lambda *_: threads.append(threading.get_ident())
    with caplog.at_level(logging.INFO, logger="sanic.root"):
        run(scenario)
    ahead = [count - (index + 1) for index, count in enumerate(seen)]
    assert decisions(caplog) == ["item window K=3 (override)"]
    assert [x[0] for x in scenario.outputs] == list(range(8))
    assert ahead[0] == 0 and all(0 <= a <= 2 for a in ahead) and max(ahead) >= 1
    [summary] = window_summaries(caplog)
    # slots 1-8 (the last one exhausted); some read ahead
    assert summary["items"] == 8 and summary["ahead"] >= 1
    # advances stay off the loop thread
    assert scenario.loop_thread not in threads
    assert scenario.created == scenario.supplied == {"g0": 1}


LATE = ValueError("g1 item 1")


@pytest.mark.parametrize(
    ("sequences", "lengths", "fail_fast", "reads_ahead"),
    [
        pytest.param(
            [[0, ValueError("g0 item 1"), 2, 3], [10, 11, ValueError("g1 item 2"), 13]],
            None,
            False,
            True,
            id="middle-slot-errors",
        ),
        pytest.param(
            [[0, 1, 2, 3], [10, 11]],
            [4, 4],
            False,
            True,
            id="later-generator-runs-out-first",
        ),
        pytest.param([[0, 1, 2], [10, 11, 12]], None, True, True, id="exhaustion"),
        pytest.param(  # nothing is read past the failing slot
            [[0, 1, 2, 3], [10, LATE, 12, 13]],
            None,
            True,
            False,
            id="fail-fast-middle-slot",
        ),
    ],
)
def test_zipped_generators_keep_the_per_slot_pattern(
    monkeypatch, tmp_path, caplog, sequences, lengths, fail_fast, reads_ahead
):
    observed = []
    for window in ("1", "3"):
        set_window(monkeypatch, window)
        scenario = Scenario(
            monkeypatch,
            tmp_path,
            sequences,
            fail_fast=fail_fast,
            source_paths=(),
            expected_lengths=lengths,
        )
        try:
            with caplog.at_level(logging.INFO, logger="sanic.root"):
                run(scenario)
            error = None
        except Exception as caught:
            error = caught
        if fail_fast and error is not None:
            assert error is LATE
        observed.append(
            (
                dict(scenario.advances),
                dict(scenario.yielded),
                scenario.outputs,
                repr(error),
                scenario.broadcasts,
                stripped(scenario.events),
            )
        )
    assert decisions(caplog) == [
        "item window K=1 (forced)",
        "item window K=3 (override)",
    ]
    summaries = window_summaries(caplog)  # only the K = 3 run has a window
    # its slots from 1 on came from the producer
    assert len(summaries) == 1 and summaries[0]["items"] >= 1
    assert (summaries[0]["ahead"] >= 1) == reads_ahead
    assert observed[0] == observed[1]


def test_loop_exit_waits_for_the_producer_before_cleanup(monkeypatch, tmp_path):
    set_window(monkeypatch, "3")
    scenario = Scenario(monkeypatch, tmp_path, [list(range(6))], source_paths=())
    reading, inside_at_failure = threading.Event(), []

    def slow_read(_node, value):
        if value == 2:
            reading.set()  # the producer is inside item 2's read
            # until the loop failed and waits for nothing else
            scenario.pool_calls.settle()

    def fail_on_one(values):
        if values[0] == 1:
            inside_at_failure.append((reading.wait(5), scenario.inside))
            raise ValueError("sink failed")

    scenario.on_yield = slow_read
    scenario.on_sink = fail_on_one
    with pytest.raises(Exception, match="sink failed"):
        run(scenario)
    # the producer was inside item 2's read when the loop failed
    assert inside_at_failure == [(True, 1)]
    assert [x[0] for x in scenario.outputs] == [0, 1]
    assert scenario.cleanups == [("g0", 1)] and scenario.inside_at_cleanup == [0]


def test_range_producer_stays_on_the_loop_thread(monkeypatch, tmp_path, caplog):
    for window in ("1", "3"):
        set_window(monkeypatch, window)
        scenario = Scenario(monkeypatch, tmp_path, [list(range(6))], source_paths=())
        as_range(scenario, "g0")
        threads = []
        scenario.on_yield = lambda *_, threads=threads: threads.append(
            threading.get_ident()
        )
        scenario.on_sink = lambda _values, calls=scenario.pool_calls: calls.settle()
        with caplog.at_level(logging.INFO, logger="sanic.root"):
            run(scenario)
        assert set(threads) == {scenario.loop_thread}  # inline, as today
        assert [x[0] for x in scenario.outputs] == list(range(6))
    assert decisions(caplog) == [
        "item window K=1 (forced)",
        "item window K=3 (override)",
    ]
    [summary] = window_summaries(caplog)
    assert summary["ahead"] >= 1  # the K = 3 run's producer read ahead, inline


def test_each_group_opens_and_closes_its_own_window(monkeypatch, tmp_path, caplog):
    observed = []
    for window in ("1", "3"):
        set_window(monkeypatch, window)
        caplog.clear()
        scenario = Scenario(monkeypatch, tmp_path, [[0, 1, 2]], source_paths=())
        second = add_group(scenario, "h0", [10, 11, 12])
        with caplog.at_level(logging.INFO, logger="sanic.root"):
            run(scenario)
        observed.append((scenario.outputs, second, stripped(scenario.events)))
        lines = [
            r.getMessage().split(":")[0]
            for r in caplog.records
            if r.getMessage().startswith("item window ")
        ]
        if window == "1":
            assert lines == ["item window K=1 (forced)"] * 2
        else:  # a summary is logged after its window drained, before the next group decides
            assert lines == ["item window K=3 (override)", "item window K=3"] * 2
    assert observed[0] == observed[1]


# SP3b: a group's single generator is described serially and materialized as engine
# jobs; without describe(), its describe is the read and its materialize the enforce.


def test_window_summary_reports_materialized(monkeypatch, tmp_path, caplog):
    # P7: "<n> materialized" follows "<n> jobs" (node jobs only), and the generic parser
    # reads it. Each described slot is materialized once: slots 1-5 of six items (slot
    # 6 is the exhaustion).
    set_window(monkeypatch, "3")
    monkeypatch.delenv(SERIAL_PRODUCER_VARIABLE, raising=False)
    scenario = Scenario(monkeypatch, tmp_path, [list(range(6))], source_paths=())
    with caplog.at_level(logging.INFO, logger="sanic.root"):
        run(scenario)
    [summary] = window_summaries(caplog)
    assert list(summary) == [
        "k",
        "items",
        "ahead",
        "jobs",
        "materialized",
        "replayed",
        "awaited",
        "discarded",
    ]
    assert (summary["items"], summary["jobs"], summary["materialized"]) == (6, 0, 5)
    assert [x[0] for x in scenario.outputs] == list(range(6))


@pytest.mark.parametrize("fail_fast", [False, True])
def test_no_split_generator_yielding_none_fails_at_its_slot_as_today(
    monkeypatch, tmp_path, caplog, fail_fast
):
    # P1: the token of a generator without describe() is the 1-tuple of the value it
    # yields, so a yielded None is an item, not the end: its output enforce fails at its
    # slot (an output is never None), as at K = 1, and later items go on unless
    # fail-fast.
    monkeypatch.delenv(SERIAL_PRODUCER_VARIABLE, raising=False)
    observed = []
    for window in ("1", "3"):
        set_window(monkeypatch, window)
        scenario = Scenario(
            monkeypatch,
            tmp_path,
            [[0, 1, None, 3, 4]],
            fail_fast=fail_fast,
            source_paths=(),
        )
        try:
            with caplog.at_level(logging.INFO, logger="sanic.root"):
                run(scenario)
            error = None
        except Exception as caught:
            error = caught
        observed.append(
            (scenario.outputs, type(error), str(error), stripped(scenario.events))
        )
    assert observed[0] == observed[1]
    outputs, kind, message, _ = observed[1]
    if fail_fast:
        assert [x[0] for x in outputs] == [0, 1] and kind is AssertionError
    else:
        assert [x[0] for x in outputs] == [0, 1, 3, 4]
        assert message == "Errors occurred during iteration:\n- "
    [summary] = window_summaries(caplog)  # the K = 3 run's
    # Slots 1-4, slot 5 being the end. Fail-fast: slots 1 and 2, where it failed; with
    # P = 1 (two pool threads) slot 3 is not described before slot 2's error ends it.
    assert summary["materialized"] == (2 if fail_fast else 4)


# SP3b Task 7: the commit path's timers, with CHAINNER_C_PROFILE on.
COMMIT_PATH_TIMERS = ("window.take_advance", "lazy.resolve", "loop.thread_cpu")


def test_commit_path_timers_appear_once_per_take(monkeypatch, tmp_path, caplog):
    # A split Load Images -> lazy Save stand-in at K = 3, run as the worker's /run runs
    # it. On, the loop's wait in take_advance is recorded once per slot the window
    # served (slots 1-6, the last one the end); each item's lazy read, once, nested
    # under its reader's timed call (the sink's here, as image_io_save is Save's); the
    # loop thread's CPU time, once. Outputs and events equal those of the run with it off.
    set_window(monkeypatch, "3")
    monkeypatch.delenv(SERIAL_PRODUCER_VARIABLE, raising=False)
    runs = {}
    for on in (False, True):
        monkeypatch.setattr(native_profile, "enabled", lambda on=on: on)
        caplog.clear()
        scenario = Scenario(
            monkeypatch,
            tmp_path,
            [list(range(6))],
            source_paths=(),
            split=True,
            lazy=True,
        )
        if on:
            sink = scenario.chain.nodes[NodeId("sink")]
            timed = native_profile.timed("save", sink.data.run)
            object.__setattr__(sink.data, "run", timed)
        with caplog.at_level(logging.INFO, logger="sanic.root"):
            line = profiled_run(scenario)
        runs[on] = SimpleNamespace(
            line=line,
            decisions=decisions(caplog),
            summaries=window_summaries(caplog),
            observed=(
                scenario.outputs,
                scenario.broadcasts,
                stripped(scenario.events),
                dict(scenario.advances),
            ),
        )
    native_profile.reset()
    off, on = runs[False], runs[True]
    assert off.line is None
    assert on.observed == off.observed
    assert [x[0] for x in on.observed[0]] == list(range(6))
    assert on.decisions == ["item window K=3 (override)"]
    [summary] = on.summaries
    assert (summary["items"], summary["materialized"]) == (6, 5)  # the split path
    assert on.line is not None
    table = json.loads(on.line.removeprefix("native profile: "))
    assert {name: table[name]["calls"] for name in (*COMMIT_PATH_TIMERS, "save")} == {
        "window.take_advance": summary["items"],
        "lazy.resolve": 6,
        "loop.thread_cpu": 1,
        "save": 6,
    }
    for name in COMMIT_PATH_TIMERS:  # none has a timed call nested in it
        entry = table[name]
        assert entry["inclusive_ns"] == entry["exclusive_ns"] >= entry["max_ns"] >= 0
    # The lazy reads' waits are the only timed calls nested in the sink's.
    save = table["save"]
    nested = save["inclusive_ns"] - save["exclusive_ns"]
    assert nested == table["lazy.resolve"]["inclusive_ns"]
