"""A node reading a lazy input holds its pool thread while the input's nodes run on
other pool threads, so a long series of lazy inputs (Conditional, Logic Operation, Save
Image) can occupy every pool thread. The run must then fail with a clear error, and
Stop must end it, instead of hanging until the worker is restarted. A read that Stop
ends leaves the input's node running: the run ends after it."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import process
from api import (
    BaseInput,
    BaseOutput,
    ExecutionOptions,
    InputId,
    NodeData,
    NodeId,
    OutputId,
)
from chain.chain import Chain, Edge, EdgeSource, EdgeTarget, FunctionNode
from events import EventConsumer
from process import ExecutionId, Executor, NodeExecutionError
from progress_controller import Aborted

POOL_SIZE = 8  # the server's


def make_node(node_id, run, inputs, outputs, *, side_effects=False):
    data = NodeData(
        schema_id=f"test:lazy-starvation:{node_id}",
        description="",
        see_also=[],
        name=node_id,
        icon="",
        kind="regularNode",
        inputs=[
            BaseInput("any", f"Input{i}").with_id(i).make_lazy() for i in range(inputs)
        ],
        outputs=[BaseOutput("any", f"Output{i}").with_id(i) for i in range(outputs)],
        group_layout=[],
        iterable_inputs=[],
        iterable_outputs=[],
        key_info=None,
        suggestions=[],
        side_effects=side_effects,
        deprecated=False,
        node_context=False,
        features=[],
        run=run,
    )
    node = object.__new__(FunctionNode)
    node.id, node.schema_id, node.data = NodeId(node_id), data.schema_id, data
    return node


class Events(EventConsumer):
    def put(self, event):
        pass


def lazy_chain(depth, seen):
    """source -> relay 1 -> ... -> relay depth -> sink, every link a lazy input: the sink
    and the relays each hold a pool thread while the next link runs."""
    chain = Chain()
    chain.add_node(make_node("source", lambda: 42, 0, 1))
    previous = "source"
    for number in range(1, depth + 1):
        node_id = f"relay{number}"
        chain.add_node(make_node(node_id, lambda lazy: lazy.value, 1, 1))
        chain.add_edge(
            Edge(
                EdgeSource(NodeId(previous), OutputId(0)),
                EdgeTarget(NodeId(node_id), InputId(0)),
            )
        )
        previous = node_id
    chain.add_node(
        make_node("sink", lambda lazy: seen.append(lazy.value), 1, 0, side_effects=True)
    )
    chain.add_edge(
        Edge(
            EdgeSource(NodeId(previous), OutputId(0)),
            EdgeTarget(NodeId("sink"), InputId(0)),
        )
    )
    return chain


def run_chain(depth, *, stop_after=None):
    """Runs the chain on the server's pool size; returns what the sink saw, or raises
    the run's error. Fails instead of hanging when the run does not end."""
    seen = []

    async def main():
        pool = ThreadPoolExecutor(max_workers=POOL_SIZE)
        try:
            executor = Executor(
                ExecutionId("lazy-starvation"),
                lazy_chain(depth, seen),
                False,
                ExecutionOptions({}),
                asyncio.get_running_loop(),
                Events(),
                pool,
                Path("."),
                pool_size=POOL_SIZE,
            )
            task = asyncio.ensure_future(executor.run())
            if stop_after is not None:
                done, _ = await asyncio.wait([task], timeout=stop_after)
                assert not done, "the chain finished before Stop"
                executor.kill()
            done, _ = await asyncio.wait([task], timeout=10)
            assert done, "the run hung"
            task.result()
        finally:
            pool.shutdown(wait=False)

    asyncio.run(main())
    return seen


@pytest.fixture(autouse=True)
def package(monkeypatch):
    monkeypatch.setattr(
        process.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )


def test_a_chain_that_fits_the_pool_runs():
    # The sink and 6 relays wait; the 8th thread runs the source.
    assert run_chain(6) == [42]


def test_a_chain_waiting_on_every_pool_thread_fails_clearly():
    with pytest.raises(
        NodeExecutionError, match="need more worker threads than the pool has"
    ):
        run_chain(7)


@pytest.mark.parametrize("depth", [12, 20])
def test_one_verdict_ends_a_series_longer_than_the_pool(monkeypatch, depth):
    # The pool is found starved once; the reads it ended wait for their nodes, which
    # fills the pool again, so every other wait that times out meanwhile ends with the
    # same error: the run fails on that verdict, however long the series (before: a
    # verdict per node beyond the pool, LAZY_STARVED_SECONDS each: 5 for 12 links,
    # 6 s, and 13 for 20, 14 s). Two waits may find the pool starved at one instant.
    starved = process._LazyWaits.starved
    verdicts = []

    def counted(waits):
        found = starved(waits)
        if found:
            verdicts.append(True)
        return found

    monkeypatch.setattr(process._LazyWaits, "starved", counted)
    with pytest.raises(
        NodeExecutionError, match="need more worker threads than the pool has"
    ):
        run_chain(depth)
    assert 1 <= len(verdicts) <= 2


def test_stop_ends_a_chain_waiting_on_every_pool_thread(monkeypatch):
    monkeypatch.setattr(process, "LAZY_STARVED_SECONDS", 60)
    with pytest.raises(Aborted):
        run_chain(7, stop_after=0.5)


def stop_during_a_lazy_read(monkeypatch, failure=None):
    """source -> slow -> sink, the sink reading slow through its lazy input and slow
    blocked until released; Stop while the sink reads. The run says (its log) that it
    waits for slow, which releases slow; a run that does not wait releases it once it
    ended. Returns the order of slow's end and the run's, and the errors logged."""
    order, errors = [], []
    started, release = threading.Event(), threading.Event()

    def slow(lazy):
        value = lazy.value
        started.set()
        assert release.wait(60), "slow was never released"
        order.append("slow returned")
        if failure is not None:
            raise failure
        return value

    def info(message, *args, **kwargs):
        if "lazy input" in message:
            release.set()

    monkeypatch.setattr(process.logger, "info", info)
    monkeypatch.setattr(
        process.logger, "error", lambda message, *args, **kwargs: errors.append(args)
    )
    chain = Chain()
    chain.add_node(make_node("source", lambda: 42, 0, 1))
    chain.add_node(make_node("slow", slow, 1, 1))
    chain.add_edge(
        Edge(
            EdgeSource(NodeId("source"), OutputId(0)),
            EdgeTarget(NodeId("slow"), InputId(0)),
        )
    )
    chain.add_node(make_node("sink", lambda lazy: lazy.value, 1, 0, side_effects=True))
    chain.add_edge(
        Edge(
            EdgeSource(NodeId("slow"), OutputId(0)),
            EdgeTarget(NodeId("sink"), InputId(0)),
        )
    )

    async def main():
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(max_workers=POOL_SIZE)
        try:
            executor = Executor(
                ExecutionId("lazy-stop"),
                chain,
                False,
                ExecutionOptions({}),
                loop,
                Events(),
                pool,
                Path("."),
                pool_size=POOL_SIZE,
            )
            task = asyncio.ensure_future(executor.run())
            assert await loop.run_in_executor(None, started.wait, 60)
            executor.kill()
            with pytest.raises(Aborted):
                await task
            order.append("run ended")
            release.set()
        finally:
            pool.shutdown(wait=True)

    asyncio.run(main())
    return order, errors


def test_stop_waits_for_the_node_a_lazy_read_waited_for(monkeypatch):
    # Stop ends the sink's read at once, but slow runs on: the run ends after it, as
    # after a node's own job, so the server releases the run's memory after it and the
    # next run does not start beside it (before: the run ended while slow ran).
    order, errors = stop_during_a_lazy_read(monkeypatch)
    assert order == ["slow returned", "run ended"]
    assert errors == []


def test_a_late_failure_of_a_lazy_inputs_node_is_logged(monkeypatch):
    # No read reports slow's error any more: the run logs it.
    order, errors = stop_during_a_lazy_read(monkeypatch, ValueError("slow failed late"))
    assert order == ["slow returned", "run ended"]
    assert [str(args[0]) for args in errors] == ["slow failed late"]
