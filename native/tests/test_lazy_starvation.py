"""A node reading a lazy input holds its pool thread while the input's nodes run on
other pool threads, so a long series of lazy inputs (Conditional, Logic Operation, Save
Image) can occupy every pool thread. The run must then fail with a clear error, and
Stop must end it, instead of hanging until the worker is restarted."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import process
from api import BaseInput, BaseOutput, ExecutionOptions, NodeData, NodeId
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
            Edge(EdgeSource(NodeId(previous), 0), EdgeTarget(NodeId(node_id), 0))
        )
        previous = node_id
    chain.add_node(
        make_node("sink", lambda lazy: seen.append(lazy.value), 1, 0, side_effects=True)
    )
    chain.add_edge(Edge(EdgeSource(NodeId(previous), 0), EdgeTarget(NodeId("sink"), 0)))
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


def test_stop_ends_a_chain_waiting_on_every_pool_thread(monkeypatch):
    monkeypatch.setattr(process, "LAZY_STARVED_SECONDS", 60)
    with pytest.raises(Aborted):
        run_chain(7, stop_after=0.5)
