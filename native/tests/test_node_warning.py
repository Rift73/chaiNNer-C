"""A node's NodeContext.warn reaches the app as one execution-warning while the run goes
on, on both executors, from a /run and from an individual run: the run of a node with
side effects as the app fills its inputs, whose output /run then takes from the cache
without running it again."""

from __future__ import annotations

import asyncio
import gc
import json
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from engine_differential import KINDS, execute
from test_engine_scenarios import Builder

import api
from api import NodeContext, NodeId
from chain.chain import Chain, FunctionNode
from events import EventConsumer
from native_executor import NativeExecutor
from process import ExecutionId, ExecutionOptions, Executor


@pytest.fixture(autouse=True)
def test_package(monkeypatch: pytest.MonkeyPatch):
    # Test nodes belong to no package: both executors read their settings from a stand-in.
    monkeypatch.setattr(
        api.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )


def warning_chain(threads: list[int]) -> Chain:
    def warns(context: NodeContext, value: int) -> int:
        threads.append(threading.get_ident())
        context.warn("This model is wrong.")
        return value * 3

    b = Builder()
    b.node("warns", warns, context=True, side_effects=True)
    b.value("warns", 0, 7)
    return b.chain


class Sink(EventConsumer):
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def put(self, event: Any) -> None:
        self.events.append(json.loads(json.dumps(event)))


def executor_of(
    kind: str,
    chain: Chain,
    sink: EventConsumer,
    pool: ThreadPoolExecutor,
    storage: Path,
) -> Executor | NativeExecutor:
    arguments: dict[str, Any] = {
        "id": ExecutionId("individual-executor warns"),
        "chain": chain,
        "send_broadcast_data": True,
        "options": ExecutionOptions.parse({}),
        "loop": asyncio.get_running_loop(),
        "queue": EventConsumer.filter(sink, EventConsumer.INDIVIDUAL_RUN_EVENTS),
        "storage_dir": storage,
        "pool": pool,
        "pool_size": 8,
    }
    return Executor(**arguments) if kind == "python" else NativeExecutor(**arguments)


async def run_individually(
    kind: str, chain: Chain, sink: Sink, storage: Path
) -> weakref.ref[Executor | NativeExecutor]:
    with ThreadPoolExecutor(8) as pool:
        executor = executor_of(kind, chain, sink, pool, storage)
        node = chain.nodes[NodeId("warns")]
        assert isinstance(node, FunctionNode)
        output = await executor.process_regular_node(node)
        assert output.output == [21]
        await executor.flush_broadcasts()
        return weakref.ref(executor)


def warnings_of(events: list[dict[str, Any]]) -> list[str]:
    return [e["data"]["message"] for e in events if e["event"] == "execution-warning"]


@pytest.mark.parametrize("kind", KINDS)
def test_a_warning_reaches_the_app_in_a_run_and_the_run_goes_on(
    tmp_path: Path, kind: str
):
    threads: list[int] = []
    result = execute(kind, warning_chain(threads), storage=tmp_path)
    assert result.error is None
    assert threads != [threading.get_ident()]
    assert warnings_of(result.events) == ["This model is wrong."]


@pytest.mark.parametrize("kind", KINDS)
def test_a_warning_reaches_the_app_in_an_individual_run(tmp_path: Path, kind: str):
    sink = Sink()
    asyncio.run(run_individually(kind, warning_chain([]), sink, tmp_path))
    assert warnings_of(sink.events) == ["This model is wrong."]


@pytest.mark.parametrize("kind", KINDS)
def test_a_context_that_warns_keeps_no_executor_alive(tmp_path: Path, kind: str):
    # server.py collects nothing after a run, so a cycle would keep its outputs.
    gc.collect()
    gc.disable()
    try:
        reference = asyncio.run(
            run_individually(kind, warning_chain([]), Sink(), tmp_path)
        )
        assert reference() is None
    finally:
        gc.enable()
