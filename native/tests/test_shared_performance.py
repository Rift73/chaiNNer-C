"""Real graph, ownership and pixel contracts for shared executor optimization."""

from __future__ import annotations

import asyncio
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from test_generator_lifecycle import EventCallback, Scenario, make_node

import process
from api import ExecutionOptions, NodeId, OutputId
from chain.chain import Chain, Edge, EdgeSource, EdgeTarget
from nodes.impl.image_utils import normalize
from nodes.impl.native_buffers import normalized_readonly
from nodes.properties.outputs.numpy_outputs import ImageOutput
from process import ExecutionId, Executor


def connect(chain, source, target, index):
    chain.add_edge(
        Edge(EdgeSource(NodeId(source), OutputId(0)), EdgeTarget(NodeId(target), index))
    )


def pure(node):
    object.__setattr__(node.data, "side_effects", False)
    object.__setattr__(node.data, "schema_id", f"chainner:image:{node.id}")
    return node


def executor(chain, pool, tmp_path, monkeypatch, events):
    monkeypatch.setattr(
        process.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )
    return Executor(
        ExecutionId("shared-perf"),
        chain,
        True,
        ExecutionOptions({}),
        asyncio.get_running_loop(),
        EventCallback(events.append),
        pool,
        tmp_path,
    )


def test_independent_cpu_branches_overlap_and_common_input_executes_once(
    monkeypatch, tmp_path
):
    chain, counts, events = Chain(), Counter(), []
    gate = threading.Barrier(2)
    active = {}

    def source():
        counts["source"] += 1
        return 17

    def branch(value):
        counts["branch"] += 1
        gate.wait(5)  # Deadlocks/fails with the original serial input resolution.
        return value + 1

    def sink(a, b):
        assert (a, b) == (18, 18)
        # Consuming a freshly computed output must count, too.
        assert not active["executor"].node_cache.has("source")
        assert not active["executor"].node_cache.has("left")
        assert not active["executor"].node_cache.has("right")
        counts["sink"] += 1

    for name, body, inputs, outputs in (
        ("source", source, 0, 1),
        ("left", branch, 1, 1),
        ("right", branch, 1, 1),
        ("sink", sink, 2, 0),
    ):
        node = make_node(name, "regularNode", body, inputs, outputs)
        chain.add_node(node if name == "sink" else pure(node))
    connect(chain, "source", "left", 0)
    connect(chain, "source", "right", 0)
    connect(chain, "left", "sink", 0)
    connect(chain, "right", "sink", 1)

    async def run():
        with ThreadPoolExecutor(max_workers=4) as pool:
            active["executor"] = executor(chain, pool, tmp_path, monkeypatch, events)
            await asyncio.wait_for(active["executor"].run(), 10)

    asyncio.run(run())
    assert counts == {"source": 1, "branch": 2, "sink": 1}


@pytest.mark.parametrize("reverse", [False, True])
def test_side_effect_root_also_consumed_by_root_runs_once(
    reverse, monkeypatch, tmp_path
):
    chain, calls = Chain(), []
    a = make_node("a", "regularNode", lambda: (calls.append("a"), 9)[1], 0, 1)
    b = make_node("b", "regularNode", lambda value: calls.append(("b", value)), 1, 0)
    for node in [b, a] if reverse else [a, b]:
        chain.add_node(node)
    connect(chain, "a", "b", 0)

    async def run():
        with ThreadPoolExecutor(max_workers=2) as pool:
            await executor(chain, pool, tmp_path, monkeypatch, []).run()

    asyncio.run(run())
    assert calls == ["a", ("b", 9)]


def test_same_schema_nodes_have_distinct_context_and_cleanup(monkeypatch, tmp_path):
    chain, contexts, cleanups = Chain(), [], []

    def run_node(context):
        contexts.append(context)
        context.add_cleanup(lambda: cleanups.append(context))

    for name in ("one", "two"):
        node = make_node(name, "regularNode", run_node, 0, 0)
        object.__setattr__(node.data, "schema_id", "chainner:utility:same")
        object.__setattr__(node.data, "node_context", True)
        chain.add_node(node)

    async def run():
        with ThreadPoolExecutor(max_workers=2) as pool:
            await executor(chain, pool, tmp_path, monkeypatch, []).run()

    asyncio.run(run())
    assert len(contexts) == 2 and contexts[0] is not contexts[1]
    assert set(cleanups) == set(contexts)


def test_generator_and_collector_leave_event_loop_responsive(monkeypatch, tmp_path):
    scenario = Scenario(monkeypatch, tmp_path, [[1, 2, 3]], collect=True)
    threads, heartbeats = [], []

    async def run():
        loop = asyncio.get_running_loop()
        main_thread = threading.get_ident()

        def blocking_boundary(*_):
            threads.append(threading.get_ident())
            released = threading.Event()

            def heartbeat():
                heartbeats.append(True)
                released.set()

            loop.call_soon_threadsafe(heartbeat)
            assert released.wait(5), "Blocking operation ran on the event loop"

        scenario.on_yield = blocking_boundary
        with ThreadPoolExecutor(max_workers=2) as pool:
            await scenario.new_executor(pool).run()
        assert all(thread != main_thread for thread in threads)

    asyncio.run(run())
    assert len(heartbeats) == 3
    assert scenario.collected == [1, 2, 3]


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_normalized_immutable_image_storage_is_borrowed(channels):
    value = (
        np.linspace(0, 1, 60 * channels, dtype=np.float32)
        .reshape(6, 10, channels)
        .copy()
    )
    value = ImageOutput().enforce(value)
    result = ImageOutput().enforce(value)
    assert np.shares_memory(value, result)
    assert not result.flags.writeable
    assert result.tobytes() == normalize(value).tobytes()


@pytest.mark.parametrize(
    "bits",
    [
        0,
        0x80000000,
        1,
        0x80000001,
        0x00800000,
        0x3F000000,
        0x3F800000,
        0x3F800001,
        0xBF000000,
        0x7F800000,
        0xFF800000,
        0x7FC01234,
        0xFFC01234,
    ],
)
def test_image_enforcement_is_bit_exact_at_numeric_boundaries(bits):
    value = np.full((3, 4, 3), bits, dtype=np.uint32).view(np.float32)
    assert value.base is not None
    value.base.flags.writeable = False
    value.flags.writeable = False
    expected = normalize(value)
    actual = ImageOutput().enforce(value)
    assert actual.tobytes() == expected.tobytes()
    if bits not in (0, 0x00800000, 0x3F000000, 0x3F800000):
        assert not np.shares_memory(value, actual)


@pytest.mark.parametrize(
    "kind", ["mutable", "readonly_alias", "strided", "float64", "uint8", "subclass"]
)
def test_mutable_or_special_images_keep_original_copy_contract(kind):
    source = np.full((8, 10, 3), 0.5, np.float32)
    value = source
    if kind == "readonly_alias":
        value = source.view()
        value.flags.writeable = False
    elif kind == "strided":
        value = source[:, ::2]
    elif kind in ("float64", "uint8"):
        value = source.astype(kind)
    elif kind == "subclass":

        class Image(np.ndarray):
            pass

        value = source.view(Image)
    assert normalized_readonly(value) is None
    actual = ImageOutput().enforce(value)
    assert not np.shares_memory(value, actual)
    assert actual.tobytes() == normalize(value).tobytes()
    assert not actual.flags.writeable
