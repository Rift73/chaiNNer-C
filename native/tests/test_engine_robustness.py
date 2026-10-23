"""The native run engine's robustness under concurrency (the review of the engine at 69d3295b, review-engine.md):
each test is built from the reviewer's repro. A loop that ends while its window's jobs are in flight, Pause then
Stop, an executor's Python references after it closes, a value a native entry point cannot take, the memory of
many static sinks, Into Directory in the item window. Look-aheads through serial nodes and failing runs' prefetched
branches are differential scenarios in test_engine_scenarios.py."""

from __future__ import annotations

import dataclasses
import gc
import importlib
import math
import os
import subprocess
import sys
import textwrap
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from engine_differential import execute, load_registry_errors, mismatches
from test_engine_scenarios import Builder, assert_same, items, run_both

import api
import native_executor
from api import NodeId
from chain.chain import Chain, Edge, EdgeSource, EdgeTarget, FunctionNode

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"
TESTS = Path(__file__).resolve().parent
TOOLS = BACKEND.parents[1] / "native" / "tools"


@pytest.fixture(autouse=True)
def test_package(monkeypatch):
    # Test nodes belong to no package: both executors read their settings from a stand-in.
    monkeypatch.setattr(
        api.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )


def run_script(script: str, threads: int, tmp_path: Path, **environment: str) -> str:
    """A script in a fresh interpreter whose engine has the given number of workers (read once per process)."""
    env = dict(
        os.environ,
        CHAINNER_C_ENGINE_THREADS=str(threads),
        CUDA_VISIBLE_DEVICES="-1",
        TOOLS=str(TOOLS),
        TESTS=str(TESTS),
        WORK=str(tmp_path),
        **environment,
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(script)],
        cwd=BACKEND,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (
        result.returncode,
        result.stdout[-2000:],
        result.stderr[-4000:],
    )
    return result.stdout.strip().splitlines()[-1]


PRELUDE = """
import os, sys, time
from pathlib import Path
from types import SimpleNamespace
sys.path[:0] = [os.environ["TOOLS"], os.environ["TESTS"]]
import api
api.registry.get_package = lambda _: SimpleNamespace(id="test")
from api import Generator
from engine_differential import execute
from nodes.impl.item_window import register_phases
from test_engine_scenarios import Builder


class Items:
    # A split iterator (Load Images' protocol): describe names an item, materialize loads it in delay(token) seconds;
    # a corrupt item is an error value.
    def __init__(self, count, delay, corrupt):
        self.count, self.delay, self.corrupt, self.next = count, delay, corrupt, 0

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def describe(self):
        if self.next >= self.count:
            raise StopIteration
        self.next += 1
        return self.next - 1

    def materialize(self, token):
        time.sleep(self.delay(token))
        return ValueError("item 3 is corrupt") if token == self.corrupt else token
"""

LOOP_END_RACE = (
    PRELUDE
    + """
fail_at = [20]


def commit(enforced, prepared):
    if prepared.result() == fail_at[0]:
        raise ValueError("the sink fails at item 2")


register_phases("chainner:utility:test_phased", lambda enforced: enforced[0], commit)


def chain(sink, corrupt, trace):
    b = Builder()
    # Later items' materializes are in flight while the loop ends.
    items = lambda: Items(40, lambda token: 0.01, corrupt)
    b.node("numbers", lambda: Generator(items, 40, source_paths=()), kind="generator", inputs=0)
    b.node("scale", lambda x: x * 10)

    def body(x):
        if x == fail_at[0]:
            raise ValueError("the sink fails at item 2")
        trace.append(x)

    b.node(sink, body, outputs=0, side_effects=True)
    b.edge("numbers", 0, "scale", 0)
    b.edge("scale", 0, sink, 0)
    return b.chain


bad = []
# A sink failing at item 2 (plain or phased, as Save Image), or Load Images' corrupt item 3 (the sink never fails).
cases = [("plain", -1, 20, "the sink fails at item 2", [0, 10]), ("phased", -1, 20, "the sink fails at item 2", None),
         ("plain", 3, -1, "item 3 is corrupt", [0, 10, 20]), ("phased", 3, -1, "item 3 is corrupt", None)]
for i in range(int(os.environ["ITERATIONS"])):
    for sink, corrupt, failing, expected, kept in cases:
        fail_at[0] = failing
        trace = []
        result = execute("native", chain(sink, corrupt, trace), storage=Path(os.environ["WORK"]))
        scale = [e["event"] for e in result.events if (e.get("data") or {}).get("nodeId") == "scale"]
        if (expected not in str(result.error) or scale.count("node-start") != scale.count("node-finish")
                or (kept is not None and trace != kept)):
            bad.append((i, sink, corrupt, repr(result.error), scale.count("node-start"), trace))
print(len(bad), bad[:3])
"""
)


def test_a_loop_ends_while_its_window_jobs_finish(tmp_path):
    # C1: a fail-fast error ends a K > 1 loop while materializes and prepares are in flight; the jobs finishing
    # during the loop's end queue nothing and reach no closed slot (a crash, or a node run with a None input).
    assert run_script(LOOP_END_RACE, 2, tmp_path, ITERATIONS="40") == "0 []"
    assert run_script(LOOP_END_RACE, 4, tmp_path, ITERATIONS="15") == "0 []"


PAUSE_THEN_STOP = (
    PRELUDE
    + """
register_phases("chainner:utility:test_paused", lambda enforced: enforced[0], lambda enforced, prepared: None)
whole = []  # the sink's body: run whole at item 0 (no window yet), never after Stop
b = Builder()
# On one worker, item 1's prepare is queued behind item 2's long materialize: the driver waits for it at item 1.
items = lambda: Items(12, lambda token: 1.0 if token == 2 else 0.05, -1)
b.node("numbers", lambda: Generator(items, 12, source_paths=()), kind="generator", inputs=0)
b.node("paused", lambda x: whole.append(x), outputs=0, side_effects=True)
b.edge("numbers", 0, "paused", 0)


def during(executor):
    # Paused while that materialize runs (the worker then takes no job), stopped once it ended.
    time.sleep(0.5)
    executor.pause()
    time.sleep(0.9)
    executor.kill()


result = execute("native", b.chain, storage=Path(os.environ["WORK"]), during=during)
print(type(result.error).__name__, whole)
"""
)


def test_pause_then_stop_ends_the_run(tmp_path):
    # H1: the driver waits for a sink's prepare queued behind a long materialize on a single worker; Pause, then
    # Stop: the run ends with Aborted (the subprocess's timeout catches a hang), and the sink whose prepare Stop
    # dropped does not run whole either.
    assert run_script(PAUSE_THEN_STOP, 1, tmp_path) == "Aborted [0]"


def test_closed_executors_are_freed(tmp_path):
    # C2: the engine's references to Python objects (hooks, cells, constants) are dropped at close: previews and runs
    # leave no executor or output alive.
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from events import EventConsumer
    from native_executor import NativeExecutor
    from process import ExecutionId

    class Output:
        """An output object whose life the test follows."""

    class Sink(EventConsumer):
        def put(self, event) -> None:
            pass

    alive: list[weakref.ref[object]] = []

    def made(_: object) -> Output:
        output = Output()
        alive.append(weakref.ref(output))
        return output

    async def preview(storage: Path) -> None:
        b = Builder()
        b.node("single", made)
        b.value("single", 0, 1)
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(2)
        executor = NativeExecutor(
            id=ExecutionId("individual-executor single"),
            chain=b.chain,
            send_broadcast_data=True,
            options=api.ExecutionOptions.parse({}),
            loop=loop,
            queue=Sink(),
            storage_dir=storage,
            pool=pool,
            pool_size=8,
        )
        alive.append(weakref.ref(executor))
        node = b.chain.nodes[NodeId("single")]
        assert isinstance(node, FunctionNode)
        await executor.process_regular_node(node)
        await executor.flush_broadcasts()
        pool.shutdown()

    for _ in range(3):
        asyncio.run(preview(tmp_path))

    def run_chain(trace: list[object]) -> Chain:
        b = Builder()
        b.node("make", made)
        b.value("make", 0, 1)
        b.node(
            "sink",
            lambda x: trace.append(type(x).__name__),
            outputs=0,
            side_effects=True,
        )
        b.edge("make", 0, "sink", 0)
        return b.chain

    assert execute("native", run_chain([]), storage=tmp_path).error is None
    gc.collect()
    assert len(alive) == 3 * 2 + 1
    assert [ref() for ref in alive] == [None] * len(alive)


def test_closed_executors_are_freed_without_a_collection(tmp_path):
    # server.py never collects after a successful run or preview, so a closed executor and the output an individual
    # run returned must go with their last reference, as process.Executor's do: a reference cycle through the
    # executor kept a preview's output alive after the server's cache dropped it (numpy_pool_growth.py's release
    # check: the first 64 MiB preview stayed after the second, and both after their cache clear).
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from events import EventConsumer
    from native_executor import NativeExecutor
    from process import ExecutionId

    class Output:
        """An output object whose life the test follows."""

    class Sink(EventConsumer):
        def put(self, event) -> None:
            pass

    alive: list[weakref.ref[object]] = []

    def made(_: object) -> Output:
        output = Output()
        alive.append(weakref.ref(output))
        return output

    def chain(side_effects: bool) -> Chain:
        b = Builder()
        b.node("single", made, side_effects=side_effects)
        b.value("single", 0, 1)
        return b.chain

    async def drive(individual: bool) -> None:
        pool = ThreadPoolExecutor(2)
        executor = NativeExecutor(
            id=ExecutionId(
                "individual-executor single" if individual else "main-executor"
            ),
            chain=chain(side_effects=not individual),
            send_broadcast_data=True,
            options=api.ExecutionOptions.parse({}),
            loop=asyncio.get_running_loop(),
            queue=Sink(),
            storage_dir=tmp_path,
            pool=pool,
            pool_size=8,
        )
        alive.append(weakref.ref(executor))
        if individual:
            node = executor.chain.nodes[NodeId("single")]
            assert isinstance(node, FunctionNode)
            await executor.process_regular_node(node)
            await executor.flush_broadcasts()
        else:
            await executor.run()
        pool.shutdown()

    gc.collect()
    gc.disable()
    try:
        for individual in (True, True, False):
            asyncio.run(drive(individual))
        assert len(alive) == 3 * 2
        assert [ref() for ref in alive] == [None] * len(alive)
    finally:
        gc.enable()


@pytest.fixture(scope="module")
def registry():
    main = sys.modules["__main__"]
    previous = getattr(main, "__file__", None)
    main.__file__ = str(BACKEND / "run.py")
    importlib.import_module("packages.chaiNNer_standard")
    errors = load_registry_errors(BACKEND / "server.py")
    assert not errors, [e.module for e in errors]
    yield
    if previous is None:
        del main.__file__
    else:
        main.__file__ = previous


def test_a_value_a_native_node_cannot_take_runs_its_python_body(registry, tmp_path):
    # H2: a 2-channel image (ImageInput.enforce takes it) into native Flip, run on a worker: the engine cannot hold
    # it, so the node's Python body runs, as process.Executor runs it (no process exit).
    def two_channels(trace: list[object]) -> Chain:
        b = Builder()
        b.node("two", lambda x: np.full((8, 8, 2), 0.25, dtype=np.float32))
        b.value("two", 0, 1)
        flip = FunctionNode(NodeId("flip"), "chainner:image:flip")
        b.chain.add_node(flip)
        b.chain.inputs.set(flip.id, flip.data.inputs[1].id, 0)
        b.chain.add_edge(
            Edge(
                EdgeSource(NodeId("two"), api.OutputId(0)),
                EdgeTarget(flip.id, flip.data.inputs[0].id),
            )
        )
        b.node(
            "sink",
            lambda image: trace.append(image.shape),
            outputs=0,
            side_effects=True,
        )
        b.chain.add_edge(
            Edge(
                EdgeSource(flip.id, flip.data.outputs[0].id),
                EdgeTarget(NodeId("sink"), api.InputId(0)),
            )
        )
        return b.chain

    engine = assert_same(two_channels, tmp_path)
    assert engine.error is None


def test_static_sinks_prefetch_only_the_next_sink(tmp_path):
    # H3: sinks outside loops run one after another; the engine computes the pure upstream of the running sink and
    # of the next one only (not every branch's outputs at once).
    made: list[int] = []
    seen: dict[str, list[list[int]]] = {"python": [], "native": []}
    current = ["python"]

    def branches(trace: list[object]) -> Chain:
        b = Builder()
        for i in range(6):
            b.node(f"make{i}", lambda x: (made.append(x), x)[1])
            b.value(f"make{i}", 0, i)

            def sink(x: int) -> None:
                seen[current[0]].append(sorted(made))
                trace.append(x)
                time.sleep(0.02)

            b.node(f"sink{i}", sink, outputs=0, side_effects=True)
            b.edge(f"make{i}", 0, f"sink{i}", 0)
        return b.chain

    for kind in ("python", "native"):
        made.clear()
        current[0] = kind
        assert execute(kind, branches([]), storage=tmp_path).error is None
    for i, computed in enumerate(seen["native"]):
        assert max(computed) <= i + 1, (i, computed)
    chain, oracle, engine, oracle_trace, engine_trace = run_both(branches, tmp_path)
    assert mismatches("branches", chain, oracle, engine, tmp_path) == []
    assert engine_trace == oracle_trace


def test_into_directory_is_never_computed_ahead(tmp_path):
    # M3: Into Directory names a folder earlier items' saves may create: as process.Executor (_NOT_SPECULATED), the
    # window never computes it ahead of its item, so it runs once every earlier item committed.
    def folders(trace: list[object]) -> Chain:
        b = Builder()
        b.node("numbers", lambda: items(list(range(10))), kind="generator", inputs=0)
        committed: list[int] = []
        into = b.node(
            "into", lambda x: (trace.append(("into", x, len(committed))), x)[1]
        )
        into.data = dataclasses.replace(
            into.data, schema_id="chainner:utility:into_directory"
        )
        into.schema_id = into.data.schema_id
        b.node("sink", committed.append, outputs=0, side_effects=True)
        b.edge("numbers", 0, "into", 0)
        b.edge("into", 0, "sink", 0)
        return b.chain

    trace: list[object] = []
    assert execute("native", folders(trace), storage=tmp_path).error is None
    assert trace == [("into", x, x) for x in range(10)]
    assert_same(folders, tmp_path)


IDLE_SERVICES = """
import asyncio, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import psutil
sys.path[:0] = [os.environ["TOOLS"], os.environ["TESTS"]]
import api
api.registry.get_package = lambda _: SimpleNamespace(id="test")
from api import ExecutionOptions, NodeId
from events import EventConsumer
from native_executor import NativeExecutor
from process import ExecutionId
from test_engine_scenarios import Builder


class Sink(EventConsumer):
    def put(self, event):
        pass


async def preview(pool, i):
    b = Builder()
    b.node("single", lambda x: x + 1)
    b.value("single", 0, i)
    executor = NativeExecutor(id=ExecutionId(f"individual-executor {i}"), chain=b.chain, send_broadcast_data=True,
                              options=ExecutionOptions.parse({}), loop=asyncio.get_running_loop(), queue=Sink(),
                              storage_dir=Path(os.environ["WORK"]), pool=pool, pool_size=8)
    await executor.process_regular_node(b.chain.nodes[NodeId("single")])
    await executor.flush_broadcasts()


async def previews(count):
    pool = ThreadPoolExecutor(4)
    await asyncio.gather(*(preview(pool, i) for i in range(count)))
    pool.shutdown()


# Counted outside the event loops, whose default executors (where runs close) end with them.
asyncio.run(previews(1))
before = psutil.Process().num_threads()
asyncio.run(previews(16))
burst = psutil.Process().num_threads()
time.sleep(3)
print(before, burst, psutil.Process().num_threads())
"""


def test_idle_service_threads_end(tmp_path):
    # A burst of previews takes a driver, event pump and broadcaster thread each; idle, they end (here after 1 s).
    before, burst, after = map(
        int,
        run_script(
            IDLE_SERVICES, 2, tmp_path, CHAINNER_C_ENGINE_IDLE_SECONDS="1"
        ).split(),
    )
    assert burst > before + 8
    assert after <= before + 2


PYTHON_THREADS = (
    PRELUDE
    + """
from nodes.impl.native_graph import graph
b = Builder()
for i in range(16):
    b.node(f"n{i}", lambda x: x + 1)
    b.value(f"n{i}", 0, i)
assert execute("native", b.chain, storage=Path(os.environ["WORK"])).error is None
print(graph().python_threads())
"""
)


@pytest.mark.parametrize(
    "value, threads", [("2000", 1024), ("1024", 1024), ("3", 3), ("0", 8)]
)
def test_python_threads_are_at_most_the_slot_limit(tmp_path, value, threads):
    # CHAINNER_C_PYTHON_THREADS above the Python slots' hard limit (1024) counts as that limit; 0 is invalid (the
    # default, 8). The engine runs Python nodes with it.
    assert (
        int(run_script(PYTHON_THREADS, 4, tmp_path, CHAINNER_C_PYTHON_THREADS=value))
        == threads
    )


def test_a_lazy_input_read_after_its_run_fails_cleanly(tmp_path):
    # A node that keeps its lazy input unread and a caller that reads it once the run closed: a Python error, not a
    # read of the closed run's cells.
    kept: list[api.Lazy[object]] = []

    def keeping(trace: list[object]) -> Chain:
        b = Builder()
        b.node("upstream", lambda x: x + 1)
        b.value("upstream", 0, 1)
        b.node(
            "keeper",
            kept.append,
            outputs=0,
            side_effects=True,
            lazy=(0,),
        )
        b.edge("upstream", 0, "keeper", 0)
        return b.chain

    assert execute("native", keeping([]), storage=tmp_path).error is None
    with pytest.raises(RuntimeError, match="after its run ended"):
        _ = kept[0].value


def test_loop_previews_are_rate_limited(registry, tmp_path):
    # ruling R4: a loop's LargeImage previews (View Image of a 1080p image per item, an item every 80 ms) start at most
    # once per 250 ms per node, the first at once, the newest kept; the loop's end sends the last item's.
    def paced(trace: list[object]) -> Chain:
        b = Builder()
        frames: list[object] = [
            np.full((1080, 1920, 3), i / 20, dtype=np.float32) for i in range(20)
        ]
        b.node("frames", lambda: items(frames), kind="generator", inputs=0)
        b.node(
            "pace", lambda context, x: (time.sleep(0.08), x)[1], context=True
        )  # on the driver, item by item
        view = FunctionNode(NodeId("view"), "chainner:image:view")
        b.chain.add_node(view)
        b.edge("frames", 0, "pace", 0)
        b.chain.add_edge(
            Edge(
                EdgeSource(NodeId("pace"), api.OutputId(0)),
                EdgeTarget(view.id, view.data.inputs[0].id),
            )
        )
        return b.chain

    def run(limit: bool) -> tuple[float, list[object]]:
        result = execute("native", paced([]), storage=tmp_path, preview_limit=limit)
        assert result.error is None
        views = [
            e["data"]
            for e in result.events
            if e["event"] == "node-broadcast" and e["data"]["nodeId"] == "view"
        ]
        return result.seconds, views

    seconds, limited = run(True)
    _, unlimited = run(False)
    assert (
        1 <= len(limited) <= math.ceil(seconds / native_executor.PREVIEW_INTERVAL) + 1
    ), (len(limited), seconds)
    assert len(limited) < len(unlimited), (len(limited), len(unlimited))
    assert limited[-1] == unlimited[-1]
