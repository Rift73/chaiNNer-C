"""Stop during a long node, through the server as the UI stops a run: the /run request dropped, then /kill, while a
node's body (a long Python body or native kernel without an abort check, e.g. PyTorch's Upscale Image on one tile)
still runs. On the native engine /kill answers at once (the host restarts a worker whose /kill takes 3 s); the node's
thread finishes in the background, its result dropped (no event, no memo entry) and its run's cleanups run once
after it; the next /run is accepted at once and starts its nodes once that node and its cleanups ended. The Python
executor (process.py's await_owned_node) waits for the node: its /kill latency follows the node and is only
recorded here."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"
TESTS = Path(__file__).resolve().parent
TOOLS = BACKEND.parents[1] / "native" / "tools"
SLOW_SECONDS = 5.0
KILL_LIMIT = 1.0  # the host's /kill timeout is 3 s (server_host.kill)

DRIVER = """
import asyncio, gc, os, sys, threading, time, weakref
from pathlib import Path
from types import SimpleNamespace
sys.path[:0] = [os.getcwd(), os.environ["TOOLS"], os.environ["TESTS"]]
sys.modules["__main__"].__file__ = os.path.abspath("run.py")
import json
import api
api.registry.get_package = lambda _: SimpleNamespace(id="test")
import native_executor
import server
from nodes.impl.native_graph import graph
from test_engine_scenarios import Builder

# The handlers run without a started Sanic app: its loop is the running one.
type(server.app).loop = property(lambda self: asyncio.get_running_loop())
SLOW = float(os.environ["SLOW_SECONDS"])
# The worker case's nodes are memo-eligible (as Upscale Image's upstream may be): a stopped node's result is not kept.
native_executor.MEMO_PYTHON_NODES |= {"chainner:utility:test_source", "chainner:utility:test_slow"}
clock = time.perf_counter


class Output:
    \"\"\"The slow node's output, whose life the test follows.\"\"\"


class Recorder:
    def __init__(self):
        self.events = []

    def put(self, event):
        self.events.append((clock(), event["event"], (event.get("data") or {}).get("nodeId")))


def stopped_chain(where, trace, alive, started):
    b = Builder()

    def source(context, x):
        context.add_cleanup(lambda: trace.append(("chain cleanup", clock())), after="chain")
        return x

    def slow(*arguments):
        started.set()
        if where == "driver":
            context = arguments[0]
            context.add_cleanup(lambda: trace.append(("node cleanup", clock())), after="node")
        time.sleep(SLOW)
        output = Output()
        alive.append(weakref.ref(output))
        trace.append(("slow returned", clock()))
        return None if where == "driver" else output

    if where == "worker":
        # Pure (and memo-eligible below): it runs on an engine worker, as Upscale Image does.
        b.node("source", lambda x: x)
        b.value("source", 0, 1)
        b.node("slow", slow)
        b.edge("source", 0, "slow", 0)
        b.node("sink", lambda x: trace.append(("sink", clock())), outputs=0, side_effects=True)
        b.edge("slow", 0, "sink", 0)
    elif where == "lazy":
        # As Save Image reads its image: a side-effect node already running reads the slow node through a lazy
        # input, which an engine worker computes; Stop ends the read, so the sink never gets the value.
        def sink(lazy):
            lazy.value
            trace.append(("sink", clock()))

        b.node("source", lambda x: x)
        b.value("source", 0, 1)
        b.node("slow", slow)
        b.edge("source", 0, "slow", 0)
        b.node("sink", sink, outputs=0, side_effects=True, lazy=(0,))
        b.edge("slow", 0, "sink", 0)
    else:
        # A side-effect node with a context: it runs on the run's driver.
        b.node("source", source, context=True)
        b.value("source", 0, 1)
        b.node("slow", slow, outputs=0, side_effects=True, context=True)
        b.edge("source", 0, "slow", 0)
    return b.chain


def next_chain(trace):
    b = Builder()
    b.node("quick", lambda x: trace.append(("next run's node", clock())), outputs=0, side_effects=True)
    b.value("quick", 0, 1)
    return b.chain


async def no_wait(*_arguments, **_keywords):
    return None


def ending_runs():
    # The native runs that ended while a node still ran (none before that fix).
    return getattr(native_executor, "_ENDING_RUNS", set())


async def stop(where, drop):
    ctx = server.app.ctx
    ctx.__dict__["storage_dir"] = Path(os.environ["WORK"]) / "storage"
    trace, alive, started = [], [], threading.Event()
    chains = [stopped_chain(where, trace, alive, started), next_chain(trace)]
    memo_before = graph().memo_stats()["entries"]
    server.parse_json = lambda _data: chains.pop(0)
    server.wait_for_nodes = no_wait
    stopped_events = Recorder()
    ctx.__dict__["queue"] = stopped_events
    request = SimpleNamespace(app=server.app, json={"data": [], "options": {}, "sendBroadcastData": True})
    run = asyncio.ensure_future(server.run(request))
    while not started.is_set():
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.3)
    executor = weakref.ref(ctx.executor)
    if drop:
        run.cancel()  # the UI's Stop drops its /run request, then sends /kill
    began = clock()
    kill = await server.kill(SimpleNamespace(app=server.app))
    killed = clock()
    kill_seconds = killed - began
    if drop:
        try:
            await run
            outcome = "returned"
        except asyncio.CancelledError:
            outcome = "cancelled"
    else:
        answer = await run
        outcome = [answer.status, json.loads(answer.body).get("type")]
    run_seconds = clock() - began
    executor_cleared = ctx.executor is None
    # The next /run, at once: accepted, its nodes after the stopped run's node and cleanups.
    next_events = Recorder()
    ctx.__dict__["queue"] = next_events
    following = await server.run(SimpleNamespace(app=server.app, json=request.json))
    await asyncio.sleep(0.2)
    for _ in range(50):
        if not ending_runs():
            break
        await asyncio.sleep(0.1)
    gc.collect()
    events = [e for e in stopped_events.events if e[0] > killed]
    return {
        "kill": [kill.status, round(kill_seconds, 3)],
        "outcome": outcome,
        "run_seconds": round(run_seconds, 3),
        "executor_cleared": executor_cleared,
        "next": [following.status, json.loads(following.body).get("type")],
        "next_finished": [e[2] for e in next_events.events if e[1] == "node-finish"],
        "late_events": [e[1:] for e in events],
        "trace": [(name, round(at - killed, 3)) for name, at in trace],
        "ending_runs": len(ending_runs()),
        "alive": [executor() is not None] + [ref() is not None for ref in alive],
        "memo_entries": graph().memo_stats()["entries"] - memo_before,
    }


results = {}
for kind in ("python", "native"):
    os.environ["CHAINNER_C_EXECUTOR"] = kind
    for where, drop in (("worker", True), ("lazy", True), ("driver", True), ("driver", False)):
        results[f"{kind} {where} {'dropped' if drop else 'kept'}"] = asyncio.run(stop(where, drop))
print(json.dumps(results))
"""


@pytest.fixture(scope="module")
def stops(tmp_path_factory) -> dict[str, Any]:
    work = tmp_path_factory.mktemp("stop")
    environment = dict(
        os.environ,
        WORK=str(work),
        TOOLS=str(TOOLS),
        TESTS=str(TESTS),
        SLOW_SECONDS=str(SLOW_SECONDS),
        CUDA_VISIBLE_DEVICES="-1",
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(DRIVER)],
        cwd=BACKEND,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout[-3000:], result.stderr[-6000:])
    stops = json.loads(result.stdout.strip().splitlines()[-1])
    print(json.dumps(stops, indent=1))
    return stops


CASES = ["worker dropped", "lazy dropped", "driver dropped", "driver kept"]


@pytest.mark.parametrize(
    ("kind", "case"), [(kind, case) for kind in ("native", "python") for case in CASES]
)
def test_stop_during_a_long_node(stops, kind, case):
    stop = stops[f"{kind} {case}"]
    trace = dict(stop["trace"])
    # Stopped: the request ends as the UI expects (dropped: cancelled; kept: the success Python's Aborted gives).
    assert stop["kill"][0] == 200, stop
    assert stop["outcome"] == ("cancelled" if case.endswith("dropped") else [200, "success"]), stop
    assert stop["executor_cleared"], stop
    if kind == "native":
        # At once, the node still running: well within the host's 3 s (no worker restart).
        assert stop["kill"][1] < KILL_LIMIT, stop
        assert stop["run_seconds"] < KILL_LIMIT, stop
        assert trace["slow returned"] > 0, stop
    # The stopped node's thread finished; its run sent nothing after /kill and ran no later node.
    assert "slow returned" in trace and "sink" not in trace, stop
    assert stop["late_events"] == [], stop
    # Its cleanups ran once each, after the node returned.
    names = [name for name, _ in stop["trace"]]
    if case.startswith("driver"):
        assert names.count("chain cleanup") == 1, stop
        assert trace["chain cleanup"] >= trace["slow returned"], stop
        assert names.count("node cleanup") == 1, stop
        assert trace["node cleanup"] >= trace["slow returned"], stop
    # The next /run: accepted at once by the same server, its node after the stopped run's node and cleanups.
    assert stop["next"] == [200, "success"] and stop["next_finished"] == ["quick"], stop
    assert names.count("next run's node") == 1, stop
    assert trace["next run's node"] >= max(at for name, at in stop["trace"] if name != "next run's node"), stop
    # Nothing of the stopped run is left: its executor, its node's output, its engine's background end.
    assert stop["ending_runs"] == 0, stop
    # The memo keeps the computed source, not the stopped node (the Python executor has no memo; the lazy case's
    # source is the worker case's entry).
    assert stop["memo_entries"] == (1 if (kind, case) == ("native", "worker dropped") else 0), stop
    assert stop["alive"] == [False] * len(stop["alive"]), stop


# The delta review of the change above: a run stopped before it started (behind a stopped run's node) leaks nothing
# (H1); a stopped run's node that fails later is logged with its traceback (M1); a second Stop while a stopped node
# still runs waits for it, so a node that never returns exceeds the host's /kill timeout and the host restarts the
# worker, and a run waiting behind it says why (M2); the server's CUDA check and collection wait for that node (M3).
# Later: a node that stops itself after Stop ends its run as stopped, with nothing logged.
REVIEW = DRIVER.split("results = {}", maxsplit=1)[0] + """
import native_executor as ne
from process import Aborted

errors, gpu = [], []
ne.logger = SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None, warning=lambda *a, **k: None,
                            error=lambda *a, **k: errors.append((a[0] % a[1:], k.get("exc_info") is not None)))
server.cuda_context_broken = lambda: gpu.append(("cuda check", clock())) or False
server.return_device_memory = lambda collect=False: gpu.append(("return memory", clock()))


def one_node(body, side_effects=True):
    b = Builder()
    b.node("slow", body, outputs=0, side_effects=side_effects)
    b.value("slow", 0, 1)
    return b.chain


async def unstarted(stop):
    loop = asyncio.get_running_loop()
    payload = Output()
    b = Builder()
    b.node("src", lambda x: x)
    b.value("src", 0, payload)
    b.node("sink", lambda x: None, outputs=0, side_effects=True)
    b.edge("src", 0, "sink", 0)
    blocker = loop.create_future()
    ending = loop.create_task(asyncio.wait((blocker,)))
    # A stopped run still ending (a set of tasks before the review's fixes).
    if isinstance(ne._ENDING_RUNS, dict):
        ne._ENDING_RUNS[ending] = "slow"
    else:
        ne._ENDING_RUNS.add(ending)
    executor = ne.NativeExecutor(id="x", chain=b.chain, send_broadcast_data=False,
                                 options=api.ExecutionOptions.parse({}), loop=loop, queue=Recorder(),
                                 storage_dir=Path(os.environ["WORK"]))
    refs = [weakref.ref(executor), weakref.ref(payload)]
    del payload, b
    task = asyncio.ensure_future(executor.run())
    await asyncio.sleep(0.1)
    executor.kill() if stop == "kill" else task.cancel()
    try:
        await task
    except (Aborted, asyncio.CancelledError):
        pass
    del executor, task
    blocker.set_result(None)
    await ending
    if isinstance(ne._ENDING_RUNS, dict):
        ne._ENDING_RUNS.pop(ending, None)
    else:
        ne._ENDING_RUNS.discard(ending)
    gc.collect()
    return [ref() is not None for ref in refs]


async def through_server(chain, then, pressed=None):
    ctx = server.app.ctx
    ctx.__dict__["storage_dir"] = Path(os.environ["WORK"]) / "storage"
    ctx.__dict__["queue"] = Recorder()
    server.parse_json = lambda _data: chain
    server.wait_for_nodes = no_wait
    request = SimpleNamespace(app=server.app, json={"data": [], "options": {}, "sendBroadcastData": True})
    run = asyncio.ensure_future(server.run(request))
    while not ctx.executor:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.3)
    run.cancel()
    if pressed is not None:
        pressed.set()
    began = clock()
    await server.kill(SimpleNamespace(app=server.app))
    first = clock() - began
    try:
        await run
    except asyncio.CancelledError:
        pass
    return {"first_kill": round(first, 3), **await then(ctx, request)}


async def failing():
    def body(x):
        time.sleep(1.5)
        raise ValueError("the node failed late")

    async def then(ctx, request):
        while ending_runs():
            await asyncio.sleep(0.05)
        return {"errors": errors[:]}

    return await through_server(one_node(body), then)


async def hung():
    release = threading.Event()

    def body(x):
        release.wait()

    async def then(ctx, request):
        result = {}
        try:
            await asyncio.wait_for(server.kill(SimpleNamespace(app=server.app)), 3.2)
            result["second_kill"] = "returned"
        except asyncio.TimeoutError:
            result["second_kill"] = "waiting"
        waiting = Recorder()
        ctx.__dict__["queue"] = waiting
        server.parse_json = lambda _data: one_node(lambda x: None)
        behind = asyncio.ensure_future(server.run(SimpleNamespace(app=server.app, json=request.json)))
        await asyncio.sleep(0.3)
        try:
            await asyncio.wait_for(server.kill(SimpleNamespace(app=server.app)), 1.5)
            result["kill_behind"] = "returned"
        except asyncio.TimeoutError:
            result["kill_behind"] = "waiting"
        result["warning"] = [e for e in waiting.events if e[1] == "execution-warning"] != []
        release.set()
        await behind
        while ending_runs():
            await asyncio.sleep(0.05)
        return result

    return await through_server(one_node(body), then)


async def gpu_checks():
    returned = []

    def body(x):
        time.sleep(1.5)
        returned.append(clock())

    async def then(ctx, request):
        # Node clears (a deleted node) while the stopped node still runs: their memory return waits for it too.
        server.return_device_memory_after_burst(ctx)
        await asyncio.sleep(0.2)
        while ending_runs():
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.3)
        return {"early": [name for name, at in gpu if at < returned[0]], "after": [name for name, at in gpu]}

    return await through_server(one_node(body), then)


async def self_stopped(delay):
    # A node that stops itself (its context's check_aborted raises api.Aborted, as Upscale Image's tile and block
    # checks do) this long after Stop: within the request's grace or after it, in the background.
    pressed, unretrieved = threading.Event(), []
    asyncio.get_running_loop().set_exception_handler(lambda _loop, context: unretrieved.append(context["message"]))

    def body(x):
        pressed.wait()
        time.sleep(delay)
        raise api.Aborted()

    async def then(ctx, request):
        while ending_runs():
            await asyncio.sleep(0.05)
        gc.collect()
        return {"errors": errors[:], "unretrieved": unretrieved}

    errors.clear()
    return await through_server(one_node(body), then, pressed)


async def lazy_release():
    # The Python executor: Stop ends the sink's lazy read while slow, which computes it, runs on. The run says it
    # waits for slow (its log line releases slow); a run that ends without waiting has the server's memory return
    # release it.
    import process

    loop = asyncio.get_running_loop()
    started, release, done, returned, memory = (threading.Event(), threading.Event(), threading.Event(), [],
                                                 asyncio.Event())
    info, return_memory = process.logger.info, server.return_device_memory

    def logged(message, *args, **kwargs):
        if "lazy input" in message:
            release.set()
        info(message, *args, **kwargs)

    def returning(collect=False):
        gpu.append(("return memory", clock()))
        release.set()
        memory.set()

    def slow(x):
        started.set()
        release.wait()
        returned.append(clock())
        done.set()
        return x

    b = Builder()
    b.node("source", lambda x: x)
    b.value("source", 0, 1)
    b.node("slow", slow)
    b.edge("source", 0, "slow", 0)
    b.node("sink", lambda lazy: lazy.value, outputs=0, side_effects=True, lazy=(0,))
    b.edge("slow", 0, "sink", 0)
    os.environ["CHAINNER_C_EXECUTOR"] = "python"
    process.logger.info, server.return_device_memory = logged, returning
    gpu.clear()
    try:
        ctx = server.app.ctx
        ctx.__dict__["storage_dir"] = Path(os.environ["WORK"]) / "storage"
        ctx.__dict__["queue"] = Recorder()
        server.parse_json = lambda _data: b.chain
        server.wait_for_nodes = no_wait
        request = SimpleNamespace(app=server.app, json={"data": [], "options": {}, "sendBroadcastData": True})
        run = asyncio.ensure_future(server.run(request))
        await loop.run_in_executor(None, started.wait)
        run.cancel()
        await server.kill(SimpleNamespace(app=server.app))
        try:
            await run
        except asyncio.CancelledError:
            pass
        await memory.wait()
        await loop.run_in_executor(None, done.wait)
    finally:
        process.logger.info, server.return_device_memory = info, return_memory
        os.environ["CHAINNER_C_EXECUTOR"] = "native"
    return {"slow returned": returned[0], "returned memory": [at for name, at in gpu if name == "return memory"]}


os.environ["CHAINNER_C_EXECUTOR"] = "native"
results = {"unstarted kill": asyncio.run(unstarted("kill")), "unstarted cancel": asyncio.run(unstarted("cancel"))}
results["failing"] = asyncio.run(failing())
gpu.clear()
results["gpu"] = asyncio.run(gpu_checks())
results["hung"] = asyncio.run(hung())
results["self-stopped within the grace"] = asyncio.run(self_stopped(0.1))
results["self-stopped after the grace"] = asyncio.run(self_stopped(1.0))
results["python lazy release"] = asyncio.run(lazy_release())
print(json.dumps(results))
"""


@pytest.fixture(scope="module")
def review(tmp_path_factory) -> dict[str, Any]:
    work = tmp_path_factory.mktemp("review")
    environment = dict(os.environ, WORK=str(work), TOOLS=str(TOOLS), TESTS=str(TESTS), SLOW_SECONDS="1",
                       CUDA_VISIBLE_DEVICES="-1")
    result = subprocess.run([sys.executable, "-B", "-c", textwrap.dedent(REVIEW)], cwd=BACKEND, env=environment,
                            capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, (result.returncode, result.stdout[-3000:], result.stderr[-6000:])
    found = json.loads(result.stdout.strip().splitlines()[-1])
    print(json.dumps(found, indent=1))
    return found


@pytest.mark.parametrize("stop", ["kill", "cancel"])
def test_a_run_stopped_before_it_started_is_freed(review, stop):
    # Its executor and its plan's constants (a previewed output, a model) are not kept by its engine.
    assert review[f"unstarted {stop}"] == [False, False]


def test_a_stopped_nodes_late_error_is_logged(review):
    found = review["failing"]
    assert found["first_kill"] < KILL_LIMIT, found
    assert any("the node failed late" in message and traceback for message, traceback in found["errors"]), found


def test_a_second_stop_waits_for_a_node_that_never_returns(review):
    found = review["hung"]
    assert found["first_kill"] < KILL_LIMIT, found
    # Past the host's 3 s: it restarts the worker, as before the prompt Stop.
    assert found["second_kill"] == "waiting", found
    assert found["kill_behind"] == "waiting", found
    assert found["warning"], found


@pytest.mark.parametrize("when", ["within the grace", "after the grace"])
def test_a_node_that_stops_itself_ends_its_run_as_stopped(review, when):
    # Not a failed node (NodeExecutionError, logged as an error), nor an error asyncio reports as never retrieved.
    found = review[f"self-stopped {when}"]
    assert found["first_kill"] < KILL_LIMIT, found
    assert found["errors"] == [] and found["unretrieved"] == [], found


def test_cuda_checks_wait_for_a_stopped_node(review):
    found = review["gpu"]
    assert found["first_kill"] < KILL_LIMIT, found
    assert found["early"] == [] and "return memory" in found["after"], found


def test_python_executor_returns_memory_after_a_stopped_lazy_reads_node(review):
    # Its run ends after the node computing the lazy input Stop stopped reading, so the server's collection and
    # memory return come after that node (before: during it, so its memory stayed reserved after it).
    found = review["python lazy release"]
    assert len(found["returned memory"]) == 1, found
    assert found["slow returned"] < found["returned memory"][0], found
