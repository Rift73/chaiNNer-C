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
    for where, drop in (("worker", True), ("driver", True), ("driver", False)):
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


CASES = ["worker dropped", "driver dropped", "driver kept"]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", ["native", "python"])
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
    # The memo keeps the computed source, not the stopped node (the Python executor has no memo).
    assert stop["memo_entries"] == (1 if (kind, case) == ("native", "worker dropped") else 0), stop
    assert stop["alive"] == [False] * len(stop["alive"]), stop
