"""The GPU lease (the installed nightly's private modification) is gone from backend/src.

Each spot it hooked behaves as upstream d56e507f again: the host proxies /run,
/run/individual, /kill and /status to the worker and closes it as upstream does, the
SSE stream sends without acknowledgments, and auto tiling budgets from the total GPU
memory. chaiNNer-C's own change at one of those spots stays: /run/individual responds
without waiting for its broadcasts, as upstream does, and a task releases the NumPy
pool once they are sent. Upstream's sources come from git; nothing here starts a
worker.
"""

import ast
import asyncio
import logging
import subprocess
from types import SimpleNamespace

import pytest
from server_source import BACKEND, ROOT, compiled, definition, source

UPSTREAM = "d56e507f"
LEASE_FILES = [
    "gpu_lease.py",
    "GPU_LEASE.md",
    "tests/test_gpu_lease.py",
    "tests/check_gpu_lease_wsl.py",
]
LEASE_NAMES = [
    "gpu_lease",
    "GpuExecutionGate",
    "gpu_gate",
    "wait_until_sent",
    "drain_execution_events",
]


def upstream_source(relative):
    shown = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{UPSTREAM}:backend/src/{relative}"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return shown.stdout


def here_and_upstream(relative, *path):
    """The definition's normalized source (no comments, one layout) here and upstream."""
    return tuple(
        ast.unparse(definition(ast.parse(text), *path))
        for text in (source(relative), upstream_source(relative))
    )


def test_no_backend_module_references_the_lease():
    assert [f for f in LEASE_FILES if (BACKEND / f).exists()] == []
    found = []
    for path in sorted(BACKEND.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            data = path.read_bytes()
            relative = path.relative_to(BACKEND).as_posix()
            found += [(relative, n) for n in LEASE_NAMES if n.encode() in data]
    assert found == []


def test_run_proxies_without_recycling_the_worker():
    # Review focus 1: a GPU chain's run no longer stops and restarts the worker.
    for handler in ("run", "run_individual"):
        here, upstream = here_and_upstream("server_host.py", handler)
        assert here == upstream
    here, upstream = here_and_upstream("server_host.py", "AppContext", "__init__")
    assert here == upstream
    trees = [
        ast.parse(text)
        for text in (source("server_host.py"), upstream_source("server_host.py"))
    ]
    imports = [
        [ast.unparse(n) for n in t.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        for t in trees
    ]
    # Sanic 25: nodes.impl.cors replaces the abandoned sanic-cors, and LOG_CONFIG
    # keeps Sanic 23's log line format.
    upstream = imports[1]
    upstream.remove("from sanic_cors import CORS")
    upstream.insert(
        upstream.index("from gpu import nvidia") + 1,
        "from nodes.impl.cors import add_cors",
    )
    upstream[upstream.index("from server_config import ServerConfig")] = (
        "from server_config import LOG_CONFIG, ServerConfig"
    )
    assert imports[0] == upstream
    # No stop/start-around-run path: the host defines upstream's functions only.
    functions = [
        [
            n.name
            for n in t.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for t in trees
    ]
    assert functions[0] == functions[1]


def test_kill_and_status_proxy_to_the_worker():
    # Review focus 2: Stop ends the run through the worker; /status asks the worker.
    for handler in ("kill", "status"):
        here, upstream = here_and_upstream("server_host.py", handler)
        assert here == upstream


def test_host_and_worker_close_as_upstream():
    # Review focus 4: shutdown awaits no lease cleanup; the worker closes as upstream.
    for relative, path in [
        ("server_host.py", ("close_server",)),
        ("server_process_helper.py", ("_WorkerProcess", "close")),
        ("server_process_helper.py", ("WorkerServer", "stop")),
        ("server_process_helper.py", ("WorkerServer", "get_sse")),
    ]:
        here, upstream = here_and_upstream(relative, *path)
        if path == ("close_server",):
            # Sanic 25 loses a stop requested before it serves, so the host waits for
            # that first, as server.py does.
            stop = "\n    sanic_app.stop()"
            assert upstream.count(stop) == 1
            upstream = upstream.replace(
                stop,
                "\n    while not sanic_app.state.is_running:"
                "\n        await asyncio.sleep(0.01)" + stop,
            )
        assert here == upstream, path


def test_event_stream_sends_without_acknowledgments():
    here, upstream = here_and_upstream("server.py", "sse")
    assert here == upstream
    methods = [
        [getattr(n, "name", "") for n in definition(ast.parse(text), "EventQueue").body]
        for text in (source("events.py"), upstream_source("events.py"))
    ]
    assert methods[0] == methods[1]


def test_individual_run_responds_without_awaiting_its_broadcasts():
    # Review focus 3: the response path awaits what upstream's does, nothing more.
    tree = ast.parse(source("server.py"))
    handlers = [
        definition(t, "run_individual")
        for t in (tree, ast.parse(upstream_source("server.py")))
    ]
    awaited = [
        [ast.unparse(n.value) for n in ast.walk(h) if isinstance(n, ast.Await)]
        for h in handlers
    ]
    assert awaited[0] == awaited[1]
    (outer,) = (n for n in handlers[0].body if isinstance(n, ast.Try))
    assert outer.finalbody == []
    (inner,) = (
        n
        for n in ast.walk(handlers[0])
        if isinstance(n, ast.Try)
        and any(ast.unparse(h.type) == "Aborted" for h in n.handlers if h.type)
    )
    # The pop on every exit path, then the release deferred past the broadcasts,
    # held in a module-level set until it finishes.
    assert [ast.unparse(s) for s in inner.finalbody] == [
        (
            "if ctx.individual_executors.get(execution_id, None) == executor:\n"
            "    ctx.individual_executors.pop(execution_id, None)"
        ),
        "release = asyncio.create_task(release_after_broadcasts(ctx, executor))",
        "individual_releases.add(release)",
        "release.add_done_callback(individual_releases.discard)",
    ]
    assert "individual_releases: set[asyncio.Task[None]] = set()" in [
        ast.unparse(n) for n in tree.body if isinstance(n, ast.AnnAssign)
    ]


def test_cache_clear_releases_the_pool_only_while_no_run_executes():
    # The deferred release guards itself; this handler guards its own call: a clear
    # that arrives mid-run does not empty the run's warm set.
    handler = definition(ast.parse(source("server.py")), "clear_cache_individual")
    releases = [
        n
        for n in ast.walk(handler)
        if isinstance(n, ast.Call) and ast.unparse(n) == "numpy_pool.release()"
    ]
    guarded = [
        c
        for n in ast.walk(handler)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "ctx.executor is None"
        for s in n.body
        for c in ast.walk(s)
        if c in releases
    ]
    assert len(releases) == 1 and guarded == releases


def deferred_release(release):
    """server.py's release_after_broadcasts, compiled alone, with release as
    numpy_pool.release and this module's logger."""
    return compiled(
        "release_after_broadcasts",
        numpy_pool=SimpleNamespace(release=release),
        logger=logging.getLogger(__name__),
    )


class Executor:
    """flush_broadcasts returns once sent is set, recording "sent" in events, then
    raises failure when one is given."""

    def __init__(self, events, failure=None):
        self.events = events
        self.failure = failure
        self.sent = asyncio.Event()

    async def flush_broadcasts(self):
        await self.sent.wait()
        self.events.append("sent")
        if self.failure is not None:
            raise self.failure


@pytest.mark.parametrize("running", [False, True], ids=["idle", "run-started"])
def test_deferred_release_follows_the_broadcasts_and_skips_a_running_run(running):
    async def scenario():
        events = []
        executor = Executor(events)
        ctx = SimpleNamespace(executor=None)
        release = deferred_release(lambda: events.append("release"))
        task = asyncio.create_task(release(ctx, executor))
        for _ in range(5):
            await asyncio.sleep(0)
        # Waiting for the broadcasts: nothing is released before they are sent.
        assert events == [] and not task.done()
        if running:
            # A /run started meanwhile; it releases at its own end.
            ctx.executor = object()
        executor.sent.set()
        await task
        return events

    assert asyncio.run(scenario()) == (["sent"] if running else ["sent", "release"])


@pytest.mark.parametrize("failing", ["broadcast", "release"])
def test_deferred_release_logs_an_error_with_its_traceback(caplog, failing):
    failure = RuntimeError(failing)
    events = []

    def pool_release():
        if failing == "release":
            raise failure
        events.append("release")

    async def scenario():
        executor = Executor(events, failure if failing == "broadcast" else None)
        release = deferred_release(pool_release)
        task = asyncio.create_task(release(SimpleNamespace(executor=None), executor))
        executor.sent.set()
        # The task ends normally: nothing reaches the loop's exception handler.
        await task

    with caplog.at_level(logging.ERROR, logger=__name__):
        asyncio.run(scenario())
    # A failed broadcast still releases: its temporaries are freed either way.
    assert events == (["sent", "release"] if failing == "broadcast" else ["sent"])
    (record,) = [r for r in caplog.records if r.name == __name__]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None and record.exc_info[1] is failure


def test_auto_tiling_budget_is_upstreams():
    # Review focus 5: the budget is 80 % of the (capped) total, not of the free memory.
    here, upstream = here_and_upstream(
        "packages/chaiNNer_pytorch/pytorch/processing/upscale_image.py",
        "upscale",
        "estimate",
    )
    assert here == upstream
    assert "budget = int(total * 0.8)" in here
