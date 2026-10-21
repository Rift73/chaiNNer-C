"""One /run at a time, also while the first waits for previews to finish.

/run checked ctx.executor, then waited for running previews before creating its
executor: a second /run arriving in that wait passed the same check, and both ran.
The guard is now set before the first await; a Stop in that wait ends the waiting
/run before it starts (the worker answered "no-executor", and the run then started).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).parent.parent / "src"

# server parses the command line when imported, so it runs in its own process.
CHECK = """
import asyncio
import json
from types import SimpleNamespace

import server

RUN = {"data": [], "options": {}, "sendBroadcastData": False}


def answer(response):
    return response.status, json.loads(response.body)["type"]


async def scenario():
    server.run_individual_counter.count = 1  # a preview is running
    first = asyncio.ensure_future(server.run(SimpleNamespace(app=server.app, json=RUN)))
    await asyncio.sleep(0.1)
    # Without the guard it waits for the preview too (and would then run): a timeout.
    second = await asyncio.wait_for(
        server.run(SimpleNamespace(app=server.app, json=RUN)), timeout=5
    )
    stop = await server.kill(SimpleNamespace(app=server.app))
    assert json.loads((await server.status(None)).body)["executor"] == "killing"
    assert json.loads((await server.status(None)).body)["executor"] == "killing"
    server.run_individual_counter.count = 0  # the preview ends
    first = await first
    assert json.loads((await server.status(None)).body)["executor"] == "ready"
    ctx = server.app.ctx
    return [answer(second), answer(stop), answer(first), ctx.run_starting, ctx.executor]


print(asyncio.run(scenario()))
"""


def test_a_run_waiting_for_previews_refuses_a_second_and_stops():
    result = subprocess.run(
        [sys.executable, "-B", "-c", CHECK],
        cwd=SRC,
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == str(
        [(500, "already-running"), (200, "success"), (200, "success"), False, None]
    )


@pytest.mark.parametrize("action", ["pause", "resume", "kill"])
def test_pause_while_waiting_for_previews_is_applied_before_execution(
    monkeypatch: pytest.MonkeyPatch, action: str
):
    monkeypatch.setattr(sys, "argv", ["server.py"])
    server = importlib.import_module("server")
    ctx = server.app.ctx
    started: list[bool] = []

    class Executor:
        def __init__(self, **_kwargs: object):
            self.paused = False

        def pause(self):
            self.paused = True

        async def run(self):
            started.append(self.paused)

    monkeypatch.setattr(server, "Executor", Executor)
    monkeypatch.setattr(server, "executor_kind", lambda: "python")
    monkeypatch.setattr(ctx, "storage_dir", Path("."))
    monkeypatch.setattr(server.run_individual_counter, "count", 1)

    async def scenario():
        app = SimpleNamespace(ctx=ctx, loop=asyncio.get_running_loop())
        monkeypatch.setattr(server, "app", app)
        request = SimpleNamespace(
            app=app, json={"data": [], "options": {}, "sendBroadcastData": False}
        )
        pending = asyncio.create_task(server.run(request))
        try:
            await asyncio.sleep(0)
            assert ctx.run_starting and ctx.executor is None
            paused = await server.pause(request)
            assert paused.status == 200
            assert json.loads((await server.status(None)).body)["executor"] == "paused"
            if action != "pause":
                reply = await getattr(server, action)(request)
                assert reply.status == 200
                expected = "killing" if action == "kill" else "running"
                assert (
                    json.loads((await server.status(None)).body)["executor"] == expected
                )
            server.run_individual_counter.count = 0
            result = await asyncio.wait_for(pending, 2)
            assert result.status == 200
            assert started == ([] if action == "kill" else [action == "pause"])
            assert json.loads((await server.status(None)).body)["executor"] == "ready"
            assert not ctx.run_paused_early
            # The next run does not inherit the startup pause or stop.
            result = await server.run(request)
            assert result.status == 200
            assert started[-1] is False
            assert (await server.pause(request)).status == 400
            assert (await server.resume(request)).status == 400
        finally:
            server.run_individual_counter.count = 0
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(scenario())
