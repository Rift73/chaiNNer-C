"""One /run at a time, also while the first waits for previews to finish.

/run checked ctx.executor, then waited for running previews before creating its
executor: a second /run arriving in that wait passed the same check, and both ran.
The guard is now set before the first await; a Stop in that wait ends the waiting
/run before it starts (the worker answered "no-executor", and the run then started).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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
    server.run_individual_counter.count = 0  # the preview ends
    first = await first
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
