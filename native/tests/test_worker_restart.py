"""Two overlapping restarts of the worker leave one worker process, the one the host
owns.

The host's /kill restarts the worker when the worker does not stop its run within
3 seconds; a second Stop in that window sends a second /kill, which fails the same
way, so two restart() calls overlap at their awaits. Each started a process, and the
second replaced the first in WorkerServer._process: the first kept running, owned by
nothing until the host exits, on the port the next worker binds. The worker process
is a stand-in here, so the test starts no Python.
"""

from __future__ import annotations

import asyncio

import pytest

import server_process_helper
from server_process_helper import WorkerServer

STARTED: list[FakeProcess] = []


class FakeProcess:
    def __init__(self, flags: list[str]):
        self.closed = False
        STARTED.append(self)

    def close(self):
        self.closed = True


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> WorkerServer:
    STARTED.clear()
    monkeypatch.setattr(server_process_helper, "_WorkerProcess", FakeProcess)

    async def ready(self: WorkerServer, timeout: float = 300):
        await asyncio.sleep(0.01)

    monkeypatch.setattr(WorkerServer, "wait_for_ready", ready)
    return WorkerServer()


def test_overlapping_restarts_leave_one_live_worker(worker: WorkerServer):
    async def scenario():
        await worker.start()
        await asyncio.gather(worker.restart(), worker.restart())
        live = [p for p in STARTED if not p.closed]
        owned = worker._process
        await worker.stop()
        return live, owned

    live, owned = asyncio.run(scenario())
    assert len(live) == 1
    assert live[0] is owned


def test_stop_then_start_closes_once(worker: WorkerServer):
    async def scenario():
        await worker.start()
        await worker.stop()
        await worker.stop()
        await worker.start()
        await worker.stop()

    asyncio.run(scenario())
    assert len(STARTED) == 2
    assert all(p.closed for p in STARTED)
