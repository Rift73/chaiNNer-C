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
from types import SimpleNamespace
from typing import Any, cast

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


def test_an_event_stream_open_while_the_worker_stops_ends_cleanly(
    worker: WorkerServer,
):
    class Response:
        def __init__(self) -> None:
            self.open = True
            self.content = self

        async def iter_chunks(self):
            while self.open:
                yield b"event", True
                await asyncio.sleep(0)

        def close(self) -> None:
            self.open = False

    class Session:
        def __init__(self) -> None:
            self.response = Response()

        def request(self, *_args: object, **_kwargs: object):
            response = self.response

            class Opened:
                async def __aenter__(self) -> Response:
                    return response

                async def __aexit__(self, *_exc: object) -> None:
                    pass

            return Opened()

        async def close(self) -> None:
            pass

    async def scenario() -> list[bytes]:
        worker._session = cast(Any, Session())
        request = cast(Any, SimpleNamespace(method="GET", headers={}, body=b""))
        received: list[bytes] = []
        async for data in worker.get_sse(request):
            received.append(data)
            if len(received) == 1:
                await worker.stop()
        return received

    assert asyncio.run(scenario()) == [b"event"]


@pytest.mark.parametrize(
    ("answers", "kills"),
    [
        ([("success", 0.0), ("no-executor", 0.0)], 2),  # a run: Stop, then its end
        ([("no-executor", 0.0)], 1),  # no run
        ([("success", 10.0)], 1),  # a Stop that hangs: given up after the timeout
    ],
    ids=["run", "idle", "hung"],
)
def test_stopping_the_run_before_closing_the_worker(
    worker: WorkerServer, answers: list[tuple[str, float]], kills: int
):
    posted: list[str] = []

    class Response:
        def __init__(self, kind: str) -> None:
            self.kind = kind

        async def json(self) -> dict[str, str]:
            return {"type": self.kind}

    class Session:
        def post(self, path: str, **_kwargs: object):
            posted.append(path)
            kind, delay = answers[min(len(posted), len(answers)) - 1]

            class Opened:
                async def __aenter__(self) -> Response:
                    await asyncio.sleep(delay)
                    return Response(kind)

                async def __aexit__(self, *_exc: object) -> None:
                    pass

            return Opened()

    async def scenario() -> float:
        worker._session = cast(Any, Session())
        start = asyncio.get_running_loop().time()
        await worker.stop_run(timeout=0.5)
        return asyncio.get_running_loop().time() - start

    elapsed = asyncio.run(scenario())
    assert posted == ["/kill"] * kills
    assert elapsed < 1
