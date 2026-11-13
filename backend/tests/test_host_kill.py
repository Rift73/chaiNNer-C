"""The host's /kill: a worker with no run to kill is left running.

The worker answers /kill with 400 "no-executor" when no run executes (Stop pressed just
as a run ended). The host treated every status above 200 as a failed kill and
restarted the worker, which drops every loaded model and engine.

A worker whose CUDA context is broken (a sticky CUDA error) says so in a header of its
/run or /kill response, and the host restarts it: nothing in that context can be freed.
"""

from __future__ import annotations

import asyncio
import logging
from json import dumps
from types import ModuleType, SimpleNamespace

import pytest
from sanic.response import HTTPResponse

from response import RESTART_WORKER_HEADER


class Worker:
    def __init__(
        self, status: int, body: object, headers: dict[str, str] | None = None
    ) -> None:
        self.response = HTTPResponse(dumps(body), status=status, headers=headers)
        self.restarts = 0

    async def proxy_request(self, _request: object, timeout: int | None = 300):
        return self.response

    async def restart(self) -> None:
        self.restarts += 1


def call(
    server_host: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    worker: Worker,
    route: str = "kill",
):
    """The host's handler of `route` (kill or run), with `worker` as its worker."""
    ctx = server_host.app.ctx
    monkeypatch.setattr(ctx, "_worker", worker)
    monkeypatch.setattr(ctx, "is_ready", True)
    request = SimpleNamespace(app=server_host.app)
    return asyncio.run(getattr(server_host, route)(request))


def test_no_run_to_kill_keeps_the_worker(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    worker = Worker(400, {"type": "no-executor"})
    response = call(server_host, monkeypatch, worker)
    assert response.status == 200
    assert worker.restarts == 0


def test_a_failed_kill_still_restarts_the_worker(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    worker = Worker(500, {"type": "error", "message": "stuck"})
    response = call(server_host, monkeypatch, worker)
    assert response.status == 200
    assert worker.restarts == 1


BROKEN = {RESTART_WORKER_HEADER: "1"}


def test_a_kill_that_finds_the_gpu_broken_restarts_the_worker(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    worker = Worker(200, {"type": "success"}, BROKEN)
    response = call(server_host, monkeypatch, worker)
    assert response.status == 200
    assert worker.restarts == 1


def test_a_run_that_broke_the_gpu_restarts_the_worker(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    worker = Worker(500, {"type": "error", "message": "CUDA error"}, BROKEN)
    response = call(server_host, monkeypatch, worker, "run")
    assert response.status == 500  # the run's own error still reaches the app
    assert worker.restarts == 1


def test_a_failed_run_keeps_a_working_worker(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    worker = Worker(500, {"type": "error", "message": "a node failed"})
    call(server_host, monkeypatch, worker, "run")
    assert worker.restarts == 0


def test_a_failed_kill_and_a_failed_restart_are_logged_with_their_errors(
    server_host: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    class Unrestartable(Worker):
        async def restart(self) -> None:
            raise RuntimeError("the new worker did not start")

    worker = Unrestartable(500, {"type": "error", "message": "stuck"})
    with caplog.at_level(logging.INFO):
        response = call(server_host, monkeypatch, worker)
    assert response.status == 500
    logged = [
        (r.levelno, str(r.exc_info[1]))
        for r in caplog.records
        if r.exc_info is not None
    ]
    assert logged == [
        (logging.WARNING, dumps({"type": "error", "message": "stuck"})),
        (logging.ERROR, "the new worker did not start"),
    ]
