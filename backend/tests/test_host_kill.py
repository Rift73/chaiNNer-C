"""The host's /kill: a worker with no run to kill is left running.

The worker answers /kill with 400 "no-executor" when no run executes (Stop pressed just
as a run ended). The host treated every status above 200 as a failed kill and
restarted the worker, which drops every loaded model and engine.

A worker whose CUDA context is broken (a sticky CUDA error) says so in a header of its
/run or /kill response, and the host restarts it: nothing in that context can be freed.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
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


@pytest.fixture
def server_host(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # The module parses its command line when imported.
    monkeypatch.setattr(sys, "argv", ["server_host.py"])
    return importlib.import_module("server_host")


def kill(server_host: ModuleType, monkeypatch: pytest.MonkeyPatch, worker: Worker):
    ctx = server_host.app.ctx
    monkeypatch.setattr(ctx, "_worker", worker)
    monkeypatch.setattr(ctx, "is_ready", True)
    request = SimpleNamespace(app=server_host.app)
    return asyncio.run(server_host.kill(request))


def test_no_run_to_kill_keeps_the_worker(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    worker = Worker(400, {"type": "no-executor"})
    response = kill(server_host, monkeypatch, worker)
    assert response.status == 200
    assert worker.restarts == 0


def test_a_failed_kill_still_restarts_the_worker(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    worker = Worker(500, {"type": "error", "message": "stuck"})
    response = kill(server_host, monkeypatch, worker)
    assert response.status == 200
    assert worker.restarts == 1


BROKEN = {RESTART_WORKER_HEADER: "1"}


def test_a_kill_that_finds_the_gpu_broken_restarts_the_worker(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    worker = Worker(200, {"type": "success"}, BROKEN)
    response = kill(server_host, monkeypatch, worker)
    assert response.status == 200
    assert worker.restarts == 1


def run(server_host: ModuleType, monkeypatch: pytest.MonkeyPatch, worker: Worker):
    ctx = server_host.app.ctx
    monkeypatch.setattr(ctx, "_worker", worker)
    monkeypatch.setattr(ctx, "is_ready", True)
    request = SimpleNamespace(app=server_host.app)
    return asyncio.run(server_host.run(request))


def test_a_run_that_broke_the_gpu_restarts_the_worker(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    worker = Worker(500, {"type": "error", "message": "CUDA error"}, BROKEN)
    response = run(server_host, monkeypatch, worker)
    assert response.status == 500  # the run's own error still reaches the app
    assert worker.restarts == 1


def test_a_failed_run_keeps_a_working_worker(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    worker = Worker(500, {"type": "error", "message": "a node failed"})
    run(server_host, monkeypatch, worker)
    assert worker.restarts == 0
