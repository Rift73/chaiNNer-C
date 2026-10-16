"""The dependency manager's pip runs (server_host.change_dependencies): a request that
closes during pip (the app reloads) neither stops pip nor starts the worker under it."""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

import server_process_helper
from server_process_helper import WorkerServer


@pytest.fixture
def server_host(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # The module parses its command line when imported.
    monkeypatch.setattr(sys, "argv", ["server_host.py"])
    return importlib.import_module("server_host")


@pytest.mark.parametrize("fails", [False, True], ids=["finished", "failed"])
def test_a_closed_request_starts_the_worker_after_pip_and_logs_the_outcome(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch, fails: bool
):
    events: list[str] = []
    logged: list[str] = []
    monkeypatch.setattr(
        server_process_helper,
        "logger",
        SimpleNamespace(
            **dict.fromkeys(("debug", "info", "warning", "error"), logged.append)
        ),
    )

    class WorkerProcess:
        def __init__(self, _flags: list[str]) -> None:
            events.append("worker started")

        def close(self) -> None:
            events.append("worker stopped")

    async def ready(_self: WorkerServer, timeout: float = 300) -> None:
        pass

    monkeypatch.setattr(server_process_helper, "_WorkerProcess", WorkerProcess)
    monkeypatch.setattr(WorkerServer, "wait_for_ready", ready)

    async def slow_pip() -> int:
        await asyncio.sleep(0.5)
        events.append("pip done")
        if fails:
            raise ValueError("An error occurred while installing dependencies.")
        return 1

    async def close_the_request_during_pip() -> None:
        worker = WorkerServer()
        request = asyncio.create_task(
            server_host.change_dependencies(worker, slow_pip())
        )
        await asyncio.sleep(0.1)
        # Sanic cancels the route's task when its connection closes.
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        await worker.stop()

    asyncio.run(close_the_request_during_pip())
    assert events == ["pip done", "worker started", "worker stopped"]
    outcome = (
        "Dependency change failed after its request closed:"
        " An error occurred while installing dependencies."
        if fails
        else "Dependency change finished after its request closed"
    )
    assert outcome in logged
