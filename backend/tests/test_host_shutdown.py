"""The host's shutdown: once the app has asked the host to end, SIGTERM ends it at once.

The app sends SIGTERM when it has the answer to its shutdown request. Sanic's handler
for it stops the event loop again, which cuts short the shutdown coroutines Sanic still
runs on that loop (RuntimeError: Event loop stopped before Future completed), and uvloop
cannot remove the handler once the loop has stopped. A Windows host never sees it: the
app ends that process with TerminateProcess.
"""

from __future__ import annotations

import asyncio
import importlib
import signal
import sys
from types import ModuleType, SimpleNamespace

import pytest


class Worker:
    def __init__(self) -> None:
        self.stops = 0

    async def stop(self) -> None:
        self.stops += 1


@pytest.fixture
def server_host(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # The module parses its command line when imported.
    monkeypatch.setattr(sys, "argv", ["server_host.py"])
    return importlib.import_module("server_host")


@pytest.mark.skipif(sys.platform != "linux", reason="Linux hosts get SIGTERM")
def test_a_closing_host_ends_on_sigterm(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    worker = Worker()
    monkeypatch.setattr(server_host.app.ctx, "_worker", worker)
    stops: list[bool] = []
    sanic_app = SimpleNamespace(
        ctx=server_host.app.ctx,
        state=SimpleNamespace(is_running=True),
        stop=lambda: stops.append(True),
    )

    # Sanic's handler is in place while the host runs
    previous = signal.signal(signal.SIGTERM, lambda _signum, _frame: None)
    try:
        asyncio.run(server_host.close_server(sanic_app))
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
    finally:
        signal.signal(signal.SIGTERM, previous)

    assert worker.stops == 1
    assert stops == [True]
