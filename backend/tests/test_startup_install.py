"""The host's startup install of package dependencies (server_host.import_packages)."""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

import api


class Worker:
    """Records the host's calls; the worker lists one package with a missing dependency."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    async def get_packages(self) -> list[api.Package]:
        dependency = api.Dependency(
            display_name="Missing",
            pypi_name="chainner-test-missing",
            version="1.0.0",
            size_estimate=0,
        )
        return [
            api.Package(
                where="",
                id="chaiNNer_standard",
                name="chaiNNer_standard",
                description="",
                icon="",
                color="",
                dependencies=[dependency],
            )
        ]

    async def stop(self) -> None:
        self.calls.append("stop")

    async def start(self, extra_flags: list[str] | None = None) -> None:
        self.calls.append(("start", extra_flags))


@pytest.fixture
def server_host(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # The module parses its command line when imported.
    monkeypatch.setattr(sys, "argv", ["server_host.py"])
    return importlib.import_module("server_host")


def run_import_packages(
    server_host: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    worker: Worker,
    close_after_start: bool,
) -> None:
    async def failing_install(*_args: object) -> None:
        raise ValueError("An error occurred while installing dependencies.")

    async def progress(*_args: object) -> None:
        pass

    monkeypatch.setattr(server_host, "install_dependencies", failing_install)
    config = SimpleNamespace(
        install_builtin_packages=False,
        error_on_failed_node=False,
        close_after_start=close_after_start,
    )
    ctx = SimpleNamespace(config=config, get_worker_unmanaged=lambda: worker)
    asyncio.run(server_host.import_packages(ctx, progress))


def test_a_failed_startup_install_restarts_the_worker(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    # A stopped worker would answer every proxied route with "Session is closed".
    worker = Worker()
    run_import_packages(server_host, monkeypatch, worker, close_after_start=False)
    assert worker.calls == ["stop", ("start", [])]


def test_close_after_start_reports_the_failed_install_after_the_restart(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    worker = Worker()
    with pytest.raises(ValueError, match="Error installing dependencies"):
        run_import_packages(server_host, monkeypatch, worker, close_after_start=True)
    assert worker.calls == ["stop", ("start", ["--close-after-start"])]
