"""The dependency manager's pip runs (server_host.change_dependencies): a request that
closes during pip (the app reloads) neither stops pip nor starts the worker under it;
an uninstall also takes what pip pulled in that nothing else needs, and an install
also installs the missing VapourSynth user modules."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import api
import server_process_helper
from dependencies import store
from dependencies.store import DependencyInfo
from nodes.impl.vapoursynth import user_modules
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


def test_an_uninstall_also_takes_what_nothing_else_needs(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    # What pip pulled in for the package goes too, unless chaiNNer-C lists it: the
    # server's own packages, another package's dependencies.
    def package(package_id: str, *names: str) -> api.Package:
        return api.Package(
            where="",
            id=package_id,
            name=package_id,
            description="",
            icon="",
            color="",
            dependencies=[
                api.Dependency(
                    display_name=name, pypi_name=name, version="1.0", size_estimate=0
                )
                for name in names
            ],
        )

    packages = [
        package("vapoursynth", "VapourSynth", "vsjetpack[deinterlace]"),
        package("pytorch", "torch", "Spandrel_Fork"),
    ]

    class Worker:
        async def get_packages(self) -> list[api.Package]:
            return packages

    async def get_worker() -> Worker:
        return Worker()

    seen: list[tuple[list[str], set[str]]] = []

    def unneeded(leaving: list[DependencyInfo], keep: set[str]) -> list[DependencyInfo]:
        seen.append(([d.package_name for d in leaving], keep))
        return [DependencyInfo(package_name="pyside6", version="6.12.0")]

    uninstalled: list[list[str]] = []

    async def uninstall(deps: list[DependencyInfo], *_args: object) -> None:
        uninstalled.append([d.package_name for d in deps])

    async def change(_worker: object, job: Any) -> None:
        await job

    ctx = SimpleNamespace(get_worker=get_worker)
    monkeypatch.setattr(server_host.AppContext, "get", lambda _app: ctx)
    monkeypatch.setattr(server_host, "unneeded_dependencies", unneeded)
    monkeypatch.setattr(server_host, "uninstall_dependencies", uninstall)
    monkeypatch.setattr(server_host, "change_dependencies", change)
    monkeypatch.setattr(server_host, "create_update_progress", lambda _ctx: None)
    request = SimpleNamespace(json={"packages": ["vapoursynth"]}, app=None)

    response = asyncio.run(server_host.uninstall_dependencies_request(request))
    assert response.status == 200
    ((leaving, keep),) = seen
    assert leaving == ["VapourSynth", "vsjetpack[deinterlace]"]
    assert {"torch", "spandrel-fork", "sanic", "numpy"} <= keep
    assert not {"vapoursynth", "vsjetpack"} & keep
    assert uninstalled == [["VapourSynth", "vsjetpack[deinterlace]", "pyside6"]]


def test_an_install_also_installs_the_missing_vapoursynth_user_modules(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # After VapourSynth's first install its user modules go in at once, with the
    # worker still stopped, not at the next start.
    (tmp_path / "vapoursynth").mkdir()
    (tmp_path / "vapoursynth" / "user_modules.toml").write_text(
        'modules = ["chainner-test-a"]\n', encoding="utf-8"
    )
    vapoursynth = api.Package(
        where="",
        id="vapoursynth",
        name="vapoursynth",
        description="",
        icon="",
        color="",
        dependencies=[
            api.Dependency(
                display_name="VapourSynth",
                pypi_name="VapourSynth",
                version="81",
                size_estimate=0,
            )
        ],
    )

    class Worker:
        async def get_packages(self) -> list[api.Package]:
            return [vapoursynth]

    async def get_worker() -> Worker:
        return Worker()

    steps: list[str] = []

    async def install(deps: list[DependencyInfo], *_args: object) -> None:
        steps.append("pip " + " ".join(d.package_name for d in deps))
        monkeypatch.setitem(store.installed_packages, "vapoursynth", "81")

    async def change(_worker: object, job: Any) -> None:
        steps.append("worker stopped")
        await job
        steps.append("worker started")

    async def progress(*_args: object) -> None:
        pass

    monkeypatch.delitem(store.installed_packages, "vapoursynth", raising=False)
    ctx = SimpleNamespace(
        get_worker=get_worker, config=SimpleNamespace(storage_dir=str(tmp_path))
    )
    monkeypatch.setattr(server_host.AppContext, "get", lambda _app: ctx)
    monkeypatch.setattr(server_host, "install_dependencies", install)
    monkeypatch.setattr(server_host, "change_dependencies", change)
    monkeypatch.setattr(server_host, "create_update_progress", lambda _ctx: progress)
    monkeypatch.setattr(
        user_modules, "install_user_module", lambda name: steps.append(f"module {name}")
    )
    monkeypatch.setattr(user_modules, "scan_installed_packages", lambda: None)
    request = SimpleNamespace(json={"packages": ["vapoursynth"]}, app=None)

    response = asyncio.run(server_host.install_dependencies_request(request))
    assert response.status == 200
    assert steps == [
        "worker stopped",
        "pip VapourSynth",
        "module chainner-test-a",
        "worker started",
    ]
