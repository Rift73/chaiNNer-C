"""The host's startup install of package dependencies and of the VapourSynth user
modules (server_host.import_packages)."""

from __future__ import annotations

import asyncio
import importlib
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import api
from dependencies import store
from dependencies.store import DependencyInfo
from nodes.impl.vapoursynth import user_modules

MISSING = api.Dependency(
    display_name="Missing",
    pypi_name="chainner-test-missing",
    version="1.0.0",
    size_estimate=0,
)


class Worker:
    """Records the host's calls; the worker lists one package with one dependency
    (by default chaiNNer_standard, with a missing one)."""

    def __init__(
        self,
        package_id: str = "chaiNNer_standard",
        dependency: api.Dependency = MISSING,
    ) -> None:
        self.calls: list[object] = []
        self.package_id = package_id
        self.dependency = dependency

    async def get_packages(self) -> list[api.Package]:
        return [
            api.Package(
                where="",
                id=self.package_id,
                name=self.package_id,
                description="",
                icon="",
                color="",
                dependencies=[self.dependency],
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


async def failing_install(*_args: object) -> None:
    raise ValueError("An error occurred while installing dependencies.")


def run_import_packages(
    server_host: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    worker: Worker,
    close_after_start: bool,
    install: Callable[..., Awaitable[object]] = failing_install,
    storage_dir: Path | None = None,
) -> list[str]:
    """The progress messages of the start."""
    messages: list[str] = []

    async def progress(message: str, *_args: object) -> None:
        messages.append(message)

    monkeypatch.setattr(server_host, "install_dependencies", install)
    config = SimpleNamespace(
        install_builtin_packages=False,
        error_on_failed_node=False,
        close_after_start=close_after_start,
        storage_dir=None if storage_dir is None else str(storage_dir),
    )
    ctx = SimpleNamespace(config=config, get_worker_unmanaged=lambda: worker)
    asyncio.run(server_host.import_packages(ctx, progress))
    return messages


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


def test_startup_updates_an_installed_dependency_with_extras(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    # pip lists "vsjetpack[deinterlace]" as "vsjetpack": an outdated one is
    # auto-updated like the package's other dependencies, not skipped as missing.
    monkeypatch.setitem(store.installed_packages, "vsjetpack", "2.2.6")
    jetpack = api.Dependency(
        display_name="vs-jetpack",
        pypi_name="vsjetpack[deinterlace]",
        version="2.2.7",
        size_estimate=0,
    )
    installed: list[list[DependencyInfo]] = []

    async def install(dependencies: list[DependencyInfo], *_args: object) -> None:
        installed.append(dependencies)

    worker = Worker("chaiNNer_vapoursynth", jetpack)
    run_import_packages(server_host, monkeypatch, worker, False, install)
    assert [[d.package_name for d in deps] for deps in installed] == [
        ["vsjetpack[deinterlace]"]
    ]


async def no_install(*_args: object) -> None:
    pass


def test_startup_installs_missing_vapoursynth_user_modules_with_the_worker_stopped(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # The worker's node import, which the app waits for, only scans them: an install
    # can take minutes. A failed one is tried again at the next start.
    for name, version in (("vapoursynth", "81"), ("chainner-test-missing", "1.0.0")):
        monkeypatch.setitem(store.installed_packages, name, version)
    folder = tmp_path / "vapoursynth"
    folder.mkdir()
    (folder / "user_modules.toml").write_text(
        'modules = ["chainner-test-missing", "chainner-test-a", "chainner-test-b>=2"]\n',
        encoding="utf-8",
    )
    worker = Worker()

    def install(name: str) -> None:
        worker.calls.append(("pip", name))

    monkeypatch.setattr(user_modules, "install_user_module", install)
    monkeypatch.setattr(user_modules, "scan_installed_packages", lambda: None)
    messages = run_import_packages(
        server_host, monkeypatch, worker, False, no_install, tmp_path
    )
    assert worker.calls == [
        "stop",
        ("pip", "chainner-test-a"),
        ("pip", "chainner-test-b>=2"),
        ("start", []),
    ]
    assert "Installing the VapourSynth user module chainner-test-b>=2..." in messages

    # A failed package install does not keep them out.
    worker = Worker()
    run_import_packages(server_host, monkeypatch, worker, False, storage_dir=tmp_path)
    assert worker.calls[1:] == [
        ("pip", "chainner-test-a"),
        ("pip", "chainner-test-b>=2"),
        ("start", []),
    ]

    # Installed: the worker keeps running.
    monkeypatch.setitem(store.installed_packages, "chainner-test-a", "1.0")
    monkeypatch.setitem(store.installed_packages, "chainner-test-b", "2.0")
    worker = Worker()
    run_import_packages(server_host, monkeypatch, worker, False, no_install, tmp_path)
    assert worker.calls == []
