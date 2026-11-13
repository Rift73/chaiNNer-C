"""The vsview launcher of Preview Clips: it reloads the script once a rewrite settles,
retries while vsview is busy, and runs vsview as `python -m vsview` when an internal
it relies on is missing. Stand-ins replace vsview and Qt; no window opens."""

from __future__ import annotations

import os
import sys
import types
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from nodes.impl.vapoursynth import viewer
from nodes.impl.vapoursynth.viewer import BUSY, Reloader


def done(result: object) -> Future[Any]:
    future: Future[Any] = Future()
    future.set_result(result)
    return future


class Vsview:
    """reload_content's stand-in: each call takes the next future."""

    def __init__(self, *futures: Future[Any]):
        self.futures = list(futures)
        self.calls = 0

    def reload(self) -> Future[Any]:
        self.calls += 1
        return self.futures.pop(0)


def rewrite(script: Path, text: str) -> None:
    script.write_text(text, encoding="utf-8")
    # A distinct stamp also where the clock is coarse.
    stamp = script.stat().st_mtime_ns + 10**9 * (1 + len(text))
    os.utime(script, ns=(stamp, stamp))


@pytest.fixture
def script(tmp_path: Path) -> Path:
    path = tmp_path / "preview.vpy"
    path.write_text("first", encoding="utf-8")
    return path


def test_a_rewrite_reloads_once_it_settled(script: Path):
    vsview = Vsview(done(0))
    reloader = Reloader(script, vsview.reload, pytest.fail)
    reloader.tick()
    assert vsview.calls == 0

    rewrite(script, "second")
    reloader.tick()
    # The check that sees the change waits for one without a further change.
    assert vsview.calls == 0
    reloader.tick()
    assert vsview.calls == 1
    for _ in range(3):
        reloader.tick()
    assert vsview.calls == 1


def test_a_busy_viewer_is_asked_again(script: Path):
    vsview = Vsview(done(BUSY), done(BUSY), done(0))
    reloader = Reloader(script, vsview.reload, pytest.fail)
    rewrite(script, "second")
    for _ in range(10):
        reloader.tick()
    assert vsview.calls == 3


def test_a_rewrite_during_a_reload_reloads_again(script: Path):
    running: Future[Any] = Future()
    vsview = Vsview(running, done(0))
    reloader = Reloader(script, vsview.reload, pytest.fail)
    rewrite(script, "second")
    reloader.tick()
    reloader.tick()
    assert vsview.calls == 1

    rewrite(script, "third")
    reloader.tick()
    running.set_result(0)
    for _ in range(5):
        reloader.tick()
    assert vsview.calls == 2


def test_a_failed_reload_is_not_repeated(script: Path):
    failed: Future[Any] = Future()
    failed.set_exception(RuntimeError("script error"))
    vsview = Vsview(failed)
    reloader = Reloader(script, vsview.reload, pytest.fail)
    rewrite(script, "second")
    for _ in range(5):
        reloader.tick()
    assert vsview.calls == 1


@pytest.mark.parametrize("gives", ["future", "result"])
def test_an_unexpected_reload_turns_auto_reload_off(script: Path, gives: str):
    off: list[str] = []
    reload = (lambda: None) if gives == "future" else (lambda: done(None))
    reloader = Reloader(script, reload, off.append)
    rewrite(script, "second")
    for _ in range(5):
        reloader.tick()
    assert reloader.stopped
    assert len(off) == 1


class Timer:
    """QTimer's stand-in."""

    def __init__(self, parent: object):
        self.interval: int | None = None
        self.timeout = types.SimpleNamespace(connect=self.connect)
        self.slot: Any = None

    def connect(self, slot: Any) -> None:
        self.slot = slot

    def start(self, interval: int) -> None:
        self.interval = interval


def stand_in_vsview(
    monkeypatch: pytest.MonkeyPatch, window: type | None, cli_window: bool = False
) -> tuple[list[list[str]], Any]:
    """vsview's modules, with `window` as `vsview.app.main.MainWindow` (and as
    `vsview.cli.MainWindow` too with `cli_window`); running `python -m vsview` records
    sys.argv. Returns the recorded argvs and the main module."""
    argvs: list[list[str]] = []
    main = types.SimpleNamespace()
    if window is not None:
        main.MainWindow = window
    modules: dict[str, Any] = {
        "vsview.cli": types.SimpleNamespace(
            **({"MainWindow": window} if cli_window else {})
        ),
        "vsview.app.main": main,
        "PySide6.QtCore": types.SimpleNamespace(QTimer=Timer),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    def run_module(name: str, run_name: str, alter_sys: bool) -> None:
        assert (name, run_name, alter_sys) == ("vsview", "__main__", True)
        argvs.append(list(sys.argv))

    monkeypatch.setattr(viewer.runpy, "run_module", run_module)
    monkeypatch.setattr(sys, "argv", ["viewer.py"])
    return argvs, main


def _set_title(window: Any, title: str) -> None:
    window.title = title


class _Window:
    def __init__(self) -> None:
        self.title = "VS View"

    def load_new_script(self, path: Path, **vsargs: Any) -> tuple[Any, Future[int]]:
        workspace = types.SimpleNamespace(reload_content=lambda: done(0))
        return types.SimpleNamespace(workspace=workspace), done(0)


# MainWindow's stand-in as vsview 0.13.0 has it (Qt's mixedCase name).
Window: Any = type("Window", (_Window,), {"setWindowTitle": _set_title})


def test_the_window_has_the_title_and_reloads_its_script(
    monkeypatch: pytest.MonkeyPatch, script: Path, capsys: pytest.CaptureFixture[str]
):
    argvs, main = stand_in_vsview(monkeypatch, Window)
    viewer.main([str(script), "Source, BM3D - chaiNNer-C"])
    # vsview runs as `python -m vsview <script>`, with the subclass as its window.
    assert argvs == [["viewer.py", str(script)]]
    hooked = main.MainWindow
    assert hooked is not Window and Window in hooked.__mro__

    window = hooked()
    assert window.title == "Source, BM3D - chaiNNer-C"
    window.load_new_script(script)
    (reloader,) = window.reloaders
    rewrite(script, "second")
    reloader.tick()
    reloader.tick()
    assert not reloader.stopped
    assert capsys.readouterr().out == ""


class OldWindow:
    """A MainWindow without load_new_script."""


class NoWorkspaceWindow(Window):
    def load_new_script(self, path: Path, **vsargs: Any) -> tuple[Any, Future[int]]:
        return types.SimpleNamespace(), done(0)


# A vsview.cli with a MainWindow of its own would not create a subclass put in
# vsview.app.main.
@pytest.mark.parametrize(
    ("window", "cli_window"), [(None, False), (OldWindow, False), (Window, True)]
)
def test_without_the_window_internals_vsview_runs_as_before(
    monkeypatch: pytest.MonkeyPatch,
    script: Path,
    capsys: pytest.CaptureFixture[str],
    window: type | None,
    cli_window: bool,
):
    argvs, main = stand_in_vsview(monkeypatch, window, cli_window)
    viewer.main([str(script), "Clip 1 - chaiNNer-C"])
    assert argvs == [["viewer.py", str(script)]]
    assert getattr(main, "MainWindow", None) is window
    (line,) = capsys.readouterr().out.splitlines()
    assert line.startswith("Auto-reload is off (") and "Ctrl+R" in line


def test_without_a_workspace_to_reload_the_window_only_says_so(
    monkeypatch: pytest.MonkeyPatch, script: Path, capsys: pytest.CaptureFixture[str]
):
    _, main = stand_in_vsview(monkeypatch, NoWorkspaceWindow)
    viewer.main([str(script), "Clip 1 - chaiNNer-C"])
    window = main.MainWindow()
    window.load_new_script(script)
    assert window.reloaders == []
    assert "Auto-reload is off (" in capsys.readouterr().out


def test_a_reload_that_raises_is_logged_once_and_the_next_change_reloads(
    script: Path, capsys: pytest.CaptureFixture[str]
):
    calls: list[int] = []

    def reload() -> Future[Any]:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("workspace gone")
        return done(0)

    reloader = Reloader(script, reload, pytest.fail)
    rewrite(script, "second")
    for _ in range(5):
        reloader.tick()
    assert len(calls) == 1
    out = capsys.readouterr().out
    assert out.count("The reload failed") == 1 and "RuntimeError: workspace gone" in out
    rewrite(script, "third")
    for _ in range(3):
        reloader.tick()
    assert len(calls) == 2 and not reloader.stopped


def test_a_failed_reload_is_logged_with_its_traceback(
    script: Path, capsys: pytest.CaptureFixture[str]
):
    failed: Future[Any] = Future()
    failed.set_exception(RuntimeError("script error"))
    reloader = Reloader(script, Vsview(failed).reload, pytest.fail)
    rewrite(script, "second")
    for _ in range(5):
        reloader.tick()
    out = capsys.readouterr().out
    assert out.count("The reload failed") == 1 and "RuntimeError: script error" in out


def test_a_viewer_that_stays_busy_is_given_up_once(
    script: Path, capsys: pytest.CaptureFixture[str]
):
    vsview = Vsview(*(done(BUSY) for _ in range(viewer.BUSY_TRIES + 5)))
    reloader = Reloader(script, vsview.reload, pytest.fail)
    rewrite(script, "second")
    for _ in range(3 * viewer.BUSY_TRIES):
        reloader.tick()
    assert vsview.calls == viewer.BUSY_TRIES
    assert capsys.readouterr().out.count("stayed busy") == 1
    assert not reloader.stopped


def test_without_vsviews_window_module_vsview_runs_as_before(
    monkeypatch: pytest.MonkeyPatch, script: Path, capsys: pytest.CaptureFixture[str]
):
    argvs, _ = stand_in_vsview(monkeypatch, Window)
    # A vsview whose window module moved: importing it fails.
    monkeypatch.setitem(sys.modules, "vsview.app.main", None)
    viewer.main([str(script), "Clip 1 - chaiNNer-C"])
    assert argvs == [["viewer.py", str(script)]]
    (line,) = capsys.readouterr().out.splitlines()
    assert line.startswith("Auto-reload is off (cannot import vsview.app.main")
