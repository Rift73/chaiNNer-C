"""Runs vsview on a Preview Clips script and reloads it when a run rewrites the script,
as Reload Script (Ctrl+R) does: the frame, tab and zoom stay.

Run as a file, `python viewer.py <script> <window title>`, in vsview's Python. It
imports only the standard library, vsview and PySide6.

vsview runs as `python -m vsview <script>` does. It has no option or interface for
the title or a reload, so these use its internals, each looked up before use. If one
is missing or differs, the log (stdout) gets one line saying auto-reload is off and
why, and vsview runs unchanged. The names relied on (vsview 0.13.0):

- `vsview.app.main.MainWindow`: the window class. `vsview.cli` imports it from that
  module when it runs (it has no `MainWindow` of its own), so a subclass put there is
  the one it creates.
- `MainWindow.load_new_script(path, **vsargs)`: returns `(button, future)`, and
  `button.workspace` is the script's workspace.
- `workspace.reload_content()`: the slot of Reload Script; returns a
  `concurrent.futures.Future` of an int, 2 when the workspace is busy (nothing done).
"""

from __future__ import annotations

import importlib
import runpy
import sys
import traceback
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable

# How often the script's file is checked for a rewrite, in milliseconds.
POLL_MS = 400
# reload_content's result when the workspace is busy and did nothing.
BUSY = 2
# Busy answers in a row before a rewrite is given up (until the next one): 2 minutes,
# above a reload that evaluates AI Upscale models on a CPU (over a minute).
BUSY_TRIES = 300


def _stamp(script: Path) -> tuple[int, int] | None:
    try:
        stat = script.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


class Reloader:
    """`tick` checks `script`: after a rewrite and one check without a further change it
    calls `reload`, and again on the next check while that says busy (BUSY_TRIES times
    in a row at most) or while the script changed during the reload. A reload that
    fails is logged with its traceback and not repeated for the same script."""

    def __init__(
        self, script: Path, reload: Callable[[], Any], off: Callable[[str], None]
    ):
        self._script = script
        self._reload = reload
        self._off = off
        self._seen = _stamp(script)
        self._pending = False
        # The running reload and the script's stamp when it started.
        self._running: tuple[Future[Any], tuple[int, int] | None] | None = None
        self._busy = 0
        self.stopped = False

    def tick(self) -> None:
        if self.stopped:
            return
        stamp = _stamp(self._script)
        if stamp != self._seen:
            self._seen = stamp
            # Gone: being replaced; the next stamp says.
            self._pending = stamp is not None
            self._busy = 0
            return
        if self._running is not None:
            done, started = self._running
            if not done.done():
                return
            self._running = None
            error = done.exception()
            if error is not None:
                _log_failure(error)
            else:
                result = done.result()
                if not isinstance(result, int):
                    self._stop(f"reload_content gave {result!r}, not a number")
                    return
                if result == BUSY:
                    self._busy += 1
                    if self._busy < BUSY_TRIES:
                        return
                    print(
                        f"vsview stayed busy for {BUSY_TRIES} reloads; the rewritten"
                        " script shows after the next change or Reload Script (Ctrl+R).",
                        flush=True,
                    )
            self._busy = 0
            self._pending = self._pending and self._seen != started
            return
        if self._pending:
            try:
                running = self._reload()
            except Exception as error:
                _log_failure(error)
                self._pending = False
                return
            if not isinstance(running, Future):
                self._stop(f"reload_content gave {running!r}, not a future")
                return
            self._running = (running, self._seen)

    def _stop(self, why: str) -> None:
        self.stopped = True
        self._off(why)


def _log_failure(error: BaseException) -> None:
    print("The reload failed; the next change of the script reloads again:", flush=True)
    traceback.print_exception(error, file=sys.stdout)
    sys.stdout.flush()


def _log_off(why: str) -> None:
    print(f"Auto-reload is off ({why}): press Reload Script (Ctrl+R).", flush=True)


def _hooked_window(cli: Any, main: Any, script: Path, title: str) -> type | None:
    """A subclass of vsview's MainWindow that sets `title` and reloads `script`, or None
    (logged) when MainWindow is not as expected."""
    if hasattr(cli, "MainWindow"):
        _log_off("vsview.cli has a MainWindow of its own")
        return None
    window = getattr(main, "MainWindow", None)
    if not isinstance(window, type) or not callable(
        getattr(window, "load_new_script", None)
    ):
        _log_off("vsview.app.main.MainWindow.load_new_script is missing")
        return None
    timer_class = importlib.import_module("PySide6.QtCore").QTimer

    class Window(window):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.setWindowTitle(title)
            self.reloaders: list[Reloader] = []

        def load_new_script(self, path: Path, **vsargs: Any) -> Any:
            loaded = super().load_new_script(path, **vsargs)
            reload = getattr(
                getattr(
                    loaded[0] if isinstance(loaded, tuple) else None, "workspace", None
                ),
                "reload_content",
                None,
            )
            if not callable(reload):
                _log_off("load_new_script gave no workspace with reload_content")
            elif Path(path).resolve() == script.resolve():
                reloader = Reloader(script, reload, _log_off)
                timer = timer_class(self)
                timer.timeout.connect(reloader.tick)
                timer.start(POLL_MS)
                self.reloaders.append(reloader)
            return loaded

    return Window


def main(argv: list[str]) -> None:
    script, title = Path(argv[0]), argv[1]
    try:
        main_module: Any = importlib.import_module("vsview.app.main")
        window = _hooked_window(
            importlib.import_module("vsview.cli"), main_module, script, title
        )
    except ImportError as error:
        _log_off(f"cannot import {error.name or error}")
        window = None
    if window is not None:
        main_module.MainWindow = window
    sys.argv = [sys.argv[0], str(script)]
    runpy.run_module("vsview", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
