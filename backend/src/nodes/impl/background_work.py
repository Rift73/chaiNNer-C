"""Work a preview leaves running in the background, one job at a time. A run waits for it
(the server's wait_for_background_work): it is what the run would do first."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock, Thread

from sanic.log import logger

_one_at_a_time = Lock()
_count_lock = Lock()
_count = 0


def start(name: str, job: Callable[[], object]) -> None:
    """Runs `job` on its own thread. A job that fails only logs why: the run that needs
    its work then does that work itself."""
    _add(1)
    try:
        Thread(target=_run, args=(name, job), name=name, daemon=True).start()
    except RuntimeError:
        _add(-1)
        logger.warning("%s: no thread to run it on", name, exc_info=True)


def pending() -> bool:
    return _count != 0


def _add(change: int) -> None:
    global _count
    with _count_lock:
        _count += change


def _run(name: str, job: Callable[[], object]) -> None:
    try:
        with _one_at_a_time:
            job()
    except Exception:
        logger.warning("%s failed", name, exc_info=True)
    finally:
        _add(-1)
