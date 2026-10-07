"""CHAINNER_C_NUMPY_POOL (SP4c): the pyd's NumPy data-memory handler, per thread.

The pyd reads the variable once, at the first install: unset or empty = installed
with the default cap, min(512 MiB, 10 % of physical memory); 0 = not installed, so
NumPy's default handler serves every request exactly as before SP4c; other ASCII
digits = installed with that cap in MiB. Any other value, or a NumPy without the
handler API, leaves the handler uninstalled, with one warning per process.

NumPy keeps the handler in a context variable that a new thread does not inherit, so
every thread that allocates node outputs installs it: the worker's main thread before
its event loop starts (the loop's tasks copy that context), and each of the worker's
executor threads through its initializer (run_in_executor copies no context). An
array keeps the handler that allocated it and is freed through it on any thread.

The worker releases every idle block at the end of each /run, and once each
/run/individual's broadcasts are sent and on /clear-cache/individual while no /run is
executing. With CHAINNER_C_PROFILE=1 the first install attaches the pool's counters
to native_profile's line, as "numpy_pool".
"""

from __future__ import annotations

import threading

from sanic.log import logger

from . import native_profile
from .native_graph import graph

VARIABLE = "CHAINNER_C_NUMPY_POOL"

# Installs run on several threads; each flag below is set once per process, under it.
_lock = threading.Lock()
_installed = False
_warned = False  # the one warning for an "invalid" or "unsupported" status
_failed = False  # the one error for an install that raised
_release_warned = False  # the one warning for VirtualFree failures


def install() -> None:
    """Install the handler on the calling thread. It never raises.

    It runs as the initializer of every worker executor, where a raise would break
    that executor for every later submit, and in the worker's main(), where it would
    stop the worker. An exception from the pyd is logged once, with its traceback,
    and the thread keeps NumPy's default handler.
    """
    global _installed, _warned, _failed
    try:
        module = graph()
        status, message = module.numpy_pool_install()
    except Exception:
        with _lock:
            first = not _failed
            _failed = True
        if first:
            logger.exception(
                "NumPy pool: install failed; the thread keeps NumPy's default handler"
            )
        return
    if status == "installed":
        with _lock:
            if not _installed:
                _installed = True
                if native_profile.enabled():
                    native_profile.attach_numpy_pool(
                        module.numpy_pool_stats, module.numpy_pool_reset
                    )
    elif status in ("invalid", "unsupported"):
        with _lock:
            first = not _warned
            _warned = True
        if first:
            logger.warning(message)


def installed() -> bool:
    """Whether some thread of this process installed the handler."""
    return _installed


def release() -> int:
    """Release every idle block; the bytes released, 0 when not installed.

    The first time the process's VirtualFree failures are non-zero, one warning says
    so; the pyd counts them for the life of the process.
    """
    global _release_warned
    if not _installed:
        return 0
    released, failures = graph().numpy_pool_release()
    if failures:
        with _lock:
            first = not _release_warned
            _release_warned = True
        if first:
            logger.warning(
                "NumPy pool: VirtualFree failed %d time(s) in this process; "
                "those blocks were not returned to the OS",
                failures,
            )
    return released
