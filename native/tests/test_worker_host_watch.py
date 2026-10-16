"""The worker exits when its host process ends.

A host that is killed or crashes runs no cleanup: measured on 2026-10-16, a worker
outlived its force-killed host by 15 s and more, still listening on its port with
905 MB resident. The host passes its pid (--host-pid) and the worker's
exit_with_host waits on it. exit_with_host is compiled alone out of server.py; the
host here is a short-lived child of this test.
"""

from __future__ import annotations

import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
from server_source import compiled, source


def watch(exits: list[int]):
    return compiled(
        "exit_with_host",
        psutil=psutil,
        os=SimpleNamespace(_exit=exits.append),
        logger=SimpleNamespace(warning=lambda message: None),
    )


def test_the_worker_exits_once_its_host_ends():
    host = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdin=subprocess.DEVNULL,
    )
    exits: list[int] = []
    started = time.monotonic()
    watch(exits)(host.pid)
    assert host.wait(timeout=30) == 0
    assert exits == [1]
    assert time.monotonic() - started < 30


def test_a_host_already_gone_ends_the_worker_at_once():
    host = subprocess.Popen([sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL)
    host.wait(timeout=30)
    exits: list[int] = []
    watch(exits)(host.pid)
    assert exits == [1]


def test_the_host_passes_its_pid_to_the_worker():
    assert '"--host-pid", str(os.getpid())' in source("server_process_helper.py")
