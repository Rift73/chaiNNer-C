"""The worker started with --close-after-start loads its nodes, closes and exits 0.

On Sanic 25, a stop requested before the server serves is lost; the worker's setup
(imports that never yield) can finish that early, which left the worker serving
forever. Its log lines keep Sanic 23's format, which the host parses with
SANIC_LOG_REGEX. The worker is this test's own child, bounded by bounded.run_python.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from bounded import run_python

from server_process_helper import SANIC_LOG_REGEX

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"

Closed = tuple[int, list[str]]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def closed_worker() -> Closed:
    done = run_python(
        ["server.py", str(free_port()), "--close-after-start"], BACKEND, timeout=300
    )
    return done.returncode, done.output.splitlines()


def test_worker_exits_after_start(closed_worker: Closed):
    returncode, lines = closed_worker
    assert returncode == 0, "\n".join(lines[-40:])
    assert any(line.endswith("Server Stopped") for line in lines)


def test_worker_log_lines_keep_sanic_23_format(closed_worker: Closed):
    _, lines = closed_worker
    parsed = [SANIC_LOG_REGEX.match(line) for line in lines]
    messages = [(m.group(1), m.group(2)) for m in parsed if m is not None]
    assert ("INFO", "Loading Nodes...") in messages
    assert ("INFO", "Closing server...") in messages
