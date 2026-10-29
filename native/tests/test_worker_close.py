"""The worker started with --close-after-start loads its nodes, closes and exits 0.

On Sanic 25, a stop requested before the server serves is lost; the worker's setup
(imports that never yield) can finish that early, which left the worker serving
forever. Its log lines keep Sanic 23's format, which the host parses with
SANIC_LOG_REGEX. The worker is this test's own child, bounded by bounded.run_python,
with a storage folder of its own: without --storage-dir it writes into the user's
backend-storage (its temporary files).
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from bounded import run_python

from server_process_helper import SANIC_LOG_REGEX

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"

Closed = tuple[int, list[str], Path]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def closed_worker(tmp_path_factory: pytest.TempPathFactory) -> Closed:
    storage = tmp_path_factory.mktemp("storage")
    done = run_python(
        [
            "server.py",
            str(free_port()),
            "--close-after-start",
            "--storage-dir",
            str(storage),
        ],
        BACKEND,
        timeout=300,
    )
    return done.returncode, done.output.splitlines(), storage


def test_worker_exits_after_start(closed_worker: Closed):
    returncode, lines, _ = closed_worker
    assert returncode == 0, "\n".join(lines[-40:])
    assert any(line.endswith("Server Stopped") for line in lines)


def test_worker_keeps_to_its_storage_folder(closed_worker: Closed):
    _, lines, storage = closed_worker
    assert any(
        line.endswith(f"Using given storage directory: {storage}") for line in lines
    ), "\n".join(lines[-40:])


def test_worker_log_lines_keep_sanic_23_format(closed_worker: Closed):
    _, lines, _ = closed_worker
    parsed = [SANIC_LOG_REGEX.match(line) for line in lines]
    messages = [(m.group(1), m.group(2)) for m in parsed if m is not None]
    assert ("INFO", "Loading Nodes...") in messages
    assert ("INFO", "Closing server...") in messages
