"""run.py, the backend host's entry, writes its log as UTF-8 to a pipe, so a line with a
character outside the ANSI code page reaches the app instead of being dropped, and the
worker's log lines reach the host even with a lone surrogate (upstream chaiNNer #1154)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from server_process_helper import ENV, SANIC_LOG_REGEX

SRC = Path(__file__).parent.parent / "src"

# Stands in for server_host: logs one line through the host's log configuration.
FAKE_HOST = """
import logging.config

from sanic.log import logger

from server_config import LOG_CONFIG


def main():
    logging.config.dictConfig(LOG_CONFIG)
    logger.info("Loaded C:/images/日本.png")
"""


def test_host_log_line_outside_the_code_page_is_written_as_utf8(tmp_path: Path):
    # run.py next to stand-ins for the modules it starts, which shadow the real ones.
    shutil.copy(SRC / "run.py", tmp_path / "run.py")
    (tmp_path / "dependencies").mkdir()
    (tmp_path / "dependencies" / "__init__.py").write_text("")
    (tmp_path / "dependencies" / "install_server_deps.py").write_text("")
    (tmp_path / "server_host.py").write_text(FAKE_HOST, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUTF8"}
    # The app pipes the host's output; a Western-European Windows gives that pipe
    # cp1252 (Chinese Windows cp936), which cannot encode the line.
    env["PYTHONIOENCODING"] = "cp1252"
    env["PYTHONPATH"] = str(SRC)
    done = subprocess.run(
        [sys.executable, "-B", str(tmp_path / "run.py")],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=env,
        timeout=120,
        check=True,
    )
    assert b"Logging error" not in done.stderr, done.stderr.decode("utf-8", "replace")
    assert "[INFO] Loaded C:/images/日本.png".encode() in done.stdout


# Stands in for the worker (server.py): logs a file name Windows could not decode,
# which Python holds as a lone surrogate, through the worker's log configuration.
FAKE_WORKER = """
import logging.config

from sanic.log import logger

from server_config import LOG_CONFIG

logging.config.dictConfig(LOG_CONFIG)
logger.info("Loaded C:/images/\\udce9.png")
"""


def test_worker_log_line_with_a_lone_surrogate_reaches_the_host():
    # The worker's environment, read the way the host reads the worker's output.
    env = {k: v for k, v in ENV.items() if k != "PYTHONUTF8"}
    env["PYTHONPATH"] = str(SRC)
    done = subprocess.run(
        [sys.executable, "-B", "-c", FAKE_WORKER],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        env=env,
        timeout=120,
        check=True,
    )
    assert "Logging error" not in done.stderr, done.stderr
    logged = [
        match.groups()
        for line in done.stdout.splitlines()
        if (match := SANIC_LOG_REGEX.match(line))
    ]
    # Escaped as backslashreplace writes it.
    assert logged == [("INFO", "Loaded C:/images/\\udce9.png")]
