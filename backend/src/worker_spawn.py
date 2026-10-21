"""How the host starts its worker process.

The standard library only: run.py starts the worker with it first, before the host
checks its own dependencies and imports Sanic, so the two processes start side by side.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

# backslashreplace: a log line with a lone surrogate (an undecodable Windows file
# name) is written escaped instead of being dropped.
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8:backslashreplace"}


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


def spawn_worker(args: list[str]) -> subprocess.Popen[str]:
    """Starts server.py with `args` (its port, then its flags), its output piped."""
    server_file = os.path.join(os.path.dirname(__file__), "server.py")
    python_location = sys.executable

    return subprocess.Popen(
        # A host that is killed or crashes stops nothing: the worker watches it.
        [python_location, server_file, *args, "--host-pid", str(os.getpid())],
        shell=False,
        stdin=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        # A byte that is not UTF-8 (native code, a tool the worker runs) is shown
        # escaped instead of ending the reader threads.
        errors="backslashreplace",
        env=ENV,
    )
