"""A child Python process with a bounded wait; on a timeout its tree is killed.

The venv's python.exe is a launcher whose child is the interpreter, so a timeout
kills the whole process tree by the launcher's PID: killing the launcher alone would
leave the interpreter running.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


class Finished:
    def __init__(self, returncode: int, output: str) -> None:
        self.returncode = returncode
        self.output = output


def run_python(
    args: list[str],
    cwd: Path,
    timeout: float,
    *,
    python: Path | None = None,
    env: dict[str, str] | None = None,
) -> Finished:
    """Runs python (this interpreter by default) with args, in this environment
    plus env; stdout and stderr are combined."""
    child = subprocess.Popen(
        [str(python or sys.executable), "-B", *args],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, **(env or {}), "PYTHONIOENCODING": "utf-8"},
    )
    try:
        output, _ = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["taskkill", "/PID", str(child.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        output, _ = child.communicate()
        raise TimeoutError(f"no exit within {timeout} s:\n{output[-4000:]}") from None
    return Finished(child.returncode, output)
