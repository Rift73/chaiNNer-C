"""DUAL's export and engine-build child (dual.run_child) ends with its parent's call.

Stop polls context.check_aborted() while the child runs and kills it; a parent that
dies takes it along (Windows kill-on-close job). Before, a Stop during an engine build
waited for the build, the host's /kill restarted the worker after 3 s, and the build
child kept the GPU until it finished. Stub children stand in for the exporter and
dual_engine_worker.py; nothing here needs a GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import psutil
import pytest

from nodes.impl.tensorrt import dual
from progress_controller import Aborted

SRC = Path(__file__).resolve().parents[1] / "src"


def sleeper(pid_file: Path) -> str:
    """A child that records its pid, then would run for a minute."""
    return (
        "import os, time; from pathlib import Path; "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )


def wait_for(condition: Callable[[], object], timeout: float = 30) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


class StopOnceStarted:
    """A NodeContext whose run is stopped once the child has started."""

    def __init__(self, pid_file: Path):
        self.pid_file = pid_file

    def check_aborted(self):
        if self.pid_file.exists() and self.pid_file.read_text():
            raise Aborted()


def test_stop_kills_the_child(tmp_path: Path):
    pid_file = tmp_path / "child.pid"
    started = time.monotonic()
    with pytest.raises(Aborted):
        dual.run_child(
            ["-c", sleeper(pid_file)], "stub", tmp_path, StopOnceStarted(pid_file)
        )
    assert time.monotonic() - started < 30
    assert not psutil.pid_exists(int(pid_file.read_text()))


@pytest.mark.skipif(os.name != "nt", reason="the kill-on-close job is Windows only")
def test_a_parent_that_dies_takes_the_child_along(tmp_path: Path):
    parent_file = tmp_path / "parent.pid"
    child_file = tmp_path / "child.pid"
    parent_code = (
        "import os, sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(SRC)!r}); "
        "from nodes.impl.tensorrt import dual; "
        f"Path({str(parent_file)!r}).write_text(str(os.getpid())); "
        f"dual.run_child(['-c', {sleeper(child_file)!r}], 'stub', Path({str(tmp_path)!r}))"
    )
    top = subprocess.Popen(
        [sys.executable, "-B", "-c", parent_code], stdin=subprocess.DEVNULL
    )
    try:
        assert wait_for(lambda: child_file.exists() and child_file.read_text())
        child = int(child_file.read_text())
        assert psutil.pid_exists(child)
        # As a host's /kill restart or a crash ends the worker: no cleanup runs.
        psutil.Process(int(parent_file.read_text())).kill()
        assert wait_for(lambda: not psutil.pid_exists(child), timeout=10)
    finally:
        try:
            top.wait(timeout=10)
        except subprocess.TimeoutExpired:
            top.kill()
            top.wait(timeout=10)


def test_the_childs_output_and_failure_come_back(tmp_path: Path):
    failing = (
        "import sys; sys.stdout.write('o' * (1 << 20)); sys.stdout.flush(); "
        "sys.stderr.write('tail-marker'); sys.exit(3)"
    )
    with pytest.raises(RuntimeError, match=r"DUAL stub failed \(exit 3\)") as error:
        dual.run_child(["-c", failing], "stub", tmp_path)
    assert str(error.value).endswith("tail-marker")
    assert dual.run_child(["-c", "print('fine')"], "stub", tmp_path).strip() == "fine"
