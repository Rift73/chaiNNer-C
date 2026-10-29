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
from typing import cast

import psutil
import pytest

from api import NodeContext
from nodes.impl.pytorch.vosr2_process import PARENT_PID_VARIABLE
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
            ["-c", sleeper(pid_file)],
            "stub",
            tmp_path,
            cast(NodeContext, StopOnceStarted(pid_file)),
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


def late_compiler(started: Path, go: Path, worked: Path) -> str:
    """A child that records its pid, waits for `go` (the seconds that a compiler's slow
    imports take), then asks for the parent-death signal as the compilers do and
    records that it still worked."""
    return (
        "import os, sys, time\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "from nodes.impl.pytorch.vosr2_process import end_with_parent\n"
        f"Path({str(started)!r}).write_text(str(os.getpid()))\n"
        f"while not Path({str(go)!r}).exists():\n"
        "    time.sleep(0.01)\n"
        "end_with_parent()\n"
        f"Path({str(worked)!r}).write_text('worked')\n"
        "time.sleep(60)\n"
    )


def has_ended(pid: int) -> bool:
    """Gone, or a zombie that nothing reaped (an init that does not wait)."""
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


@pytest.mark.skipif(
    sys.platform != "linux", reason="the parent-death signal is Linux only"
)
def test_a_child_whose_parent_died_before_it_asked_for_the_signal_exits(tmp_path: Path):
    """The kernel signals a child only for a parent that ends after the child asked, so
    end_with_parent also checks the parent that run_child names in the environment."""
    parent_file = tmp_path / "parent.pid"
    started = tmp_path / "started.pid"
    go = tmp_path / "go"
    worked = tmp_path / "worked"
    parent_code = (
        "import os, sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(SRC)!r}); "
        "from nodes.impl.tensorrt import dual; "
        f"Path({str(parent_file)!r}).write_text(str(os.getpid())); "
        f"dual.run_child(['-c', {late_compiler(started, go, worked)!r}], 'stub', Path({str(tmp_path)!r}))"
    )
    child = None
    top = subprocess.Popen(
        [sys.executable, "-B", "-c", parent_code], stdin=subprocess.DEVNULL
    )
    try:
        assert wait_for(lambda: started.exists() and started.read_text())
        child = int(started.read_text())
        # As a host's /kill restart or a crash ends the worker: no cleanup runs.
        psutil.Process(int(parent_file.read_text())).kill()
        top.wait(timeout=10)
        assert not has_ended(child)
        go.write_text("go")
        assert wait_for(lambda: has_ended(child), timeout=10)
        assert not worked.exists()
    finally:
        try:
            top.wait(timeout=10)
        except subprocess.TimeoutExpired:
            top.kill()
            top.wait(timeout=10)
        if child is not None and not has_ended(child):
            psutil.Process(child).kill()


def test_the_child_is_given_its_parents_pid_outside_windows(tmp_path: Path):
    output = dual.run_child(
        ["-c", f"import os; print(os.environ.get({PARENT_PID_VARIABLE!r}, 'none'))"],
        "stub",
        tmp_path,
    )
    assert output.strip() == ("none" if os.name == "nt" else str(os.getpid()))


def test_the_childs_output_and_failure_come_back(tmp_path: Path):
    failing = (
        "import sys; sys.stdout.write('o' * (1 << 20)); sys.stdout.flush(); "
        "sys.stderr.write('tail-marker'); sys.exit(3)"
    )
    with pytest.raises(RuntimeError, match=r"^DUAL stub failed \(exit 3\)") as error:
        dual.run_child(["-c", failing], "DUAL stub", tmp_path)
    assert str(error.value).endswith("tail-marker")
    assert dual.run_child(["-c", "print('fine')"], "stub", tmp_path).strip() == "fine"
