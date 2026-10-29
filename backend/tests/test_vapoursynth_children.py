"""The worker's VapourSynth child processes: vspipe and the catalog scan end with the
worker, also when it is killed (a crash, the host's kill), and a scan that hangs names
the import it hung in."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from nodes.impl.vapoursynth import catalog, scan

BACKEND_SRC = Path(catalog.__file__).parents[3]
# The interpreter itself: on Windows a venv's python.exe starts it as a child.
PYTHON = getattr(sys, "_base_executable", sys.executable)

# A worker that runs one VapourSynth child: `fake` stands in for vspipe, or for scan.py.
WORKER = """\
import os, sys
from pathlib import Path
from nodes.impl.vapoursynth import catalog, runner

run, fake, folder = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
runner.vspipe = lambda: fake
runner.vs_env = catalog.vs_env = lambda: dict(os.environ)
catalog.vs_python = lambda: Path(sys.argv[4])
catalog.SCAN_PY = fake
(folder / "worker.pid").write_text(str(os.getpid()))
script = folder / "chain.vpy"
if run == "evaluation":
    runner.clip_info(script, 0)
elif run == "frames":
    info = runner.ClipInfo(8, 6, 2, 24, 1, "RGBS")
    list(runner.read_frames(script, 0, 0, 2, info=info))
else:
    catalog.run_scan(folder / "catalog.json", folder / "user", folder / "modules.toml")
"""


def written_pid(path: Path) -> int:
    deadline = time.monotonic() + 30
    while not (path.exists() and path.read_text(encoding="utf-8")):
        assert time.monotonic() < deadline, f"{path.name} was not written"
        time.sleep(0.05)
    return int(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("run", ["evaluation", "frames", "scan"])
def test_children_end_with_a_killed_worker(tmp_path: Path, run: str):
    # A vspipe that renders for minutes (an AI model in the script) or indexes a long
    # video, a user file whose import hangs in the scan.
    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text(
        "import os, time\n"
        "from pathlib import Path\n"
        f"Path({str(tmp_path / 'child.pid')!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    if run == "scan":
        fake = sleeper
    elif os.name == "nt":
        fake = tmp_path / "vspipe.cmd"
        fake.write_text(f'@"{PYTHON}" "{sleeper}"\n', encoding="utf-8")
    else:
        fake = tmp_path / "vspipe"
        fake.write_text(f'#!/bin/sh\nexec "{PYTHON}" "{sleeper}"\n', encoding="utf-8")
        fake.chmod(0o755)
    (tmp_path / "chain.vpy").write_text("# the chain\n", encoding="utf-8")
    worker_script = tmp_path / "worker.py"
    worker_script.write_text(WORKER, encoding="utf-8")

    launched = subprocess.Popen(
        [sys.executable, str(worker_script), run, str(fake), str(tmp_path), PYTHON],
        env={**os.environ, "PYTHONPATH": str(BACKEND_SRC)},
        stdin=subprocess.DEVNULL,
    )
    child = None
    try:
        worker = psutil.Process(written_pid(tmp_path / "worker.pid"))
        child = psutil.Process(written_pid(tmp_path / "child.pid"))
        worker.kill()
        try:
            child.wait(timeout=10)
        except psutil.TimeoutExpired:
            pytest.fail(f"The {run} child outlived its killed worker.")
    finally:
        launched.kill()
        launched.wait()
        if child is not None and child.is_running():
            child.kill()


def test_a_scan_that_hangs_names_the_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    user = tmp_path / "user"
    user.mkdir()
    (user / "quick_user_functions.py").write_text(
        "print('a user file prints too', end='')\n", encoding="utf-8"
    )
    # It also starts a process that keeps the scan's stderr open after the scan ends.
    held = tmp_path / "held.pid"
    (user / "hangs_on_import.py").write_text(
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(60)'], stderr=sys.stderr\n"
        ")\n"
        f"Path({str(held)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    # The scan's import of the user files, without VapourSynth.
    child = tmp_path / "scan_user_files.py"
    child.write_text(
        "import importlib.util, pathlib, sys\n"
        f"spec = importlib.util.spec_from_file_location('scan', {scan.__file__!r})\n"
        "scan = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(scan)\n"
        "scan._import_user_files(pathlib.Path(sys.argv[sys.argv.index('--user-dir') + 1]), [])\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(catalog, "SCAN_PY", child)
    # Time for the child to start and reach the hanging file on a busy machine.
    monkeypatch.setattr(catalog, "SCAN_TIMEOUT", 8)
    monkeypatch.setattr(catalog, "vs_python", lambda: Path(PYTHON))
    monkeypatch.setattr(catalog, "vs_env", lambda: dict(os.environ))

    begin = time.monotonic()
    try:
        with pytest.raises(RuntimeError) as raised:
            catalog.run_scan(tmp_path / "catalog.json", user, tmp_path / "modules.toml")
        assert time.monotonic() - begin < 30
    finally:
        # (Ended by itself after 60 s, when run_scan waited for it.)
        if held.exists() and psutil.pid_exists(pid := int(held.read_text("utf-8"))):
            psutil.Process(pid).kill()
    assert str(raised.value) == (
        "The scan took longer than 8 s; it was still importing the user file"
        " hangs_on_import.py."
    )
