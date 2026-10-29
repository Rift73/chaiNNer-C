"""run.py starts the worker only once the host's server dependencies are installed.

The worker started first, before the host ran pip: on a first start pip then replaced
modules the worker had loaded (NumPy's, psutil's .pyd), which fails on Windows or
leaves ~-prefixed folders behind. The installed packages are now checked first (a
metadata scan, no pip), pip runs when one is missing, and the worker starts after it.

The leftovers of interrupted installs and downloads are swept in between: the sweep
loads psutil, whose .pyd pip cannot replace once a process has it loaded (Windows), so
a sweep before pip failed the start of an update that raised psutil's pin. The sweep is
best effort: an error in it is logged with its traceback, and the start goes on.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"

# run.py with stand-ins for the worker, the host, pip and the leftovers' sweep (which
# fails when told to); the environment's installed distributions are a JSON file, which
# pip's stand-in adds the pins it installs to.
CHECK = """
import json
import runpy
import subprocess
import sys
from types import SimpleNamespace

from dependencies import store

ENVIRONMENT, SWEEP = sys.argv[1:3]
sys.argv = ["run.py"]
events = []


def scan():
    with open(ENVIRONMENT, encoding="utf-8") as f:
        store.installed_packages.clear()
        store.installed_packages.update(json.load(f))


def pip(command, **_kwargs):
    events.append("pip")
    with open(ENVIRONMENT, encoding="utf-8") as f:
        installed = json.load(f)
    for pin in command[4:]:
        if pin.endswith(".whl"):
            name, version = pin.rsplit("/", 1)[1].split("-")[:2]
        elif "==" in pin:
            name, version = pin.split("==")
        else:
            continue
        installed[store.distribution_name(name)] = version
    with open(ENVIRONMENT, "w", encoding="utf-8") as f:
        json.dump(installed, f)
    return 0


def sweep(_storage_dir, _prefix, _executable):
    events.append("sweep")
    if SWEEP == "fails":
        raise OSError("the storage folder cannot be listed")


def spawn_worker(_args):
    events.append("spawn")
    return SimpleNamespace(kill=lambda: events.append("kill"))


def main(worker):
    events.append(("host", "with worker" if worker is not None else "no worker"))


store.scan_installed_packages = scan
subprocess.check_call = pip
sys.modules["leftovers"] = SimpleNamespace(remove=sweep)
sys.modules["worker_spawn"] = SimpleNamespace(
    find_free_port=lambda: 1, spawn_worker=spawn_worker
)
sys.modules["server_host"] = SimpleNamespace(main=main)
runpy.run_path("run.py", run_name="__main__")
print(json.dumps(events))
"""


def start(environment: Path, sweep: str = "works") -> tuple[list[object], str]:
    """The start's events, and what it wrote to stderr."""
    result = subprocess.run(
        [sys.executable, "-B", "-c", CHECK, str(environment), sweep],
        cwd=SRC,
        env=os.environ,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1]), result.stderr


def test_the_worker_starts_after_the_first_start_install(tmp_path: Path):
    environment = tmp_path / "installed.json"
    environment.write_text("{}", encoding="utf-8")
    # A first start: pip runs before any worker exists, then the sweep, then the
    # worker starts once.
    events, _ = start(environment)
    assert events == ["pip", "sweep", "spawn", ["host", "with worker"]]
    # The next start finds them installed: no pip, the sweep and the worker at once.
    events, _ = start(environment)
    assert events == ["sweep", "spawn", ["host", "with worker"]]


def test_a_failing_sweep_is_logged_and_the_start_goes_on(tmp_path: Path):
    environment = tmp_path / "installed.json"
    environment.write_text("{}", encoding="utf-8")
    events, stderr = start(environment, sweep="fails")
    assert events == ["pip", "sweep", "spawn", ["host", "with worker"]]
    assert "Traceback (most recent call last)" in stderr
    assert "OSError: the storage folder cannot be listed" in stderr
