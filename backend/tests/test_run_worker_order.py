"""run.py starts the worker only once the host's server dependencies are installed.

The worker started first, before the host ran pip: on a first start pip then replaced
modules the worker had loaded (NumPy's, psutil's .pyd), which fails on Windows or
leaves ~-prefixed folders behind. The installed packages are now checked first (a
metadata scan, no pip), pip runs when one is missing, and the worker starts after it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"

# run.py with stand-ins for the worker, the host and pip; the environment's installed
# distributions are a JSON file, which pip's stand-in adds the pins it installs to.
CHECK = """
import json
import runpy
import subprocess
import sys
from types import SimpleNamespace

from dependencies import store

ENVIRONMENT = sys.argv[1]
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


def spawn_worker(_args):
    events.append("spawn")
    return SimpleNamespace(kill=lambda: events.append("kill"))


def main(worker):
    events.append(("host", "with worker" if worker is not None else "no worker"))


store.scan_installed_packages = scan
subprocess.check_call = pip
sys.modules["worker_spawn"] = SimpleNamespace(
    find_free_port=lambda: 1, spawn_worker=spawn_worker
)
sys.modules["server_host"] = SimpleNamespace(main=main)
runpy.run_path("run.py", run_name="__main__")
print(json.dumps(events))
"""


def start(environment: Path) -> list[object]:
    result = subprocess.run(
        [sys.executable, "-B", "-c", CHECK, str(environment)],
        cwd=SRC,
        env=os.environ,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_worker_starts_after_the_first_start_install(tmp_path: Path):
    environment = tmp_path / "installed.json"
    environment.write_text("{}", encoding="utf-8")
    # A first start: pip runs before any worker exists, which then starts once.
    assert start(environment) == ["pip", "spawn", ["host", "with worker"]]
    # The next start finds them installed: no pip, the worker starts at once.
    assert start(environment) == ["spawn", ["host", "with worker"]]
