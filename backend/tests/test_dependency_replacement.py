"""A dependency that replaces another distribution of the same module, installed by the
dependency manager's installer (dependencies/store.py) with pip itself: in a new virtual
environment, wheel a, then wheel b, which replaces a, both providing module m, as
onnxruntime-migraphx and onnxruntime-directml both provide onnxruntime. Local wheels, no
network."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from dependencies import store

BACKEND_SRC = Path(store.__file__).resolve().parents[1]
CHAINNER_PIP = next((BACKEND_SRC / "dependencies/whls/chainner-pip").glob("*.whl"))
# Seconds a step may take: a new environment installs pip first
TIMEOUT = 300

# Runs in the new environment as the dependency manager runs the installer (with
# progress, through chainner_pip): a, then b.
INSTALL_A_THEN_B = """
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from dependencies import store
from dependencies.store import DependencyInfo


async def progress(message, *_args):
    print("progress:", message)


store.scan_installed_packages()
for name, replaces in (("a", ()), ("b", ("a",))):
    wheel = Path(sys.argv[2]) / f"{name}-1.0-py3-none-any.whl"
    dependency = DependencyInfo(name, "1.0", url=wheel.as_uri(), replaces=replaces)
    asyncio.run(store.install_dependencies([dependency], progress))
"""
# What the environment has afterwards, read by a new process
LISTING = """
import importlib.metadata
import json
from pathlib import Path

import m

print(json.dumps({
    "distributions": [d.metadata["Name"] for d in importlib.metadata.distributions()],
    "owners": importlib.metadata.packages_distributions()["m"],
    "source": m.SOURCE,
    "files": sorted(p.name for p in Path(m.__file__).parent.glob("*.py")),
}))
"""


def record_line(path: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{path},sha256={digest.decode()},{len(data)}"


def build_wheel(folder: Path, name: str, modules: dict[str, str]) -> None:
    """Wheel `name` 1.0 (py3-none-any): package m with these modules, and a RECORD."""
    info = f"{name}-1.0.dist-info"
    files = {
        **{f"m/{module}": text.encode() for module, text in modules.items()},
        f"{info}/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n".encode()
        ),
        f"{info}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: chaiNNer-C tests\n"
            b"Root-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    record = [*(record_line(p, d) for p, d in files.items()), f"{info}/RECORD,,"]
    with zipfile.ZipFile(folder / f"{name}-1.0-py3-none-any.whl", "w") as wheel:
        for path, data in files.items():
            wheel.writestr(path, data)
        wheel.writestr(f"{info}/RECORD", "\n".join(record) + "\n")


def run(command: list[str | Path], env: dict[str, str]) -> str:
    """The command's output; a failure shows all it printed."""
    result = subprocess.run(
        command,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=TIMEOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_the_replacing_wheel_alone_stays_listed_and_provides_the_module(
    tmp_path: Path,
):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    build_wheel(wheels, "a", {"__init__.py": 'SOURCE = "a"\n', "old.py": ""})
    build_wheel(wheels, "b", {"__init__.py": 'SOURCE = "b"\n', "new.py": ""})
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "install_a_then_b.py").write_text(INSTALL_A_THEN_B, encoding="utf-8")
    (scripts / "listing.py").write_text(LISTING, encoding="utf-8")
    # This suite's own paths stay out of the new environment and its pip runs.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONNOUSERSITE"] = "1"

    environment = tmp_path / "venv"
    run([sys.executable, "-m", "venv", environment], env)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--disable-pip-version-check",
            CHAINNER_PIP,
        ],
        env,
    )

    output = run([python, scripts / "install_a_then_b.py", BACKEND_SRC, wheels], env)
    assert "progress: Uninstalling a..." in output.splitlines()
    listing = json.loads(run([python, scripts / "listing.py"], env))
    assert "a" not in listing["distributions"]
    assert listing["distributions"].count("b") == 1
    assert listing["owners"] == ["b"]
    assert listing["source"] == "b"
    # a's own module left with it.
    assert listing["files"] == ["__init__.py", "new.py"]
