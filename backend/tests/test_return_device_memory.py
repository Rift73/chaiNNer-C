"""Clearing a node's resources hands the GPU memory they freed back to the driver
(server.return_device_memory), and never imports torch to do so."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"

# server parses the command line when imported, so the case comes in an environment variable.
CHECK = """
import os
import sys
from types import SimpleNamespace

import server

emptied = []
if os.environ["TORCH_STATE"] == "loaded":
    sys.modules["torch"] = SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: True, empty_cache=lambda: emptied.append(1))
    )
server.return_device_memory()
print(len(emptied), "torch" in sys.modules)
"""


def run(state: str) -> str:
    result = subprocess.run(
        [sys.executable, "-B", "-c", CHECK],
        cwd=SRC,
        env={**os.environ, "TORCH_STATE": state},
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


def test_a_loaded_initialized_torch_is_emptied():
    assert run("loaded") == "1 True"


def test_without_torch_nothing_is_imported():
    assert run("absent") == "0 False"
