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


# Deleting a selection sends one /clear-cache/individual per node, each answered at once.
BURST = """
import asyncio
import sys
from types import SimpleNamespace

import server

calls = []
sys.modules["torch"] = SimpleNamespace(
    cuda=SimpleNamespace(is_initialized=lambda: True, empty_cache=lambda: calls.append("empty"))
)
server.gc = SimpleNamespace(collect=lambda: calls.append("collect"))


async def burst():
    for i in range(60):
        request = SimpleNamespace(app=server.app, json={"id": f"node-{i}"})
        await server.clear_cache_individual(request)
    answered = list(calls)
    await asyncio.sleep(0.5)
    return answered


answered = asyncio.run(burst())
print(answered == [], calls)
"""


def run(state: str, check: str = CHECK) -> str:
    result = subprocess.run(
        [sys.executable, "-B", "-c", check],
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


def test_a_burst_of_clears_returns_device_memory_once_after_the_last():
    # Each return is a full collection (tens of ms with a loaded chain), serialized on
    # the worker's loop: 60 deleted nodes made 60 of them.
    assert run("loaded", BURST) == "True ['collect', 'empty']"
