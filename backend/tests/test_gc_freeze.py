"""The worker's one gc.freeze after its node loading (gate ruling R1,
the decision log "instant-exec gate R1").

Frozen objects are never collected: the freeze is skipped, and never retried, while a
run is in flight or once a run of a node outside the light set has completed (its
model may sit in the output cache). What runs make later is not frozen; the model a
Load Model reads and the image a Load Image reads are freed by their reference counts,
with the collector off.
"""

from __future__ import annotations

import asyncio
import gc
import importlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    # The module parses its command line when imported.
    monkeypatch.setattr(sys, "argv", ["server.py"])
    module = importlib.import_module("server")
    monkeypatch.setattr(module, "run_tracker", module.RunTracker())
    try:
        yield module
    finally:
        gc.unfreeze()


def freeze(server: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Runs the freeze step of the node loading; returns what it logged."""
    logged: list[str] = []
    monkeypatch.setattr(server, "logger", SimpleNamespace(info=logged.append))
    server.NodeLoading().freeze_objects()
    return logged


def test_the_freeze_moves_the_objects_of_the_start_out_of_the_collectors_reach(
    server: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    (logged,) = freeze(server, monkeypatch)
    assert re.fullmatch(r"gc frozen: \d+ objects", logged)
    assert int(logged.split()[2]) > 0
    assert gc.get_freeze_count() > 0


def test_no_freeze_while_a_run_is_in_flight(
    server: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    server.run_tracker.started()
    assert freeze(server, monkeypatch) == ["freeze skipped: 1 run(s) in flight"]
    # never retried: the run's answer freezes nothing
    server.run_tracker.answered(False)
    assert gc.get_freeze_count() == 0


def test_no_freeze_after_a_run_of_a_node_outside_the_light_set(
    server: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    server.run_tracker.started()
    server.run_tracker.answered(True)
    assert freeze(server, monkeypatch) == [
        "freeze skipped: a run of a node outside the light set has completed"
    ]
    assert gc.get_freeze_count() == 0


def request(path: str, body: object) -> SimpleNamespace:
    return SimpleNamespace(path=path, json=body)


@pytest.mark.parametrize(
    ("path", "body", "heavy"),
    [
        ("/run/individual", {"schemaId": "chainner:image:load"}, False),
        ("/run/individual", {"schemaId": "chainner:pytorch:load_model"}, True),
        (
            "/run",
            {"data": [{"schemaId": "chainner:image:load"}, {"schemaId": "x:y:z"}]},
            True,
        ),
        ("/run", {"data": [{"schemaId": "chainner:image:load"}]}, False),
        # unreadable: counted as heavy
        ("/run", None, True),
    ],
)
def test_a_run_counts_while_its_handler_runs(
    server: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    body: object,
    heavy: bool,
):
    loading = server.NodeLoading()
    loading.light_schemata = frozenset({"chainner:image:load"})
    monkeypatch.setattr(server, "node_loading", loading)
    counted: list[int] = []

    async def handler(_request: object) -> str:
        counted.append(server.run_tracker.in_flight)
        return "reply"

    assert asyncio.run(server.track_run(handler)(request(path, body))) == "reply"
    assert counted == [1]
    assert server.run_tracker.in_flight == 0
    assert server.run_tracker.ran_heavy_node == heavy


def test_a_cancelled_run_stops_counting(
    server: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    # The app's Stop drops its /run request, and Sanic cancels the handler, which
    # skips the response middleware: the freeze would be skipped for good.
    monkeypatch.setattr(server, "node_loading", server.NodeLoading())
    started = asyncio.Event()

    async def handler(_request: object) -> str:
        started.set()
        await asyncio.Event().wait()
        return "never"

    async def stop() -> None:
        run = asyncio.create_task(
            server.track_run(handler)(
                request("/run", {"data": [{"schemaId": "chainner:pytorch:load_model"}]})
            )
        )
        await started.wait()
        assert server.run_tracker.in_flight == 1
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

    asyncio.run(stop())
    assert server.run_tracker.in_flight == 0
    assert server.run_tracker.ran_heavy_node


def test_both_run_routes_are_counted(server: ModuleType):
    assert server.run.__wrapped__ is not None
    assert server.run_individual.__wrapped__ is not None


# After the start's freeze, with the collector off: a CPU-loaded model of the
# default.chn's architecture (a DRFT, built small here) and a Load Image output.
AFTER_FREEZE = """
import gc, json, os, sys, tempfile, weakref
from pathlib import Path
from types import SimpleNamespace

sys.argv = ["server.py"]
# The worker runs server.py: modules find the backend's files by its path.
sys.modules["__main__"].__file__ = os.path.abspath("server.py")
from system import load_system_cpp_runtime

load_system_cpp_runtime()
import cv2
import numpy as np
import torch
from spandrel.architectures.DRFT import DRFT

import server
from packages.chaiNNer_pytorch.pytorch.io.load_model import finish_model, read_model
from packages.chaiNNer_standard.image.io.load_image import load_image_node

folder = Path(tempfile.mkdtemp())
model_file = folder / "drft.pth"
torch.save(
    DRFT(embed_dim=32, depths=(1,), num_heads=(2,), window_size=8, num_feat=16, rank=8,
         upscale=2).state_dict(),
    model_file,
)
image_file = folder / "image.png"
cv2.imwrite(str(image_file), np.full((32, 48, 3), 128, np.uint8))

server.NodeLoading().freeze_objects()
gc.disable()
cpu = SimpleNamespace(device=torch.device("cpu"), use_fp16=False)
descriptor = finish_model(read_model(model_file, torch.device("cpu")), cpu)
image = load_image_node(image_file)[0]
alive = {
    "architecture": descriptor.architecture.name,
    "frozen": gc.get_freeze_count() > 0,
    "descriptor": weakref.ref(descriptor),
    "model": weakref.ref(descriptor.model),
    "image": weakref.ref(image),
}
del descriptor, image
print(json.dumps({k: v() is not None if isinstance(v, weakref.ref) else v
                  for k, v in alive.items()}))
"""


def test_a_model_and_an_image_read_after_the_freeze_die_by_their_reference_counts():
    pytest.importorskip("nodes.impl._chainner_graph")
    done = subprocess.run(
        [sys.executable, "-B", "-c", AFTER_FREEZE],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=300,
        # loading a model starts no GPU
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.splitlines()[-1]) == {
        "architecture": "DRFT",
        "frozen": True,
        "descriptor": False,
        "model": False,
        "image": False,
    }
