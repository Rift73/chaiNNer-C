"""Clearing a node's resources hands the GPU memory they freed back to the driver
(server.return_device_memory), and never imports torch to do so. So does a stopped
run: the app's Stop drops its /run request, so the run ends cancelled. A CUDA context
with a sticky error frees nothing; the worker then asks the host to restart it."""

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
        cuda=SimpleNamespace(
            is_initialized=lambda: True,
            device_count=lambda: 0,
            empty_cache=lambda: emptied.append(1),
        )
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
    cuda=SimpleNamespace(
        is_initialized=lambda: True,
        device_count=lambda: 0,
        empty_cache=lambda: calls.append("empty"),
    )
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


# A cleared or removed node lets go of what it holds open: Load Video's Preview reader.
PREVIEW = """
import asyncio
from types import SimpleNamespace

import server
from nodes.impl import video

closed = []
video.close_preview = closed.append
request = SimpleNamespace(app=server.app, json={"id": "load-video-1"})
asyncio.run(server.clear_cache_individual(request))
print(closed)
"""


# A reader that fails to close: the clear still drops the node's output and returns its
# memory, then answers with the close's error.
PREVIEW_FAILS = """
import asyncio
import json
from types import SimpleNamespace

import server
from nodes.impl import video


def fail(node_id):
    raise OSError("the reader did not close")


video.close_preview = fail
returned = []
server.return_device_memory_after_burst = returned.append
ctx = server.AppContext.get(server.app)
ctx.cache["load-video-1"] = object()
request = SimpleNamespace(app=server.app, json={"id": "load-video-1"})
response = asyncio.run(server.clear_cache_individual(request))
print("load-video-1" in ctx.cache, returned == [ctx], json.loads(response.body))
"""


# The app's Stop: /run's handler is cancelled while its executor runs.
STOPPED = """
import asyncio
import contextlib
import sys
from types import SimpleNamespace

import server

calls = []
sys.modules["torch"] = SimpleNamespace(
    cuda=SimpleNamespace(
        is_initialized=lambda: True,
        device_count=lambda: 0,
        empty_cache=lambda: calls.append("empty"),
    )
)
server.gc = SimpleNamespace(collect=lambda: calls.append("collect"))
server.parse_json = lambda _data: SimpleNamespace(nodes={})
server.optimize = lambda _chain: None
server.ExecutionOptions = SimpleNamespace(parse=lambda _options: None)


class Executor:
    def __init__(self, **_):
        pass

    async def run(self):
        await asyncio.sleep(60)


# Either executor CHAINNER_C_EXECUTOR selects.
server.Executor = Executor
server.NativeExecutor = Executor
# Sanic gives its loop only to a serving app; this one is not served.
type(server.app).loop = property(lambda _app: asyncio.get_running_loop())


async def stop():
    request = SimpleNamespace(
        app=server.app, json={"data": [], "options": {}, "sendBroadcastData": False}
    )
    handler = asyncio.ensure_future(server.run(request))
    await asyncio.sleep(0.1)
    handler.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await handler
    await asyncio.sleep(0.1)


asyncio.run(stop())
print(calls)
"""


# A sticky CUDA error: synchronizing the context fails, and so does freeing memory.
BROKEN = """
import sys
from types import SimpleNamespace

import server


def failing():
    raise RuntimeError("CUDA error: an illegal memory access was encountered")


def cuda(synchronize):
    return SimpleNamespace(
        is_initialized=lambda: True,
        device_count=lambda: 2,
        memory_reserved=lambda device: 0 if device == 1 else 1 << 30,
        synchronize=synchronize,
        empty_cache=failing,
    )


sys.modules["torch"] = SimpleNamespace(cuda=cuda(lambda device: None))
healthy = server.restart_headers()
sys.modules["torch"] = SimpleNamespace(cuda=cuda(lambda device: failing()))
broken = server.restart_headers()
server.collect_failed_run(server.AppContext.get(server.app))  # logs, does not raise
print(healthy == {}, broken == {server.RESTART_WORKER_HEADER: "1"})
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


def test_a_stopped_run_returns_device_memory():
    # The app's Stop cancels /run's handler; the run used to end without returning
    # memory, and a 4x upscale left 27.7 GB taken until chaiNNer was closed.
    assert run("loaded", STOPPED) == "['collect', 'empty']"


def test_a_broken_cuda_context_asks_the_host_to_restart_the_worker():
    assert run("loaded", BROKEN) == "True True"


def test_clearing_a_node_closes_its_preview_reader():
    assert run("absent", PREVIEW) == "['load-video-1']"


def test_a_reader_that_fails_to_close_still_clears_the_node():
    error = "{'success': False, 'error': 'the reader did not close'}"
    assert run("absent", PREVIEW_FAILS) == f"False True {error}"
