"""PyTorch, ONNX and NCNN Upscale Image may run ahead of earlier items
(LOCKED_DEVICE_NODES): each runs its whole upscale on the thread of its device, the
GPU's or the CPU's (nodes/impl/device_thread.py), so items computed ahead take the
device one at a time from one calling thread. Fakes stand in for the models and
sessions; no GPU."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import BrokenThreadPool
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from api import Aborted, NodeContext
from nodes.impl import device_thread
from nodes.impl.device_thread import on_device_thread
from nodes.impl.execution_scheduler import LOCKED_DEVICE_NODES
from nodes.impl.upscale.auto_split_tiles import NO_TILING
from nodes.impl.upscale.basic_upscale import PaddingType
from packages.chaiNNer_ncnn.ncnn.processing import upscale_image as ncnn_node
from packages.chaiNNer_ncnn.settings import CPU_DEVICE
from packages.chaiNNer_onnx.onnx.processing import upscale_image as onnx_node
from packages.chaiNNer_pytorch.pytorch.processing import upscale_image as torch_node
from util import waited_seconds, waiting

CONTEXT = cast(
    NodeContext,
    SimpleNamespace(add_cleanup=lambda *_, **__: None, suspend=lambda: None),
)
IMAGE = np.zeros((4, 5, 3), np.float32)


def on_device(gpu: bool) -> bool:
    """Whether this runs on the device's own thread."""
    name = threading.current_thread().name
    return name.startswith("gpu-upscale_" if gpu else "cpu-upscale_")


def test_the_upscale_nodes_may_run_ahead():
    assert {
        "chainner:tensorrt:upscale_image",
        "chainner:migraphx:upscale_image",
        "chainner:pytorch:upscale_image",
        "chainner:onnx:upscale_image",
        "chainner:ncnn:upscale_image",
    } == LOCKED_DEVICE_NODES


@pytest.mark.parametrize(("device", "gpu"), [("cpu", False), ("cuda:1", True)])
def test_pytorch_upscales_on_its_device_thread(monkeypatch, device, gpu):
    seen: list[bool] = []
    settings = SimpleNamespace(device=torch.device(device), force_cache_wipe=False)
    monkeypatch.setattr(torch_node, "get_settings", lambda _: settings)

    def upscale(img: np.ndarray, *_: Any, **__: Any) -> np.ndarray:
        seen.append(on_device(gpu))
        return img

    monkeypatch.setattr(torch_node, "basic_upscale", upscale)
    model = cast(Any, SimpleNamespace(input_channels=3, output_channels=3, scale=1))
    torch_node.upscale_image_node(
        CONTEXT, IMAGE, model, False, 1, NO_TILING, 256, PaddingType.NONE, False
    )
    assert seen == [True]


@pytest.mark.parametrize(
    ("provider", "gpu"),
    [
        ("CPUExecutionProvider", False),
        ("CUDAExecutionProvider", True),
        ("MIGraphXExecutionProvider", True),
    ],
)
def test_onnx_creates_and_runs_its_session_on_its_device_thread(
    monkeypatch, provider, gpu
):
    seen: list[bool] = []
    settings = SimpleNamespace(
        gpu_index=2,
        execution_provider=provider,
        tensorrt_fp16_mode=False,
        tensorrt_cache_path=None,
        migraphx_fp16_mode=False,
        migraphx_cache_path=None,
        migraphx_exhaustive_tune=False,
    )
    monkeypatch.setattr(onnx_node, "get_settings", lambda _: settings)
    monkeypatch.setattr(
        onnx_node, "get_onnx_session", lambda *_: seen.append(on_device(gpu))
    )

    def upscale(img: np.ndarray, *_: Any, **__: Any) -> np.ndarray:
        seen.append(on_device(gpu))
        return img

    monkeypatch.setattr(onnx_node, "upscale_with_session", upscale)
    onnx_node.upscale_image_node(CONTEXT, IMAGE, cast(Any, None), NO_TILING, 256, False)
    assert seen == [True, True]


@pytest.mark.parametrize(
    ("vulkan", "gpu_index", "gpu"),
    [(False, 3, False), (True, 3, True), (True, CPU_DEVICE, False)],
)
def test_ncnn_upscales_on_its_device_thread(monkeypatch, vulkan, gpu_index, gpu):
    # Without Vulkan, or with the CPU chosen in the settings, NCNN runs on the CPU.
    monkeypatch.setattr(ncnn_node, "use_gpu", vulkan)
    seen: list[bool] = []
    monkeypatch.setattr(
        ncnn_node, "get_settings", lambda _: SimpleNamespace(gpu_index=gpu_index)
    )

    def upscale(img: np.ndarray, *_: Any) -> np.ndarray:
        seen.append(on_device(gpu))
        return img

    monkeypatch.setattr(ncnn_node, "convenient_upscale", upscale)
    model = cast(Any, SimpleNamespace(in_nc=3, out_nc=3))
    ncnn_node.upscale_image_node(CONTEXT, IMAGE, model, NO_TILING, 256, False)
    assert seen == [True]


def test_calls_from_many_threads_run_one_at_a_time_on_one_thread_per_device():
    # The frameworks keep state per calling thread: the engine's workers must not each
    # make one.
    threads: list[tuple[bool, int]] = []
    running = {True: 0, False: 0}
    most = {True: 0, False: 0}
    count = threading.Lock()

    def upscale(gpu: bool) -> tuple[bool, int]:
        with count:
            running[gpu] += 1
            most[gpu] = max(most[gpu], running[gpu])
        time.sleep(0.01)
        with count:
            running[gpu] -= 1
        return gpu, threading.get_ident()

    def call(gpu: bool) -> None:
        threads.append(on_device_thread(gpu, lambda: upscale(gpu)))

    callers = [threading.Thread(target=call, args=(gpu,)) for gpu in [True, False] * 4]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join()
    assert len(threads) == 8 and len(set(threads)) == 2
    assert {gpu for gpu, _ in threads} == {True, False}
    assert threading.get_ident() not in {ident for _, ident in threads}
    assert most == {True: 1, False: 1}


def test_a_call_s_error_reaches_its_caller():
    def fail() -> None:
        raise ValueError("inference failed")

    with pytest.raises(ValueError, match="inference failed"):
        on_device_thread(True, fail)
    assert on_device_thread(True, lambda: 5) == 5


def test_the_wait_for_the_device_is_waiting_and_the_call_is_not():
    # As `held` for a lock: the run engine leaves the wait out of the node's time.
    busy = threading.Event()
    release = threading.Event()
    waited: list[float] = []

    def occupy() -> None:
        busy.set()
        release.wait()

    def caller() -> None:
        before = waited_seconds()
        on_device_thread(True, lambda: time.sleep(0.3))
        waited.append(waited_seconds() - before)

    first = threading.Thread(target=on_device_thread, args=(True, occupy))
    first.start()
    busy.wait()
    second = threading.Thread(target=caller)
    second.start()
    time.sleep(0.3)
    release.set()
    first.join()
    second.join()
    assert 0.2 <= waited[0] < 0.6
    before = waited_seconds()
    on_device_thread(True, lambda: time.sleep(0.3))
    assert waited_seconds() - before < 0.2


def stopped() -> NodeContext:
    def suspend() -> None:
        raise Aborted()

    return cast(
        NodeContext, SimpleNamespace(add_cleanup=lambda *_, **__: None, suspend=suspend)
    )


def test_a_queued_upscale_suspends_before_it_starts(monkeypatch):
    # Paused or stopped while queued behind other upscales: each node waits, or ends,
    # before its session, model or tiles (ONNX's and NCNN's tile loops check only Stop).
    def started(*_: Any, **__: Any) -> None:
        pytest.fail("the upscale started")

    settings = SimpleNamespace(
        device=torch.device("cpu"),
        force_cache_wipe=False,
        gpu_index=0,
        execution_provider="CPUExecutionProvider",
    )
    for module in (torch_node, onnx_node, ncnn_node):
        monkeypatch.setattr(module, "get_settings", lambda _: settings)
    monkeypatch.setattr(torch_node, "basic_upscale", started)
    monkeypatch.setattr(onnx_node, "get_onnx_session", started)
    monkeypatch.setattr(ncnn_node, "convenient_upscale", started)
    model = cast(Any, SimpleNamespace(input_channels=3, output_channels=3, scale=1))
    with pytest.raises(Aborted):
        torch_node.upscale_image_node(
            stopped(), IMAGE, model, False, 1, NO_TILING, 256, PaddingType.NONE, False
        )
    with pytest.raises(Aborted):
        onnx_node.upscale_image_node(
            stopped(), IMAGE, cast(Any, None), NO_TILING, 256, False
        )
    with pytest.raises(Aborted):
        ncnn_node.upscale_image_node(stopped(), IMAGE, model, NO_TILING, 256, False)


def finishes(call: Any) -> list[object]:
    """call()'s outcome, or nothing when it hangs past 5 s."""
    outcome: list[object] = []

    def run() -> None:
        try:
            outcome.append(call())
        except Exception as error:
            outcome.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(5)
    return outcome


def test_a_call_on_the_device_thread_runs_at_once():
    def nested() -> int:
        return on_device_thread(True, lambda: 7)

    assert finishes(lambda: on_device_thread(True, nested)) == [7]


def test_a_device_thread_that_never_runs_the_call_raises(monkeypatch):
    def broken() -> None:
        raise RuntimeError("no thread")

    pool = ThreadPoolExecutor(max_workers=1, initializer=broken)
    monkeypatch.setitem(device_thread._threads, True, pool)
    monkeypatch.setitem(device_thread._idents, True, -1)
    [outcome] = finishes(lambda: on_device_thread(True, lambda: 7))
    assert isinstance(outcome, BrokenThreadPool)


def test_a_wait_inside_the_call_is_the_caller_s_waiting():
    # A shared session's or model's lock taken on the device thread.
    def call() -> None:
        with waiting():
            time.sleep(0.3)

    before = waited_seconds()
    on_device_thread(True, call)
    assert waited_seconds() - before >= 0.25
