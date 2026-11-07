"""A Load Model preview readies the GPU for its model in the background (one tiny upscale),
so that the first run does not wait for the GPU driver. A run waits for that work; its
failure stays in the log."""

from __future__ import annotations

import asyncio
import importlib
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.Compact.__arch.SRVGG import SRVGGNetCompact

from api import NodeContext, NodeId, Progress
from nodes.impl import background_work
from nodes.impl.pytorch import warm_up
from nodes.impl.tensorrt.cache import Preview, in_preview
from packages.chaiNNer_pytorch.pytorch.io import load_model as load_model_module

CUDA = torch.device("cuda")
CPU = torch.device("cpu")


def tiny_model(channels: int = 3) -> ImageModelDescriptor:
    network = SRVGGNetCompact(
        num_in_ch=channels, num_out_ch=channels, num_feat=4, num_conv=2, upscale=2
    )
    model = ModelLoader(CPU).load_from_state_dict(network.state_dict()).eval()
    assert isinstance(model, ImageModelDescriptor)
    return model


def wait_until_done() -> None:
    deadline = time.monotonic() + 30
    while background_work.pending():
        assert time.monotonic() < deadline
        time.sleep(0.001)


def recorded_upscales(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    upscales: list[tuple] = []

    def record(
        image: np.ndarray,
        model: ImageModelDescriptor,
        device: torch.device,
        use_fp16: bool,
        tiler: object,
        progress: Progress,
    ) -> None:
        upscales.append((image.shape, image.dtype, model, device, use_fp16))
        assert not progress.aborted and not progress.paused
        progress.set_progress(1.0)

    monkeypatch.setattr(warm_up, "pytorch_auto_split", record)
    monkeypatch.setattr(warm_up, "safe_cuda_cache_empty", lambda: None)
    return upscales


def test_pending_lasts_until_the_job_ends():
    release = threading.Event()
    background_work.start("job", release.wait)
    assert background_work.pending()
    release.set()
    wait_until_done()


def test_jobs_run_one_at_a_time():
    running = 0
    most = 0
    lock = threading.Lock()

    def job() -> None:
        nonlocal running, most
        with lock:
            running += 1
            most = max(most, running)
        time.sleep(0.01)
        with lock:
            running -= 1

    for _ in range(4):
        background_work.start("job", job)
    wait_until_done()
    assert most == 1


def test_a_failed_job_only_logs_with_its_traceback(monkeypatch: pytest.MonkeyPatch):
    logged: list[tuple] = []
    monkeypatch.setattr(
        background_work,
        "logger",
        SimpleNamespace(
            warning=lambda *args, **kwargs: logged.append((args, kwargs)),
        ),
    )

    def job() -> None:
        raise RuntimeError("the job failed")

    background_work.start("GPU warm-up", job)
    wait_until_done()
    assert logged == [(("%s failed", "GPU warm-up"), {"exc_info": True})]


def test_no_thread_for_a_job_leaves_nothing_pending(monkeypatch: pytest.MonkeyPatch):
    logged: list[tuple] = []
    monkeypatch.setattr(
        background_work,
        "logger",
        SimpleNamespace(warning=lambda *args, **kwargs: logged.append(args)),
    )

    class NoThread:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(background_work, "Thread", NoThread)
    background_work.start("GPU warm-up", lambda: None)
    assert not background_work.pending()
    assert len(logged) == 1


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # The module parses its command line when imported.
    monkeypatch.setattr(sys, "argv", ["server.py"])
    return importlib.import_module("server")


def test_a_run_waits_for_the_work(server: ModuleType):
    release = threading.Event()
    background_work.start("job", release.wait)

    async def wait() -> bool:
        waiting = asyncio.create_task(server.wait_for_background_work(lambda: False))
        await asyncio.sleep(0.05)
        waited = not waiting.done()
        release.set()
        await waiting
        return waited

    assert asyncio.run(wait())
    wait_until_done()


def test_a_stop_ends_the_wait(server: ModuleType):
    release = threading.Event()
    background_work.start("job", release.wait)
    asyncio.run(server.wait_for_background_work(lambda: True))
    assert background_work.pending()
    release.set()
    wait_until_done()


@pytest.mark.parametrize("setting", [True, False])
def test_a_gpu_upscales_a_tiny_image_in_the_runs_precision(
    monkeypatch: pytest.MonkeyPatch, setting: bool
):
    upscales = recorded_upscales(monkeypatch)
    model = tiny_model()
    warm_up.warm_up_in_background(model, CUDA, setting)
    wait_until_done()
    assert upscales == [((64, 64, 3), np.float32, model, CUDA, setting)]


def test_a_model_without_half_precision_warms_up_in_full(
    monkeypatch: pytest.MonkeyPatch,
):
    upscales = recorded_upscales(monkeypatch)
    model = tiny_model()
    monkeypatch.setattr(model, "supports_half", False)
    warm_up.warm_up_in_background(model, CUDA, True)
    wait_until_done()
    assert upscales[0][4] is False


def test_the_cpu_needs_no_warm_up(monkeypatch: pytest.MonkeyPatch):
    upscales = recorded_upscales(monkeypatch)
    warm_up.warm_up_in_background(tiny_model(), CPU, False)
    wait_until_done()
    assert upscales == []


def test_a_model_upscale_image_does_not_take_needs_no_warm_up(
    monkeypatch: pytest.MonkeyPatch,
):
    upscales = recorded_upscales(monkeypatch)
    model = tiny_model()
    monkeypatch.setattr(ImageModelDescriptor, "purpose", "FaceSR")
    warm_up.warm_up_in_background(model, CUDA, True)
    wait_until_done()
    assert upscales == []


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_the_warm_up_runs_the_real_upscale(
    monkeypatch: pytest.MonkeyPatch, channels: int
):
    logged: list[str] = []
    monkeypatch.setattr(warm_up, "logger", SimpleNamespace(info=logged.append))
    warm_up.tiny_upscale(tiny_model(channels), CPU, False)
    assert len(logged) == 1
    assert "Compact" in logged[0] or "RealESRGAN" in logged[0]


@pytest.mark.parametrize("previewing", [True, False])
def test_only_a_preview_warms_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, previewing: bool
):
    model = tiny_model()
    settings = SimpleNamespace(device=CUDA, use_fp16=True)
    started: list[tuple] = []
    monkeypatch.setattr(load_model_module, "check_model_file", lambda _path: None)
    monkeypatch.setattr(load_model_module, "get_settings", lambda _context: settings)
    monkeypatch.setattr(load_model_module, "load_model", lambda _path, _s: model)
    monkeypatch.setattr(
        warm_up, "warm_up_in_background", lambda *args: started.append(args)
    )
    path = tmp_path / "tiny.pth"
    context = cast(NodeContext, SimpleNamespace())
    if previewing:
        output = Preview(NodeId("node")).run(
            load_model_module.load_model_node, context, path
        )
    else:
        output = load_model_module.load_model_node(context, path)
    assert output[0] is model
    assert not in_preview()
    assert started == ([(model, CUDA, True)] if previewing else [])
