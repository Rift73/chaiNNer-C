"""Stop ends a PyTorch upscale between the model's blocks. A tile's forward otherwise
runs to its end after Stop and keeps its memory until then: minutes when the GPU's
memory overflows into shared memory."""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch
from spandrel import ImageModelDescriptor
from spandrel.architectures.ESRGAN import ESRGAN, ESRGANArch

# Tiling runs in the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

import nodes.impl.pytorch.auto_split as torch_split
from api import Aborted, Progress
from nodes.impl.pytorch.auto_split import pytorch_auto_split
from nodes.impl.upscale.tiler import NoTiling

CPU = torch.device("cpu")


class StopProgress(Progress):
    """A node's progress whose Stop the test presses."""

    def __init__(self) -> None:
        self.stopped = False

    @property
    def aborted(self) -> bool:
        return self.stopped

    @property
    def paused(self) -> bool:
        return False

    def set_progress(self, progress: float) -> None:
        self.check_aborted()


def esrgan() -> ImageModelDescriptor[torch.nn.Module]:
    torch.manual_seed(0)
    net = ESRGAN(num_filters=8, num_blocks=3, scale=2)
    model = ESRGANArch().load(net.state_dict())
    assert isinstance(model, ImageModelDescriptor)
    return model


def rrdbs(model: ImageModelDescriptor[torch.nn.Module]) -> list[torch.nn.Module]:
    return [m for m in model.model.modules() if type(m).__name__ == "RRDB"]


def image() -> np.ndarray:
    return np.random.default_rng(0).random((24, 32, 3), dtype=np.float32)


def upscale(model: ImageModelDescriptor[torch.nn.Module], progress: Progress):
    return pytorch_auto_split(image(), model, CPU, False, NoTiling(), progress)


def test_stop_during_a_forward_ends_it_at_the_next_block():
    model = esrgan()
    progress = StopProgress()
    first, *rest = rrdbs(model)
    ran: list[torch.nn.Module] = []

    def press_stop(_module: torch.nn.Module, _args: object, _output: object) -> None:
        progress.stopped = True

    first.register_forward_hook(press_stop)
    for block in rest:
        for conv in block.modules():
            if isinstance(conv, torch.nn.Conv2d):
                conv.register_forward_pre_hook(lambda m, _args: ran.append(m))

    with pytest.raises(Aborted) as error:
        upscale(model, progress)

    assert ran == []
    # Raised outside the forward: its frames, the activations, are not kept.
    frames = []
    traceback = error.value.__traceback__
    while traceback is not None:
        frames.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert "forward" not in frames
    assert error.value.__context__ is None


def test_the_checks_leave_the_model_as_it_was(monkeypatch: pytest.MonkeyPatch):
    model = esrgan()
    before = model(torch.zeros(1, 3, 8, 8))
    checks: list[torch.nn.Module] = []
    monkeypatch.setattr(
        torch_split, "_stop_between_blocks", lambda m, _args: checks.append(m)
    )
    upscale(model, StopProgress())
    during = len(checks)
    after = model(torch.zeros(1, 3, 8, 8))

    # Checked during the upscale, not by a later use of the model.
    assert during > 0 and len(checks) == during
    torch.testing.assert_close(after, before)


def test_another_runs_forward_of_the_same_model_is_not_stopped():
    model = esrgan()
    stopped = StopProgress()
    other: list[np.ndarray] = []

    def stop_and_run_another(_module: torch.nn.Module, _args: object, _output: object):
        # Stop of this run pressed while another run's thread upscales with the model.
        if stopped.stopped:
            return  # the other run's forward
        stopped.stopped = True
        thread = threading.Thread(
            target=lambda: other.append(upscale(model, StopProgress()))
        )
        thread.start()
        thread.join()

    rrdbs(model)[0].register_forward_hook(stop_and_run_another)
    with pytest.raises(Aborted):
        upscale(model, stopped)

    assert len(other) == 1
    assert other[0].shape == (48, 64, 3)
