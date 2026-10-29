"""PyTorch's FP16 mode falls back to FP32 wherever PyTorch runs on the CPU: in CPU mode,
and when no GPU is found. On a GPU it stays on."""

from __future__ import annotations

import pytest
import torch

from packages.chaiNNer_pytorch.settings import PyTorchSettings


def settings(use_cpu: bool) -> PyTorchSettings:
    return PyTorchSettings(use_cpu=use_cpu, use_fp16=True, gpu_index=0, budget_limit=0)


@pytest.fixture
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


@pytest.mark.usefixtures("no_gpu")
@pytest.mark.parametrize("use_cpu", [True, False], ids=["cpu-mode", "no-gpu"])
def test_fp16_falls_back_to_fp32_on_the_cpu(use_cpu: bool):
    options = settings(use_cpu)
    assert options.device == torch.device("cpu")
    assert not options.use_fp16


@pytest.mark.parametrize("device", ["cuda:0", "mps"])
def test_fp16_stays_on_a_gpu(monkeypatch: pytest.MonkeyPatch, device: str):
    monkeypatch.setattr(
        PyTorchSettings, "device", property(lambda _: torch.device(device))
    )
    assert settings(use_cpu=False).use_fp16


def test_cpu_mode_on_a_gpu_machine_is_fp32(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    options = settings(use_cpu=True)
    assert options.device == torch.device("cpu")
    assert not options.use_fp16
