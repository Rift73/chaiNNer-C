"""PyTorch's FP16 mode falls back to FP32 wherever PyTorch runs on the CPU: in CPU mode,
and when no GPU is found. On a GPU it stays on. BF16 Mode (CPU) is the reverse: on
where PyTorch runs on a CPU with AVX-512 BF16 or AMX, off on a GPU, and FP32 on a CPU
without them."""

from __future__ import annotations

import importlib
import logging

import pytest
import torch

from api import ToggleSetting
from packages.chaiNNer_pytorch import settings as settings_module
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


def bf16_settings(use_cpu: bool) -> PyTorchSettings:
    return PyTorchSettings(
        use_cpu=use_cpu, use_fp16=False, gpu_index=0, budget_limit=0, use_cpu_bf16=True
    )


@pytest.mark.usefixtures("no_gpu")
@pytest.mark.parametrize("use_cpu", [True, False], ids=["cpu-mode", "no-gpu"])
def test_bf16_stays_on_a_cpu_with_bf16(monkeypatch: pytest.MonkeyPatch, use_cpu: bool):
    monkeypatch.setattr(settings_module, "cpu_has_bf16", True)
    assert bf16_settings(use_cpu).use_cpu_bf16


@pytest.mark.usefixtures("no_gpu")
def test_bf16_falls_back_to_fp32_on_a_cpu_without_bf16(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr(settings_module, "cpu_has_bf16", False)
    settings_module.log_no_bf16.cache_clear()
    with caplog.at_level(logging.INFO):
        options = bf16_settings(use_cpu=True)
        # Every PyTorch node builds its settings: one line per process
        assert not bf16_settings(use_cpu=True).use_cpu_bf16
    assert not options.use_cpu_bf16
    assert [r.message for r in caplog.records if "BF16" in r.message] == [
        "Falling back to FP32 mode: this CPU has neither AVX-512 BF16 nor AMX."
    ]


@pytest.mark.usefixtures("no_gpu")
@pytest.mark.parametrize("bf16", [True, False])
def test_the_fp16_fallback_names_bf16_when_it_runs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bf16: bool
):
    monkeypatch.setattr(settings_module, "cpu_has_bf16", True)
    with caplog.at_level(logging.INFO):
        options = PyTorchSettings(
            use_cpu=True, use_fp16=True, gpu_index=0, budget_limit=0, use_cpu_bf16=bf16
        )
    assert not options.use_fp16 and options.use_cpu_bf16 == bf16
    assert [r.message for r in caplog.records] == [
        "FP16 is off on the CPU; Upscale Image runs BF16 Mode (CPU)."
        if bf16
        else "Falling back to FP32 mode on the CPU."
    ]


@pytest.mark.parametrize("device", ["cuda:0", "mps"])
def test_bf16_is_off_on_a_gpu(monkeypatch: pytest.MonkeyPatch, device: str):
    monkeypatch.setattr(settings_module, "cpu_has_bf16", True)
    monkeypatch.setattr(
        PyTorchSettings, "device", property(lambda _: torch.device(device))
    )
    assert not bf16_settings(use_cpu=False).use_cpu_bf16


def test_bf16_toggle_is_disabled_on_a_cpu_without_bf16():
    """Off by default after Use FP16 Mode; shown disabled where the CPU has neither
    AVX-512 BF16 nor AMX (detected at import)."""
    registered = list(settings_module.package.settings)

    def toggle(avx512_bf16: bool | None, amx: bool | None) -> ToggleSetting:
        """The toggle with PyTorch's checks giving these answers (None: a torch
        without the check)."""
        with pytest.MonkeyPatch.context() as m:
            for check, answer in (
                ("_is_avx512_bf16_supported", avx512_bf16),
                ("_is_amx_tile_supported", amx),
            ):
                if answer is None:
                    m.delattr(torch.cpu, check)
                else:
                    m.setattr(torch.cpu, check, lambda answer=answer: answer)
            settings_module.package.settings[:] = []
            importlib.reload(settings_module)
            keys = [s.key for s in settings_module.package.settings]
            assert keys.index("use_cpu_bf16") == keys.index("use_fp16") + 1
            [found] = [
                s for s in settings_module.package.settings if s.key == "use_cpu_bf16"
            ]
            assert isinstance(found, ToggleSetting)
            return found

    unavailable = (
        "This CPU has neither AVX-512 BF16 nor AMX, so the setting is unavailable."
    )
    try:
        for avx512_bf16, amx in [(True, False), (False, True)]:
            available = toggle(avx512_bf16, amx)
            assert available.label == "Use BF16 Mode (CPU)"
            assert not available.default and not available.disabled
            assert not available.description.endswith(unavailable)
        for avx512_bf16, amx in [(False, False), (None, None)]:
            missing = toggle(avx512_bf16, amx)
            assert not missing.default and missing.disabled
            assert missing.description.endswith(" " + unavailable)
    finally:
        # The module as this machine's PyTorch makes it, its settings registered once
        importlib.reload(settings_module)
        settings_module.package.settings[:] = registered
