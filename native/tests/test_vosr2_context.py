"""Scene windows must never copy context pixels into the restored output."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from api import Progress
from nodes.impl.pytorch import vosr2_context, vosr2_memory, vosr2_scene
from nodes.impl.pytorch.vosr2 import MemoryPlan, VosrBundle, VosrMemoryMode
from nodes.impl.pytorch.vosr2_context import (
    context_denoise,
    context_position,
    context_strengths,
    pad_image,
)


@pytest.mark.parametrize("size", [(1, 1), (16, 32), (512, 512), (528, 720)])
def test_minimum_window_padding_preserves_input(size):
    height, width = size
    value = torch.arange(height * width).float().reshape(1, 1, height, width)
    actual = pad_image(value)
    assert actual.shape[-2:] == (max(512, height), max(512, width))
    torch.testing.assert_close(actual[:, :, :height, :width], value, atol=0, rtol=0)
    assert torch.isfinite(actual).all()
    if height > 1 and height < 512:
        torch.testing.assert_close(actual[:, :, height, :width], value[:, :, -2])


def test_context_selection_is_input_derived_and_phase_aligned():
    image = torch.zeros(1, 3, 544, 720)
    image[:, :, :, 320:384:2] = 1
    image[:, :, :, 321:384:2] = -1
    y, x = context_position(image)
    assert (y, x) == (32, 320)
    assert y % 16 == x % 16 == 0


@pytest.mark.parametrize("shape", [(64, 64), (66, 98), (96, 130)])
@pytest.mark.parametrize("stream_blocks", [False, True])
@torch.inference_mode()
def test_appended_context_never_escapes_into_output(shape, stream_blocks):
    class Dino(torch.nn.Module):
        def forward_intermediate_layer(self, image, layer):
            assert image.shape == (1, 3, 448, 448) and layer == 17
            return torch.zeros(1, 1024, 2)

    class Dit(torch.nn.Module):
        def forward_flexible(self, inputs, time, **kwargs):
            assert inputs.shape in ((1, 32, 64, 64), (1, 32, 64, 72))
            assert time.item() == 1
            kwargs["check_cancel"]()
            # A perfect conditional identity flow. Any leaked context, wrong
            # noise coordinates, uncovered output or crop shift breaks parity.
            return inputs[:, 16:] - inputs[:, :16]

    height, width = shape
    generator = torch.Generator().manual_seed(123)
    latent = torch.randn(1, 16, height, width, generator=generator)
    image = torch.rand(1, 3, height * 8, width * 8, generator=generator) * 2 - 1
    before = latent.clone()
    rng = torch.random.get_rng_state().clone()
    bundle = cast(
        VosrBundle, SimpleNamespace(dino=Dino(), dit=Dit(), dtype=torch.float32)
    )
    plan = MemoryPlan(7_000_000_000, 128 * 1024**2, 256, stream_blocks)
    result = context_denoise(
        bundle,
        latent,
        image,
        torch.device("cpu"),
        4,
        plan,
        Progress.noop_progress(),
        lambda: None,
    )
    torch.testing.assert_close(result, before, atol=1e-6, rtol=1e-5)
    assert torch.equal(latent, before)
    assert torch.equal(torch.random.get_rng_state(), rng)


@torch.inference_mode()
def test_disagreeing_windows_crossfade_without_a_step():
    class Dino(torch.nn.Module):
        def forward_intermediate_layer(self, image, layer):
            return torch.zeros(1, 1024, 2)

    class Dit(torch.nn.Module):
        window = 0

        def forward_flexible(self, inputs, time, **kwargs):
            result = inputs[:, 16:] + self.window
            self.window += 1
            return result

    bundle = cast(
        VosrBundle, SimpleNamespace(dino=Dino(), dit=Dit(), dtype=torch.float32)
    )
    result = context_denoise(
        bundle,
        torch.zeros(1, 16, 64, 120),
        torch.zeros(1, 3, 512, 960),
        torch.device("cpu"),
        0,
        MemoryPlan.for_mode(VosrMemoryMode.DEFAULT),
        Progress.noop_progress(),
        lambda: None,
    )
    line = -result[0, 0, 32]
    overlap = line[56:64]
    assert overlap[0] < 0.02 and overlap[-1] > 0.98
    assert bool(torch.all(torch.diff(overlap) > 0))
    # No single pixel may carry a majority of the unit jump between windows.
    assert torch.diff(line).abs().max() < 0.5
    torch.testing.assert_close(overlap + overlap.flip(0), torch.ones(8))


def test_context_activity_transition_preserves_sharp_windows(monkeypatch):
    energy = np.concatenate(
        [np.full((512, 512), v, dtype=np.float32) for v in (1, 0.15, 0)], axis=1
    )
    monkeypatch.setattr(vosr2_scene, "detail_energy", lambda image: energy)
    result = context_strengths(torch.empty(0), [(0, 0), (0, 64), (0, 128)], (0, 0))
    assert result == pytest.approx([0, 0.5, 1], abs=1e-5)


def test_flat_input_does_not_invent_a_sharpness_reference():
    assert context_strengths(torch.ones(1, 3, 512, 960), [(0, 0), (0, 56)], (0, 0)) == [
        0,
        0,
    ]


@torch.inference_mode()
def test_context_preserves_local_pixels_noise_and_dino_view(monkeypatch):
    generator = torch.Generator().manual_seed(23)
    latent = torch.randn(1, 16, 64, 176, generator=generator)
    image = torch.rand(1, 3, 512, 1408, generator=generator) * 2 - 1
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(4))
    inputs = torch.cat((latent, noise), dim=1)
    calls = []
    dino_calls = []
    monkeypatch.setattr(vosr2_context, "context_position", lambda pixels: (0, 16))
    monkeypatch.setattr(vosr2_context, "context_strengths", lambda *args: [0, 0.5, 1])

    class Dino(torch.nn.Module):
        def forward_intermediate_layer(self, pixels, layer):
            dino_calls.append(pixels.clone())
            return torch.zeros(1, 1024, 2)

    class Dit(torch.nn.Module):
        def forward_flexible(self, value, time, **kwargs):
            calls.append(value.clone())
            return value[:, 16:] - value[:, :16] + (2 if value.shape[-1] == 72 else 0)

    bundle = cast(
        VosrBundle, SimpleNamespace(dino=Dino(), dit=Dit(), dtype=torch.float32)
    )
    result = context_denoise(
        bundle,
        latent,
        image,
        torch.device("cpu"),
        4,
        MemoryPlan.for_mode(VosrMemoryMode.DEFAULT),
        Progress.noop_progress(),
        lambda: None,
    )
    assert [value.shape[-1] for value in calls] == [64, 64, 72, 72]
    for value, start in zip(calls, (0, 56, 56, 112), strict=True):
        torch.testing.assert_close(
            value[:, :, :, :64], inputs[:, :, :, start : start + 64]
        )
        if value.shape[-1] == 72:
            torch.testing.assert_close(value[:, :, :, 64:], inputs[:, :, :, 2:10])
    for actual, start in zip(dino_calls, (0, 56, 112), strict=True):
        crop = image[:, :, :, start * 8 : start * 8 + 512]
        expected = torch.nn.functional.interpolate(
            crop * 0.5 + 0.5, size=(448, 448), mode="bicubic", align_corners=False
        ).clamp_(0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        torch.testing.assert_close(actual, (expected - mean) / std)
    # Only the context prediction changes; its contribution is 0, 1/2, then 1.
    for column, change in ((16, 0), (88, 1), (160, 2)):
        torch.testing.assert_close(
            result[:, :, :, column], latent[:, :, :, column] - change
        )


def memory_mock(monkeypatch, used=2_000_000_000, previous=0.4):
    state = SimpleNamespace(used=used, fraction=previous, history=[])
    gpu = SimpleNamespace(
        index=0,
        get_current_vram_usage=lambda: SimpleNamespace(
            total=32_000_000_000, used=state.used
        ),
    )
    monkeypatch.setattr(vosr2_memory, "nvidia", SimpleNamespace(devices=[gpu]))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=32_000_000_000),
    )
    monkeypatch.setattr(
        torch.cuda, "get_per_process_memory_fraction", lambda _: state.fraction
    )

    def set_fraction(value, device):
        assert device == torch.device("cuda:0")
        state.fraction = value
        state.history.append(value)

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", set_fraction)
    return state


@pytest.mark.parametrize("failure", [None, "cancel", "oom", "external"])
def test_vram_quota_restores_allocator_after_every_exit(monkeypatch, failure):
    state = memory_mock(monkeypatch)

    def work():
        with vosr2_memory.vram_budget(
            torch.device("cuda:0"), VosrMemoryMode.LOW, Progress.noop_progress()
        ) as check:
            assert state.fraction == pytest.approx(4_200_000_000 / 32_000_000_000)
            if failure == "cancel":
                raise InterruptedError("cancelled")
            if failure == "oom":
                raise torch.OutOfMemoryError("test allocator cap")
            if failure == "external":
                state.used = 6_950_000_000
            check()

    if failure:
        expected = InterruptedError if failure == "cancel" else RuntimeError
        with pytest.raises(expected):
            work()
    else:
        work()
    assert state.fraction == 0.4
    assert len(state.history) == 2


def test_vram_quota_keeps_existing_stricter_limit_and_app_budget(monkeypatch):
    state = memory_mock(monkeypatch, previous=0.05)
    with vosr2_memory.vram_budget(
        torch.device("cuda:0"), VosrMemoryMode.DEFAULT, Progress.noop_progress(), 1
    ):
        assert state.fraction == pytest.approx(1024**3 / 32_000_000_000)
    assert state.fraction == 0.05


def test_default_profile_has_no_ceiling_below_the_device(monkeypatch):
    state = memory_mock(monkeypatch, previous=1.0)
    with vosr2_memory.vram_budget(
        torch.device("cuda:0"), VosrMemoryMode.DEFAULT, Progress.noop_progress()
    ) as check:
        # The whole device minus other use and the fixed headroom; no 30 GB cap.
        assert state.fraction == pytest.approx(29_200_000_000 / 32_000_000_000)
        state.used = 31_000_000_000
        check()
        state.used = 31_950_000_000
        with pytest.raises(RuntimeError, match="safety boundary"):
            check()
    assert state.fraction == 1.0


def test_vram_quota_rejects_busy_device_before_mutation(monkeypatch):
    state = memory_mock(monkeypatch, used=6_300_000_000)
    with pytest.raises(RuntimeError, match="Not enough free"):
        with vosr2_memory.vram_budget(
            torch.device("cuda:0"), VosrMemoryMode.LOW, Progress.noop_progress()
        ):
            pytest.fail("Busy device admitted")
    assert state.history == []
    state.used = 2_000_000_000
    with vosr2_memory.vram_budget(
        torch.device("cuda:0"), VosrMemoryMode.LOW, Progress.noop_progress()
    ) as check:
        check()


def test_cpu_profile_never_mutates_cuda_allocator(monkeypatch):
    def forbidden(*args):
        pytest.fail("CPU inference touched the CUDA allocator")

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", forbidden)
    with vosr2_memory.vram_budget(
        torch.device("cpu"), VosrMemoryMode.LOW, Progress.noop_progress()
    ) as check:
        check()


def test_context_selection_on_flat_input_has_stable_tie_break():
    image = torch.from_numpy(np.zeros((1, 3, 512, 512), dtype=np.float32))
    assert context_position(image) == (0, 448)
