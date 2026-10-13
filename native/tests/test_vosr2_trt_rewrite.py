"""Raw conditioning reuse, bounded state and colour parity without a CUDA dependency."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from nodes.impl.pytorch.vosr2 import VosrColorCorrection, color_correct
from nodes.impl.pytorch.vosr2_scene import (
    context_position,
    context_strengths,
    scene_routing,
)
from nodes.impl.pytorch.vosr2_trt_execution import BoundedQueue, StageExecution
from nodes.impl.pytorch.vosr2_trt_models import CachedDiT, DiTKv, StableDit
from nodes.impl.pytorch.vosr2_trt_pipeline import FullSession
from nodes.impl.pytorch.vosr2_trt_restore import FiniteChecks, gpu_colour
from vendor.vosr2.lightningdit import LightningDiT


def test_raw_kv_cut_keeps_the_unmodified_mixed_model_prediction():
    torch.manual_seed(913)
    original = LightningDiT(
        input_size=64,
        hidden_size=16,
        depth=2,
        num_heads=4,
        use_rmsnorm=True,
        use_qknorm=True,
        use_swiglu=True,
        z_dims=16,
    ).eval()
    baseline, projection, body = (
        StableDit(original),
        DiTKv(original),
        CachedDiT(original),
    )
    condition = torch.randn(1, 7, 16).bfloat16()
    with torch.inference_mode():
        kv = projection(condition)
        assert kv.shape == (4, 1, 7, 16)
        for width in (4, 6):
            latent = torch.randn(1, 32, 64, width).bfloat16()
            cos = torch.ones(32 * (width // 2), 4).bfloat16()
            sin = torch.zeros_like(cos)
            expected = baseline(latent, condition, cos, sin)
            actual = body(latent, kv, cos, sin)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_kv_is_recomputed_for_new_windows_and_reused_for_both_predictions():
    calls = []
    session = FullSession.__new__(FullSession)
    session.device = torch.device("cpu")
    session.ropes = {}
    session.kv_tensor = None

    def stage(name, *args):
        calls.append((name, args))
        return args[0].clone()

    session.stage = stage
    condition = torch.randn(1, 5, 16)
    for _ in range(2):
        session.begin_window()
        for width in (64, 72):
            session.predict(torch.zeros(1, 32, 64, width), condition)
    assert [name for name, _ in calls] == ["dit_kv", "dit_cached", "dit_cached"] * 2
    assert calls[1][1][1] is calls[2][1][1]
    assert calls[1][1][1] is not calls[4][1][1]


def test_geometry_and_seed_changes_bound_retained_state():
    clears = []
    session = FullSession.__new__(FullSession)
    session.execution = cast(
        StageExecution, SimpleNamespace(clear=lambda: clears.append(True))
    )
    session.geometry = None
    session.constants = {}
    session.ropes = {}
    session.begin_image(512, 512, 42)
    value = torch.ones(1)
    session.constants["noise"] = value
    session.constants["geometry_constant"] = value
    session.begin_image(512, 512, 42)
    assert session.constants["noise"] is value and len(clears) == 1
    session.begin_image(512, 512, 43)
    assert "noise" not in session.constants and len(clears) == 1
    assert session.constants["geometry_constant"] is value
    session.constants["noise"] = value
    session.begin_image(512, 768, 43)
    assert not session.constants and len(clears) == 2
    with pytest.raises(ValueError, match="padded outputs"):
        session.begin_image(2177, 3840, 43)
    assert len(clears) == 2


def test_deferred_finite_checks_retain_errors_even_when_output_is_clamped():
    checks = FiniteChecks()
    value = torch.tensor([float("inf")])
    assert checks.add(value, "DINO conditioning") is value
    value.clamp_(0, 1)
    checks.add(value, "VAE decoding")
    with pytest.raises(ValueError, match="non-finite DINO conditioning"):
        checks.validate()
    valid = FiniteChecks()
    valid.add(torch.ones(2), "VAE encoding")
    valid.validate()


def test_gpu_submission_queue_bounds_work_ahead_without_waiting_each_window(
    monkeypatch,
):
    calls = []

    class Event:
        def record(self):
            calls.append("record")

        def synchronize(self):
            calls.append("wait")

    monkeypatch.setattr(torch.cuda, "Event", Event)
    queue = BoundedQueue()
    for _ in range(12):
        queue.advance(lambda: calls.append("check"))
    assert calls == ["record", "wait", "check", "record", "wait", "check", "record"]


def test_global_attention_upload_is_reused_only_within_its_vae_epoch():
    execution = StageExecution(torch.device("cpu"), cast(torch.cuda.Stream, None))
    target = torch.zeros(3)
    execution._copy("attention", "k", target, torch.ones(3))
    execution._copy("attention", "k", target, torch.full((3,), 2.0))
    assert target.tolist() == [1, 1, 1]
    execution.attention_epoch += 1
    execution._copy("attention", "k", target, torch.full((3,), 2.0))
    assert target.tolist() == [2, 2, 2]
    with pytest.raises(RuntimeError, match="binding changed"):
        execution._copy("dit_cached", "kv", target, target.clone())


@pytest.mark.parametrize(
    "name,binding",
    [
        ("encoder_qkv", "x"),
        ("decoder_qkv", "x"),
        ("encoder_post", "x"),
        ("decoder_post", "x"),
        ("dit_kv", "conditioning"),
        ("dit_cached", "cos"),
        ("dit_cached", "sin"),
    ],
)
def test_resident_bindings_cannot_silently_read_a_previous_buffer(name, binding):
    execution = StageExecution(torch.device("cpu"), cast(torch.cuda.Stream, None))
    value = torch.ones(3)
    execution._copy(name, binding, value, value)
    with pytest.raises(RuntimeError, match="binding changed"):
        execution._copy(name, binding, value, value.clone())


def test_shared_scene_analysis_preserves_anchor_ties_and_strengths():
    for image in (torch.ones(1, 3, 512, 960), torch.randn(1, 3, 512, 960)):
        grid = [(0, 0), (0, 56)]
        anchor = context_position(image)
        assert scene_routing(image, grid) == (
            anchor,
            context_strengths(image, grid, anchor),
        )


@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (20, 20), (53, 79)])
def test_gpu_colour_algorithm_matches_opencv_including_tiny_references(shape):
    rng = np.random.default_rng(31)
    decoded = rng.uniform(-1, 1, (1, 3, *shape)).astype(np.float32)
    reference = rng.random((*shape, 3), dtype=np.float32)
    session = FullSession.__new__(FullSession)
    session.device = torch.device("cpu")
    session.constants = {}
    actual = gpu_colour(session, torch.from_numpy(decoded), reference)
    rgb = decoded[0].transpose(1, 2, 0) * 0.5 + 0.5
    expected = color_correct(rgb, reference, VosrColorCorrection.WAVELET)[:, :, ::-1]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
