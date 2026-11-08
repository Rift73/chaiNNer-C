"""Raw conditioning reuse, bounded state and colour parity without a CUDA dependency."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import cast

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

from nodes.impl.pytorch import vosr2_trt_pipeline
from nodes.impl.pytorch.vosr2 import VosrColorCorrection, color_correct
from nodes.impl.pytorch.vosr2_scene import (
    _context_position,
    _context_strengths,
    detail_energy,
    scene_routing,
    scene_windows,
)
from nodes.impl.pytorch.vosr2_trt_execution import BoundedQueue, StageExecution
from nodes.impl.pytorch.vosr2_trt_models import (
    AttentionCore,
    CachedDiT,
    DiTKv,
    GroupedAttention,
    StableDit,
    profiles,
)
from nodes.impl.pytorch.vosr2_trt_pipeline import FullSession
from nodes.impl.pytorch.vosr2_trt_restore import FiniteChecks, gpu_colour
from nodes.impl.pytorch.vosr2_trt_rewrite import attention_keys_transposed
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
        energy = detail_energy(image)
        layout = scene_windows(energy, 64, 120)
        anchor = _context_position(energy)
        assert scene_routing(image) == (
            layout,
            anchor,
            _context_strengths(energy, layout.grid, anchor),
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


def test_attention_takes_its_keys_transposed_with_the_same_result(tmp_path):
    k_info = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 1, "tokens", 4])
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["k"], ["kt"], perm=[0, 1, 3, 2]),
            helper.make_node("MatMul", ["q", "kt"], ["scores"]),
        ],
        "attention",
        [helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 1, 2, 4]), k_info],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, None)],
    )
    source, target = tmp_path / "source.onnx", tmp_path / "target.onnx"
    onnx.save(helper.make_model(graph), source)
    attention_keys_transposed(source, target)
    model = onnx.load(target)
    assert [node.op_type for node in model.graph.node] == ["MatMul"]
    dims = next(
        i for i in model.graph.input if i.name == "k"
    ).type.tensor_type.shape.dim
    assert dims[2].dim_value == 4 and dims[3].dim_param == "tokens"
    rng = np.random.default_rng(1)
    q = rng.random((1, 1, 2, 4), dtype=np.float32)
    k = rng.random((1, 1, 3, 4), dtype=np.float32)
    before = ReferenceEvaluator(str(source)).run(None, {"q": q, "k": k})
    after = ReferenceEvaluator(str(target)).run(
        None, {"q": q, "k": k.transpose(0, 1, 3, 2)}
    )
    assert np.array_equal(cast(list, before)[0], cast(list, after)[0])


def test_grouped_attention_has_the_bits_of_the_flat_one():
    generator = torch.Generator().manual_seed(4)
    q, k, v = (
        torch.randn(shape, generator=generator).bfloat16()
        for shape in ((1, 1, 5, 384), (1, 1, 384, 24), (1, 1, 24, 384))
    )
    expected = AttentionCore()(q, k, v)
    actual = GroupedAttention()(q, k.unflatten(-1, (-1, 8)), v.unflatten(2, (-1, 8)))
    assert torch.equal(actual, expected)
    flat, grouped = profiles("attention"), profiles("attention_grouped")
    for a, b in zip(flat["k"], grouped["k"], strict=True):
        assert b == (*a[:3], a[3] // 8, 8)
    for a, b in zip(flat["v"], grouped["v"], strict=True):
        assert b == (*a[:2], a[2] // 8, 8, a[3])


def test_the_hosted_attention_graph_takes_grouped_keys_and_values(tmp_path):
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["k"], ["kt"], perm=[0, 1, 3, 2]),
            helper.make_node("MatMul", ["q", "kt"], ["s"]),
            helper.make_node("MatMul", ["s", "v"], ["result"]),
        ],
        "attention",
        [
            helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 1, 2, 4]),
            helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 1, "n", 4]),
            helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 1, "n", 4]),
        ],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, None)],
    )
    source, target = tmp_path / "source.onnx", tmp_path / "target.onnx"
    onnx.save(helper.make_model(graph), source)
    attention_keys_transposed(source, target, grouped=True)
    model = onnx.load(target)
    shapes = {
        i.name: [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
        for i in model.graph.input
    }
    assert shapes["k"] == [1, 1, 4, "k_groups", 8]
    assert shapes["v"] == [1, 1, "v_groups", 8, 4]
    rng = np.random.default_rng(2)
    q = rng.random((1, 1, 2, 4), dtype=np.float32)
    k = rng.random((1, 1, 16, 4), dtype=np.float32)
    v = rng.random((1, 1, 16, 4), dtype=np.float32)
    before = ReferenceEvaluator(str(source)).run(None, {"q": q, "k": k, "v": v})
    keys = k.transpose(0, 1, 3, 2).reshape(1, 1, 4, 2, 8)
    after = ReferenceEvaluator(str(target)).run(
        None, {"q": q, "k": keys, "v": v.reshape(1, 1, 2, 8, 4)}
    )
    assert np.array_equal(cast(list, before)[0], cast(list, after)[0])


class Launched:
    def __init__(self, label: str):
        self.label = label

    def record_stream(self, stream) -> None:
        pass


def lane_session(monkeypatch, reserve):
    launched = []
    streams: dict[int, SimpleNamespace] = {}

    def lane_stream(number):
        return streams.setdefault(
            number,
            SimpleNamespace(wait_stream=lambda other: None, wait_event=lambda e: None),
        )

    current = SimpleNamespace(wait_stream=lambda other: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: current)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(record=lambda stream: None)
    )
    session = FullSession.__new__(FullSession)
    session.device, session.reserve = torch.device("cpu"), reserve
    session.lane_count, session.images = 3, 40
    session.stream = cast(torch.cuda.Stream, lane_stream(0))
    session.stages = cast(
        dict, {name: name for name in ("dino", "dit_kv", "dit_cached")}
    )
    monkeypatch.setattr(session, "rope", lambda width: (f"cos{width}", f"sin{width}"))

    def launch(stage, values, stream, number):
        assert stream is streams[number]
        width = values[0].shape[-1] if stage == "dit_cached" else ""
        launched.append(f"lane{number} {stage}{width}")
        return Launched(f"lane{number} {stage}{width}")

    session.execution = cast(
        StageExecution, SimpleNamespace(launch=launch, lane_stream=lane_stream)
    )
    return session, launched


def labels(results):
    return [(c.label, [p.label for p in ps]) for c, ps in results]


def test_windows_run_side_by_side_and_second_predictions_on_pair_lanes(monkeypatch):
    session, launched = lane_session(monkeypatch, None)
    image = torch.zeros(1, 3, 448, 448)
    local, wide = torch.zeros(1, 32, 64, 64), torch.zeros(1, 32, 64, 72)
    results = session.windows([(image, [local, wide]), (image, [wide])])
    assert labels(results) == [
        ("lane0 dino", ["lane0 dit_cached64", "lane8 dit_cached72"]),
        ("lane1 dino", ["lane1 dit_cached72"]),
    ]
    assert launched == [
        "lane0 dino",
        "lane0 dit_kv",
        "lane0 dit_cached64",
        "lane8 dit_cached72",
        "lane1 dino",
        "lane1 dit_kv",
        "lane1 dit_cached72",
    ]


def test_a_lone_windows_second_prediction_reads_its_lanes_keys(monkeypatch):
    session, launched = lane_session(monkeypatch, None)
    image = torch.zeros(1, 3, 448, 448)
    local, wide = torch.zeros(1, 32, 64, 64), torch.zeros(1, 32, 64, 72)
    results = session.windows([(image, [local, wide])])
    assert labels(results) == [
        ("lane0 dino", ["lane0 dit_cached64", "lane8 dit_cached72"])
    ]
    assert "lane8 dit_kv" not in launched and "lane8 dino" not in launched


def test_low_memory_takes_a_windows_predictions_in_turn(monkeypatch):
    session, launched = lane_session(monkeypatch, 1)
    assert session.lanes() == 1
    calls = []
    monkeypatch.setattr(
        session, "condition", lambda image: calls.append("condition") or "condition"
    )
    monkeypatch.setattr(
        session,
        "predict",
        lambda latent, condition: calls.append(latent.shape[-1]) or latent.shape[-1],
    )
    session.kv_tensor = None
    image = torch.zeros(1, 3, 448, 448)
    local, wide = torch.zeros(1, 32, 64, 64), torch.zeros(1, 32, 64, 72)
    assert session.windows([(image, [local, wide])]) == [("condition", [64, 72])]
    assert calls == ["condition", 64, 72] and launched == []


@pytest.mark.parametrize(
    ("free", "images", "lanes"),
    [
        (10 * 2**30, 6, 1),
        (10 * 2**30, 7, 2),
        (10 * 2**30, 40, 2),
        (1.5 * 2**30, 7, 2),
        (2**30, 7, 1),
    ],
)
def test_lanes_follow_the_images_and_the_free_memory(monkeypatch, free, images, lanes):
    session = FullSession.__new__(FullSession)
    session.device, session.reserve, session.lane_count = (
        torch.device("cpu"),
        None,
        None,
    )
    session.images = images
    monkeypatch.setattr(vosr2_trt_pipeline, "available_memory", lambda device: free)
    assert session.lanes() == lanes
    # The free memory is read once per geometry.
    monkeypatch.setattr(vosr2_trt_pipeline, "available_memory", lambda device: 0)
    assert session.lanes() == lanes
    session.reserve = 1
    assert session.lanes() == 1


def test_images_are_counted_per_geometry():
    session = FullSession.__new__(FullSession)
    session.geometry, session.constants, session.ropes = None, {}, {}
    session.execution = cast(StageExecution, SimpleNamespace(clear=lambda: None))
    session.begin_image(512, 512, 1)
    assert session.images == 0
    session.begin_image(512, 512, 2)
    session.begin_image(512, 512, 2)
    assert session.images == 2
    session.begin_image(512, 640, 2)
    assert session.images == 0
