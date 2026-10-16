"""BF16 precision boundaries, global attention and complete cache publication."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch

from api import NodeContext
from nodes.impl.pytorch import vosr2_engine_cache, vosr2_trt_pipeline
from nodes.impl.pytorch.vosr2 import BundlePaths, VosrBundle, load_weights
from nodes.impl.pytorch.vosr2_export import rotary_tables
from nodes.impl.pytorch.vosr2_official import FILES
from nodes.impl.pytorch.vosr2_trt_compile import (
    report_stage,
    store_timing_cache,
    timing_cache,
    use_timing_cache,
)
from nodes.impl.pytorch.vosr2_trt_models import (
    STAGES,
    AttentionCore,
    Norm32,
    StableDino,
    mixed,
    profiles,
)
from nodes.impl.pytorch.vosr2_trt_pipeline import FullSession, Stage
from vendor.vosr2.dinov2 import Block, DinoVisionTransformer
from vendor.vosr2.qwenimage_vae2d import RMSNorm2D

if TYPE_CHECKING:
    import tensorrt as trt


def test_bf16_weights_keep_original_fp32_normalization_parameters():
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 1), RMSNorm2D(8)).eval()
    with torch.no_grad():
        cast(RMSNorm2D, model[1]).gamma.copy_(
            torch.linspace(0.1111, 1.2345, 8).reshape(8, 1, 1)
        )
    result = mixed(model, torch.bfloat16)
    assert result[0].weight.dtype == torch.bfloat16
    norm = cast(RMSNorm2D, cast(Norm32, result[1]).original)
    assert norm.gamma.dtype == torch.float32
    torch.testing.assert_close(
        norm.gamma, cast(RMSNorm2D, model[1]).gamma, rtol=0, atol=0
    )
    assert model[0].weight.dtype == torch.float32


def test_bounded_vae_queries_retain_all_keys_and_values():
    generator = torch.Generator().manual_seed(93)
    q, k, v = [
        torch.randn((1, 1, 19, 384), generator=generator).bfloat16() for _ in range(3)
    ]
    graph = AttentionCore()
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float()
    ).bfloat16()
    chunked = torch.cat(
        [graph(q[:, :, start : start + 5], k, v) for start in range(0, 19, 5)], dim=2
    )
    torch.testing.assert_close(chunked, expected, rtol=0.008, atol=0.004)
    local_only = graph(q[:, :, :5], k[:, :, :5], v[:, :, :5])
    assert not torch.allclose(local_only, chunked[:, :, :5], rtol=0.05, atol=0.05)


def test_dino_sensitive_weights_avoid_a_bf16_round_trip():
    original = DinoVisionTransformer(
        img_size=448, embed_dim=32, depth=2, num_heads=4
    ).eval()
    result = StableDino(original)
    assert result.model.patch_embed.proj.weight.dtype == torch.float32
    for result_module, source_module in zip(
        result.model.blocks, original.blocks, strict=True
    ):
        block, source = cast(Block, result_module), cast(Block, source_module)
        assert block.attn.qkv.weight.dtype == torch.float32
        torch.testing.assert_close(
            block.attn.qkv.weight, source.attn.qkv.weight, rtol=0, atol=0
        )
        assert block.mlp.fc1.weight.dtype == torch.bfloat16
        assert source.mlp.fc1.weight.dtype == torch.float32


@pytest.mark.parametrize("width", [64, 72])
def test_bf16_rotary_tables_round_directly_from_fp32(width):
    f32 = rotary_tables(width, torch.device("cpu"), torch.float32)
    bf16 = rotary_tables(width, torch.device("cpu"), torch.bfloat16)
    for actual, reference in zip(bf16, f32, strict=True):
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, reference.bfloat16(), rtol=0, atol=0)


def test_engine_input_order_is_semantic_for_equal_shape_keys_and_values():
    assert tuple(profiles("attention")) == ("q", "k", "v")
    assert tuple(profiles("encoder_post")) == ("x", "attention")
    assert tuple(profiles("dit")) == ("latent", "conditioning", "cos", "sin")


def test_partial_bf16_set_is_never_published_and_corruption_rebuilds(
    monkeypatch, tmp_path
):
    context = cast(
        NodeContext,
        SimpleNamespace(
            storage_dir=tmp_path, node_id="bf16-compiler", suspend=lambda: None
        ),
    )
    paths = BundlePaths(tmp_path / "bundle")
    for file in paths.files():
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"test-checkpoint")
    monkeypatch.setattr(vosr2_engine_cache, "read_configs", lambda _: None)
    plugin = tmp_path / "test-plugin.dll"
    plugin.write_bytes(b"test-plugin")
    monkeypatch.setattr(vosr2_engine_cache, "plugin_path", lambda: plugin)
    monkeypatch.setattr(
        vosr2_engine_cache, "cache_identity", lambda *args: {"test": True}
    )
    attempts = []

    def build(context, paths, directory, device):
        attempts.append(True)
        engines = {}
        for name in STAGES:
            content = name.encode()
            (directory / f"{name}.plan").write_bytes(content)
            engines[name] = hashlib.sha256(content).hexdigest()
        if len(attempts) == 1:
            engines.pop("decoder_post")
        (directory / "validation.json").write_text(json.dumps({"engines": engines}))

    monkeypatch.setattr(vosr2_engine_cache, "build", build)
    args = (context, paths, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="changed after validation"):
        vosr2_engine_cache.ensure_full_engine(*args)
    assert not list(tmp_path.rglob("manifest.json"))
    assert not list(tmp_path.rglob("building-*"))
    path = vosr2_engine_cache.ensure_full_engine(*args)
    assert {p.stem for p in path.glob("*.plan")} == set(STAGES)
    assert vosr2_engine_cache.ensure_full_engine(*args) == path
    assert len(attempts) == 2
    (path / "decoder_post.plan").write_bytes(b"corrupt")
    assert vosr2_engine_cache.ensure_full_engine(*args) == path
    assert (path / "decoder_post.plan").read_bytes() == b"decoder_post"
    assert len(attempts) == 3
    assert len(list(path.parent.glob("quarantine-*"))) == 1
    (path / "manifest.json").write_text("[]")
    assert vosr2_engine_cache.ensure_full_engine(*args) == path
    assert len(attempts) == 4
    assert len(list(path.parent.glob("quarantine-*/manifest.json"))) == 2
    plugin.write_bytes(b"changed-plugin")
    assert vosr2_engine_cache.ensure_full_engine(*args) != path
    assert len(attempts) == 5


class TimingConfig:
    """A builder config's timing-cache calls: data recorded elsewhere is refused."""

    def __init__(self):
        self.attached = []

    def create_timing_cache(self, data):
        return SimpleNamespace(data=data, serialize=lambda: data + b"+timed")

    def set_timing_cache(self, cache, ignore_mismatch):
        assert ignore_mismatch is False
        self.attached.append(cache.data)
        return not cache.data.startswith(b"other-gpu")


def test_timing_cache_is_reused_stored_whole_and_restarted_when_foreign(tmp_path):
    path = tmp_path / "timing-key.cache"
    config = TimingConfig()
    cache = use_timing_cache(cast("trt.IBuilderConfig", config), path)
    assert config.attached == [b""]
    store_timing_cache(cache, path)
    assert path.read_bytes() == b"+timed"
    assert [p.name for p in tmp_path.iterdir()] == [path.name]
    use_timing_cache(cast("trt.IBuilderConfig", config), path)
    assert config.attached[-1] == b"+timed"
    # Recorded by another GPU or TensorRT version: refused, then a new cache.
    path.write_bytes(b"other-gpu timings")
    cache = use_timing_cache(cast("trt.IBuilderConfig", config), path)
    assert config.attached[-2:] == [b"other-gpu timings", b""]
    store_timing_cache(cache, path)
    assert path.read_bytes() == b"+timed"


def test_timing_cache_file_is_per_gpu_model_and_tensorrt_version(monkeypatch, tmp_path):
    gpu = SimpleNamespace(name="Test GPU", major=12, minor=0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: gpu)
    runtime = SimpleNamespace(__version__="11.2.1.2")
    monkeypatch.setitem(sys.modules, "tensorrt", runtime)
    device = torch.device("cuda:0")
    path = timing_cache(tmp_path, device)
    assert path.parent == tmp_path and path.name.startswith("timing-")
    assert timing_cache(tmp_path, device) == path
    runtime.__version__ = "11.3.0.0"
    assert timing_cache(tmp_path, device) != path
    runtime.__version__ = "11.2.1.2"
    gpu.name = "Other GPU"
    assert timing_cache(tmp_path, device) != path


def test_build_stages_are_logged_and_reported_as_the_worker_prints_them(capsys, caplog):
    for index in (1, 2, 3):
        report_stage(index, 3, f"Compiling BF16 stage {index}")
    printed = capsys.readouterr().out.replace("\n", "\r\n").encode()
    progress = []
    context = cast(NodeContext, SimpleNamespace(set_progress=progress.append))
    # The worker's output arrives in pieces; only whole lines count.
    head, tail = printed[:45], printed[45:]
    with caplog.at_level(logging.INFO):
        unread = vosr2_engine_cache.report_stages(context, b"Exporting\n" + head)
        assert progress == [0] and unread == head[head.rindex(b"\n") + 1 :]
        assert vosr2_engine_cache.report_stages(context, unread + tail) == b""
    assert progress == [0, 1 / 3, 2 / 3]
    assert "VOSR TensorRT build: stage 3 of 3, Compiling BF16 stage 3" in caplog.text


def test_a_build_without_nvidia_memory_monitoring_says_so(monkeypatch, tmp_path):
    # NVML lists no device for the CUDA index (NVML failed to start): the build refuses
    # with the message vram_budget gives, not a bare StopIteration, whose empty text
    # would be the node's whole error.
    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(vosr2_engine_cache, "nvidia", SimpleNamespace(devices=[]))
    context = cast(NodeContext, SimpleNamespace())
    with pytest.raises(RuntimeError, match="NVIDIA memory monitoring"):
        vosr2_engine_cache.compile_worker(
            context, "unused.py", tmp_path, tmp_path, torch.device("cuda:0")
        )


def test_bf16_identity_uses_the_pins_not_the_model_folder(monkeypatch, tmp_path):
    context = cast(
        NodeContext,
        SimpleNamespace(
            storage_dir=tmp_path, node_id="bf16-hashes", suspend=lambda: None
        ),
    )
    monkeypatch.setattr(vosr2_engine_cache, "read_configs", lambda _: None)
    plugin = tmp_path / "test-plugin.dll"
    plugin.write_bytes(b"test-plugin")
    monkeypatch.setattr(vosr2_engine_cache, "plugin_path", lambda: plugin)
    monkeypatch.setattr(vosr2_engine_cache, "cache_identity", lambda _: {})
    builds = []

    def build(context, paths, directory, device):
        builds.append(True)
        engines = {}
        for name in STAGES:
            (directory / f"{name}.plan").write_bytes(name.encode())
            engines[name] = hashlib.sha256(name.encode()).hexdigest()
        (directory / "validation.json").write_text(json.dumps({"engines": engines}))

    monkeypatch.setattr(vosr2_engine_cache, "build", build)
    hashed = []
    file_hash = vosr2_engine_cache.file_hash

    def recorded(path):
        hashed.append(path)
        return file_hash(path)

    monkeypatch.setattr(vosr2_engine_cache, "file_hash", recorded)
    device = torch.device("cpu")
    first = BundlePaths(tmp_path / "bundle")
    path = vosr2_engine_cache.ensure_full_engine(context, first, device)
    # A moved model folder keeps its engines: no bundle file is read or rebuilt.
    moved = BundlePaths(tmp_path / "moved")
    assert vosr2_engine_cache.ensure_full_engine(context, moved, device) == path
    assert not {*first.files(), *moved.files()} & set(hashed) and len(builds) == 1
    identity = json.loads((path / "manifest.json").read_text())["identity"]
    assert identity["checkpoints"] == {
        Path(local).name: digest for _, local, _, digest in FILES
    }


def test_bf16_identity_hashes_the_engine_code_not_whole_vosr2_or_the_driver(
    monkeypatch,
):
    monkeypatch.setitem(
        sys.modules, "tensorrt", SimpleNamespace(__version__="11.2.1.2")
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(name="Test GPU", major=12, minor=0),
    )
    hashed = []
    file_hash = vosr2_engine_cache.file_hash

    def recorded(path):
        hashed.append(path.name)
        return file_hash(path)

    monkeypatch.setattr(vosr2_engine_cache, "file_hash", recorded)
    device = torch.device("cuda:0")
    identity = vosr2_engine_cache.cache_identity(device)
    assert set(identity) == {
        "architecture",
        "torch",
        "onnx",
        "tensorrt",
        "gpu",
        "capability",
    }
    assert "vosr2_export.py" in hashed and "lightningdit.py" in hashed
    assert "vosr2.py" not in hashed
    # Only the bundle's construction and weight loading of vosr2.py count.
    source = inspect.getsource
    for code in (VosrBundle.__init__, load_weights):
        monkeypatch.setattr(
            inspect,
            "getsource",
            lambda value, code=code: source(value) + ("\n" if value is code else ""),
        )
        assert vosr2_engine_cache.cache_identity(device) != identity
    monkeypatch.setattr(inspect, "getsource", source)
    assert vosr2_engine_cache.cache_identity(device) == identity


def test_resident_external_weights_are_subtracted_only_when_newly_admitted(monkeypatch):
    pytest.importorskip("tensorrt")
    session = FullSession.__new__(FullSession)
    session.device = torch.device("cuda:0")
    session.budget = 0
    engine = SimpleNamespace(
        streamable_weights_size=1000,
        weight_streaming_budget_v2=0,
        create_execution_context=lambda strategy: object(),
    )

    class WeightlessEngine:
        streamable_weights_size = 0
        weight_streaming_budget_v2 = -1

        def create_execution_context(self, strategy):
            return object()

    weightless = WeightlessEngine()
    session.stages = {
        "dit": cast(Stage, SimpleNamespace(engine=engine, context=None)),
        "attention": cast(Stage, SimpleNamespace(engine=weightless, context=None)),
    }
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=4_000_000_000),
    )
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda _: 0.5)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: 0)
    fractions = []
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: fractions.append(fraction),
    )
    for _ in range(2):
        session.activate(lambda: None)
    assert fractions == [(2_000_000_000 - 1000) / 4_000_000_000]
    assert session.budget == 1000
    assert weightless.weight_streaming_budget_v2 == -1


@pytest.mark.parametrize(
    "used,expected", [(2_000_000_000, 0), (30_000_000_000, 0), (30_900_000_000, 1)]
)
def test_warm_session_is_released_only_when_the_device_is_nearly_full(
    monkeypatch, used, expected
):
    calls = []
    session = FullSession.__new__(FullSession)
    session.device = torch.device("cuda:0")
    monkeypatch.setattr(session, "offload", lambda: calls.append(True))
    gpu = SimpleNamespace(
        index=0,
        get_current_vram_usage=lambda: SimpleNamespace(used=used, total=32_000_000_000),
    )
    monkeypatch.setattr(vosr2_trt_pipeline, "nvidia", SimpleNamespace(devices=[gpu]))
    session.prepare()
    assert len(calls) == expected
