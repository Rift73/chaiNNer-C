"""Both TensorRT backends route through the node and fail clearly before any work."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from api import NodeContext
from nodes.impl.pytorch.vosr2 import (
    BundlePaths,
    VosrBackend,
    VosrColorCorrection,
    VosrMemoryMode,
    prepare_image,
)
from nodes.impl.pytorch.vosr2_scene import WindowPlan
from nodes.impl.pytorch.vosr2_temporal import TemporalMode
from nodes.impl.pytorch.vosr2_trt_pipeline import FullSession
from nodes.utils.seed import Seed


def use_device(monkeypatch, node, device):
    monkeypatch.setattr(
        node,
        "get_settings",
        lambda _: SimpleNamespace(
            device=torch.device(device), budget_limit=0, force_cache_wipe=False
        ),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)


@pytest.mark.parametrize(
    "memory,backend,expected",
    [
        (VosrMemoryMode.DEFAULT, VosrBackend.TENSORRT, "bf16"),
        (VosrMemoryMode.DEFAULT, VosrBackend.TENSORRT_FP8, "fp8"),
        (VosrMemoryMode.DEFAULT, VosrBackend.PYTORCH, "pytorch"),
        (VosrMemoryMode.LOW, VosrBackend.TENSORRT, "bf16"),
        (VosrMemoryMode.LOW, VosrBackend.TENSORRT_FP8, "fp8"),
        (VosrMemoryMode.LOW, VosrBackend.PYTORCH, "pytorch"),
    ],
)
def test_node_routes_backends_in_both_memory_modes(
    monkeypatch, tmp_path, memory, backend, expected
):
    from packages.chaiNNer_pytorch.pytorch.processing import upscale_image_vosr as node

    class SelectedEngine(Exception):
        pass

    def select(name):
        def selected(*args):
            raise SelectedEngine(name)

        return selected

    use_device(monkeypatch, node, "cuda")
    monkeypatch.setattr(node, "find_spec", lambda _: object())
    monkeypatch.setattr(node, "ensure_fp8_engine", select("fp8"))
    monkeypatch.setattr(node, "ensure_full_engine", select("bf16"))
    monkeypatch.setattr(node, "release_full_session", select("pytorch"))
    monkeypatch.setattr(node, "ensure_model", lambda _: BundlePaths(tmp_path))
    monkeypatch.setattr(node, "ensure_face_model", lambda _: tmp_path / "faces.onnx")
    with pytest.raises(SelectedEngine, match=expected):
        node.upscale_image_vosr_node(
            cast(NodeContext, SimpleNamespace(node_id="routing", iteration=None)),
            np.zeros((16, 16, 3), np.float32),
            4,
            Seed(0),
            VosrColorCorrection.WAVELET,
            memory,
            backend,
            TemporalMode.THREE_FRAMES,
            65,
            texture=True,
        )


@pytest.mark.parametrize("backend", [VosrBackend.TENSORRT, VosrBackend.TENSORRT_FP8])
def test_unusable_tensorrt_fails_clearly_before_any_work(
    monkeypatch, tmp_path, backend
):
    from packages.chaiNNer_pytorch.pytorch.processing import upscale_image_vosr as node

    def unexpected(*args, **kwargs):
        pytest.fail("An unusable TensorRT request started work")

    for name in (
        "ensure_model",
        "ensure_face_model",
        "prepare_image",
        "ensure_fp8_engine",
        "ensure_full_engine",
    ):
        monkeypatch.setattr(node, name, unexpected)

    def run(device):
        use_device(monkeypatch, node, device)
        node.upscale_image_vosr_node(
            cast(NodeContext, SimpleNamespace(node_id="unusable", iteration=None)),
            np.zeros((16, 16, 3), np.float32),
            4,
            Seed(0),
            VosrColorCorrection.WAVELET,
            VosrMemoryMode.DEFAULT,
            backend,
            TemporalMode.THREE_FRAMES,
            65,
            texture=True,
        )

    monkeypatch.setattr(node, "find_spec", lambda _: object())
    with pytest.raises(ValueError, match="requires an NVIDIA GPU"):
        run("cpu")
    monkeypatch.setattr(node, "find_spec", lambda _: None)
    with pytest.raises(ValueError, match="Install TensorRT in the dependency manager"):
        run("cuda")


@pytest.mark.parametrize("backend", [VosrBackend.TENSORRT, VosrBackend.TENSORRT_FP8])
def test_full_trt_node_does_not_load_unused_pytorch_models(
    monkeypatch, tmp_path, backend
):
    from packages.chaiNNer_pytorch.pytorch.processing import upscale_image_vosr as node

    session = FullSession.__new__(FullSession)
    statistics = object()
    expected = np.zeros((64, 64, 3), np.float32)
    released = []
    context = cast(
        NodeContext,
        SimpleNamespace(
            node_id="full-trt",
            add_cleanup=lambda *args, **kwargs: None,
            iteration=None,
        ),
    )
    use_device(monkeypatch, node, "cuda")
    monkeypatch.setattr(node, "find_spec", lambda _: object())
    monkeypatch.setattr(node, "release_bundle", released.append)
    monkeypatch.setattr(node, "ensure_model", lambda _: BundlePaths(tmp_path))
    monkeypatch.setattr(node, "ensure_face_model", lambda _: tmp_path / "faces.onnx")
    calls = []

    def engine(*args):
        calls.append("engine")
        return tmp_path

    def prepare(*args):
        calls.append("resize")
        return prepare_image(*args)

    monkeypatch.setattr(node, "ensure_fp8_engine", engine)
    monkeypatch.setattr(node, "ensure_full_engine", engine)
    monkeypatch.setattr(node, "prepare_image", prepare)
    monkeypatch.setattr(
        node, "lease_full_session", lambda *args, **kwargs: nullcontext(session)
    )
    monkeypatch.setattr(
        node, "vram_budget", lambda *args, **kwargs: nullcontext(lambda: None)
    )
    monkeypatch.setattr(node.LatentStatistics, "load", lambda _: statistics)

    def unexpected(*args, **kwargs):
        pytest.fail("Full TensorRT attempted to load PyTorch model weights")

    def restore(*args, windows, low):
        assert args[0] is statistics and args[-1] is session
        assert isinstance(windows, WindowPlan)
        assert low is None  # Default memory
        return expected

    monkeypatch.setattr(node, "lease_bundle", unexpected)
    monkeypatch.setattr(node, "restore", restore)
    result = node.upscale_image_vosr_node(
        context,
        np.zeros((16, 16, 3), np.float32),
        4,
        Seed(0),
        VosrColorCorrection.WAVELET,
        VosrMemoryMode.DEFAULT,
        backend,
        TemporalMode.THREE_FRAMES,
        65,
        texture=True,
    )
    assert result is expected and released == ["full-trt"]
    # The image is resized once the engine is ready, never during its build.
    assert calls == ["engine", "resize"]
