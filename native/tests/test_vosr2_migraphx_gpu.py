"""VOSR 2.0 on an AMD GPU through MIGraphX and PyTorch on ROCm, for the AMD kit: the HIP
Q/K/V op against its PyTorch reference; the BF16 program set's build (its child checks
every program against PyTorch as the TensorRT engines are checked: relative L2 0.03, DINO
0.05 against FP32) and reuse; its restoration against the PyTorch backend's; GPU Memory
Low on PyTorch; and the reduced set (MIGraphX FP8: FP8 calibrated by MIGraphX on a GPU
with FP8 math, else FP16) against BF16, whose numbers the build's manifest keeps. They
need an AMD GPU, the ROCm PyTorch, the MIGraphX runner and vosr2_migx_ops (anything
missing fails them); all but the first also the official VOSR 2.0 model files
(ensure_model's folder, about 7 GB) at CHAINNER_VOSR2_MODEL, and skip without them."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

import gpu
from api import NodeContext
from nodes.impl.migraphx import vosr2_migx
from nodes.impl.migraphx.vosr2_migx_compile import load_ops, migraphx_graph
from nodes.impl.pytorch.vosr2 import (
    BundlePaths,
    VosrBackend,
    VosrColorCorrection,
    VosrMemoryMode,
)
from nodes.impl.pytorch.vosr2_cache import release_bundle
from nodes.impl.pytorch.vosr2_scene import WindowPlan
from nodes.impl.pytorch.vosr2_trt_pipeline import release_full_session
from nodes.utils.seed import Seed

pytestmark = pytest.mark.amd_gpu


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not gpu.amd.is_available:
        pytest.skip("no AMD GPU")
    assert torch.version.hip and torch.cuda.is_available(), "no working ROCm PyTorch"
    return torch.device("cuda", gpu.amd.devices[0].index)


@pytest.fixture(scope="module")
def model() -> BundlePaths:
    folder = os.environ.get("CHAINNER_VOSR2_MODEL")
    if not folder:
        pytest.skip("CHAINNER_VOSR2_MODEL names no VOSR 2.0 model folder")
    return BundlePaths(Path(folder))


def node_context(storage: Path) -> NodeContext:
    return cast(
        NodeContext,
        SimpleNamespace(
            node_id="vosr-amd",
            storage_dir=storage,
            iteration=None,
            aborted=False,
            paused=False,
            add_cleanup=lambda *args, **kwargs: None,
            suspend=lambda: None,
            check_aborted=lambda: None,
            set_progress=lambda value: None,
        ),
    )


def bf16(value: torch.Tensor) -> torch.Tensor:
    return value.bfloat16().float()


def qkv_reference(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> list[torch.Tensor]:
    """The plugin's Q, K and V (1, 24, tokens, 64) in its rounding order: RMS
    normalization, BF16, weight, BF16, then the adjacent-pair rotation with each product
    and the sum rounded to BF16."""
    tokens = x.shape[1]
    parts = x.float().reshape(tokens, 3, 24, 64)
    cos, sin = cos.float()[:, None], sin.float()[:, None]

    def pack(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(value.square().sum(-1, keepdim=True) / 64 + 1e-6)
        value = bf16(bf16(value * scale) * weight.float())
        other = torch.stack((-value[..., 1::2], value[..., 0::2]), -1).flatten(-2)
        value = bf16(bf16(value * cos) + bf16(other * sin))
        return value.permute(1, 0, 2)[None]

    return [
        pack(parts[:, 0], q_weight),
        pack(parts[:, 1], k_weight),
        parts[:, 2].permute(1, 0, 2)[None],
    ]


@pytest.mark.parametrize("tokens", [1, 7, 1024, 1152])
def test_the_qkv_op_matches_its_pytorch_reference(device, tmp_path, tokens):
    import onnx
    from onnx import TensorProto, helper

    shapes = {
        "x": (1, tokens, 4608),
        "q_norm": (64,),
        "k_norm": (64,),
        "cos": (tokens, 64),
        "sin": (tokens, 64),
    }
    graph = helper.make_graph(
        [
            helper.make_node(
                "VosrQkvPack",
                list(shapes),
                ["q", "k", "v"],
                name="/attn/NativePack",
                domain="trt.plugins",
            )
        ],
        "qkv",
        [
            helper.make_tensor_value_info(name, TensorProto.BFLOAT16, list(shape))
            for name, shape in shapes.items()
        ],
        [
            helper.make_tensor_value_info(
                name, TensorProto.BFLOAT16, [1, 24, tokens, 64]
            )
            for name in ("q", "k", "v")
        ],
    )
    path = tmp_path / "qkv.onnx"
    onnx.save(
        helper.make_model(
            graph,
            ir_version=10,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("trt.plugins", 1),
            ],
        ),
        path,
    )
    migraphx = load_ops()
    program = migraphx.compile_onnx(
        str(migraphx_graph(path)), device=device.index, input_dims=shapes
    )
    try:
        generator = torch.Generator().manual_seed(tokens)
        inputs = {
            "x": torch.randn(shapes["x"], generator=generator) * 3,
            "q_norm": 0.5 + torch.rand(64, generator=generator),
            "k_norm": 0.5 + torch.rand(64, generator=generator),
        }
        angles = torch.rand(tokens, 64, generator=generator) * 6.3
        inputs.update(cos=angles.cos(), sin=angles.sin())
        values = {name: value.bfloat16().to(device) for name, value in inputs.items()}
        outputs = [
            torch.empty(tuple(shape), dtype=torch.bfloat16, device=device)
            for shape in program.output_shapes
        ]
        stream = torch.cuda.current_stream(device)
        program.run(
            {name: value.data_ptr() for name, value in values.items()},
            [output.data_ptr() for output in outputs],
            stream.cuda_stream,
        )
        stream.synchronize()
    finally:
        program.close()
    expected = qkv_reference(*(value.cpu() for value in values.values()))
    for name, output, reference in zip("qkv", outputs, expected, strict=True):
        actual = output.float().cpu()
        if name == "v":
            assert torch.equal(actual, reference), "V is a copy"
        else:
            # One BF16 step at most: the sum of squares and rsqrt may round otherwise
            torch.testing.assert_close(actual, reference, rtol=2**-7, atol=1e-3)


def synthetic_image() -> np.ndarray:
    """A 128 x 128 BGR image with edges, a gradient and fine noise."""
    rng = np.random.default_rng(5)
    y, x = np.mgrid[0:128, 0:128] / 127
    image = np.stack((x, y, (x + y) / 2), -1)
    image[32:96, 40:88] = (0.9, 0.3, 0.2)
    return np.clip(image + rng.normal(0, 0.02, image.shape), 0, 1).astype(np.float32)


def upscale(
    monkeypatch,
    context: NodeContext,
    device: torch.device,
    model: BundlePaths,
    memory: VosrMemoryMode,
    backend: VosrBackend,
) -> np.ndarray:
    from packages.chaiNNer_pytorch.pytorch.processing import upscale_image_vosr as node

    monkeypatch.setattr(
        node,
        "get_settings",
        lambda _: SimpleNamespace(
            device=device, budget_limit=0, force_cache_wipe=False
        ),
    )
    try:
        return node._upscale(
            context,
            synthetic_image(),
            model,
            4,
            Seed(1),
            VosrColorCorrection.WAVELET,
            memory,
            backend,
            None,
            WindowPlan(),
        )
    finally:
        release_full_session(context.node_id)
        release_bundle(context.node_id)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(1 / np.mean((a - b) ** 2)))


def test_the_bf16_program_set_restores_as_pytorch_does_and_is_reused(
    device, model, tmp_path, monkeypatch
):
    context = node_context(tmp_path)
    directory = vosr2_migx.ensure_migraphx_engine(context, model, device, reduced=False)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["profiles"]
    print("VOSR MIGraphX BF16 stages against PyTorch:", json.dumps(records))
    assert len(records) == 4, records
    assert all(record["relative_l2"] <= 0.03 for record in records), records

    def compile_again(*args, **kwargs):
        pytest.fail("a cached program set was compiled again")

    monkeypatch.setattr(vosr2_migx, "compile_worker", compile_again)
    assert (
        vosr2_migx.ensure_migraphx_engine(context, model, device, reduced=False)
        == directory
    )
    migraphx = upscale(
        monkeypatch,
        context,
        device,
        model,
        VosrMemoryMode.DEFAULT,
        VosrBackend.MIGRAPHX,
    )
    pytorch = upscale(
        monkeypatch,
        context,
        device,
        model,
        VosrMemoryMode.DEFAULT,
        VosrBackend.PYTORCH,
    )
    assert migraphx.shape == pytorch.shape == (512, 512, 3)
    assert np.isfinite(migraphx).all()
    value = psnr(migraphx, pytorch)
    print(f"VOSR MIGraphX BF16 against PyTorch: {value:.2f} dB PSNR")
    assert value >= 35, f"MIGraphX against PyTorch: {value:.1f} dB PSNR"


def test_low_memory_runs_on_pytorch_under_7_gb(device, model, tmp_path, monkeypatch):
    context = node_context(tmp_path)
    torch.cuda.reset_peak_memory_stats(device)
    result = upscale(
        monkeypatch, context, device, model, VosrMemoryMode.LOW, VosrBackend.MIGRAPHX
    )
    assert result.shape == (512, 512, 3) and np.isfinite(result).all()
    assert not (tmp_path / "vosr2-migraphx-bf16").exists()
    assert torch.cuda.max_memory_reserved(device) < 7_000_000_000


def test_the_reduced_program_set_restores_close_to_bf16(
    device, model, tmp_path, monkeypatch
):
    context = node_context(tmp_path)
    reduced = upscale(
        monkeypatch,
        context,
        device,
        model,
        VosrMemoryMode.DEFAULT,
        VosrBackend.MIGRAPHX_FP8,
    )
    bf16_result = upscale(
        monkeypatch,
        context,
        device,
        model,
        VosrMemoryMode.DEFAULT,
        VosrBackend.MIGRAPHX,
    )
    (manifest,) = tmp_path.glob("vosr2-migraphx-fp*/*/manifest.json")
    records = json.loads(manifest.read_text(encoding="utf-8"))["profiles"]
    assert np.isfinite(reduced).all()
    value = psnr(reduced, bf16_result)
    # The kit's report: the set's precision, its stages' distance from PyTorch BF16 and
    # the restoration's from MIGraphX BF16
    print(
        f"VOSR {manifest.parent.parent.name} against MIGraphX BF16: {value:.2f} dB "
        f"PSNR; stages against PyTorch BF16: {json.dumps(records)}"
    )
    # TensorRT FP8 against BF16 measured 42.5 and 50.0 dB (VOSR2.md)
    assert value >= 35, (
        f"MIGraphX {manifest.parent.parent.name} against BF16: {value:.1f} dB PSNR; stages {records}"
    )
