"""DUAL on an AMD GPU through PyTorch on ROCm: spandrel's DUAL is plain PyTorch (its
context layers' neighborhood attention is masked scaled-dot-product attention; no
Triton), so the GPU gives the CPU's FP32 result within GPU arithmetic, and FP16 and BF16
within their rounding. Needs an AMD GPU and PyTorch on ROCm (the AMD kit)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.DUAL.__arch.dual_arch import dual_light, dual_xs

pytestmark = [
    pytest.mark.amd_gpu,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and torch.version.hip),
        reason="needs an AMD GPU and PyTorch on ROCm",
    ),
]


def _model(factory: Callable, scale: int, tmp_path: Path) -> ImageModelDescriptor:
    """A folded DUAL whose zero-initialized parameters are made nonzero."""
    torch.manual_seed(0)
    network = factory(scale=scale).eval()
    with torch.no_grad():
        for parameter in network.parameters():
            if not parameter.any():
                parameter.normal_(std=0.02)
    path = tmp_path / "dual.pth"
    torch.save(network.state_dict(), path)
    descriptor = ModelLoader("cpu").load_from_file(path)
    assert isinstance(descriptor, ImageModelDescriptor)
    return descriptor


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(10 * np.log10(1 / ((a.double() - b.double()) ** 2).mean().item()))


@pytest.mark.parametrize(
    ("factory", "scale"), [(dual_xs, 2), (dual_light, 4)], ids=["XS", "Light"]
)
def test_dual_runs_on_the_gpu_as_on_the_cpu(
    factory: Callable, scale: int, tmp_path: Path
):
    descriptor = _model(factory, scale, tmp_path)
    # 48x72: past the M32 window on both axes, so those sites are neighborhoods
    x = torch.rand(1, 3, 48, 72, generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        cpu = descriptor.model(x)
        network = descriptor.model.to("cuda")
        fp32 = network(x.cuda()).cpu()
        half = {
            dtype: network.to(dtype)(x.cuda().to(dtype)).float().cpu()
            for dtype in (torch.float16, torch.bfloat16)
        }
    assert fp32.shape == (1, 3, 48 * scale, 72 * scale)
    # TF32-class matrix products are allowed on the GPU (torch_runtime.py).
    assert _psnr(fp32, cpu) > 50
    for dtype, out in half.items():
        assert torch.isfinite(out).all(), dtype
        assert _psnr(out, cpu) > 30, dtype
