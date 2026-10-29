"""HAT's and DAT's attention on an AMD GPU: every scaled_dot_product_attention call of
their window, overlapping-window and spatial attention runs in one of AOTriton's fused
kernels (flash or memory-efficient attention), in FP32 and FP16, with 30-channel heads
(HAT-L's and DAT2's) padded to 32, and gives the math kernel's result. Runs with
chaiNNer's process settings (torch_runtime), which let AOTriton run on every GPU it has
kernels for. Needs PyTorch on ROCm and an AMD GPU; the AMD kit runs it."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.DAT import DAT
from spandrel.architectures.HAT import HAT
from torch.nn.attention import SDPBackend, sdpa_kernel

from packages.chaiNNer_pytorch.torch_runtime import configure_torch

pytestmark = [
    pytest.mark.amd_gpu,
    pytest.mark.skipif(
        not (torch.version.hip and torch.cuda.is_available()),
        reason="needs PyTorch on ROCm and an AMD GPU",
    ),
]

# Before the first attention call, where AOTriton reads its switch
configure_torch()

# Heads of 30 channels, as HAT-L's and DAT2's (180 channels in 6 heads)
ARCHS: dict[str, Callable[[], torch.nn.Module]] = {
    # Two blocks (the second shifts its windows) and the overlapping cross-attention
    "HAT": lambda: HAT(
        embed_dim=60,
        depths=(2,),
        num_heads=(2,),
        window_size=8,
        compress_ratio=2,
        squeeze_factor=4,
        upsampler="pixelshuffle",
        upscale=2,
    ),
    # Spatial, channel and shifted spatial attention in DAT2's 8x32 windows; each
    # spatial branch has half of the channels and heads
    "DAT": lambda: DAT(
        embed_dim=60,
        depth=[3],
        num_heads=[2],
        split_size=[8, 32],
        upsampler="pixelshuffle",
        upscale=2,
    ),
}
# Absolute tolerances against the math kernel in the same precision
TOLERANCE = {torch.float32: 1e-4, torch.float16: 2e-2}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("arch", ["HAT", "DAT"])
def test_attention_runs_in_fused_kernels(arch: str, dtype: torch.dtype):
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(ARCHS[arch]().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    net = model.model.eval().to("cuda", dtype)
    # Not a multiple of the windows: padded, with the shifted windows' masks for this size
    image = torch.rand(1, 3, 37, 51, device="cuda", dtype=dtype)
    with torch.inference_mode():
        # A call no fused kernel takes raises here instead of running the math kernel
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            fused = net(image)
        with sdpa_kernel([SDPBackend.MATH]):
            reference = net(image)
    assert fused.isfinite().all()
    torch.testing.assert_close(fused, reference, rtol=0, atol=TOLERANCE[dtype])
