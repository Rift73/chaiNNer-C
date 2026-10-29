"""Use BF16 Mode (CPU) in Upscale Image (PyTorch): an FP32 model's forward runs in
bfloat16 on the CPU (autocast, so the weights stay FP32) and gives a finite output
within BF16's precision of the FP32 one; a model without bfloat16 support runs in
FP32."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from spandrel import ImageModelDescriptor
from spandrel.architectures.ESRGAN import ESRGAN, ESRGANArch

# Tiling runs in the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import Progress
from nodes.impl.upscale.auto_split_tiles import NO_TILING
from packages.chaiNNer_pytorch import settings as settings_module
from packages.chaiNNer_pytorch.pytorch.processing import upscale_image
from packages.chaiNNer_pytorch.settings import PyTorchSettings

# This CPU's own detection, before the fixture below passes it on every CPU
NATIVE_BF16 = settings_module.cpu_has_bf16


@pytest.fixture(autouse=True)
def cpu_with_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    """The setting's detection passed on any CPU: the model is tiny, and BF16 runs
    (emulated, slower) on one without AVX-512 BF16 too."""
    monkeypatch.setattr(settings_module, "cpu_has_bf16", True)


# Four 8-bit codes; BF16 keeps 8 significant bits (an output near 0.5 rounds to
# 1/512), and this model's outputs differ from FP32 by about one code.
TOLERANCE = 4 / 255


def esrgan() -> ImageModelDescriptor:
    """A random 2x ESRGAN of one block: convolutions, LeakyReLU and scaled residual
    sums in BF16, nearest-neighbour upsampling. Its output sits around 0.5, so none
    of it is clipped."""
    torch.manual_seed(0)
    net = ESRGAN(num_filters=16, num_blocks=1, scale=2)
    last = [m for m in net.modules() if isinstance(m, torch.nn.Conv2d)][-1]
    assert last.bias is not None
    with torch.no_grad():
        last.bias.add_(0.5)
    model = ESRGANArch().load(net.state_dict())
    assert isinstance(model, ImageModelDescriptor)
    return model


def image() -> np.ndarray:
    return np.random.default_rng(0).random((24, 32, 3), dtype=np.float32)


def upscale(model: ImageModelDescriptor, use_cpu_bf16: bool) -> np.ndarray:
    options = PyTorchSettings(
        use_cpu=True,
        use_fp16=False,
        gpu_index=0,
        budget_limit=0,
        use_cpu_bf16=use_cpu_bf16,
    )
    assert options.use_cpu_bf16 == use_cpu_bf16
    return upscale_image.upscale(
        image(), model, NO_TILING, options, Progress.noop_progress()
    )


def test_bf16_is_within_bf16_precision_of_fp32():
    model = esrgan()
    fp32 = upscale(model, use_cpu_bf16=False)
    bf16 = upscale(model, use_cpu_bf16=True)
    assert bf16.dtype == np.float32 and bf16.shape == fp32.shape == (48, 64, 3)
    assert np.isfinite(bf16).all()
    difference = np.abs(bf16 - fp32)
    assert 0 < difference.max() <= TOLERANCE
    # Autocast: the model the next node gets is still FP32
    assert model.dtype == torch.float32
    np.testing.assert_array_equal(upscale(model, use_cpu_bf16=False), fp32)


def test_a_model_without_bfloat16_support_runs_fp32():
    model = esrgan()
    fp32 = upscale(model, use_cpu_bf16=False)
    model.supports_bfloat16 = False
    np.testing.assert_array_equal(upscale(model, use_cpu_bf16=True), fp32)


@pytest.mark.skipif(
    not NATIVE_BF16, reason="oneDNN's BF16 convolutions need AVX-512 BF16 or AMX"
)
def test_bf16_runs_in_channels_last_order():
    """The first convolution's output is channels-last: oneDNN ran it on the
    channels-last input without reordering it to NCHW."""
    model = esrgan()
    first = next(m for m in model.model.modules() if isinstance(m, torch.nn.Conv2d))
    outputs: list[torch.Tensor] = []

    def record(_module: torch.nn.Module, _input: object, output: torch.Tensor) -> None:
        outputs.append(output)

    first.register_forward_hook(record)
    upscale(model, use_cpu_bf16=True)
    [output] = outputs
    assert output.dtype == torch.bfloat16
    assert output.is_contiguous(memory_format=torch.channels_last)
    assert not output.is_contiguous()
