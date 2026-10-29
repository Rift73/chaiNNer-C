"""DRFT on PyTorch with ROCm: spandrel's legacy and current DRFT on an AMD GPU give their
CPU result (attention through PyTorch's scaled dot product attention, convolutions
through MIOpen). Without an AMD GPU (PyTorch built for ROCm, with a GPU) it is skipped."""

from __future__ import annotations

import copy

import pytest
import torch
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.DRFT.__arch.drft_arch import drft_nano
from test_tensorrt_drft_export import legacy_drft


def current_drft() -> ImageModelDescriptor:
    torch.manual_seed(1)
    network = drft_nano(scale=2)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.normal_(0.0, 0.05)
    descriptor = ModelLoader().load_from_state_dict(network.state_dict())
    assert isinstance(descriptor, ImageModelDescriptor)
    return descriptor


@pytest.mark.amd_gpu
@pytest.mark.parametrize("variant", ["legacy", "current"])
@pytest.mark.parametrize(("height", "width"), [(64, 64), (80, 112)])
def test_drft_runs_on_the_amd_gpu_as_on_the_cpu(variant: str, height: int, width: int):
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("no AMD GPU")
    descriptor = legacy_drft(32, 0.5) if variant == "legacy" else current_drft()
    model = descriptor.model.eval()
    # Moved before any forward: the neural bias caches its first result on its device.
    gpu_model = copy.deepcopy(model).cuda()
    image = torch.rand(1, 3, height, width, generator=torch.Generator().manual_seed(2))
    with torch.inference_mode():
        expected = model(image)
        actual = gpu_model(image.cuda()).cpu()
    assert actual.shape == (1, 3, 2 * height, 2 * width)
    # TF32 convolutions where the GPU has them (PyTorch's default): within 0.5 % of the
    # output's magnitude (an RTX 5090 gives 0.15 % on the random legacy model, whose
    # output reaches +-35, and 0.02 % on the current one).
    tolerance = 5e-3 * float(expected.abs().max())
    torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)
