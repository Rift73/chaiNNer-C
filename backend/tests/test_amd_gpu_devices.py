"""The AMD GPU list against HIP on a machine with AMD GPUs: the same GPUs in the same
order as PyTorch's ROCm build numbers them, with their processors, and memory counters
that follow the GPU an allocation lands on. Needs an AMD GPU and PyTorch's ROCm
build; the AMD kit runs it (also with HIP_VISIBLE_DEVICES set, to check the order)."""

from __future__ import annotations

import pytest
import torch

import gpu

pytestmark = [
    pytest.mark.amd_gpu,
    pytest.mark.skipif(
        torch.version.hip is None or not torch.cuda.is_available(),
        reason="needs an AMD GPU and PyTorch's ROCm build",
    ),
]


def test_the_list_is_hips():
    devices = gpu.amd.devices
    assert len(devices) == torch.cuda.device_count()
    for device in devices:
        properties = torch.cuda.get_device_properties(device.index)
        assert device.gfx == properties.gcnArchName.split(":")[0]
        if device.sysfs is not None:
            # Linux: the PCI address the list read the GPU's counters at
            domain, bus, _ = device.sysfs.name.split(":", 2)
            assert (int(domain, 16), int(bus, 16)) == (
                properties.pci_domain_id,
                properties.pci_bus_id,
            )


@pytest.mark.parametrize("index", range(torch.cuda.device_count()))
def test_memory_counters_follow_the_gpu(index: int):
    before = gpu.get_vram_usage(index)
    assert before is not None
    assert 0 < before.used <= before.total
    block = torch.empty(1 << 30, dtype=torch.uint8, device=f"cuda:{index}")
    torch.cuda.synchronize(index)
    after = gpu.get_vram_usage(index)
    assert after is not None
    # An integrated GPU's allocations can land in system memory (GTT) instead.
    if not torch.cuda.get_device_properties(index).is_integrated:
        assert after.used - before.used >= 1 << 29
    del block
    torch.cuda.empty_cache()
