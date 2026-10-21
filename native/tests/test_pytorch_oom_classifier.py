"""CPU stand-ins distinguish a CUDA fault from memory failures that may split."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from spandrel import ImageModelDescriptor
from test_tiling_control import OLD

from api import Progress
from nodes.impl.pytorch import auto_split
from nodes.impl.upscale.tiler import MaxTileSize


@pytest.mark.parametrize(
    "error,is_oom",
    [
        (
            RuntimeError(
                "CUDA error: no kernel image is available for execution on the device"
            ),
            False,
        ),
        (
            torch.OutOfMemoryError(
                "CUDA memory is insufficient for activation offload and its workspace."
            ),
            True,
        ),
        (MemoryError("host allocation failed"), True),
        (RuntimeError("DefaultCPUAllocator: can't allocate memory"), True),
        (RuntimeError("CUDA out of memory."), True),
        (RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"), True),
    ],
)
def test_only_memory_failures_retry(monkeypatch, error, is_oom):
    calls = []
    splits = []
    cleared = []

    class Tiler(MaxTileSize):
        def split(self, tile_size):
            splits.append(tile_size)
            return super().split(tile_size)

    def forward(_model, tensor, _progress):
        calls.append(tensor.shape)
        if len(calls) == 1:
            raise error
        return tensor

    monkeypatch.setattr(auto_split, "auto_split", OLD[0].auto_split)
    monkeypatch.setattr(auto_split, "Split", OLD[0].Split)
    monkeypatch.setattr(auto_split, "_forward_or_stopped", forward)
    monkeypatch.setattr(
        auto_split, "safe_cuda_cache_empty", lambda: cleared.append(True)
    )
    model = SimpleNamespace(
        dtype=torch.float32, device=torch.device("cpu"), model=torch.nn.Identity()
    )
    image = np.ones((8, 8, 3), np.float32)

    def run():
        return auto_split.pytorch_auto_split(
            image,
            cast(ImageModelDescriptor[torch.nn.Module], model),
            torch.device("cpu"),
            False,
            Tiler(64),
            Progress.noop_progress(),
        )

    if is_oom:
        np.testing.assert_array_equal(run(), image)
        assert len(calls) == 2
        assert len(splits) == len(cleared) == 1
    else:
        with pytest.raises(RuntimeError) as raised:
            run()
        assert raised.value is error
        assert len(calls) == 1
        assert not splits and not cleared
