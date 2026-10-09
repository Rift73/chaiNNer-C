"""PyTorch's CPU tile estimate takes its RAM budget from the helper NCNN shares, and
the budget is the one it computed itself before: 80% of the available RAM, or of the
Memory Budget Limit when that is lower."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import psutil
import pytest
from spandrel import ImageModelDescriptor
from spandrel.architectures.Compact import Compact, CompactArch

# Tiling runs in the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import Progress
from nodes.impl.upscale.auto_split_tiles import ESTIMATE
from packages.chaiNNer_pytorch.pytorch.processing import upscale_image
from packages.chaiNNer_pytorch.settings import PyTorchSettings

AVAILABLE = 3 * 1024**3 + 12345


@pytest.mark.parametrize("budget_limit", [0, 1, 2, 3, 4])
def test_cpu_budget_is_unchanged(monkeypatch: pytest.MonkeyPatch, budget_limit: int):
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=AVAILABLE)
    )
    budgets: list[float] = []
    estimate_tile_size = upscale_image.estimate_tile_size

    def record(
        budget: float, model_size: int, img: np.ndarray, img_element_size: int = 4
    ) -> int:
        budgets.append(budget)
        return estimate_tile_size(budget, model_size, img, img_element_size)

    monkeypatch.setattr(upscale_image, "estimate_tile_size", record)
    net = Compact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=2, upscale=2)
    model = CompactArch().load(net.state_dict())
    assert isinstance(model, ImageModelDescriptor)
    options = PyTorchSettings(
        use_cpu=True, use_fp16=False, gpu_index=0, budget_limit=budget_limit
    )
    image = np.zeros((8, 8, 3), np.float32)
    upscale_image.upscale(image, model, ESTIMATE, options, Progress.noop_progress())

    free = AVAILABLE
    if budget_limit > 0:
        free = min(budget_limit * 1024**3, free)
    assert budgets == [int(free * 0.8)]
