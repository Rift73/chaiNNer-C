"""PyTorch Interpolate Models tests an inpainting model's mix with a one-channel mask:
upstream made the mask and passed the image instead, which spandrel refused for LaMa
("Expected mask shape to be [1, 1, 16, 16], but got [1, 3, 16, 16]"), so no inpainting
model could be interpolated."""

from __future__ import annotations

import torch
from spandrel import ModelLoader
from spandrel.architectures.LaMa import LaMa

from packages.chaiNNer_pytorch.pytorch.utility.interpolate_models import (
    _check_interp_state,
)


def test_an_inpainting_mix_is_tested_with_a_mask():
    torch.manual_seed(0)
    state = LaMa(in_nc=4, out_nc=3).state_dict()  # 3 image channels and the mask
    assert ModelLoader().load_from_state_dict(state).purpose == "Inpainting"
    # Random weights may or may not pass the brightness test; the test itself must run.
    assert _check_interp_state(state) in (True, False)
