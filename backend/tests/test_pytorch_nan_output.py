"""PyTorch upscaling refuses a result with NaN instead of returning it to be saved as
black pixels (upstream chaiNNer #3071: OmniSR overflows in FP16)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from spandrel import ImageModelDescriptor
from spandrel.architectures.Compact import Compact, CompactArch

# Tiling runs in the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import Progress
from nodes.impl.pytorch.auto_split import pytorch_auto_split
from nodes.impl.upscale.tiler import ExactTileSize, NoTiling, Tiler

CPU = torch.device("cpu")


def compact(first_weight_scale: float) -> ImageModelDescriptor:
    """A random 2x Compact; its first convolution's weights are scaled by
    `first_weight_scale`."""
    torch.manual_seed(0)
    net = Compact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=2, upscale=2)
    first = net.body[0]
    assert isinstance(first, torch.nn.Conv2d)
    with torch.no_grad():
        first.weight.mul_(first_weight_scale)
    model = CompactArch().load(net.state_dict())
    assert isinstance(model, ImageModelDescriptor)
    return model


def overflowing() -> ImageModelDescriptor:
    """Activations past FP16's 65504: finite in FP32, NaN in FP16, as OmniSR's
    variance."""
    return compact(1e5)


def image() -> np.ndarray:
    return np.random.default_rng(0).random((24, 32, 3), dtype=np.float32)


def upscale(
    img: np.ndarray,
    model: ImageModelDescriptor,
    use_fp16: bool,
    tiler: Tiler | None = None,
):
    return pytorch_auto_split(
        img, model, CPU, use_fp16, tiler or NoTiling(), Progress.noop_progress()
    )


def tiled_image() -> np.ndarray:
    """Several tiles at a 32 px tile size."""
    return np.random.default_rng(0).random((64, 96, 3), dtype=np.float32)


def test_finite_result_is_the_model_output_unchanged():
    model = overflowing()
    img = image()
    # BGR HWC to the model's RGB NCHW and back, as pytorch_auto_split converts.
    tensor = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0)
    expected = model(tensor).squeeze(0).permute(1, 2, 0).flip(2).numpy()
    np.testing.assert_array_equal(upscale(img, model, use_fp16=False), expected)


def test_fp16_overflow_names_fp16_mode():
    with pytest.raises(RuntimeError, match="turn off 'Use FP16 Mode'"):
        upscale(image(), overflowing(), use_fp16=True)


def test_fp32_nan_is_an_error_without_the_fp16_hint():
    with pytest.raises(RuntimeError, match="invalid values") as error:
        upscale(image(), compact(float("nan")), use_fp16=False)
    assert "FP16" not in str(error.value)


def test_nan_from_the_input_passes_through():
    img = image()
    img[5, 7] = np.nan
    result = upscale(img, compact(1.0), use_fp16=False)
    assert np.isnan(result).any()
    assert np.isfinite(result[-8:, -8:]).all()  # beyond the receptive field


def test_tiled_nan_fails_at_the_first_tile():
    model = compact(float("nan"))
    tiles = 0

    def count(*_: object) -> None:
        nonlocal tiles
        tiles += 1

    model.model.register_forward_hook(count)
    with pytest.raises(RuntimeError, match="invalid values"):
        upscale(tiled_image(), model, use_fp16=False, tiler=ExactTileSize((32, 32)))
    assert tiles == 1


def test_tiled_nan_from_the_input_passes_through():
    img = tiled_image()
    img[5, 7] = np.nan
    result = upscale(img, compact(1.0), use_fp16=False, tiler=ExactTileSize((32, 32)))
    assert np.isnan(result).any()
    assert np.isfinite(result[-8:, -8:]).all()  # beyond the receptive field
