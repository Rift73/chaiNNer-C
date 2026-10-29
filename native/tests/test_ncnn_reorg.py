"""NCNN models that start with a Reorg, run on ncnn's CPU (upstream #3327).

A Reorg (space-to-depth) of stride 2 drops the last row or column of an odd-sized
input, so ncnn_auto_split pads an odd image to even and crops the result back, and
keeps every tile even. Vulkan stays hidden (conftest), so ncnn runs on the CPU.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nodes.impl.ncnn.auto_split import _MultipleTiler, ncnn_auto_split
from nodes.impl.ncnn.model import (
    DTYPE_FP32,
    NcnnLayer,
    NcnnModel,
    NcnnModelWrapper,
    NcnnWeight,
)
from nodes.impl.ncnn.session import create_ncnn_net
from nodes.impl.upscale.tiler import MaxTileSize, NoTiling
from packages.chaiNNer_ncnn.settings import NcnnSettings

SETTINGS = NcnnSettings(
    gpu_index=0, winograd=False, sgemm=False, threads=1, blocktime=0, budget_limit=0
)


def nearest_2x_reorg_model() -> NcnnModelWrapper:
    """Reorg 2, a 1x1 convolution 12 -> 48 and PixelShuffle 4: a nearest 2x upscale.

    Output pixel (4i + a, 4j + b) of the half-size grid is PixelShuffle channel
    16c + 4a + b, and the input pixel it repeats, (2i + a // 2, 2j + b // 2), is
    Reorg channel 4c + 2 (a // 2) + b // 2.
    """
    weight = np.zeros((48, 12, 1, 1), np.float32)
    for c in range(3):
        for a in range(4):
            for b in range(4):
                weight[16 * c + 4 * a + b, 4 * c + 2 * (a // 2) + b // 2] = 1
    model = NcnnModel(4, 4)
    layers = [
        ("Input", "in", (), ("input",), {}),
        ("Reorg", "reorg", ("input",), ("r",), {0: 2}),
        ("Convolution", "conv", ("r",), ("c",), {0: 48, 1: 1, 6: weight.size}),
        ("PixelShuffle", "ps", ("c",), ("output",), {0: 4}),
    ]
    for op, name, inputs, outputs, params in layers:
        layer = NcnnLayer(
            op, name, len(inputs), len(outputs), list(inputs), list(outputs)
        )
        for key, value in params.items():
            layer.params[key] = value
        if op == "Convolution":
            layer.weight_data["weight"] = NcnnWeight(weight, DTYPE_FP32)
        model.layers.append(layer)
    return NcnnModelWrapper(model)


MODEL = nearest_2x_reorg_model()
NET = create_ncnn_net(MODEL, SETTINGS)


def upscale(img: np.ndarray, tiler) -> np.ndarray:
    return ncnn_auto_split(
        img,
        NET,
        input_name="input",
        output_name="output",
        blob_vkallocator=None,
        staging_vkallocator=None,
        tiler=tiler,
        size_multiple=MODEL.size_multiple,
    )


def image(h: int, w: int) -> np.ndarray:
    # Whole uint8 steps, which the NCNN input conversion keeps exactly.
    rng = np.random.default_rng(h * 1000 + w)
    return (rng.integers(0, 256, (h, w, 3)) / 255).astype(np.float32)


def nearest_2x(img: np.ndarray) -> np.ndarray:
    return img.repeat(2, axis=0).repeat(2, axis=1)


def test_reorg_model_reports_2x_rgb_and_an_even_multiple():
    assert (MODEL.scale, MODEL.in_nc, MODEL.out_nc, MODEL.size_multiple) == (
        2,
        3,
        3,
        2,
    )


@pytest.mark.parametrize("size", [(37, 45), (36, 45), (37, 44), (36, 44), (1, 1)])
def test_odd_sized_image_is_padded_and_cropped_back(size):
    img = image(*size)
    result = upscale(img, NoTiling())
    assert result.shape == (2 * size[0], 2 * size[1], 3)
    np.testing.assert_allclose(result, nearest_2x(img), atol=1e-6)


@pytest.mark.parametrize("tile", [16, 13, 25, 7])
@pytest.mark.parametrize("size", [(37, 45), (64, 50)])
def test_odd_tiles_are_kept_even(tile, size):
    # MaxTileSize(16) cuts the 38x46 padded image into tiles of ceil(38 / 3) = 13
    # and ceil(46 / 3) = 16; a 13-pixel tile lost a row in the Reorg and failed
    # the tiler's whole-multiple check.
    img = image(*size)
    result = upscale(img, MaxTileSize(tile))
    assert result.shape == (2 * size[0], 2 * size[1], 3)
    np.testing.assert_allclose(result, nearest_2x(img), atol=1e-6)


# The image is padded to the multiple first, so its length is one.
@pytest.mark.parametrize(
    ("multiple", "length"),
    [(m, n) for m in (2, 4) for n in (2, 4, 38, 46, 100, 822, 1300) if n % m == 0],
)
@pytest.mark.parametrize("max_size", [1, 7, 13, 16, 25, 256, 4096])
def test_multiple_tiler_keeps_the_tile_size_a_multiple(multiple, length, max_size):
    wrapped = MaxTileSize(max_size).starting_tile_size(length, length, 3)[0]
    tiler = _MultipleTiler(MaxTileSize(max_size), multiple)
    size = tiler.starting_tile_size(length, length, 3)[0]
    if wrapped >= length:
        assert size == wrapped
        return
    # The size the tiler cuts the axis into is this size itself.
    assert math.ceil(length / math.ceil(length / size)) == size
    assert size % multiple == 0
    assert size <= max(wrapped, multiple)


def test_other_models_keep_their_tiler():
    tiler = MaxTileSize(13)
    img = image(5, 5)
    calls = []

    class Spy(MaxTileSize):
        def starting_tile_size(self, width, height, channels):
            calls.append((width, height))
            return tiler.starting_tile_size(width, height, channels)

    ncnn_auto_split(
        img,
        NET,
        input_name="input",
        output_name="output",
        blob_vkallocator=None,
        staging_vkallocator=None,
        tiler=Spy(13),
    )
    # No multiple: the image is neither padded nor given a wrapped tiler.
    assert calls == [(5, 5)]


def test_a_reorg_stride_given_as_a_list_is_refused_by_name():
    # A param file can give any parameter as an array; the stride is one number (it
    # failed with int()'s TypeError).
    model = NcnnModel(2, 2)
    for op, name, inputs, outputs in (
        ("Input", "in", (), ("input",)),
        ("Reorg", "reorg", ("input",), ("output",)),
    ):
        model.layers.append(
            NcnnLayer(op, name, len(inputs), len(outputs), list(inputs), list(outputs))
        )
    model.layers[1].params[0] = [2, 2]
    with pytest.raises(ValueError, match="Reorg layer reorg's stride is a list"):
        NcnnModelWrapper.get_size_multiple(model)
