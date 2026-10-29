"""PyTorch nodes leave the models they are given as they are: a model output feeds several
nodes (Upscale Image, Save Model, Convert To ONNX), so an upscale in FP16 or an
interpolation of models on two devices must not convert or move it in place."""

from __future__ import annotations

import gc
import weakref
from typing import cast

import numpy as np
import pytest
import torch
from spandrel import ImageModelDescriptor, ModelDescriptor
from spandrel.architectures.Compact import Compact, CompactArch

# Tiling runs in the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import NodeContext, Progress, SettingsParser
from nodes.impl.pytorch import utils
from nodes.impl.pytorch.auto_split import pytorch_auto_split
from nodes.impl.pytorch.utils import on_device
from nodes.impl.upscale.tiler import NoTiling
from packages.chaiNNer_pytorch.pytorch.utility.interpolate_models import (
    interpolate_models_node,
)

CPU = torch.device("cpu")


def compact(weight_scale: float = 1.0) -> ImageModelDescriptor:
    """A random 2x Compact, its weights scaled by `weight_scale`, as Interpolate Models
    returns a model: in FP32."""
    torch.manual_seed(0)
    net = Compact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=2, upscale=2)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.mul_(weight_scale)
    model = CompactArch().load(net.state_dict())
    assert isinstance(model, ImageModelDescriptor)
    model.model.requires_grad_(False)
    return model


def weights(model: ModelDescriptor) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.model.state_dict().items()}


def upscale(model: ImageModelDescriptor, use_fp16: bool) -> np.ndarray:
    image = np.random.default_rng(0).random((24, 32, 3), dtype=np.float32)
    return pytorch_auto_split(
        image, model, CPU, use_fp16, NoTiling(), Progress.noop_progress()
    )


def test_an_fp16_upscale_leaves_the_model_in_fp32():
    model = compact()
    before = weights(model)
    fp32 = upscale(model, use_fp16=False)
    fp16 = upscale(model, use_fp16=True)
    assert model.dtype == torch.float32
    after = weights(model)
    assert all(torch.equal(after[k], v) for k, v in before.items())
    # The model's FP32 weights after an FP16 upscale, not FP16-rounded ones
    np.testing.assert_array_equal(upscale(model, use_fp16=False), fp32)
    assert not np.array_equal(fp16, fp32)


def test_the_converted_copy_is_made_once_and_released_with_the_model():
    model = compact()
    copy = on_device(model, CPU, torch.float16)
    assert copy is not model and copy.dtype == torch.float16
    assert on_device(model, CPU, torch.float16) is copy
    assert on_device(model, CPU, torch.float32) is model
    released = weakref.ref(copy)
    del copy, model
    gc.collect()
    assert released() is None


def test_a_copy_is_dropped_once_the_model_runs_as_it_is():
    # FP16 turned off again (or the device the model is on chosen again): the copy's
    # memory is freed, not kept while the model lives.
    model = compact()
    released = weakref.ref(on_device(model, CPU, torch.float16))
    assert on_device(model, CPU, torch.float32) is model
    gc.collect()
    assert released() is None


def test_a_superseded_copy_is_dropped_before_the_next_is_made(
    monkeypatch: pytest.MonkeyPatch,
):
    model = compact()
    released = weakref.ref(on_device(model, CPU, torch.float16))
    alive_at_copy: list[bool] = []
    deepcopy = utils.copy.deepcopy

    def recorded(value: object) -> object:
        gc.collect()
        alive_at_copy.append(released() is not None)
        return deepcopy(value)

    monkeypatch.setattr(utils.copy, "deepcopy", recorded)
    assert on_device(model, CPU, torch.bfloat16).dtype == torch.bfloat16
    assert alive_at_copy == [False]


class Context:
    """The NodeContext member Interpolate Models uses: PyTorch on the CPU."""

    def __init__(self) -> None:
        self.settings = SettingsParser({"use_cpu": True})


def test_interpolating_models_on_two_devices_does_not_move_them(
    monkeypatch: pytest.MonkeyPatch,
):
    # Brightness-tested on an image of ones: Compact adds its input, so small weights
    # pass the test.
    model_a = compact(0.01)
    model_b = compact(0.02)
    # model_b on another device (the CPU stands in for it: CUDA is hidden)
    monkeypatch.setattr(
        ImageModelDescriptor,
        "device",
        property(lambda self: torch.device("meta") if self is model_b else CPU),
    )
    # Everything is on the CPU, so a move is recorded and changes nothing
    moved: list[ImageModelDescriptor] = []

    def recorded(
        self: ImageModelDescriptor, *args: object, **kwargs: object
    ) -> ImageModelDescriptor:
        moved.append(self)
        return self

    monkeypatch.setattr(ImageModelDescriptor, "to", recorded)
    before_a, before_b = weights(model_a), weights(model_b)
    mixed, _, _ = interpolate_models_node(
        cast(NodeContext, Context()), model_a, model_b, 50
    )
    assert not any(model is model_a or model is model_b for model in moved)
    mixed_weights = weights(mixed)
    for key, value in before_a.items():
        torch.testing.assert_close(mixed_weights[key], (value + before_b[key]) / 2)
