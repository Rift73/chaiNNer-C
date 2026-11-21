"""WSR's context gates for an FP16 ONNX export: their 4x upsampling runs in FP32.

Each gate projects the LR/4 context map to a scale and a shift per channel and upsamples
them bilinearly to the trunk's size, where a sigmoid of the scale multiplies the trunk.
In FP16 those maps lose the precision the sigmoid needs, as RSR's engine sets found for
the same gates (tensorrt/rsr_export.py runs every Resize in FP32 for that reason): the
export takes the upsampling in FP32 and the rest of the gate in FP16. The ONNX then holds
a Cast to FP32, the Resize and a Cast back per gate.

The loaded model is not changed: the rewrite works on a copy, so Save Model keeps it as it
is."""

from __future__ import annotations

import copy
from typing import TypeVar

import torch
from spandrel import ModelDescriptor
from spandrel.architectures.WSR.__arch.wsr_arch import ContextGate
from torch import nn
from torch.nn import functional

_Descriptor = TypeVar("_Descriptor", bound=ModelDescriptor)


class Fp32UpsamplingGate(ContextGate):
    """ContextGate's evaluation forward with its upsampling in FP32."""

    def __init__(self, source: ContextGate) -> None:
        nn.Module.__init__(self)
        self.__dict__.update(source.__dict__)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        maps = self.project(context)
        maps = functional.interpolate(
            maps.float(),
            size=(4 * maps.shape[-2], 4 * maps.shape[-1]),
            mode="bilinear",
            align_corners=False,
        ).to(maps.dtype)
        gamma, beta = maps[..., : x.shape[-2], : x.shape[-1]].chunk(2, dim=1)
        return x * (2.0 * torch.sigmoid(gamma + self.local_scale * x)) + beta


def for_export(model: _Descriptor) -> _Descriptor:
    """A copy of an FP16 WSR model with its context gates' upsampling in FP32; any other
    model as it is."""
    if model.architecture.id != "WSR" or model.dtype != torch.float16:
        return model
    model = copy.deepcopy(model)
    net = model.model
    for name, module in list(net.named_modules()):
        if isinstance(module, ContextGate):
            net.set_submodule(name, Fp32UpsamplingGate(module))
    return model
