"""DySample's sampling grid at every input size in an ONNX export.

DySample divides its sampling coordinates by the feature map's size through
`torch.tensor([W, H])`, which a trace stores as a constant: the size of the export's
example. Every other size then samples the wrong part of the image: a 4x RealPLKSR with
DySample, exported at 32x32, reaches 22 dB against PyTorch at 64x64 and 16 dB at
128x128. The export divides by the traced size instead, which computes the same values
at the example's size and stays right at the others. The loaded model is not changed:
the rewrite works on a copy, so Save Model keeps it as it is."""

from __future__ import annotations

import copy
from typing import TypeVar

import torch
from spandrel import ModelDescriptor
from spandrel.architectures.__arch_helpers.dysample import DySample
from spandrel.architectures.FDAT.__arch.fdat import DySample as FdatDySample
from spandrel.architectures.GaterV3.__arch.gaterv3 import DySample as GaterDySample
from spandrel.architectures.GFISRv2.__arch.gfisrv2 import DySample as GfisrDySample
from spandrel.architectures.SMoSR.__arch.smosr_arch import DySample as SmosrDySample
from torch import nn
from torch.nn import functional
from torch.onnx.operators import shape_as_tensor

_Descriptor = TypeVar("_Descriptor", bound=ModelDescriptor)

# Spandrel's copies of DySample, which differ only in their end convolution's kernel
_DYSAMPLES = (DySample, FdatDySample, GaterDySample, GfisrDySample, SmosrDySample)


class _TracedSizeDySample(DySample):
    def __init__(self, source: nn.Module) -> None:
        nn.Module.__init__(self)
        self.__dict__.update(source.__dict__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        init_pos = self.get_buffer("init_pos")
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + init_pos
        batch, _, height, width = offset.shape
        offset = offset.view(batch, 2, -1, height, width)
        coords = (
            torch.stack(
                torch.meshgrid(
                    [torch.arange(width) + 0.5, torch.arange(height) + 0.5],
                    indexing="ij",
                )
            )
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .to(x)
        )
        # The width and height, which a trace records as the graph's own
        normalizer = shape_as_tensor(offset)[[4, 3]].to(x).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        out_height, out_width = self.scale * height, self.scale * width
        coords = (
            functional.pixel_shuffle(
                coords.reshape(batch, -1, height, width), self.scale
            )
            .view(batch, 2, -1, out_height, out_width)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = functional.grid_sample(
            x.reshape(batch * self.groups, -1, height, width),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(batch, -1, out_height, out_width)
        if self.end_convolution:
            output = self.end_conv(output)
        return output


def for_export(model: _Descriptor) -> _Descriptor:
    """A copy of a model with DySample with its sampling grid following the input's
    size; any other model as it is."""
    if not any(isinstance(m, _DYSAMPLES) for m in model.model.modules()):
        return model
    model = copy.deepcopy(model)
    net = model.model
    for name, module in list(net.named_modules()):
        if isinstance(module, _DYSAMPLES):
            net.set_submodule(name, _TracedSizeDySample(module))
    return model
