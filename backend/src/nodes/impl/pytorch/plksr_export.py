"""PLKSR's layout for an ONNX export: its large convolution, and DySample's precision.

A block writes the convolution's output over its first channels in place, which the trace
exports as a ScatterND with the arithmetic for its indices, half of a block's 94 nodes. The
module's training path computes the same values with a split and a join. ONNX Runtime
then has no indices to build per image: at 256x256 in FP16 a 64-channel RealPLKSR with 28
blocks runs in 26 ms instead of 108 ms on CUDA, with the same output. TensorRT folds either
form (6.2 ms). The loaded model is not changed: the rewrite works on a copy, so Save Model
keeps it as it is.

DySample computes the coordinates it samples at in pixels, which FP16 rounds to a quarter
of a pixel from 256 px up. A 2x RealPLKSR at 256x256 reaches 42 dB against PyTorch FP32 and
36 dB at 512x512; a 4x at 256x256 51 dB. With DySample alone in FP32 they reach 71, 66 and
68 dB, at most 0.8 ms slower."""

from __future__ import annotations

import copy
from typing import TypeVar

import torch
from spandrel import ModelDescriptor
from spandrel.architectures.__arch_helpers.dysample import DySample
from spandrel.architectures.PLKSR.__arch.PLKSR import PLKConv2d, RectSparsePLKConv2d
from spandrel.architectures.PLKSR.__arch.RealPLKSR import PLKConv2d as RealPLKConv2d
from spandrel.architectures.PLKSR.__arch.RealPLKSR import RealPLKSR
from torch import nn

_Descriptor = TypeVar("_Descriptor", bound=ModelDescriptor)


class _TrainingPath(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Here and not at the rewrite: the export turns the whole model to evaluation.
        self.source.training = True
        return self.source(x)


class _Fp32Sampling(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.source(x.float()).to(x.dtype)


def fp32_sampler(model: ModelDescriptor) -> nn.Module | None:
    """A copy of a RealPLKSR's DySample with its weights as loaded, taken before the
    export copy turns to FP16 (sample_in_fp32 puts it back); None for any other model."""
    net = model.model
    if isinstance(net, RealPLKSR) and isinstance(net.to_img, DySample):
        return copy.deepcopy(net.to_img).float()
    return None


def sample_in_fp32(model: ModelDescriptor, sampler: nn.Module | None) -> None:
    """Puts `sampler` (fp32_sampler's) in the RealPLKSR's DySample's place in `model`, an
    FP16 copy for export, so that the sampling runs in FP32 on its FP32 weights."""
    if sampler is not None:
        model.model.set_submodule("to_img", _Fp32Sampling(sampler))


def for_export(model: _Descriptor) -> _Descriptor:
    """A copy of a PLKSR model with its large convolutions split and joined; any other
    model as it is."""
    if model.architecture.id != "PLKSR":
        return model
    model = copy.deepcopy(model)
    net = model.model
    for name, module in list(net.named_modules()):
        if isinstance(module, (PLKConv2d, RealPLKConv2d, RectSparsePLKConv2d)):
            net.set_submodule(name, _TrainingPath(module))
    return model
