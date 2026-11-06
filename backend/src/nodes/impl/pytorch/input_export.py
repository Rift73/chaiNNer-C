"""An FP16 ONNX export's input and first convolution in FP32, its body in FP16.

Near 1.0 FP16 rounds an image by up to 1/16 of an 8-bit step. Trained models are very
sensitive to that on smooth content: on a diagonal gradient a 4x RealESRGANv2 goes from
FP32's output to 31.9 dB (14.8 of 255 at worst) from the rounded input alone, with every
layer behind it in FP32, and 44.2 dB on the same gradient in 8-bit steps. The first
convolution's output is small on smooth content, where FP16 is exact enough: with an FP32
input and first convolution that model reaches 74.2 dB with the whole body in FP16. A
body's rounding is not what costs these images.

The output stays FP16, which is what says the ONNX is FP16 (onnx/load.py) and lets Build
Engine build it as one. ESRGAN stays plain FP16: 4x-UltraSharp gains 0.4 dB from this on a
real crop. HAT and DAT stay as they are too. The loaded model is not changed: the rewrite
works on the export's copy."""

from __future__ import annotations

import copy

import torch
from spandrel import ModelDescriptor
from spandrel.architectures.PLKSR.__arch.PLKSR import PLKSR
from spandrel.architectures.PLKSR.__arch.RealPLKSR import RealPLKSR
from spandrel.architectures.SPAN.__arch.span import Conv3XC
from torch import nn

# The module each architecture's image enters by, by the name FP16_ARCHS holds
_FIRST_CONVOLUTION = {
    "RealESRGAN Compact": "body.0",
    "SPAN": "conv_1",
    "PLKSR": "feats.0",
    "OmniSR": "input",
}


class _Fp32Convolution(nn.Module):
    """The convolution in FP32 and an FP16 output. Its weights are buffers: they are the
    only FP32 tensors of the copy, and the copy's type is its parameters'."""

    weight: torch.Tensor
    bias: torch.Tensor | None

    def __init__(self, source: nn.Conv2d) -> None:
        super().__init__()
        bias = source.bias
        self.register_buffer("weight", source.weight.detach().float())
        self.register_buffer("bias", None if bias is None else bias.detach().float())
        self.stride = source.stride
        self.padding = source.padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.nn.functional.conv2d(
            x, self.weight, self.bias, self.stride, self.padding
        )
        return output.half()


class _Fp16Repeat(nn.Module):
    """PLKSR's image skip, which would turn the sum after it and the output to FP32."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.source(x).half()


def fp32_first_convolution(model: ModelDescriptor) -> _Fp32Convolution | None:
    """The first convolution of an FP16-converting architecture with its weights as
    loaded, taken before the export copy turns to FP16 (take_fp32_input puts it in); None
    for any other model."""
    name = _FIRST_CONVOLUTION.get(model.architecture.name)
    if name is None:
        return None
    module = model.model.get_submodule(name)
    if isinstance(module, Conv3XC):
        # SPAN's convolution in its evaluation form, folded here from the loaded weights
        module = copy.deepcopy(module).float()
        module.update_params()
        module = module.eval_conv
    return _Fp32Convolution(module)


def take_fp32_input(model: ModelDescriptor, first: _Fp32Convolution | None) -> None:
    """Puts `first` (fp32_first_convolution's) in its place in `model`, an FP16 copy for
    export."""
    if first is None:
        return
    net = model.model
    net.set_submodule(_FIRST_CONVOLUTION[model.architecture.name], first)
    if isinstance(net, (PLKSR, RealPLKSR)):
        net.repeat_op = _Fp16Repeat(net.repeat_op)


def input_dtype(model: ModelDescriptor) -> torch.dtype:
    """The type the export copy `model` takes its image in."""
    name = _FIRST_CONVOLUTION.get(model.architecture.name)
    if name is not None and isinstance(
        model.model.get_submodule(name), _Fp32Convolution
    ):
        return torch.float32
    return model.dtype
