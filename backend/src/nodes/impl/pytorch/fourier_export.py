"""GFISRv2's and FIGSR's Fourier transforms in an ONNX export that ONNX Runtime loads.

Their Fourier units export through symbolics in spandrel that ONNX Runtime refuses at
every opset: a one-sided DFT of a signal in complex form, where the operator takes a
real one, and from opset 20 the axis as an attribute, which became an input. The export
uses the same transforms in operators as ONNX defines them.

An FP16 export takes each RMSNorm's division in FP32. The length squares every value, which
FP16 rounds to zero below 2^-12 and overflows above 256, and a spectrum holds both: a
Fourier unit's bins run from below the first to above the second (DC grows with the square
root of the tile's area). Its unit scales each bin's channels to about 1 whatever their size,
so the transform's FP32 goes on into the norm, and the rest of the unit is FP16.

The loaded model is not changed: the rewrite works on a copy, so Save Model keeps it as it
is."""

from __future__ import annotations

import copy
from typing import Any, TypeVar

import onnx
import torch
from spandrel import ModelDescriptor
from spandrel.architectures.FIGSR.__arch import figsr_arch
from spandrel.architectures.GFISRv2.__arch import gfisrv2
from torch import nn

_Descriptor = TypeVar("_Descriptor", bound=ModelDescriptor)

_RFFT2 = (figsr_arch.CustomRfft2Wrap, gfisrv2.CustomRfft2Wrap)
_IRFFT2 = (figsr_arch.CustomIrfft2Wrap, gfisrv2.CustomIrfft2Wrap)
_RMS_NORM = (figsr_arch.RMSNorm, gfisrv2.RMSNorm)
FOURIER_UNITS = (figsr_arch.FourierUnit, gfisrv2.FourierUnit)

# ONNX has no DFT before this opset
_DFT_OPSET = 17
# DFT's axis is an input from this opset, after its optional length
_DFT_AXIS_INPUT_OPSET = 20
# Of a spectrum's batch, channels, height, width, real and imaginary parts
_SWAP_HEIGHT_AND_WIDTH = [0, 1, 3, 2, 4]
# Both architectures' RMSNorm, in FP32: an FP16 copy holds it rounded
_RMS_EPSILON = 1e-6


def _integer(g: Any, value: int) -> torch.Value:
    return g.op("Constant", value_t=torch.tensor(value, dtype=torch.int64))


def _integers(g: Any, value: int) -> torch.Value:
    return g.op("Constant", value_t=torch.tensor([value], dtype=torch.int64))


def _size(g: Any, x: torch.Value, axis: int) -> torch.Value:
    return g.op("Gather", g.op("Shape", x), _integer(g, axis), axis_i=0)


def _dft(
    g: Any, x: torch.Value, axis: int, inverse: int = 0, onesided: int = 0
) -> torch.Value:
    if g.opset < _DFT_AXIS_INPUT_OPSET:
        return g.op("DFT", x, axis_i=axis, inverse_i=inverse, onesided_i=onesided)
    # No length: the signal's own, which shape inference then knows
    length = g.op("prim::Constant").setType(torch.OptionalType.ofTensor())
    return g.op(
        "DFT", x, length, _integer(g, axis), inverse_i=inverse, onesided_i=onesided
    )


def _height_dft(g: Any, x: torch.Value, inverse: int = 0) -> torch.Value:
    """The DFT over the height, with the height moved to the one axis TensorRT's parser
    takes a DFT on: the last before the real and imaginary parts."""
    swapped = g.op("Transpose", x, perm_i=_SWAP_HEIGHT_AND_WIDTH)
    y = _dft(g, swapped, 3, inverse=inverse)
    return g.op("Transpose", y, perm_i=_SWAP_HEIGHT_AND_WIDTH)


def _orthonormal_scale(g: Any, height: torch.Value, width: torch.Value) -> torch.Value:
    area = g.op("Mul", height, width)
    return g.op("Sqrt", g.op("Cast", area, to_i=onnx.TensorProto.FLOAT))


class _Rfft2(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_real(torch.fft.rfft2(x, dim=(2, 3), norm="ortho"))

    @staticmethod
    def symbolic(g: Any, x: torch.Value) -> torch.Value:
        height = _size(g, x, 2)
        width = _size(g, x, 3)
        signal = g.op("Unsqueeze", x, _integers(g, 4))
        y = _dft(g, signal, 3, onesided=1)
        y = _height_dft(g, y)
        return g.op("Div", y, _orthonormal_scale(g, height, width))


class _Irfft2(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfft2(torch.view_as_complex(x), dim=(2, 3), norm="ortho")

    @staticmethod
    def symbolic(g: Any, x: torch.Value) -> torch.Value:
        height = _size(g, x, 2)
        bins = _size(g, x, 3)
        width = g.op("Mul", g.op("Sub", bins, _integer(g, 1)), _integer(g, 2))
        y = _height_dft(g, x, inverse=1)
        # The bins a one-sided spectrum leaves out: the others' conjugates, mirrored
        # (by a Slice of constants: ModelOpt's AutoCast could not follow a computed Range)
        mirrored = g.op(
            "Slice",
            y,
            _integers(g, -2),
            _integers(g, 0),
            _integers(g, 3),
            _integers(g, -1),
        )
        conjugate = g.op("Constant", value_t=torch.tensor([[[[[1.0, -1.0]]]]]))
        rest = g.op("Mul", mirrored, conjugate)
        y = _dft(g, g.op("Concat", y, rest, axis_i=3), 3, inverse=1)
        real = g.op("Gather", y, _integer(g, 0), axis_i=4)
        return g.op("Mul", real, _orthonormal_scale(g, height, width))


class _Rfft2Module(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _Rfft2.apply(x)


class _Irfft2Module(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _Irfft2.apply(x)


class _Fp32DivisionNorm(nn.Module):
    """RMSNorm with its division in FP32, in whichever precision `x` comes, and its scale and
    offset in the copy's."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.scale = nn.Parameter(source.scale.detach().view(-1, 1, 1))
        self.offset = nn.Parameter(source.offset.detach().view(-1, 1, 1))
        self.rms = self.scale.numel() ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        length = x.norm(2, dim=1, keepdim=True)
        unit = (x / (length * self.rms + _RMS_EPSILON)).to(self.scale.dtype)
        return unit * self.scale + self.offset


class _Fp16FourierUnit(nn.Module):
    """A Fourier unit that hands its norms the transforms' FP32, not a rounding of it."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.__dict__.update(source.__dict__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, _ = x.shape
        ffted = self.rfft2(x.float()).permute(0, 4, 1, 2, 3).contiguous()
        ffted = self.rn(ffted.view(b, c * 2, h, -1))
        ffted = self.fpe(ffted) + ffted
        ffted = self.gelu(self.fdc(ffted))
        ffted = ffted.view(b, c, 2, h, -1).permute(0, 1, 3, 4, 2).contiguous()
        return self.post_norm(self.irfft2(ffted.float()))


def for_export(model: _Descriptor, opset_version: int) -> _Descriptor:
    """A copy of a model with Fourier units, which export in valid operators and in FP16 with
    what FP16 cannot hold in FP32; any other model as it is."""
    if not any(isinstance(m, _RFFT2) for m in model.model.modules()):
        return model
    if opset_version < _DFT_OPSET:
        raise ValueError(
            f"{model.architecture.name} needs Opset {_DFT_OPSET} or newer: ONNX has no"
            " Fourier transform before it."
        )
    half = model.dtype == torch.float16
    model = copy.deepcopy(model)
    net = model.model
    for name, module in list(net.named_modules()):
        if isinstance(module, _RFFT2):
            net.set_submodule(name, _Rfft2Module())
        elif isinstance(module, _IRFFT2):
            net.set_submodule(name, _Irfft2Module())
        elif half and isinstance(module, _RMS_NORM):
            net.set_submodule(name, _Fp32DivisionNorm(module))
        elif half and isinstance(module, FOURIER_UNITS):
            net.set_submodule(name, _Fp16FourierUnit(module))
    return model
