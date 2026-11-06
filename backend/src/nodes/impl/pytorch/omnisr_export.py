"""OmniSR's attention for an FP16 ONNX export: two sums that FP16 cannot hold run in FP32.

The channel attention scales each vector to unit length. The length squares every value,
and in FP16 a small square rounds to zero: TensorRT's FP16 ReduceL2 then returns a length
of zero, the epsilon of 1e-12 is itself zero in FP16, and the division gives NaN that the
attention spreads over the whole image. Just above that the length is badly rounded. The
export takes the length in FP32 and divides by at least FP16's smallest number, so an
all-zero vector stays zero as in FP32.

The squeeze-and-excitation gate averages over the whole image. ONNX Runtime's CUDA FP16
ReduceMean adds up first, which overflows FP16 from about 256x256 and also gives NaN, and
loses accuracy below that. The export takes that mean in FP32.

The loaded model is not changed: the rewrite works on a copy, so Save Model keeps it as it
is."""

from __future__ import annotations

import copy
from typing import TypeVar

import torch
from einops import rearrange
from spandrel import ModelDescriptor
from spandrel.architectures.OmniSR.__arch.OSA import (
    Channel_Attention,
    Channel_Attention_grid,
    SqueezeExcitation,
)
from torch import nn

_Descriptor = TypeVar("_Descriptor", bound=ModelDescriptor)

_SMALLEST_FP16 = 2.0**-24


def unit_length(x: torch.Tensor) -> torch.Tensor:
    length = x.float().norm(2, dim=-1, keepdim=True).to(x.dtype)
    return x / length.clamp_min(_SMALLEST_FP16)


def _forward(
    module: Channel_Attention | Channel_Attention_grid,
    x: torch.Tensor,
    split: str,
    join: str,
) -> torch.Tensor:
    _, _, h, w = x.shape
    q, k, v = (
        rearrange(t, split, ph=module.ps, pw=module.ps, head=module.heads)
        for t in module.qkv_dwconv(module.qkv(x)).chunk(3, dim=1)
    )
    attn = unit_length(q) @ unit_length(k).transpose(-2, -1)
    out = (attn * module.temperature).softmax(dim=-1) @ v
    out = rearrange(
        out,
        join,
        h=h // module.ps,
        w=w // module.ps,
        ph=module.ps,
        pw=module.ps,
        head=module.heads,
    )
    return module.project_out(out)


class ChannelAttention(Channel_Attention):
    def __init__(self, source: Channel_Attention) -> None:
        nn.Module.__init__(self)
        self.__dict__.update(source.__dict__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _forward(
            self,
            x,
            "b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)",
            "b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)",
        )


class ChannelAttentionGrid(Channel_Attention_grid):
    def __init__(self, source: Channel_Attention_grid) -> None:
        nn.Module.__init__(self)
        self.__dict__.update(source.__dict__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _forward(
            self,
            x,
            "b (head d) (h ph) (w pw) -> b (ph pw) head d (h w)",
            "b (ph pw) head d (h w) -> b (head d) (h ph) (w pw)",
        )


class _ImageMean(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().mean((2, 3)).to(x.dtype)


def for_export(model: _Descriptor) -> _Descriptor:
    """A copy of an FP16 OmniSR model with its vector lengths and image means in FP32; any
    other model as it is."""
    if model.architecture.id != "OmniSR" or model.dtype != torch.float16:
        return model
    model = copy.deepcopy(model)
    net = model.model
    for name, module in list(net.named_modules()):
        if isinstance(module, Channel_Attention):
            net.set_submodule(name, ChannelAttention(module))
        elif isinstance(module, Channel_Attention_grid):
            net.set_submodule(name, ChannelAttentionGrid(module))
        elif isinstance(module, SqueezeExcitation):
            module.get_submodule("gate")[0] = _ImageMean()
    return model
