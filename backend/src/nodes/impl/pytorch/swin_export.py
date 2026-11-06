"""SwinIR, Swin2SR, RGT and DRCT for an FP16 ONNX export: a copy that traces in FP16 and
keeps the end of its last residual group in FP32.

Their shifted windows build the attention mask in FP32 at any size but the one they were
trained at. An FP16 score plus an FP32 mask is an FP32 score, which the next product with
an FP16 value refuses, so the trace stops. The copy casts the mask to the weights'
precision: its 0 and -100 are exact in FP16.

As in HAT and DAT (attention_export), the end of the last residual group outgrows FP16's
65504, flat images first and natural ones too, right before the final norm scales the
values back down. In FP16 that is inf, and every pixel after it NaN. Measured in FP32 at 48
and 64 px on nine trained SwinIR, Swin2SR and RGT files, it reaches 707,133 in a SwinIR-M,
192,836 in an RGT, 70,875 in a Swin2SR and 55,158 in a SwinIR-L, while all that stays in
FP16 peaks at 8,304, 7.9 times under 65504. The copy keeps that end in FP32: the closing
convolution, in Swin2SR also the patch embedding's convolution after it (the one that
outgrows FP16 there), the group's sum, and the final norm, which hands FP16 back. Three
trained DRCT files stay in range (4,521 at most), but this end differs over 300 times
between files of one architecture, so DRCT's, at its group's last adjusting convolution,
is kept too.

Swin2SR's cosine attention scales each vector to unit length, which TensorRT's FP16
ReduceL2 cannot do (omnisr_export), so the copy takes those lengths in FP32 too.

The loaded model is not changed: the rewrite works on a copy, so Save Model keeps it as it
is. An export in another precision is not rewritten."""

from __future__ import annotations

import copy
import math
from typing import TypeVar

import torch
from spandrel import ModelDescriptor
from spandrel.architectures.__arch_helpers.fp32_last_group import (
    FP32Conv2d,
    FP32LayerNorm,
    keep_last_group_in_fp32,
)
from spandrel.architectures.DRCT.__arch.drct_arch import (
    SwinTransformerBlock as DrctBlock,
)
from spandrel.architectures.RGT.__arch.rgt import L_SA
from spandrel.architectures.Swin2SR.__arch.Swin2SR import (
    SwinTransformerBlock as Swin2srBlock,
)
from spandrel.architectures.Swin2SR.__arch.Swin2SR import (
    WindowAttention as Swin2srAttention,
)
from spandrel.architectures.SwinIR.__arch.SwinIR import (
    SwinTransformerBlock as SwinirBlock,
)
from torch import nn

from .omnisr_export import unit_length

ARCHS = ("SwinIR", "Swin2SR", "RGT", "DRCT")

_Descriptor = TypeVar("_Descriptor", bound=ModelDescriptor)


def _adopt(module: nn.Module, source: nn.Module) -> None:
    """`module` takes over `source`'s parameters, buffers and submodules."""
    nn.Module.__init__(module)
    module.__dict__.update(source.__dict__)


class _SwinirBlock(SwinirBlock):
    def __init__(self, source: SwinirBlock) -> None:
        _adopt(self, source)

    def calculate_mask(self, x_size: tuple[int, int]) -> torch.Tensor:
        return super().calculate_mask(x_size).to(self.norm1.weight.dtype)


class _Swin2srBlock(Swin2srBlock):
    def __init__(self, source: Swin2srBlock) -> None:
        _adopt(self, source)

    def calculate_mask(self, x_size: tuple[int, int]) -> torch.Tensor:
        return super().calculate_mask(x_size).to(self.norm1.weight.dtype)


class _DrctBlock(DrctBlock):
    def __init__(self, source: DrctBlock) -> None:
        _adopt(self, source)

    def calculate_mask(self, x_size: tuple[int, int]) -> torch.Tensor:
        return super().calculate_mask(x_size).to(self.norm1.weight.dtype)


class _RgtAttention(L_SA):
    def __init__(self, source: L_SA) -> None:
        _adopt(self, source)

    def calculate_mask(self, height: int, width: int) -> tuple[torch.Tensor, ...]:
        dtype = self.proj.weight.dtype
        return tuple(mask.to(dtype) for mask in super().calculate_mask(height, width))


_MASKED: dict[type[nn.Module], type[nn.Module]] = {
    SwinirBlock: _SwinirBlock,
    Swin2srBlock: _Swin2srBlock,
    DrctBlock: _DrctBlock,
    L_SA: _RgtAttention,
}


class _UnitLengthAttention(Swin2srAttention):
    def __init__(self, source: Swin2srAttention) -> None:
        _adopt(self, source)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, n, c = x.shape
        bias = None
        if self.q_bias is not None:
            bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = nn.functional.linear(x, self.qkv.weight, bias)
        qkv = qkv.reshape(b, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = unit_length(q) @ unit_length(k).transpose(-2, -1)
        attn = attn * self.logit_scale.clamp_max(math.log(100.0)).exp()

        table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        side = self.window_size[0] * self.window_size[1]
        position = table[self.relative_position_index.view(-1)].view(side, side, -1)
        attn = attn + 16 * torch.sigmoid(position.permute(2, 0, 1)).unsqueeze(0)

        if mask is not None:
            windows = mask.shape[0]
            attn = attn.view(b // windows, windows, self.num_heads, n, n)
            attn = (attn + mask.unsqueeze(1).unsqueeze(0)).view(
                -1, self.num_heads, n, n
            )
        attn = self.attn_drop(self.softmax(attn))
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(b, n, c)))


def _keep_last_group_end_in_fp32(arch: str, net: nn.Module) -> None:
    group = net.layers[-1]
    if arch == "DRCT":
        group.adjust5 = FP32Conv2d(group.adjust5)
    else:
        group.conv = keep_last_group_in_fp32(group.conv)
    if arch == "Swin2SR":
        group.patch_embed.proj = FP32Conv2d(group.patch_embed.proj)
    net.norm = FP32LayerNorm(net.norm)


def for_export(model: _Descriptor) -> _Descriptor:
    """A copy of an FP16 SwinIR, Swin2SR, RGT or DRCT rewritten for export; any other model
    as it is."""
    if model.architecture.id not in ARCHS or model.dtype != torch.float16:
        return model
    model = copy.deepcopy(model)
    net = model.model
    for name, module in list(net.named_modules()):
        block = _MASKED.get(type(module))
        if block is not None:
            net.set_submodule(name, block(module))
        elif type(module) is Swin2srAttention:
            net.set_submodule(name, _UnitLengthAttention(module))
    _keep_last_group_end_in_fp32(model.architecture.id, net)
    return model
