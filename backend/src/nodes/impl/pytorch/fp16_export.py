"""Rewrites that let an FP16 ONNX export hold where FP16 alone gives NaN or loses the image.

RetinexFormer's attention scales each vector to unit length, and an all-zero vector, which a
black image gives, divides by zero in FP16: its epsilon of 1e-12 is itself zero there, and
every pixel turns NaN. It takes OmniSR's length, in FP32 and at least FP16's smallest number.
Its product of two unit vectors runs over every pixel: at 256x256 each term is about 1 / 65536,
which FP16 keeps only as a subnormal, so a real FP16 runtime loses what a rounding of the sum
alone does not show. In FP32 that one product takes a trained FiveK on TensorRT, FP16 engine
against FP32 engine at 256x256, from 54.8 to 68.6 dB on natural crops to 68.4 to 72.8 dB, and
from 44.5 to 48.2 dB on flat white, at the same 1.6 ms (FP32: 3.6 to 4.1 ms).

FDAT's channel attention takes the same product over every pixel of the tile, so its FP16 loses
more as the tile grows: a trained FDAT M, FP16 engine against FP32 engine on natural crops, read
60 to 68 dB at 128 px, 49 to 58 dB at 256 px and 31 to 51 dB at 512 px, with no NaN and with
smooth images above 66 dB. It takes the same lengths and the same FP32 product: 64 to 70 dB at
each of those sizes, at the same speed (Fixed 512 px: 65.25 -> 64.74 ms; FP32: 223.24 ms).

DAT's channel attention takes that product too, after lengths it already takes in FP32. A
trained DAT2, Dynamic FP16 engine against Dynamic FP32 engine on natural crops, read 65 to 70
dB at 128 px, 56 to 65 dB at 256 px and 41 to 58 dB at 512 px; with the product in FP32, 66 to
70 dB at each. Here it costs speed: at 512 px 234.82 -> 357.30 ms (FP32: 573.03 ms).

Below opset 17 ONNX has no LayerNormalization: the exporter writes it out of plain operators
and casts FP16 to float64 for them, so an FP16 graph holds float64 in every LayerNorm of HAT,
DAT, OmniSR, RetinexFormer, Uformer and the SwinIR family. The export's LayerNorms take FP32
there, where LayerNormalization itself computes in FP32 from opset 17 on.

The loaded model is not changed: the rewrite works on the export's copy."""

from __future__ import annotations

import copy
from typing import TypeVar

import torch
from einops import rearrange
from spandrel import ModelDescriptor
from spandrel.architectures.__arch_helpers.attention import in_migraphx_export
from spandrel.architectures.__arch_helpers.fp32_last_group import FP32LayerNorm
from spandrel.architectures.DAT.__arch.DAT import Adaptive_Channel_Attention
from spandrel.architectures.FDAT.__arch.fdat import FastChannelAttention
from spandrel.architectures.RetinexFormer.__arch.retinexformer_arch import IG_MSA
from torch import nn

from .omnisr_export import unit_length

_Descriptor = TypeVar("_Descriptor", bound=ModelDescriptor)


class _UnitLengthAttention(IG_MSA):
    def __init__(self, source: IG_MSA) -> None:
        nn.Module.__init__(self)
        self.__dict__.update(source.__dict__)

    def forward(self, x_in: torch.Tensor, illu_fea_trans: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        v_inp = self.to_v(x)
        q, k, v, illu_attn = (
            rearrange(t, "b n (h d) -> b h n d", h=self.num_heads)
            for t in (self.to_q(x), self.to_k(x), v_inp, illu_fea_trans.flatten(1, 2))
        )
        v = (v * illu_attn).transpose(-2, -1)
        q = unit_length(q.transpose(-2, -1))
        k = unit_length(k.transpose(-2, -1))
        attn = k.float() @ q.float().transpose(-2, -1)
        attn = attn.to(x_in.dtype) * self.rescale
        out = (attn.softmax(dim=-1) @ v).permute(0, 3, 1, 2)
        out = out.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(out).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2))
        return out_c + out_p.permute(0, 2, 3, 1)


class _UnitLengthChannelAttention(FastChannelAttention):
    def __init__(self, source: FastChannelAttention) -> None:
        nn.Module.__init__(self)
        self.__dict__.update(source.__dict__)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.nh, c // self.nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = unit_length(q.transpose(-2, -1))
        k = unit_length(k.transpose(-2, -1))
        attn = q.float() @ k.float().transpose(-2, -1)
        attn = (attn.to(x.dtype) * self.temp).softmax(dim=-1)
        return self.proj(
            (attn @ v.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(b, n, c)
        )


class _Fp32ProductChannelAttention(Adaptive_Channel_Attention):
    def __init__(self, source: Adaptive_Channel_Attention) -> None:
        nn.Module.__init__(self)
        self.__dict__.update(source.__dict__)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = torch.nn.functional.normalize(q.transpose(-2, -1).float(), dim=-1)
        k = torch.nn.functional.normalize(k.transpose(-2, -1).float(), dim=-1)
        v = v.transpose(-2, -1)
        v_ = v.reshape(b, c, n).contiguous().view(b, c, height, width)
        attn = (q @ k.transpose(-2, -1)).to(x.dtype) * self.temperature
        attn = self.attn_drop(attn.softmax(dim=-1))
        if in_migraphx_export():
            attened_x = v.transpose(-2, -1) @ attn.transpose(-2, -1)
            attened_x = attened_x.permute(0, 2, 1, 3).reshape(b, n, c)
        else:
            attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(b, n, c)
        conv_x = self.dwconv(v_)
        attention_map = (
            attened_x.transpose(-2, -1).contiguous().view(b, c, height, width)
        )
        channel_map = self.channel_interaction(attention_map)
        spatial_map = self.spatial_interaction(conv_x).permute(0, 2, 3, 1)
        spatial_map = spatial_map.contiguous().view(b, n, 1)
        attened_x = attened_x * torch.sigmoid(spatial_map)
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, n, c)
        return self.proj_drop(self.proj(attened_x + conv_x))


_ATTENTIONS: dict[type[nn.Module], type[nn.Module]] = {
    Adaptive_Channel_Attention: _Fp32ProductChannelAttention,
    FastChannelAttention: _UnitLengthChannelAttention,
    IG_MSA: _UnitLengthAttention,
}


def for_export(model: _Descriptor) -> _Descriptor:
    """A copy of an FP16 RetinexFormer, FDAT or DAT with its vector lengths and attention
    product in FP32; any other model as it is."""
    names = ("RetinexFormer", "FDAT", "DAT")
    if model.dtype != torch.float16 or model.architecture.name not in names:
        return model
    model = copy.deepcopy(model)
    net = model.model
    for name, module in list(net.named_modules()):
        attention = _ATTENTIONS.get(type(module))
        if attention is not None:
            net.set_submodule(name, attention(module))
    return model


def for_opset(model: _Descriptor, opset_version: int) -> _Descriptor:
    """A copy of an FP16 model with its LayerNorms in FP32, below opset 17; any other model
    as it is."""
    if opset_version >= 17:
        return model
    norms = [
        name
        for name, module in model.model.named_modules()
        if type(module) is nn.LayerNorm
        and module.elementwise_affine
        and module.weight.dtype == torch.float16
    ]
    if not norms:
        return model
    model = copy.deepcopy(model)
    net = model.model
    for name in norms:
        net.set_submodule(name, FP32LayerNorm(net.get_submodule(name)))
    return model
