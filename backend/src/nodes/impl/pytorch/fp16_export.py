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
trained DAT2, Dynamic FP16 engine against strict FP32 on natural crops, read 53 to 65 dB at 256
px and 43 to 53 dB at 512 px, and scaling both sides so that no term is subnormal changes
nothing: what loses the image is one FP16 sum of every pixel's small term. FP32 holds, 66 to 69
dB, at a price: TensorRT's FP32 GEMM has no fast kernel, 32.6 of 79.7 ms at Fixed 256 px. So it
takes one FP16 product per row of the tile, each a short sum of scaled terms, and adds the rows
in FP32: the same 66 to 69 dB, Fixed 256 px 49.60 -> 47.60 ms and Dynamic 512 px 260.08 -> 249.04
ms against the plain FP16 product (FP32 product: 79.69 and 366.72 ms).

On a large image its lengths and its spatial attention's moves grow faster than the pixels: a
trained 4x DAT2 at 1200x864 took 1166.0 ms, 277.6 of them in one FP32 sum per channel over every
pixel. The lengths now sum along each row and then the rows, and the spatial attention takes each
branch's own channels in one pass, its spatial map from the tokens as linear layers and its mix in
tokens: 970.8 ms (Fixed: 1141.8 -> 921.8 ms), context 14.63 -> 13.68 GiB, crops the same to 0.1 dB.
Keeping the channel attention's v and attention B, C, H, W as well removes more moves and is no
faster: its GPU clock falls from about 2700 to about 1600 MHz, 1043.3 ms.

Below opset 17 ONNX has no LayerNormalization: the exporter writes it out of plain operators
and casts FP16 to float64 for them, so an FP16 graph holds float64 in every LayerNorm of HAT,
DAT, OmniSR, RetinexFormer, Uformer and the SwinIR family. The export's LayerNorms take FP32
there, where LayerNormalization itself computes in FP32 from opset 17 on.

The loaded model is not changed: the rewrite works on the export's copy."""

from __future__ import annotations

import copy
from typing import TypeVar

import torch
import torch.nn.functional as F
from einops import rearrange
from spandrel import ModelDescriptor
from spandrel.architectures.__arch_helpers.attention import in_migraphx_export
from spandrel.architectures.__arch_helpers.fp32_last_group import FP32LayerNorm
from spandrel.architectures.DAT.__arch.DAT import (
    Adaptive_Channel_Attention,
    Adaptive_Spatial_Attention,
)
from spandrel.architectures.FDAT.__arch.fdat import FastChannelAttention
from spandrel.architectures.RetinexFormer.__arch.retinexformer_arch import IG_MSA
from torch import nn

from .omnisr_export import unit_length

_Descriptor = TypeVar("_Descriptor", bound=ModelDescriptor)

# Both sides of a row product are scaled by this: terms of 1024 / pixels stay normal FP16 numbers
# to 2048x2048, and no row's sum, 1024 at most, comes near FP16's 65504.
_ROW_PRODUCT_SCALE = 32.0


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


def _unit_rows(x: torch.Tensor) -> torch.Tensor:
    """Each channel of B, heads, H, W, C at _ROW_PRODUCT_SCALE's length, in x's type: the
    squares in FP32, summed along each row and then the rows."""
    wide = x.float()
    length = wide.square().sum(3, keepdim=True).sum(2, keepdim=True).sqrt()
    return (wide / length.clamp_min(1e-12) * _ROW_PRODUCT_SCALE).to(x.dtype)


class _RowProductChannelAttention(Adaptive_Channel_Attention):
    def __init__(self, source: Adaptive_Channel_Attention) -> None:
        nn.Module.__init__(self)
        self.__dict__.update(source.__dict__)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        b, n, c = x.shape
        head = c // self.num_heads
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        rows = (b, self.num_heads, height, width, head)
        q, k = _unit_rows(q.reshape(rows)), _unit_rows(k.reshape(rows))
        v = v.transpose(-2, -1)
        v_ = v.reshape(b, c, n).contiguous().view(b, c, height, width)
        attn = (q.transpose(-2, -1) @ k).float().sum(2) / _ROW_PRODUCT_SCALE**2
        attn = attn.to(x.dtype) * self.temperature
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


class _TokenSpatialAttention(Adaptive_Spatial_Attention):
    def __init__(self, source: Adaptive_Spatial_Attention) -> None:
        nn.Module.__init__(self)
        self.__dict__.update(source.__dict__)

    def _spatial_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """spatial_interaction on B, L, C: its 1x1 convolutions as linear layers, with its
        batch norm folded into the first."""
        conv_0, norm, activation, conv_1 = self.spatial_interaction
        scale = norm.weight.float() / (norm.running_var.float() + norm.eps).sqrt()
        offset = norm.bias.float() - norm.running_mean.float() * scale
        weight = (conv_0.weight[:, :, 0, 0].float() * scale[:, None]).to(tokens.dtype)
        bias = (conv_0.bias.float() * scale + offset).to(tokens.dtype)
        x = activation(F.linear(tokens, weight, bias))
        return F.linear(x, conv_1.weight[:, :, 0, 0], conv_1.bias)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        d = C // self.num_heads
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, -1)
        v = qkv[:, :, :, 2, :, :d].permute(0, 3, 4, 1, 2).reshape(B, C, H, W)
        half = self.num_heads * qkv.shape[-1] // 2
        qkv = qkv.reshape(B, H, W, 3, 2 * half).permute(3, 0, 1, 2, 4)
        split = max(self.split_size)
        pad_r = (split - W % split) % split
        pad_b = (split - H % split) % split
        qkv = F.pad(qkv, (0, 0, 0, pad_r, 0, pad_b))
        full_h, full_w = H + pad_b, W + pad_r
        first, second = qkv[..., :half], qkv[..., half:]
        if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (
            self.rg_idx % 2 != 0 and self.b_idx % 4 == 0
        ):
            s_0, s_1 = self.shift_size
            mask_0, mask_1 = self.calculate_mask(
                full_h, full_w, dtype=x.dtype, device=x.device
            )
            first = torch.roll(first, (-s_0, -s_1), (2, 3))
            second = torch.roll(second, (-s_1, -s_0), (2, 3))
            x_0 = self.attns[0](first, full_h, full_w, mask=mask_0)
            x_1 = self.attns[1](second, full_h, full_w, mask=mask_1)
            x_0 = torch.roll(x_0, (s_0, s_1), (1, 2))
            x_1 = torch.roll(x_1, (s_1, s_0), (1, 2))
        else:
            x_0 = self.attns[0](first, full_h, full_w)
            x_1 = self.attns[1](second, full_h, full_w)
        heads = self.num_heads // 2
        attened = torch.cat(
            [
                part[:, :H, :W].reshape(B, H, W, heads, -1)[..., :d].reshape(B, L, C // 2)
                for part in (x_0, x_1)
            ],
            dim=2,
        )
        conv_x = self.dwconv(v)
        channel_map = self.channel_interaction(conv_x).reshape(B, 1, C)
        spatial_map = self._spatial_map(attened)
        conv_x = conv_x.flatten(2).transpose(1, 2)
        return self.proj_drop(
            self.proj(
                attened * torch.sigmoid(channel_map)
                + conv_x * torch.sigmoid(spatial_map)
            )
        )


_ATTENTIONS: dict[type[nn.Module], type[nn.Module]] = {
    Adaptive_Channel_Attention: _RowProductChannelAttention,
    Adaptive_Spatial_Attention: _TokenSpatialAttention,
    FastChannelAttention: _UnitLengthChannelAttention,
    IG_MSA: _UnitLengthAttention,
}


def for_export(model: _Descriptor) -> _Descriptor:
    """A copy of an FP16 RetinexFormer or FDAT with its vector lengths and attention product
    in FP32, or an FP16 DAT with its lengths in FP32, its product by rows and fewer passes over
    memory; any other model as it is."""
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
