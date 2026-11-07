"""An exact trunk of a multiple of 8 channels for HAT's FP16 ONNX export.

TensorRT's FP16 convolutions and matrix products work on widths in multiples of 8. HAT's trunk
is usually 180 channels wide, so around each of them TensorRT copies 180 channels out to 184
and drops 4 afterwards. The export pads them in the weights instead: zero outputs after each
layer's channels and zero weights on the extra inputs, so the 4 channels hold exact zeros from
the first convolution to the last. Only LayerNorm reads them, and it takes its statistics over
the channels that are real. A trained 4x HAT-L in FP16, with PatchEmbed's
permute, paired: Fixed 1200x864 1871.7 -> 1559.3 ms, Dynamic 1200x864 about 1812 -> 1607 ms, an
automatic engine's Dynamic 64-512 at 512x512 about 441 -> 388 ms and at 256x256 78.6 -> 63.1 ms,
with crops the same to 0.2 dB against strict FP32. The loaded model is not changed: the rewrite works on a copy."""

from __future__ import annotations

import copy
from typing import Any, TypeVar

import torch
from spandrel import ModelDescriptor
from spandrel.architectures.__arch_helpers.inference import Derived
from torch import nn
from torch.nn import functional as F

_Descriptor = TypeVar("_Descriptor", bound=ModelDescriptor)

_WIDTH_MULTIPLE = 8


def _padded(width: int) -> int:
    return width + -width % _WIDTH_MULTIPLE


def _grow(layer: Any, inputs: int | None = None, outputs: int | None = None) -> None:
    """`layer` in place with zero weights on more inputs and zero outputs after its own."""
    weight = layer.weight
    extra_in = 0 if inputs is None else inputs - weight.shape[1]
    extra_out = 0 if outputs is None else outputs - weight.shape[0]
    pads = (0, 0) * (weight.dim() - 2) + (0, extra_in, 0, extra_out)
    layer.weight = nn.Parameter(F.pad(weight, pads), requires_grad=False)
    if layer.bias is not None:
        layer.bias = nn.Parameter(F.pad(layer.bias, (0, extra_out)), requires_grad=False)
    names = ("in_channels", "out_channels")
    if isinstance(layer, nn.Linear):
        names = ("in_features", "out_features")
    setattr(layer, names[0], weight.shape[1] + extra_in)
    setattr(layer, names[1], weight.shape[0] + extra_out)


class _PaddedLayerNorm(nn.Module):
    """A LayerNorm whose last channels hold exact zeros: its mean and variance are over the
    `width` real channels of `padded`, in FP32 as LayerNormalization computes, and its output
    is zero on the rest."""

    def __init__(self, source: Any, width: int, padded: int) -> None:
        super().__init__()
        extra = (0, padded - width)
        self.register_buffer("kind", torch.zeros(0, dtype=source.weight.dtype))
        self.register_buffer("gain", F.pad(source.weight.detach(), extra).float())
        self.register_buffer("offset", F.pad(source.bias.detach(), extra).float())
        self.eps = source.eps
        self.to_real = padded / width
        self.zeros_per_real = (padded - width) / width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide = x.float()
        mean = wide.mean(-1, keepdim=True) * self.to_real
        centered = wide - mean
        variance = centered.square().mean(-1, keepdim=True) * self.to_real
        variance = variance - mean.square() * self.zeros_per_real
        out = centered / (variance + self.eps).sqrt()
        out = out * self.get_buffer("gain") + self.get_buffer("offset")
        return out.to(self.get_buffer("kind").dtype)


def _grow_mlp(mlp: Any, padded: int) -> None:
    hidden = _padded(mlp.fc1.out_features)
    _grow(mlp.fc1, inputs=padded, outputs=hidden)
    _grow(mlp.fc2, inputs=hidden, outputs=padded)


def _grow_closing(conv: nn.Module, padded: int) -> None:
    """A residual group's closing convolution, '1conv' or '3conv', from and to the trunk."""
    if isinstance(conv, nn.Conv2d):
        _grow(conv, inputs=padded, outputs=padded)
    elif isinstance(conv, nn.Sequential):
        layers = [layer for layer in conv if isinstance(layer, nn.Conv2d)]
        _grow(layers[0], inputs=padded)
        _grow(layers[-1], outputs=padded)


def _grow_block(block: Any, width: int, padded: int) -> None:
    """A HAB or an OCAB: what reads and writes the trunk."""
    for name in ("norm1", "norm2"):
        block.set_submodule(
            name, _PaddedLayerNorm(block.get_submodule(name), width, padded)
        )
    attention = block.attn if hasattr(block, "attn") else block
    _grow(attention.qkv, inputs=padded)
    _grow(attention.proj, outputs=padded)
    _grow_mlp(block.mlp, padded)
    if hasattr(block, "conv_block"):
        conv_0, _, conv_1, channel = block.conv_block.cab
        middle = _padded(conv_0.out_channels)
        _grow(conv_0, inputs=padded, outputs=middle)
        _grow(conv_1, inputs=middle, outputs=padded)
        _grow(channel.attention[1], inputs=padded)
        _grow(channel.attention[3], outputs=padded)


def for_export(model: _Descriptor) -> _Descriptor:
    """A copy of an FP16 HAT with its trunk a multiple of 8 channels wide; any other model
    as it is."""
    if model.architecture.id != "HAT" or model.dtype != torch.float16:
        return model
    net: Any = model.model
    width = net.embed_dim
    padded = _padded(width)
    if padded == width or net.ape:
        return model
    model = copy.deepcopy(model)
    net = model.model
    _grow(net.conv_first, outputs=padded)
    if net.patch_embed.norm is not None:
        net.patch_embed.norm = _PaddedLayerNorm(net.patch_embed.norm, width, padded)
    net.patch_unembed.embed_dim = padded
    for group in net.layers:
        group.patch_unembed.embed_dim = padded
        _grow_closing(group.conv, padded)
        for block in group.residual_group.blocks:
            _grow_block(block, width, padded)
        _grow_block(group.residual_group.overlap_attn, width, padded)
    net.norm = _PaddedLayerNorm(net.norm, width, padded)
    _grow_closing(net.conv_after_body, padded)
    _grow(net.conv_before_upsample[0], inputs=padded)
    for module in net.modules():
        if isinstance(module, Derived):
            module.derived.clear()
    return model
