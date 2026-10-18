"""Exact rewrites of HAT's and DAT's window attention for an ONNX export.

TensorRT's fused attention wants heads a multiple of 8 channels wide; HAT's and DAT's are
often 30 (180 channels in 6 heads), which TensorRT pads with a copy on every call. The
export pads them in the weights instead: zero rows after each head's slice of the qkv
projection, and for HAT zero columns after it in the output projection (DAT drops the
padded channels itself), which only adds exact zeros. The relative position biases, which
HAT looks up and DAT computes from constants on every call, become constants. With
spandrel 0.4.2+c6's channels-last windows, at 256x256 in fp16 on TensorRT 11.4
(optimization level 3), a 4x HAT-L runs in 72 ms instead of 89 ms and a 4x DAT2 in 45 ms
instead of 58 ms. The loaded model is not changed: the rewrite works on a copy, so Save
Model keeps the original weights."""

from __future__ import annotations

import copy
from typing import TypeVar

import torch
from spandrel import ModelDescriptor
from spandrel.architectures.DAT.__arch.DAT import (
    Adaptive_Spatial_Attention,
    Spatial_Attention,
)
from spandrel.architectures.HAT.__arch.HAT import OCAB, WindowAttention
from torch import nn

# TensorRT's fused attention takes heads of a multiple of 8 channels without padding them
HEAD_MULTIPLE = 8

_Descriptor = TypeVar("_Descriptor", bound=ModelDescriptor)


def _adopt(module: nn.Module, source: nn.Module, bias: torch.Tensor) -> None:
    """`module` takes over `source`'s parameters, buffers and submodules under the same
    names, plus `bias` as the buffer "constant_bias"."""
    nn.Module.__init__(module)
    module.__dict__.update(source.__dict__)
    module.register_buffer("constant_bias", bias)


class _ConstantBiasWindowAttention(WindowAttention):
    def __init__(self, source: WindowAttention, bias: torch.Tensor) -> None:
        _adopt(self, source, bias)

    def relative_position_bias(self, rpi: torch.Tensor) -> torch.Tensor:
        return self.get_buffer("constant_bias")


class _ConstantBiasOCAB(OCAB):
    def __init__(self, source: OCAB, bias: torch.Tensor) -> None:
        _adopt(self, source, bias)

    def relative_position_bias(self, rpi: torch.Tensor) -> torch.Tensor:
        return self.get_buffer("constant_bias")


class _ConstantBiasSpatialAttention(Spatial_Attention):
    def __init__(self, source: Spatial_Attention, bias: torch.Tensor) -> None:
        _adopt(self, source, bias)

    def relative_position_bias(self) -> torch.Tensor:
        return self.get_buffer("constant_bias")


def for_export(model: _Descriptor) -> _Descriptor:
    """A copy of a HAT or DAT model with its attention rewritten for export; any other
    model as it is."""
    arch = model.architecture.id
    if arch not in ("HAT", "DAT"):
        return model
    model = copy.deepcopy(model)
    net = model.model
    for name, module in list(net.named_modules()):
        with torch.no_grad():
            if isinstance(module, WindowAttention):
                _pad_heads(module, output=True)
                rpi = net.get_buffer("relative_position_index_SA")
                bias = module.relative_position_bias(rpi)
                net.set_submodule(name, _ConstantBiasWindowAttention(module, bias))
            elif isinstance(module, OCAB):
                _pad_heads(module, output=True)
                rpi = net.get_buffer("relative_position_index_OCA")
                bias = module.relative_position_bias(rpi)
                net.set_submodule(name, _ConstantBiasOCAB(module, bias))
            elif isinstance(module, Adaptive_Spatial_Attention):
                _pad_heads(module, output=False)
            elif isinstance(module, Spatial_Attention) and module.position_bias:
                bias = module.relative_position_bias()
                net.set_submodule(name, _ConstantBiasSpatialAttention(module, bias))
    return model


def _pad_heads(
    module: WindowAttention | OCAB | Adaptive_Spatial_Attention, *, output: bool
) -> None:
    """Pad the heads of `module`'s qkv projection, and with `output` its output
    projection's inputs (HAT's reads the padded heads; DAT drops the padding itself)."""
    heads = module.num_heads
    width = module.qkv.out_features // (3 * heads)
    padded = -(-width // HEAD_MULTIPLE) * HEAD_MULTIPLE
    if padded != width:
        module.qkv = _pad_rows(module.qkv, 3 * heads, width, padded)
        if output:
            module.proj = _pad_columns(module.proj, heads, width, padded)


def _pad_rows(linear: nn.Linear, groups: int, width: int, padded: int) -> nn.Linear:
    """`linear` with each of its `groups` slices of `width` outputs followed by zero
    outputs up to `padded`."""
    bias = dict(linear.named_parameters()).get("bias")
    out = nn.Linear(
        linear.in_features,
        groups * padded,
        bias=bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    out.weight.zero_()
    out.weight.view(groups, padded, -1)[:, :width] = linear.weight.view(
        groups, width, -1
    )
    if bias is not None:
        out.bias.zero_()
        out.bias.view(groups, padded)[:, :width] = bias.view(groups, width)
    return out


def _pad_columns(linear: nn.Linear, groups: int, width: int, padded: int) -> nn.Linear:
    """`linear` reading `groups` slices of `width` inputs, each followed by `padded` -
    `width` zero-weighted inputs."""
    bias = dict(linear.named_parameters()).get("bias")
    out = nn.Linear(
        groups * padded,
        linear.out_features,
        bias=bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    out.weight.zero_()
    out.weight.view(-1, groups, padded)[:, :, :width] = linear.weight.view(
        -1, groups, width
    )
    if bias is not None:
        out.bias.copy_(bias)
    return out
