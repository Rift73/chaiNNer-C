"""An exact trunk of a multiple of 8 channels for HAT's FP16 ONNX export.

TensorRT's FP16 convolutions and matrix products work on widths in multiples of 8. HAT's trunk
is usually 180 channels wide, so around each of them TensorRT copies 180 channels out to 184
and drops 4 afterwards. The export pads them in the weights instead: zero outputs after each
layer's channels and zero weights on the extra inputs, so the 4 channels hold exact zeros from
the first convolution to the last. Only LayerNorm reads them, and it takes its statistics over
the channels that are real. A trained 4x HAT-L in FP16, with PatchEmbed's
permute, paired: Fixed 1200x864 1871.7 -> 1559.3 ms, Dynamic 1200x864 about 1812 -> 1607 ms, an
automatic engine's Dynamic 64-512 at 512x512 about 441 -> 388 ms and at 256x256 78.6 -> 63.1 ms,
with crops the same to 0.2 dB against strict FP32. The loaded model is not changed: the rewrite works on a copy.

The same copy serves eager PyTorch inference (for_eager), whose LayerNorms run spandrel's
PaddedLayerNorm kernel instead: at a 1024x1024 input's 1,048,576 tokens, 184-wide linear layers
took 0.92-0.97 ms against 1.14-1.16 ms at 180 on an RTX 5090, while _PaddedLayerNorm's
separate steps took 7.21 ms against F.layer_norm's 1.06 ms.

HEART's trunk is 180 channels wide too, and cuDNN pads every 180-channel channels-last
convolution input to a multiple of 8: its eager inference runs the same copy, with spandrel's
PaddedILN, whose kernels take the image-wide statistics over the real channels, for its i-LN,
and its affine norms padded with zeros. It is not exported this way."""

from __future__ import annotations

import copy
import importlib
import weakref
from collections.abc import Callable
from typing import Any, TypeVar, cast

import torch
from spandrel import ModelDescriptor
from spandrel.architectures.__arch_helpers.inference import Derived
from torch import nn
from torch.nn import functional

_Descriptor = TypeVar("_Descriptor", bound=ModelDescriptor)

_WIDTH_MULTIPLE = 8

# A LayerNorm's replacement: (source, real width, padded width) -> module.
_Norm = Callable[[Any, int, int], nn.Module]


def _padded(width: int) -> int:
    return width + -width % _WIDTH_MULTIPLE


def _grow(layer: Any, inputs: int | None = None, outputs: int | None = None) -> None:
    """`layer` in place with zero weights on more inputs and zero outputs after its own."""
    weight = layer.weight
    extra_in = 0 if inputs is None else inputs - weight.shape[1]
    extra_out = 0 if outputs is None else outputs - weight.shape[0]
    pads = (0, 0) * (weight.dim() - 2) + (0, extra_in, 0, extra_out)
    layer.weight = nn.Parameter(functional.pad(weight, pads), requires_grad=False)
    if layer.bias is not None:
        layer.bias = nn.Parameter(
            functional.pad(layer.bias, (0, extra_out)), requires_grad=False
        )
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
        self.register_buffer(
            "gain", functional.pad(source.weight.detach(), extra).float()
        )
        self.register_buffer(
            "offset", functional.pad(source.bias.detach(), extra).float()
        )
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


def _grow_affine(affine: Any, padded: int) -> None:
    """An elementwise weight and bias (HEART's affine norms) with zeros on the padding."""
    extra = (0, padded - affine.weight.shape[0])
    affine.weight = nn.Parameter(
        functional.pad(affine.weight, extra), requires_grad=False
    )
    affine.bias = nn.Parameter(functional.pad(affine.bias, extra), requires_grad=False)


def _grow_cab(cab: Any, padded: int) -> None:
    """A CAB: its convolutions' middle also a multiple of 8."""
    conv_0, _, conv_1, channel = cab.cab
    middle = _padded(conv_0.out_channels)
    _grow(conv_0, inputs=padded, outputs=middle)
    _grow(conv_1, inputs=middle, outputs=padded)
    _grow(channel.attention[1], inputs=padded)
    _grow(channel.attention[3], outputs=padded)


def _grow_block(block: Any, width: int, padded: int, norm: _Norm) -> None:
    """A HAB or an OCAB: what reads and writes the trunk."""
    for name in ("norm1", "norm2"):
        block.set_submodule(name, norm(block.get_submodule(name), width, padded))
    attention = block.attn if hasattr(block, "attn") else block
    _grow(attention.qkv, inputs=padded)
    _grow(attention.proj, outputs=padded)
    _grow_mlp(block.mlp, padded)
    if hasattr(block, "conv_block"):
        _grow_cab(block.conv_block, padded)


def _grow_heart_block(block: Any, width: int, padded: int, norm: _Norm) -> None:
    """A HEART block: its i-LNs, its attention when it has one, its CAB and its MLP."""
    for name in ("norm1", "norm2"):
        block.set_submodule(name, norm(block.get_submodule(name), width, padded))
    if block.use_attention:
        _grow(block.attn.to_qkv, inputs=padded)
        _grow(block.attn.to_out, outputs=padded)
    _grow_cab(block.conv_block, padded)
    _grow_mlp(block.mlp, padded)


def _padded_width(model: ModelDescriptor) -> int | None:
    """The padded trunk width of a HAT or a HEART that needs one, else None."""
    if model.architecture.id not in ("HAT", "HEART"):
        return None
    net: Any = model.model
    padded = _padded(net.embed_dim)
    if padded == net.embed_dim or net.ape:
        return None
    return padded


def _padded_copy(model: _Descriptor, padded: int, norm: _Norm) -> _Descriptor:
    """A copy of a HAT or a HEART with its trunk `padded` channels wide, each LayerNorm or
    i-LN made by `norm`."""
    net: Any = model.model
    width: int = net.embed_dim
    heart = model.architecture.id == "HEART"
    model = copy.deepcopy(model)
    net = model.model
    _grow(net.conv_first, outputs=padded)
    if net.patch_embed.norm is not None:
        if heart:
            _grow_affine(net.patch_embed.norm, padded)
        else:
            net.patch_embed.norm = norm(net.patch_embed.norm, width, padded)
    net.patch_unembed.embed_dim = padded
    for group in net.layers:
        group.patch_unembed.embed_dim = padded
        _grow_closing(group.conv, padded)
        for block in group.residual_group.blocks:
            if heart:
                _grow_heart_block(block, width, padded, norm)
            else:
                _grow_block(block, width, padded, norm)
        if not heart:
            _grow_block(group.residual_group.overlap_attn, width, padded, norm)
    if heart:
        _grow_affine(net.norm, padded)
    else:
        net.norm = norm(net.norm, width, padded)
    _grow_closing(net.conv_after_body, padded)
    _grow(net.conv_before_upsample[0], inputs=padded)
    for module in net.modules():
        if isinstance(module, Derived):
            module.derived.clear()
    # The new norms are built in training mode: the copy keeps the source's mode.
    net.train(model.model.training)
    return model


def for_export(model: _Descriptor) -> _Descriptor:
    """A copy of an FP16 HAT with its trunk a multiple of 8 channels wide; any other model
    as it is."""
    padded = _padded_width(model)
    if padded is None or model.architecture.id != "HAT" or model.dtype != torch.float16:
        return model
    return _padded_copy(model, padded, _PaddedLayerNorm)


def _eager_norm(arch: str) -> _Norm | None:
    """spandrel's PaddedLayerNorm (HAT) or PaddedILN (HEART) when its kernel can run in
    this process, else None (an older spandrel without it, no Triton, or a kernel that
    failed)."""
    # Imported by name: a spandrel built before these does not have them.
    try:
        if arch == "HEART":
            iln: Any = importlib.import_module(
                "spandrel.architectures.HEART.__arch.iln_triton"
            )
            if not iln.ILN_KERNEL:
                return None
            return lambda source, _width, padded: iln.PaddedILN(source, padded)
        padded_norm: Any = importlib.import_module(
            "spandrel.architectures.__arch_helpers.padded_norm"
        )
    except ImportError:
        return None
    if not padded_norm.PADDED_NORM:
        return None
    return lambda source, _width, padded: padded_norm.PaddedLayerNorm(source, padded)


_EAGER_COPIES: weakref.WeakKeyDictionary[ModelDescriptor, ModelDescriptor] = (
    weakref.WeakKeyDictionary()
)


def for_eager(
    model: _Descriptor, device: torch.device, use_fp16: bool, plain: bool
) -> _Descriptor:
    """The model eager PyTorch inference runs: for a HAT or a HEART in FP16 on an NVIDIA
    GPU where spandrel's PaddedLayerNorm or PaddedILN kernel can run, a copy with its
    trunk a multiple of 8 channels wide, made once per model; any other model, device or precision, the model
    itself. The model, its weights and its exports are not changed.

    `plain`: the forwards run as is, not under the activation offload's dispatch mode
    (offload.runs_plain), which keeps spandrel's kernels off; there the padded copy
    would run its PyTorch path and was slower than the model (a 4x HAT-L's first
    1448x1448 image on an RTX 5090: 7.5 s against 7.2 s), so the model runs."""
    padded = _padded_width(model)
    if (
        padded is None
        or not plain
        or not use_fp16
        or device.type != "cuda"
        or torch.version.hip is not None
    ):
        return model
    norm = _eager_norm(model.architecture.id)
    if norm is None:
        return model
    copied = _EAGER_COPIES.get(model)
    if copied is None:
        copied = _padded_copy(model, padded, norm)
        _EAGER_COPIES[model] = copied
    return cast(_Descriptor, copied)
