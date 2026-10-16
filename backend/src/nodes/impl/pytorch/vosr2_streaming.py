"""Whole-image VOSR execution with bounded accelerator intermediates.

Convolution bands include their exact receptive-field halo. Only attention
queries are chunked: every query still sees the entire image's keys and values.
There is no independently generated image tile, overlap blend, or seam filter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import nn
from torch.nn import functional

from vendor.vosr2.qwenimage_vae2d import (
    AttentionBlock2D,
    AutoencoderKLQwenImage2D,
    MidBlock2D,
    Resample2D,
    ResidualBlock2D,
    UpBlock2D,
)


class VaeStreamer:
    """Keep full feature maps in host RAM, including residual connections."""

    def __init__(
        self,
        device: torch.device,
        activation_bytes: int,
        check: Callable[[], None],
        map_device: torch.device | None = None,
    ):
        if activation_bytes < 1:
            raise ValueError("VAE activation budget must be positive.")
        self.device = device
        self.activation_bytes = activation_bytes
        self.check = check
        self.map_device = map_device or torch.device("cpu")

    def conv(
        self,
        x: torch.Tensor,
        layer: nn.Conv2d,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        *,
        up: int = 1,
        pad_end: int = 0,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate convolution at the original global stride/padding phase."""
        if x.device != self.map_device or layer.dilation != (1, 1):
            raise ValueError(
                "VAE streaming requires maps on its home device and unit dilation."
            )
        b, channels, height, width = x.shape
        kh, kw = layer.kernel_size
        sy, sx = layer.stride
        if isinstance(layer.padding, str):
            raise ValueError("VAE streaming requires explicit numeric padding.")
        py, px = layer.padding
        vh, vw = height * up, width * up
        oh = (vh + pad_end + 2 * py - kh) // sy + 1
        ow = (vw + pad_end + 2 * px - kw) // sx + 1
        result = torch.empty(
            b, layer.out_channels, oh, ow, dtype=x.dtype, device=self.map_device
        )
        # Several simultaneously live input/norm/output buffers, plus halo.
        bytes_per_row = (
            b * max(channels, layer.out_channels) * max(vw, ow) * x.element_size()
        )
        rows = max(1, self.activation_bytes // max(1, bytes_per_row * 8 * sy))
        for start in range(0, oh, rows):
            self.check()
            end = min(start + rows, oh)
            lo, hi = start * sy - py, (end - 1) * sy - py + kh
            valid_lo, valid_hi = max(0, lo), min(vh, hi)
            input_lo, input_hi = valid_lo // up, (valid_hi + up - 1) // up
            band = x[:, :, input_lo:input_hi].to(self.device)
            if up != 1:
                band = functional.interpolate(band, scale_factor=up, mode="nearest")
            band = band[:, :, valid_lo - input_lo * up : valid_hi - input_lo * up]
            if transform is not None:
                band = transform(band)
            band = functional.pad(
                band, (px, px + pad_end, valid_lo - lo, hi - valid_hi)
            )
            value = functional.conv2d(
                band, layer.weight, layer.bias, layer.stride, groups=layer.groups
            )
            value = value.to(self.map_device)
            if residual is not None:
                value.add_(residual[:, :, start:end])
            result[:, :, start:end] = value
            del band, value
        return result

    def residual(self, x: torch.Tensor, layer: ResidualBlock2D) -> torch.Tensor:
        shortcut = (
            x
            if isinstance(layer.conv_shortcut, nn.Identity)
            else self.conv(x, layer.conv_shortcut)
        )
        value = self.conv(
            x, layer.conv1, lambda band: functional.silu(layer.norm1(band))
        )
        return self.conv(
            value,
            layer.conv2,
            lambda band: functional.silu(layer.norm2(band)),
            residual=shortcut,
        )

    def attention(self, x: torch.Tensor, layer: AttentionBlock2D) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.conv(x, layer.to_qkv, layer.norm).reshape(b, 3, c, h * w)
        q, k, v = qkv.unbind(1)
        k = k.transpose(1, 2).unsqueeze(1).to(self.device).contiguous()
        v = v.transpose(1, 2).unsqueeze(1).to(self.device).contiguous()
        result = torch.empty(b, c, h * w, dtype=x.dtype, device=self.map_device)
        # Also bound the explicit score matrix if fused SDPA is unavailable.
        queries = max(1, min(1024, self.activation_bytes // (b * h * w * 8)))
        for start in range(0, h * w, queries):
            self.check()
            query = (
                q[:, :, start : start + queries]
                .transpose(1, 2)
                .unsqueeze(1)
                .to(self.device)
            )
            value = functional.scaled_dot_product_attention(query, k, v)
            result[:, :, start : start + queries] = (
                value.squeeze(1).transpose(1, 2).to(self.map_device)
            )
            del query, value
        del qkv, q, k, v
        return self.conv(result.reshape(b, c, h, w), layer.proj, residual=x)

    def mid(self, x: torch.Tensor, layer: MidBlock2D) -> torch.Tensor:
        x = self.residual(x, cast(ResidualBlock2D, layer.resnets[0]))
        for attention, residual in zip(layer.attentions, layer.resnets[1:]):
            x = self.residual(
                self.attention(x, cast(AttentionBlock2D, attention)),
                cast(ResidualBlock2D, residual),
            )
        return x

    def resample(self, x: torch.Tensor, layer: Resample2D) -> torch.Tensor:
        if not isinstance(layer.resample, nn.Sequential):
            raise ValueError("Unsupported VOSR VAE resampling layer.")
        first = layer.resample[0]
        convolution = cast(nn.Conv2d, layer.resample[1])
        if isinstance(first, nn.Upsample):
            return self.conv(x, convolution, up=2)
        if isinstance(first, nn.ZeroPad2d):
            return self.conv(x, convolution, pad_end=1)
        raise ValueError("Unsupported VOSR VAE resampling layer.")

    def encode(self, x: torch.Tensor, vae: AutoencoderKLQwenImage2D) -> torch.Tensor:
        encoder = vae.encoder
        x = self.conv(x, encoder.conv_in)
        for layer in encoder.down_blocks:
            if isinstance(layer, ResidualBlock2D):
                x = self.residual(x, layer)
            elif isinstance(layer, Resample2D):
                x = self.resample(x, layer)
            elif isinstance(layer, AttentionBlock2D):
                x = self.attention(x, layer)
            else:
                raise ValueError(f"Unsupported VOSR encoder layer: {type(layer)}")
        x = self.mid(x, encoder.mid_block)
        x = self.conv(
            x, encoder.conv_out, lambda band: functional.silu(encoder.norm_out(band))
        )
        return self.conv(x, vae.quant_conv).chunk(2, dim=1)[0].contiguous()

    def decode(self, x: torch.Tensor, vae: AutoencoderKLQwenImage2D) -> torch.Tensor:
        decoder = vae.decoder
        x = self.conv(self.conv(x, vae.post_quant_conv), decoder.conv_in)
        x = self.mid(x, decoder.mid_block)
        for module in decoder.up_blocks:
            block = cast(UpBlock2D, module)
            for residual in block.resnets:
                x = self.residual(x, cast(ResidualBlock2D, residual))
            if block.upsamplers is not None:
                x = self.resample(x, cast(Resample2D, block.upsamplers[0]))
        return self.conv(
            x, decoder.conv_out, lambda band: functional.silu(decoder.norm_out(band))
        ).clamp_(-1, 1)
