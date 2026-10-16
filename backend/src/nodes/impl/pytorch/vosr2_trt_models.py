"""BF16 VOSR engine graphs with measured FP32 normalization and residual islands.

The one-step checkpoint uses time=1. Its small modulation constants are evaluated
once in FP32; large linear/conv weights and activations remain BF16. VAE queries
are bounded outside the attention graph and always retain global keys/values.
"""

from __future__ import annotations

import copy
from typing import TypeVar, cast

import torch
from torch import nn
from torch.nn import functional

from vendor.vosr2.dinov2 import Block, DinoVisionTransformer
from vendor.vosr2.lightningdit import LightningDiT, LightningDiTBlock
from vendor.vosr2.pos_embed import rotate_half
from vendor.vosr2.qwenimage_vae2d import (
    AttentionBlock2D,
    AutoencoderKLQwenImage2D,
    RMSNorm2D,
)

from .vosr2 import VosrBundle

Model = TypeVar("Model", bound=nn.Module)


class StableDino(nn.Module):
    position: torch.Tensor

    def __init__(self, original: DinoVisionTransformer):
        super().__init__()
        self.model = mixed(original, torch.bfloat16)
        # Conditioning is sensitive to patch/attention rounding. Its larger
        # feed-forward layers stay BF16; preserve attention and the stem in FP32.
        self.model.patch_embed.float().load_state_dict(
            original.patch_embed.state_dict()
        )
        self.position = nn.Buffer(
            original.interpolate_pos_encoding(torch.zeros(1, 1025, 1024), 448, 448)
        )
        self.model.cls_token = nn.Parameter(
            original.cls_token.float().clone(), requires_grad=False
        )
        for block, source in zip(
            cast(list[Block], list(self.model.blocks)),
            cast(list[Block], list(original.blocks)),
            strict=True,
        ):
            block.ls1.float().load_state_dict(source.ls1.state_dict())
            block.ls2.float().load_state_dict(source.ls2.state_dict())
            block.set_submodule("attn", copy.deepcopy(source.attn).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        x = torch.cat((m.cls_token, m.patch_embed(x).float()), dim=1) + self.position
        for block in cast(list[Block], list(m.blocks[:18])):
            x = x + block.ls1(block.attn(block.norm1(x)).float())
            x = x + block.ls2(block.mlp(block.norm2(x).bfloat16()).float())
        return x[:, 1:].bfloat16()


class StableDit(nn.Module):
    modulation: torch.Tensor
    final_shift: torch.Tensor
    final_scale: torch.Tensor

    def __init__(self, original: LightningDiT):
        super().__init__()
        self.dit = copy.deepcopy(original).bfloat16()
        with torch.inference_mode():
            time = original.t_embedder(torch.ones(1))
            modulation = original.t_block(time)
            shift, scale = original.final_layer.adaLN_modulation(time).chunk(2, dim=1)
        self.modulation = nn.Buffer(modulation)
        self.final_shift = nn.Buffer(shift[:, None])
        self.final_scale = nn.Buffer(scale[:, None])
        # Keep small normalization/gating parameters and residuals in FP32;
        # all large convolutions and GEMMs still execute in BF16.
        for block, source in zip(
            cast(list[LightningDiTBlock], list(self.dit.blocks)),
            cast(list[LightningDiTBlock], list(original.blocks)),
            strict=True,
        ):
            block.norm1.float().load_state_dict(source.norm1.state_dict())
            block.norm2.float().load_state_dict(source.norm2.state_dict())
            block.scale_shift_table = nn.Parameter(
                source.scale_shift_table.float().clone(), requires_grad=False
            )
        self.dit.final_layer.norm_final.float().load_state_dict(
            original.final_layer.norm_final.state_dict()
        )
        self.dit.layer_norm.float().load_state_dict(original.layer_norm.state_dict())

    def forward(
        self,
        latent: torch.Tensor,
        conditioning: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict(latent, conditioning, cos, sin)

    def predict(
        self,
        latent: torch.Tensor,
        conditioning: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = self.dit
        x = m.x_embedder(latent).float()
        condition = (
            m.mlp_ca(m.layer_norm(conditioning.float()).bfloat16())
            if conditioning is not None
            else None
        )

        def rope(value: torch.Tensor) -> torch.Tensor:
            return value * cos + rotate_half(value) * sin

        for index, b in enumerate(cast(list[LightningDiTBlock], list(m.blocks))):
            shift, scale, gate, shift_m, scale_m, gate_m = (
                b.scale_shift_table[None] + self.modulation.reshape(1, 6, -1)
            ).chunk(6, dim=1)
            x = (
                x
                + gate
                * b.attn(
                    (b.norm1(x) * (1 + scale) + shift).bfloat16(), rope=rope
                ).float()
            )
            if kv is None:
                x = x + b.cross_attn(x.bfloat16(), condition).float()
            else:
                cross = b.cross_attn
                batch, tokens, channels = x.shape
                heads, depth = cross.num_heads, cross.head_dim
                q = cross.q_linear(x.bfloat16()).reshape(batch, tokens, heads, depth)
                k = kv[index * 2].reshape(batch, -1, heads, depth)
                v = kv[index * 2 + 1].reshape(batch, -1, heads, depth)
                q, k = (
                    cross.q_norm(q.permute(0, 2, 1, 3)),
                    cross.k_norm(k.permute(0, 2, 1, 3)),
                )
                attention = functional.scaled_dot_product_attention(
                    q, k, v.permute(0, 2, 1, 3)
                )
                x = (
                    x
                    + cross.proj(
                        attention.permute(0, 2, 1, 3)
                        .contiguous()
                        .reshape(batch, tokens, channels)
                    ).float()
                )
            x = (
                x
                + gate_m
                * b.mlp((b.norm2(x) * (1 + scale_m) + shift_m).bfloat16()).float()
            )
        x = m.final_layer.linear(
            (
                m.final_layer.norm_final(x) * (1 + self.final_scale) + self.final_shift
            ).bfloat16()
        )
        return m.unpatchify(x, (32, latent.shape[-1] // 2))


class DiTKv(nn.Module):
    def __init__(self, original: LightningDiT):
        super().__init__()
        self.model = StableDit(original)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        m = self.model.dit
        condition = m.mlp_ca(m.layer_norm(conditioning.float()).bfloat16())
        values = []
        for b in cast(list[LightningDiTBlock], list(m.blocks)):
            values.extend(
                (b.cross_attn.k_linear(condition), b.cross_attn.v_linear(condition))
            )
        return torch.stack(values)


class CachedDiT(nn.Module):
    def __init__(self, original: LightningDiT):
        super().__init__()
        self.model = StableDit(original)

    def forward(
        self,
        latent: torch.Tensor,
        kv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.predict(latent, None, cos, sin, kv)


class Norm32(nn.Module):
    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = copy.deepcopy(original).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x.float()).to(x.dtype)


def mixed(model: Model, dtype: torch.dtype) -> Model:
    result = copy.deepcopy(model)
    # Preserve the original norm parameters before reducing the other weights.
    norms = [
        (name, Norm32(module))
        for name, module in result.named_modules()
        if isinstance(module, (RMSNorm2D, nn.LayerNorm))
    ]
    result.to(dtype=dtype)
    for name, replacement in norms:
        result.set_submodule(name, replacement)
    return result.eval()


class EncoderPre(nn.Module):
    def __init__(self, vae: AutoencoderKLQwenImage2D):
        super().__init__()
        self.conv = vae.encoder.conv_in
        self.down = vae.encoder.down_blocks
        self.mid = vae.encoder.mid_block.resnets[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        for layer in self.down:
            x = layer(x)
        return self.mid(x)


class AttentionProjection(nn.Module):
    def __init__(self, attention: AttentionBlock2D):
        super().__init__()
        self.norm = attention.norm
        self.qkv = attention.to_qkv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return (
            self.qkv(self.norm(x))
            .reshape(b, 3, c, h * w)
            .permute(0, 1, 3, 2)
            .contiguous()
        )


class AttentionCore(nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        # Global keys/values, bounded queries; never independent image tiles.
        scores = (q @ k.transpose(-1, -2)).float() * (384**-0.5)
        return scores.softmax(dim=-1).to(q.dtype) @ v


class EncoderPost(nn.Module):
    def __init__(self, vae: AutoencoderKLQwenImage2D):
        super().__init__()
        self.proj = cast(AttentionBlock2D, vae.encoder.mid_block.attentions[0]).proj
        self.mid = vae.encoder.mid_block.resnets[1]
        self.norm = vae.encoder.norm_out
        self.conv = vae.encoder.conv_out
        self.quant = vae.quant_conv

    def forward(self, x: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        x = self.mid(self.proj(attention) + x)
        return self.quant(self.conv(functional.silu(self.norm(x))))[:, :16].contiguous()


class DecoderPre(nn.Module):
    def __init__(self, vae: AutoencoderKLQwenImage2D):
        super().__init__()
        self.post_quant = vae.post_quant_conv
        self.conv = vae.decoder.conv_in
        self.mid = vae.decoder.mid_block.resnets[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mid(self.conv(self.post_quant(x)))


class DecoderPost(nn.Module):
    def __init__(self, vae: AutoencoderKLQwenImage2D):
        super().__init__()
        self.proj = cast(AttentionBlock2D, vae.decoder.mid_block.attentions[0]).proj
        self.mid = vae.decoder.mid_block.resnets[1]
        self.up = vae.decoder.up_blocks
        self.norm = vae.decoder.norm_out
        self.conv = vae.decoder.conv_out

    def forward(self, x: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        x = self.mid(self.proj(attention) + x)
        for layer in self.up:
            x = layer(x)
        return self.conv(functional.silu(self.norm(x))).clamp(-1, 1)


STAGES = (
    "dit_kv",
    "dit_cached",
    "dino",
    "encoder_pre",
    "encoder_qkv",
    "encoder_post",
    "decoder_pre",
    "decoder_qkv",
    "decoder_post",
    "attention",
)


def export_model(name: str, bundle: VosrBundle) -> nn.Module:
    if name == "dit":
        return StableDit(bundle.dit)
    if name == "dit_kv":
        return DiTKv(bundle.dit)
    if name == "dit_cached":
        return CachedDiT(bundle.dit)
    if name == "dino":
        return StableDino(bundle.dino)
    vae = mixed(bundle.vae, torch.bfloat16)
    models = {
        "encoder_pre": EncoderPre(vae),
        "encoder_qkv": AttentionProjection(
            cast(AttentionBlock2D, vae.encoder.mid_block.attentions[0])
        ),
        "encoder_post": EncoderPost(vae),
        "decoder_pre": DecoderPre(vae),
        "decoder_qkv": AttentionProjection(
            cast(AttentionBlock2D, vae.decoder.mid_block.attentions[0])
        ),
        "decoder_post": DecoderPost(vae),
        "attention": AttentionCore(),
    }
    return models[name].eval()


def profiles(name: str) -> dict[str, tuple[tuple[int, ...], ...]]:
    if name == "dit_kv":
        return {"conditioning": ((1, 1024, 1024),) * 3}
    if name == "dit_cached":
        original = profiles("dit")
        return {
            "latent": original["latent"],
            "kv": ((72, 1, 1024, 1536),) * 3,
            "cos": original["cos"],
            "sin": original["sin"],
        }
    if name == "dit":
        return {
            "latent": ((1, 32, 64, 64), (1, 32, 64, 64), (1, 32, 64, 72)),
            "conditioning": ((1, 1024, 1024),) * 3,
            "cos": ((1024, 64), (1024, 64), (1152, 64)),
            "sin": ((1024, 64), (1024, 64), (1152, 64)),
        }
    image = ((1, 3, 512, 512), (1, 3, 1440, 2560), (1, 3, 2176, 3840))
    feature = ((1, 384, 64, 64), (1, 384, 180, 320), (1, 384, 272, 480))
    latent = ((1, 16, 64, 64), (1, 16, 180, 320), (1, 16, 272, 480))
    if name == "dino":
        return {"x": ((1, 3, 448, 448),) * 3}
    if name == "encoder_pre":
        return {"x": image}
    if name == "decoder_pre":
        return {"x": latent}
    if name in ("encoder_post", "decoder_post"):
        return {"x": feature, "attention": feature}
    if name.endswith("_qkv"):
        return {"x": feature}
    if name == "attention":
        return {
            "q": ((1, 1, 1, 384), (1, 1, 256, 384), (1, 1, 1024, 384)),
            "k": ((1, 1, 4096, 384), (1, 1, 57600, 384), (1, 1, 130560, 384)),
            "v": ((1, 1, 4096, 384), (1, 1, 57600, 384), (1, 1, 130560, 384)),
        }
    raise ValueError(name)
