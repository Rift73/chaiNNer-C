# Modified for chaiNNer-C: standalone torch modules; see NOTICE.md.
# Vendored from VOSR's models/lightningdit.py (by Jingfeng Yao, HUST-VL;
# built on facebookresearch/DiT and willisma/SiT).
#
# Trimmed for VOSR2 v1:
# - timm's PatchEmbed/Mlp are replaced by the small local equivalents below
#   (VOSR2 targets zero new dependencies and timm is not otherwise required).
# - torch.compile decorators on module methods are dropped; ComfyUI runs on
#   accelerators/platforms (e.g. Windows without Triton) where compiling
#   arbitrary custom-node code is not a safe default.
# - Only forward_flexible's code path is kept; the fixed-size forward(),
#   sin-cos position-embedding helpers, and pos-embed interpolation were
#   dead code for this one-step model (LightningDiT is RoPE-conditioned and
#   never used its optional absolute pos_embed) and are removed.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from .pos_embed import RopeTable, VisionRotaryEmbeddingFast
from .rmsnorm import RMSNorm
from .swiglu_ffn import SwiGLUFFN



class PatchEmbed(nn.Module):
    """Conv2d patchify, (N, C, H, W) -> (N, T, D). Accepts any H, W divisible by patch_size."""

    def __init__(self, img_size, patch_size, in_chans, embed_dim, bias=True):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    @property
    def num_patches(self):
        return (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, qk_norm=False, fused_attn: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.fused_attn = fused_attn

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def keys(self, cond):
        """The normalized keys and the values of `cond`, for `keyed`."""
        B_cond, N_cond, _ = cond.shape
        k = self.k_linear(cond).view(B_cond, N_cond, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_linear(cond).view(B_cond, N_cond, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return self.k_norm(k), v

    def keyed(self, x, k, v):
        B, N, C = x.shape
        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        x = F.scaled_dot_product_attention(self.q_norm(q), k, v)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return self.proj(x)

    def forward(self, x, cond):
        return self.keyed(x, *self.keys(cond))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, use_rmsnorm=False):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope=None, query_chunk=None, check_cancel=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        if query_chunk is None or query_chunk >= N:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            # All queries retain the full-image key/value context.
            x = torch.empty_like(q)
            for start in range(0, N, query_chunk):
                if check_cancel is not None:
                    check_cancel()
                x[:, :, start:start + query_chunk] = F.scaled_dot_product_attention(
                    q[:, :, start:start + query_chunk], k, v
                )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class LightningDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False,
        use_rmsnorm=False,
        z_dims=None,
    ):
        super().__init__()
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=use_qknorm, use_rmsnorm=use_rmsnorm)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(hidden_size, mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"))

        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        self.z_dims = z_dims
        if self.z_dims is not None:
            self.cross_attn = MultiHeadCrossAttention(d_model=hidden_size, num_heads=num_heads, qk_norm=use_qknorm)

    def modulation(self, c):
        """The block's shifts, gains (1 + scale) and gates: they follow from the time alone."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + c.reshape(c.shape[0], 6, -1)
        ).chunk(6, dim=1)
        return shift_msa, 1 + scale_msa, gate_msa, shift_mlp, 1 + scale_mlp, gate_mlp

    def forward(self, x, modulation, keys=None, feat_rope=None, token_chunk=None, query_chunk=None, check_cancel=None):
        N = x.shape[1]
        shift_msa, gain_msa, gate_msa, shift_mlp, gain_mlp, gate_mlp = modulation

        x = x + gate_msa * self.attn(
            self.norm1(x) * gain_msa + shift_msa, rope=feat_rope,
            query_chunk=query_chunk, check_cancel=check_cancel,
        )
        if token_chunk is not None:
            # Cross attention, normalization and the MLP are independent per
            # query/token; only self attention above needs all image tokens.
            result = torch.empty_like(x)
            for start in range(0, N, token_chunk):
                if check_cancel is not None:
                    check_cancel()
                part = x[:, start:start + token_chunk]
                if self.z_dims is not None:
                    part = part + self.cross_attn.keyed(part, *keys)
                result[:, start:start + token_chunk] = part + gate_mlp * self.mlp(
                    self.norm2(part) * gain_mlp + shift_mlp
                )
            return result
        if self.z_dims is not None:
            x = x + self.cross_attn.keyed(x, *keys)
        x = x + gate_mlp * self.mlp(self.norm2(x) * gain_mlp + shift_mlp)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size) if use_rmsnorm else nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def modulation(self, c):
        """The final shift and gain (1 + scale): they follow from the time alone."""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return shift.unsqueeze(1), 1 + scale.unsqueeze(1)

    def forward(self, x, modulation):
        shift, gain = modulation
        return self.linear(self.norm_final(x) * gain + shift)


class LightningDiT(nn.Module):
    """One-step VOSR2 flow-matching backbone. Only forward_flexible is used at inference."""

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=32,
        out_channels=16,
        hidden_size=1536,
        depth=36,
        num_heads=24,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
        z_dims=None,
        encdim_ratio=2,
        num_fused_layers=1,
        auxiliary_time_cond=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len)
        else:
            self.feat_rope = None

        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, use_qknorm=use_qknorm,
                use_swiglu=use_swiglu, use_rmsnorm=use_rmsnorm, z_dims=z_dims,
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels, use_rmsnorm=use_rmsnorm)

        self.z_dims = z_dims
        if self.z_dims is not None:
            self.num_fused_layers = num_fused_layers
            self.layer_norm = nn.LayerNorm(z_dims)
            self.mlp_ca = Mlp(z_dims, hidden_size * encdim_ratio, out_features=hidden_size, act_layer=lambda: nn.GELU(approximate="tanh"))

        self.auxiliary_time_cond = auxiliary_time_cond
        self.r_embedder = TimestepEmbedder(hidden_size) if auxiliary_time_cond else None

    def unpatchify(self, x, grid=None):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h, w = grid if grid is not None else (int(x.shape[1] ** 0.5),) * 2
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def _get_dynamic_rope(self, hw_seq_len, device, dtype):
        if not self.use_rope:
            return None

        half_head_dim = self.hidden_size // self.num_heads // 2
        pt_seq_len = self.x_embedder.img_size[0] // self.patch_size

        from einops import repeat

        from .pos_embed import broadcat

        theta = 10000
        freqs = 1.0 / (theta ** (torch.arange(0, half_head_dim, 2, device=device)[: (half_head_dim // 2)].float() / half_head_dim))
        height, width = hw_seq_len if isinstance(hw_seq_len, tuple) else (hw_seq_len,) * 2
        # Extend upstream's square interpolation independently along each axis.
        # This matches the whole-image DINO conditioning resized to 448 x 448.
        y = torch.arange(height, device=device).float() / height * pt_seq_len
        x = torch.arange(width, device=device).float() / width * pt_seq_len
        fy = repeat(torch.einsum("..., f -> ... f", y, freqs), "... n -> ... (n r)", r=2)
        fx = repeat(torch.einsum("..., f -> ... f", x, freqs), "... n -> ... (n r)", r=2)
        freqs = broadcat((fy[:, None, :], fx[None, :, :]), dim=-1)
        return RopeTable(
            freqs.cos().view(-1, freqs.shape[-1]).to(dtype),
            freqs.sin().view(-1, freqs.shape[-1]).to(dtype),
        )

    def rope(self, grid, device, dtype):
        """The RoPE of a token grid; None without RoPE."""
        if not self.use_rope:
            return None
        train_hw_seq_len = self.x_embedder.img_size[0] // self.patch_size
        if grid != (train_hw_seq_len, train_hw_seq_len):
            return self._get_dynamic_rope(grid, device, dtype)
        return RopeTable(self.feat_rope.freqs_cos.to(dtype), self.feat_rope.freqs_sin.to(dtype))

    def time_condition(self, t, r=None):
        """What the blocks and the final layer take from the time: (c, c0)."""
        t_raw = t
        t = self.t_embedder(t)
        r = self.r_embedder(r) * (t_raw - r).unsqueeze(-1) if self.r_embedder is not None else 0
        c = t + r
        return c, self.t_block(c)

    def condition_keys(self, z):
        """Each block's cross-attention keys and values for the conditioning `z`."""
        z = self.mlp_ca(self.layer_norm(z))
        return [block.cross_attn.keys(z) for block in self.blocks]

    def forward_prepared(self, x, keys, feat_rope, modulations, final, token_chunk=None, query_chunk=None):
        """One whole prediction from a prepared time (`modulations`, `final`), conditioning
        (`keys`) and RoPE: a fixed sequence of operations with the bits of forward_flexible
        with these chunks, fit to capture as a CUDA graph."""
        grid = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)
        x = self.x_embedder(x)
        for block, modulation, pair in zip(self.blocks, modulations, keys):
            x = block(x, modulation, pair, feat_rope, token_chunk, query_chunk)
        return self.unpatchify(self._final(x, final, token_chunk), grid)

    def _final(self, x, final, token_chunk):
        if token_chunk is None:
            return self.final_layer(x, final)
        return torch.cat([self.final_layer(part, final) for part in x.split(token_chunk, dim=1)], dim=1)

    def forward_flexible(self, x, t, r=None, z=None, check_cancel=None,
                         token_chunk=None, query_chunk=None, stream_blocks=False, release_block=None):
        """Full-image inference, optionally streaming weights and intermediates.
        `release_block` takes a streamed block off the device (default: copied back)."""
        N, C, H, W = x.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError("DiT dimensions must be divisible by its patch size.")
        if (token_chunk is not None and token_chunk < 1) or (query_chunk is not None and query_chunk < 1):
            raise ValueError("DiT chunk sizes must be positive.")

        grid = (H // self.patch_size, W // self.patch_size)

        x = self.x_embedder(x)
        c, c0 = self.time_condition(t, r)

        if self.z_dims is not None and z is not None:
            z = self.mlp_ca(self.layer_norm(z[0]))

        feat_rope = self.rope(grid, x.device, x.dtype)

        for block in self.blocks:
            if check_cancel is not None:
                check_cancel()
            try:
                if stream_blocks:
                    block.to(x.device)
                keys = block.cross_attn.keys(z) if self.z_dims is not None else None
                x = block(x, block.modulation(c0), keys, feat_rope, token_chunk, query_chunk, check_cancel)
            finally:
                if stream_blocks:
                    if release_block is None:
                        block.cpu()
                    else:
                        release_block(block)

        return self.unpatchify(self._final(x, self.final_layer.modulation(c), token_chunk), grid)
