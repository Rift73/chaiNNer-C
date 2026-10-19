"""Legacy DRFT's TensorRT ONNX (drft.py): a deep copy of the model whose modules are
swapped for stand-ins that compute the original maths and export as chaiNNer-C's DRFT
TensorRT plugins (native/src/drft, Codex's LegacyBarrierV1 set; node attributes and
layouts as in Codex's prepared graph), for every input size from MIN_SIZE up. FP16
inside (the plugins take BF16 too: ONNX files exported in BF16 before still build), input
and output FP32, opset 20. FP16, not Codex's BF16: on photos a DRFT-L x4 in BF16 was
25.9-35.7 dB against PyTorch FP32 (PyTorch's own BF16 alike), in FP16 38.7-58.6 dB."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from spandrel import ImageModelDescriptor
from spandrel.architectures.DRFT.__arch import legacy_drft_arch as legacy
from torch import nn
from torch.nn import functional

from . import drft

OPSET = 20
PLUGINS = "legacy-v1"
# The relative-bias plugin's (query, key) OCAB windows; other windows keep the dense
# bias as plain maths.
BIAS_WINDOWS = {(32, 40), (32, 48)}
# A barrier after every third ACT block (Codex: blocks 2 and 5 of each group).
BARRIER_EVERY = 3


def is_legacy_drft(model: ImageModelDescriptor) -> bool:
    """spandrel's legacy DRFT with standard LayerNorm (no i-LN): the plugin route."""
    return isinstance(model.model, legacy.DRFT) and not any(
        isinstance(module, legacy.iLN) for module in model.model.modules()
    )


def min_size(window_size: int) -> int:
    """The smallest input side: two windows, so the shifted blocks' interior windows
    exist (and at least Codex's 64)."""
    return max(64, 2 * window_size)


def _relative_index(q_window: int, k_window: int, device: torch.device) -> torch.Tensor:
    """OCAB's relative-position index (q*q, k*k) into its (q+k-1)^2 table, as spandrel
    builds it."""
    span = q_window + k_window - 1
    q = torch.arange(q_window, device=device)
    k = torch.arange(k_window, device=device)
    q_y, q_x = q.repeat_interleave(q_window), q.repeat(q_window)
    k_y, k_x = k.repeat_interleave(k_window), k.repeat(k_window)
    rows = (k_y[None, :] - q_y[:, None] + q_window - 1) * span
    return rows + (k_x[None, :] - q_x[:, None] + q_window - 1)


def _attention(
    q: torch.Tensor,
    k_t: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Attention (k given transposed) with the scale, bias and softmax in FP32, and
    q @ k^T in BF16 for an FP16 body. A trained DRFT's logits pass FP16's range by far
    (up to 48.6 million on a DRFT-L x4's photos); in FP32 the 512x512 run filled 32 GB,
    and in FP16 on logits divided by 2^16 TensorRT flushed the small ones to zero
    (42.0-52.8 dB). BF16 logits with the rest in FP16: 48.8-60.9 dB against PyTorch FP32
    (in PyTorch).
    Not spandrel's export-safe maths, which clamp the logits to +-50 (the softmax
    subtracts their maximum, so there is no overflow to prevent there, and the clamp
    changed the result wherever a logit passed 50: DRFT-L x4 on photos in FP32, 43-52
    dB against PyTorch's own attention)."""
    if q.dtype == torch.float16:
        q, k_t = q.bfloat16(), k_t.bfloat16()
    attn = torch.matmul(q, k_t).float() * scale
    attn = torch.softmax(attn + bias.float(), dim=-1).to(dtype=v.dtype)
    return torch.matmul(attn, v)


def _plugin(g: Any, op: str, *inputs: Any, **attributes: Any) -> Any:
    return g.op(op, *inputs, plugin_version_s="1", plugin_namespace_s="", **attributes)


class _WindowNeuralBias(torch.autograd.Function):
    """ACT attention's rank-factored bias (hnr,hmr->hnm) from its constant factors."""

    @staticmethod
    def forward(ctx: Any, bias_q: torch.Tensor, bias_k: torch.Tensor) -> torch.Tensor:
        return torch.einsum("hnr,hmr->hnm", bias_q, bias_k)

    @staticmethod
    def symbolic(g: Any, bias_q: Any, bias_k: Any) -> Any:
        # Codex's node carries no attributes.
        return g.op("DrftWindowNeuralBias_TRT", bias_q, bias_k)


class _RelativeBias(torch.autograd.Function):
    """OCAB's dense relative bias (1, heads, q*q, k*k) from its compact table."""

    @staticmethod
    def forward(
        ctx: Any, table: torch.Tensor, q_window: int, k_window: int
    ) -> torch.Tensor:
        index = _relative_index(q_window, k_window, table.device)
        bias = table[index.view(-1)]
        bias = bias.view(q_window**2, k_window**2, -1).permute(2, 0, 1).contiguous()
        return bias.unsqueeze(0)

    @staticmethod
    def symbolic(g: Any, table: Any, q_window: int, k_window: int) -> Any:
        return _plugin(
            g,
            "DrftOCABRelativeBiasSafe_TRT",
            table,
            q_window_size_i=q_window,
            kv_window_size_i=k_window,
        )


class _QKVPack(torch.autograd.Function):
    """OCAB's q windows and overlapping k/v windows from qkv (B, H, W, 3, C), token
    major: (tokens, windows, heads, head_dim) each."""

    @staticmethod
    def forward(
        ctx: Any,
        qkv: torch.Tensor,
        channels: int,
        heads: int,
        q_window: int,
        k_window: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = qkv.shape[0]
        head_dim = channels // heads
        qkv = qkv.permute(3, 0, 4, 1, 2)
        q = qkv[0].permute(0, 2, 3, 1)
        kv = torch.cat((qkv[1], qkv[2]), dim=1)
        q_windows = legacy.window_partition(q, q_window)
        q_windows = q_windows.view(-1, q_window * q_window, channels)
        kv_windows = functional.unfold(
            kv,
            kernel_size=(k_window, k_window),
            stride=q_window,
            padding=(k_window - q_window) // 2,
        )
        windows = kv_windows.shape[2]
        kv_windows = kv_windows.view(batch, 2, channels, k_window, k_window, windows)
        kv_windows = kv_windows.permute(1, 0, 5, 3, 4, 2).contiguous()
        kv_windows = kv_windows.view(2, batch * windows, k_window * k_window, channels)
        k_windows, v_windows = kv_windows.unbind(0)
        b_windows, q_tokens, _ = q_windows.shape
        k_tokens = k_windows.shape[1]
        q = q_windows.reshape(b_windows, q_tokens, heads, head_dim).permute(0, 2, 1, 3)
        k = k_windows.reshape(b_windows, k_tokens, heads, head_dim).permute(0, 2, 1, 3)
        v = v_windows.reshape(b_windows, k_tokens, heads, head_dim).permute(0, 2, 1, 3)
        return q.permute(2, 0, 1, 3), k.permute(2, 0, 1, 3), v.permute(2, 0, 1, 3)

    @staticmethod
    def symbolic(
        g: Any, qkv: Any, channels: int, heads: int, q_window: int, k_window: int
    ) -> Any:
        return _plugin(
            g,
            "DrftOCABQKVPackTokenMajor_TRT",
            qkv,
            channels_i=channels,
            heads_i=heads,
            query_window_size_i=q_window,
            key_window_size_i=k_window,
            outputs=3,
        )


class _FinalAffineCrop(torch.autograd.Function):
    """The output tail for img_range 1: add the mean, crop to the input's size times
    the scale, FP32."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        image: torch.Tensor,
        scale: int,
        mean: tuple[float, float, float],
    ) -> torch.Tensor:
        height, width = image.shape[2:]
        means = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        x = x + means
        return x[:, :, : height * scale, : width * scale].to(dtype=torch.float32)

    @staticmethod
    def symbolic(
        g: Any, x: Any, image: Any, scale: int, mean: tuple[float, float, float]
    ) -> Any:
        return _plugin(
            g,
            "DrftFinalAffineCrop_TRT",
            x,
            image,
            scale_i=scale,
            mean_r_f=mean[0],
            mean_g_f=mean[1],
            mean_b_f=mean[2],
        )


class _IdentityBarrier(torch.autograd.Function):
    """A TensorRT partition boundary; the identity."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    @staticmethod
    def symbolic(g: Any, x: Any) -> Any:
        return _plugin(g, "DrftIdentityBarrier_TRT", x)


def _barrier(x: torch.Tensor) -> torch.Tensor:
    return _IdentityBarrier.apply(x)


class _Barriered(nn.Module):
    """A module whose output passes an identity barrier."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, *args: Any) -> torch.Tensor:
        return _barrier(self.inner(*args))


class _WindowAttention(nn.Module):
    """ACT window attention (WindowAttentionRFB's additive-bias maths) with its
    neural-bias factors as constants."""

    def __init__(self, attn: legacy.WindowAttentionRFB) -> None:
        super().__init__()
        self.qkv = attn.qkv
        self.proj = attn.proj
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.content_scale = attn.content_scale
        attn.neural_bias.clear_cache()
        with torch.no_grad():
            bias_q, bias_k = attn.neural_bias()
        self.register_buffer("bias_q", bias_q.detach().clone(), persistent=False)
        self.register_buffer("bias_k", bias_k.detach().clone(), persistent=False)

    def bias(self) -> torch.Tensor:
        return _WindowNeuralBias.apply(self.bias_q, self.bias_k)

    def forward(
        self, x: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b_windows, n_tokens, c = x.shape
        qkv = self.qkv(x).reshape(b_windows, n_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn_mask = bias.unsqueeze(0)
        if mask is not None:
            attn_mask = attn_mask + mask.to(dtype=attn_mask.dtype)
        attn_mask = attn_mask.to(dtype=q.dtype)
        out = _attention(q, k.transpose(-2, -1), v, attn_mask, self.content_scale)
        out = out.transpose(1, 2).reshape(b_windows, n_tokens, c)
        return self.proj(out)


Masks = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class _ACTBlock(nn.Module):
    """ACTBlock with one neural bias per block and the shifted windows split into
    interior, right column, bottom row and corner (each with its constant region mask),
    so every input size exports (Codex's dynamic route, at the model's window and
    shift)."""

    def __init__(self, block: legacy.ACTBlock, barrier: bool) -> None:
        super().__init__()
        self.norm1 = block.norm1
        self.conv_block = block.conv_block
        self.attn = _WindowAttention(block.attn)
        self.ls_attn = block.ls_attn
        self.ls_conv = block.ls_conv
        self.ls_ffn = block.ls_ffn
        self.norm2 = block.norm2
        self.ffn = block.ffn
        self.window_size = block.window_size
        self.shift_size = block.shift_size
        self.conv_scale = block.conv_scale
        self.barrier = barrier

    def _shifted_attention(
        self,
        windows: torch.Tensor,
        bias: torch.Tensor,
        batch: int,
        height: int,
        width: int,
        masks: Masks,
    ) -> torch.Tensor:
        rows = height // self.window_size
        columns = width // self.window_size
        tokens, channels = windows.shape[1], windows.shape[2]
        grid = windows.reshape(batch, rows, columns, tokens, channels)
        right_mask, bottom_mask, corner_mask = masks
        interior = self.attn(
            grid[:, : rows - 1, : columns - 1].reshape(-1, tokens, channels), bias
        )
        right = self.attn(
            grid[:, : rows - 1, columns - 1 :].reshape(-1, tokens, channels),
            bias,
            right_mask,
        )
        bottom = self.attn(
            grid[:, rows - 1 :, : columns - 1].reshape(-1, tokens, channels),
            bias,
            bottom_mask,
        )
        corner = self.attn(
            grid[:, rows - 1 :, columns - 1 :].reshape(-1, tokens, channels),
            bias,
            corner_mask,
        )
        top = torch.cat(
            (
                interior.reshape(batch, rows - 1, columns - 1, tokens, channels),
                right.reshape(batch, rows - 1, 1, tokens, channels),
            ),
            dim=2,
        )
        last = torch.cat(
            (
                bottom.reshape(batch, 1, columns - 1, tokens, channels),
                corner.reshape(batch, 1, 1, tokens, channels),
            ),
            dim=2,
        )
        return torch.cat((top, last), dim=1).reshape(-1, tokens, channels)

    def forward(
        self, x: torch.Tensor, x_size: tuple[int, int], masks: Masks
    ) -> torch.Tensor:
        height, width = x_size
        batch, length, channels = x.shape
        shift, window = self.shift_size, self.window_size
        shortcut = x
        x_2d = self.norm1(x).view(batch, height, width, channels)
        conv_x = self.conv_block(x_2d.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(batch, length, channels)
        shifted = torch.roll(x_2d, (-shift, -shift), (1, 2)) if shift else x_2d
        windows = legacy.window_partition(shifted, window)
        windows = windows.view(-1, window * window, channels)
        bias = self.attn.bias()
        if shift:
            windows = self._shifted_attention(
                windows, bias, batch, height, width, masks
            )
        else:
            windows = self.attn(windows, bias)
        windows = windows.view(-1, window, window, channels)
        attn_x = legacy.window_reverse(windows, window, height, width)
        if shift:
            attn_x = torch.roll(attn_x, (shift, shift), (1, 2))
        attn_x = self.ls_attn(attn_x.view(batch, length, channels))
        conv_x = self.ls_conv(conv_x)
        x = shortcut + attn_x + conv_x * self.conv_scale
        x = x + self.ls_ffn(self.ffn(self.norm2(x), height, width))
        return _barrier(x) if self.barrier else x


class _OCAB(nn.Module):
    """OCAB with its q/k/v window packing and (for the plugin's windows) its relative
    bias as plugins; the attention itself stays plain maths."""

    def __init__(self, ocab: legacy.OCAB) -> None:
        super().__init__()
        self.norm1 = ocab.norm1
        self.qkv = ocab.qkv
        self.relative_position_bias_table = ocab.relative_position_bias_table
        self.proj = ocab.proj
        self.ls_attn = ocab.ls_attn
        self.ls_ffn = ocab.ls_ffn
        self.norm2 = ocab.norm2
        self.ffn = ocab.ffn
        self.window_size = ocab.window_size
        self.overlap_win_size = ocab.overlap_win_size
        self.num_heads = ocab.num_heads
        self.dim = ocab.dim
        self.scale = ocab.scale
        self.bias_plugin = (ocab.window_size, ocab.overlap_win_size) in BIAS_WINDOWS

    def _dense_bias(self, dtype: torch.dtype) -> torch.Tensor:
        """OCAB._relative_position_bias's plain maths."""
        window, overlap = self.window_size, self.overlap_win_size
        table = self.relative_position_bias_table
        index = _relative_index(window, overlap, table.device)
        bias = table[index.view(-1)].view(window**2, overlap**2, self.num_heads)
        return bias.permute(2, 0, 1).contiguous().to(dtype=dtype).unsqueeze(0)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        height, width = x_size
        batch, length, channels = x.shape
        window, overlap = self.window_size, self.overlap_win_size
        shortcut = x
        x_norm = self.norm1(x).view(batch, height, width, channels)
        qkv = self.qkv(x_norm).reshape(batch, height, width, 3, channels)
        q, k, v = _QKVPack.apply(qkv, self.dim, self.num_heads, window, overlap)
        q = q.permute(1, 2, 0, 3)
        k_t = k.permute(1, 2, 3, 0)
        v = v.permute(1, 2, 0, 3)
        if self.bias_plugin:
            bias = _RelativeBias.apply(
                self.relative_position_bias_table, window, overlap
            )
            bias = bias.to(dtype=q.dtype)
        else:
            bias = self._dense_bias(q.dtype)
        x = _attention(q, k_t, v, bias, self.scale)
        x = x.transpose(1, 2).reshape(-1, window * window, self.dim)
        x = x.view(-1, window, window, self.dim)
        x = legacy.window_reverse(x, window, height, width).view(
            batch, length, self.dim
        )
        x = shortcut + self.ls_attn(self.proj(x))
        return x + self.ls_ffn(self.ffn(self.norm2(x), height, width))


class _DenseFusion(torch.autograd.Function):
    """The dense skip's Linear over the concatenated block outputs (exactly, from the
    weight's per-block slices, given transposed), exported as one MatMul per block and
    their sum: TensorRT 11 computes the concatenated BF16 form wrongly in this graph
    (DRFT-L x4: 19 dB against PyTorch, 46.6 dB split; Codex split it too)."""

    @staticmethod
    def forward(
        ctx: Any, bias: torch.Tensor | None, blocks: int, *tensors: torch.Tensor
    ) -> torch.Tensor:
        weight = torch.cat([part.t() for part in tensors[:blocks]], dim=1)
        return functional.linear(torch.cat(tensors[blocks:], dim=-1), weight, bias)

    @staticmethod
    def symbolic(g: Any, bias: Any, blocks: int, *values: Any) -> Any:
        total = None
        for part, feature in zip(values[:blocks], values[blocks:]):
            product = g.op("MatMul", feature, part)
            total = product if total is None else g.op("Add", total, product)
        return total if bias is None else g.op("Add", total, bias)


class _AttentionBlocks(nn.Module):
    """A residual group's evaluation forward with its dense skip as _DenseFusion."""

    def __init__(self, group: legacy.AttentionBlocks) -> None:
        super().__init__()
        self.blocks = group.blocks
        self.ocab = group.ocab
        self.dense_skip = group.dense_skip
        if self.dense_skip:
            self.ls_dense = group.ls_dense
            self.dense_bias = group.dense_fusion.bias
            parts = group.dense_fusion.weight.detach().chunk(len(group.blocks), dim=1)
            for index, part in enumerate(parts):
                self.register_buffer(f"dense_part_{index}", part.t().contiguous())

    def forward(
        self, x: torch.Tensor, x_size: tuple[int, int], hybrid_ctx: Any = None
    ) -> torch.Tensor:
        outputs = []
        for block in self.blocks:
            x = block(x, x_size, hybrid_ctx)
            outputs.append(x)
        x = self.ocab(x, x_size)
        if not self.dense_skip:
            return x
        parts = [self.get_buffer(f"dense_part_{i}") for i in range(len(outputs))]
        dense = _DenseFusion.apply(self.dense_bias, len(outputs), *parts, *outputs)
        return x + self.ls_dense(dense)


def _region_masks(window: int) -> list[torch.Tensor]:
    """The shifted blocks' region masks (spandrel's Swin-style regions, shift half the
    window) of a right-column, bottom-row and corner window: the same at every size."""
    shift = window // 2
    image = torch.zeros(1, 2 * window, 2 * window, 1)
    bands = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
    for row, rows in enumerate(bands):
        for column, columns in enumerate(bands):
            image[:, rows, columns, :] = row * len(bands) + column
    region_ids = legacy.window_partition(image, window).view(-1, window * window).long()
    masks = []
    for regions in region_ids[1:]:
        same = regions.unsqueeze(1) == regions.unsqueeze(0)
        mask = torch.zeros(same.shape, dtype=torch.float32, device=regions.device)
        mask.masked_fill_(~same, float("-inf"))
        masks.append(mask.unsqueeze(0).unsqueeze(0))
    return masks


class Deployment(nn.Module):
    """A legacy DRFT (folded, swapped in place) for TensorRT export: the original
    forward with the stand-ins, identity barriers at Codex's sites, and the precision
    casts (body: `body`; upsampler and last convolution: `tail`)."""

    def __init__(
        self,
        model: legacy.DRFT,
        body: torch.dtype = torch.float32,
        tail: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        for index, group in enumerate(model.layers):
            assert isinstance(group, legacy.RHAG)
            residual = group.residual_group
            for position, block in enumerate(residual.blocks):
                assert isinstance(block, legacy.ACTBlock)
                barrier = position % BARRIER_EVERY == BARRIER_EVERY - 1
                residual.blocks[position] = _ACTBlock(block, barrier)
            residual.add_module("ocab", _OCAB(residual.ocab))
            group.add_module("residual_group", _AttentionBlocks(residual))
            model.layers[index] = _Barriered(group)
        model.add_module("norm", _Barriered(model.norm))
        model.add_module("conv_before_upsample", _Barriered(model.conv_before_upsample))
        up = model.upsample.up
        if isinstance(up, nn.Sequential):
            for index, layer in enumerate(up):
                if isinstance(layer, nn.PixelShuffle):
                    up[index] = _Barriered(layer)
        model.to(dtype=body)
        model.upsample.to(dtype=tail)
        model.conv_last.to(dtype=tail)
        self.inner = model
        self.body = body
        self.tail = tail
        self.final_plugin = (
            model.img_range == 1 and model.get_buffer("mean").shape[1] == 3
        )
        right, bottom, corner = _region_masks(model.window_size)
        self.register_buffer("right_mask", right, persistent=False)
        self.register_buffer("bottom_mask", bottom, persistent=False)
        self.register_buffer("corner_mask", corner, persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        inner = self.inner
        height, width = image.shape[2:]
        x = image.to(dtype=self.body)
        mean = inner.get_buffer("mean").to(dtype=x.dtype, device=x.device)
        x = (x - mean) * inner.img_range
        x = legacy.pad_to_multiple(x, inner.window_size)
        shallow = inner.conv_first(x)
        x_size = (shallow.shape[2], shallow.shape[3])
        masks = (self.right_mask, self.bottom_mask, self.corner_mask)
        deep = inner.patch_embed(shallow)
        for layer in inner.layers:
            deep = layer(deep, x_size, masks)
        deep = inner.patch_unembed(inner.norm(deep), x_size)
        deep = _barrier(inner.conv_after_body(deep) + shallow)
        x = inner.conv_before_upsample(deep).to(dtype=self.tail)
        x = inner.conv_last(inner.upsample(x)).to(dtype=self.body)
        if self.final_plugin:
            means = tuple(mean.flatten().tolist())
            return _FinalAffineCrop.apply(x, image, inner.upscale, means)
        x = x / inner.img_range + mean
        return x[:, :, : height * inner.upscale, : width * inner.upscale].float()


def deployment(
    model: legacy.DRFT,
    body: torch.dtype = torch.float32,
    tail: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> Deployment:
    """A deep copy of a legacy DRFT, folded, in FP32 on `device` (the CPU by
    default), swapped for export."""
    device = device or torch.device("cpu")
    model = copy.deepcopy(model).to(device=device, dtype=torch.float32).eval()
    with torch.no_grad():
        model.fold_reparam_conv()
    return Deployment(model, body, tail).to(device).eval()


def export_onnx(
    model: ImageModelDescriptor, device: torch.device | None = None
) -> bytes:
    """The TensorRT-only ONNX of a legacy DRFT (is_legacy_drft), with the plugin
    nodes and the exporter's manifest (drft.METADATA_KEY) in its metadata, traced on
    `device` (the CPU by default). Convert To ONNX traces where Load Model put the
    model: PyTorch's CPU convolutions in FP16 are about 3,000 times slower than in
    BF16 (a DRFT-L x4 export took 605 s on the CPU, 37 s in BF16)."""
    import onnx

    network = model.model
    if not isinstance(network, legacy.DRFT) or not is_legacy_drft(model):
        raise ValueError(
            "Only legacy DRFT models with LayerNorm take the DRFT plugins."
        )
    window = network.window_size
    for module in network.modules():
        if isinstance(module, legacy.ACTBlock) and module.window_size != window:
            raise ValueError("This DRFT's blocks do not all use its window size.")
    wrapper = deployment(network, torch.float16, torch.float16, device)
    side = 3 * window
    sample = torch.rand(1, model.input_channels, side, side, device=device)
    with tempfile.TemporaryDirectory(prefix="chainner-drft-export-") as temporary:
        path = Path(temporary) / "model.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (sample,),
                path,
                dynamo=False,
                opset_version=OPSET,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {2: "height", 3: "width"},
                    "output": {2: "output_height", 3: "output_width"},
                },
                do_constant_folding=True,
            )
        proto = onnx.load(str(path))
    manifest: dict[str, Any] = {
        "format": 1,
        "plugins": PLUGINS,
        "scale": model.scale,
        "window_size": window,
        "min_size": min_size(window),
        # The graph pads its input to the window itself, as PyTorch does.
        "alignment": 1,
    }
    entry = proto.metadata_props.add()
    entry.key, entry.value = drft.METADATA_KEY, json.dumps(manifest)
    return proto.SerializeToString()
