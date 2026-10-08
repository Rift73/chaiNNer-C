"""DUAL: standalone DUAL3X with differentiable pre-mixed dynamic convolutions.

Presets: Light C128/PU2/depths(4,3,3,6), XS C128/native with a latency-selected
schedule, S C160/G6, M C192/G6, L C224/G10, XL C256/G12. Full groups contain
six layers. All retain head width32, F64, EDBB, XR/XG and ordinary (not
factorized) FFN/attention 1x1 projections. Four zero-initialized experts at
each group write-back preserve that preset's static-base initial function.
Kernels are mixed per image BEFORE convolution; no stacked expert responses.

Native output scales1/2/4 work for every preset. PU2 packs mean-shifted RGB
and the coarse RGB branch, pads to a multiple of4 then crops the output.
Its reconstruction scale is twice the requested net scale; the x8 head
uses a 12-channel XG readout followed by PixelShuffle2.

DUAL(scale=4) retains the original C96/G4x6 DUAL3X state/init contract.
Factory names now describe new geometries: old consolidated DUAL4 factory
checkpoints are NOT compatible. Context windows are not in checkpoints;
set c_windows to the geometry used for training (default32/64).

Dependencies: PyTorch, optional traiNNer registry, optional Triton/direct
attention APIs. All model and kernel code is bundled; no other architecture
imports. Eager and exported attention use SDPA. prepare_for_compile selects
the original direct attention implementation, not the rolled-back speed
optimization. Export folds an evaluation COPY's static EDBB bases only:
routers and expert kernels remain input-dependent. Never fold for training.

Attention-kernel source SHA256:
cd50b9a8e761d59f66130151ea55ff721b148d64eeab6ac9e945debe45c98ca7.
"""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ["DUAL", "dual_light", "dual_xs", "dual_s", "dual_m", "dual_l", "dual_xl"]


def _init_weights(module: nn.Module) -> None:
    """Preserve the source trunk/transfer initialization and RNG order."""
    if isinstance(module, nn.Linear) or (
        isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1)
    ):
        nn.init.trunc_normal_(module.weight, std=0.02)
    if isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
        nn.init.zeros_(module.bias)


PositionBias = Literal["rpb", "rib", "none"]
WindowMode = Literal["anchored"]
QKMode = Literal["dot", "cosine"]


@dataclass(frozen=True)
class AxisWindows:
    """Gather indices and the unique designated output slot for each pixel."""

    indices: Tensor
    restore: Tensor


def anchored_axis(
    length: int, window: int, offset: int, device: torch.device | str = "cpu"
) -> AxisWindows:
    """Clamp segment windows inward without changing their output ownership.

    Segments start at -offset + k*window. Each real position belongs to its
    segment, but its attention window is anchored inside the image. When an
    axis is shorter than one window, both offsets use one clamped window.
    """
    if length < 1 or window < 1 or offset not in (0, window // 2):
        raise ValueError("positive length/window and offset 0 or window//2 required")
    positions = torch.arange(length, device=device)
    if length <= window:
        return AxisWindows(positions.unsqueeze(0), positions)
    starts = torch.arange(-offset, length, window, device=device)
    starts = starts.clamp(0, length - window)
    owner = (positions + offset) // window
    local = positions - starts.index_select(0, owner)
    indices = starts[:, None] + torch.arange(window, device=device)[None, :]
    return AxisWindows(indices, owner * window + local)


def gather_windows(x: Tensor, rows: AxisWindows, cols: AxisWindows) -> Tensor:
    """BHWC -> (batch*row_windows*column_windows, tokens, channels)."""
    batch, _, _, channels = x.shape
    nr, wh = rows.indices.shape
    nc, ww = cols.indices.shape
    gathered = x.index_select(1, rows.indices.flatten()).index_select(
        2, cols.indices.flatten()
    )
    return (
        gathered.reshape(batch, nr, wh, nc, ww, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch * nr * nc, wh * ww, channels)
    )


def restore_windows(
    windows: Tensor, rows: AxisWindows, cols: AxisWindows, batch: int
) -> Tensor:
    """Gather designated outputs, discarding duplicated overlap queries."""
    nr, wh = rows.indices.shape
    nc, ww = cols.indices.shape
    channels = windows.shape[-1]
    expanded = (
        windows.reshape(batch, nr, nc, wh, ww, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, nr * wh, nc * ww, channels)
    )
    return expanded.index_select(1, rows.restore).index_select(2, cols.restore)


def relative_position_index(
    height: int, width: int, nominal_window: int, device: torch.device
) -> Tensor:
    """Swin query-minus-key row-major table index, including rectangular crops."""
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    y, x = yy.flatten(), xx.flatten()
    stride = 2 * nominal_window - 1
    return (y[:, None] - y[None, :] + nominal_window - 1) * stride + (
        x[:, None] - x[None, :] + nominal_window - 1
    )


def calculation_dtype(dtype: torch.dtype) -> torch.dtype:
    """Use FP32 for low-precision stability, preserve FP64 reference tests."""
    return torch.float64 if dtype == torch.float64 else torch.float32


class SignedFactoredBias(nn.Module):
    """SST-style Fourier/ReLU signed positional factors, shared across windows."""

    def __init__(
        self, window: int, heads: int, rank: int = 16, hidden: int = 32
    ) -> None:
        super().__init__()
        if min(window, heads, rank, hidden) < 1:
            raise ValueError("RIB dimensions must be positive")
        self.window = window
        self.heads = heads
        self.rank = rank
        yy, xx = torch.meshgrid(
            torch.arange(window, dtype=torch.float64),
            torch.arange(window, dtype=torch.float64),
            indexing="ij",
        )
        coords = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        coords = 2 * (coords + 0.5) / window - 1
        features = [coords]
        for exponent in range(10):
            phase = coords * (2**exponent)
            features.extend((phase.sin(), phase.cos()))
        self.register_buffer(
            "coordinates", torch.cat(features, dim=-1), persistent=False
        )
        self.to_hidden = nn.Parameter(torch.empty(42, hidden))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden))
        self.to_query = nn.Parameter(torch.empty(heads, hidden, rank))
        self.to_key = nn.Parameter(torch.empty(heads, hidden, rank))
        for parameter in (self.to_hidden, self.to_query, self.to_key):
            nn.init.normal_(parameter, std=0.05)

    def forward(
        self, height: int, width: int, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        # Tiny rectangular inputs take leading rows AND columns of the nominal
        # window grid, rather than the first height*width entries of its flatten.
        device = self.to_hidden.device
        index = (
            torch.arange(height, device=device)[:, None] * self.window
            + torch.arange(width, device=device)[None, :]
        ).flatten()
        precision = calculation_dtype(dtype)
        with torch.autocast(device_type=device.type, enabled=False):
            features = self.coordinates.index_select(0, index).to(precision)
            hidden = F.relu(
                features @ self.to_hidden.to(precision) + self.hidden_bias.to(precision)
            )
            query = torch.matmul(hidden, self.to_query.to(precision))
            key = torch.matmul(hidden, self.to_key.to(precision))
            # SST's rank normalization is absorbed into the query factor.
            query = query * (self.rank**-0.5)
        return query.to(dtype), key.to(dtype)


class _WindowAttention(nn.Module):
    """Attention branch only: caller supplies normalized BCHW features.

    No LayerNorm or residual is hidden here; these belong to the M2 layer.
    ``pre_gate_core`` exposes window attention before the spatial gate/proj.
    The module respects an enclosing sdpa_kernel context for backend testing.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        window: int,
        *,
        shifted: bool = False,
        pos_bias: PositionBias = "rpb",
        window_mode: WindowMode = "anchored",
        rank: int = 16,
        qk: QKMode = "dot",
    ) -> None:
        super().__init__()
        if min(channels, heads, window) < 1 or channels % heads:
            raise ValueError(
                "positive dimensions and channels divisible by heads required"
            )
        if pos_bias not in ("rpb", "rib", "none"):
            raise ValueError("pos_bias must be rpb, rib or none")
        if window_mode != "anchored":
            raise ValueError("DUAL requires anchored spatial windows")
        if qk not in ("dot", "cosine"):
            raise ValueError("qk must be dot or cosine")
        self.channels = channels
        self.heads = heads
        self.window = window
        self.offset = window // 2 if shifted else 0
        self.pos_bias = pos_bias
        self.window_mode = window_mode
        self.qk_mode = qk
        self.qkv = nn.Linear(channels, 3 * channels)
        self.logit_scale = (
            nn.Parameter(torch.full((heads, 1, 1), math.log(10)))
            if qk == "cosine"
            else None
        )
        self.relative_bias: nn.Parameter | None = None
        self.factored_bias: SignedFactoredBias | None = None
        if pos_bias == "rpb":
            self.relative_bias = nn.Parameter(torch.empty((2 * window - 1) ** 2, heads))
            nn.init.trunc_normal_(self.relative_bias, std=0.02)
        elif pos_bias == "rib":
            self.factored_bias = SignedFactoredBias(window, heads, rank)
        self.gate_depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )
        self.gate_pointwise = nn.Conv2d(channels, channels, 1)
        self.project = nn.Linear(channels, channels)

    def position_bias(
        self, height: int, width: int, dtype: torch.dtype
    ) -> Tensor | None:
        """Dense RPB only; never construct an N*N RIB tensor in the core."""
        if self.relative_bias is None:
            return None
        index = relative_position_index(
            height, width, self.window, self.relative_bias.device
        )
        precision = calculation_dtype(dtype)
        values = self.relative_bias.to(precision)[index.flatten()]
        return (
            values.reshape(height * width, height * width, self.heads)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(dtype)
        )

    def attend_windows(
        self, tokens: Tensor, window_shape: tuple[int, int], mask: Tensor | None = None
    ) -> Tensor:
        """Pre-gate single/multi-window primitive; no routing or image context."""
        batch_windows, count, channels = tokens.shape
        head_dim = channels // self.heads
        qkv = self.qkv(tokens).reshape(batch_windows, count, 3, self.heads, head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        dtype = query.dtype
        precision = calculation_dtype(dtype)
        if self.logit_scale is None:
            query = query * (head_dim**-0.5)
        else:
            with torch.autocast(device_type=tokens.device.type, enabled=False):
                temperature = (
                    self.logit_scale.to(precision).clamp(max=math.log(100)).exp()
                )
                query = (
                    F.normalize(query.to(precision), dim=-1, eps=1e-6) * temperature
                ).to(dtype)
                key = F.normalize(key.to(precision), dim=-1, eps=1e-6).to(dtype)
        if self.factored_bias is not None:
            bq, bk = self.factored_bias(*window_shape, dtype)
            query = torch.cat(
                (query, bq.unsqueeze(0).expand(batch_windows, -1, -1, -1)), dim=-1
            )
            key = torch.cat(
                (key, bk.unsqueeze(0).expand(batch_windows, -1, -1, -1)), dim=-1
            )
            value = F.pad(value, (0, self.factored_bias.rank))
        bias = self.position_bias(*window_shape, dtype)
        if mask is not None:
            bias = mask.to(dtype) if bias is None else bias + mask.to(dtype)
        output = F.scaled_dot_product_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            attn_mask=None if bias is None else bias.contiguous(),
            dropout_p=0.0,
            scale=1.0,
        )[..., :head_dim]
        return output.transpose(1, 2).reshape(batch_windows, count, channels)

    def pre_gate_core(self, x: Tensor) -> Tensor:
        """BCHW window-attention output, excluding the gate and projection."""
        if x.ndim != 4 or x.shape[1] != self.channels or min(x.shape[2:]) < 1:
            raise ValueError("expected nonempty BCHW input with matching channels")
        batch, _, height, width = x.shape
        image = x.permute(0, 2, 3, 1)
        rows = anchored_axis(height, self.window, self.offset, x.device)
        cols = anchored_axis(width, self.window, self.offset, x.device)
        tokens = gather_windows(image, rows, cols)
        output = self.attend_windows(
            tokens, (rows.indices.shape[1], cols.indices.shape[1])
        )
        return restore_windows(output, rows, cols, batch).permute(0, 3, 1, 2)

    def forward(self, x: Tensor) -> Tensor:
        attended = self.pre_gate_core(x)
        gate = self.gate_pointwise(self.gate_depthwise(x)).sigmoid()
        return self.project((attended * gate).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class TokenNorm(nn.Module):
    """BCHW per-token LayerNorm control with FP32 low-precision statistics."""

    def __init__(self, channels: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        precision = calculation_dtype(x.dtype)
        with torch.autocast(device_type=x.device.type, enabled=False):
            result = F.layer_norm(
                x.permute(0, 2, 3, 1).to(precision),
                self.norm.normalized_shape,
                self.norm.weight.to(precision),
                self.norm.bias.to(precision),
                self.norm.eps,
            )
        return result.permute(0, 3, 1, 2).to(x.dtype)


class ConvFFN(nn.Module):
    """Non-gated SST-style convolutional feed-forward branch, expansion two."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = 2 * channels
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.project = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        hidden = F.gelu(self.expand(x))
        return self.project(hidden + F.gelu(self.depthwise(hidden)))


def direct_g16_attention(
    query: Tensor, key: Tensor, value: Tensor, table: Tensor
) -> Tensor:
    """Retain the existing G16 entry point and equation."""
    return direct_g_attention(query, key, value, table, 16)


def direct_g_attention(
    query: Tensor, key: Tensor, value: Tensor, table: Tensor, window: int
) -> Tensor:
    """G16/G32 adapter to the retained, unmodified DRFT dense-RPB kernel.

    Q is already scaled. DRFT indexes key-minus-query, whereas GRAFT uses
    query-minus-key: their indices sum to (2*window-1)**2-1 (960 or 3968).
    """
    return ocab_attention(
        query, key, value, table.flip(0), window, window, window, window, 1.0
    )


class RelativeFourierBias(nn.Module):
    """Factor a stationary signed relative bias without constructing N*N.

    Dot(q_factor(i), k_factor(j)) = sum(a*cos(w*(i-j))+b*sin(w*(i-j))).
    Frequency and amplitude shapes are independent of the attention window.
    """

    def __init__(self, heads: int, rank: int = 16) -> None:
        super().__init__()
        if heads < 1 or rank < 2 or rank % 2:
            raise ValueError("positive heads and a positive even rank required")
        self.rank = rank
        count = rank // 2
        magnitudes = torch.logspace(
            math.log10(2 * math.pi / 128), math.log10(math.pi / 2), count
        )
        angles = torch.arange(count) * (math.pi * (3 - math.sqrt(5)))
        self.frequency = nn.Parameter(
            magnitudes[:, None] * torch.stack((angles.cos(), angles.sin()), dim=-1)
        )
        self.amplitude_cos = nn.Parameter(torch.full((heads, count), 1 / count))
        self.amplitude_sin = nn.Parameter(torch.zeros(heads, count))

    def factors(self, coordinates: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        precision = calculation_dtype(dtype)
        with torch.autocast(device_type=self.frequency.device.type, enabled=False):
            phase = coordinates.to(precision) @ self.frequency.to(precision).T
            cosine, sine = phase.cos()[None], phase.sin()[None]
            a = self.amplitude_cos.to(precision)[:, None]
            b = self.amplitude_sin.to(precision)[:, None]
            query = torch.cat((a * cosine + b * sine, a * sine - b * cosine), dim=-1)
            key = torch.cat((cosine, sine), dim=-1).expand(query.shape[0], -1, -1)
        return query.to(dtype), key.to(dtype)

    def forward(
        self, height: int, width: int, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        yy, xx = torch.meshgrid(
            torch.arange(height, device=self.frequency.device),
            torch.arange(width, device=self.frequency.device),
            indexing="ij",
        )
        return self.factors(torch.stack((yy, xx), dim=-1).reshape(-1, 2), dtype)


class _SpatialAttention(_WindowAttention):
    """Reuse v0 attention math, with RFB and exact regular window routing."""

    def __init__(
        self,
        channels: int,
        heads: int,
        window: int,
        *,
        pos_bias: str = "rfb",
        g_attention_impl: Literal["sdpa", "direct"] = "sdpa",
        **kwargs,
    ) -> None:
        if pos_bias not in ("rfb", "rpb", "rib", "none"):
            raise ValueError("pos_bias must be rfb/rpb/rib/none")
        if g_attention_impl not in ("sdpa", "direct"):
            raise ValueError("g_attention_impl must be sdpa/direct")
        super().__init__(
            channels,
            heads,
            window,
            pos_bias="none" if pos_bias == "rfb" else pos_bias,
            **kwargs,
        )
        if pos_bias == "rfb":
            self.factored_bias = RelativeFourierBias(heads, kwargs.get("rank", 16))
            self.pos_bias = "rfb"
        self.g_attention_impl = g_attention_impl
        self.direct_fallback_reason: str | None = None

    def _record_direct_fallback(self, reason: str) -> None:
        if self.direct_fallback_reason is None:
            self.direct_fallback_reason = reason
            warnings.warn(
                f"GRAFT G{self.window} direct fallback to SDPA: {reason}", stacklevel=3
            )

    def attend_windows(
        self, tokens: Tensor, window_shape: tuple[int, int], mask: Tensor | None = None
    ) -> Tensor:
        if self.g_attention_impl == "sdpa" or self.pos_bias != "rpb":
            return super().attend_windows(tokens, window_shape, mask)
        if (
            tokens.device.type != "cuda"
            or self.window_mode != "anchored"
            or self.window not in (16, 32)
            or window_shape != (self.window, self.window)
            or mask is not None
            or self.logit_scale is not None
        ):
            self._record_direct_fallback(
                "requires CUDA, anchored full G16/G32, no mask and dot attention"
            )
            return super().attend_windows(tokens, window_shape, mask)
        batch_windows, count, channels = tokens.shape
        head_dim = channels // self.heads
        qkv = self.qkv(tokens).reshape(batch_windows, count, 3, self.heads, head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if query.dtype not in (torch.bfloat16, torch.float16):
            self._record_direct_fallback("requires BF16 or FP16 attention inputs")
            return super().attend_windows(tokens, window_shape, mask)
        operands = (
            (query * (head_dim**-0.5)).contiguous(),
            key.contiguous(),
            value.contiguous(),
            self.relative_bias,
        )
        output = (
            direct_g16_attention(*operands)
            if self.window == 16
            else direct_g_attention(*operands, self.window)
        )
        return output.transpose(1, 2).reshape(batch_windows, count, channels)

    def pre_gate_core(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels or min(x.shape[2:]) < 1:
            raise ValueError("expected nonempty BCHW input with matching channels")
        batch, channels, height, width = x.shape
        wh, ww = min(height, self.window), min(width, self.window)
        if (
            self.window_mode == "anchored"
            and self.offset == 0
            and height % wh == 0
            and width % ww == 0
        ):
            nr, nc = height // wh, width // ww
            tokens = (
                x.permute(0, 2, 3, 1)
                .reshape(batch, nr, wh, nc, ww, channels)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(batch * nr * nc, wh * ww, channels)
            )
            output = self.attend_windows(tokens, (wh, ww))
            return (
                output.reshape(batch, nr, nc, wh, ww, channels)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(batch, height, width, channels)
                .permute(0, 3, 1, 2)
            )
        return super().pre_gate_core(x)


def axis_spans(
    length: int, window: int, offset: int = 0
) -> tuple[tuple[int, int, int], ...]:
    """(start, stop, window extent); each input position is owned exactly once."""
    if length < 1 or window < 1 or not 0 <= offset < window:
        raise ValueError("nonempty axis and 0 <= offset < window required")
    if length <= window:
        return ((0, length, length),)
    spans = []
    if offset:
        spans.append((0, offset, offset))
    end = offset + ((length - offset) // window) * window
    if end > offset:
        spans.append((offset, end, window))
    if end < length:
        spans.append((end, length, length - end))
    return tuple(spans)


def block_layout(height: int, width: int, window: int, offset: int = 0) -> tuple:
    """Row-major rectangular blocks with equal-sized windows in each block."""
    return tuple(
        (y0, y1, x0, x1, wh, ww)
        for y0, y1, wh in axis_spans(height, window, offset)
        for x0, x1, ww in axis_spans(width, window, offset)
    )


def pack_block(x: Tensor, block: tuple[int, ...]) -> Tensor:
    y0, y1, x0, x1, wh, ww = block
    batch, channels = x.shape[:2]
    return (
        x[:, :, y0:y1, x0:x1]
        .reshape(batch, channels, (y1 - y0) // wh, wh, (x1 - x0) // ww, ww)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(-1, wh * ww, channels)
    )


def unpack_block(tokens: Tensor, block: tuple[int, ...], batch: int) -> Tensor:
    y0, y1, x0, x1, wh, ww = block
    channels = tokens.shape[-1]
    return (
        tokens.reshape(batch, (y1 - y0) // wh, (x1 - x0) // ww, wh, ww, channels)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, channels, y1 - y0, x1 - x0)
    )


def join_blocks(outputs: list[Tensor], layout: tuple) -> Tensor:
    rows, current = [], []
    previous = layout[0][0]
    for result, block in zip(outputs, layout, strict=True):
        if block[0] != previous:
            rows.append(torch.cat(current, dim=3))
            current = []
            previous = block[0]
        current.append(result)
    rows.append(torch.cat(current, dim=3))
    return torch.cat(rows, dim=2)


def grouped_layout(layout: tuple) -> tuple:
    """Batch equal window shapes; never cache tensor outputs across forwards."""
    groups = {}
    for index, block in enumerate(layout):
        groups.setdefault(block[-2:], []).append((index, block))
    return tuple(groups.items())


def pad_even(x: Tensor) -> Tensor:
    return F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")


def pack_children(x: Tensor) -> Tensor:
    """B,8*heads,H,W -> B,32*heads,H/2,W/2; head/slot/channel order."""
    b, c, h, w = x.shape
    return (
        F.pixel_unshuffle(x, 2)
        .reshape(b, c // 8, 8, 4, h // 2, w // 2)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(b, 4 * c, h // 2, w // 2)
    )


def unpack_children(x: Tensor) -> Tensor:
    """Canonical head/slot/c retrieval -> B,8*heads,2H,2W."""
    b, c, h, w = x.shape
    slots = (
        x.reshape(b, c // 32, 4, 8, h, w).permute(0, 1, 3, 2, 4, 5).reshape(b, c, h, w)
    )
    return F.pixel_shuffle(slots, 2)


class CrossScaleRetrieval(nn.Module):
    """Parameters/operators for DUAL's shared packed cross-scale retrieval."""

    def __init__(self, channels: int = 96, *, include_hr: bool = True) -> None:
        super().__init__()
        self.channels, self.heads = channels, channels // 32
        self.norm_f = TokenNorm(channels, 1e-5)
        self.norm_u = TokenNorm(channels, 1e-5)
        self.query = nn.Conv2d(2 * channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels // 4, 1)
        self.gate = nn.Conv2d(channels, channels, 1)
        self.project = nn.Conv2d(channels, channels, 1)
        self.hr = nn.Conv2d(channels // 4, 64, 1) if include_hr else None
        self.log_tau = nn.Parameter(torch.zeros(self.heads))
        self.register_buffer(
            "down_taps",
            torch.tensor(
                [
                    -0.01171875,
                    -0.03515625,
                    0.11328125,
                    0.43359375,
                    0.43359375,
                    0.11328125,
                    -0.03515625,
                    -0.01171875,
                ]
            ),
            persistent=False,
        )
        self.apply(_init_weights)  # noqa: SLF001 - preserve the parent initialization contract
        for layer in (self.project, self.hr):
            if layer is not None:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def downsample(self, shifted: Tensor) -> Tensor:
        x = pad_even(shifted)
        taps = self.down_taps.to(x.dtype)
        h = taps.view(1, 1, 1, 8).expand(3, -1, -1, -1)
        v = taps.view(1, 1, 8, 1).expand(3, -1, -1, -1)
        x = F.conv2d(
            F.pad(x, (3, 3, 0, 0), mode="replicate"), h, stride=(1, 2), groups=3
        )
        return F.conv2d(
            F.pad(x, (0, 0, 3, 3), mode="replicate"), v, stride=(2, 1), groups=3
        )


def pack_grandchildren(x: Tensor) -> Tensor:
    """B,2*heads,H,W -> B,32*heads,H/4,W/4; head/slot/channel order."""
    b, c, h, w = x.shape
    return (
        F.pixel_unshuffle(x, 4)
        .reshape(b, c // 2, 2, 16, h // 4, w // 4)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(b, 16 * c, h // 4, w // 4)
    )


def unpack_grandchildren(x: Tensor) -> Tensor:
    b, c, h, w = x.shape
    raw = x.reshape(b, c // 32, 16, 2, h, w).permute(0, 1, 3, 2, 4, 5)
    return F.pixel_shuffle(raw.reshape(b, c, h, w), 4)


def pack_qk(q: Tensor, k: Tensor) -> tuple:
    layout = block_layout(*q.shape[-2:], 96)
    classes = []
    for _, items in grouped_layout(layout):
        qs = [pack_block(q, block) for _, block in items]
        ks = [pack_block(k, tuple(t // 2 for t in block)) for _, block in items]
        qq, kk = (
            torch.cat(z, 0)
            .reshape(-1, z[0].shape[1], q.shape[1] // 32, 32)
            .transpose(1, 2)
            for z in (qs, ks)
        )
        classes.append((items, qq, kk.contiguous()))
    return layout, classes, q.shape[0]


def attend_packed(packed: tuple, v: Tensor, log_tau: Tensor) -> Tensor:
    layout, classes, batch = packed
    outputs = [None] * len(layout)
    for items, qq, kk in classes:
        vs = [pack_block(v, tuple(t // 2 for t in block)) for _, block in items]
        vv = (
            torch.cat(vs, 0)
            .reshape(-1, vs[0].shape[1], qq.shape[1], 32)
            .transpose(1, 2)
        )
        tau = log_tau.exp().clamp(0.1, 10).to(qq.dtype)
        out = (
            F.scaled_dot_product_attention(
                (qq * tau[None, :, None, None]).contiguous(),
                kk,
                vv.contiguous(),
                scale=math.log(kk.shape[2]) / (math.log(1024) * math.sqrt(32)),
            )
            .transpose(1, 2)
            .reshape(-1, qq.shape[2], 32 * qq.shape[1])
        )
        start = 0
        for (index, block), chunk in zip(items, vs, strict=True):
            count = chunk.shape[0]
            outputs[index] = unpack_block(out[start : start + count], block, batch)
            start += count
    return join_blocks(outputs, layout)


class SecondOctaveTransfer(nn.Module):
    def __init__(self, channels: int = 96) -> None:
        super().__init__()
        self.value = nn.Conv2d(64, channels // 16, 1)
        self.out = nn.Conv2d(channels // 16, 3, 3, padding=1, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(channels // 32))
        self.apply(_init_weights)  # noqa: SLF001 - preserve base initialization
        nn.init.zeros_(self.out.weight)

    def forward(self, t2: Tensor, packed: tuple, shape: tuple[int, int]) -> Tensor:
        h, w = shape
        g = F.pad(self.value(t2), (0, 2 * (w % 2), 0, 2 * (h % 2)), mode="replicate")
        o2 = attend_packed(packed, pack_grandchildren(g), self.log_tau)
        r = unpack_grandchildren(o2)[:, :, : 4 * h, : 4 * w]
        return self.out(r)


def hat_weights(
    length: int,
    shift: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Nominal-window hats on real pixels, including clipped border windows."""
    if length < 1 or shift not in (0, 16):
        raise ValueError("DUAL-R needs a positive axis and a 0/16 shift")
    t = (torch.arange(length, device=device) + shift) % 32
    return (1 - ((t.to(dtype) - 15.5).abs() / 16)).to(dtype)


class RegionalChannelAttention(nn.Module):
    """Pre-norm residual BRANCH (the caller adds x); no hidden mutable state."""

    def __init__(
        self, channels: int = 96, heads: int = 3, *, core_checkpoint: bool = True
    ) -> None:
        super().__init__()
        if channels < 1 or heads < 1 or channels % heads:
            raise ValueError("channels must be positive and divisible by heads")
        self.channels, self.heads = channels, heads
        self.core_checkpoint = core_checkpoint
        self.norm = TokenNorm(channels, eps=1e-5)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.depthwise = nn.Conv2d(
            channels * 3, channels * 3, 3, padding=1, groups=channels * 3, bias=False
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        self.theta = nn.Parameter(torch.zeros(heads))
        self.apply(_init_weights)  # noqa: SLF001 - same new-module initialization

    def channel_attention(self, q: Tensor, k: Tensor, count: Tensor) -> Tensor:
        """(..., heads, d, tokens) -> (..., heads, d, d), FP32 or FP64.

        `count` is the number of REAL tokens, never the padded window area.
        Normalize after the raw Gram to avoid low-precision normalized operands.
        """
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=q.device.type, enabled=False):
            qf, kf = q.to(dtype), k.to(dtype)
            count = count.to(dtype)
            qnorm = (qf.square().sum(-1) + count * 1e-6).sqrt()
            knorm = (kf.square().sum(-1) + count * 1e-6).sqrt()
            gram = qf @ kf.transpose(-2, -1)
            cosine = gram / (qnorm[..., :, None] * knorm[..., None, :])
            tau = self.theta.to(dtype).clamp(max=math.log(100)).exp()
            return (cosine * tau[:, None, None]).softmax(-1)

    def _class_core(self, q: Tensor, k: Tensor, v: Tensor, sy: int, sx: int) -> Tensor:
        b, c, h, w = q.shape
        ph, pw = h + sy + (-(h + sy) % 32), w + sx + (-(w + sx) % 32)
        ny, nx, d = ph // 32, pw // 32, c // self.heads

        def windows(t: Tensor) -> Tensor:
            t = F.pad(t, (sx, pw - w - sx, sy, ph - h - sy))
            return (
                t.reshape(b, self.heads, d, ny, 32, nx, 32)
                .permute(0, 3, 5, 1, 2, 4, 6)
                .reshape(b, ny * nx, self.heads, d, 1024)
            )

        # Real clipped lengths rather than a stored pixel-validity tensor.
        y0 = torch.arange(ny, device=q.device) * 32 - sy
        x0 = torch.arange(nx, device=q.device) * 32 - sx
        yc = (y0 + 32).clamp(max=h) - y0.clamp(min=0)
        xc = (x0 + 32).clamp(max=w) - x0.clamp(min=0)
        count = (yc[:, None] * xc[None, :]).reshape(1, ny * nx, 1, 1)
        a = self.channel_attention(windows(q), windows(k), count)
        # The probability cast is part of the stated BF16 training function.
        # Keep matmul dtype explicit even under a surrounding autocast context.
        with torch.autocast(device_type=v.device.type, enabled=False):
            o = a.to(v.dtype) @ windows(v)
            o = (
                o.reshape(b, ny, nx, self.heads, d, 32, 32)
                .permute(0, 3, 4, 1, 5, 2, 6)
                .reshape(b, c, ph, pw)[:, :, sy : sy + h, sx : sx + w]
                .to(a.dtype)
            )
            wy = hat_weights(h, sy, device=q.device, dtype=a.dtype)
            wx = hat_weights(w, sx, device=q.device, dtype=a.dtype)
            return o * wy[None, None, :, None] * wx[None, None, None, :]

    def mix(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Blend four class outputs without an H*W*d*d matrix field."""
        check = self.core_checkpoint and self.training and torch.is_grad_enabled()
        result = None
        for sy, sx in ((0, 0), (0, 16), (16, 0), (16, 16)):
            out = (
                checkpoint(self._class_core, q, k, v, sy, sx, use_reentrant=False)
                if check
                else self._class_core(q, k, v, sy, sx)
            )
            result = out if result is None else result + out
        return result

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels or min(x.shape[-2:]) < 1:
            raise ValueError("expected nonempty BCHW with matching channels")
        q, k, v = self.depthwise(self.qkv(self.norm(x))).chunk(3, dim=1)
        return self.project(self.mix(q, k, v).to(x.dtype))


class DualContextLayer(nn.Module):
    def __init__(self, source: nn.Module, *, core_checkpoint: bool) -> None:
        super().__init__()
        self.norm1, self.attention = source.norm1, source.attention
        self.norm2, self.ffn = source.norm2, source.ffn
        self.dual = RegionalChannelAttention(
            self.attention.channels,
            self.attention.heads,
            core_checkpoint=core_checkpoint,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.dual(x)
        return x + self.ffn(self.norm2(x))


class _EDBBDenseBranch(nn.Module):
    """Independent gain-2 1x1 -> 3x3 -> 1x1 factors, never applied to images."""

    def __init__(self, ci: int, co: int, *, bias: bool, factory: dict) -> None:
        super().__init__()
        self.expand = nn.Conv2d(ci, 2 * ci, 1, bias=bias, **factory)
        self.spatial = nn.Conv2d(2 * ci, 2 * co, 3, bias=bias, **factory)
        self.reduce = nn.Conv2d(2 * co, co, 1, bias=bias, **factory)

    def equivalent_kernel(self, dtype: torch.dtype) -> tuple[Tensor, Tensor | None]:
        expand = self.expand.weight[:, :, 0, 0].to(dtype)
        spatial = self.spatial.weight.to(dtype)
        reduce = self.reduce.weight[:, :, 0, 0].to(dtype)
        projected_spatial = torch.einsum("om,mnxy->onxy", reduce, spatial)
        kernel = torch.einsum("omxy,mi->oixy", projected_spatial, expand)
        bias = None
        if self.expand.bias is not None:
            bias = reduce @ (
                spatial.sum((-2, -1)) @ self.expand.bias.to(dtype)
                + self.spatial.bias.to(dtype)
            ) + self.reduce.bias.to(dtype)
        return kernel, bias


class _EDBBPointwiseBranch(nn.Module):
    """Gain-2 pointwise factors; their effective kernel remains 1x1."""

    def __init__(self, ci: int, co: int, *, bias: bool, factory: dict) -> None:
        super().__init__()
        self.expand = nn.Conv2d(ci, 2 * ci, 1, bias=bias, **factory)
        self.reduce = nn.Conv2d(2 * ci, co, 1, bias=bias, **factory)

    def equivalent_kernel(self, dtype: torch.dtype) -> tuple[Tensor, Tensor | None]:
        reduce = self.reduce.weight[:, :, 0, 0].to(dtype)
        kernel = (reduce @ self.expand.weight[:, :, 0, 0].to(dtype))[:, :, None, None]
        bias = None
        if self.expand.bias is not None:
            bias = reduce @ self.expand.bias.to(dtype) + self.reduce.bias.to(dtype)
        return kernel, bias


class _EDBBEdgeBranch(nn.Module):
    """Expanded pointwise -> scaled depthwise edge filter -> output projection."""

    def __init__(
        self, ci: int, co: int, mask: Tensor, *, bias: bool, factory: dict
    ) -> None:
        super().__init__()
        # A projection bias cancels against the zero-sum bias-padded filter.
        self.project = nn.Conv2d(ci, 2 * co, 1, bias=False, **factory)
        self.scale = nn.Parameter(torch.zeros(2 * co, 1, 1, 1, **factory))
        self.bias = nn.Parameter(torch.zeros(2 * co, **factory)) if bias else None
        self.reduce = nn.Conv2d(2 * co, co, 1, bias=bias, **factory)
        self.register_buffer("mask", mask.reshape(1, 1, 3, 3), persistent=False)

    def equivalent_kernel(self, dtype: torch.dtype) -> tuple[Tensor, Tensor | None]:
        reduce = self.reduce.weight[:, :, 0, 0].to(dtype)
        expanded = (
            self.project.weight.to(dtype) * self.scale.to(dtype) * self.mask.to(dtype)
        )
        kernel = torch.einsum("om,mixy->oixy", reduce, expanded)
        bias = None
        if self.bias is not None:
            bias = reduce @ self.bias.to(dtype) + self.reduce.bias.to(dtype)
        return kernel, bias


class EDBBReparamConv3x3(nn.Module):
    """All learnable branches expanded; one differentiably composed convolution."""

    def __init__(self, original: nn.Conv2d) -> None:
        super().__init__()
        if not (
            original.kernel_size == (3, 3)
            and original.stride == original.dilation == (1, 1)
            and original.padding == (1, 1)
            and original.groups == 1
            and original.padding_mode == "zeros"
        ):
            raise ValueError("EDBB reparameterization requires a dense stride-1 3x3")
        ci, co = original.in_channels, original.out_channels
        self.in_channels, self.out_channels = ci, co
        self.gain = 2
        bias = original.bias is not None
        factory = {"device": original.weight.device, "dtype": original.weight.dtype}
        # Expanding the former direct 3x3 makes its topology match the sequential
        # branch. They remain independent factors, not shared/duplicated weights.
        self.direct = _EDBBDenseBranch(ci, co, bias=bias, factory=factory)
        self.sequential = _EDBBDenseBranch(ci, co, bias=bias, factory=factory)
        self.pointwise = _EDBBPointwiseBranch(ci, co, bias=bias, factory=factory)
        masks = (
            ((1, 0, -1), (2, 0, -2), (1, 0, -1)),
            ((1, 2, 1), (0, 0, 0), (-1, -2, -1)),
            ((0, 1, 0), (1, -4, 1), (0, 1, 0)),
        )
        self.edges = nn.ModuleList(
            _EDBBEdgeBranch(
                ci, co, torch.tensor(mask, **factory), bias=bias, factory=factory
            )
            for mask in masks
        )
        # Three dimensions keep channels_last conversion from rewriting strides.
        identity = torch.zeros(co, ci, 1, **factory)
        if ci == co:
            identity[:, :, 0].copy_(torch.eye(ci, **factory))
        self.register_buffer("identity_weight", identity, persistent=False)
        with torch.no_grad():
            # An exact identity route carries the original dense kernel. Other
            # factors stay random where their zero readout initially gates them:
            # this avoids tying duplicated channels or a permanently dead pair.
            self.direct.expand.weight[:ci, :, 0, 0].copy_(torch.eye(ci, **factory))
            self.direct.spatial.weight[:co].zero_()
            self.direct.spatial.weight[:co, :ci].copy_(original.weight)
            self.direct.reduce.weight.zero_()
            self.direct.reduce.weight[:, :co, 0, 0].copy_(torch.eye(co, **factory))
            # The independent sequential branch and edge branches start at zero.
            self.sequential.spatial.weight.zero_()
            # Cancel identity through an exact pointwise route, adding these
            # terms before the dense kernel to preserve exactly-zero readouts.
            self.pointwise.expand.weight[:ci, :, 0, 0].copy_(torch.eye(ci, **factory))
            self.pointwise.reduce.weight.zero_()
            self.pointwise.reduce.weight[:, :ci, 0, 0].copy_(-identity[:, :, 0])
            if bias:
                for module in self.modules():
                    if isinstance(module, nn.Conv2d) and module.bias is not None:
                        module.bias.zero_()
                self.direct.reduce.bias.copy_(original.bias)

    def equivalent_kernel(self, dtype: torch.dtype) -> tuple[Tensor, Tensor | None]:
        """Compose EVERY branch using ordinary autograd, never feature maps."""
        kernel, bias = self.direct.equivalent_kernel(dtype)
        point, point_bias = self.pointwise.equivalent_kernel(dtype)
        point = point + self.identity_weight.to(dtype).unsqueeze(-1)
        kernel = kernel + F.pad(point, (1, 1, 1, 1))
        if bias is not None:
            bias = bias + point_bias
        for branch in (self.sequential, *self.edges):
            branch_kernel, branch_bias = branch.equivalent_kernel(dtype)
            kernel = kernel + branch_kernel
            if bias is not None:
                bias = bias + branch_bias
        return kernel, bias

    def forward(self, x: Tensor) -> Tensor:
        parameter_dtype = self.direct.expand.weight.dtype
        fold_dtype = (
            torch.float64 if parameter_dtype == torch.float64 else torch.float32
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            weight, bias = self.equivalent_kernel(fold_dtype)
        return F.conv2d(
            x,
            weight.to(parameter_dtype),
            None if bias is None else bias.to(parameter_dtype),
            padding=1,
        )

    @torch.no_grad()
    def folded(self) -> nn.Conv2d:
        parameter = self.direct.expand.weight
        module = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            3,
            padding=1,
            bias=self.direct.reduce.bias is not None,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        with torch.autocast(device_type=parameter.device.type, enabled=False):
            weight, bias = self.equivalent_kernel(torch.float64)
        module.weight.copy_(weight)
        if bias is not None:
            module.bias.copy_(bias)
        return module.eval()


def _check_experts(experts: int) -> None:
    if type(experts) is not int or experts < 2:
        raise ValueError("dynamic_experts must be an integer >= 2 (not bool)")


class DynamicRouter(nn.Module):
    """FP32/FP64 per-image GAP/LayerNorm/MLP routing, centred around 1/K."""

    def __init__(
        self,
        channels: int,
        experts: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _check_experts(experts)
        self.channels, self.num_experts = channels, experts
        self.fc1 = nn.Linear(channels, channels // 4, device=device, dtype=dtype)
        self.fc2 = nn.Linear(channels // 4, experts, device=device, dtype=dtype)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        dtype = (
            torch.float64 if self.fc1.weight.dtype == torch.float64 else torch.float32
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            pooled = x.mean((2, 3), dtype=dtype)
            normalized = F.layer_norm(pooled, (self.channels,), eps=1e-5)
            hidden = F.gelu(
                F.linear(normalized, self.fc1.weight.to(dtype), self.fc1.bias.to(dtype))
            )
            logits = F.linear(
                hidden, self.fc2.weight.to(dtype), self.fc2.bias.to(dtype)
            )
            return (logits.softmax(-1) - 1.0 / self.num_experts)[:, :, None, None]


def _premixed_conv3x3(
    x: Tensor, weight0: Tensor, bias0: Tensor, experts: Tensor, coefficients: Tensor
) -> Tensor:
    """One differentiable mixed kernel per sample, including batch > 1."""
    batch, channels, height, width = x.shape
    dtype = torch.float64 if experts.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        residual = coefficients.reshape(batch, experts.shape[0]).to(dtype) @ (
            experts.reshape(experts.shape[0], -1).to(dtype)
        )
        weight = (weight0.to(dtype).reshape(1, -1) + residual).reshape(
            batch * channels, channels, 3, 3
        )
        bias = bias0.to(dtype).repeat(batch)
    weight, bias = weight.to(experts.dtype), bias.to(experts.dtype)
    if batch == 1:
        return F.conv2d(x, weight, bias, padding=1)
    output = F.conv2d(
        x.reshape(1, batch * channels, height, width),
        weight,
        bias,
        padding=1,
        groups=batch,
    ).reshape(batch, channels, height, width)
    if x.is_contiguous(memory_format=torch.channels_last):
        output = output.contiguous(memory_format=torch.channels_last)
    return output


class PremixedDynamicGroupConv3x3(nn.Module):
    """Unchanged EDBB base plus K learnable, input-routed residual kernels."""

    def __init__(self, base: EDBBReparamConv3x3, experts: int = 4) -> None:
        super().__init__()
        _check_experts(experts)
        if not isinstance(base, EDBBReparamConv3x3):
            raise TypeError("A dynamic group convolution requires an EDBB base")
        if base.in_channels != base.out_channels or base.direct.reduce.bias is None:
            raise ValueError(
                "A dynamic group convolution requires a square biased base"
            )
        self.base = base
        parameter = base.direct.expand.weight
        factory = {"device": parameter.device, "dtype": parameter.dtype}
        channels = base.in_channels
        self.experts = nn.Parameter(
            torch.zeros(experts, channels, channels * 9, **factory)
        )
        self.router = DynamicRouter(channels, experts, **factory)
        self.train(base.training)

    def coefficients(self, x: Tensor) -> Tensor:
        return self.router(x)

    def forward(self, x: Tensor) -> Tensor:
        dtype = torch.float64 if self.experts.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            weight, bias = self.base.equivalent_kernel(dtype)
        return _premixed_conv3x3(x, weight, bias, self.experts, self.coefficients(x))

    @torch.no_grad()
    def folded(self) -> PremixedDeployedDynamicConv3x3:
        return PremixedDeployedDynamicConv3x3(
            self.base.folded(), self.experts, self.router
        ).eval()


class PremixedDeployedDynamicConv3x3(nn.Module):
    """Static base folded; routing is recomputed for every input."""

    def __init__(self, base: nn.Conv2d, experts: Tensor, router: DynamicRouter) -> None:
        super().__init__()
        self.base = base
        self.experts = nn.Parameter(experts.detach().clone())
        self.router = deepcopy(router)

    def coefficients(self, x: Tensor) -> Tensor:
        return self.router(x)

    def forward(self, x: Tensor) -> Tensor:
        return _premixed_conv3x3(
            x, self.base.weight, self.base.bias, self.experts, self.coefficients(x)
        )


class _Layer(nn.Module):
    """Original token-normalized spatial attention and convolutional FFN."""

    def __init__(
        self,
        channels: int,
        window: int,
        *,
        shifted: bool,
        pos_bias: str,
        qk: QKMode,
        rank: int,
        g_attention_impl: Literal["sdpa", "direct"],
    ) -> None:
        super().__init__()
        # Match the source construction order, not just its forward equation.
        self.norm1 = TokenNorm(channels, 1e-5)
        self.norm2 = TokenNorm(channels, 1e-5)
        self.attention = _SpatialAttention(
            channels,
            channels // 32,
            window,
            shifted=shifted,
            pos_bias=pos_bias,
            window_mode="anchored",
            qk=qk,
            rank=rank,
            g_attention_impl=g_attention_impl,
        )
        self.ffn = ConvFFN(channels)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class _Group(nn.Module):
    """Six layers and a group residual; optional activation checkpointing."""

    def __init__(
        self,
        channels: int,
        c_windows: tuple[int, int],
        *,
        c_pos_bias: str,
        use_checkpoint: bool,
        g_window: int,
        qk: QKMode,
        rank: int,
        g_attention_impl: Literal["sdpa", "direct"],
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        schedule = (
            (g_window, False, "rpb"),
            (g_window, True, "rpb"),
            (c_windows[0], False, c_pos_bias),
            (g_window, False, "rpb"),
            (g_window, True, "rpb"),
            (c_windows[1], False, c_pos_bias),
        )
        self.layers = nn.ModuleList(
            _Layer(
                channels,
                window,
                shifted=shifted,
                pos_bias=bias,
                qk=qk,
                rank=rank,
                g_attention_impl=g_attention_impl,
            )
            for window, shifted, bias in schedule
        )
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        for layer in self.layers:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return residual + self.conv(x)


class DUAL(nn.Module):
    """DUAL3X with explicit original-index schedules and optional RGB PU2.

    RGB BCHW -> RGB B,C,scale*H,scale*W. The shared two-local-layer prefix,
    retrieval after group two and F64 reconstruction are retained. Static
    factor folding is destructive and terminal; use an evaluation copy.
    """

    def __init__(
        self,
        scale: int = 4,
        *,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (6, 6, 6, 6),
        num_heads: tuple[int, ...] | None = None,
        layer_indices: tuple[tuple[int, ...], ...] | None = None,
        pixel_unshuffle: bool = False,
        dynamic_experts: int = 4,
        c_windows: tuple[int, int] = (32, 64),
        g_window: int = 16,
        norm: Literal["token"] = "token",
        c_pos_bias: str = "rfb",
        rfb_rank: int = 16,
        window_mode: WindowMode = "anchored",
        qk: QKMode = "dot",
        ffn: Literal["conv"] = "conv",
        use_checkpoint: bool = False,
        dual_core_checkpoint: bool = True,
        g_attention_impl: Literal["sdpa", "direct"] = "sdpa",
        compile_g_attention_impl: Literal["sdpa", "direct"] = "direct",
    ) -> None:
        super().__init__()
        _check_experts(dynamic_experts)
        if type(pixel_unshuffle) is not bool:
            raise ValueError("pixel_unshuffle must be a bool")
        if scale not in (1, 2, 4) or norm != "token":
            raise ValueError("DUAL requires scale 1/2/4 and token normalization")
        if not isinstance(embed_dim, int) or embed_dim < 32 or embed_dim % 32:
            raise ValueError("embed_dim must be a positive multiple of 32")
        depths = tuple(depths)
        if len(depths) < 2 or any(
            type(d) is not int or not 1 <= d <= 6 for d in depths
        ):
            raise ValueError("DUAL requires at least two groups, with depths in 1..6")
        if layer_indices is None:
            if any(d != 6 for d in depths):
                raise ValueError("Pruned groups require explicit layer_indices")
            layer_indices = ((0, 1, 2, 3, 4, 5),) * len(depths)
        selected = tuple(tuple(indices) for indices in layer_indices)
        if len(selected) != len(depths) or any(
            len(indices) != depth
            or any(type(i) is not int or not 0 <= i < 6 for i in indices)
            or tuple(sorted(set(indices))) != indices
            for indices, depth in zip(selected, depths, strict=True)
        ):
            raise ValueError(
                "layer_indices must be ordered unique original indices matching depths"
            )
        if selected[0][:2] != (0, 1):
            raise ValueError("The first group must retain shared local prefix (0, 1)")
        expected_heads = (embed_dim // 32,) * len(depths)
        num_heads = expected_heads if num_heads is None else tuple(num_heads)
        if num_heads != expected_heads:
            raise ValueError("num_heads must provide embed_dim//32 for each group")
        if window_mode != "anchored":
            raise ValueError("DUAL requires anchored spatial windows")
        if ffn != "conv":
            raise ValueError("DUAL requires the original convolutional FFN")
        if len(c_windows) != 2 or min(c_windows) < 1:
            raise ValueError("two positive context windows required")
        if not isinstance(g_window, int) or g_window < 2 or g_window % 2:
            raise ValueError("g_window must be a positive even integer")
        if c_pos_bias not in ("rfb", "rib", "none"):
            raise ValueError("c_pos_bias must be rfb/rib/none")
        if compile_g_attention_impl not in ("sdpa", "direct"):
            raise ValueError("compile_g_attention_impl must be sdpa/direct")
        self.compile_g_attention_impl = compile_g_attention_impl
        self.scale = scale
        self.unshuffle = pixel_unshuffle
        self.internal_scale = scale * (2 if pixel_unshuffle else 1)
        self.selected_original_indices = selected
        self.embed_dim, self.depths, self.num_heads = embed_dim, depths, num_heads
        self.register_buffer(
            "mean", torch.tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1)
        )
        channels, groups, tail_channels = embed_dim, len(depths), 64
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.groups = nn.Sequential(
            *(
                _Group(
                    channels,
                    c_windows,
                    c_pos_bias=c_pos_bias,
                    use_checkpoint=use_checkpoint,
                    g_window=g_window,
                    qk=qk,
                    rank=rfb_rank,
                    g_attention_impl=g_attention_impl,
                )
                for _ in range(groups)
            )
        )
        self.final_norm = TokenNorm(channels, 1e-5)
        self.trunk_end = nn.Conv2d(channels, channels, 3, padding=1)
        tail: list[nn.Module] = [
            nn.Conv2d(channels, tail_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
        ]
        stages = (2, 2) if scale == 4 else ((2,) if scale == 2 else ())
        for stage in stages:
            tail.extend(
                (
                    nn.Conv2d(tail_channels, stage**2 * tail_channels, 3, padding=1),
                    nn.PixelShuffle(stage),
                )
            )
        tail.append(nn.Conv2d(tail_channels, 3, 3, padding=1))
        self.tail = nn.Sequential(*tail)
        self.apply(_init_weights)

        # Preserve the full native C96 DUAL3X construction/RNG order.
        self.xr = CrossScaleRetrieval(channels, include_hr=self.internal_scale >= 2)
        self.xg = SecondOctaveTransfer(channels) if self.internal_scale >= 4 else None
        for group in self.groups:
            for index in (2, 5):
                group.layers[index] = DualContextLayer(
                    group.layers[index], core_checkpoint=dual_core_checkpoint
                )
        for group, indices in zip(self.groups, selected, strict=True):
            group.layers = nn.ModuleList(group.layers[i] for i in indices)
        if self.unshuffle:
            self.stem = nn.Conv2d(12, channels, 3, padding=1)
            self.tail = nn.Sequential(
                *list(self.tail.children())[:-1],
                nn.Conv2d(64, 256, 3, padding=1),
                nn.PixelShuffle(2),
                self.tail[-1],
            )
            if self.internal_scale == 8:
                self.xg.out = nn.Conv2d(channels // 16, 12, 3, padding=1, bias=False)
                nn.init.zeros_(self.xg.out.weight)
        targets = [
            (name, module)
            for name, module in self.named_modules()
            if isinstance(module, nn.Conv2d)
            and module.groups == 1
            and module.kernel_size == (3, 3)
        ]
        self.reparameterized_names = tuple(name for name, _ in targets)
        for name, module in targets:
            parent, _, leaf = name.rpartition(".")
            setattr(self.get_submodule(parent), leaf, EDBBReparamConv3x3(module))
        self.deployed = False

        self.dynamic_conv_names = tuple(f"groups.{i}.conv" for i in range(groups))
        with torch.random.fork_rng(devices=[]):
            for name in self.dynamic_conv_names:
                parent, _, leaf = name.rpartition(".")
                setattr(
                    self.get_submodule(parent),
                    leaf,
                    PremixedDynamicGroupConv3x3(
                        self.get_submodule(name), dynamic_experts
                    ),
                )

    def forward(self, x: Tensor) -> Tensor:
        original_h, original_w = x.shape[-2:]
        if self.unshuffle:
            ph, pw = (-original_h) % 4, (-original_w) % 4
            if ph or pw:
                mode = "reflect" if original_h > ph and original_w > pw else "replicate"
                x = F.pad(x, (0, pw, 0, ph), mode=mode)
        mean = self.mean.to(dtype=x.dtype)
        shifted = x - mean
        stem = self.stem(F.pixel_unshuffle(shifted, 2) if self.unshuffle else shifted)
        h = stem
        group = self.groups[0]
        check = group.use_checkpoint and self.training and torch.is_grad_enabled()
        fine = h
        for index, layer in enumerate(group.layers):
            h = checkpoint(layer, h, use_reentrant=False) if check else layer(h)
            if index == 1:
                fine = h
        h = stem + group.conv(h)
        h = self.groups[1](h)
        coarse_input = self.xr.downsample(shifted)
        coarse = self.stem(
            F.pixel_unshuffle(coarse_input, 2) if self.unshuffle else coarse_input
        )
        for layer in group.layers[:2]:
            coarse = (
                checkpoint(layer, coarse, use_reentrant=False)
                if check
                else layer(coarse)
            )
        height, width = h.shape[-2:]
        u = self.xr.norm_u(pad_even(h))
        q = self.xr.query(torch.cat((self.xr.norm_f(pad_even(fine)), u), 1))
        k = self.xr.key(self.xr.norm_f(coarse))
        packed = pack_qk(q, k)
        o = attend_packed(packed, pack_children(self.xr.value(u)), self.xr.log_tau)
        lr = self.xr.project(o * self.xr.gate(u).sigmoid())[:, :, :height, :width]
        hr = (
            self.xr.hr(unpack_children(o))[:, :, : 2 * height, : 2 * width]
            if self.internal_scale >= 2
            else None
        )
        h = h + lr
        for group in self.groups[2:]:
            h = group(h)
        features = stem + self.trunk_end(self.final_norm(h))
        if self.internal_scale == 1:
            out = self.tail(features)
        else:
            t2 = self.tail[:4](features) + hr
            out = self.tail[4:](t2)
            if self.internal_scale >= 4:
                extra = self.xg(t2, packed, (height, width))
                out = out + (
                    F.pixel_shuffle(extra, 2) if self.internal_scale == 8 else extra
                )
        out = (
            out + F.interpolate(shifted, scale_factor=self.scale, mode="nearest") + mean
        )
        return out[:, :, : original_h * self.scale, : original_w * self.scale]

    def prepare_for_compile(
        self,
        input_shape: tuple[int, ...] | None = None,
        input_dtype: torch.dtype | None = None,
    ) -> None:
        """Select the opt-in G16/G32 direct implementation on the compile path.

        No caches need priming. Eager attention defaults to SDPA; export copies
        remain on SDPA. Per-call device/dtype/geometry checks govern direct use.
        """
        if (
            self.deployed
            or self.compile_g_attention_impl != "direct"
            or input_dtype
            not in (
                torch.bfloat16,
                torch.float16,
            )
        ):
            return
        for layer in self.modules():
            if (
                isinstance(layer, _SpatialAttention)
                and layer.pos_bias == "rpb"
                and layer.window_mode == "anchored"
                and layer.window in (16, 32)
            ):
                layer.g_attention_impl = "direct"

    def prepare_for_export(self) -> None:
        """Select portable SDPA and fold factors; call only on an eval COPY."""
        if self.training:
            raise RuntimeError("Call eval() on an export copy before folding DUAL")
        for layer in self.modules():
            if isinstance(layer, _SpatialAttention):
                layer.g_attention_impl = "sdpa"
        if self.deployed:
            return
        for name in self.reparameterized_names:
            parent, _, leaf = name.rpartition(".")
            module = self.get_submodule(name)
            setattr(self.get_submodule(parent), leaf, module.folded())
        self.deployed = True


# =============================================================================
# Model Factory Functions
# Explicit schedules retain both context scales and the shared local prefix.
# =============================================================================

_LIGHT_INDICES = ((0, 1, 2, 3), (3, 4, 5), (0, 1, 2), (0, 1, 2, 3, 4, 5))
# Native x4 C128: 57.23ms median at static B1/LR512 on the recorded RTX5090
# TensorRT11.2 Opt3 BF16/FP32 CUDA-Graph contract. Not a universal timing promise.
_XS_INDICES = _LIGHT_INDICES


def dual_light(scale: int = 4, use_checkpoint: bool = False, **kwargs: Any) -> DUAL:
    """PU2 C128/H4/F64, four groups with 16 selected DUAL3X blocks."""
    return DUAL(
        scale=scale,
        embed_dim=128,
        depths=tuple(map(len, _LIGHT_INDICES)),
        layer_indices=_LIGHT_INDICES,
        pixel_unshuffle=True,
        use_checkpoint=use_checkpoint,
        **kwargs,
    )


def dual_xs(scale: int = 4, use_checkpoint: bool = False, **kwargs: Any) -> DUAL:
    """Native C128/H4/F64, depths(4,3,3,6); measured XS target preset."""
    return DUAL(
        scale=scale,
        embed_dim=128,
        depths=tuple(map(len, _XS_INDICES)),
        layer_indices=_XS_INDICES,
        use_checkpoint=use_checkpoint,
        **kwargs,
    )


def dual_s(scale: int = 4, use_checkpoint: bool = False, **kwargs: Any) -> DUAL:
    """C160/G6xD6/H5/F64 DUAL3X."""
    return DUAL(
        scale=scale,
        embed_dim=160,
        depths=(6,) * 6,
        use_checkpoint=use_checkpoint,
        **kwargs,
    )


def dual_m(scale: int = 4, use_checkpoint: bool = False, **kwargs: Any) -> DUAL:
    """C192/G6xD6/H6/F64 DUAL3X."""
    return DUAL(
        scale=scale,
        embed_dim=192,
        depths=(6,) * 6,
        use_checkpoint=use_checkpoint,
        **kwargs,
    )


def dual_l(scale: int = 4, use_checkpoint: bool = False, **kwargs: Any) -> DUAL:
    """C224/G10xD6/H7/F64 DUAL3X."""
    return DUAL(
        scale=scale,
        embed_dim=224,
        depths=(6,) * 10,
        use_checkpoint=use_checkpoint,
        **kwargs,
    )


def dual_xl(scale: int = 4, use_checkpoint: bool = False, **kwargs: Any) -> DUAL:
    """C256/G12xD6/H8/F64 DUAL3X."""
    return DUAL(
        scale=scale,
        embed_dim=256,
        depths=(6,) * 12,
        use_checkpoint=use_checkpoint,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Bundled DRFT-derived dense-RPB Triton attention, used by direct G16/G32.
# Kernel math, launch settings, layout, custom autograd and table convention
# are retained; only the torch.library registration namespace is renamed.
# Optional import keeps the reference usable on CPU-only PyTorch installations.
# ---------------------------------------------------------------------------
_DIRECT_ATTENTION_ERROR: str | None = None
try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as error:
    if error.name not in {"triton", "triton.language"}:
        raise
    _DIRECT_ATTENTION_ERROR = f"missing optional dependency: {error.name}"
else:
    _missing_torch_apis = [
        name
        for name in ("triton_op", "wrap_triton", "register_autograd")
        if not callable(getattr(torch.library, name, None))
    ]
    if _missing_torch_apis:
        _DIRECT_ATTENTION_ERROR = "missing torch.library API(s): " + ", ".join(
            _missing_torch_apis
        )
    else:
        from torch.library import triton_op, wrap_triton

_HAS_DIRECT_ATTENTION = _DIRECT_ATTENTION_ERROR is None

if _HAS_DIRECT_ATTENTION:
    _BF16 = 1
    _FP16 = 2

    _FWD_BLOCK_M = 64
    _FWD_BLOCK_N = 64
    _FWD_WARPS = 4
    _FWD_STAGES = 3
    _DQ_BLOCK_M = 16
    _DQ_BLOCK_N = 64
    _DQ_WARPS = 4
    _DQ_STAGES = 3
    _DKV_BLOCK_M = 64
    _DKV_BLOCK_N = 64
    _DKV_WARPS = 4
    _DKV_STAGES = 3
    _DBIAS_BLOCK_M = 32
    _DBIAS_BLOCK_N = 64
    _DBIAS_WARPS = 4
    _DBIAS_STAGES = 3
    _DELTA_WARPS = 8
    _REDUCE_WARPS = 8

    def ocab_attention_dtype_code(dtype: torch.dtype) -> int:
        if dtype is torch.bfloat16:
            return _BF16
        if dtype is torch.float16:
            return _FP16
        raise ValueError(f"unsupported fused OCAB attention dtype: {dtype}")

    @triton.jit
    def _rounded_bias(value, DTYPE_CODE: tl.constexpr):
        if DTYPE_CODE == 1:
            return value.to(tl.bfloat16).to(tl.float32)
        return value.to(tl.float16).to(tl.float32)

    @triton.jit
    def _attention_forward_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        table_ptr,
        output_ptr,
        lse_ptr,
        query_stride_batch,
        query_stride_head,
        query_stride_token,
        query_stride_channel,
        key_stride_batch,
        key_stride_head,
        key_stride_token,
        key_stride_channel,
        value_stride_batch,
        value_stride_head,
        value_stride_token,
        value_stride_channel,
        output_stride_batch,
        output_stride_head,
        output_stride_token,
        output_stride_channel,
        table_stride_relative,
        table_stride_head,
        scale,
        query_tokens: tl.constexpr,
        key_tokens: tl.constexpr,
        query_height: tl.constexpr,
        query_width: tl.constexpr,
        key_width: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        DTYPE_CODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // num_heads
        head = batch_head - batch * num_heads

        query_index = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        channel = tl.arange(0, BLOCK_D)
        query_valid = query_index < query_tokens
        channel_valid = channel < head_dim

        query_offsets = (
            batch * query_stride_batch
            + head * query_stride_head
            + query_index[:, None] * query_stride_token
            + channel[None, :] * query_stride_channel
        )
        query = tl.load(
            query_ptr + query_offsets,
            mask=query_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )

        row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        query_y = query_index // query_width
        query_x = query_index - query_y * query_width
        relative_width = query_width + key_width - 1

        for key_start in range(0, key_tokens, BLOCK_N):
            key_index = key_start + tl.arange(0, BLOCK_N)
            key_valid = key_index < key_tokens
            key_offsets = (
                batch * key_stride_batch
                + head * key_stride_head
                + key_index[:, None] * key_stride_token
                + channel[None, :] * key_stride_channel
            )
            key = tl.load(
                key_ptr + key_offsets,
                mask=key_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            score = tl.dot(query, tl.trans(key)) * scale.to(tl.float32)

            key_y = key_index // key_width
            key_x = key_index - key_y * key_width
            relative_y = key_y[None, :] - query_y[:, None] + query_height - 1
            relative_x = key_x[None, :] - query_x[:, None] + query_width - 1
            relative_index = relative_y * relative_width + relative_x
            bias = tl.load(
                table_ptr
                + relative_index * table_stride_relative
                + head * table_stride_head,
                mask=query_valid[:, None] & key_valid[None, :],
                other=0.0,
            )
            score += _rounded_bias(bias, DTYPE_CODE)
            score = tl.where(
                query_valid[:, None] & key_valid[None, :],
                score,
                -float("inf"),
            )

            block_max = tl.max(score, axis=1)
            new_max = tl.maximum(row_max, block_max)
            probability = tl.exp(score - new_max[:, None])
            correction = tl.exp(row_max - new_max)
            block_sum = tl.sum(probability, axis=1)
            row_sum = row_sum * correction + block_sum
            accumulator *= correction[:, None]

            value_offsets = (
                batch * value_stride_batch
                + head * value_stride_head
                + key_index[:, None] * value_stride_token
                + channel[None, :] * value_stride_channel
            )
            value = tl.load(
                value_ptr + value_offsets,
                mask=key_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            if DTYPE_CODE == 1:
                accumulator += tl.dot(probability.to(tl.bfloat16), value)
            else:
                accumulator += tl.dot(probability.to(tl.float16), value)
            row_max = new_max

        accumulator /= row_sum[:, None]
        output_offsets = (
            batch * output_stride_batch
            + head * output_stride_head
            + query_index[:, None] * output_stride_token
            + channel[None, :] * output_stride_channel
        )
        tl.store(
            output_ptr + output_offsets,
            accumulator,
            mask=query_valid[:, None] & channel_valid[None, :],
        )
        lse_offset = (batch * num_heads + head) * query_tokens + query_index
        tl.store(
            lse_ptr + lse_offset,
            row_max + tl.log(row_sum),
            mask=query_valid,
        )

    @triton.jit
    def _attention_delta_kernel(
        output_ptr,
        grad_output_ptr,
        delta_ptr,
        output_stride_batch,
        output_stride_head,
        output_stride_token,
        output_stride_channel,
        grad_stride_batch,
        grad_stride_head,
        grad_stride_token,
        grad_stride_channel,
        query_tokens: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_index = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // num_heads
        head = batch_head - batch * num_heads
        channel = tl.arange(0, BLOCK_D)
        valid = channel < head_dim
        output_offset = (
            batch * output_stride_batch
            + head * output_stride_head
            + query_index * output_stride_token
            + channel * output_stride_channel
        )
        grad_offset = (
            batch * grad_stride_batch
            + head * grad_stride_head
            + query_index * grad_stride_token
            + channel * grad_stride_channel
        )
        output = tl.load(output_ptr + output_offset, mask=valid, other=0.0).to(
            tl.float32
        )
        grad_output = tl.load(
            grad_output_ptr + grad_offset,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        delta = tl.sum(output * grad_output, axis=0)
        tl.store(delta_ptr + batch_head * query_tokens + query_index, delta)

    @triton.jit
    def _attention_backward_query_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        table_ptr,
        grad_output_ptr,
        lse_ptr,
        delta_ptr,
        grad_query_ptr,
        query_stride_batch,
        query_stride_head,
        query_stride_token,
        query_stride_channel,
        key_stride_batch,
        key_stride_head,
        key_stride_token,
        key_stride_channel,
        value_stride_batch,
        value_stride_head,
        value_stride_token,
        value_stride_channel,
        grad_output_stride_batch,
        grad_output_stride_head,
        grad_output_stride_token,
        grad_output_stride_channel,
        grad_query_stride_batch,
        grad_query_stride_head,
        grad_query_stride_token,
        grad_query_stride_channel,
        table_stride_relative,
        table_stride_head,
        scale,
        query_tokens: tl.constexpr,
        key_tokens: tl.constexpr,
        query_height: tl.constexpr,
        query_width: tl.constexpr,
        key_width: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        DTYPE_CODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // num_heads
        head = batch_head - batch * num_heads
        query_index = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        channel = tl.arange(0, BLOCK_D)
        query_valid = query_index < query_tokens
        channel_valid = channel < head_dim

        query_offset = (
            batch * query_stride_batch
            + head * query_stride_head
            + query_index[:, None] * query_stride_token
            + channel[None, :] * query_stride_channel
        )
        query = tl.load(
            query_ptr + query_offset,
            mask=query_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        grad_output_offset = (
            batch * grad_output_stride_batch
            + head * grad_output_stride_head
            + query_index[:, None] * grad_output_stride_token
            + channel[None, :] * grad_output_stride_channel
        )
        grad_output = tl.load(
            grad_output_ptr + grad_output_offset,
            mask=query_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        lse = tl.load(
            lse_ptr + batch_head * query_tokens + query_index,
            mask=query_valid,
            other=0.0,
        )
        delta = tl.load(
            delta_ptr + batch_head * query_tokens + query_index,
            mask=query_valid,
            other=0.0,
        )
        grad_query = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        query_y = query_index // query_width
        query_x = query_index - query_y * query_width
        relative_width = query_width + key_width - 1

        for key_start in range(0, key_tokens, BLOCK_N):
            key_index = key_start + tl.arange(0, BLOCK_N)
            key_valid = key_index < key_tokens
            key_offset = (
                batch * key_stride_batch
                + head * key_stride_head
                + key_index[:, None] * key_stride_token
                + channel[None, :] * key_stride_channel
            )
            key = tl.load(
                key_ptr + key_offset,
                mask=key_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            value_offset = (
                batch * value_stride_batch
                + head * value_stride_head
                + key_index[:, None] * value_stride_token
                + channel[None, :] * value_stride_channel
            )
            value = tl.load(
                value_ptr + value_offset,
                mask=key_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )

            score = tl.dot(query, tl.trans(key)) * scale.to(tl.float32)
            key_y = key_index // key_width
            key_x = key_index - key_y * key_width
            relative_y = key_y[None, :] - query_y[:, None] + query_height - 1
            relative_x = key_x[None, :] - query_x[:, None] + query_width - 1
            relative_index = relative_y * relative_width + relative_x
            bias = tl.load(
                table_ptr
                + relative_index * table_stride_relative
                + head * table_stride_head,
                mask=query_valid[:, None] & key_valid[None, :],
                other=0.0,
            )
            score += _rounded_bias(bias, DTYPE_CODE)
            probability = tl.exp(score - lse[:, None])
            probability = tl.where(
                query_valid[:, None] & key_valid[None, :],
                probability,
                0.0,
            )
            grad_probability = tl.dot(grad_output, tl.trans(value))
            grad_score = probability * (grad_probability - delta[:, None])
            if DTYPE_CODE == 1:
                grad_query += tl.dot(grad_score.to(tl.bfloat16), key) * scale.to(
                    tl.float32
                )
            else:
                grad_query += tl.dot(grad_score.to(tl.float16), key) * scale.to(
                    tl.float32
                )

        grad_query_offset = (
            batch * grad_query_stride_batch
            + head * grad_query_stride_head
            + query_index[:, None] * grad_query_stride_token
            + channel[None, :] * grad_query_stride_channel
        )
        tl.store(
            grad_query_ptr + grad_query_offset,
            grad_query,
            mask=query_valid[:, None] & channel_valid[None, :],
        )

    @triton.jit
    def _attention_backward_key_value_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        table_ptr,
        grad_output_ptr,
        lse_ptr,
        delta_ptr,
        grad_key_ptr,
        grad_value_ptr,
        query_stride_batch,
        query_stride_head,
        query_stride_token,
        query_stride_channel,
        key_stride_batch,
        key_stride_head,
        key_stride_token,
        key_stride_channel,
        value_stride_batch,
        value_stride_head,
        value_stride_token,
        value_stride_channel,
        grad_output_stride_batch,
        grad_output_stride_head,
        grad_output_stride_token,
        grad_output_stride_channel,
        grad_key_stride_batch,
        grad_key_stride_head,
        grad_key_stride_token,
        grad_key_stride_channel,
        grad_value_stride_batch,
        grad_value_stride_head,
        grad_value_stride_token,
        grad_value_stride_channel,
        table_stride_relative,
        table_stride_head,
        scale,
        query_tokens: tl.constexpr,
        key_tokens: tl.constexpr,
        query_height: tl.constexpr,
        query_width: tl.constexpr,
        key_width: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        DTYPE_CODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        key_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // num_heads
        head = batch_head - batch * num_heads
        key_index = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
        channel = tl.arange(0, BLOCK_D)
        key_valid = key_index < key_tokens
        channel_valid = channel < head_dim

        key_offset = (
            batch * key_stride_batch
            + head * key_stride_head
            + key_index[:, None] * key_stride_token
            + channel[None, :] * key_stride_channel
        )
        key = tl.load(
            key_ptr + key_offset,
            mask=key_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        value_offset = (
            batch * value_stride_batch
            + head * value_stride_head
            + key_index[:, None] * value_stride_token
            + channel[None, :] * value_stride_channel
        )
        value = tl.load(
            value_ptr + value_offset,
            mask=key_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        grad_key = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
        grad_value = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
        key_y = key_index // key_width
        key_x = key_index - key_y * key_width
        relative_width = query_width + key_width - 1

        for query_start in range(0, query_tokens, BLOCK_M):
            query_index = query_start + tl.arange(0, BLOCK_M)
            query_valid = query_index < query_tokens
            query_offset = (
                batch * query_stride_batch
                + head * query_stride_head
                + query_index[:, None] * query_stride_token
                + channel[None, :] * query_stride_channel
            )
            query = tl.load(
                query_ptr + query_offset,
                mask=query_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            grad_output_offset = (
                batch * grad_output_stride_batch
                + head * grad_output_stride_head
                + query_index[:, None] * grad_output_stride_token
                + channel[None, :] * grad_output_stride_channel
            )
            grad_output = tl.load(
                grad_output_ptr + grad_output_offset,
                mask=query_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            lse = tl.load(
                lse_ptr + batch_head * query_tokens + query_index,
                mask=query_valid,
                other=0.0,
            )
            delta = tl.load(
                delta_ptr + batch_head * query_tokens + query_index,
                mask=query_valid,
                other=0.0,
            )

            score = tl.dot(query, tl.trans(key)) * scale.to(tl.float32)
            query_y = query_index // query_width
            query_x = query_index - query_y * query_width
            relative_y = key_y[None, :] - query_y[:, None] + query_height - 1
            relative_x = key_x[None, :] - query_x[:, None] + query_width - 1
            relative_index = relative_y * relative_width + relative_x
            bias = tl.load(
                table_ptr
                + relative_index * table_stride_relative
                + head * table_stride_head,
                mask=query_valid[:, None] & key_valid[None, :],
                other=0.0,
            )
            score += _rounded_bias(bias, DTYPE_CODE)
            probability = tl.exp(score - lse[:, None])
            probability = tl.where(
                query_valid[:, None] & key_valid[None, :],
                probability,
                0.0,
            )
            grad_probability = tl.dot(grad_output, tl.trans(value))
            grad_score = probability * (grad_probability - delta[:, None])
            if DTYPE_CODE == 1:
                grad_key += tl.dot(
                    tl.trans(grad_score.to(tl.bfloat16)), query
                ) * scale.to(tl.float32)
                grad_value += tl.dot(tl.trans(probability.to(tl.bfloat16)), grad_output)
            else:
                grad_key += tl.dot(
                    tl.trans(grad_score.to(tl.float16)), query
                ) * scale.to(tl.float32)
                grad_value += tl.dot(tl.trans(probability.to(tl.float16)), grad_output)

        grad_key_offset = (
            batch * grad_key_stride_batch
            + head * grad_key_stride_head
            + key_index[:, None] * grad_key_stride_token
            + channel[None, :] * grad_key_stride_channel
        )
        grad_value_offset = (
            batch * grad_value_stride_batch
            + head * grad_value_stride_head
            + key_index[:, None] * grad_value_stride_token
            + channel[None, :] * grad_value_stride_channel
        )
        mask = key_valid[:, None] & channel_valid[None, :]
        tl.store(grad_key_ptr + grad_key_offset, grad_key, mask=mask)
        tl.store(grad_value_ptr + grad_value_offset, grad_value, mask=mask)

    @triton.jit
    def _attention_backward_dense_bias_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        table_ptr,
        grad_output_ptr,
        lse_ptr,
        delta_ptr,
        grad_bias_ptr,
        query_stride_batch,
        query_stride_head,
        query_stride_token,
        query_stride_channel,
        key_stride_batch,
        key_stride_head,
        key_stride_token,
        key_stride_channel,
        value_stride_batch,
        value_stride_head,
        value_stride_token,
        value_stride_channel,
        grad_output_stride_batch,
        grad_output_stride_head,
        grad_output_stride_token,
        grad_output_stride_channel,
        table_stride_relative,
        table_stride_head,
        scale,
        batch_windows: tl.constexpr,
        query_tokens: tl.constexpr,
        key_tokens: tl.constexpr,
        query_height: tl.constexpr,
        query_width: tl.constexpr,
        key_width: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        DTYPE_CODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        key_block = tl.program_id(1)
        head = tl.program_id(2)
        query_index = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        key_index = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
        channel = tl.arange(0, BLOCK_D)
        query_valid = query_index < query_tokens
        key_valid = key_index < key_tokens
        channel_valid = channel < head_dim
        query_y = query_index // query_width
        query_x = query_index - query_y * query_width
        key_y = key_index // key_width
        key_x = key_index - key_y * key_width
        relative_width = query_width + key_width - 1
        relative_y = key_y[None, :] - query_y[:, None] + query_height - 1
        relative_x = key_x[None, :] - query_x[:, None] + query_width - 1
        relative_index = relative_y * relative_width + relative_x
        bias = tl.load(
            table_ptr
            + relative_index * table_stride_relative
            + head * table_stride_head,
            mask=query_valid[:, None] & key_valid[None, :],
            other=0.0,
        )
        bias = _rounded_bias(bias, DTYPE_CODE)
        grad_bias = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for batch in range(batch_windows):
            query_offset = (
                batch * query_stride_batch
                + head * query_stride_head
                + query_index[:, None] * query_stride_token
                + channel[None, :] * query_stride_channel
            )
            key_offset = (
                batch * key_stride_batch
                + head * key_stride_head
                + key_index[:, None] * key_stride_token
                + channel[None, :] * key_stride_channel
            )
            value_offset = (
                batch * value_stride_batch
                + head * value_stride_head
                + key_index[:, None] * value_stride_token
                + channel[None, :] * value_stride_channel
            )
            grad_output_offset = (
                batch * grad_output_stride_batch
                + head * grad_output_stride_head
                + query_index[:, None] * grad_output_stride_token
                + channel[None, :] * grad_output_stride_channel
            )
            query = tl.load(
                query_ptr + query_offset,
                mask=query_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            key = tl.load(
                key_ptr + key_offset,
                mask=key_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            value = tl.load(
                value_ptr + value_offset,
                mask=key_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            grad_output = tl.load(
                grad_output_ptr + grad_output_offset,
                mask=query_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            lse = tl.load(
                lse_ptr + (batch * num_heads + head) * query_tokens + query_index,
                mask=query_valid,
                other=0.0,
            )
            row_delta = tl.load(
                delta_ptr + (batch * num_heads + head) * query_tokens + query_index,
                mask=query_valid,
                other=0.0,
            )
            score = tl.dot(query, tl.trans(key)) * scale.to(tl.float32) + bias
            probability = tl.exp(score - lse[:, None])
            probability = tl.where(
                query_valid[:, None] & key_valid[None, :],
                probability,
                0.0,
            )
            grad_probability = tl.dot(grad_output, tl.trans(value))
            grad_bias += probability * (grad_probability - row_delta[:, None])

        output_offset = (
            head * query_tokens + query_index[:, None]
        ) * key_tokens + key_index[None, :]
        tl.store(
            grad_bias_ptr + output_offset,
            grad_bias,
            mask=query_valid[:, None] & key_valid[None, :],
        )

    @triton.jit
    def _reduce_dense_bias_gradient_kernel(
        grad_bias_ptr,
        grad_table_ptr,
        query_tokens: tl.constexpr,
        key_tokens: tl.constexpr,
        query_height: tl.constexpr,
        query_width: tl.constexpr,
        key_height: tl.constexpr,
        key_width: tl.constexpr,
        relative_width: tl.constexpr,
        num_heads: tl.constexpr,
        BLOCK_QUERY: tl.constexpr,
    ):
        output_index = tl.program_id(0)
        head = output_index % num_heads
        relative_index = output_index // num_heads
        relative_y = relative_index // relative_width
        relative_x = relative_index - relative_y * relative_width
        delta_y = relative_y - (query_height - 1)
        delta_x = relative_x - (query_width - 1)

        query_index = tl.arange(0, BLOCK_QUERY)
        query_y = query_index // query_width
        query_x = query_index - query_y * query_width
        key_y = query_y + delta_y
        key_x = query_x + delta_x
        valid = (
            (query_index < query_tokens)
            & (key_y >= 0)
            & (key_y < key_height)
            & (key_x >= 0)
            & (key_x < key_width)
        )
        key_index = key_y * key_width + key_x
        dense_offset = (head * query_tokens + query_index) * key_tokens + key_index
        gradient = tl.load(
            grad_bias_ptr + dense_offset,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            grad_table_ptr + relative_index * num_heads + head,
            tl.sum(gradient, axis=0),
        )

    @triton_op("dual_standalone::ocab_dense_rpb_attention", mutates_args={})
    def ocab_dense_rpb_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        table: torch.Tensor,
        query_height: int,
        query_width: int,
        key_height: int,
        key_width: int,
        scale: float,
        dtype_code: int,
        preserve_layout: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_windows, num_heads, query_tokens, head_dim = query.shape
        if key.shape != (batch_windows, num_heads, key_height * key_width, head_dim):
            raise ValueError("fused OCAB key geometry does not match query geometry")
        if value.shape != key.shape:
            raise ValueError("fused OCAB value geometry must match key geometry")
        if query_tokens != query_height * query_width:
            raise ValueError("fused OCAB query geometry does not match token count")
        expected_relative = (query_height + key_height - 1) * (
            query_width + key_width - 1
        )
        if table.shape != (expected_relative, num_heads):
            raise ValueError("fused OCAB relative-bias table has the wrong shape")

        # Q arrives as a dense head view of token-major QKV storage.  Preserving
        # those strides makes the immediately following head->channel reshape a
        # view instead of a materialized transpose.  Keep the historical layout
        # available for controlled A/B validation.
        output = (
            torch.empty_like(query, memory_format=torch.preserve_format)
            if preserve_layout
            else torch.empty(query.shape, device=query.device, dtype=query.dtype)
        )
        lse = torch.empty(
            (batch_windows, num_heads, query_tokens),
            device=query.device,
            dtype=torch.float32,
        )
        BLOCK_D = triton.next_power_of_2(head_dim)
        grid = (
            triton.cdiv(query_tokens, _FWD_BLOCK_M),
            batch_windows * num_heads,
        )
        wrap_triton(_attention_forward_kernel)[grid](
            query,
            key,
            value,
            table,
            output,
            lse,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *output.stride(),
            *table.stride(),
            scale,
            query_tokens,
            key_height * key_width,
            query_height,
            query_width,
            key_width,
            num_heads,
            head_dim,
            dtype_code,
            BLOCK_M=_FWD_BLOCK_M,
            BLOCK_N=_FWD_BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=_FWD_WARPS,
            num_stages=_FWD_STAGES,
        )
        return output, lse

    @triton_op("dual_standalone::ocab_dense_rpb_attention_backward", mutates_args={})
    def ocab_dense_rpb_attention_backward(
        grad_output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        table: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
        query_height: int,
        query_width: int,
        key_height: int,
        key_width: int,
        scale: float,
        dtype_code: int,
        preserve_layout: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not preserve_layout:
            grad_output = grad_output.contiguous()
        batch_windows, num_heads, query_tokens, head_dim = query.shape
        key_tokens = key_height * key_width
        BLOCK_D = triton.next_power_of_2(head_dim)
        delta = torch.empty_like(lse)
        delta_grid = (query_tokens, batch_windows * num_heads)
        wrap_triton(_attention_delta_kernel)[delta_grid](
            output,
            grad_output,
            delta,
            *output.stride(),
            *grad_output.stride(),
            query_tokens,
            num_heads,
            head_dim,
            BLOCK_D=BLOCK_D,
            num_warps=_DELTA_WARPS,
        )

        # In the projection-ready mode, return gradients in the source Q/K/V
        # layouts so their inverse reshape/permute into the fused QKV-window
        # operator remains a view.  Every kernel already honors explicit strides.
        grad_query = (
            torch.empty_like(query, memory_format=torch.preserve_format)
            if preserve_layout
            else torch.empty(query.shape, device=query.device, dtype=query.dtype)
        )
        query_grid = (
            triton.cdiv(query_tokens, _DQ_BLOCK_M),
            batch_windows * num_heads,
        )
        wrap_triton(_attention_backward_query_kernel)[query_grid](
            query,
            key,
            value,
            table,
            grad_output,
            lse,
            delta,
            grad_query,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *grad_output.stride(),
            *grad_query.stride(),
            *table.stride(),
            scale,
            query_tokens,
            key_tokens,
            query_height,
            query_width,
            key_width,
            num_heads,
            head_dim,
            dtype_code,
            BLOCK_M=_DQ_BLOCK_M,
            BLOCK_N=_DQ_BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=_DQ_WARPS,
            num_stages=_DQ_STAGES,
        )

        grad_key = (
            torch.empty_like(key, memory_format=torch.preserve_format)
            if preserve_layout
            else torch.empty(key.shape, device=key.device, dtype=key.dtype)
        )
        grad_value = (
            torch.empty_like(value, memory_format=torch.preserve_format)
            if preserve_layout
            else torch.empty(value.shape, device=value.device, dtype=value.dtype)
        )
        key_value_grid = (
            triton.cdiv(key_tokens, _DKV_BLOCK_N),
            batch_windows * num_heads,
        )
        wrap_triton(_attention_backward_key_value_kernel)[key_value_grid](
            query,
            key,
            value,
            table,
            grad_output,
            lse,
            delta,
            grad_key,
            grad_value,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *grad_output.stride(),
            *grad_key.stride(),
            *grad_value.stride(),
            *table.stride(),
            scale,
            query_tokens,
            key_tokens,
            query_height,
            query_width,
            key_width,
            num_heads,
            head_dim,
            dtype_code,
            BLOCK_M=_DKV_BLOCK_M,
            BLOCK_N=_DKV_BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=_DKV_WARPS,
            num_stages=_DKV_STAGES,
        )

        grad_bias = torch.empty(
            (num_heads, query_tokens, key_tokens),
            device=query.device,
            dtype=query.dtype,
        )
        dense_bias_grid = (
            triton.cdiv(query_tokens, _DBIAS_BLOCK_M),
            triton.cdiv(key_tokens, _DBIAS_BLOCK_N),
            num_heads,
        )
        wrap_triton(_attention_backward_dense_bias_kernel)[dense_bias_grid](
            query,
            key,
            value,
            table,
            grad_output,
            lse,
            delta,
            grad_bias,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *grad_output.stride(),
            *table.stride(),
            scale,
            batch_windows,
            query_tokens,
            key_tokens,
            query_height,
            query_width,
            key_width,
            num_heads,
            head_dim,
            dtype_code,
            BLOCK_M=_DBIAS_BLOCK_M,
            BLOCK_N=_DBIAS_BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=_DBIAS_WARPS,
            num_stages=_DBIAS_STAGES,
        )

        grad_table = torch.empty_like(table, memory_format=torch.contiguous_format)
        relative_entries = table.shape[0]
        relative_width = query_width + key_width - 1
        reduction_grid = (relative_entries * num_heads,)
        wrap_triton(_reduce_dense_bias_gradient_kernel)[reduction_grid](
            grad_bias,
            grad_table,
            query_tokens,
            key_tokens,
            query_height,
            query_width,
            key_height,
            key_width,
            relative_width,
            num_heads,
            BLOCK_QUERY=triton.next_power_of_2(query_tokens),
            num_warps=_REDUCE_WARPS,
        )
        return grad_query, grad_key, grad_value, grad_table

    def _setup_context(ctx, inputs, output) -> None:
        (
            query,
            key,
            value,
            table,
            query_height,
            query_width,
            key_height,
            key_width,
            scale,
            dtype_code,
            preserve_layout,
        ) = inputs
        attention_output, lse = output
        ctx.save_for_backward(query, key, value, table, attention_output, lse)
        ctx.query_height = query_height
        ctx.query_width = query_width
        ctx.key_height = key_height
        ctx.key_width = key_width
        ctx.scale = scale
        ctx.dtype_code = dtype_code
        ctx.preserve_layout = preserve_layout

    def _backward(ctx, grad_output: torch.Tensor, grad_lse: torch.Tensor | None):
        del grad_lse
        query, key, value, table, attention_output, lse = ctx.saved_tensors
        grad_query, grad_key, grad_value, grad_table = (
            ocab_dense_rpb_attention_backward(
                grad_output,
                query,
                key,
                value,
                table,
                attention_output,
                lse,
                ctx.query_height,
                ctx.query_width,
                ctx.key_height,
                ctx.key_width,
                ctx.scale,
                ctx.dtype_code,
                ctx.preserve_layout,
            )
        )
        return (
            grad_query,
            grad_key,
            grad_value,
            grad_table,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        ocab_dense_rpb_attention,
        _backward,
        setup_context=_setup_context,
    )

    def ocab_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        table: torch.Tensor,
        query_height: int,
        query_width: int,
        key_height: int,
        key_width: int,
        scale: float,
        preserve_layout: bool = False,
    ) -> torch.Tensor:
        output, _lse = ocab_dense_rpb_attention(
            query,
            key,
            value,
            table,
            query_height,
            query_width,
            key_height,
            key_width,
            scale,
            ocab_attention_dtype_code(query.dtype),
            preserve_layout,
        )
        return output

else:

    def ocab_attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        table: Tensor,
        query_height: int,
        query_width: int,
        key_height: int,
        key_width: int,
        scale: float,
        preserve_layout: bool = False,
    ) -> Tensor:
        raise RuntimeError(
            f"DUAL direct CUDA attention is unavailable: {_DIRECT_ATTENTION_ERROR}. Use "
            "g_attention_impl='sdpa' and compile_g_attention_impl='sdpa' "
            "or provide the required Triton/PyTorch APIs."
        )


# =============================================================================
# Optional: traiNNer Registry
# =============================================================================
try:
    from traiNNer.utils.registry import ARCH_REGISTRY
except ModuleNotFoundError as error:
    # Standalone use is supported, but an unrelated broken dependency must
    # remain visible rather than silently disabling registration.
    if error.name not in {"traiNNer", "traiNNer.utils", "traiNNer.utils.registry"}:
        raise
else:
    ARCH_REGISTRY.register(dual_light)
    ARCH_REGISTRY.register(dual_xs)
    ARCH_REGISTRY.register(dual_s)
    ARCH_REGISTRY.register(dual_m)
    ARCH_REGISTRY.register(dual_l)
    ARCH_REGISTRY.register(dual_xl)
