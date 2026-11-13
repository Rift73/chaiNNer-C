"""DUAL's local (window 16, dense relative position bias) attention as neighbourhood attention
(the source's local_na) in one Triton kernel for a TensorRT AOT plugin, the way dual_na_kernel.na_attention_blocks works:
the neighbourhood is read straight from the image-layout qkv tensor, no gathered copies, no mask
tensor; the bias is read from the layer's small (2K-1)^2 table inside the kernel.

The operator: query (y, x) of an H x W map attends keys start = clamp(y - K // 2, 0, H - K),
K = min(16, H) per axis; logit = (BF16(q / sqrt(32)) . k) + table[head, (yq - yk + 15) * 31 +
(xq - xk + 15)]; online FP32 softmax, BF16 P and output. A program takes a qh x qw block of
queries and one head; its keys are the block's union, (K + qh - 1) x (K + qw - 1), row stride
K + qw - 1, walked in 64-key blocks. Arguments: inputs, scalars, then outputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import triton
import triton.language as tl


@triton.jit
def local_na(
    qkv_ptr: tl.tensor,
    table_ptr: tl.tensor,
    height: tl.int32,
    width: tl.int32,
    out_ptr: tl.tensor,
    channels: tl.constexpr,
    window: tl.constexpr,
    qh: tl.constexpr,
    qw: tl.constexpr,
    block_n: tl.constexpr,
) -> None:
    """qkv [1, H, W, 3C] (all Q, all K, all V; heads of 32), table [C / 32, (2M - 1)^2] FP32,
    out [1, H, W, C] BF16."""
    block_m: tl.constexpr = qh * qw
    tile, head = tl.program_id(0), tl.program_id(1)
    oy = tile // tl.cdiv(width, qw) * qh
    ox = tile % tl.cdiv(width, qw) * qw
    m = tl.arange(0, block_m)
    qy = tl.minimum(oy + m // qw, height - 1)
    qx = tl.minimum(ox + m % qw, width - 1)
    us: tl.constexpr = window + qw - 1
    stride: tl.constexpr = 2 * window - 1
    half: tl.constexpr = window // 2
    kh, kw = tl.minimum(height, window), tl.minimum(width, window)
    rows = tl.minimum(height, kh + qh - 1)
    cols = tl.minimum(width, kw + qw - 1)
    ky0 = tl.minimum(tl.maximum(tl.minimum(oy, height - 1) - half, 0), height - rows)
    kx0 = tl.minimum(tl.maximum(tl.minimum(ox, width - 1) - half, 0), width - cols)
    d = tl.arange(0, 32)
    n = tl.arange(0, block_n)
    pix = qy * width + qx
    q = tl.load(qkv_ptr + pix[:, None] * (3 * channels) + head * 32 + d[None, :])
    q = (q.to(tl.float32) * 0.1767766952966369).to(tl.bfloat16)
    sy = tl.minimum(tl.maximum(qy - half, 0), height - kh)
    sx = tl.minimum(tl.maximum(qx - half, 0), width - kw)
    maximum = tl.full((block_m,), -float("inf"), tl.float32)
    denominator = tl.full((block_m,), 0, tl.float32)
    acc = tl.full((block_m, 32), 0, tl.float32)
    trips: tl.constexpr = ((window + qh - 1) * us + block_n - 1) // block_n
    for start in range(0, trips * block_n, block_n):
        ns = start + n
        ky, kx = ky0 + ns // us, kx0 + ns % us
        inside = (ns < rows * us) & (ns % us < cols)
        kp = ky * width + kx
        k = tl.load(
            qkv_ptr + kp[None, :] * (3 * channels) + channels + head * 32 + d[:, None],
            inside[None, :],
            0,
        )
        v = tl.load(
            qkv_ptr
            + kp[:, None] * (3 * channels)
            + 2 * channels
            + head * 32
            + d[None, :],
            inside[:, None],
            0,
        )
        valid = (
            (ky[None, :] >= sy[:, None])
            & (ky[None, :] < sy[:, None] + kh)
            & (kx[None, :] >= sx[:, None])
            & (kx[None, :] < sx[:, None] + kw)
        )
        dy = tl.minimum(
            tl.maximum(qy[:, None] - ky[None, :] + window - 1, 0), stride - 1
        )
        dx = tl.minimum(
            tl.maximum(qx[:, None] - kx[None, :] + window - 1, 0), stride - 1
        )
        bias = tl.load(table_ptr + head * stride * stride + dy * stride + dx)
        score = (tl.dot(q, k) + bias) * 1.4426950408889634
        score = tl.where(valid, score, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        p = tl.exp2(score - safe_max[:, None])
        alpha = tl.exp2(maximum - safe_max)
        acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
        denominator = denominator * alpha + tl.sum(p, 1)
        maximum = new_max
    stored = (oy + m // qw < height) & (ox + m % qw < width)
    tl.store(
        out_ptr + pix[:, None] * channels + head * 32 + d[None, :],
        (acc / denominator[:, None]).to(tl.bfloat16),
        stored[:, None],
    )


# 8x8 query blocks: measured fastest of 8x16, 4x16, 4x8 and 4x4 on the 2x DUAL Light at 960x540
LAUNCH = {"window": 16, "qh": 8, "qw": 8, "block_n": 64}
OPTIONS = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": True}


def grid(
    height: Any, width: Any, channels: int, cdiv: Callable[[Any, int], Any]
) -> tuple[Any, int, int]:
    """The launch grid, the image-sized count on the first axis; ints (cdiv =
    triton.cdiv) or TensorRT shape expressions."""
    return (cdiv(height, LAUNCH["qh"]) * cdiv(width, LAUNCH["qw"]), channels // 32, 1)
