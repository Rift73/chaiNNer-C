"""DUAL's cross-scale retrieval (xr, xg) as full neighbourhood attention in one Triton
kernel for a TensorRT AOT plugin, built the way dual_na_kernel.na_attention_blocks is:
each query's neighbourhood is read straight from the image-layout tensors (no gathered
copies, no mask tensor), online FP32 softmax, BF16 operands and output.

The operator (the source's retrieval_na): fine query (y, x) of an H x W trunk has coarse
cell c = (y // 2, x // 2) on the Hc x Wc key grid; per axis K = min(N, Hc) and keys
start = clamp(c - N // 2, 0, Hc - K) (N = 48). No position bias. The graph scales the
query by the head's tau and log(Ky * Kx) / (log(1024) * sqrt(32)) before the node.

A program takes a qh x qw block of fine queries (qh / 2 x qw / 2 coarse cells) and one
head; its keys are the block's union, (K + qh / 2 - 1) x (K + qw / 2 - 1) cells with row
stride N + qw / 2 - 1, walked in 64-key blocks; a per-query test keeps its own K x K.
Arguments follow TensorRT AOT order: inputs, scalars, then outputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import triton
import triton.language as tl


@triton.jit
def xr_attention(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    v_ptr: tl.tensor,
    height: tl.int32,
    width: tl.int32,
    kheight: tl.int32,
    kwidth: tl.int32,
    out_ptr: tl.tensor,
    channels: tl.constexpr,
    neighbourhood: tl.constexpr,
    qh: tl.constexpr,
    qw: tl.constexpr,
    block_n: tl.constexpr,
) -> None:
    """q [1, H, W, C], k and v [1, Hc, Wc, C], out [1, H, W, C], all BF16; heads of 32."""
    block_m: tl.constexpr = qh * qw
    tile, head = tl.program_id(0), tl.program_id(1)
    oy = tile // tl.cdiv(width, qw) * qh
    ox = tile % tl.cdiv(width, qw) * qw
    m = tl.arange(0, block_m)
    qy = tl.minimum(oy + m // qw, height - 1)
    qx = tl.minimum(ox + m % qw, width - 1)
    cy, cx = qy // 2, qx // 2
    us: tl.constexpr = neighbourhood + qw // 2 - 1
    kh, kw = tl.minimum(kheight, neighbourhood), tl.minimum(kwidth, neighbourhood)
    rows = tl.minimum(kheight, kh + qh // 2 - 1)
    cols = tl.minimum(kwidth, kw + qw // 2 - 1)
    half: tl.constexpr = neighbourhood // 2
    ky0 = tl.minimum(
        tl.maximum(tl.minimum(oy, height - 1) // 2 - half, 0), kheight - rows
    )
    kx0 = tl.minimum(
        tl.maximum(tl.minimum(ox, width - 1) // 2 - half, 0), kwidth - cols
    )
    d = tl.arange(0, 32)
    n = tl.arange(0, block_n)
    pix = qy * width + qx
    q = tl.load(q_ptr + pix[:, None] * channels + head * 32 + d[None, :])
    sy = tl.minimum(tl.maximum(cy - half, 0), kheight - kh)
    sx = tl.minimum(tl.maximum(cx - half, 0), kwidth - kw)
    maximum = tl.full((block_m,), -float("inf"), tl.float32)
    denominator = tl.full((block_m,), 0, tl.float32)
    acc = tl.full((block_m, 32), 0, tl.float32)
    trips: tl.constexpr = ((neighbourhood + qh // 2 - 1) * us + block_n - 1) // block_n
    for start in range(0, trips * block_n, block_n):
        ns = start + n
        ky, kx = ky0 + ns // us, kx0 + ns % us
        inside = (ns < rows * us) & (ns % us < cols)
        kp = ky * kwidth + kx
        k = tl.load(
            k_ptr + kp[None, :] * channels + head * 32 + d[:, None], inside[None, :], 0
        )
        v = tl.load(
            v_ptr + kp[:, None] * channels + head * 32 + d[None, :], inside[:, None], 0
        )
        score = tl.dot(q, k) * 1.4426950408889634
        valid = (
            (ky[None, :] >= sy[:, None])
            & (ky[None, :] < sy[:, None] + kh)
            & (kx[None, :] >= sx[:, None])
            & (kx[None, :] < sx[:, None] + kw)
        )
        score = tl.where(valid, score, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        # A query may have no valid key yet: avoid -inf - (-inf) until its first one.
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


# 8x8 fine queries (4x4 coarse cells) per program: 16x16 and 16x8 blocks were slower
LAUNCH = {"neighbourhood": 48, "qh": 8, "qw": 8, "block_n": 64}
OPTIONS = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": True}


def grid(
    height: Any, width: Any, channels: int, cdiv: Callable[[Any, int], Any]
) -> tuple[Any, int, int]:
    """The launch grid, the image-sized count on the first axis; ints (cdiv =
    triton.cdiv) or TensorRT shape expressions."""
    return (cdiv(height, LAUNCH["qh"]) * cdiv(width, LAUNCH["qw"]), channels // 32, 1)
