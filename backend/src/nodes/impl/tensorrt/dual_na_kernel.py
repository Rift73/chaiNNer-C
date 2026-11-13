"""DUAL's neighborhood attention (NA) core as one Triton kernel for the TensorRT
plugins: exact inward-clamped neighborhoods with the learned relative bias (RFB) given as
two factor tables, an online FP32 softmax and the value product, fused. BF16 operands
and output, FP32 accumulation; no sparse keys, atomics or scratch memory.

The operator, for a query (y, x) of an H x W trunk and window M (32 or 64):
    Kh = min(H, M), sy = clamp(y - Kh // 2, 0, H - Kh), key rows [sy, sy + Kh)
(columns alike with W), so every query sees exactly Kh x Kw keys, borders shift the
neighborhood inward and an axis no longer than M sees all of it. Q is scaled by
1/sqrt(32) and rounded to BF16. The 32 content channels and 16 RFB channels use
separate dot products into one FP32 accumulator, avoiding padding 48 channels to 64.
The resulting logit is Q.K / sqrt(32) + bias(query - key).

The RFB factors are low-precision, so their coordinate frame is part of the result:
queries form 16 x 16 tiles anchored at multiples of 16; a tile's key union has
Uh = min(H, Kh + 15) rows starting at by = clamp(ty - Kh // 2, 0, H - Uh) (columns
alike), and both tables are indexed by union-local coordinates with row stride
u = M + 15. Each original 128-query parent covers eight query rows. Two 64-query
programs share that parent's key origin, min(H, Kh + 7)-row union and 64-key block
boundaries. Each child skips blocks outside its four query rows' support without
regrouping the remaining keys. A per-query mask retains exactly its neighborhood.
Lanes past the trunk's edge repeat the last valid query and are not stored.

The trunk size is a run-time argument shared by fixed and dynamic engines. The
parent key grid preserves the existing per-row reduction order; changing the key
origin with the smaller query block would change BF16 rounding. Finite activations
are required: skipping masked blocks need not preserve NaN/Inf propagation.
Arguments follow TensorRT AOT order: inputs, scalars, then outputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import triton
import triton.language as tl


@triton.jit
def na_attention(
    qkv_ptr: tl.tensor,
    bq_ptr: tl.tensor,
    bk_ptr: tl.tensor,
    height: tl.int32,
    width: tl.int32,
    out_ptr: tl.tensor,
    channels: tl.constexpr,
    window: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    minimal: tl.constexpr,
) -> None:
    """qkv [1, H, W, 3C] (all Q, all K, all V; heads of 32), factor tables
    [C / 32, (M + 15)^2, 16] and out [1, H, W, C], all BF16."""
    tl.static_assert(block_m == 64)
    tl.static_assert(block_n == 64)
    tile, sub, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ty = tile // tl.cdiv(width, 16) * 16
    tx = tile % tl.cdiv(width, 16) * 16
    m = sub * block_m + tl.arange(0, block_m)
    qy = tl.minimum(ty + m // 16, height - 1)
    qx = tl.minimum(tx + m % 16, width - 1)
    u: tl.constexpr = window + 15
    kh, kw = tl.minimum(height, window), tl.minimum(width, window)
    uh, uw = tl.minimum(height, kh + 15), tl.minimum(width, kw + 15)
    by = tl.minimum(tl.maximum(ty - kh // 2, 0), height - uh)
    bx = tl.minimum(tl.maximum(tx - kw // 2, 0), width - uw)
    if minimal:
        rows = tl.minimum(height, kh + 7)
        ky0 = tl.minimum(tl.maximum(ty + (sub // 2) * 8 - kh // 2, 0), height - rows)
    else:
        rows = uh
        ky0 = by
    d = tl.arange(0, 32)
    r = tl.arange(0, 16)
    dv = tl.arange(0, 32)
    n = tl.arange(0, block_n)
    pix = qy * width + qx
    q0 = tl.load(
        qkv_ptr + pix[:, None] * (3 * channels) + head * 32 + d[None, :],
        d[None, :] < 32,
        0,
    )
    q0 = (q0.to(tl.float32) * 0.1767766952966369).to(tl.bfloat16)
    bqi = (qy - by) * u + (qx - bx)
    bq = tl.load(
        bq_ptr + (head * u * u + bqi[:, None]) * 16 + r[None, :],
        r[None, :] < 16,
        0,
    )
    q = q0
    sy = tl.minimum(tl.maximum(qy - kh // 2, 0), height - kh)
    sx = tl.minimum(tl.maximum(qx - kw // 2, 0), width - kw)
    maximum = tl.full((block_m,), -float("inf"), tl.float32)
    denominator = tl.full((block_m,), 0, tl.float32)
    acc = tl.full((block_m, 32), 0, tl.float32)
    # Four child query rows need at most window+3 key rows, plus the leading
    # block's alignment remainder. Keep the parent's 64-key block boundaries.
    first_qy = tl.minimum(ty + sub * 4, height - 1)
    first_sy = tl.minimum(tl.maximum(first_qy - kh // 2, 0), height - kh)
    first = ((first_sy - ky0) * u // block_n) * block_n
    trips: tl.constexpr = ((window + 3) * u + 2 * block_n - 2) // block_n
    for start in range(0, trips * block_n, block_n):
        ns = first + start + n
        ky, kx = ky0 + ns // u, bx + ns % u
        # The union's row stride stays M + 15; columns past its width are not loaded.
        # Keys outside the loaded union fail every query's neighborhood test below.
        inside = (ns < rows * u) & (ns % u < uw)
        kp = ky * width + kx
        bi = (ky - by) * u + kx - bx
        k0 = tl.load(
            qkv_ptr + kp[None, :] * (3 * channels) + channels + head * 32 + d[:, None],
            (d[:, None] < 32) & inside[None, :],
            0,
        )
        bk = tl.load(
            bk_ptr + (head * u * u + bi[None, :]) * 16 + r[:, None],
            inside[None, :] & (r[:, None] < 16),
            0,
        )
        k = k0
        v = tl.load(
            qkv_ptr
            + kp[:, None] * (3 * channels)
            + 2 * channels
            + head * 32
            + dv[None, :],
            inside[:, None],
            0,
        )
        score = tl.dot(q, k)
        score = tl.dot(bq, bk, score) * 1.4426950408889634
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
    stored = (ty + m // 16 < height) & (tx + m % 16 < width)
    tl.store(
        out_ptr + pix[:, None] * channels + head * 32 + dv[None, :],
        (acc / denominator[:, None]).to(tl.bfloat16),
        stored[:, None],
    )


# Split query work, retaining the parent's key grouping and per-row reductions.
LAUNCH = {"block_m": 64, "block_n": 64, "minimal": True}
OPTIONS = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": True}


def grid(
    height: Any, width: Any, channels: int, cdiv: Callable[[Any, int], Any]
) -> tuple[Any, int, int]:
    """The launch grid; height and width are ints (cdiv = triton.cdiv) or TensorRT
    shape expressions (cdiv = tensorrt.plugin.cdiv)."""
    return (
        cdiv(height, 16) * cdiv(width, 16),
        256 // LAUNCH["block_m"],
        channels // 32,
    )


# TensorRT's kernel: the operator above with square query blocks. A program's queries
# are a qh x qw block of a 16 x 16 tile (8 x 8: four programs per tile, as above) and its
# keys only that block's own union, (Kh + qh - 1) x (Kw + qw - 1) with row stride
# window + qw - 1, walked in 64-key blocks; the RFB tables keep the tile's frame. The
# kernel is bound by matrix-multiply issue, and every 64-key block costs the same
# whether its keys are needed, so fewer blocks are faster: window 64 issues 79 instead
# of 84, window 32 24 instead of 27. The result differs from na_attention only by the
# online softmax's block grouping (another running maximum when p is rounded to BF16,
# and the FP32 summation order). MIGraphX keeps na_attention (LAUNCH, OPTIONS, grid).
@triton.jit
def na_attention_blocks(
    qkv_ptr: tl.tensor,
    bq_ptr: tl.tensor,
    bk_ptr: tl.tensor,
    height: tl.int32,
    width: tl.int32,
    out_ptr: tl.tensor,
    channels: tl.constexpr,
    window: tl.constexpr,
    qh: tl.constexpr,
    qw: tl.constexpr,
    block_n: tl.constexpr,
) -> None:
    block_m: tl.constexpr = qh * qw
    tile, sub, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ty = tile // tl.cdiv(width, 16) * 16
    tx = tile % tl.cdiv(width, 16) * 16
    oy = ty + (sub // (16 // qw)) * qh
    ox = tx + (sub % (16 // qw)) * qw
    m = tl.arange(0, block_m)
    qy = tl.minimum(oy + m // qw, height - 1)
    qx = tl.minimum(ox + m % qw, width - 1)
    u: tl.constexpr = window + 15
    us: tl.constexpr = window + qw - 1
    kh, kw = tl.minimum(height, window), tl.minimum(width, window)
    uh, uw = tl.minimum(height, kh + 15), tl.minimum(width, kw + 15)
    by = tl.minimum(tl.maximum(ty - kh // 2, 0), height - uh)
    bx = tl.minimum(tl.maximum(tx - kw // 2, 0), width - uw)
    rows = tl.minimum(height, kh + qh - 1)
    cols = tl.minimum(width, kw + qw - 1)
    ky0 = tl.minimum(tl.maximum(tl.minimum(oy, height - 1) - kh // 2, 0), height - rows)
    kx0 = tl.minimum(tl.maximum(tl.minimum(ox, width - 1) - kw // 2, 0), width - cols)
    d = tl.arange(0, 32)
    r = tl.arange(0, 16)
    n = tl.arange(0, block_n)
    pix = qy * width + qx
    q = tl.load(qkv_ptr + pix[:, None] * (3 * channels) + head * 32 + d[None, :])
    q = (q.to(tl.float32) * 0.1767766952966369).to(tl.bfloat16)
    bqi = (qy - by) * u + (qx - bx)
    bq = tl.load(bq_ptr + (head * u * u + bqi[:, None]) * 16 + r[None, :])
    sy = tl.minimum(tl.maximum(qy - kh // 2, 0), height - kh)
    sx = tl.minimum(tl.maximum(qx - kw // 2, 0), width - kw)
    maximum = tl.full((block_m,), -float("inf"), tl.float32)
    denominator = tl.full((block_m,), 0, tl.float32)
    acc = tl.full((block_m, 32), 0, tl.float32)
    trips: tl.constexpr = ((window + qh - 1) * us + block_n - 1) // block_n
    for start in range(0, trips * block_n, block_n):
        ns = start + n
        ky, kx = ky0 + ns // us, kx0 + ns % us
        inside = (ns < rows * us) & (ns % us < cols)
        kp = ky * width + kx
        bi = (ky - by) * u + kx - bx
        k = tl.load(
            qkv_ptr + kp[None, :] * (3 * channels) + channels + head * 32 + d[:, None],
            inside[None, :],
            0,
        )
        bk = tl.load(
            bk_ptr + (head * u * u + bi[None, :]) * 16 + r[:, None], inside[None, :], 0
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
        score = tl.dot(q, k)
        score = tl.dot(bq, bk, score) * 1.4426950408889634
        valid = (
            (ky[None, :] >= sy[:, None])
            & (ky[None, :] < sy[:, None] + kh)
            & (kx[None, :] >= sx[:, None])
            & (kx[None, :] < sx[:, None] + kw)
        )
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


# Swept on an RTX 5090: 8 x 8 with 4 warps; 4 x 8 with 2 warps, 4 x 4 with 1 warp and
# 8 x 16 were slower.
BLOCKS_LAUNCH = {"qh": 8, "qw": 8, "block_n": 64}
BLOCKS_OPTIONS = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": True}


def blocks_grid(
    height: Any, width: Any, channels: int, cdiv: Callable[[Any, int], Any]
) -> tuple[Any, int, int]:
    """na_attention_blocks' launch grid, as `grid`'s."""
    return (
        cdiv(height, 16) * cdiv(width, 16),
        256 // (BLOCKS_LAUNCH["qh"] * BLOCKS_LAUNCH["qw"]),
        channels // 32,
    )
