"""Two pointwise chains of DUAL's trunk as one Triton kernel each, for the TensorRT plugins
dual_aot.lower puts in their place (static 128-channel engines only):

- chain A: attention output projection [P, 128] x [128, 128] + bias -> residual add ->
  the FFN's Norm -> FFN expand [128, 256] + bias -> GELU (erf). Writes the new residual
  (BF16, 128 channels) and the GELU output (BF16, 256 channels).
- chain C: chain A after a DualCore layer, whose output projection is DualProject's
  (no bias) instead of a matmul + bias.
- chain B: FFN contraction [P, 256] x [256, 128] + bias -> residual add -> the next
  layer's Norm. Writes the new residual and the Norm output (both BF16, 128 channels).
- chain D: the attention layers' tail: gate [P, 128] x [128, 128] + bias -> sigmoid ->
  x the attention output -> projection + bias -> residual add -> Norm -> the next
  core's qkv [128, 384] (no bias). Writes the new residual and qkv (BF16). Its gate
  pre-activation, sigmoid, gated product and projection are rounded to BF16 as the
  graph rounds them.

Both matmuls run on tl.dot with FP32 accumulation; the Norm is the Norm plugin's FP32
two-pass arithmetic. BF16 roundings are kept where the tensor exists in BF16 anyway
(projection + bias, residual sum, Norm output as the next matmul's operand, GELU and
Norm outputs); the raw matmul results before their bias and chain A's GELU
intermediates stay FP32. Chain B reproduces TensorRT's own layers bit for bit; chain A
differs from a per-operator BF16 evaluation by those skipped roundings. The 256-long
reduction of chain B runs as two 128-long steps on one accumulator, within the AOT
plugins' 48 KiB of shared memory. Arguments follow TensorRT AOT order: inputs, then
outputs, then constants."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def chain_a(
    g_ptr: tl.tensor,
    r_ptr: tl.tensor,
    w1_ptr: tl.tensor,
    b1_ptr: tl.tensor,
    gamma_ptr: tl.tensor,
    beta_ptr: tl.tensor,
    w2_ptr: tl.tensor,
    b2_ptr: tl.tensor,
    res_ptr: tl.tensor,
    out_ptr: tl.tensor,
    pixels: tl.constexpr,
    eps: tl.constexpr,
    c_div: tl.constexpr,
    c_add: tl.constexpr,
    c_mul: tl.constexpr,
    bm: tl.constexpr,
    bn: tl.constexpr,
    project: tl.constexpr,
) -> None:
    """With `project`, chain C: the projection is DualProject's (no bias; four 32-long
    products summed into the FP32 accumulator as its kernel does, then BF16), and
    b1_ptr is not read."""
    rows = tl.program_id(0) * bm + tl.arange(0, bm)
    c = tl.arange(0, 128)
    mask = (rows < pixels)[:, None]
    if project:
        ks = tl.arange(0, 32)
        acc = tl.full((bm, 128), 0, tl.float32)
        for k0 in tl.static_range(0, 128, 32):
            gk = tl.load(g_ptr + rows[:, None] * 128 + (k0 + ks)[None, :], mask, 0)
            acc += tl.dot(gk, tl.load(w1_ptr + (k0 + ks)[:, None] * 128 + c[None, :]))
        proj = acc.to(tl.bfloat16)
    else:
        g = tl.load(g_ptr + rows[:, None] * 128 + c[None, :], mask, 0)
        w1 = tl.load(w1_ptr + c[:, None] * 128 + c[None, :])
        proj = (tl.dot(g, w1) + tl.load(b1_ptr + c).to(tl.float32)[None, :]).to(
            tl.bfloat16
        )
    r = tl.load(r_ptr + rows[:, None] * 128 + c[None, :], mask, 0)
    s = (proj.to(tl.float32) + r.to(tl.float32)).to(tl.bfloat16)
    tl.store(res_ptr + rows[:, None] * 128 + c[None, :], s, mask)
    x = s.to(tl.float32)
    mean = tl.math.div_rn(tl.sum(x, 1), 128)
    centered = tl.where(mask, x - mean[:, None], 0)
    variance = tl.math.div_rn(tl.sum(centered * centered, 1), 128) + eps
    inv = tl.math.rsqrt(variance)
    y = (
        (tl.load(gamma_ptr + c)[None, :] * centered) * inv[:, None]
        + tl.load(beta_ptr + c)[None, :]
    ).to(tl.bfloat16)
    for n0 in tl.static_range(0, 256, bn):
        nn = n0 + tl.arange(0, bn)
        h = (
            tl.dot(y, tl.load(w2_ptr + c[:, None] * 256 + nn[None, :]))
            + tl.load(b2_ptr + nn).to(tl.float32)[None, :]
        )
        act = (h * c_mul) * (tl.math.erf(h / c_div) + c_add)
        tl.store(out_ptr + rows[:, None] * 256 + nn[None, :], act.to(tl.bfloat16), mask)


@triton.jit
def chain_b(
    g_ptr: tl.tensor,
    r_ptr: tl.tensor,
    w_ptr: tl.tensor,
    b_ptr: tl.tensor,
    gamma_ptr: tl.tensor,
    beta_ptr: tl.tensor,
    res_ptr: tl.tensor,
    y_ptr: tl.tensor,
    pixels: tl.constexpr,
    eps: tl.constexpr,
    bm: tl.constexpr,
) -> None:
    rows = tl.program_id(0) * bm + tl.arange(0, bm)
    c = tl.arange(0, 128)
    mask = (rows < pixels)[:, None]
    g0 = tl.load(g_ptr + rows[:, None] * 256 + c[None, :], mask, 0)
    acc = tl.dot(g0, tl.load(w_ptr + c[:, None] * 128 + c[None, :]))
    g1 = tl.load(g_ptr + rows[:, None] * 256 + 128 + c[None, :], mask, 0)
    acc = tl.dot(g1, tl.load(w_ptr + (128 + c[:, None]) * 128 + c[None, :]), acc)
    proj = (acc + tl.load(b_ptr + c).to(tl.float32)[None, :]).to(tl.bfloat16)
    r = tl.load(r_ptr + rows[:, None] * 128 + c[None, :], mask, 0)
    s = (proj.to(tl.float32) + r.to(tl.float32)).to(tl.bfloat16)
    tl.store(res_ptr + rows[:, None] * 128 + c[None, :], s, mask)
    x = s.to(tl.float32)
    mean = tl.math.div_rn(tl.sum(x, 1), 128)
    centered = tl.where(mask, x - mean[:, None], 0)
    variance = tl.math.div_rn(tl.sum(centered * centered, 1), 128) + eps
    inv = tl.math.rsqrt(variance)
    y = (tl.load(gamma_ptr + c)[None, :] * centered) * inv[:, None] + tl.load(
        beta_ptr + c
    )[None, :]
    tl.store(y_ptr + rows[:, None] * 128 + c[None, :], y.to(tl.bfloat16), mask)


@triton.jit
def chain_d(
    a_ptr: tl.tensor,
    wg_ptr: tl.tensor,
    bg_ptr: tl.tensor,
    na_ptr: tl.tensor,
    wp_ptr: tl.tensor,
    bp_ptr: tl.tensor,
    r_ptr: tl.tensor,
    gamma_ptr: tl.tensor,
    beta_ptr: tl.tensor,
    wq_ptr: tl.tensor,
    res_ptr: tl.tensor,
    qkv_ptr: tl.tensor,
    pixels: tl.constexpr,
    eps: tl.constexpr,
    bm: tl.constexpr,
) -> None:
    rows = tl.program_id(0) * bm + tl.arange(0, bm)
    c = tl.arange(0, 128)
    mask = (rows < pixels)[:, None]
    a = tl.load(a_ptr + rows[:, None] * 128 + c[None, :], mask, 0)
    pre = (
        tl.dot(a, tl.load(wg_ptr + c[:, None] * 128 + c[None, :]))
        + tl.load(bg_ptr + c).to(tl.float32)[None, :]
    )
    gate = tl.sigmoid(pre.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    na = tl.load(na_ptr + rows[:, None] * 128 + c[None, :], mask, 0)
    m = (na.to(tl.float32) * gate.to(tl.float32)).to(tl.bfloat16)
    proj = (
        tl.dot(m, tl.load(wp_ptr + c[:, None] * 128 + c[None, :]))
        + tl.load(bp_ptr + c).to(tl.float32)[None, :]
    )
    r = tl.load(r_ptr + rows[:, None] * 128 + c[None, :], mask, 0)
    s = (proj.to(tl.bfloat16).to(tl.float32) + r.to(tl.float32)).to(tl.bfloat16)
    tl.store(res_ptr + rows[:, None] * 128 + c[None, :], s, mask)
    x = s.to(tl.float32)
    mean = tl.math.div_rn(tl.sum(x, 1), 128)
    centered = tl.where(mask, x - mean[:, None], 0)
    variance = tl.math.div_rn(tl.sum(centered * centered, 1), 128) + eps
    inv = tl.math.rsqrt(variance)
    y = (
        (tl.load(gamma_ptr + c)[None, :] * centered) * inv[:, None]
        + tl.load(beta_ptr + c)[None, :]
    ).to(tl.bfloat16)
    for n0 in tl.static_range(0, 384, 128):
        nn = n0 + c
        q = tl.dot(y, tl.load(wq_ptr + c[:, None] * 384 + nn[None, :]))
        tl.store(qkv_ptr + rows[:, None] * 384 + nn[None, :], q.to(tl.bfloat16), mask)


# The fastest launch shapes inside the AOT plugins' shared-memory limit (measured in
# the engines: chain A 32 x 128, chains B and D 32; 4 warps each)
CHAIN_A_LAUNCH = {"bm": 32, "bn": 128}
CHAIN_B_LAUNCH = {"bm": 32}
CHAIN_D_LAUNCH = {"bm": 32}
OPTIONS = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
# The width the kernels are written for
CHANNELS = 128
