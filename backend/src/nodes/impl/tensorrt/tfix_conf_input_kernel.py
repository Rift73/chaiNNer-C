"""Temporal Fix's confidence-head input in one pass: per support s and pixel p of the
half-size grid, the 25 values the head's first convolution reads, in its order: centre (3),
warped (3), |warped - centre| (3), centre features (4), support features sampled along the
grid (4), |sampled - centre features| (4), flow (2), |flow| (1), the support's time offset
(1). The half-size warped frames come from the full-size ones here: the 5x5 binomial
low-pass (zero padding) and the 2x bilinear downsample are one 6x6 stencil, weights
[1, 5, 10, 10, 5, 1] / 32 per axis. The grid sample is bilinear with zero padding and
half-pixel centres (grid_sample, align_corners=False); its positions, its weights, the
stencil and the flow norm are computed in FP32. Every thread computes all 25 values of
its pixel, so no branch splits a warp per part. The result is NCHW planes, or channel-last
rows padded to 32 channels (`padded_rows`: TensorRT's HWC8 layout for the convolution,
zeros in the padding), assembled per pixel and stored as whole rows. Inputs are NCHW
FP16 (the grid (S, H, W, 2)); arguments: inputs, output."""

from __future__ import annotations

import triton
import triton.language as tl

CHANNELS: tl.constexpr = tl.constexpr(25)
ROW: tl.constexpr = tl.constexpr(32)


@triton.jit
def _put(
    row: tl.tensor,
    out_ptr: tl.tensor,
    plane: tl.tensor,
    p: tl.tensor,
    c: tl.constexpr,
    value: tl.tensor,
    inside: tl.tensor,
    pixels: tl.constexpr,
    padded_rows: tl.constexpr,
) -> tl.tensor:
    """Channel c of pixels p of one support: into the rows being assembled, or stored in
    its NCHW plane."""
    if padded_rows:
        k = tl.arange(0, ROW)
        row = tl.where(k[None, :] == c, value[:, None], row)
    else:
        at = out_ptr + plane * CHANNELS + c * pixels + p
        tl.store(at, value.to(tl.float16), inside)
    return row


@triton.jit
def _stencil_weight(t: tl.constexpr) -> float:
    """Tap t of [1, 5, 10, 10, 5, 1] / 32 (binomial(5, t) / 32)."""
    if t * (5 - t) == 0:
        weight = 1.0 / 32.0
    elif t * (5 - t) == 4:
        weight = 5.0 / 32.0
    else:
        weight = 10.0 / 32.0
    return weight


@triton.jit
def conf_input(
    centre_ptr: tl.tensor,
    warped_ptr: tl.tensor,
    fcentre_ptr: tl.tensor,
    features_ptr: tl.tensor,
    grid_ptr: tl.tensor,
    flow_ptr: tl.tensor,
    delta_ptr: tl.tensor,
    out_ptr: tl.tensor,
    height: tl.constexpr,
    width: tl.constexpr,
    padded_rows: tl.constexpr,
    block: tl.constexpr,
) -> None:
    s = tl.program_id(1).to(tl.int64)
    pixels = height * width
    p = tl.program_id(0) * block + tl.arange(0, block)
    inside = p < pixels
    y = p // width
    x = p % width
    plane = s * pixels
    row = tl.zeros((block, ROW), tl.float32)

    gx = tl.load(grid_ptr + (plane + p) * 2, inside, 0.0).to(tl.float32)
    gy = tl.load(grid_ptr + (plane + p) * 2 + 1, inside, 0.0).to(tl.float32)
    sx = ((gx + 1.0) * width - 1.0) * 0.5
    sy = ((gy + 1.0) * height - 1.0) * 0.5
    x0f = tl.floor(sx)
    y0f = tl.floor(sy)
    ax = sx - x0f
    ay = sy - y0f
    x0 = x0f.to(tl.int32)
    y0 = y0f.to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    vx0 = (x0 >= 0) & (x0 < width)
    vx1 = (x1 >= 0) & (x1 < width)
    vy0 = (y0 >= 0) & (y0 < height)
    vy1 = (y1 >= 0) & (y1 < height)
    w00 = (1.0 - ax) * (1.0 - ay)
    w01 = ax * (1.0 - ay)
    w10 = (1.0 - ax) * ay
    w11 = ax * ay

    full_w = 2 * width
    for c in tl.static_range(3):
        centre = tl.load(centre_ptr + (s * 3 + c) * pixels + p, inside, 0.0)
        centre = centre.to(tl.float32)
        full = warped_ptr + (s * 3 + c) * (4 * pixels)
        warped = tl.zeros_like(centre)
        for a in tl.static_range(6):
            fy = 2 * y + (a - 2)
            ok_y = inside & (fy >= 0) & (fy < 2 * height)
            line = tl.zeros_like(centre)
            for b in tl.static_range(6):
                fx = 2 * x + (b - 2)
                ok = ok_y & (fx >= 0) & (fx < full_w)
                v = tl.load(full + fy * full_w + fx, ok, 0.0).to(tl.float32)
                line += v * _stencil_weight(b)
            warped += line * _stencil_weight(a)
        # the network's FP16 half-size warped frames
        warped = warped.to(tl.float16).to(tl.float32)
        difference = tl.abs(warped - centre)
        row = _put(row, out_ptr, plane, p, c, centre, inside, pixels, padded_rows)
        row = _put(row, out_ptr, plane, p, 3 + c, warped, inside, pixels, padded_rows)
        row = _put(
            row, out_ptr, plane, p, 6 + c, difference, inside, pixels, padded_rows
        )
    for c in tl.static_range(4):
        fcentre = tl.load(fcentre_ptr + (s * 4 + c) * pixels + p, inside, 0.0)
        fcentre = fcentre.to(tl.float32)
        base = features_ptr + (s * 4 + c) * pixels
        v = (
            tl.load(base + y0 * width + x0, inside & vy0 & vx0, 0.0).to(tl.float32)
            * w00
        )
        v += (
            tl.load(base + y0 * width + x1, inside & vy0 & vx1, 0.0).to(tl.float32)
            * w01
        )
        v += (
            tl.load(base + y1 * width + x0, inside & vy1 & vx0, 0.0).to(tl.float32)
            * w10
        )
        v += (
            tl.load(base + y1 * width + x1, inside & vy1 & vx1, 0.0).to(tl.float32)
            * w11
        )
        # grid_sample's FP16 result, as the network's sampled features
        v = v.to(tl.float16).to(tl.float32)
        difference = tl.abs(v - fcentre)
        row = _put(row, out_ptr, plane, p, 9 + c, fcentre, inside, pixels, padded_rows)
        row = _put(row, out_ptr, plane, p, 13 + c, v, inside, pixels, padded_rows)
        row = _put(
            row, out_ptr, plane, p, 17 + c, difference, inside, pixels, padded_rows
        )
    fx = tl.load(flow_ptr + (s * 2) * pixels + p, inside, 0.0).to(tl.float32)
    fy = tl.load(flow_ptr + (s * 2 + 1) * pixels + p, inside, 0.0).to(tl.float32)
    delta = tl.load(delta_ptr + plane + p, inside, 0.0).to(tl.float32)
    norm = tl.sqrt(fx * fx + fy * fy)
    row = _put(row, out_ptr, plane, p, 21, fx, inside, pixels, padded_rows)
    row = _put(row, out_ptr, plane, p, 22, fy, inside, pixels, padded_rows)
    row = _put(row, out_ptr, plane, p, 23, norm, inside, pixels, padded_rows)
    row = _put(row, out_ptr, plane, p, 24, delta, inside, pixels, padded_rows)
    if padded_rows:
        k = tl.arange(0, ROW)
        at = out_ptr + (plane + p)[:, None] * ROW + k[None, :]
        tl.store(at, row.to(tl.float16), inside[:, None])


LAUNCH = {"block": 128}
OPTIONS = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}


def grid(supports: int, height: int, width: int) -> tuple[int, int, int]:
    return (triton.cdiv(height * width, LAUNCH["block"]), supports, 1)
