"""Temporal Fix's full-size tail in two passes, from the aligner's half-size flow (FP32):

- `warp`: each of the six support frames (one input each, the window's own frames)
  warped along the flow, as the network's fp32_warp: the flow upsampled 2x (bilinear,
  half-pixel centres, edge clamp) and scaled per channel by flow_x and flow_y, the
  sampling positions the stored base grid's (FP16 values, as the model holds it) plus
  the scaled flow, bilinear sampling with zero padding (grid_sample,
  align_corners=False). FP16 frames in and out.
- `blend`: the confidence-weighted temporal average: the confidence logits upsampled 2x
  and squashed, times each position's share inside the frame (the warp of a tensor of
  ones, in closed form from the same positions), the gate, the weights and the blend
  with the centre frame (conservative_temporal_average), out in FP32. The confidence is
  rounded to FP16 where the graph rounds it (the upsampled logits, their sigmoid, the
  share and their product), so the pass computes the graph's operations; the gate and
  the average are FP32.

The flow's scales and the average's constants are the graph's (tfix_aot reads them where
they act and passes them as constants). Positions, interpolation weights and every sum
are FP32. The tensors the passes read and
write are NCHW, one plane per support; the half-size flow is (S, 2, H/2, W/2), the
confidence logits (S, 1, H/2, W/2). Arguments: inputs, outputs."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _up(
    plane: tl.tensor,
    y: tl.tensor,
    x: tl.tensor,
    half_h: tl.constexpr,
    half_w: tl.constexpr,
    inside: tl.tensor,
) -> tl.tensor:
    """A half-size plane bilinearly upsampled 2x at full-size pixels (y, x)."""
    sy = tl.maximum((y.to(tl.float32) + 0.5) * 0.5 - 0.5, 0.0)
    sx = tl.maximum((x.to(tl.float32) + 0.5) * 0.5 - 0.5, 0.0)
    y0 = tl.minimum(sy.to(tl.int32), half_h - 1)
    x0 = tl.minimum(sx.to(tl.int32), half_w - 1)
    ly = sy - y0.to(tl.float32)
    lx = sx - x0.to(tl.float32)
    y1 = tl.minimum(y0 + 1, half_h - 1)
    x1 = tl.minimum(x0 + 1, half_w - 1)
    a = tl.load(plane + y0 * half_w + x0, inside, 0.0).to(tl.float32)
    b = tl.load(plane + y0 * half_w + x1, inside, 0.0).to(tl.float32)
    c = tl.load(plane + y1 * half_w + x0, inside, 0.0).to(tl.float32)
    d = tl.load(plane + y1 * half_w + x1, inside, 0.0).to(tl.float32)
    top = a * (1.0 - lx) + b * lx
    bottom = c * (1.0 - lx) + d * lx
    return top * (1.0 - ly) + bottom * ly


@triton.jit
def _pixels(
    flow_ptr: tl.tensor,
    s: tl.tensor,
    y: tl.tensor,
    x: tl.tensor,
    height: tl.constexpr,
    width: tl.constexpr,
    inside: tl.tensor,
    flow_x: tl.constexpr,
    flow_y: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor]:
    """The sampling position of full-size pixel (y, x) of support s, in pixels
    (grid_sample's unnormalized coordinates), the flow scaled by flow_x and flow_y."""
    half_h: tl.constexpr = height // 2
    half_w: tl.constexpr = width // 2
    plane = flow_ptr + s * 2 * half_h * half_w
    fx = _up(plane, y, x, half_h, half_w, inside) * flow_x
    fy = _up(plane + half_h * half_w, y, x, half_h, half_w, inside) * flow_y
    # The stored base grid's values ((i + 0.5) * 2 / n - 1, which the model holds in
    # FP16) plus the scaled flow
    bx = ((x.to(tl.float32) + 0.5) * (2.0 / width) - 1.0).to(tl.float16).to(tl.float32)
    by = ((y.to(tl.float32) + 0.5) * (2.0 / height) - 1.0).to(tl.float16).to(tl.float32)
    gx = bx + fx * (2.0 / width)
    gy = by + fy * (2.0 / height)
    return ((gx + 1.0) * width - 1.0) * 0.5, ((gy + 1.0) * height - 1.0) * 0.5


@triton.jit
def warp(
    f0_ptr: tl.tensor,
    f1_ptr: tl.tensor,
    f2_ptr: tl.tensor,
    f3_ptr: tl.tensor,
    f4_ptr: tl.tensor,
    f5_ptr: tl.tensor,
    flow_ptr: tl.tensor,
    out_ptr: tl.tensor,
    height: tl.constexpr,
    width: tl.constexpr,
    flow_x: tl.constexpr,
    flow_y: tl.constexpr,
    block: tl.constexpr,
) -> None:
    s = tl.program_id(1).to(tl.int64)
    pixels: tl.constexpr = height * width
    p = tl.program_id(0) * block + tl.arange(0, block)
    inside = p < pixels
    y = p // width
    x = p % width
    px, py = _pixels(flow_ptr, s, y, x, height, width, inside, flow_x, flow_y)
    # Support s's frame, one (1, 3, H, W) input each
    frame = f0_ptr
    if s == 1:
        frame = f1_ptr
    if s == 2:
        frame = f2_ptr
    if s == 3:
        frame = f3_ptr
    if s == 4:
        frame = f4_ptr
    if s == 5:
        frame = f5_ptr
    x0f = tl.floor(px)
    y0f = tl.floor(py)
    ax = px - x0f
    ay = py - y0f
    x0 = x0f.to(tl.int32)
    y0 = y0f.to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    vx0 = (x0 >= 0) & (x0 < width)
    vx1 = (x1 >= 0) & (x1 < width)
    vy0 = (y0 >= 0) & (y0 < height)
    vy1 = (y1 >= 0) & (y1 < height)
    for c in tl.static_range(3):
        base = frame + c * pixels
        v = tl.load(base + y0 * width + x0, inside & vy0 & vx0, 0.0).to(tl.float32)
        v = v * ((1.0 - ax) * (1.0 - ay))
        b = tl.load(base + y0 * width + x1, inside & vy0 & vx1, 0.0).to(tl.float32)
        v += b * (ax * (1.0 - ay))
        b = tl.load(base + y1 * width + x0, inside & vy1 & vx0, 0.0).to(tl.float32)
        v += b * ((1.0 - ax) * ay)
        b = tl.load(base + y1 * width + x1, inside & vy1 & vx1, 0.0).to(tl.float32)
        v += b * (ax * ay)
        tl.store(out_ptr + (s * 3 + c) * pixels + p, v.to(tl.float16), inside)


@triton.jit
def _share(pixel: tl.tensor, size: tl.constexpr) -> tl.tensor:
    """The weight of a bilinear sample's corners inside 0..size-1 along one axis."""
    return tl.minimum(tl.maximum(tl.minimum(pixel + 1.0, size - pixel), 0.0), 1.0)


@triton.jit
def blend(
    centre_ptr: tl.tensor,
    warped_ptr: tl.tensor,
    logits_ptr: tl.tensor,
    flow_ptr: tl.tensor,
    out_ptr: tl.tensor,
    height: tl.constexpr,
    width: tl.constexpr,
    supports: tl.constexpr,
    flow_x: tl.constexpr,
    flow_y: tl.constexpr,
    threshold: tl.constexpr,
    gate_slope: tl.constexpr,
    min_support: tl.constexpr,
    offset: tl.constexpr,
    count_slope: tl.constexpr,
    one: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
) -> None:
    pixels: tl.constexpr = height * width
    half_h: tl.constexpr = height // 2
    half_w: tl.constexpr = width // 2
    p = tl.program_id(0) * block + tl.arange(0, block)
    inside = p < pixels
    y = p // width
    x = p % width
    zero = tl.zeros((block,), tl.float32)
    sum_r, sum_g, sum_b = zero, zero, zero
    weights = zero
    count = zero
    for s in tl.static_range(supports):
        logit = _up(logits_ptr + s * half_h * half_w, y, x, half_h, half_w, inside)
        logit = logit.to(tl.float16).to(tl.float32)
        px, py = _pixels(flow_ptr, s, y, x, height, width, inside, flow_x, flow_y)
        valid = (_share(px, width) * _share(py, height)).to(tl.float16)
        squashed = (1.0 / (1.0 + tl.exp(-logit))).to(tl.float16)
        conf = (squashed.to(tl.float32) * valid.to(tl.float32)).to(tl.float16)
        conf = conf.to(tl.float32)
        gate = 1.0 / (1.0 + tl.exp(-(conf - threshold) * gate_slope))
        weight = conf * gate
        frame = warped_ptr + s * 3 * pixels + p
        sum_r += weight * tl.load(frame, inside, 0.0).to(tl.float32)
        sum_g += weight * tl.load(frame + pixels, inside, 0.0).to(tl.float32)
        sum_b += weight * tl.load(frame + 2 * pixels, inside, 0.0).to(tl.float32)
        weights += weight
        count += gate
    ready = 1.0 / (1.0 + tl.exp(-(count - min_support + offset) * count_slope))
    norm = tl.maximum(weights + one, eps)
    _out(centre_ptr, out_ptr, p, 0, sum_r, norm, ready, pixels, inside)
    _out(centre_ptr, out_ptr, p, 1, sum_g, norm, ready, pixels, inside)
    _out(centre_ptr, out_ptr, p, 2, sum_b, norm, ready, pixels, inside)


@triton.jit
def _out(
    centre_ptr: tl.tensor,
    out_ptr: tl.tensor,
    p: tl.tensor,
    c: tl.constexpr,
    total: tl.tensor,
    norm: tl.tensor,
    ready: tl.tensor,
    pixels: tl.constexpr,
    inside: tl.tensor,
) -> None:
    """Channel c of the blend: the centre moved towards the weighted mean by `ready`."""
    centre = tl.load(centre_ptr + c * pixels + p, inside, 0.0).to(tl.float32)
    blended = (centre + total) / norm
    tl.store(out_ptr + c * pixels + p, centre + ready * (blended - centre), inside)


LAUNCH = {"block": 256}
OPTIONS = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}


def grid(planes: int, height: int, width: int) -> tuple[int, int, int]:
    return (triton.cdiv(height * width, LAUNCH["block"]), planes, 1)
