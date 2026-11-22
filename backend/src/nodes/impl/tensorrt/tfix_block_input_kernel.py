"""The input of Temporal Fix's last flow refinement stage (the aligner's third block) in one
pass, per support s and pixel p of the half-size grid, in the block's channel order: the
centre frame (3), the support frame warped along the flow (3), the centre features (4),
the support features warped (4), the support's time offset (1), the previous stage's
confidence logit (1) and latent (8) brought to this size, and the flow (2).

The flow arrives at quarter size in FP32 and is upsampled 2x (bilinear, half-pixel
centres, edge clamp) and scaled per channel by flow_x and flow_y (the graph's scales) in
FP32, as the stage's _resize_flow; the warp's
positions are the stored grid's (FP16 values, as the model holds them) plus the scaled
flow, its bilinear sampling with zero padding in FP32, rounded to FP16 as the export's
fp32_warp. The logit and latent arrive as the previous block's FP32 quarter-size output
(channels 2 and 3..10), rounded to FP16 and upsampled 2x. The result is 26 FP16 NCHW
planes per support, or with `padded_rows` one row of 32 per pixel (the 26 channels, then
zeros: TensorRT's HWC8 layout of 26 channels), assembled in registers and stored once, for
an engine whose input this is. Arguments: inputs, output."""

from __future__ import annotations

import triton
import triton.language as tl

from .tfix_tail_kernel import _up

CHANNELS: tl.constexpr = tl.constexpr(26)
ROW: tl.constexpr = tl.constexpr(32)


@triton.jit
def _put(
    row: tl.tensor,
    out_ptr: tl.tensor,
    s: tl.tensor,
    p: tl.tensor,
    c: tl.constexpr,
    value: tl.tensor,
    inside: tl.tensor,
    pixels: tl.constexpr,
    padded_rows: tl.constexpr,
) -> tl.tensor:
    """Channel c of pixels p of support s: into the rows being assembled, or stored in
    its NCHW plane."""
    if padded_rows:
        k = tl.arange(0, ROW)
        row = tl.where(k[None, :] == c, value.to(tl.float32)[:, None], row)
    else:
        at = out_ptr + (s * CHANNELS + c) * pixels + p
        tl.store(at, value.to(tl.float16), inside)
    return row


@triton.jit
def block_input(
    centre_ptr: tl.tensor,
    fcentre_ptr: tl.tensor,
    supports_ptr: tl.tensor,
    flow_ptr: tl.tensor,
    previous_ptr: tl.tensor,
    delta_ptr: tl.tensor,
    out_ptr: tl.tensor,
    height: tl.constexpr,
    width: tl.constexpr,
    flow_x: tl.constexpr,
    flow_y: tl.constexpr,
    padded_rows: tl.constexpr,
    block: tl.constexpr,
) -> None:
    s = tl.program_id(1).to(tl.int64)
    pixels: tl.constexpr = height * width
    quarter_h: tl.constexpr = height // 2
    quarter_w: tl.constexpr = width // 2
    quarter: tl.constexpr = quarter_h * quarter_w
    p = tl.program_id(0) * block + tl.arange(0, block)
    inside = p < pixels
    y = p // width
    x = p % width
    row = tl.zeros((block, ROW), tl.float32)

    flow = flow_ptr + s * 2 * quarter
    fx = _up(flow, y, x, quarter_h, quarter_w, inside) * flow_x
    fy = _up(flow + quarter, y, x, quarter_h, quarter_w, inside) * flow_y
    bx = ((x.to(tl.float32) + 0.5) * (2.0 / width) - 1.0).to(tl.float16).to(tl.float32)
    by = ((y.to(tl.float32) + 0.5) * (2.0 / height) - 1.0).to(tl.float16).to(tl.float32)
    px = ((bx + fx * (2.0 / width) + 1.0) * width - 1.0) * 0.5
    py = ((by + fy * (2.0 / height) + 1.0) * height - 1.0) * 0.5
    x0f = tl.floor(px)
    y0f = tl.floor(py)
    ax = px - x0f
    ay = py - y0f
    x0 = x0f.to(tl.int32)
    y0 = y0f.to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    m00 = inside & (y0 >= 0) & (y0 < height) & (x0 >= 0) & (x0 < width)
    m01 = inside & (y0 >= 0) & (y0 < height) & (x1 >= 0) & (x1 < width)
    m10 = inside & (y1 >= 0) & (y1 < height) & (x0 >= 0) & (x0 < width)
    m11 = inside & (y1 >= 0) & (y1 < height) & (x1 >= 0) & (x1 < width)
    w00 = (1.0 - ax) * (1.0 - ay)
    w01 = ax * (1.0 - ay)
    w10 = (1.0 - ax) * ay
    w11 = ax * ay

    # The support's frame (its channels 0..2) and features (3..6), warped: the block's
    # channels 3..5 and 10..13
    for c in tl.static_range(7):
        plane = supports_ptr + (s * 7 + c) * pixels
        v = tl.load(plane + y0 * width + x0, m00, 0.0).to(tl.float32) * w00
        v += tl.load(plane + y0 * width + x1, m01, 0.0).to(tl.float32) * w01
        v += tl.load(plane + y1 * width + x0, m10, 0.0).to(tl.float32) * w10
        v += tl.load(plane + y1 * width + x1, m11, 0.0).to(tl.float32) * w11
        v = v.to(tl.float16)
        row = _put(
            row,
            out_ptr,
            s,
            p,
            3 + c + 4 * ((c + 3) // 6),
            v,
            inside,
            pixels,
            padded_rows,
        )
    for c in tl.static_range(3):
        centre = tl.load(centre_ptr + c * pixels + p, inside, 0.0)
        row = _put(row, out_ptr, s, p, c, centre, inside, pixels, padded_rows)
    for c in tl.static_range(4):
        fcentre = tl.load(fcentre_ptr + c * pixels + p, inside, 0.0)
        row = _put(row, out_ptr, s, p, 6 + c, fcentre, inside, pixels, padded_rows)
    delta = tl.load(delta_ptr + s + p * 0, inside, 0.0)
    row = _put(row, out_ptr, s, p, 14, delta, inside, pixels, padded_rows)
    for c in tl.static_range(9):
        # The previous block's output channel 2 + c (logit, then latent), in FP16
        plane = previous_ptr + (s * 11 + 2 + c) * quarter
        value = _up16(plane, y, x, quarter_h, quarter_w, inside)
        row = _put(row, out_ptr, s, p, 15 + c, value, inside, pixels, padded_rows)
    row = _put(row, out_ptr, s, p, 24, fx, inside, pixels, padded_rows)
    row = _put(row, out_ptr, s, p, 25, fy, inside, pixels, padded_rows)
    if padded_rows:
        k = tl.arange(0, ROW)
        at = out_ptr + (s * pixels + p)[:, None] * ROW + k[None, :]
        tl.store(at, row.to(tl.float16), inside[:, None])


@triton.jit
def _up16(
    plane: tl.tensor,
    y: tl.tensor,
    x: tl.tensor,
    small_h: tl.constexpr,
    small_w: tl.constexpr,
    inside: tl.tensor,
) -> tl.tensor:
    """_up of an FP32 plane rounded to FP16 first, as the network casts it."""
    sy = tl.maximum((y.to(tl.float32) + 0.5) * 0.5 - 0.5, 0.0)
    sx = tl.maximum((x.to(tl.float32) + 0.5) * 0.5 - 0.5, 0.0)
    y0 = tl.minimum(sy.to(tl.int32), small_h - 1)
    x0 = tl.minimum(sx.to(tl.int32), small_w - 1)
    ly = sy - y0.to(tl.float32)
    lx = sx - x0.to(tl.float32)
    y1 = tl.minimum(y0 + 1, small_h - 1)
    x1 = tl.minimum(x0 + 1, small_w - 1)
    a = tl.load(plane + y0 * small_w + x0, inside, 0.0).to(tl.float16).to(tl.float32)
    b = tl.load(plane + y0 * small_w + x1, inside, 0.0).to(tl.float16).to(tl.float32)
    c = tl.load(plane + y1 * small_w + x0, inside, 0.0).to(tl.float16).to(tl.float32)
    d = tl.load(plane + y1 * small_w + x1, inside, 0.0).to(tl.float16).to(tl.float32)
    top = a * (1.0 - lx) + b * lx
    bottom = c * (1.0 - lx) + d * lx
    return top * (1.0 - ly) + bottom * ly


LAUNCH = {"block": 128}
OPTIONS = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}


def grid(supports: int, height: int, width: int) -> tuple[int, int, int]:
    return (triton.cdiv(height * width, LAUNCH["block"]), supports, 1)
