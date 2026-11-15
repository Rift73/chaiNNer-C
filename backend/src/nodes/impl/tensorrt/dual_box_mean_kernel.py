"""The routed convolution's router input: the FP32 mean of an NHWC BF16 tensor over each
position's clamped K x K neighborhood, start = clamp(p - K // 2, 0, L - K), K =
min(window, L) per axis; out[0, y, x, c] FP32, NHWC. One program per 32 output columns, 32
output rows and 32 channels: the sums of its span source columns over the first row's K
rows, then one row in and one out per output row (each sum has at most K + 31 terms; a
cumulative sum over the map would lose digits), and per output row the horizontal sums as
an FP32 product with the 0/1 band of the windows. Arguments: inputs, scalars, outputs."""

from __future__ import annotations

from typing import Any, Callable

import triton
import triton.language as tl


@triton.jit
def box_mean(
    x_ptr: tl.tensor,
    height: tl.int32,
    width: tl.int32,
    out_ptr: tl.tensor,
    channels: tl.constexpr,
    window: tl.constexpr,
    block_x: tl.constexpr,
    block_y: tl.constexpr,
    block_c: tl.constexpr,
    span: tl.constexpr,
) -> None:
    x0 = tl.program_id(0) * block_x
    y0 = tl.program_id(1) * block_y
    c = tl.program_id(2) * block_c + tl.arange(0, block_c)
    kx = tl.minimum(width, window)
    ky = tl.minimum(height, window)
    # The tile's windows lie in [base, base + span) (span >= block_x - 1 + window).
    base = tl.maximum(tl.minimum(x0 - kx // 2, width - span), 0)
    columns = base + tl.arange(0, span)
    mask = (columns < width)[:, None] & (c < channels)[None, :]
    offsets = columns[:, None] * channels + c[None, :]
    px = x0 + tl.arange(0, block_x)
    starts = tl.maximum(tl.minimum(px - kx // 2, width - kx), 0)
    band = (columns[None, :] >= starts[:, None]) & (
        columns[None, :] < starts[:, None] + kx
    )
    band = band.to(tl.float32)
    row = width * channels
    top = tl.maximum(tl.minimum(y0 - ky // 2, height - ky), 0)
    sums = tl.zeros((span, block_c), tl.float32)
    for r in range(ky):
        sums += tl.load(x_ptr + (top + r) * row + offsets, mask, 0).to(tl.float32)
    count = (kx * ky).to(tl.float32)
    out_mask = (px < width)[:, None] & (c < channels)[None, :]
    for y in range(y0, tl.minimum(y0 + block_y, height)):
        # The window moves down by one row or stays.
        start = tl.maximum(tl.minimum(y - ky // 2, height - ky), 0)
        step = start - top
        entering = tl.load(x_ptr + (top + ky * step) * row + offsets, mask, 0)
        leaving = tl.load(x_ptr + top * row + offsets, mask, 0)
        sums += (entering.to(tl.float32) - leaving.to(tl.float32)) * step.to(tl.float32)
        top = start
        means = tl.math.div_rn(tl.dot(band, sums, input_precision="ieee"), count)
        tl.store(
            out_ptr + y * row + px[:, None] * channels + c[None, :], means, out_mask
        )


LAUNCH = {"window": 96, "block_x": 32, "block_y": 32, "block_c": 32, "span": 128}
OPTIONS = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}


def grid(
    height: Any, width: Any, channels: int, cdiv: Callable[[Any, int], Any]
) -> tuple[Any, Any, int]:
    """The launch grid; ints (cdiv = triton.cdiv) or TensorRT shape expressions."""
    return (
        cdiv(width, LAUNCH["block_x"]),
        cdiv(height, LAUNCH["block_y"]),
        triton.cdiv(channels, LAUNCH["block_c"]),
    )
