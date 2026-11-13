"""Row sums of an NHWC BF16 tensor in FP32 for the dynamic convolution's router: out[0, y, c] =
sum over x of in[0, y, x, c]. One program per row, coalesced 32-pixel x C tiles; the graph then
sums the H rows and divides by H * W (a few hundred values). Arguments: inputs, scalars, outputs."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def row_sum(
    x_ptr: tl.tensor,
    width: tl.int32,
    out_ptr: tl.tensor,
    channels: tl.constexpr,
    block: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    c = tl.arange(0, channels)
    p = tl.arange(0, block)
    acc = tl.zeros((channels,), tl.float32)
    for x0 in range(0, width, block):
        px = x0 + p
        tile = tl.load(
            x_ptr + (row * width + px)[:, None] * channels + c[None, :],
            (px < width)[:, None],
            0,
        )
        acc += tl.sum(tile.to(tl.float32), 0)
    tl.store(out_ptr + row * channels + c, acc)


LAUNCH = {"block": 32}
OPTIONS = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
