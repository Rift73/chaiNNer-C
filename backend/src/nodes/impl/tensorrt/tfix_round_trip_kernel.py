"""Temporal Fix's resize round trips (a half-resolution map bilinearly upsampled 2x, low-passed
with the 5x5 binomial filter, bilinearly downsampled back) as one stencil on the half-resolution
grid. The three steps are linear and separable, so each output pixel is sum_a sum_b ch[i, a] *
cw[j, b] * x[ih[i, a], iw[j, b]] with per-axis tap tables composed in float64 (`taps`), borders
included (the upsample's edge clamp, the filter's zero padding). FP32 accumulation, FP16 in and
out (FP32 for the flow: x_ptr's type), a per-channel scale (a flow's 2x before the filter). The map and its result are NCHW or
channel-last (`channels_last`: the layout TensorRT keeps around the plugin, so that it inserts no
reformat). Arguments: inputs, outputs."""

from __future__ import annotations

import numpy as np
import triton
import triton.language as tl


def _bilinear(n_in: int, n_out: int) -> np.ndarray:
    """Bilinear resize along one axis, half-pixel centres (align_corners=False)."""
    m = np.zeros((n_out, n_in))
    scale = n_in / n_out
    for x in range(n_out):
        source = max((x + 0.5) * scale - 0.5, 0.0)
        i0 = min(int(np.floor(source)), n_in - 1)
        lam = source - np.floor(source)
        i1 = min(i0 + 1, n_in - 1)
        m[x, i0] += 1 - lam
        m[x, i1] += lam
    return m


def _filter(n: int, kernel: np.ndarray) -> np.ndarray:
    """A centred 1-D filter along one axis, zero padding."""
    m = np.zeros((n, n))
    r = len(kernel) // 2
    for x in range(n):
        for d in range(-r, r + 1):
            if 0 <= x + d < n:
                m[x, x + d] += kernel[d + r]
    return m


def taps(n: int, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tap indices (int32) and weights (float32) of down . filter . up for an axis of
    n half-resolution pixels, one row per output pixel; short rows repeat their last
    index with weight 0."""
    a = _bilinear(2 * n, n) @ _filter(2 * n, kernel) @ _bilinear(n, 2 * n)
    k = int((a != 0).sum(1).max())
    index = np.zeros((n, k), np.int32)
    weight = np.zeros((n, k), np.float32)
    for row in range(n):
        nz = np.nonzero(a[row])[0]
        index[row, : len(nz)] = nz
        index[row, len(nz) :] = nz[-1]
        weight[row, : len(nz)] = a[row, nz]
    return index, weight


@triton.jit
def round_trip(
    x_ptr: tl.tensor,
    ih_ptr: tl.tensor,
    ch_ptr: tl.tensor,
    iw_ptr: tl.tensor,
    cw_ptr: tl.tensor,
    scale_ptr: tl.tensor,
    y_ptr: tl.tensor,
    height: tl.constexpr,
    width: tl.constexpr,
    channels: tl.constexpr,
    k: tl.constexpr,
    channels_last: tl.constexpr,
    bh: tl.constexpr,
    bw: tl.constexpr,
) -> None:
    plane = tl.program_id(2)
    i = tl.program_id(1) * bh + tl.arange(0, bh)
    j = tl.program_id(0) * bw + tl.arange(0, bw)
    mi = i < height
    mj = j < width
    inside = mi[:, None] & mj[None, :]
    if channels_last:
        n = plane // channels
        base = n.to(tl.int64) * height * width * channels + plane % channels
        step = channels
    else:
        base = plane.to(tl.int64) * height * width
        step = 1
    acc = tl.zeros((bh, bw), tl.float32)
    for a in tl.static_range(k):
        r = tl.load(ih_ptr + i * k + a, mi, 0)
        cr = tl.load(ch_ptr + i * k + a, mi, 0.0)
        row = tl.zeros((bh, bw), tl.float32)
        for b in tl.static_range(k):
            c = tl.load(iw_ptr + j * k + b, mj, 0)
            cc = tl.load(cw_ptr + j * k + b, mj, 0.0)
            at = (r[:, None] * width + c[None, :]) * step
            v = tl.load(x_ptr + base + at, inside, 0.0)
            row += v.to(tl.float32) * cc[None, :]
        acc += row * cr[:, None]
    acc = acc * tl.load(scale_ptr + plane % channels)
    out = y_ptr + base + (i[:, None] * width + j[None, :]) * step
    tl.store(out, acc.to(y_ptr.dtype.element_ty), inside)


LAUNCH = {"bh": 8, "bw": 64}
OPTIONS = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}


def grid(planes: int, height: int, width: int) -> tuple[int, int, int]:
    return (
        triton.cdiv(width, LAUNCH["bw"]),
        triton.cdiv(height, LAUNCH["bh"]),
        planes,
    )
