"""DUAL's plugin kernels for dynamic-shape engines: the vendored RCA and projection
kernels (vendor/dual_tensorrt/scripts/dual_tensorrt/kernels.py) and dual_aot.py's Norm,
with the trunk size, region counts and pixel count as run-time arguments instead of
tl.constexpr. Arguments are ordered as TensorRT's AOT plugins pass them: inputs, then
scalars, then outputs. The math is the originals' (bit-identical results); the
fixed-size engines keep the originals."""

import triton
import triton.language as tl


@triton.jit
def _convolve(
    x_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    y: tl.tensor,
    x: tl.tensor,
    c: tl.tensor,
    height: tl.int32,
    width: tl.int32,
    channels: tl.constexpr,
):
    acc = tl.full((c.shape[0], y.shape[0]), 0, tl.float32)
    for iy in tl.static_range(3):
        for ix in tl.static_range(3):
            yy, xx = y + iy - 1, x + ix - 1
            valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            value = tl.load(
                x_ptr
                + (yy[None, :] * width + xx[None, :]) * (3 * channels)
                + c[:, None],
                valid[None, :],
                0,
            ).to(tl.float32)
            weight = tl.load(weight_ptr + (iy * 3 + ix) * (3 * channels) + c).to(
                tl.float32
            )
            acc = acc + value * weight[:, None]
    # Invalid output pixels must remain zero even when their halo touches data.
    return tl.where(((y < height) & (x < width))[None, :], acc, 0).to(tl.bfloat16)


@triton.jit
def cell_stats(
    x_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    height: tl.int32,
    width: tl.int32,
    stats_ptr: tl.tensor,
    channels: tl.constexpr,
    block: tl.constexpr,
):
    cell, head = tl.program_id(0), tl.program_id(1)
    cy, cx = cell // tl.cdiv(width, 16), cell % tl.cdiv(width, 16)
    c = tl.arange(0, 32)
    gram = tl.full((32, 32), 0, tl.float32)
    qn, kn = tl.full((32,), 0, tl.float32), tl.full((32,), 0, tl.float32)
    for start in range(0, 256, block):
        t = start + tl.arange(0, block)
        y, x = cy * 16 + t // 16, cx * 16 + t % 16
        q = _convolve(x_ptr, weight_ptr, y, x, head * 32 + c, height, width, channels)
        k = _convolve(
            x_ptr, weight_ptr, y, x, channels + head * 32 + c, height, width, channels
        )
        gram += tl.dot(q, tl.trans(k))
        qf, kf = q.to(tl.float32), k.to(tl.float32)
        qn += tl.sum(qf * qf, 1)
        kn += tl.sum(kf * kf, 1)
    base = (cell * (channels // 32) + head) * 1088
    tl.store(stats_ptr + base + c[:, None] * 32 + c[None, :], gram)
    tl.store(stats_ptr + base + 1024 + c, qn)
    tl.store(stats_ptr + base + 1056 + c, kn)


@triton.jit
def region_stats(
    stats_ptr: tl.tensor,
    tau_ptr: tl.tensor,
    qkv_ptr: tl.tensor,
    height: tl.int32,
    width: tl.int32,
    n0: tl.int32,
    n1: tl.int32,
    n2: tl.int32,
    matrices_ptr: tl.tensor,
    channels: tl.constexpr,
):
    """qkv_ptr is unused: the plugin takes qkv only to see the trunk's size."""
    r, head = tl.program_id(0), tl.program_id(1)
    cls = tl.where(
        r < n0, 0, tl.where(r < n0 + n1, 1, tl.where(r < n0 + n1 + n2, 2, 3))
    )
    rid = r - tl.where(
        cls == 0, 0, tl.where(cls == 1, n0, tl.where(cls == 2, n0 + n1, n0 + n1 + n2))
    )
    sy, sx = (cls // 2) * 16, (cls % 2) * 16
    nx = (width + sx + 31) // 32
    y0, x0 = (rid // nx) * 32 - sy, (rid % nx) * 32 - sx
    c = tl.arange(0, 32)
    gram = tl.full((32, 32), 0, tl.float32)
    qn, kn = tl.full((32,), 0, tl.float32), tl.full((32,), 0, tl.float32)
    for iy in tl.static_range(2):
        for ix in tl.static_range(2):
            cy, cx = y0 // 16 + iy, x0 // 16 + ix
            valid = (
                (cy >= 0)
                & (cy < tl.cdiv(height, 16))
                & (cx >= 0)
                & (cx < tl.cdiv(width, 16))
            )
            base = ((cy * tl.cdiv(width, 16) + cx) * (channels // 32) + head) * 1088
            gram += tl.load(stats_ptr + base + c[:, None] * 32 + c[None, :], valid, 0)
            qn += tl.load(stats_ptr + base + 1024 + c, valid, 0)
            kn += tl.load(stats_ptr + base + 1056 + c, valid, 0)
    count = (tl.minimum(y0 + 32, height) - tl.maximum(y0, 0)) * (
        tl.minimum(x0 + 32, width) - tl.maximum(x0, 0)
    )
    qn, kn = tl.sqrt(qn + count * 1e-6), tl.sqrt(kn + count * 1e-6)
    logits = gram / (qn[:, None] * kn[None, :]) * tl.load(tau_ptr + head)
    p = tl.exp(logits - tl.max(logits, 1)[:, None])
    p = p / tl.sum(p, 1)[:, None]
    tl.store(
        matrices_ptr
        + (r * (channels // 32) + head) * 1024
        + c[:, None] * 32
        + c[None, :],
        p,
    )


@triton.jit
def apply_dw(
    x_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    matrices_ptr: tl.tensor,
    height: tl.int32,
    width: tl.int32,
    n0: tl.int32,
    n1: tl.int32,
    n2: tl.int32,
    out_ptr: tl.tensor,
    channels: tl.constexpr,
    block: tl.constexpr,
):
    cell, head, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cy, cx = cell // tl.cdiv(width, 16), cell % tl.cdiv(width, 16)
    t = part * block + tl.arange(0, block)
    y, x = cy * 16 + t // 16, cx * 16 + t % 16
    c = tl.arange(0, 32)
    v = _convolve(
        x_ptr, weight_ptr, y, x, 2 * channels + head * 32 + c, height, width, channels
    )
    result = tl.full((32, block), 0, tl.float32)
    for cls in tl.static_range(4):
        sy, sx = (cls // 2) * 16, (cls % 2) * 16
        nx = (width + sx + 31) // 32
        start = (
            0
            if cls == 0
            else (n0 if cls == 1 else (n0 + n1 if cls == 2 else n0 + n1 + n2))
        )
        rid = start + ((cy * 16 + sy) // 32) * nx + (cx * 16 + sx) // 32
        a = tl.load(
            matrices_ptr
            + (rid * (channels // 32) + head) * 1024
            + c[:, None] * 32
            + c[None, :]
        )
        out = tl.dot(a, v).to(tl.bfloat16).to(tl.float32)
        wy = 1.0 - tl.abs(((y + sy) % 32).to(tl.float32) - 15.5) / 16.0
        wx = 1.0 - tl.abs(((x + sx) % 32).to(tl.float32) - 15.5) / 16.0
        result = result + (out * wy[None, :]) * wx[None, :]
    tl.store(
        out_ptr + (y[None, :] * width + x[None, :]) * channels + head * 32 + c[:, None],
        result,
        ((y < height) & (x < width))[None, :],
    )


@triton.jit
def project_add(
    x_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    residual_ptr: tl.tensor,
    pixels: tl.int32,
    out_ptr: tl.tensor,
    channels: tl.constexpr,
    block_c: tl.constexpr,
    block_m: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_c)
    ks = tl.arange(0, 32)
    acc = tl.full((block_m, block_c), 0, tl.float32)
    for start in tl.static_range(0, channels, 32):
        x = tl.load(
            x_ptr + rows[:, None] * channels + (ks + start)[None, :],
            (rows < pixels)[:, None],
            0,
        )
        w = tl.load(
            weight_ptr + (ks + start)[:, None] * channels + cols[None, :],
            (cols < channels)[None, :],
            0,
        )
        acc += tl.dot(x, w)
    mask = (rows < pixels)[:, None] & (cols < channels)[None, :]
    residual = tl.load(
        residual_ptr + rows[:, None] * channels + cols[None, :], mask, 0
    ).to(tl.float32)
    # Preserve the standalone projection's BF16 output before the residual add.
    result = acc.to(tl.bfloat16).to(tl.float32) + residual
    tl.store(out_ptr + rows[:, None] * channels + cols[None, :], result, mask)


@triton.jit
def norm(
    x_ptr: tl.tensor,
    gamma_ptr: tl.tensor,
    beta_ptr: tl.tensor,
    pixels: tl.int32,
    y_ptr: tl.tensor,
    sigma_ptr: tl.tensor,
    channels: tl.constexpr,
    block_c: tl.constexpr,
    block_p: tl.constexpr,
    eps: tl.constexpr,
):
    """dual_aot.norm_kernel with a run-time pixel count."""
    rows = tl.program_id(0) * block_p + tl.arange(0, block_p)
    cols = tl.arange(0, block_c)
    mask = (rows < pixels)[:, None] & (cols < channels)[None, :]
    x = tl.load(x_ptr + rows[:, None] * channels + cols[None, :], mask, 0)
    x = x.to(tl.float32)
    mean = tl.math.div_rn(tl.sum(x, 1), channels)
    centered = tl.where(mask, x - mean[:, None], 0)
    variance = tl.math.div_rn(tl.sum(centered * centered, 1), channels) + eps
    inv = tl.math.rsqrt(variance)
    gamma = tl.load(gamma_ptr + cols, cols < channels, 0)
    beta = tl.load(beta_ptr + cols, cols < channels, 0)
    y = (gamma[None, :] * centered) * inv[:, None] + beta[None, :]
    tl.store(y_ptr + rows[:, None] * channels + cols[None, :], y.to(tl.bfloat16), mask)
    tl.store(sigma_ptr + rows, tl.sqrt(variance), rows < pixels)
