"""Shape/width-specialized RCA kernels, generalized from the accepted C128 path.

Compilation is explicit and offline; importing this module launches nothing.
Fixed-owner reductions, BF16 boundaries and canonical four-class blending.
"""

# ruff: noqa: ANN001, ANN201, N803
import triton as tr
import triton.language as tl


@tr.jit
def convolve(X, WEIGHT, y, x, c, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr):
    acc = tl.full((c.shape[0], y.shape[0]), 0, tl.float32)
    for iy in tl.static_range(3):
        for ix in tl.static_range(3):
            yy, xx = y + iy - 1, x + ix - 1
            valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            value = tl.load(
                X + (yy[None, :] * W + xx[None, :]) * (3 * C) + c[:, None],
                valid[None, :],
                0,
            ).to(tl.float32)
            weight = tl.load(WEIGHT + (iy * 3 + ix) * (3 * C) + c).to(tl.float32)
            acc = acc + value * weight[:, None]
    # Invalid output pixels must remain zero even when their halo touches data.
    return tl.where(((y < H) & (x < W))[None, :], acc, 0).to(tl.bfloat16)


@tr.jit
def cell_stats(
    X,
    WEIGHT,
    STATS,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    BT: tl.constexpr,
):
    cell, head = tl.program_id(0), tl.program_id(1)
    cy, cx = cell // tr.cdiv(W, 16), cell % tr.cdiv(W, 16)
    c = tl.arange(0, 32)
    gram = tl.full((32, 32), 0, tl.float32)
    qn, kn = tl.full((32,), 0, tl.float32), tl.full((32,), 0, tl.float32)
    for start in range(0, 256, BT):
        t = start + tl.arange(0, BT)
        y, x = cy * 16 + t // 16, cx * 16 + t % 16
        q = convolve(X, WEIGHT, y, x, head * 32 + c, H, W, C)
        k = convolve(X, WEIGHT, y, x, C + head * 32 + c, H, W, C)
        gram += tl.dot(q, tl.trans(k))
        qf, kf = q.to(tl.float32), k.to(tl.float32)
        qn += tl.sum(qf * qf, 1)
        kn += tl.sum(kf * kf, 1)
    base = (cell * (C // 32) + head) * 1088
    tl.store(STATS + base + c[:, None] * 32 + c[None, :], gram)
    tl.store(STATS + base + 1024 + c, qn)
    tl.store(STATS + base + 1056 + c, kn)


@tr.jit
def region_stats(
    STATS,
    TAU,
    A,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
):
    r, head = tl.program_id(0), tl.program_id(1)
    cls = tl.where(
        r < N0, 0, tl.where(r < N0 + N1, 1, tl.where(r < N0 + N1 + N2, 2, 3))
    )
    rid = r - tl.where(
        cls == 0, 0, tl.where(cls == 1, N0, tl.where(cls == 2, N0 + N1, N0 + N1 + N2))
    )
    sy, sx = (cls // 2) * 16, (cls % 2) * 16
    nx = (W + sx + 31) // 32
    y0, x0 = (rid // nx) * 32 - sy, (rid % nx) * 32 - sx
    c = tl.arange(0, 32)
    gram = tl.full((32, 32), 0, tl.float32)
    qn, kn = tl.full((32,), 0, tl.float32), tl.full((32,), 0, tl.float32)
    for iy in tl.static_range(2):
        for ix in tl.static_range(2):
            cy, cx = y0 // 16 + iy, x0 // 16 + ix
            valid = (
                (cy >= 0) & (cy < tr.cdiv(H, 16)) & (cx >= 0) & (cx < tr.cdiv(W, 16))
            )
            base = ((cy * tr.cdiv(W, 16) + cx) * (C // 32) + head) * 1088
            gram += tl.load(STATS + base + c[:, None] * 32 + c[None, :], valid, 0)
            qn += tl.load(STATS + base + 1024 + c, valid, 0)
            kn += tl.load(STATS + base + 1056 + c, valid, 0)
    count = (tl.minimum(y0 + 32, H) - tl.maximum(y0, 0)) * (
        tl.minimum(x0 + 32, W) - tl.maximum(x0, 0)
    )
    qn, kn = tl.sqrt(qn + count * 1e-6), tl.sqrt(kn + count * 1e-6)
    logits = gram / (qn[:, None] * kn[None, :]) * tl.load(TAU + head)
    p = tl.exp(logits - tl.max(logits, 1)[:, None])
    p = p / tl.sum(p, 1)[:, None]
    tl.store(A + (r * (C // 32) + head) * 1024 + c[:, None] * 32 + c[None, :], p)


@tr.jit
def apply_dw(
    X,
    WEIGHT,
    A,
    OUT,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    BT: tl.constexpr,
):
    cell, head, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cy, cx = cell // tr.cdiv(W, 16), cell % tr.cdiv(W, 16)
    t = part * BT + tl.arange(0, BT)
    y, x = cy * 16 + t // 16, cx * 16 + t % 16
    c = tl.arange(0, 32)
    v = convolve(X, WEIGHT, y, x, 2 * C + head * 32 + c, H, W, C)
    result = tl.full((32, BT), 0, tl.float32)
    for cls in tl.static_range(4):
        sy, sx = (cls // 2) * 16, (cls % 2) * 16
        nx = (W + sx + 31) // 32
        start = (
            0
            if cls == 0
            else (N0 if cls == 1 else (N0 + N1 if cls == 2 else N0 + N1 + N2))
        )
        rid = start + ((cy * 16 + sy) // 32) * nx + (cx * 16 + sx) // 32
        a = tl.load(A + (rid * (C // 32) + head) * 1024 + c[:, None] * 32 + c[None, :])
        out = tl.dot(a, v).to(tl.bfloat16).to(tl.float32)
        wy = 1.0 - tl.abs(((y + sy) % 32).to(tl.float32) - 15.5) / 16.0
        wx = 1.0 - tl.abs(((x + sx) % 32).to(tl.float32) - 15.5) / 16.0
        result = result + (out * wy[None, :]) * wx[None, :]
    tl.store(
        OUT + (y[None, :] * W + x[None, :]) * C + head * 32 + c[:, None],
        result,
        ((y < H) & (x < W))[None, :],
    )


@tr.jit
def project_add(
    X,
    WEIGHT,
    RESIDUAL,
    OUT,
    PIXELS: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    BM: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BC)
    ks = tl.arange(0, 32)
    acc = tl.full((BM, BC), 0, tl.float32)
    for start in tl.static_range(0, C, 32):
        x = tl.load(
            X + rows[:, None] * C + (ks + start)[None, :], (rows < PIXELS)[:, None], 0
        )
        w = tl.load(
            WEIGHT + (ks + start)[:, None] * C + cols[None, :],
            (cols < C)[None, :],
            0,
        )
        acc += tl.dot(x, w)
    residual = tl.load(
        RESIDUAL + rows[:, None] * C + cols[None, :],
        (rows < PIXELS)[:, None] & (cols < C)[None, :],
        0,
    ).to(tl.float32)
    # Preserve the standalone projection's BF16 output before the residual add.
    result = acc.to(tl.bfloat16).to(tl.float32) + residual
    tl.store(
        OUT + rows[:, None] * C + cols[None, :],
        result,
        (rows < PIXELS)[:, None] & (cols < C)[None, :],
    )
