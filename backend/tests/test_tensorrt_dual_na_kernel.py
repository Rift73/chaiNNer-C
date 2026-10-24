from __future__ import annotations

import pytest
import torch

# Needs Triton (the engine build's environment); the numeric checks also need CUDA.
dual_na_kernel = pytest.importorskip("nodes.impl.tensorrt.dual_na_kernel")
triton = pytest.importorskip("triton")

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="needs a CUDA GPU with BF16 tensor cores",
)

POINTERS = {
    "qkv_ptr": "*bf16",
    "bq_ptr": "*bf16",
    "bk_ptr": "*bf16",
    "height": "i32",
    "width": "i32",
    "out_ptr": "*bf16",
}


def factor_tables(
    heads: int, window: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """RFB factor tables over the (M + 15)^2 union coordinates, as DUAL makes them:
    FP32 trigonometry of random rank-8 frequencies and amplitudes, cast to BF16."""
    u = window + 15
    device = generator.device
    frequency = torch.randn(8, 2, device=device, generator=generator) * 0.5
    a = torch.randn(heads, 1, 8, device=device, generator=generator) * 0.5
    b = torch.randn(heads, 1, 8, device=device, generator=generator) * 0.5
    y, x = torch.meshgrid(
        torch.arange(u, device=device), torch.arange(u, device=device), indexing="ij"
    )
    phase = torch.stack((y, x), -1).reshape(-1, 2).float() @ frequency.T
    cos, sin = torch.cos(phase)[None], torch.sin(phase)[None]
    bq = torch.cat((a * cos + b * sin, a * sin - b * cos), -1)
    bk = torch.cat((cos, sin), -1).expand(heads, -1, -1)
    return bq.to(torch.bfloat16), bk.contiguous().to(torch.bfloat16)


def trunk(
    height: int, width: int, heads: int, pattern: str, generator: torch.Generator
) -> torch.Tensor:
    """qkv [1, H, W, 3C] with unit-scale random, constant or checkerboard values."""
    shape = (1, height, width, 3 * heads * 32)
    device = generator.device
    x = torch.randn(shape, device=device, generator=generator).to(torch.bfloat16)
    if pattern == "random":
        return x
    x = x[:, :1, :1].expand(shape).clone()
    if pattern == "checker":
        y, xx = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        sign = ((y + xx) % 2 * 2 - 1).to(torch.bfloat16)
        x = x * sign[None, :, :, None]
    return x


def oracle(
    qkv: torch.Tensor, bq: torch.Tensor, bk: torch.Tensor, window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The NA operator in FP64 from its definition: each query's own Kh x Kw
    inward-clamped neighborhood (no key tiles or minimal unions); the RFB tables are
    read in the 16 x 16 query tile's union frame. The operands are the kernel's: BF16
    Q times 1/sqrt(32) rounded to BF16, BF16 K, V and factors. Also returns the
    softmax-weighted mean of |V|, the size of what each output element sums."""
    _, height, width, c3 = qkv.shape
    heads = c3 // 96
    u = window + 15
    kh, kw = min(height, window), min(width, window)
    uh, uw = min(height, kh + 15), min(width, kw + 15)
    raw = qkv[0].reshape(height, width, 3, heads, 32)
    q = (raw[:, :, 0].float() * 0.1767766952966369).to(torch.bfloat16).double()
    k, v = raw[:, :, 1].double(), raw[:, :, 2].double()
    bq, bk = bq.double(), bk.double()
    device = qkv.device
    out = torch.empty(height, width, heads, 32, dtype=torch.float64, device=device)
    magnitude = torch.empty_like(out)
    for ty in range(0, height, 16):
        for tx in range(0, width, 16):
            yq = torch.arange(ty, min(ty + 16, height), device=device)
            xq = torch.arange(tx, min(tx + 16, width), device=device)
            sy = (yq - kh // 2).clamp(0, height - kh)
            sx = (xq - kw // 2).clamp(0, width - kw)
            by = min(max(ty - kh // 2, 0), height - uh)
            bx = min(max(tx - kw // 2, 0), width - uw)
            # The tile's queries' neighborhoods' bounding box; every key must lie in
            # the union frame, or the tables could not give its factors.
            r0, r1 = int(sy[0]), int(sy[-1]) + kh
            c0, c1 = int(sx[0]), int(sx[-1]) + kw
            assert by <= r0 and r1 <= by + uh and bx <= c0 and c1 <= bx + uw
            ky, kx = (
                t.reshape(-1)
                for t in torch.meshgrid(
                    torch.arange(r0, r1, device=device),
                    torch.arange(c0, c1, device=device),
                    indexing="ij",
                )
            )
            qy, qx = (t.reshape(-1) for t in torch.meshgrid(yq, xq, indexing="ij"))
            qsy, qsx = (t.reshape(-1) for t in torch.meshgrid(sy, sx, indexing="ij"))
            mask = (
                (ky[None] >= qsy[:, None])
                & (ky[None] < qsy[:, None] + kh)
                & (kx[None] >= qsx[:, None])
                & (kx[None] < qsx[:, None] + kw)
            )
            assert bool((mask.sum(1) == kh * kw).all())
            qf = torch.cat(
                (q[qy, qx], bq[:, (qy - by) * u + qx - bx].transpose(0, 1)), -1
            )
            kf = torch.cat(
                (k[ky, kx], bk[:, (ky - by) * u + kx - bx].transpose(0, 1)), -1
            )
            logits = torch.einsum("qhd,khd->hqk", qf, kf)
            weights = torch.softmax(logits.masked_fill(~mask[None], -torch.inf), -1)
            out[qy, qx] = torch.einsum("hqk,khd->qhd", weights, v[ky, kx])
            magnitude[qy, qx] = torch.einsum("hqk,khd->qhd", weights, v[ky, kx].abs())
    shape = (1, height, width, heads * 32)
    return out.reshape(shape), magnitude.reshape(shape)


def run(
    qkv: torch.Tensor,
    bq: torch.Tensor,
    bk: torch.Tensor,
    window: int,
    minimal: bool = True,
) -> torch.Tensor:
    _, height, width, c3 = qkv.shape
    channels = c3 // 3
    # NaN-filled, so an element the kernel never stores shows up.
    out = torch.full(
        (1, height, width, channels),
        torch.nan,
        device=qkv.device,
        dtype=torch.bfloat16,
    )
    g = dual_na_kernel.grid(height, width, channels, triton.cdiv)
    dual_na_kernel.na_attention[g](
        qkv,
        bq,
        bk,
        height,
        width,
        out,
        channels=channels,
        window=window,
        **{**dual_na_kernel.LAUNCH, "minimal": minimal},
        **dual_na_kernel.OPTIONS,
    )
    return out


def assert_matches_the_oracle(
    out: torch.Tensor,
    qkv: torch.Tensor,
    bq: torch.Tensor,
    bk: torch.Tensor,
    window: int,
):
    """Every element within the worst case of the kernel's own rounding: BF16 (unit
    roundoff 2^-8) on each softmax probability in the value product and on the output,
    plus FP32 slack. A wrong neighborhood, mask or bias frame moves outputs by a sizable
    fraction of |V| instead."""
    assert not out.isnan().any()
    exact, magnitude = oracle(qkv, bq, bk, window)
    unit = 2.0**-8
    bound = unit * exact.abs() + (1 + unit) * (unit * magnitude + 1e-5)
    error = (out.double() - exact).abs()
    assert bool((error <= bound).all()), f"error / bound {(error / bound).max()}"


def test_grid_covers_every_16x16_tile_and_head():
    assert dual_na_kernel.grid(40, 70, 160, triton.cdiv) == (3 * 5, 2, 5)
    assert dual_na_kernel.grid(256, 256, 128, triton.cdiv) == (256, 2, 4)


@pytest.mark.parametrize("window", [32, 64])
@pytest.mark.parametrize("channels", [128, 160, 256])
def test_compiles_ahead_of_time_within_the_plugin_limits(window: int, channels: int):
    # TensorRT's AOT plugins launch with at most 48 KiB of shared memory and no
    # Triton scratch buffers; dual_aot compiles with 16-byte aligned pointers.
    fn = dual_na_kernel.na_attention
    constants = {"channels": channels, "window": window, **dual_na_kernel.LAUNCH}
    source = triton.compiler.ASTSource(
        fn=fn,
        signature={**POINTERS, **dict.fromkeys(constants, "constexpr")},
        constexprs=constants,
        attrs={
            (fn.arg_names.index(name),): [["tt.divisibility", 16]]
            for name, kind in POINTERS.items()
            if kind.startswith("*")
        },
    )
    kernel = triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("cuda", 120, 32),
        options=dual_na_kernel.OPTIONS,
    )
    metadata = kernel.metadata
    assert metadata.shared <= 49152
    assert not metadata.global_scratch_size and not metadata.profile_scratch_size


@needs_cuda
@pytest.mark.parametrize(
    ("window", "height", "width", "heads"),
    [
        (32, 2, 100, 4),  # one axis shorter than a tile and than M
        (32, 31, 33, 5),  # partial tiles, both axes inside M + 15
        (32, 47, 47, 4),  # exactly M + 15: the fixed-size kernel's smallest trunk
        (32, 5, 70, 8),
        (64, 16, 16, 4),  # both axes within M: whole-trunk attention
        (64, 48, 200, 5),
        (64, 101, 37, 4),
        (64, 79, 80, 8),
    ],
)
def test_matches_the_fp64_oracle(window: int, height: int, width: int, heads: int):
    generator = torch.Generator(device="cuda").manual_seed(height * 1000 + width)
    bq, bk = factor_tables(heads, window, generator)
    qkv = trunk(height, width, heads, "random", generator)
    assert_matches_the_oracle(run(qkv, bq, bk, window), qkv, bq, bk, window)


@needs_cuda
@pytest.mark.parametrize(("window", "height", "width"), [(32, 31, 70), (64, 48, 80)])
def test_a_checkerboard_matches_the_fp64_oracle(window: int, height: int, width: int):
    generator = torch.Generator(device="cuda").manual_seed(7)
    bq, bk = factor_tables(4, window, generator)
    qkv = trunk(height, width, 4, "checker", generator)
    assert_matches_the_oracle(run(qkv, bq, bk, window), qkv, bq, bk, window)


@needs_cuda
@pytest.mark.parametrize(("window", "height", "width"), [(32, 40, 90), (64, 101, 37)])
def test_the_whole_tile_union_matches_the_fp64_oracle(
    window: int, height: int, width: int
):
    # minimal=False: every program visits its 16 x 16 tile's whole key union.
    generator = torch.Generator(device="cuda").manual_seed(3)
    bq, bk = factor_tables(4, window, generator)
    qkv = trunk(height, width, 4, "random", generator)
    out = run(qkv, bq, bk, window, minimal=False)
    assert_matches_the_oracle(out, qkv, bq, bk, window)


@needs_cuda
@pytest.mark.parametrize(("window", "height", "width"), [(32, 5, 70), (64, 79, 80)])
def test_a_constant_trunk_returns_its_value_exactly(
    window: int, height: int, width: int
):
    heads = 5
    generator = torch.Generator(device="cuda").manual_seed(11)
    bq, bk = factor_tables(heads, window, generator)
    qkv = trunk(height, width, heads, "constant", generator)
    out = run(qkv, bq, bk, window)
    value = qkv[..., 2 * heads * 32 :].contiguous()
    assert torch.equal(out.view(torch.int16), value.view(torch.int16))
