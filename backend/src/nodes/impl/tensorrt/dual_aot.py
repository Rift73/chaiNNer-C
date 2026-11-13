"""DUAL's TensorRT plugins as ahead-of-time Python plugins (TensorRT's tensorrt.plugin
API): Triton kernels compiled to PTX for the building GPU and embedded in the engine, so
neither a C++ compiler nor the CUDA Toolkit is needed. Core's and Project's kernels are
the C++ plugins' own (vendor/dual_tensorrt/scripts/dual_tensorrt/kernels.py, compiled
with aot.py's options); Norm and the attention barrier get Triton versions of their CUDA
code. An AOT plugin launches one kernel, so Core's three become three plugins whose
statistics and attention matrices are tensors. Neighborhood attention (DualNA, which has
no C++ plugin) launches dual_na_kernel.py's kernel, compiled with its own options.
In a static 128-channel model, four pointwise chains around the Norms (attention output
projection or DualProject to FFN expand, FFN contraction to the next Norm, attention gate
to the next core's qkv) become one plugin each (dual_chain_kernels.py, fuse_chains).

The exported ONNX keeps the guide's C++ plugin nodes. TensorRT 11.2's ONNX parser mixes
up Python plugin instances (a layer calls another plugin's functions), so `lower`
replaces each plugin node with stand-in operators of the same output shapes and types,
the parser reads that graph, and `attach` adds the plugins through the network API and
moves the stand-ins' consumers onto them; the builder drops the unused stand-ins. Used
only by the engine builds' child processes, dual_engine_worker.py and, through
rdual_aot.py, rdual_engine_worker.py (needs Triton).

tensorrt.plugin reads the registered functions' annotations as objects: `resolved`
evaluates the postponed ones first."""

from __future__ import annotations

import importlib.util
import typing
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Tuple, TypeVar, Union

import numpy as np
import onnx
import tensorrt as trt
import tensorrt.plugin as trtp
import triton
import triton.language as tl
from onnx import TensorProto, helper, numpy_helper
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from . import (
    dual_chain_kernels,
    dual_dynamic_kernels,
    dual_local_na_kernel,
    dual_na_kernel,
    dual_retrieval_na_kernel,
    dual_row_sum_kernel,
)

KERNELS = (
    Path(__file__).resolve().parents[3]
    / "vendor/dual_tensorrt/scripts/dual_tensorrt/kernels.py"
)
NAMESPACE = "dual_aot"
# What an aot_impl returns: kernel name, PTX, launch parameters, scalar arguments.
Launch = Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymIntExprs
]
# aot.py's compile options for the RCA and projection kernels
OPTIONS = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
# Split the value application over four programs per cell; each output keeps the
# same four-region accumulation order. Shared by fixed and dynamic engines.
APPLY_BT = 64
PROJECT_BM = 64
# The cell statistics' pixels per step: a cell's 256 in four chunks keeps the two
# convolution tiles in registers (one 256-pixel tile spilled).
CELL_BT = 64
NORM_PIXELS = 16
COPY_BLOCK = 1024
# The shared memory TensorRT's AOT plugin path launches with (a 57344-byte NA candidate
# failed there)
MAX_SHARED = 49152


@triton.jit
def norm_kernel(
    x_ptr: tl.tensor,
    gamma_ptr: tl.tensor,
    beta_ptr: tl.tensor,
    y_ptr: tl.tensor,
    sigma_ptr: tl.tensor,
    pixels: tl.constexpr,
    channels: tl.constexpr,
    block_c: tl.constexpr,
    block_p: tl.constexpr,
    eps: tl.constexpr,
):
    """Norm.cu without the residual variant: centered two-pass FP32 moments per pixel,
    the affine in FP32 (no fused multiply-add), BF16 out, sigma = sqrt(var + eps)."""
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


@triton.jit
def copy_kernel(
    x_ptr: tl.tensor, count: tl.int32, y_ptr: tl.tensor, block: tl.constexpr
):
    """The attention barrier: a bitwise copy that TensorRT cannot see through."""
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < count
    tl.store(y_ptr + offsets, tl.load(x_ptr + offsets, mask), mask)


@triton.jit
def router_add_kernel(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    count: tl.int32,
    out_ptr: tl.tensor,
    block: tl.constexpr,
) -> None:
    """Coalesced residual addition with the original BF16 rounding boundary."""
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < count
    a = tl.load(a_ptr + offsets, mask, 0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask, 0).to(tl.float32)
    tl.store(out_ptr + offsets, (a + b).to(tl.bfloat16), mask)


@cache
def _register_router_add(sm: int) -> None:
    @trtp.register(f"{NAMESPACE}::router_add")
    @resolved
    def router_add_shape(a: trtp.TensorDesc, b: trtp.TensorDesc) -> trtp.TensorDesc:
        return a.like()

    @trtp.aot_impl(f"{NAMESPACE}::router_add")
    @resolved
    def router_add_launch(
        a: trtp.TensorDesc,
        b: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        count = a.shape_expr.numel()
        kernel = compile_kernel(
            router_add_kernel,
            {"a_ptr": "*bf16", "b_ptr": "*bf16", "count": "i32", "out_ptr": "*bf16"},
            {"block": COPY_BLOCK},
            sm,
        )
        return kernel_launch(
            kernel, (trtp.max(trtp.cdiv(count, COPY_BLOCK), 1), 1, 1), [count]
        )


@cache
def vendored_kernels() -> ModuleType:
    """The vendored Triton kernels (a standalone module)."""
    spec = importlib.util.spec_from_file_location("dual_tensorrt_kernels", KERNELS)
    if spec is None or spec.loader is None:
        raise ImportError(KERNELS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = TypeVar("F", bound=Callable[..., Any])


def resolved(function: F) -> F:
    """The function with its postponed annotations evaluated (tensorrt.plugin reads
    them as objects)."""
    function.__annotations__ = typing.get_type_hints(function)
    return function


def compile_kernel(
    fn: Any,
    pointers: dict[str, str],
    constants: dict[str, Any],
    sm: int,
    options: dict[str, Any] = OPTIONS,
):
    signature = {**pointers, **dict.fromkeys(constants, "constexpr")}
    # TensorRT's buffers are 16-byte aligned. A launch tells Triton's JIT so; an AOT
    # compile must, or every load and store moves one element (the core ran ~14x slower).
    aligned = {
        (fn.arg_names.index(name),): [["tt.divisibility", 16]]
        for name, kind in pointers.items()
        if kind.startswith("*")
    }
    kernel = triton.compile(
        ASTSource(fn=fn, signature=signature, constexprs=constants, attrs=aligned),
        target=GPUTarget("cuda", sm, 32),
        options=options,
    )
    metadata = kernel.metadata
    if metadata.global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(f"{fn.__name__} needs Triton scratch memory")
    return kernel


@cache
def na_kernel(channels: int, window: int, sm: int) -> Any:
    """The neighborhood-attention kernel of a width and window, compiled once: static
    and dynamic engines launch the same kernel (the trunk's height and width are
    run-time arguments)."""
    kernel = compile_kernel(
        dual_na_kernel.na_attention_blocks,
        {
            "qkv_ptr": "*bf16",
            "bq_ptr": "*bf16",
            "bk_ptr": "*bf16",
            "height": "i32",
            "width": "i32",
            "out_ptr": "*bf16",
        },
        {"channels": channels, "window": window, **dual_na_kernel.BLOCKS_LAUNCH},
        sm,
        dual_na_kernel.BLOCKS_OPTIONS,
    )
    if kernel.metadata.shared > MAX_SHARED:
        raise RuntimeError(
            f"The neighborhood-attention kernel needs {kernel.metadata.shared} bytes of"
            f" shared memory; TensorRT's AOT plugins launch with at most {MAX_SHARED}"
        )
    return kernel


def _checked_shared(kernel: Any, label: str) -> Any:
    if kernel.metadata.shared > MAX_SHARED:
        raise RuntimeError(
            f"The {label} kernel needs {kernel.metadata.shared} bytes of shared memory;"
            f" TensorRT's AOT plugins launch with at most {MAX_SHARED}"
        )
    return kernel


@cache
def local_na_kernel(channels: int, sm: int) -> Any:
    """The local layers' neighborhood-attention kernel (dual_local_na_kernel), compiled
    once per width: the trunk's height and width are run-time arguments."""
    return _checked_shared(
        compile_kernel(
            dual_local_na_kernel.local_na,
            {
                "qkv_ptr": "*bf16",
                "table_ptr": "*fp32",
                "height": "i32",
                "width": "i32",
                "out_ptr": "*bf16",
            },
            {"channels": channels, **dual_local_na_kernel.LAUNCH},
            sm,
            dual_local_na_kernel.OPTIONS,
        ),
        "local neighborhood-attention",
    )


@cache
def retrieval_na_kernel(channels: int, sm: int) -> Any:
    """The retrieval's neighborhood-attention kernel (dual_retrieval_na_kernel), once
    per width: fine and coarse sizes are run-time arguments."""
    return _checked_shared(
        compile_kernel(
            dual_retrieval_na_kernel.xr_attention,
            {
                "q_ptr": "*bf16",
                "k_ptr": "*bf16",
                "v_ptr": "*bf16",
                "height": "i32",
                "width": "i32",
                "kheight": "i32",
                "kwidth": "i32",
                "out_ptr": "*bf16",
            },
            {"channels": channels, **dual_retrieval_na_kernel.LAUNCH},
            sm,
            dual_retrieval_na_kernel.OPTIONS,
        ),
        "retrieval neighborhood-attention",
    )


@cache
def row_sum_kernel(channels: int, sm: int) -> Any:
    """The router mean's FP32 row sums (dual_row_sum_kernel), once per width."""
    return _checked_shared(
        compile_kernel(
            dual_row_sum_kernel.row_sum,
            {"x_ptr": "*bf16", "width": "i32", "out_ptr": "*fp32"},
            {"channels": channels, **dual_row_sum_kernel.LAUNCH},
            sm,
            dual_row_sum_kernel.OPTIONS,
        ),
        "row-sum",
    )


Size = Union[int, trtp.ShapeExpr]


def kernel_launch(
    kernel: Any, grid: tuple[Size, Size, Size], extra: list[Size] | None = None
) -> Launch:
    params = trtp.KernelLaunchParams()
    params.grid_x, params.grid_y, params.grid_z = grid
    params.block_x = kernel.metadata.num_warps * 32
    params.shared_mem = kernel.metadata.shared
    extra_args = trtp.SymIntExprs(len(extra or []))
    for i, value in enumerate(extra or []):
        extra_args[i] = trtp.SymInt32(value)
    return kernel.metadata.name, kernel.asm["ptx"], params, extra_args


def _elements(desc: trtp.TensorDesc) -> int:
    """A static tensor's element count (DUAL engines have one fixed input shape)."""
    count = desc.shape_expr.numel()
    if not count.is_constant:
        raise ValueError("DUAL's TensorRT plugins need static shapes")
    return count.constant_value()


def device_sm() -> int:
    from cuda.bindings import runtime as cudart

    error, device = cudart.cudaGetDevice()
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGetDevice failed: {error}")
    error, properties = cudart.cudaGetDeviceProperties(device)
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGetDeviceProperties failed: {error}")
    return properties.major * 10 + properties.minor


@dataclass(frozen=True)
class Geometry:
    """A plugin key c<C>_h<H>_w<W>: width and trunk shape, and the RCA layout."""

    channels: int
    height: int
    width: int

    @staticmethod
    def of(key: str) -> Geometry:
        c, h, w = (int(part[1:]) for part in key.split("_"))
        return Geometry(c, h, w)

    @property
    def heads(self) -> int:
        return self.channels // 32

    @property
    def cells(self) -> int:
        return ((self.height + 15) // 16) * ((self.width + 15) // 16)

    @property
    def region_counts(self) -> list[int]:
        return [
            ((self.height + sy + 31) // 32) * ((self.width + sx + 31) // 32)
            for sy, sx in ((0, 0), (0, 16), (16, 0), (16, 16))
        ]

    @property
    def stats_size(self) -> int:
        return self.cells * self.heads * 1088

    @property
    def matrices_size(self) -> int:
        return sum(self.region_counts) * self.heads * 1024


@cache
def register(key: str) -> None:
    """Register this specialization's plugins (and the shared barrier) once."""
    g = Geometry.of(key)
    c, h, w = g.channels, g.height, g.width
    common = {"H": h, "W": w, "C": c}
    split = {
        **common,
        **dict(zip(("N0", "N1", "N2"), g.region_counts[:3], strict=True)),
    }
    pixels = h * w
    sm = device_sm()
    kernels = vendored_kernels()
    _register_router_add(sm)

    def plugin(name: str) -> str:
        return f"{NAMESPACE}::{name}_{key}"

    @trtp.register(plugin("norm"))
    @resolved
    def norm_shape(
        x: trtp.TensorDesc,
        gamma: trtp.TensorDesc,
        beta: trtp.TensorDesc,
        epsilon: float,
    ) -> tuple[trtp.TensorDesc, trtp.TensorDesc]:
        s = x.shape_expr
        return x.like(), trtp.from_shape_expr((s[0], s[1], s[2], 1), trt.float32)

    @trtp.aot_impl(plugin("norm"))
    @resolved
    def norm_launch(
        x: trtp.TensorDesc,
        gamma: trtp.TensorDesc,
        beta: trtp.TensorDesc,
        epsilon: float,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        count = _elements(x) // c
        kernel = compile_kernel(
            norm_kernel,
            {
                "x_ptr": "*bf16",
                "gamma_ptr": "*fp32",
                "beta_ptr": "*fp32",
                "y_ptr": "*bf16",
                "sigma_ptr": "*fp32",
            },
            {
                "pixels": count,
                "channels": c,
                "block_c": 1 << (c - 1).bit_length(),
                "block_p": NORM_PIXELS,
                "eps": epsilon,
            },
            sm,
        )
        return kernel_launch(kernel, ((count + NORM_PIXELS - 1) // NORM_PIXELS, 1, 1))

    @trtp.register(plugin("cell"))
    @resolved
    def cell_shape(qkv: trtp.TensorDesc, weight: trtp.TensorDesc) -> trtp.TensorDesc:
        return trtp.from_shape_expr((g.stats_size,), trt.float32)

    @trtp.aot_impl(plugin("cell"))
    @resolved
    def cell_launch(
        qkv: trtp.TensorDesc,
        weight: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        kernel = compile_kernel(
            kernels.cell_stats,
            {"X": "*bf16", "WEIGHT": "*bf16", "STATS": "*fp32"},
            {**common, "BT": CELL_BT},
            sm,
        )
        return kernel_launch(kernel, (g.cells, g.heads, 1))

    @trtp.register(plugin("region"))
    @resolved
    def region_shape(stats: trtp.TensorDesc, tau: trtp.TensorDesc) -> trtp.TensorDesc:
        return trtp.from_shape_expr((g.matrices_size,), trt.bfloat16)

    @trtp.aot_impl(plugin("region"))
    @resolved
    def region_launch(
        stats: trtp.TensorDesc,
        tau: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        kernel = compile_kernel(
            kernels.region_stats,
            {"STATS": "*fp32", "TAU": "*fp32", "A": "*bf16"},
            split,
            sm,
        )
        return kernel_launch(kernel, (sum(g.region_counts), g.heads, 1))

    @trtp.register(plugin("apply"))
    @resolved
    def apply_shape(
        qkv: trtp.TensorDesc, weight: trtp.TensorDesc, matrices: trtp.TensorDesc
    ) -> trtp.TensorDesc:
        s = qkv.shape_expr
        return trtp.from_shape_expr((s[0], s[1], s[2], c), trt.bfloat16)

    @trtp.aot_impl(plugin("apply"))
    @resolved
    def apply_launch(
        qkv: trtp.TensorDesc,
        weight: trtp.TensorDesc,
        matrices: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        kernel = compile_kernel(
            kernels.apply_dw,
            {"X": "*bf16", "WEIGHT": "*bf16", "A": "*bf16", "OUT": "*bf16"},
            {**split, "BT": APPLY_BT},
            sm,
        )
        return kernel_launch(kernel, (g.cells, g.heads, 256 // APPLY_BT))

    @trtp.register(plugin("project"))
    @resolved
    def project_shape(
        x: trtp.TensorDesc, weight: trtp.TensorDesc, residual: trtp.TensorDesc
    ) -> trtp.TensorDesc:
        return residual.like()

    @trtp.aot_impl(plugin("project"))
    @resolved
    def project_launch(
        x: trtp.TensorDesc,
        weight: trtp.TensorDesc,
        residual: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        kernel = compile_kernel(
            kernels.project_add,
            {"X": "*bf16", "WEIGHT": "*bf16", "RESIDUAL": "*bf16", "OUT": "*bf16"},
            {
                "PIXELS": pixels,
                "C": c,
                "BC": 1 << (c - 1).bit_length(),
                "BM": PROJECT_BM,
            },
            sm,
        )
        return kernel_launch(kernel, ((pixels + PROJECT_BM - 1) // PROJECT_BM, 1, 1))

    @trtp.register(plugin("na"))
    @resolved
    def na_shape(
        qkv: trtp.TensorDesc, bq: trtp.TensorDesc, bk: trtp.TensorDesc, window: int
    ) -> trtp.TensorDesc:
        s = qkv.shape_expr
        return trtp.from_shape_expr((s[0], s[1], s[2], c), trt.bfloat16)

    @trtp.aot_impl(plugin("na"))
    @resolved
    def na_launch(
        qkv: trtp.TensorDesc,
        bq: trtp.TensorDesc,
        bk: trtp.TensorDesc,
        window: int,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        return kernel_launch(
            na_kernel(c, window, sm),
            dual_na_kernel.blocks_grid(h, w, c, triton.cdiv),
            [h, w],
        )

    @trtp.register(plugin("lna"))
    @resolved
    def lna_shape(qkv: trtp.TensorDesc, table: trtp.TensorDesc) -> trtp.TensorDesc:
        s = qkv.shape_expr
        return trtp.from_shape_expr((s[0], s[1], s[2], c), trt.bfloat16)

    @trtp.aot_impl(plugin("lna"))
    @resolved
    def lna_launch(
        qkv: trtp.TensorDesc,
        table: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        # Local layers run at the trunk's size and at the coarse branch's: the site's own.
        height, width = qkv.shape_expr[1], qkv.shape_expr[2]
        return kernel_launch(
            local_na_kernel(c, sm),
            dual_local_na_kernel.grid(height, width, c, trtp.cdiv),
            [height, width],
        )

    @trtp.register(plugin("xr"))
    @resolved
    def xr_shape(
        q: trtp.TensorDesc, k: trtp.TensorDesc, v: trtp.TensorDesc
    ) -> trtp.TensorDesc:
        return q.like()

    @trtp.aot_impl(plugin("xr"))
    @resolved
    def xr_launch(
        q: trtp.TensorDesc,
        k: trtp.TensorDesc,
        v: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        return kernel_launch(
            retrieval_na_kernel(c, sm),
            dual_retrieval_na_kernel.grid(h, w, c, triton.cdiv),
            [h, w, h // 2, w // 2],
        )

    @trtp.register(plugin("rowsum"))
    @resolved
    def rowsum_shape(x: trtp.TensorDesc) -> trtp.TensorDesc:
        s = x.shape_expr
        return trtp.from_shape_expr((s[0], s[1], s[3]), trt.float32)

    @trtp.aot_impl(plugin("rowsum"))
    @resolved
    def rowsum_launch(
        x: trtp.TensorDesc, outputs: tuple[trtp.TensorDesc], tactic: int
    ) -> Launch:
        return kernel_launch(row_sum_kernel(c, sm), (h, 1, 1), [w])

    _register_barrier(sm)
    if c == dual_chain_kernels.CHANNELS:
        _register_chains(key, sm)


@cache
def _register_chains(key: str, sm: int) -> None:
    """Chains A's, B's, C's and D's plugins (dual_chain_kernels.py) of a static key."""

    def plugin(name: str) -> str:
        return f"{NAMESPACE}::{name}_{key}"

    def checked(kernel: Any) -> Any:
        if kernel.metadata.shared > MAX_SHARED:
            raise RuntimeError(
                f"{kernel.metadata.name} needs {kernel.metadata.shared} bytes of shared"
                f" memory; TensorRT's AOT plugins launch with at most {MAX_SHARED}"
            )
        return kernel

    def chain_a_plugin(symbol: str, project: bool) -> None:
        """Chain A's plugin, or with `project` chain C's (dual_chain_kernels.chain_a)."""

        @trtp.register(symbol)
        @resolved
        def chain_a_shape(
            g: trtp.TensorDesc,
            r: trtp.TensorDesc,
            w1: trtp.TensorDesc,
            b1: trtp.TensorDesc,
            gamma: trtp.TensorDesc,
            beta: trtp.TensorDesc,
            w2: trtp.TensorDesc,
            b2: trtp.TensorDesc,
            eps: float,
            c_div: float,
            c_add: float,
            c_mul: float,
        ) -> tuple[trtp.TensorDesc, trtp.TensorDesc]:
            s = r.shape_expr
            return r.like(), trtp.from_shape_expr((s[0], s[1], s[2], 256), trt.bfloat16)

        @trtp.aot_impl(symbol)
        @resolved
        def chain_a_launch(
            g: trtp.TensorDesc,
            r: trtp.TensorDesc,
            w1: trtp.TensorDesc,
            b1: trtp.TensorDesc,
            gamma: trtp.TensorDesc,
            beta: trtp.TensorDesc,
            w2: trtp.TensorDesc,
            b2: trtp.TensorDesc,
            eps: float,
            c_div: float,
            c_add: float,
            c_mul: float,
            outputs: tuple[trtp.TensorDesc],
            tactic: int,
        ) -> Launch:
            # The site's own pixels: the coarse branch runs at a quarter of the trunk's.
            pixels = _elements(r) // dual_chain_kernels.CHANNELS
            launch = dual_chain_kernels.CHAIN_A_LAUNCH
            kernel = compile_kernel(
                dual_chain_kernels.chain_a,
                {
                    **dict.fromkeys(("g_ptr", "r_ptr", "w1_ptr", "b1_ptr"), "*bf16"),
                    **dict.fromkeys(("gamma_ptr", "beta_ptr"), "*fp32"),
                    **dict.fromkeys(
                        ("w2_ptr", "b2_ptr", "res_ptr", "out_ptr"), "*bf16"
                    ),
                },
                {
                    "pixels": pixels,
                    "eps": eps,
                    "c_div": c_div,
                    "c_add": c_add,
                    "c_mul": c_mul,
                    **launch,
                    "project": project,
                },
                sm,
                dual_chain_kernels.OPTIONS,
            )
            blocks = (pixels + launch["bm"] - 1) // launch["bm"]
            return kernel_launch(checked(kernel), (blocks, 1, 1))

    chain_a_plugin(plugin("chaina"), False)
    chain_a_plugin(plugin("chainc"), True)

    @trtp.register(plugin("chainb"))
    @resolved
    def chain_b_shape(
        g: trtp.TensorDesc,
        r: trtp.TensorDesc,
        w: trtp.TensorDesc,
        b: trtp.TensorDesc,
        gamma: trtp.TensorDesc,
        beta: trtp.TensorDesc,
        eps: float,
    ) -> tuple[trtp.TensorDesc, trtp.TensorDesc]:
        return r.like(), r.like()

    @trtp.aot_impl(plugin("chainb"))
    @resolved
    def chain_b_launch(
        g: trtp.TensorDesc,
        r: trtp.TensorDesc,
        w: trtp.TensorDesc,
        b: trtp.TensorDesc,
        gamma: trtp.TensorDesc,
        beta: trtp.TensorDesc,
        eps: float,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        pixels = _elements(r) // dual_chain_kernels.CHANNELS
        launch = dual_chain_kernels.CHAIN_B_LAUNCH
        kernel = compile_kernel(
            dual_chain_kernels.chain_b,
            {
                **dict.fromkeys(("g_ptr", "r_ptr", "w_ptr", "b_ptr"), "*bf16"),
                **dict.fromkeys(("gamma_ptr", "beta_ptr"), "*fp32"),
                **dict.fromkeys(("res_ptr", "y_ptr"), "*bf16"),
            },
            {"pixels": pixels, "eps": eps, **launch},
            sm,
            dual_chain_kernels.OPTIONS,
        )
        blocks = (pixels + launch["bm"] - 1) // launch["bm"]
        return kernel_launch(checked(kernel), (blocks, 1, 1))

    @trtp.register(plugin("chaind"))
    @resolved
    def chain_d_shape(
        a: trtp.TensorDesc,
        wg: trtp.TensorDesc,
        bg: trtp.TensorDesc,
        na: trtp.TensorDesc,
        wp: trtp.TensorDesc,
        bp: trtp.TensorDesc,
        r: trtp.TensorDesc,
        gamma: trtp.TensorDesc,
        beta: trtp.TensorDesc,
        wq: trtp.TensorDesc,
        eps: float,
    ) -> tuple[trtp.TensorDesc, trtp.TensorDesc]:
        s = r.shape_expr
        return r.like(), trtp.from_shape_expr((s[0], s[1], s[2], 384), trt.bfloat16)

    @trtp.aot_impl(plugin("chaind"))
    @resolved
    def chain_d_launch(
        a: trtp.TensorDesc,
        wg: trtp.TensorDesc,
        bg: trtp.TensorDesc,
        na: trtp.TensorDesc,
        wp: trtp.TensorDesc,
        bp: trtp.TensorDesc,
        r: trtp.TensorDesc,
        gamma: trtp.TensorDesc,
        beta: trtp.TensorDesc,
        wq: trtp.TensorDesc,
        eps: float,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        pixels = _elements(r) // dual_chain_kernels.CHANNELS
        launch = dual_chain_kernels.CHAIN_D_LAUNCH
        names = (
            "a",
            "wg",
            "bg",
            "na",
            "wp",
            "bp",
            "r",
            "gamma",
            "beta",
            "wq",
            "res",
            "qkv",
        )
        kernel = compile_kernel(
            dual_chain_kernels.chain_d,
            {
                f"{name}_ptr": "*fp32" if name in ("gamma", "beta") else "*bf16"
                for name in names
            },
            {"pixels": pixels, "eps": eps, **launch},
            sm,
            dual_chain_kernels.OPTIONS,
        )
        blocks = (pixels + launch["bm"] - 1) // launch["bm"]
        return kernel_launch(checked(kernel), (blocks, 1, 1))


@cache
def _register_barrier(sm: int) -> None:
    @trtp.register(f"{NAMESPACE}::barrier")
    @resolved
    def barrier_shape(x: trtp.TensorDesc) -> trtp.TensorDesc:
        return x.like()

    @trtp.aot_impl(f"{NAMESPACE}::barrier")
    @resolved
    def barrier_launch(
        x: trtp.TensorDesc, outputs: tuple[trtp.TensorDesc], tactic: int
    ) -> Launch:
        count = _elements(x)
        kernel = compile_kernel(
            copy_kernel,
            {"x_ptr": "*bf16", "count": "i32", "y_ptr": "*bf16"},
            {"block": COPY_BLOCK},
            sm,
        )
        return kernel_launch(
            kernel, ((count + COPY_BLOCK - 1) // COPY_BLOCK, 1, 1), [count]
        )


def _trunk(
    qkv: trtp.TensorDesc,
) -> tuple[trtp.ShapeExpr, trtp.ShapeExpr, trtp.ShapeExpr, list[trtp.ShapeExpr]]:
    """A run-time trunk's height, width, RCA cell count and region counts."""
    s = qkv.shape_expr
    height, width = s[1], s[2]
    cells = trtp.cdiv(height, 16) * trtp.cdiv(width, 16)
    counts = [
        trtp.cdiv(height + sy, 32) * trtp.cdiv(width + sx, 32)
        for sy, sx in ((0, 0), (0, 16), (16, 0), (16, 16))
    ]
    return height, width, cells, counts


@cache
def register_dynamic(channels: int) -> None:
    """Register one width's dynamic-shape plugins (and the shared dynamic barrier)
    once: sizes, grids and scalar arguments are shape expressions of the inputs."""
    heads = channels // 32
    sm = device_sm()
    _register_router_add(sm)

    def plugin(name: str) -> str:
        return f"{NAMESPACE}::{name}_c{channels}"

    @trtp.register(plugin("norm"))
    @resolved
    def dynamic_norm_shape(
        x: trtp.TensorDesc,
        gamma: trtp.TensorDesc,
        beta: trtp.TensorDesc,
        epsilon: float,
    ) -> tuple[trtp.TensorDesc, trtp.TensorDesc]:
        s = x.shape_expr
        return x.like(), trtp.from_shape_expr((s[0], s[1], s[2], 1), trt.float32)

    @trtp.aot_impl(plugin("norm"))
    @resolved
    def dynamic_norm_launch(
        x: trtp.TensorDesc,
        gamma: trtp.TensorDesc,
        beta: trtp.TensorDesc,
        epsilon: float,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        s = x.shape_expr
        pixels = s[0] * s[1] * s[2]
        kernel = compile_kernel(
            dual_dynamic_kernels.norm,
            {
                "x_ptr": "*bf16",
                "gamma_ptr": "*fp32",
                "beta_ptr": "*fp32",
                "pixels": "i32",
                "y_ptr": "*bf16",
                "sigma_ptr": "*fp32",
            },
            {
                "channels": channels,
                "block_c": 1 << (channels - 1).bit_length(),
                "block_p": NORM_PIXELS,
                "eps": epsilon,
            },
            sm,
        )
        return kernel_launch(kernel, (trtp.cdiv(pixels, NORM_PIXELS), 1, 1), [pixels])

    @trtp.register(plugin("cell"))
    @resolved
    def dynamic_cell_shape(
        qkv: trtp.TensorDesc, weight: trtp.TensorDesc
    ) -> trtp.TensorDesc:
        _, _, cells, _ = _trunk(qkv)
        return trtp.from_shape_expr((cells * (heads * 1088),), trt.float32)

    @trtp.aot_impl(plugin("cell"))
    @resolved
    def dynamic_cell_launch(
        qkv: trtp.TensorDesc,
        weight: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        height, width, cells, _ = _trunk(qkv)
        kernel = compile_kernel(
            dual_dynamic_kernels.cell_stats,
            {
                "x_ptr": "*bf16",
                "weight_ptr": "*bf16",
                "height": "i32",
                "width": "i32",
                "stats_ptr": "*fp32",
            },
            {"channels": channels, "block": 256},
            sm,
        )
        return kernel_launch(kernel, (cells, heads, 1), [height, width])

    @trtp.register(plugin("region"))
    @resolved
    def dynamic_region_shape(
        stats: trtp.TensorDesc, tau: trtp.TensorDesc, qkv: trtp.TensorDesc
    ) -> trtp.TensorDesc:
        _, _, _, counts = _trunk(qkv)
        regions = counts[0] + counts[1] + counts[2] + counts[3]
        return trtp.from_shape_expr((regions * (heads * 1024),), trt.bfloat16)

    @trtp.aot_impl(plugin("region"))
    @resolved
    def dynamic_region_launch(
        stats: trtp.TensorDesc,
        tau: trtp.TensorDesc,
        qkv: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        height, width, _, counts = _trunk(qkv)
        kernel = compile_kernel(
            dual_dynamic_kernels.region_stats,
            {
                "stats_ptr": "*fp32",
                "tau_ptr": "*fp32",
                "qkv_ptr": "*bf16",
                "height": "i32",
                "width": "i32",
                "n0": "i32",
                "n1": "i32",
                "n2": "i32",
                "matrices_ptr": "*bf16",
            },
            {"channels": channels},
            sm,
        )
        regions = counts[0] + counts[1] + counts[2] + counts[3]
        return kernel_launch(kernel, (regions, heads, 1), [height, width, *counts[:3]])

    @trtp.register(plugin("apply"))
    @resolved
    def dynamic_apply_shape(
        qkv: trtp.TensorDesc, weight: trtp.TensorDesc, matrices: trtp.TensorDesc
    ) -> trtp.TensorDesc:
        s = qkv.shape_expr
        return trtp.from_shape_expr((s[0], s[1], s[2], channels), trt.bfloat16)

    @trtp.aot_impl(plugin("apply"))
    @resolved
    def dynamic_apply_launch(
        qkv: trtp.TensorDesc,
        weight: trtp.TensorDesc,
        matrices: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        height, width, cells, counts = _trunk(qkv)
        kernel = compile_kernel(
            dual_dynamic_kernels.apply_dw,
            {
                "x_ptr": "*bf16",
                "weight_ptr": "*bf16",
                "matrices_ptr": "*bf16",
                "height": "i32",
                "width": "i32",
                "n0": "i32",
                "n1": "i32",
                "n2": "i32",
                "out_ptr": "*bf16",
            },
            {"channels": channels, "block": APPLY_BT},
            sm,
        )
        return kernel_launch(
            kernel,
            (cells, heads, 256 // APPLY_BT),
            [height, width, *counts[:3]],
        )

    @trtp.register(plugin("project"))
    @resolved
    def dynamic_project_shape(
        x: trtp.TensorDesc, weight: trtp.TensorDesc, residual: trtp.TensorDesc
    ) -> trtp.TensorDesc:
        return residual.like()

    @trtp.aot_impl(plugin("project"))
    @resolved
    def dynamic_project_launch(
        x: trtp.TensorDesc,
        weight: trtp.TensorDesc,
        residual: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        s = x.shape_expr
        pixels = s[0] * s[1] * s[2]
        kernel = compile_kernel(
            dual_dynamic_kernels.project_add,
            {
                "x_ptr": "*bf16",
                "weight_ptr": "*bf16",
                "residual_ptr": "*bf16",
                "pixels": "i32",
                "out_ptr": "*bf16",
            },
            {
                "channels": channels,
                "block_c": 1 << (channels - 1).bit_length(),
                "block_m": PROJECT_BM,
            },
            sm,
        )
        return kernel_launch(kernel, (trtp.cdiv(pixels, PROJECT_BM), 1, 1), [pixels])

    @trtp.register(plugin("na"))
    @resolved
    def dynamic_na_shape(
        qkv: trtp.TensorDesc, bq: trtp.TensorDesc, bk: trtp.TensorDesc, window: int
    ) -> trtp.TensorDesc:
        s = qkv.shape_expr
        return trtp.from_shape_expr((s[0], s[1], s[2], channels), trt.bfloat16)

    @trtp.aot_impl(plugin("na"))
    @resolved
    def dynamic_na_launch(
        qkv: trtp.TensorDesc,
        bq: trtp.TensorDesc,
        bk: trtp.TensorDesc,
        window: int,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        height, width, _, _ = _trunk(qkv)
        return kernel_launch(
            na_kernel(channels, window, sm),
            dual_na_kernel.blocks_grid(height, width, channels, trtp.cdiv),
            [height, width],
        )

    @trtp.register(plugin("lna"))
    @resolved
    def dynamic_lna_shape(
        qkv: trtp.TensorDesc, table: trtp.TensorDesc
    ) -> trtp.TensorDesc:
        s = qkv.shape_expr
        return trtp.from_shape_expr((s[0], s[1], s[2], channels), trt.bfloat16)

    @trtp.aot_impl(plugin("lna"))
    @resolved
    def dynamic_lna_launch(
        qkv: trtp.TensorDesc,
        table: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        height, width, _, _ = _trunk(qkv)
        return kernel_launch(
            local_na_kernel(channels, sm),
            dual_local_na_kernel.grid(height, width, channels, trtp.cdiv),
            [height, width],
        )

    @trtp.register(plugin("xr"))
    @resolved
    def dynamic_xr_shape(
        q: trtp.TensorDesc, k: trtp.TensorDesc, v: trtp.TensorDesc
    ) -> trtp.TensorDesc:
        return q.like()

    @trtp.aot_impl(plugin("xr"))
    @resolved
    def dynamic_xr_launch(
        q: trtp.TensorDesc,
        k: trtp.TensorDesc,
        v: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        height, width = q.shape_expr[1], q.shape_expr[2]
        return kernel_launch(
            retrieval_na_kernel(channels, sm),
            dual_retrieval_na_kernel.grid(height, width, channels, trtp.cdiv),
            [height, width, k.shape_expr[1], k.shape_expr[2]],
        )

    @trtp.register(plugin("rowsum"))
    @resolved
    def dynamic_rowsum_shape(x: trtp.TensorDesc) -> trtp.TensorDesc:
        s = x.shape_expr
        return trtp.from_shape_expr((s[0], s[1], s[3]), trt.float32)

    @trtp.aot_impl(plugin("rowsum"))
    @resolved
    def dynamic_rowsum_launch(
        x: trtp.TensorDesc, outputs: tuple[trtp.TensorDesc], tactic: int
    ) -> Launch:
        height, width = x.shape_expr[1], x.shape_expr[2]
        return kernel_launch(row_sum_kernel(channels, sm), (height, 1, 1), [width])

    _register_dynamic_barrier(sm)


@cache
def _register_dynamic_barrier(sm: int) -> None:
    @trtp.register(f"{NAMESPACE}::barrier_dynamic")
    @resolved
    def dynamic_barrier_shape(x: trtp.TensorDesc) -> trtp.TensorDesc:
        return x.like()

    @trtp.aot_impl(f"{NAMESPACE}::barrier_dynamic")
    @resolved
    def dynamic_barrier_launch(
        x: trtp.TensorDesc, outputs: tuple[trtp.TensorDesc], tactic: int
    ) -> Launch:
        count = x.shape_expr.numel()
        kernel = compile_kernel(
            copy_kernel,
            {"x_ptr": "*bf16", "count": "i32", "y_ptr": "*bf16"},
            {"block": COPY_BLOCK},
            sm,
        )
        # An absent retrieval class is empty: launch one block, which copies nothing
        # (a launch of zero blocks is an error).
        blocks = trtp.max(trtp.cdiv(count, COPY_BLOCK), 1)
        return kernel_launch(kernel, (blocks, 1, 1), [count])


@dataclass
class Plugin:
    """One plugin to attach: its op, input and output tensor names, attributes."""

    op: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Lowered:
    model: onnx.ModelProto
    plugins: list[Plugin]
    # reader layer name -> the plugin input it reads
    readers: dict[str, str]


def _lower_router_additions(lowered: Lowered, channels: int) -> None:
    """Materialize router residual sums without replacing native FP32 means.

    Splitting this addition prevents an expensive strided Add/Transpose/Cast/Mean
    fusion. Only provably same-shaped BF16 NHWC operands qualify; broadcasts and
    unresolved shapes keep the native graph. No preset or output-scale gates.
    """
    graph = lowered.model.graph
    inferred = onnx.shape_inference.infer_shapes(lowered.model)
    shapes = {}
    for value in [
        *inferred.graph.value_info,
        *inferred.graph.input,
        *inferred.graph.output,
    ]:
        tensor = value.type.tensor_type
        if tensor.elem_type != TensorProto.BFLOAT16:
            continue
        shape = tuple(
            dim.dim_value if dim.HasField("dim_value") else dim.dim_param or None
            for dim in tensor.shape.dim
        )
        if len(shape) == 4 and shape[-1] == channels and all(shape):
            shapes[value.name] = shape
    nodes = {output: node for node in graph.node for output in node.output}
    constants = {
        value.name: numpy_helper.to_array(value) for value in graph.initializer
    }
    seen: set[str] = set()
    for node in list(graph.node):
        if node.op_type != "ReduceMean" or len(node.input) != 2:
            continue
        axes = constants.get(node.input[1])
        if axes is None or axes.tolist() != [2, 3]:
            continue
        cast = nodes.get(node.input[0])
        if cast is None or cast.op_type != "Cast":
            continue
        if {a.name: helper.get_attribute_value(a) for a in cast.attribute}.get(
            "to"
        ) != TensorProto.FLOAT:
            continue
        transpose = nodes.get(cast.input[0])
        if transpose is None or transpose.op_type != "Transpose":
            continue
        if {a.name: helper.get_attribute_value(a) for a in transpose.attribute}.get(
            "perm"
        ) != [0, 3, 1, 2]:
            continue
        addition = nodes.get(transpose.input[0])
        if addition is None or addition.op_type != "Add" or len(addition.input) != 2:
            continue
        output = addition.output[0]
        shape = shapes.get(addition.input[0])
        if output in seen or shape is None or shapes.get(addition.input[1]) != shape:
            continue
        for index, source in enumerate(addition.input):
            reader = f"{output}/router_add_reader_{index}"
            graph.node.append(
                helper.make_node("Identity", [source], [reader], name=reader)
            )
            lowered.readers[reader] = source
        # Attach before direct plugin consumers so they also see the replacement;
        # native consumers are rewired by attach's original network inventory.
        position = next(
            (i for i, plugin in enumerate(lowered.plugins) if output in plugin.inputs),
            len(lowered.plugins),
        )
        lowered.plugins.insert(
            position, Plugin("router_add", list(addition.input), [output])
        )
        seen.add(output)


def _scalar(tensor: onnx.TensorProto) -> float:
    """A one-element constant's value (BF16 ones included, which numpy cannot read)."""
    if tensor.data_type == TensorProto.BFLOAT16:
        if tensor.raw_data:
            raw = np.frombuffer(tensor.raw_data, np.uint16).copy()
        else:
            raw = np.array(tensor.int32_data, np.uint16)
        return float((raw.astype(np.uint32) << 16).view(np.float32).reshape(-1)[0])
    return float(numpy_helper.to_array(tensor).reshape(-1)[0])


class _Sites:
    """What the chain fusions read of a lowered model: its constants, each tensor's
    producer and node readers, and the plugins that read each tensor."""

    def __init__(self, lowered: Lowered) -> None:
        graph = lowered.model.graph
        self.consts: dict[str, onnx.TensorProto] = {
            t.name: t for t in graph.initializer
        }
        self.consts.update(
            {
                n.output[0]: n.attribute[0].t
                for n in graph.node
                if n.op_type == "Constant"
            }
        )
        self.producer = {o: n for n in graph.node for o in n.output}
        self.outputs = {output.name for output in graph.output}
        self.users: dict[str, list[onnx.NodeProto]] = {}
        for node in graph.node:
            for name in node.input:
                self.users.setdefault(name, []).append(node)
        self.plugin_users: dict[str, list[Plugin]] = {}
        for plugin in lowered.plugins:
            for name in plugin.inputs:
                self.plugin_users.setdefault(name, []).append(plugin)

    def dims(self, name: str) -> list[int] | None:
        return list(self.consts[name].dims) if name in self.consts else None

    def only_user(self, tensor: str, op: str) -> onnx.NodeProto | None:
        users = self.users.get(tensor, [])
        if (
            len(users) == 1
            and users[0].op_type == op
            and not self.plugin_users.get(tensor)
        ):
            return users[0]
        return None

    def unused(self, tensor: str) -> bool:
        return not (
            self.users.get(tensor)
            or self.plugin_users.get(tensor)
            or tensor in self.outputs
        )

    def bias_matmul(
        self, tensor: str, weight: list[int], bias: list[int]
    ) -> tuple[onnx.NodeProto, onnx.NodeProto] | None:
        """Add(MatMul(x, W of `weight`'s shape), b of `bias`'s) producing `tensor`, the
        sum and the product each read once."""
        add = self.producer.get(tensor)
        if (
            add is None
            or add.op_type != "Add"
            or self.dims(add.input[1]) != bias
            or len(self.users.get(tensor, [])) != 1
        ):
            return None
        matmul = self.producer.get(add.input[0])
        if (
            matmul is None
            or matmul.op_type != "MatMul"
            or self.dims(matmul.input[1]) != weight
            or len(self.users.get(matmul.output[0], [])) != 1
        ):
            return None
        return add, matmul

    def projection(
        self, residual: onnx.NodeProto | None, weight: list[int], bias: list[int]
    ) -> tuple[str, onnx.NodeProto, onnx.NodeProto] | None:
        """The residual addition's operand that is a bias_matmul, its bias addition and
        product, when exactly one is; its other operand is the residual."""
        if residual is None or residual.op_type != "Add":
            return None
        found = []
        for name in residual.input:
            site = self.bias_matmul(name, weight, bias)
            if site is not None:
                found.append((name, *site))
        return found[0] if len(found) == 1 else None

    def gelu(
        self, h: str
    ) -> tuple[onnx.NodeProto, onnx.NodeProto, onnx.NodeProto, onnx.NodeProto] | None:
        """(h * c_mul) * (erf(h / c_div) + c_add) of the exported GELU: its Div, Add, Mul
        by c_mul, and the output Mul."""
        readers = self.users.get(h, [])
        div = next((n for n in readers if n.op_type == "Div"), None)
        scale = next((n for n in readers if n.op_type == "Mul"), None)
        if (
            len(readers) != 2
            or div is None
            or scale is None
            or self.plugin_users.get(h)
        ):
            return None
        erf = self.only_user(div.output[0], "Erf")
        add = None if erf is None else self.only_user(erf.output[0], "Add")
        out = None if add is None else self.only_user(add.output[0], "Mul")
        if (
            out is None
            or add is None
            or self.only_user(scale.output[0], "Mul") is not out
        ):
            return None
        return div, add, scale, out


def _add_readers(lowered: Lowered, label: str, tensors: list[str]) -> None:
    for index, tensor in enumerate(tensors):
        reader = f"{NAMESPACE}/{label}_reader_{index}"
        lowered.model.graph.node.append(
            helper.make_node("Identity", [tensor], [reader], name=reader)
        )
        lowered.readers[reader] = tensor


def _fuse_chain_a(lowered: Lowered, key: str) -> int:
    """Chain A sites (dual_chain_kernels.chain_a) as one plugin each:
    g -> MatMul(W1 [C, C]) -> Add(b1) -> Add(r, .) = s -> Norm (sigma unused) ->
    MatMul(W2 [C, 2C]) -> Add(b2) = h -> GELU. The plugin takes (g, r, W1, b1, gamma,
    beta, W2, b2) and gives (s, the GELU output); the replaced nodes stay as stand-ins,
    dead once `attach` moves their readers to the plugin. Chain C sites, where s is a
    DualProject plugin's output (x, W1, r), become one plugin with that one's place
    (x, r, W1, W1 for the unread bias, gamma, beta, W2, b2)."""
    sites = _Sites(lowered)
    c = dual_chain_kernels.CHANNELS
    projects = {p.outputs[0]: p for p in lowered.plugins if p.op.startswith("project_")}
    fused_projects: set[int] = set()
    kept: list[Plugin] = []
    count = 0
    for plugin in lowered.plugins:
        if not plugin.op.startswith("norm_"):
            kept.append(plugin)
            continue
        s, y, sigma = plugin.inputs[0], plugin.outputs[0], plugin.outputs[1]
        found = sites.projection(sites.producer.get(s), [c, c], [c])
        expand = sites.only_user(y, "MatMul")
        bias = None
        if expand is not None and sites.dims(expand.input[1]) == [c, 2 * c]:
            bias = sites.only_user(expand.output[0], "Add")
        if bias is not None and sites.dims(bias.input[1]) != [2 * c]:
            bias = None
        gelu = None if bias is None else sites.gelu(bias.output[0])
        project = projects.get(s)
        if sites.unused(sigma) and project is not None and gelu is not None:
            assert expand is not None and bias is not None
            div, add, scale, out = gelu
            x, w1, r = project.inputs
            weights = [expand.input[1], bias.input[1]]
            kept.append(
                Plugin(
                    f"chainc_{key}",
                    [x, r, w1, w1, plugin.inputs[1], plugin.inputs[2], *weights],
                    [s, out.output[0]],
                    {
                        "eps": float(plugin.attributes["epsilon"]),
                        "c_div": _scalar(sites.consts[div.input[1]]),
                        "c_add": _scalar(sites.consts[add.input[1]]),
                        "c_mul": _scalar(sites.consts[scale.input[1]]),
                    },
                )
            )
            fused_projects.add(id(project))
            _add_readers(lowered, f"chainc_{count}", weights)
            count += 1
            continue
        if not sites.unused(sigma) or found is None or gelu is None:
            kept.append(plugin)
            continue
        assert expand is not None and bias is not None
        source, bias_add, matmul = found
        residual = sites.producer[s]
        r = next(name for name in residual.input if name != source)
        div, add, scale, out = gelu
        c_div, c_add, c_mul = (
            _scalar(sites.consts[n.input[1]]) for n in (div, add, scale)
        )
        weights = [matmul.input[1], bias_add.input[1], expand.input[1], bias.input[1]]
        kept.append(
            Plugin(
                f"chaina_{key}",
                [
                    matmul.input[0],
                    r,
                    weights[0],
                    weights[1],
                    plugin.inputs[1],
                    plugin.inputs[2],
                    weights[2],
                    weights[3],
                ],
                [s, out.output[0]],
                {
                    "eps": float(plugin.attributes["epsilon"]),
                    "c_div": c_div,
                    "c_add": c_add,
                    "c_mul": c_mul,
                },
            )
        )
        _add_readers(lowered, f"chaina_{count}", weights)
        count += 1
    lowered.plugins[:] = [p for p in kept if id(p) not in fused_projects]
    return count


def _fuse_chain_b(lowered: Lowered, key: str) -> int:
    """Chain B sites (dual_chain_kernels.chain_b) as one plugin each, after chain A,
    whose Norms are no longer Norm plugins: g [.., 2C] -> MatMul(W [2C, C]) -> Add(b)
    -> Add(r, .) = s -> Norm (sigma unused) = y; (g, r, W, b, gamma, beta) -> (s, y)."""
    sites = _Sites(lowered)
    c = dual_chain_kernels.CHANNELS
    kept: list[Plugin] = []
    count = 0
    for plugin in lowered.plugins:
        if not plugin.op.startswith("norm_"):
            kept.append(plugin)
            continue
        s, sigma = plugin.inputs[0], plugin.outputs[1]
        found = sites.projection(sites.producer.get(s), [2 * c, c], [c])
        if not sites.unused(sigma) or found is None:
            kept.append(plugin)
            continue
        source, bias_add, matmul = found
        r = next(name for name in sites.producer[s].input if name != source)
        weights = [matmul.input[1], bias_add.input[1]]
        kept.append(
            Plugin(
                f"chainb_{key}",
                [matmul.input[0], r, *weights, plugin.inputs[1], plugin.inputs[2]],
                [s, plugin.outputs[0]],
                {"eps": float(plugin.attributes["epsilon"])},
            )
        )
        _add_readers(lowered, f"chainb_{count}", weights)
        count += 1
    lowered.plugins[:] = kept
    return count


def _fuse_chain_d(lowered: Lowered, key: str) -> int:
    """Chain D sites (dual_chain_kernels.chain_d) as one plugin each, after chain A:
    a -> MatMul(Wg [C, C]) -> Add(bg) -> Sigmoid -> Mul(na, .) -> MatMul(Wp [C, C]) ->
    Add(bp) -> Add(r, .) = s -> Norm (sigma unused) = y -> MatMul(Wq [C, 3C]) = qkv;
    (a, Wg, bg, na, Wp, bp, r, gamma, beta, Wq) -> (s, qkv). Every intermediate is read
    once, by the next layer only."""
    sites = _Sites(lowered)
    c = dual_chain_kernels.CHANNELS

    def single(tensor: str) -> bool:
        return len(sites.users.get(tensor, [])) == 1 and not sites.plugin_users.get(
            tensor
        )

    def linear(tensor: str) -> tuple[str, str, str] | None:
        """(x, W, b) of tensor = Add(MatMul(x, W [C, C]), b [C])."""
        add = sites.producer.get(tensor)
        if (
            add is None
            or add.op_type != "Add"
            or sites.dims(add.input[1]) != [c]
            or not single(tensor)
        ):
            return None
        matmul = sites.producer.get(add.input[0])
        if (
            matmul is None
            or matmul.op_type != "MatMul"
            or sites.dims(matmul.input[1]) != [c, c]
            or not single(matmul.output[0])
        ):
            return None
        return matmul.input[0], matmul.input[1], add.input[1]

    kept: list[Plugin] = []
    count = 0
    for plugin in lowered.plugins:
        if not plugin.op.startswith("norm_"):
            kept.append(plugin)
            continue
        s, y, sigma = plugin.inputs[0], plugin.outputs[0], plugin.outputs[1]
        residual = sites.producer.get(s)
        hit = None
        if sites.unused(sigma) and residual is not None and residual.op_type == "Add":
            for name in residual.input:
                projection = linear(name)
                if projection is None:
                    continue
                mul = sites.producer.get(projection[0])
                if mul is None or mul.op_type != "Mul" or not single(projection[0]):
                    continue
                for attended, gated in (
                    (mul.input[0], mul.input[1]),
                    (mul.input[1], mul.input[0]),
                ):
                    sigmoid = sites.producer.get(gated)
                    if (
                        sigmoid is None
                        or sigmoid.op_type != "Sigmoid"
                        or not single(gated)
                    ):
                        continue
                    gate = linear(sigmoid.input[0])
                    if gate is None:
                        continue
                    readers = sites.users.get(y, [])
                    if (
                        len(readers) == 1
                        and readers[0].op_type == "MatMul"
                        and sites.dims(readers[0].input[1]) == [c, 3 * c]
                        and not sites.plugin_users.get(y)
                    ):
                        hit = (name, projection, attended, gate, readers[0])
                    break
        if hit is None or residual is None:
            kept.append(plugin)
            continue
        source, projection, attended, gate, qkv = hit
        r = next(name for name in residual.input if name != source)
        weights = [gate[1], gate[2], projection[1], projection[2], qkv.input[1]]
        kept.append(
            Plugin(
                f"chaind_{key}",
                [
                    gate[0],
                    gate[1],
                    gate[2],
                    attended,
                    projection[1],
                    projection[2],
                    r,
                    plugin.inputs[1],
                    plugin.inputs[2],
                    qkv.input[1],
                ],
                [s, qkv.output[0]],
                {"eps": float(plugin.attributes["epsilon"])},
            )
        )
        _add_readers(lowered, f"chaind_{count}", weights)
        count += 1
    lowered.plugins[:] = kept
    return count


def fuse_chains(lowered: Lowered, key: str) -> tuple[int, int, int]:
    """Chain A's, chain D's, then chain B's sites as plugins (the order the kernels were
    measured in), for a static key of the kernels' width (a dynamic or other-width model
    is left as it is); the sites fused of each."""
    if "_h" not in key or Geometry.of(key).channels != dual_chain_kernels.CHANNELS:
        return 0, 0, 0
    a = _fuse_chain_a(lowered, key)
    d = _fuse_chain_d(lowered, key)
    return a, _fuse_chain_b(lowered, key), d


def lower(model: onnx.ModelProto, key: str, chains: bool = True) -> Lowered:
    """The model with each of the exporter's plugin nodes (the guide's and DualNA)
    replaced by stand-in operators of the same output shapes and types, and the
    plugins to attach in their place; with `chains`, also the chain sites
    (fuse_chains), which a caller adding plugins of its own fuses after them.
    A width-only key (c<C>) is a dynamic-shape model: only plugins read the core's
    statistics and matrices, so their stand-ins are single elements."""
    dynamic = "_h" not in key
    channels = int(key.partition("_")[0][1:])
    stats_size, matrices_size = (
        (1, 1)
        if dynamic
        else (Geometry.of(key).stats_size, Geometry.of(key).matrices_size)
    )
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    plugins: list[Plugin] = []

    def constant(name: str, values: np.ndarray) -> str:
        initializers.append(numpy_helper.from_array(values, name))
        return name

    def zeros(output: str, size: int, dtype: int) -> None:
        shape = constant(f"{output}/shape", np.array([size], np.int64))
        nodes.append(helper.make_node("ConstantOfShape", [shape], [output + "/f32"]))
        nodes.append(helper.make_node("Cast", [output + "/f32"], [output], to=dtype))

    def identity(source: str, output: str) -> None:
        nodes.append(helper.make_node("Identity", [source], [output]))

    for node in model.graph.node:
        op, name = node.op_type, node.name
        inputs, outputs = list(node.input), list(node.output)
        if op == f"DualNorm_{key}_TRT":
            attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if attributes.get("residual", 0) != 0:
                raise ValueError("DUAL's residual Norm variant is not supported")
            identity(inputs[0], outputs[0])
            starts = constant(f"{name}/starts", np.array([0], np.int64))
            ends = constant(f"{name}/ends", np.array([1], np.int64))
            axes = constant(f"{name}/axes", np.array([3], np.int64))
            nodes.append(
                helper.make_node(
                    "Slice", [inputs[0], starts, ends, axes], [outputs[1] + "/bf16"]
                )
            )
            nodes.append(
                helper.make_node(
                    "Cast", [outputs[1] + "/bf16"], [outputs[1]], to=TensorProto.FLOAT
                )
            )
            plugins.append(
                Plugin(
                    f"norm_{key}",
                    inputs,
                    outputs,
                    {"epsilon": float(attributes["epsilon"])},
                )
            )
        elif op == f"DualCore_{key}_TRT":
            qkv, weight, tau = inputs
            stats, matrices = f"{name}/stats", f"{name}/matrices"
            zeros(stats, stats_size, TensorProto.FLOAT)
            zeros(matrices, matrices_size, TensorProto.BFLOAT16)
            starts = constant(f"{name}/starts", np.array([0], np.int64))
            ends = constant(f"{name}/ends", np.array([channels], np.int64))
            axes = constant(f"{name}/axes", np.array([3], np.int64))
            nodes.append(helper.make_node("Slice", [qkv, starts, ends, axes], outputs))
            # The dynamic region plugin reads the trunk's size from qkv's shape.
            region_inputs = [stats, tau, qkv] if dynamic else [stats, tau]
            plugins += [
                Plugin(f"cell_{key}", [qkv, weight], [stats]),
                Plugin(f"region_{key}", region_inputs, [matrices]),
                Plugin(f"apply_{key}", [qkv, weight, matrices], outputs),
            ]
        elif op == f"DualProject_{key}_TRT":
            identity(inputs[2], outputs[0])
            plugins.append(Plugin(f"project_{key}", inputs, outputs))
        elif op == f"DualNA_{key}_TRT":
            # Inputs qkv, bq, bk; the output has qkv's first C channels' shape.
            attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            starts = constant(f"{name}/starts", np.array([0], np.int64))
            ends = constant(f"{name}/ends", np.array([channels], np.int64))
            axes = constant(f"{name}/axes", np.array([3], np.int64))
            nodes.append(
                helper.make_node("Slice", [inputs[0], starts, ends, axes], outputs)
            )
            plugins.append(
                Plugin(
                    f"na_{key}", inputs, outputs, {"window": int(attributes["window"])}
                )
            )
        elif op == f"DualLocalNA_{key}_TRT":
            # Inputs qkv, table; the output has qkv's first C channels' shape.
            starts = constant(f"{name}/starts", np.array([0], np.int64))
            ends = constant(f"{name}/ends", np.array([channels], np.int64))
            axes = constant(f"{name}/axes", np.array([3], np.int64))
            nodes.append(
                helper.make_node("Slice", [inputs[0], starts, ends, axes], outputs)
            )
            plugins.append(Plugin(f"lna_{key}", inputs, outputs))
        elif op == f"DualXR_{key}_TRT":
            # Inputs q, k, v; the output has q's shape.
            attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            neighbourhood = dual_retrieval_na_kernel.LAUNCH["neighbourhood"]
            if int(attributes["neighbourhood"]) != neighbourhood:
                raise ValueError(
                    "DUAL's retrieval plugin takes a neighborhood of"
                    f" {neighbourhood}, not {attributes['neighbourhood']}"
                )
            identity(inputs[0], outputs[0])
            plugins.append(Plugin(f"xr_{key}", inputs, outputs))
        elif op == f"DualRowSum_{key}_TRT":
            # Input x [1, H, W, C] BF16; the output, its FP32 sums over W [1, H, C].
            axes = constant(f"{name}/axes", np.array([2], np.int64))
            nodes.append(
                helper.make_node(
                    "Cast", [inputs[0]], [outputs[0] + "/f32"], to=TensorProto.FLOAT
                )
            )
            nodes.append(
                helper.make_node(
                    "ReduceSum", [outputs[0] + "/f32", axes], outputs, keepdims=0
                )
            )
            plugins.append(Plugin(f"rowsum_{key}", inputs, outputs))
        elif op == "DualAttentionBarrier_TRT":
            identity(inputs[0], outputs[0])
            barrier = "barrier_dynamic" if dynamic else "barrier"
            plugins.append(Plugin(barrier, inputs, outputs))
        elif op.startswith("Dual") and op.endswith("_TRT"):
            raise ValueError(f"Unexpected DUAL plugin node {op}")
        else:
            nodes.append(node)
    # The parser makes a tensor of an initializer only for a node that reads it, and
    # names it its own way: give every plugin input a named reader (unused, so the
    # builder drops it), through which `attach` finds the tensor.
    produced = {name for plugin in plugins for name in plugin.outputs}
    needed = {name for plugin in plugins for name in plugin.inputs} - produced
    readers = {}
    for index, name in enumerate(sorted(needed)):
        reader = f"{NAMESPACE}/reader_{index}"
        nodes.append(helper.make_node("Identity", [name], [reader], name=reader))
        readers[reader] = name
    lowered = onnx.ModelProto()
    lowered.CopyFrom(model)
    del lowered.graph.node[:]
    lowered.graph.node.extend(nodes)
    lowered.graph.initializer.extend(initializers)
    result = Lowered(lowered, plugins, readers)
    _lower_router_additions(result, channels)
    if chains:
        fuse_chains(result, key)
    return result


def attach(network: trt.INetworkDefinition, lowered: Lowered) -> None:
    """Add the plugins to a network parsed from `lower`'s model, in graph order, and
    move every consumer of a stand-in output onto the plugin's output."""
    tensors: dict[str, trt.ITensor] = {}
    consumers: dict[str, list[tuple[trt.ILayer, int]]] = {}
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        tensors[tensor.name] = tensor
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        if layer.name in lowered.readers:
            read = layer.get_input(0)
            if read is None:
                raise RuntimeError(f"The reader {layer.name} lost its input")
            tensors[lowered.readers[layer.name]] = read
        for j in range(layer.num_inputs):
            tensor = layer.get_input(j)
            if tensor is not None:
                consumers.setdefault(tensor.name, []).append((layer, j))
        for j in range(layer.num_outputs):
            tensor = layer.get_output(j)
            tensors[tensor.name] = tensor
    outputs = {network.get_output(i).name for i in range(network.num_outputs)}
    namespace = getattr(trtp.op, NAMESPACE)
    for plugin in lowered.plugins:
        attributes = plugin.attributes
        call = getattr(namespace, plugin.op)
        layer = network.add_plugin(
            call(*(tensors[name] for name in plugin.inputs), **attributes)
        )
        layer.name = f"{plugin.op}/{plugin.outputs[0]}"
        for k, name in enumerate(plugin.outputs):
            stand_in = tensors[name]
            stand_in.name = name + "/stand_in"
            replacement = layer.get_output(k)
            replacement.name = name
            tensors[name] = replacement
            for consumer, j in consumers.get(name, []):
                consumer.set_input(j, replacement)
            if name in outputs:
                network.unmark_output(stand_in)
                network.mark_output(replacement)


__all__ = [
    "MAX_SHARED",
    "NAMESPACE",
    "Geometry",
    "Launch",
    "Lowered",
    "Plugin",
    "attach",
    "compile_kernel",
    "device_sm",
    "fuse_chains",
    "kernel_launch",
    "local_na_kernel",
    "lower",
    "na_kernel",
    "register",
    "register_dynamic",
    "resolved",
    "retrieval_na_kernel",
    "row_sum_kernel",
    "vendored_kernels",
]
