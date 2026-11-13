"""RDUAL's own TensorRT plugins as ahead-of-time Python plugins, beside DUAL's
(dual_aot.py), in its namespace and with its compile and launch helpers: Triton kernels
compiled to PTX for the building GPU and embedded in the engine.

- temporal_h<H>_w<W>_s<S>_r<R>: the temporal attention of the state fusion, RDUAL's own
  forward kernel (the model's file), over S donors in a radius R.
- flow_kv_h<H>_w<W>: the joint key/value warp of a donor's state along its flow, RDUAL's
  own forward kernel.
- corr_window_b<B>_h<H>_w<W>_<f16|f32> and corr_sample_*: the flow network's radius-4
  correlation lookup without the all-pairs volume. Per pixel the window plugin takes the
  11 x 11 integer neighbourhood that the 81 bilinear taps can touch, as dot products of
  the two channel-last feature maps (FP32 tree sums, then the FP16 rounding the
  volume's matmul gave in the mixed recipe); the sample plugin applies the lookup's
  1/sqrt(C) scale, rounding and bilinear blend to those 121 values. B*H*W x 128 values
  instead of a (H*W)^2 volume.
- match_b<B>_n<N>_c<C>: the global matching at 1/16, softmax(Q K^T / sqrt(C)) times the
  pixel coordinates, as an online softmax over 64-key blocks (3xTF32 dot products, FP32
  accurate), without the N x N matrix.

The exported ONNX holds them as RDUALFlowKV, RDUALTemporal, RDUALCorrWindow,
RDUALCorrSample and RDUALMatch nodes
(rdual_export.py). `lower` replaces them with stand-ins as dual_aot.lower does DUAL's,
then lowers DUAL's own (a trunk part; the flow part has none). Used only by
rdual_engine_worker.py (needs Triton)."""

from __future__ import annotations

from functools import cache

import numpy as np
import onnx
import tensorrt as trt
import tensorrt.plugin as trtp
import triton
import triton.language as tl
from onnx import TensorProto, helper, numpy_helper
from spandrel.architectures.RDUAL.__arch import rdual_arch

from . import dual_aot
from .dual_aot import Launch, resolved

if not rdual_arch.HAS_DIRECT_ATTENTION:
    raise ImportError(
        f"RDUAL's kernels are unavailable: {rdual_arch.DIRECT_ATTENTION_ERROR}"
    )

# The exporter's custom node types
FLOW_KV = "RDUALFlowKV"
TEMPORAL = "RDUALTemporal"
CORR_WINDOW = "RDUALCorrWindow"
CORR_SAMPLE = "RDUALCorrSample"
MATCH = "RDUALMatch"
NODES = (FLOW_KV, TEMPORAL, CORR_WINDOW, CORR_SAMPLE, MATCH)
# The temporal kernel's query tile and key block
TEMPORAL_TILE = 8
TEMPORAL_BN = 64
KV_BLOCK = 256
# Correlation positions per program; the window's neighbours per step
CORRELATION_BP = 4
WINDOW_SLOTS = 16
CORRELATION_OPTIONS = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
# The matching's query and key blocks
MATCH_LAUNCH = {"bm": 32, "bn": 64}


@triton.jit
def window_kernel(
    f0_ptr: tl.tensor,
    f1_ptr: tl.tensor,
    flow_ptr: tl.tensor,
    d_ptr: tl.tensor,
    height: tl.constexpr,
    width: tl.constexpr,
    batch: tl.constexpr,
    channels: tl.constexpr,
    block_p: tl.constexpr,
    slots: tl.constexpr,
    half: tl.constexpr,
) -> None:
    """D[p, j] = f0[p] . f1[n_j(p)] for the 11 x 11 neighbours n_j (origin floor(p +
    flow) - 5, zero outside the image) of channel-last features (B, H*W, C); C-long tree
    sums in FP32, rounded to FP16 with `half` as the volume's FP16 matmul output was."""
    p = tl.program_id(0) * block_p + tl.arange(0, block_p)
    valid = p < batch * height * width
    b = p // (height * width)
    xy = p % (height * width)
    cc = tl.arange(0, channels)
    fx = tl.load(flow_ptr + (b * 2) * height * width + xy, valid, 0)
    fy = tl.load(flow_ptr + (b * 2 + 1) * height * width + xy, valid, 0)
    ox = tl.floor(xy % width + fx).to(tl.int32) - 5
    oy = tl.floor(xy // width + fy).to(tl.int32) - 5
    a = tl.load(f0_ptr + (p[:, None] * channels + cc[None, :]), valid[:, None], 0).to(
        tl.float32
    )
    for s in tl.static_range(0, 128, slots):
        j = s + tl.arange(0, slots)
        nx = ox[:, None] + (j % 11)[None, :]
        ny = oy[:, None] + (j // 11)[None, :]
        inside = (
            valid[:, None]
            & (j < 121)[None, :]
            & (nx >= 0)
            & (nx < width)
            & (ny >= 0)
            & (ny < height)
        )
        q = tl.where(inside, b[:, None] * (height * width) + ny * width + nx, 0)
        v = tl.load(
            f1_ptr + (q[:, :, None] * channels + cc[None, None, :]),
            inside[:, :, None],
            0,
        ).to(tl.float32)
        dot = tl.sum(a[:, None, :] * v, axis=2)
        if half:
            dot = dot.to(tl.float16).to(tl.float32)
        tl.store(d_ptr + p[:, None] * 128 + j[None, :], dot, valid[:, None])


@triton.jit
def _in_window(lx: tl.tensor, ly: tl.tensor) -> tl.tensor:
    return (lx >= 0) & (lx < 11) & (ly >= 0) & (ly < 11)


@triton.jit
def sample_kernel(
    d_ptr: tl.tensor,
    flow_ptr: tl.tensor,
    out_ptr: tl.tensor,
    height: tl.constexpr,
    width: tl.constexpr,
    batch: tl.constexpr,
    block_p: tl.constexpr,
    inv: tl.constexpr,
    half: tl.constexpr,
) -> None:
    """The radius-4 bilinear lookup (zero borders, align_corners=True) reading the
    window's values: each tap scaled by inv and, with half, rounded to FP16 as the scaled
    half-precision volume held it."""
    p = tl.program_id(0) * block_p + tl.arange(0, block_p)
    j = tl.arange(0, 128)
    k = j % 9 * 9 + j // 9
    b = p // (height * width)
    xy = p % (height * width)
    valid = p < batch * height * width
    fx = tl.load(flow_ptr + (b * 2) * height * width + xy, valid, 0)
    fy = tl.load(flow_ptr + (b * 2 + 1) * height * width + xy, valid, 0)
    ox = tl.floor(xy % width + fx).to(tl.int32) - 5
    oy = tl.floor(xy // width + fy).to(tl.int32) - 5
    x = (xy % width + fx)[:, None] + (k // 9 - 4)[None, :]
    y = (xy // width + fy)[:, None] + (k % 9 - 4)[None, :]
    x = ((2 * x / (width - 1) - 1) + 1) * 0.5 * (width - 1)
    y = ((2 * y / (height - 1) - 1) + 1) * 0.5 * (height - 1)
    x0 = tl.floor(x).to(tl.int32)
    y0 = tl.floor(y).to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = x - x0
    wy = y - y0
    ok = valid[:, None] & (j[None, :] < 81)
    row = p[:, None] * 128
    lx0, lx1 = x0 - ox[:, None], x1 - ox[:, None]
    ly0, ly1 = y0 - oy[:, None], y1 - oy[:, None]
    in00 = (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height) & _in_window(lx0, ly0)
    in10 = (x1 >= 0) & (x1 < width) & (y0 >= 0) & (y0 < height) & _in_window(lx1, ly0)
    in01 = (x0 >= 0) & (x0 < width) & (y1 >= 0) & (y1 < height) & _in_window(lx0, ly1)
    in11 = (x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height) & _in_window(lx1, ly1)
    a = tl.load(d_ptr + row + ly0 * 11 + lx0, ok & in00, 0) * inv
    z = tl.load(d_ptr + row + ly0 * 11 + lx1, ok & in10, 0) * inv
    c = tl.load(d_ptr + row + ly1 * 11 + lx0, ok & in01, 0) * inv
    e = tl.load(d_ptr + row + ly1 * 11 + lx1, ok & in11, 0) * inv
    if half:
        a = a.to(tl.float16).to(tl.float32)
        z = z.to(tl.float16).to(tl.float32)
        c = c.to(tl.float16).to(tl.float32)
        e = e.to(tl.float16).to(tl.float32)
    v = a * ((1 - wx) * (1 - wy)) + z * (wx * (1 - wy)) + c * ((1 - wx) * wy)
    v = v + e * (wx * wy)
    tl.store(
        out_ptr + (b[:, None] * 81 + k[None, :]) * height * width + xy[:, None], v, ok
    )


@triton.jit
def match_kernel(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    g_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n: tl.constexpr,
    channels: tl.constexpr,
    bm: tl.constexpr,
    bn: tl.constexpr,
    scale: tl.constexpr,
    prec: tl.constexpr,
) -> None:
    """softmax(Q K^T * scale) @ G for G = pixel coordinates (n x 2), with a running
    maximum and sum over key blocks; Q, K (batch, n, channels), G and the output
    (batch, n, 2), all FP32."""
    pid, b = tl.program_id(0), tl.program_id(1)
    rows = pid * bm + tl.arange(0, bm)
    cc = tl.arange(0, channels)
    q = tl.load(
        q_ptr + b * n * channels + rows[:, None] * channels + cc[None, :],
        (rows < n)[:, None],
        0,
    )
    m = tl.full((bm,), float("-inf"), tl.float32)
    s_sum = tl.zeros((bm,), tl.float32)
    ax = tl.zeros((bm,), tl.float32)
    ay = tl.zeros((bm,), tl.float32)
    for start in range(0, n, bn):
        cols = start + tl.arange(0, bn)
        ok = cols < n
        k = tl.load(
            k_ptr + b * n * channels + cols[:, None] * channels + cc[None, :],
            ok[:, None],
            0,
        )
        s = tl.dot(q, tl.trans(k), input_precision=prec) * scale
        s = tl.where(ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        gx = tl.load(g_ptr + b * n * 2 + cols * 2, ok, 0)
        gy = tl.load(g_ptr + b * n * 2 + cols * 2 + 1, ok, 0)
        s_sum = s_sum * alpha + tl.sum(p, 1)
        ax = ax * alpha + tl.sum(p * gx[None, :], 1)
        ay = ay * alpha + tl.sum(p * gy[None, :], 1)
        m = m_new
    tl.store(out_ptr + b * n * 2 + rows * 2, ax / s_sum, rows < n)
    tl.store(out_ptr + b * n * 2 + rows * 2 + 1, ay / s_sum, rows < n)


def corr_names(batch: int, height: int, width: int, half: int) -> tuple[str, str]:
    tag = f"b{batch}_h{height}_w{width}_{'f16' if half else 'f32'}"
    return f"corr_window_{tag}", f"corr_sample_{tag}"


def match_name(batch: int, n: int, channels: int) -> str:
    return f"match_b{batch}_n{n}_c{channels}"


def temporal_name(height: int, width: int, slots: int, radius: int) -> str:
    return f"temporal_h{height}_w{width}_s{slots}_r{radius}"


def flow_kv_name(height: int, width: int) -> str:
    return f"flow_kv_h{height}_w{width}"


@cache
def register_corr(
    batch: int, height: int, width: int, channels: int, half: int
) -> None:
    window, sample = corr_names(batch, height, width, half)
    sm = dual_aot.device_sm()
    pixels = batch * height * width
    blocks = (pixels + CORRELATION_BP - 1) // CORRELATION_BP

    @trtp.register(f"{dual_aot.NAMESPACE}::{window}")
    @resolved
    def window_shape(
        f0: trtp.TensorDesc, f1: trtp.TensorDesc, flow: trtp.TensorDesc
    ) -> trtp.TensorDesc:
        return trtp.from_shape_expr((pixels, 128), trt.float32)

    @trtp.aot_impl(f"{dual_aot.NAMESPACE}::{window}")
    @resolved
    def window_launch(
        f0: trtp.TensorDesc,
        f1: trtp.TensorDesc,
        flow: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        features = "*fp16" if half else "*fp32"
        kernel = dual_aot.compile_kernel(
            window_kernel,
            {
                "f0_ptr": features,
                "f1_ptr": features,
                "flow_ptr": "*fp32",
                "d_ptr": "*fp32",
            },
            {
                "height": height,
                "width": width,
                "batch": batch,
                "channels": channels,
                "block_p": CORRELATION_BP,
                "slots": WINDOW_SLOTS,
                "half": bool(half),
            },
            sm,
            CORRELATION_OPTIONS,
        )
        if kernel.metadata.shared > dual_aot.MAX_SHARED:
            raise RuntimeError(
                f"The correlation window needs {kernel.metadata.shared} bytes of shared"
                f" memory; TensorRT's AOT plugins launch with at most"
                f" {dual_aot.MAX_SHARED}"
            )
        return dual_aot.kernel_launch(kernel, (blocks, 1, 1))

    @trtp.register(f"{dual_aot.NAMESPACE}::{sample}")
    @resolved
    def sample_shape(d: trtp.TensorDesc, flow: trtp.TensorDesc) -> trtp.TensorDesc:
        return trtp.from_shape_expr((batch, 81, height, width), trt.float32)

    @trtp.aot_impl(f"{dual_aot.NAMESPACE}::{sample}")
    @resolved
    def sample_launch(
        d: trtp.TensorDesc,
        flow: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        kernel = dual_aot.compile_kernel(
            sample_kernel,
            {"d_ptr": "*fp32", "flow_ptr": "*fp32", "out_ptr": "*fp32"},
            {
                "height": height,
                "width": width,
                "batch": batch,
                "block_p": CORRELATION_BP,
                "inv": 1.0 / channels**0.5,
                "half": bool(half),
            },
            sm,
            CORRELATION_OPTIONS,
        )
        return dual_aot.kernel_launch(kernel, (blocks, 1, 1))


@cache
def register_match(batch: int, n: int, channels: int) -> None:
    symbol = f"{dual_aot.NAMESPACE}::{match_name(batch, n, channels)}"
    sm = dual_aot.device_sm()

    @trtp.register(symbol)
    @resolved
    def match_shape(
        q: trtp.TensorDesc, k: trtp.TensorDesc, g: trtp.TensorDesc
    ) -> trtp.TensorDesc:
        return trtp.from_shape_expr((batch, n, 2), trt.float32)

    @trtp.aot_impl(symbol)
    @resolved
    def match_launch(
        q: trtp.TensorDesc,
        k: trtp.TensorDesc,
        g: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        kernel = dual_aot.compile_kernel(
            match_kernel,
            {"q_ptr": "*fp32", "k_ptr": "*fp32", "g_ptr": "*fp32", "out_ptr": "*fp32"},
            {
                "n": n,
                "channels": channels,
                **MATCH_LAUNCH,
                "scale": 1.0 / channels**0.5,
                "prec": "tf32x3",
            },
            sm,
            CORRELATION_OPTIONS,
        )
        if kernel.metadata.shared > dual_aot.MAX_SHARED:
            raise RuntimeError(
                f"The matching needs {kernel.metadata.shared} bytes of shared memory;"
                f" TensorRT's AOT plugins launch with at most {dual_aot.MAX_SHARED}"
            )
        bm = MATCH_LAUNCH["bm"]
        return dual_aot.kernel_launch(kernel, ((n + bm - 1) // bm, batch, 1))


@cache
def register_temporal(height: int, width: int, slots: int, radius: int) -> None:
    symbol = f"{dual_aot.NAMESPACE}::{temporal_name(height, width, slots, radius)}"
    sm = dual_aot.device_sm()

    @trtp.register(symbol)
    @resolved
    def temporal_shape(
        q: trtp.TensorDesc,
        k: trtp.TensorDesc,
        v: trtp.TensorDesc,
        table: trtp.TensorDesc,
        null_key: trtp.TensorDesc,
        null_bias: trtp.TensorDesc,
        slot_bias: trtp.TensorDesc,
        valid: trtp.TensorDesc,
    ) -> tuple[trtp.TensorDesc, trtp.TensorDesc]:
        e = q.shape_expr
        return (
            trtp.from_shape_expr((e[0], e[1], e[2], e[3]), trt.float32),
            trtp.from_shape_expr((e[0], e[1], e[2]), trt.float32),
        )

    @trtp.aot_impl(symbol)
    @resolved
    def temporal_launch(
        q: trtp.TensorDesc,
        k: trtp.TensorDesc,
        v: trtp.TensorDesc,
        table: trtp.TensorDesc,
        null_key: trtp.TensorDesc,
        null_bias: trtp.TensorDesc,
        slot_bias: trtp.TensorDesc,
        valid: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        tile = TEMPORAL_TILE
        kernel = dual_aot.compile_kernel(
            rdual_arch.TEMPORAL_FORWARD_KERNEL,
            {
                "Q": "*bf16",
                "K": "*bf16",
                "V": "*bf16",
                "R": "*fp32",
                "NK": "*bf16",
                "NB": "*fp32",
                "SB": "*fp32",
                "Valid": "*i1",
                "Output": "*fp32",
                "LSE": "*fp32",
            },
            {
                "H": height,
                "W": width,
                "S": slots,
                "RADIUS": radius,
                "UY": min(height, tile + 2 * radius),
                "UX": min(width, tile + 2 * radius),
                "TILE": tile,
                "BN": TEMPORAL_BN,
            },
            sm,
            {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": True},
        )
        if kernel.metadata.shared > dual_aot.MAX_SHARED:
            raise RuntimeError(
                f"RDUAL's temporal kernel needs {kernel.metadata.shared} bytes of shared"
                f" memory; TensorRT's AOT plugins launch with at most"
                f" {dual_aot.MAX_SHARED}"
            )
        tiles = ((height + tile - 1) // tile) * ((width + tile - 1) // tile)
        return dual_aot.kernel_launch(kernel, (tiles, 4, 1))


@cache
def register_flow_kv(height: int, width: int) -> None:
    symbol = f"{dual_aot.NAMESPACE}::{flow_kv_name(height, width)}"
    sm = dual_aot.device_sm()

    @trtp.register(symbol)
    @resolved
    def flow_kv_shape(x: trtp.TensorDesc, flow: trtp.TensorDesc) -> trtp.TensorDesc:
        return trtp.from_shape_expr((2, 1, 4, height * width, 32), trt.bfloat16)

    @trtp.aot_impl(symbol)
    @resolved
    def flow_kv_launch(
        x: trtp.TensorDesc,
        flow: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        pixels = height * width
        kernel = dual_aot.compile_kernel(
            rdual_arch.KV_WARP_FORWARD_KERNEL,
            {"X": "*bf16", "Flow": "*fp32", "Y": "*bf16"},
            {
                "H": height,
                "W": width,
                "N": 1,
                "S0": pixels * 256,
                "S1": 1,
                "S2": width * 256,
                "S3": 256,
                "BLOCK": KV_BLOCK,
            },
            sm,
        )
        count = 256 * pixels
        return dual_aot.kernel_launch(
            kernel, ((count + KV_BLOCK - 1) // KV_BLOCK, 1, 1)
        )


def _attributes(node: onnx.NodeProto) -> dict[str, int]:
    """A plugin node's integer attributes (its geometry), without the plugin's
    version and namespace strings."""
    return {a.name: a.i for a in node.attribute if a.type == onnx.AttributeProto.INT}


def register(model: onnx.ModelProto) -> None:
    """Register the plugins of the model's RDUAL nodes (once per geometry); needs the
    building GPU, whose architecture the kernels are compiled for."""
    for node in model.graph.node:
        if node.op_type not in NODES:
            continue
        a = _attributes(node)
        if node.op_type == FLOW_KV:
            register_flow_kv(a["height"], a["width"])
        elif node.op_type == TEMPORAL:
            register_temporal(a["height"], a["width"], a["slots"], a["radius"])
        elif node.op_type == MATCH:
            register_match(a["batch"], a["n"], a["channels"])
        else:
            register_corr(a["batch"], a["height"], a["width"], a["channels"], a["half"])


def lower(model: onnx.ModelProto, key: str | None) -> dual_aot.Lowered:
    """The model with RDUAL's plugin nodes replaced by stand-ins of their outputs'
    shapes and types, then DUAL's (dual_aot.lower) for a trunk part's plugin `key`;
    None for the flow part, which holds none of DUAL's. RDUAL's plugins join the
    plugin list in graph order, each input with a named reader, as dual_aot's. The
    plugins are registered apart (`register`, and dual_aot.register for `key`)."""
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    extra: list[dual_aot.Plugin] = []
    for node in model.graph.node:
        if node.op_type not in NODES:
            nodes.append(node)
            continue
        attributes = _attributes(node)
        inputs, outputs = list(node.input), list(node.output)
        if node.op_type == MATCH:
            # (B, N, 2) FP32, as the coordinates input
            nodes.append(helper.make_node("Identity", [inputs[2]], outputs))
            name = match_name(
                attributes["batch"], attributes["n"], attributes["channels"]
            )
            extra.append(dual_aot.Plugin(name, inputs, outputs))
            continue
        height, width = attributes["height"], attributes["width"]
        if node.op_type == FLOW_KV:
            shape = outputs[0] + "/shape"
            initializers.append(
                helper.make_tensor(
                    shape, TensorProto.INT64, [5], [2, 1, 4, height * width, 32]
                )
            )
            nodes.append(helper.make_node("Reshape", [inputs[0], shape], outputs))
            extra.append(dual_aot.Plugin(flow_kv_name(height, width), inputs, outputs))
        elif node.op_type == TEMPORAL:
            geometry = (height, width, attributes["slots"], attributes["radius"])
            nodes.append(
                helper.make_node(
                    "Cast", [inputs[0]], [outputs[0]], to=TensorProto.FLOAT
                )
            )
            axes = outputs[1] + "/axes"
            initializers.append(numpy_helper.from_array(np.array([-1], np.int64), axes))
            nodes.append(
                helper.make_node(
                    "ReduceSum", [outputs[0], axes], [outputs[1]], keepdims=0
                )
            )
            extra.append(dual_aot.Plugin(temporal_name(*geometry), inputs, outputs))
        else:
            batch = attributes["batch"]
            window, sample = corr_names(batch, height, width, attributes["half"])
            if node.op_type == CORR_WINDOW:
                # [B*H*W, 128] FP32, from the flow input [B, 2, H, W]
                pixels = batch * height * width
                summed, axis = outputs[0] + "/summed", outputs[0] + "/axis"
                column, wide = outputs[0] + "/column", outputs[0] + "/wide"
                initializers += [
                    helper.make_tensor(axis, TensorProto.INT64, [1], [1]),
                    helper.make_tensor(column, TensorProto.INT64, [2], [pixels, 1]),
                    helper.make_tensor(wide, TensorProto.INT64, [2], [pixels, 128]),
                ]
                nodes += [
                    helper.make_node(
                        "ReduceSum", [inputs[2], axis], [summed], keepdims=0
                    ),
                    helper.make_node("Reshape", [summed, column], [summed + "/r"]),
                    helper.make_node("Expand", [summed + "/r", wide], outputs),
                ]
                extra.append(dual_aot.Plugin(window, inputs, outputs))
            else:
                # [B, 81, H, W] FP32, from the flow input [B, 2, H, W]
                axis = outputs[0] + "/axis"
                summed = outputs[0] + "/summed"
                initializers.append(
                    helper.make_tensor(axis, TensorProto.INT64, [1], [1])
                )
                nodes.append(
                    helper.make_node(
                        "ReduceSum", [inputs[1], axis], [summed], keepdims=1
                    )
                )
                nodes.append(helper.make_node("Concat", [summed] * 81, outputs, axis=1))
                extra.append(dual_aot.Plugin(sample, inputs, outputs))
    clean = onnx.ModelProto()
    clean.CopyFrom(model)
    del clean.graph.node[:]
    clean.graph.node.extend(nodes)
    clean.graph.initializer.extend(initializers)
    lowered = (
        dual_aot.Lowered(clean, [], {})
        if key is None
        else dual_aot.lower(clean, key, chains=False)
    )
    for index, name in enumerate(sorted({n for p in extra for n in p.inputs})):
        reader = f"rdual/reader_{index}"
        lowered.model.graph.node.append(
            helper.make_node("Identity", [name], [reader], name=reader)
        )
        lowered.readers[reader] = name
    order = {
        output: index
        for index, node in enumerate(lowered.model.graph.node)
        for output in node.output
    }
    for plugin in extra:
        position = next(
            (
                index
                for index, other in enumerate(lowered.plugins)
                if order[other.outputs[0]] > order[plugin.outputs[0]]
            ),
            len(lowered.plugins),
        )
        lowered.plugins.insert(position, plugin)
    # DUAL's chain sites last, so that a tensor RDUAL's plugins read stays outside them
    if key is not None:
        dual_aot.fuse_chains(lowered, key)
    return lowered


__all__ = [
    "CORR_SAMPLE",
    "CORR_WINDOW",
    "FLOW_KV",
    "MATCH",
    "TEMPORAL",
    "corr_names",
    "flow_kv_name",
    "lower",
    "match_name",
    "register",
    "temporal_name",
]
