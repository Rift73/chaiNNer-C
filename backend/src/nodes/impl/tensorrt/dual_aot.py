"""DUAL's TensorRT plugins as ahead-of-time Python plugins (TensorRT's tensorrt.plugin
API): Triton kernels compiled to PTX for the building GPU and embedded in the engine, so
neither a C++ compiler nor the CUDA Toolkit is needed. Core's and Project's kernels are
the C++ plugins' own (vendor/dual_tensorrt/scripts/dual_tensorrt/kernels.py, compiled
with aot.py's options); Norm and the attention barrier get Triton versions of their CUDA
code. An AOT plugin launches one kernel, so Core's three become three plugins whose
statistics and attention matrices are tensors.

The exported ONNX keeps the guide's C++ plugin nodes. TensorRT 11.2's ONNX parser mixes
up Python plugin instances (a layer calls another plugin's functions), so `lower`
replaces each plugin node with stand-in operators of the same output shapes and types,
the parser reads that graph, and `attach` adds the plugins through the network API and
moves the stand-ins' consumers onto them; the builder drops the unused stand-ins. Used
only by dual_engine_worker.py (needs Triton).

tensorrt.plugin reads the registered functions' annotations as objects: `_resolved`
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
APPLY_BT = 256
PROJECT_BM = 64
NORM_PIXELS = 16
COPY_BLOCK = 1024


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


@cache
def _kernels() -> ModuleType:
    """The vendored Triton kernels (a standalone module)."""
    spec = importlib.util.spec_from_file_location("dual_tensorrt_kernels", KERNELS)
    if spec is None or spec.loader is None:
        raise ImportError(KERNELS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = TypeVar("F", bound=Callable[..., Any])


def _resolved(function: F) -> F:
    """The function with its postponed annotations evaluated (tensorrt.plugin reads
    them as objects)."""
    function.__annotations__ = typing.get_type_hints(function)
    return function


def _compile(fn: Any, pointers: dict[str, str], constants: dict[str, Any], sm: int):
    signature = {**pointers, **dict.fromkeys(constants, "constexpr")}
    kernel = triton.compile(
        ASTSource(fn=fn, signature=signature, constexprs=constants),
        target=GPUTarget("cuda", sm, 32),
        options=OPTIONS,
    )
    metadata = kernel.metadata
    if metadata.global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(f"{fn.__name__} needs Triton scratch memory")
    return kernel


def _launch(
    kernel: Any, grid: tuple[int, int, int], extra: list[int] | None = None
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


def _sm() -> int:
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
    sm = _sm()
    kernels = _kernels()

    def plugin(name: str) -> str:
        return f"{NAMESPACE}::{name}_{key}"

    @trtp.register(plugin("norm"))
    @_resolved
    def norm_shape(
        x: trtp.TensorDesc,
        gamma: trtp.TensorDesc,
        beta: trtp.TensorDesc,
        epsilon: float,
    ) -> tuple[trtp.TensorDesc, trtp.TensorDesc]:
        s = x.shape_expr
        return x.like(), trtp.from_shape_expr((s[0], s[1], s[2], 1), trt.float32)

    @trtp.aot_impl(plugin("norm"))
    @_resolved
    def norm_launch(
        x: trtp.TensorDesc,
        gamma: trtp.TensorDesc,
        beta: trtp.TensorDesc,
        epsilon: float,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        count = _elements(x) // c
        kernel = _compile(
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
        return _launch(kernel, ((count + NORM_PIXELS - 1) // NORM_PIXELS, 1, 1))

    @trtp.register(plugin("cell"))
    @_resolved
    def cell_shape(qkv: trtp.TensorDesc, weight: trtp.TensorDesc) -> trtp.TensorDesc:
        return trtp.from_shape_expr((g.stats_size,), trt.float32)

    @trtp.aot_impl(plugin("cell"))
    @_resolved
    def cell_launch(
        qkv: trtp.TensorDesc,
        weight: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        kernel = _compile(
            kernels.cell_stats,
            {"X": "*bf16", "WEIGHT": "*bf16", "STATS": "*fp32"},
            {**common, "BT": 256},
            sm,
        )
        return _launch(kernel, (g.cells, g.heads, 1))

    @trtp.register(plugin("region"))
    @_resolved
    def region_shape(stats: trtp.TensorDesc, tau: trtp.TensorDesc) -> trtp.TensorDesc:
        return trtp.from_shape_expr((g.matrices_size,), trt.bfloat16)

    @trtp.aot_impl(plugin("region"))
    @_resolved
    def region_launch(
        stats: trtp.TensorDesc,
        tau: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        kernel = _compile(
            kernels.region_stats,
            {"STATS": "*fp32", "TAU": "*fp32", "A": "*bf16"},
            split,
            sm,
        )
        return _launch(kernel, (sum(g.region_counts), g.heads, 1))

    @trtp.register(plugin("apply"))
    @_resolved
    def apply_shape(
        qkv: trtp.TensorDesc, weight: trtp.TensorDesc, matrices: trtp.TensorDesc
    ) -> trtp.TensorDesc:
        s = qkv.shape_expr
        return trtp.from_shape_expr((s[0], s[1], s[2], c), trt.bfloat16)

    @trtp.aot_impl(plugin("apply"))
    @_resolved
    def apply_launch(
        qkv: trtp.TensorDesc,
        weight: trtp.TensorDesc,
        matrices: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        kernel = _compile(
            kernels.apply_dw,
            {"X": "*bf16", "WEIGHT": "*bf16", "A": "*bf16", "OUT": "*bf16"},
            {**split, "BT": APPLY_BT},
            sm,
        )
        return _launch(kernel, (g.cells, g.heads, 256 // APPLY_BT))

    @trtp.register(plugin("project"))
    @_resolved
    def project_shape(
        x: trtp.TensorDesc, weight: trtp.TensorDesc, residual: trtp.TensorDesc
    ) -> trtp.TensorDesc:
        return residual.like()

    @trtp.aot_impl(plugin("project"))
    @_resolved
    def project_launch(
        x: trtp.TensorDesc,
        weight: trtp.TensorDesc,
        residual: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        kernel = _compile(
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
        return _launch(kernel, ((pixels + PROJECT_BM - 1) // PROJECT_BM, 1, 1))

    _register_barrier(sm)


@cache
def _register_barrier(sm: int) -> None:
    @trtp.register(f"{NAMESPACE}::barrier")
    @_resolved
    def barrier_shape(x: trtp.TensorDesc) -> trtp.TensorDesc:
        return x.like()

    @trtp.aot_impl(f"{NAMESPACE}::barrier")
    @_resolved
    def barrier_launch(
        x: trtp.TensorDesc, outputs: tuple[trtp.TensorDesc], tactic: int
    ) -> Launch:
        count = _elements(x)
        kernel = _compile(
            copy_kernel,
            {"x_ptr": "*bf16", "count": "i32", "y_ptr": "*bf16"},
            {"block": COPY_BLOCK},
            sm,
        )
        return _launch(kernel, ((count + COPY_BLOCK - 1) // COPY_BLOCK, 1, 1), [count])


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


def lower(model: onnx.ModelProto, key: str) -> Lowered:
    """The model with each of the guide's plugin nodes replaced by stand-in operators
    of the same output shapes and types, and the plugins to attach in their place."""
    g = Geometry.of(key)
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
            zeros(stats, g.stats_size, TensorProto.FLOAT)
            zeros(matrices, g.matrices_size, TensorProto.BFLOAT16)
            starts = constant(f"{name}/starts", np.array([0], np.int64))
            ends = constant(f"{name}/ends", np.array([g.channels], np.int64))
            axes = constant(f"{name}/axes", np.array([3], np.int64))
            nodes.append(helper.make_node("Slice", [qkv, starts, ends, axes], outputs))
            plugins += [
                Plugin(f"cell_{key}", [qkv, weight], [stats]),
                Plugin(f"region_{key}", [stats, tau], [matrices]),
                Plugin(f"apply_{key}", [qkv, weight, matrices], outputs),
            ]
        elif op == f"DualProject_{key}_TRT":
            identity(inputs[2], outputs[0])
            plugins.append(Plugin(f"project_{key}", inputs, outputs))
        elif op == "DualAttentionBarrier_TRT":
            identity(inputs[0], outputs[0])
            plugins.append(Plugin("barrier", inputs, outputs))
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
    return Lowered(lowered, plugins, readers)


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


__all__ = ["NAMESPACE", "Geometry", "Lowered", "Plugin", "attach", "lower", "register"]
