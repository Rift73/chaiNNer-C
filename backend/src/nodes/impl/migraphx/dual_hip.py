"""DUAL's MIGraphX custom ops (the dual_migx_ops library, native/src/dual_hip): the
TensorRT route's Triton kernels, compiled for the building GPU and carried by the ops'
nodes, so a program holds its kernels as a TensorRT engine holds its AOT plugins
(tensorrt/dual_aot.py). They are the dynamic-shape kernels (tensorrt/
dual_dynamic_kernels.py, tensorrt/dual_na_kernel.py: the trunk's size is a run-time
argument), so one kernel of each per width serves fixed and dynamic programs, compiled
with dual_aot.py's options. A chainner op has one output: Norm's sigma, which no DUAL
graph reads, goes to the op's scratch memory, and the attention barrier is a device copy
(no kernel).

An AMD target compiles to a code object (HSACO) for its gfx; a CUDA target to PTX, which
the library loads through HIP on NVIDIA (how the ops are checked against the TensorRT
plugins on an NVIDIA GPU). Used only by dual_lower_worker.py and the tests (needs Triton
and ONNX)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache
from typing import Any

import onnx
import triton
from onnx import helper
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from ..tensorrt import dual_dynamic_kernels, dual_na_kernel

DOMAIN = "chainner"
OP_PREFIX = "chainner_dual_"
# tensorrt/dual_aot.py's compile options and tile sizes
OPTIONS = {"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False}
NORM_PIXELS = 16
APPLY_BLOCK = 256
PROJECT_ROWS = 64
# Shared memory (LDS) a kernel may take: AMD's 64 KiB, or on NVIDIA what a launch gets
# without opting in to more
MAX_SHARED = {"hip": 65536, "cuda": 49152}
# The dynamic exporter's retrieval block (vendor/dual_tensorrt/scripts/dual_tensorrt/
# dynamic.py, WINDOW)
RETRIEVAL_BLOCK = 96


@dataclass(frozen=True)
class Target:
    """What the kernels are compiled for: an AMD gfx (backend "hip") or, for the check
    on NVIDIA, a CUDA compute capability (backend "cuda", arch e.g. 120)."""

    backend: str
    arch: str | int

    @staticmethod
    def amd(gfx: str) -> Target:
        """An AMD GPU by its gfx name (e.g. "gfx1100"; target features after a colon
        are dropped, as Triton's own HIP target does)."""
        return Target("hip", gfx.partition(":")[0])

    @property
    def warp_size(self) -> int:
        # GCN and CDNA (gfx9) run 64-wide waves; RDNA (gfx10 and later) 32-wide.
        if self.backend == "hip" and str(self.arch).startswith("gfx9"):
            return 64
        return 32

    def gpu_target(self) -> GPUTarget:
        return GPUTarget(self.backend, self.arch, self.warp_size)


def compile_kernel(
    fn: Any,
    pointers: dict[str, str],
    constants: dict[str, Any],
    target: Target,
    options: dict[str, Any] = OPTIONS,
    unaligned: tuple[str, ...] = (),
) -> dict[str, Any]:
    """A kernel's node attributes: its code object (`code`, PTX NUL-terminated for the
    driver) and the code's SHA-256 (`code_id`, which the library keys its loaded code
    by), function name (`kernel`), threads per block (`block`) and shared memory bytes
    (`shared`). As dual_aot.compile_kernel, pointers are declared 16-byte aligned (the
    library checks the buffers it is given), except those `unaligned` names, and
    Triton's scratch buffers refused."""
    signature = {**pointers, **dict.fromkeys(constants, "constexpr")}
    aligned = {
        (fn.arg_names.index(name),): [["tt.divisibility", 16]]
        for name, kind in pointers.items()
        if kind.startswith("*") and name not in unaligned
    }
    kernel = triton.compile(
        ASTSource(fn=fn, signature=signature, constexprs=constants, attrs=aligned),
        target=target.gpu_target(),
        options=options,
    )
    metadata = kernel.metadata
    if metadata.global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(f"{fn.__name__} needs Triton scratch memory")
    if metadata.shared > MAX_SHARED[target.backend]:
        raise RuntimeError(
            f"{fn.__name__} needs {metadata.shared} bytes of shared memory; at most"
            f" {MAX_SHARED[target.backend]} are available"
        )
    if target.backend == "hip":
        code = kernel.asm["hsaco"]
    else:
        code = kernel.asm["ptx"].encode() + b"\0"
    return {
        "code": code,
        "code_id": hashlib.sha256(code).hexdigest(),
        "kernel": metadata.name,
        "block": metadata.num_warps * target.warp_size,
        "shared": metadata.shared,
    }


@cache
def kernel(op: str, channels: int, target: Target, parameter: float = 0.0) -> Any:
    """One op's kernel attributes for a width (compiled once per process); `parameter`
    is Norm's epsilon or neighborhood attention's window."""
    if op == "norm":
        return compile_kernel(
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
                "eps": parameter,
            },
            target,
            # sigma, which nothing reads, goes to the op's scratch memory, whose
            # alignment MIGraphX does not promise
            unaligned=("sigma_ptr",),
        )
    if op == "cell":
        return compile_kernel(
            dual_dynamic_kernels.cell_stats,
            {
                "x_ptr": "*bf16",
                "weight_ptr": "*bf16",
                "height": "i32",
                "width": "i32",
                "stats_ptr": "*fp32",
            },
            {"channels": channels, "block": 256},
            target,
        )
    trunk = {"height": "i32", "width": "i32", "n0": "i32", "n1": "i32", "n2": "i32"}
    if op == "region":
        return compile_kernel(
            dual_dynamic_kernels.region_stats,
            {
                "stats_ptr": "*fp32",
                "tau_ptr": "*fp32",
                "qkv_ptr": "*bf16",
                **trunk,
                "matrices_ptr": "*bf16",
            },
            {"channels": channels},
            target,
        )
    if op == "apply":
        return compile_kernel(
            dual_dynamic_kernels.apply_dw,
            {
                "x_ptr": "*bf16",
                "weight_ptr": "*bf16",
                "matrices_ptr": "*bf16",
                **trunk,
                "out_ptr": "*bf16",
            },
            {"channels": channels, "block": APPLY_BLOCK},
            target,
        )
    if op == "project":
        return compile_kernel(
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
                "block_m": PROJECT_ROWS,
            },
            target,
        )
    if op == "na":
        return compile_kernel(
            dual_na_kernel.na_attention,
            {
                "qkv_ptr": "*bf16",
                "bq_ptr": "*bf16",
                "bk_ptr": "*bf16",
                "height": "i32",
                "width": "i32",
                "out_ptr": "*bf16",
            },
            {"channels": channels, "window": int(parameter), **dual_na_kernel.LAUNCH},
            target,
            dual_na_kernel.OPTIONS,
        )
    raise ValueError(f"DUAL has no {op} kernel")


def custom_node(
    op: str, inputs: list[str], output: str, name: str, **attributes: Any
) -> onnx.NodeProto:
    return helper.make_node(
        OP_PREFIX + op, inputs, [output], name=name, domain=DOMAIN, **attributes
    )


def lower(model: onnx.ModelProto, key: str, target: Target) -> onnx.ModelProto:
    """The exporter's DUAL ONNX (the TensorRT plugin key `key`, c<C>_h<H>_w<W> or the
    dynamic c<C>) with each plugin node replaced by `chainner`-domain nodes of the
    library's ops, each carrying its kernel compiled for `target`:
    Norm (x, gamma, beta) -> norm; Core (qkv, weight, tau) -> cell (qkv, weight),
    region (stats, tau, qkv) and apply (qkv, weight, matrices), statistics and
    attention matrices passing between them as tensors; Project (x, weight, residual)
    -> project; NA (qkv, bq, bk; window) -> na; the attention barrier -> barrier.
    Tile sizes the library's grids depend on travel as attributes too (rows, parts)."""
    channels = int(key.partition("_")[0][1:])
    read = {name for node in model.graph.node for name in node.input}
    read |= {output.name for output in model.graph.output}
    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        op, name = node.op_type, node.name
        inputs, outputs = list(node.input), list(node.output)
        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if op == f"DualNorm_{key}_TRT":
            if attributes.get("residual", 0) != 0:
                raise ValueError("DUAL's residual Norm variant is not supported")
            if outputs[1] in read:
                raise ValueError(
                    f"{name}'s sigma output is read; DUAL's MIGraphX Norm has none"
                )
            epsilon = float(attributes["epsilon"])
            nodes.append(
                custom_node(
                    "norm",
                    inputs,
                    outputs[0],
                    name,
                    rows=NORM_PIXELS,
                    **kernel("norm", channels, target, epsilon),
                )
            )
        elif op == f"DualCore_{key}_TRT":
            qkv, weight, tau = inputs
            stats, matrices = f"{name}/stats", f"{name}/matrices"
            nodes += [
                custom_node(
                    "cell",
                    [qkv, weight],
                    stats,
                    f"{name}/cell",
                    **kernel("cell", channels, target),
                ),
                custom_node(
                    "region",
                    [stats, tau, qkv],
                    matrices,
                    f"{name}/region",
                    **kernel("region", channels, target),
                ),
                custom_node(
                    "apply",
                    [qkv, weight, matrices],
                    outputs[0],
                    name,
                    parts=256 // APPLY_BLOCK,
                    **kernel("apply", channels, target),
                ),
            ]
        elif op == f"DualProject_{key}_TRT":
            nodes.append(
                custom_node(
                    "project",
                    inputs,
                    outputs[0],
                    name,
                    rows=PROJECT_ROWS,
                    **kernel("project", channels, target),
                )
            )
        elif op == f"DualNA_{key}_TRT":
            window = int(attributes["window"])
            nodes.append(
                custom_node(
                    "na",
                    inputs,
                    outputs[0],
                    name,
                    window=window,
                    parts=256 // dual_na_kernel.LAUNCH["block_m"],
                    **kernel("na", channels, target, window),
                )
            )
        elif op == "DualAttentionBarrier_TRT":
            nodes.append(custom_node("barrier", inputs, outputs[0], name))
        elif op.startswith("Dual") and op.endswith("_TRT"):
            raise ValueError(f"Unexpected DUAL plugin node {op}")
        else:
            nodes.append(node)
    lowered = onnx.ModelProto()
    lowered.CopyFrom(model)
    del lowered.graph.node[:]
    lowered.graph.node.extend(nodes)
    opsets = [o for o in lowered.opset_import if o.domain not in ("trt", DOMAIN)]
    del lowered.opset_import[:]
    lowered.opset_import.extend([*opsets, helper.make_opsetid(DOMAIN, 1)])
    return lowered


def _concat_of(
    producers: dict[str, onnx.NodeProto], name: str, axis: int
) -> onnx.NodeProto | None:
    node = producers.get(name)
    if node is None or node.op_type != "Concat" or len(node.input) != 2:
        return None
    attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    return node if attributes.get("axis") == axis else None


def fix_size(model: onnx.ModelProto, height: int, width: int, unshuffle: bool) -> None:
    """A dynamic DUAL ONNX made one of `height` x `width` (a MIGraphX program has one
    size): its input's size fixed, and cross-scale retrieval's window classes that are
    empty at that size taken out. Per trunk axis the dynamic exporter retrieves in
    96-px blocks and a leftover strip, so four classes, concatenated back as two rows of
    two (retrieve); a trunk axis shorter than a block has no blocks, one that is a
    multiple of it no strip, and such a class is a tensor with no elements, which
    TensorRT takes and MIGraphX refuses. Removing them from the concatenations, and
    what computed only them, leaves every other value as it was."""
    graph = model.graph
    dims = graph.input[0].type.tensor_type.shape.dim
    for dim, value in zip(dims[2:], (height, width), strict=True):
        dim.Clear()
        dim.dim_value = value
    divisor = 2 if unshuffle else 1
    counts = [
        (length // RETRIEVAL_BLOCK, min(length % RETRIEVAL_BLOCK, 1))
        for length in (height // divisor, width // divisor)
    ]
    producers = {output: node for node in graph.node for output in node.output}
    retrievals = 0
    for node in graph.node:
        if (
            not node.name.startswith(("/xr/", "/xg/"))
            or _concat_of(producers, node.output[0], 1) is None
        ):
            continue
        rows = [
            row
            for name in node.input
            if (row := _concat_of(producers, name, 2)) is not None
        ]
        if len(rows) != 2:
            continue
        retrievals += 1
        kept_rows = []
        for y, row in enumerate(rows):
            kept = [c for x, c in enumerate(row.input) if counts[0][y] and counts[1][x]]
            if kept:
                del row.input[:]
                row.input.extend(kept)
                kept_rows.append(row.output[0])
        del node.input[:]
        node.input.extend(kept_rows)
    if retrievals == 0:
        raise ValueError("This dynamic DUAL ONNX has no cross-scale retrieval to size")
    # A concatenation of one value is that value.
    for node in graph.node:
        if node.op_type == "Concat" and len(node.input) == 1:
            node.op_type = "Identity"
            del node.attribute[:]
    _remove_unread(model)


def _remove_unread(model: onnx.ModelProto) -> None:
    """Remove the nodes and initializers nothing reads (the graph's outputs are read)."""
    graph = model.graph
    while True:
        read = {name for node in graph.node for name in node.input}
        read |= {output.name for output in graph.output}
        unread = [n for n in graph.node if not any(o in read for o in n.output)]
        if not unread:
            break
        for node in unread:
            graph.node.remove(node)
    unused = [i for i in graph.initializer if i.name not in read]
    for initializer in unused:
        graph.initializer.remove(initializer)


def name_input(model: onnx.ModelProto, name: str) -> None:
    """Rename the model's (first) input, and every read of it, to `name`."""
    graph = model.graph
    source = graph.input[0].name
    graph.input[0].name = name
    for node in graph.node:
        for i, value in enumerate(node.input):
            if value == source:
                node.input[i] = name


__all__ = [
    "DOMAIN",
    "OP_PREFIX",
    "Target",
    "compile_kernel",
    "fix_size",
    "kernel",
    "lower",
    "name_input",
]
