"""VOSR's MIGraphX programs: which stages, at which fixed shapes, compiled from which ONNX
graph, and the one change the graphs need, the TensorRT plugin's node turned into the
custom op chainner_vosr2_qkv (native/src/vosr2_hip). The program sets' identity hashes this
file (vosr2_migx.py)."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from ..pytorch.vosr2_trt_models import profiles

if TYPE_CHECKING:
    import numpy as np
    import onnx

# MIGraphX compiles fixed shapes: each program is a stage at one of its profile's shapes
# (index 0, the smallest, or 2, the largest). The DiT's cached stage takes windows of 64
# latent columns, and of 72 with the context strip: a program for each, each holding its
# own copy of the weights.
PROGRAMS = {
    "dino": ("dino", 0),
    "dit_kv": ("dit_kv", 0),
    "dit_cached_64": ("dit_cached", 0),
    "dit_cached_72": ("dit_cached", 2),
}
# The TensorRT plugin node that the export's rewrite puts in the DiT (vosr2_trt_rewrite)
PLUGIN = ("trt.plugins", "VosrQkvPack")
OP = "chainner_vosr2_qkv"


def program_inputs(name: str) -> dict[str, tuple[int, ...]]:
    """A program's input shapes, by input name in the stage's order."""
    stage, index = PROGRAMS[name]
    return {key: shapes[index] for key, shapes in profiles(stage).items()}


def program_name(stage: str, latent_width: int) -> str:
    """The program that runs `stage`; for the DiT's cached stage, the one of the window's
    width."""
    if stage != "dit_cached":
        return stage
    name = f"dit_cached_{latent_width}"
    if name not in PROGRAMS:
        raise ValueError("VOSR MIGraphX supports 64 or 72 latent columns.")
    return name


def ops_path() -> Path:
    path = Path(__file__).with_name(
        "vosr2_migx_ops.dll" if sys.platform == "win32" else "vosr2_migx_ops.so"
    )
    if not path.is_file():
        raise RuntimeError(
            f"VOSR's MIGraphX op library {path} is missing: this build was made "
            "without it. Use the PyTorch backend, or build it with the ROCm 7.14 SDK"
            " (native/build_linux.sh -D CHAINNER_C_ROCM_ROOT=<SDK>, on Windows"
            " native/Build.ps1 -RocmRoot <SDK>; native/README.md, Build)."
        )
    return path


@lru_cache(maxsize=1)
def load_ops() -> ModuleType:
    """MIGraphX's binding with chainner_vosr2_qkv registered, as compiling or loading a
    DiT program needs."""
    from .runner import load_migraphx

    module = load_migraphx()
    module.load_op_library(str(ops_path()))
    return module


def plugin_ops(model: onnx.ModelProto) -> bool:
    """Turn each VosrQkvPack plugin node of `model` into the custom op, whose one output
    holds Q, K and V one after another, split and squeezed back into the plugin's three
    outputs. Whether the graph had the plugin."""
    from onnx import TensorProto, helper

    nodes = []
    for node in model.graph.node:
        if (node.domain, node.op_type) != PLUGIN:
            nodes.append(node)
            continue
        if len(node.input) != 5 or len(node.output) != 3:
            raise ValueError(f"Unexpected VOSR plugin node {node.name}.")
        packed = f"{node.name}/packed"
        parts = [f"{node.name}/part{i}" for i in range(3)]
        nodes.append(
            helper.make_node(
                OP,
                list(node.input),
                [packed],
                name=f"{node.name}/Op",
                domain="chainner",
            )
        )
        nodes.append(
            helper.make_node(
                "Split",
                [packed, "vosr_qkv_split"],
                parts,
                name=f"{node.name}/Split",
                axis=0,
            )
        )
        nodes.extend(
            helper.make_node(
                "Squeeze",
                [part, "vosr_qkv_axes"],
                [output],
                name=f"{node.name}/Squeeze{i}",
            )
            for i, (part, output) in enumerate(zip(parts, node.output, strict=True))
        )
    if len(nodes) == len(model.graph.node):
        return False
    model.graph.initializer.extend(
        [
            helper.make_tensor("vosr_qkv_split", TensorProto.INT64, [3], [1, 1, 1]),
            helper.make_tensor("vosr_qkv_axes", TensorProto.INT64, [1], [0]),
        ]
    )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    opsets = [opset for opset in model.opset_import if opset.domain != PLUGIN[0]]
    del model.opset_import[:]
    model.opset_import.extend([*opsets, helper.make_opsetid("chainner", 1)])
    return True


# Ops whose output has their first input's element type, which fp8_products follows back
# from a product without a weight (attention) to a Cast, an initializer or a graph input
_SAME_TYPE = {
    "Add",
    "Concat",
    "Div",
    "Gather",
    "Identity",
    "MatMul",
    "Mul",
    "Reshape",
    "Slice",
    "Softmax",
    "Split",
    "Squeeze",
    "Sub",
    "Transpose",
    "Unsqueeze",
}


def fp8_products(model: onnx.ModelProto, stage: str) -> int:
    """Give the matrix products TensorRT's FP8 recipe quantizes (the hosted set's Q/DQ
    pairs) FP32 inputs, which MIGraphX's quantize_fp8 turns into FP8, and keep the rest as
    exported: it never quantizes BF16 products. The DiT: every product with a weight but
    its output projection, the cached stage's last one (attention stays BF16); DINO:
    every product, and its FP32 patch embedding becomes BF16 to stay out of FP8. The
    number of FP8 products."""
    from onnx import TensorProto, helper

    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    producers = {name: node for node in graph.node for name in node.output}
    inputs = {value.name: value.type.tensor_type.elem_type for value in graph.input}

    def element_type(name: str) -> int:
        while name not in initializers and name not in inputs:
            node = producers[name]
            if node.op_type == "Cast":
                return next(a.i for a in node.attribute if a.name == "to")
            if node.op_type not in _SAME_TYPE:
                raise ValueError(f"No VOSR type rule for {node.op_type} ({name}).")
            name = node.input[0]
        if name in inputs:
            return inputs[name]
        return initializers[name].data_type

    products = [node for node in graph.node if node.op_type == "MatMul"]
    if stage != "dino":
        products = [
            node
            for node in products
            if any(name in initializers for name in node.input)
        ]
        if stage == "dit_cached":
            products.pop()
    # Each product's type before any changes: its inputs share it, a weight's says it
    kinds = {
        node.name: initializers[node.input[1]].data_type
        if node.input[1] in initializers
        else element_type(node.input[0])
        for node in products
    }

    nodes: list[onnx.NodeProto] = []
    casts: dict[tuple[str, int], str] = {}

    def converted(name: str, to: int) -> str:
        if (name, to) not in casts:
            casts[name, to] = f"{name}/cast{to}"
            nodes.append(
                helper.make_node(
                    "Cast", [name], [casts[name, to]], to=to, name=casts[name, to]
                )
            )
        return casts[name, to]

    def retyped(node: onnx.NodeProto, to: int, back: int) -> None:
        for index, name in enumerate(node.input):
            if name:
                node.input[index] = converted(name, to)
        result = node.output[0]
        node.output[0] = f"{result}/computed{to}"
        nodes.append(node)
        nodes.append(
            helper.make_node("Cast", [node.output[0]], [result], to=back, name=result)
        )

    for node in list(graph.node):
        weight = initializers.get(node.input[1]) if len(node.input) > 1 else None
        kind = kinds.get(node.name)
        if node.op_type == "Conv" and weight is not None and weight.data_type == 1:
            retyped(node, TensorProto.BFLOAT16, TensorProto.FLOAT)
        elif kind == TensorProto.BFLOAT16:
            retyped(node, TensorProto.FLOAT, TensorProto.BFLOAT16)
        elif kind in (None, TensorProto.FLOAT):
            nodes.append(node)
        else:
            raise ValueError(f"Unexpected VOSR product type in {node.name}.")
    del graph.node[:]
    graph.node.extend(nodes)
    return len(products)


def fp16_types(model: onnx.ModelProto) -> None:
    """Make every BF16 tensor of `model` (weights loaded) FP16: weights, casts, constants,
    inputs and outputs; FP32 stays FP32. The custom op keeps BF16, between casts."""
    import numpy as np
    from onnx import AttributeProto, TensorProto, helper, numpy_helper

    bf16, fp16 = TensorProto.BFLOAT16, TensorProto.FLOAT16
    graph = model.graph

    def half(tensor: onnx.TensorProto) -> onnx.TensorProto:
        values = numpy_helper.to_array(tensor).astype(np.float32).astype(np.float16)
        return numpy_helper.from_array(values, tensor.name)

    for tensor in graph.initializer:
        if tensor.data_type == bf16:
            tensor.CopyFrom(half(tensor))
    for value in (*graph.input, *graph.output, *graph.value_info):
        if value.type.tensor_type.elem_type == bf16:
            value.type.tensor_type.elem_type = fp16
    nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        for attribute in node.attribute:
            if (
                node.op_type == "Cast"
                and attribute.name == "to"
                and attribute.i == bf16
            ):
                attribute.i = fp16
            elif (
                attribute.type == AttributeProto.TENSOR
                and attribute.t.data_type == bf16
            ):
                attribute.t.CopyFrom(half(attribute.t))
        if node.domain != "chainner":
            nodes.append(node)
            continue
        for index, name in enumerate(node.input):
            cast = f"{node.name}/input{index}"
            nodes.append(helper.make_node("Cast", [name], [cast], to=bf16, name=cast))
            node.input[index] = cast
        result = node.output[0]
        node.output[0] = f"{result}/bf16"
        nodes.append(node)
        nodes.append(
            helper.make_node("Cast", [node.output[0]], [result], to=fp16, name=result)
        )
    del graph.node[:]
    graph.node.extend(nodes)


def migraphx_graph(path: Path, precision: str = "bf16") -> Path:
    """The ONNX graph at `path` as MIGraphX takes it, written beside it: the plugin as the
    custom op (plugin_ops) and, for `precision` "fp8", FP32 inputs on the products to
    quantize (fp8_products), for "fp16" FP16 for every BF16 tensor (fp16_types, with its
    own external data). A graph that needs nothing is returned as it is; one made before
    in the same folder (a build's own) is reused."""
    import onnx

    target = path.with_name(f"{path.stem}.{precision}.migraphx.onnx")
    if target.is_file():
        return target
    # FP16 rewrites the weights; the others keep the external data where it is
    model = onnx.load(path, load_external_data=precision == "fp16")
    changed = plugin_ops(model)
    if precision == "fp8":
        changed = fp8_products(model, path.stem) > 0 or changed
    elif precision == "fp16":
        fp16_types(model)
        onnx.save(
            model,
            target,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{target.name}.data",
        )
        return target
    if not changed:
        return path
    onnx.save(model, target)
    return target


def compile_program(
    name: str,
    graphs: Path,
    plan: Path,
    gpu: int,
    precision: str = "bf16",
    calibration: Sequence[dict[str, np.ndarray]] = (),
) -> None:
    """Compile program `name` from its stage's graph in `graphs` (this code's BF16
    export) for GPU `gpu` (HIP's index) and save it at `plan`, in `precision`: "bf16",
    the graph's own types, as TensorRT's strongly typed builds have them; "fp16", FP16
    instead of BF16 (FP32 stays); "fp8", the products of TensorRT's FP8 recipe quantized
    by MIGraphX to the GPU's FP8 format, calibrated on `calibration` (typical inputs)."""
    stage, _ = PROGRAMS[name]
    migraphx = load_ops()
    program = migraphx.compile_onnx(
        str(migraphx_graph(graphs / f"{stage}.onnx", precision)),
        device=gpu,
        input_dims=program_inputs(name),
        fp8=precision == "fp8",
        calibration=tuple(calibration),
    )
    try:
        program.save(str(plan))
    finally:
        program.close()
