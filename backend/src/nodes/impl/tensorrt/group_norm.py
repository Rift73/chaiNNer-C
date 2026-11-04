"""TensorRT 11 fuses GroupNorm's ONNX, a Reshape to [0, groups, -1] around an
InstanceNormalization, into a group normalization kernel. On a dynamic engine whose
largest group is small the kernel is silently wrong: a 16-channel RealPLKSR in 4 groups
comes out at 10 dB on sizes 32 to 64 and a 64-channel one at 6 dB on 16 to 32. Elsewhere it
is right and about twice as fast as the same arithmetic written out, so only these engines
are built from the written-out ONNX. FP32 only: written out in FP16 a trained model loses
too much (34 to 40 dB), and a mixed ONNX can hold its GroupNorm in FP16."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper

# Values per group at the engine's largest size. Measured wrong up to 20736 and right from
# 25088 on four channel and group counts at optimization levels 0 and 3. The line is not a
# documented TensorRT constant, and an engine this small loses little to the arithmetic,
# so the margin is wide.
WRONG_PLAN_LIMIT = 65536


def _written_out(
    nodes: list[onnx.NodeProto], norm: onnx.NodeProto, epsilon: np.ndarray, opset: int
) -> None:
    """Appends `norm`'s values over its axis 2 as plain arithmetic."""
    source, name = norm.input[0], norm.output[0]

    def mean(values: str, output: str) -> None:
        if opset >= 18:
            axes = f"{output}_axes"
            axis = numpy_helper.from_array(np.array([2], np.int64))
            nodes.append(helper.make_node("Constant", [], [axes], value=axis))
            nodes.append(helper.make_node("ReduceMean", [values, axes], [output], keepdims=1))
        else:
            nodes.append(helper.make_node("ReduceMean", [values], [output], axes=[2], keepdims=1))

    mean(source, f"{name}_mean")
    nodes.append(helper.make_node("Sub", [source, f"{name}_mean"], [f"{name}_deviation"]))
    nodes.append(
        helper.make_node(
            "Mul", [f"{name}_deviation", f"{name}_deviation"], [f"{name}_square"]
        )
    )
    mean(f"{name}_square", f"{name}_variance")
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            [f"{name}_epsilon"],
            value=numpy_helper.from_array(epsilon),
        )
    )
    nodes.append(
        helper.make_node("Add", [f"{name}_variance", f"{name}_epsilon"], [f"{name}_sum"])
    )
    nodes.append(helper.make_node("Sqrt", [f"{name}_sum"], [f"{name}_root"]))
    nodes.append(helper.make_node("Div", [f"{name}_deviation", f"{name}_root"], [name]))


def written_out_group_norm(
    onnx_bytes: bytes,
    precision: str,
    min_shape: tuple[int, int],
    max_shape: tuple[int, int],
    dynamic: bool,
) -> bytes | None:
    """The ONNX with each GroupNorm that takes the wrong plan on this engine written out;
    None when the engine has none, which then builds from `onnx_bytes` as it is."""
    area = max_shape[0] * max_shape[1]
    # Two channels per group is the fewest of a group of several channels.
    if (
        not dynamic
        or min_shape == max_shape
        or precision != "fp32"
        or 2 * area > WRONG_PLAN_LIMIT
        or b"InstanceNormalization" not in onnx_bytes
    ):
        return None
    model = onnx.load_from_string(onnx_bytes)
    graph = model.graph
    makers = {output: node for node in graph.node for output in node.output}
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    opset = next(
        (entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")),
        None,
    )
    if opset is None:
        return None
    inferred = onnx.shape_inference.infer_shapes(model).graph.value_info
    shapes = {value.name: value.type.tensor_type.shape.dim for value in inferred}

    def constant(name: str) -> np.ndarray | None:
        maker = makers.get(name)
        if maker is not None and maker.op_type == "Constant":
            if maker.attribute[0].name != "value":
                return None
            return numpy_helper.to_array(maker.attribute[0].t)
        return numpy_helper.to_array(initializers[name]) if name in initializers else None

    nodes: list[onnx.NodeProto] = []
    rewritten = False
    for node in graph.node:
        reshape = makers.get(node.input[0]) if node.op_type == "InstanceNormalization" else None
        target = None
        if reshape is not None and reshape.op_type == "Reshape":
            target = constant(reshape.input[1])
        if target is None or len(target) != 3 or target[0] != 0 or target[2] != -1:
            nodes.append(node)
            continue
        groups = int(target[1])
        dims = shapes.get(reshape.input[0])
        channels = dims[1].dim_value if dims is not None and len(dims) > 1 else 0
        scale, bias = constant(node.input[1]), constant(node.input[2])
        if (
            scale is None
            or bias is None
            or not (scale == 1).all()
            or not (bias == 0).all()
            or groups <= 1
            or 0 < channels <= groups
            or max(channels // groups, 2) * area > WRONG_PLAN_LIMIT
        ):
            nodes.append(node)
            continue
        epsilon = next((a.f for a in node.attribute if a.name == "epsilon"), 1e-5)
        _written_out(nodes, node, np.array(epsilon, scale.dtype), opset)
        rewritten = True
    if not rewritten:
        return None
    del graph.node[:]
    graph.node.extend(nodes)
    return model.SerializeToString()
