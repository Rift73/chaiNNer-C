"""An ONNX of an ESRGAN in FP16 from an earlier version joins each dense block's tensors
from scratch: cat(x, x1), cat(x, x1, x2) and so on. Convert To ONNX now exports
cat(cat(x, x1), x2) (nodes/impl/pytorch/esrgan_export.py): the same channels in the same
order, which TensorRT lays out with far fewer copies. Build Engine gives such a file the
same layout. FP16 only: nested, an FP32 engine runs slower."""

from __future__ import annotations

import onnx
from onnx import numpy_helper

# A dense block ends in x5 * 0.2 + x.
RESIDUAL_SCALE = 0.2


def nested_dense_blocks(onnx_bytes: bytes, precision: str) -> bytes | None:
    """The ONNX with the joins of each ESRGAN dense block nested; None when the build is
    not in FP16 or the ONNX is anything but an ESRGAN with plain joins, which then builds
    from `onnx_bytes` as it is: every join must be part of a whole dense block."""
    if precision != "fp16" or b"LeakyRelu" not in onnx_bytes:
        return None
    model = onnx.load_from_string(onnx_bytes)
    graph = model.graph
    makers = {output: node for node in graph.node for output in node.output}
    readers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            readers.setdefault(name, []).append(node)
    initializers = {tensor.name: tensor for tensor in graph.initializer}

    def made_by(name: str, op_type: str) -> onnx.NodeProto | None:
        maker = makers.get(name)
        return maker if maker is not None and maker.op_type == op_type else None

    def read_only_by(name: str, op_type: str) -> onnx.NodeProto | None:
        found = readers.get(name, [])
        return found[0] if len(found) == 1 and found[0].op_type == op_type else None

    def conv_reads(name: str) -> str | None:
        conv = made_by(name, "Conv")
        return None if conv is None else conv.input[0]

    def layer_reads(name: str) -> str | None:
        activation = made_by(name, "LeakyRelu")
        return None if activation is None else conv_reads(activation.input[0])

    def is_residual_scale(name: str) -> bool:
        maker = made_by(name, "Constant")
        tensor = initializers.get(name) if maker is None else maker.attribute[0].t
        if tensor is None or tensor.dims or not tensor.data_type:
            return False
        return abs(float(numpy_helper.to_array(tensor)) - RESIDUAL_SCALE) < 1e-3

    def ends_block(join: str, x: str) -> bool:
        """Whether only a convolution reads `join`, and its output, scaled, adds to `x`."""
        conv = read_only_by(join, "Conv")
        mul = None if conv is None else read_only_by(conv.output[0], "Mul")
        if mul is None:
            return False
        add = read_only_by(mul.output[0], "Add")
        return (
            add is not None
            and any(is_residual_scale(name) for name in mul.input)
            and sorted(add.input) == sorted((mul.output[0], x))
        )

    joins = [
        node
        for node in graph.node
        if node.op_type == "Concat"
        and next((a.i for a in node.attribute if a.name == "axis"), None) == 1
    ]
    by_inputs = {tuple(node.input): node for node in joins}
    nestings = []
    for inputs, last in by_inputs.items():
        second, third, fourth = (by_inputs.get(inputs[:size]) for size in (2, 3, 4))
        if len(inputs) != 5 or second is None or third is None or fourth is None:
            continue
        x, x1, x2, x3, x4 = inputs
        # ESRGAN+ (spandrel's `plus`) adds a 1x1 convolution of x to x2, and x2 to x4.
        layer2, layer4 = x2, x4
        sum2, sum4 = made_by(x2, "Add"), made_by(x4, "Add")
        if sum2 is not None and sum4 is not None:
            if conv_reads(sum2.input[1]) != x or sum4.input[1] != x2:
                continue
            layer2, layer4 = sum2.input[0], sum4.input[0]
        if (
            layer_reads(x1) == x
            and layer_reads(layer2) == second.output[0]
            and layer_reads(x3) == third.output[0]
            and layer_reads(layer4) == fourth.output[0]
            and ends_block(last.output[0], x)
        ):
            nestings += [(third, second, x2), (fourth, third, x3), (last, fourth, x4)]
    # Three of a block's four joins change. A network with any other join, such as a
    # TSCUNet with dense blocks inside, is not an ESRGAN.
    if not nestings or len(joins) != len(nestings) // 3 * 4:
        return None
    for join, previous, grown in nestings:
        del join.input[:]
        join.input.extend((previous.output[0], grown))
    return model.SerializeToString()
