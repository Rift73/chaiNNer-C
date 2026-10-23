from __future__ import annotations

import heapq

import onnx
from onnx.onnx_pb import ModelProto, NodeProto

from nodes.impl.ncnn.model import NcnnModelWrapper
from nodes.impl.onnx.model import OnnxModel
from nodes.impl.onnx.onnx_to_ncnn import Onnx2NcnnConverter
from nodes.impl.onnx.utils import safely_optimize_onnx_model
from nodes.properties.inputs import OnnxFpDropdown, OnnxModelInput
from nodes.properties.outputs import NcnnModelOutput, TextOutput

from .. import utility_group


def put_in_order(model: ModelProto) -> None:
    """Order the graph's nodes so each comes after the nodes making its inputs, as the
    converter needs: its own reordering fails on a file that breaks ONNX's rule of
    ordered nodes, and never ends on some cycles. An ordered graph keeps its order;
    a cycle raises ValueError. Inputs no node makes stay for the converter to name."""
    graph = model.graph
    nodes = list(graph.node)
    given = {i.name for i in graph.input} | {w.name for w in graph.initializer}
    makers: dict[str, int] = {}
    for index, node in enumerate(nodes):
        for name in node.output:
            if name and name not in given:
                makers.setdefault(name, index)
    waits = [0] * len(nodes)
    users: list[list[int]] = [[] for _ in nodes]
    for index, node in enumerate(nodes):
        for name in set(node.input):
            if name in makers:
                waits[index] += 1
                users[makers[name]].append(index)

    ready = [index for index, count in enumerate(waits) if count == 0]
    order: list[int] = []
    while ready:
        index = heapq.heappop(ready)
        order.append(index)
        for user in users[index]:
            waits[user] -= 1
            if waits[user] == 0:
                heapq.heappush(ready, user)

    if len(order) < len(nodes):
        # Every node left waits on a node left, so following them comes back round.
        walk: dict[int, None] = {}
        index = next(i for i, count in enumerate(waits) if count)
        while index not in walk:
            walk[index] = None
            index = next(
                makers[name]
                for name in nodes[index].input
                if name in makers and waits[makers[name]]
            )
        cycle = list(walk)[list(walk).index(index) :][::-1]
        start = cycle.index(min(cycle))
        cycle = cycle[start:] + cycle[:start] + cycle[start : start + 1]
        names = " -> ".join(nodes[i].name or f"{nodes[i].op_type} #{i}" for i in cycle)
        raise ValueError(
            f"The ONNX model's graph has a cycle: {names}. The file is damaged."
        )

    if order != list(range(len(nodes))):
        ordered = [NodeProto() for _ in order]
        for copy, index in zip(ordered, order):
            copy.CopyFrom(nodes[index])
        del graph.node[:]
        graph.node.extend(ordered)


def fill_kernel_shapes(model: ModelProto) -> None:
    """Give each Conv and ConvTranspose without a kernel_shape the one its weights have:
    ONNX infers it from them when the attribute is absent, while the converter reads
    only the attribute and made a convolution with a kernel 0 wide. A weight that is no
    initializer is left to the converter, as before."""
    weights = {w.name: w for w in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type not in ("Conv", "ConvTranspose") or len(node.input) < 2:
            continue
        if any(a.name == "kernel_shape" for a in node.attribute):
            continue
        weight = weights.get(node.input[1])
        if weight is not None and len(weight.dims) > 2:
            node.attribute.append(
                onnx.helper.make_attribute("kernel_shape", list(weight.dims[2:]))
            )


@utility_group.register(
    schema_id="chainner:onnx:convert_to_ncnn",
    name="Convert To NCNN",
    description=[
        "Convert an ONNX model to NCNN.",
        "It is recommended to save converted models as a separate step, then load the converted models instead of converting them every time you run the chain.",
        "Note: Converted models are not guaranteed to work with other programs that support NCNN models. This is for a variety of reasons and cannot be changed.",
    ],
    icon="NCNN",
    inputs=[
        OnnxModelInput("ONNX Model"),
        OnnxFpDropdown(),
    ],
    outputs=[
        NcnnModelOutput(
            model_type="NcnnNetwork & onnxToNcnn(Input0)",
            label="NCNN Model",
        ),
        TextOutput("FP Mode", "FpMode::toString(Input1)"),
    ],
)
def convert_to_ncnn_node(
    model: OnnxModel, is_fp16: int
) -> tuple[NcnnModelWrapper, str]:
    fp16 = bool(is_fp16)

    model_proto = onnx.load_model_from_string(model.bytes)
    put_in_order(model_proto)  # first, as the optimizer optimizes only valid models
    fill_kernel_shapes(model_proto)
    opt_model = safely_optimize_onnx_model(model_proto)

    converter = Onnx2NcnnConverter(opt_model)
    converted = converter.convert(fp16, False)
    # Without its MemoryData layers (chaiNNer writes none) a model that needs a constant
    # tensor (one that no scalar fusion absorbs) names a blob no layer makes, and ncnn
    # refuses the files ("find_blob_index_by_name ... failed").
    written = [layer for layer in converted.layers if layer.op_type != "ncnnfused"]
    made = {name for layer in written for name in layer.outputs}
    missing = sorted({name for layer in written for name in layer.inputs} - made)
    if missing:
        raise ValueError(
            "This model needs constant tensors, which NCNN keeps in MemoryData layers"
            " that chaiNNer does not write, so it cannot be converted to NCNN"
            f" (constants: {', '.join(missing[:5])}{', ...' if len(missing) > 5 else ''})."
        )
    ncnn_model = NcnnModelWrapper(converted)

    fp_mode = "fp16" if fp16 else "fp32"

    return ncnn_model, fp_mode
