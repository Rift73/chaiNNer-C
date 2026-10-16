"""Validated ONNX boundaries for the one-step VOSR checkpoint.

Raw cross-attention K/V are cut before normalization, preserving the established
BF16 rounding order. Self-attention packing uses a native CUDA plugin; DINO keeps
FP32 projection weights and residuals around native BF16 fused attention.
"""

from __future__ import annotations

import copy
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def split_dit(model: onnx.ModelProto) -> tuple[onnx.ModelProto, onnx.ModelProto]:
    producer = {v: n for n in model.graph.node for v in n.output}
    cuts = []
    for n in model.graph.node:
        if n.op_type != "MatMul" or not n.name.startswith("/cross_attn"):
            continue
        soft = producer.get(n.input[0])
        if soft is None or soft.op_type != "Softmax":
            continue
        qk = producer[soft.input[0]]
        scale = producer[qk.input[1]]
        transpose = producer[scale.input[0]]
        key = transpose.input[0]
        value = n.input[1]
        if "k_norm" not in key or producer[value].op_type != "Transpose":
            raise ValueError("Unexpected VOSR cross-attention normalization boundary.")
        block = len(cuts) // 2
        key = next(
            v
            for v, p in producer.items()
            if p.op_type == "Add"
            and f"dit.blocks.{block}.cross_attn.k_linear.bias" in p.input
        )
        value = next(
            v
            for v, p in producer.items()
            if p.op_type == "Add"
            and f"dit.blocks.{block}.cross_attn.v_linear.bias" in p.input
        )
        cuts.extend((key, value))
    if len(cuts) != 72 or len(set(cuts)) != 72:
        raise ValueError("Expected 36 VOSR raw cross-attention K/V pairs.")
    cut_shape = [1, 1024, 1536]
    packed_shape = [72, *cut_shape]
    cut_info = {
        name: helper.make_tensor_value_info(name, TensorProto.BFLOAT16, cut_shape)
        for name in cuts
    }
    original_inputs = {v.name: v for v in model.graph.input}
    original_outputs = {v.name: v for v in model.graph.output}

    def extract(inputs: list[str], outputs: list[str]) -> onnx.ModelProto:
        needed = set(outputs)
        stack = list(outputs)
        selected = set()
        boundaries = set(inputs)
        initializers = {t.name: t for t in model.graph.initializer}
        while stack:
            name = stack.pop()
            if name in boundaries or name in initializers or not name:
                continue
            node = producer.get(name)
            if node is None:
                raise ValueError(f"Unresolved VOSR graph dependency: {name}")
            if node.name in selected:
                continue
            selected.add(node.name)
            for value in node.input:
                if value not in needed:
                    needed.add(value)
                    stack.append(value)
        result = copy.deepcopy(model)
        del result.graph.node[:]
        result.graph.node.extend(n for n in model.graph.node if n.name in selected)
        del result.graph.input[:]
        result.graph.input.extend(
            original_inputs[n] if n in original_inputs else cut_info[n] for n in inputs
        )
        del result.graph.output[:]
        result.graph.output.extend(
            original_outputs[n] if n in original_outputs else cut_info[n]
            for n in outputs
        )
        del result.graph.initializer[:]
        result.graph.initializer.extend(
            t for t in model.graph.initializer if t.name in needed
        )
        return result

    projection = extract(["conditioning"], cuts)
    for i, name in enumerate(cuts):
        axes = helper.make_tensor(f"cache_axes_{i}", TensorProto.INT64, [1], [0])
        projection.graph.initializer.append(axes)
        projection.graph.node.append(
            helper.make_node(
                "Unsqueeze",
                [name, axes.name],
                [f"cache_piece_{i}"],
                name=f"cache_unsqueeze_{i}",
            )
        )
    projection.graph.node.append(
        helper.make_node(
            "Concat",
            [f"cache_piece_{i}" for i in range(72)],
            ["result"],
            axis=0,
            name="cache_pack",
        )
    )
    del projection.graph.output[:]
    projection.graph.output.append(
        helper.make_tensor_value_info("result", TensorProto.BFLOAT16, packed_shape)
    )

    body = extract(["latent", *cuts, "cos", "sin"], ["result"])
    del body.graph.input[:]
    body.graph.input.extend(
        [
            original_inputs["latent"],
            helper.make_tensor_value_info("kv", TensorProto.BFLOAT16, packed_shape),
            original_inputs["cos"],
            original_inputs["sin"],
        ]
    )
    gathers = []
    for i, name in enumerate(cuts):
        index = helper.make_tensor(f"cache_index_{i}", TensorProto.INT64, [], [i])
        body.graph.initializer.append(index)
        gathers.append(
            helper.make_node(
                "Gather", ["kv", index.name], [name], axis=0, name=f"cache_unpack_{i}"
            )
        )
    nodes = list(body.graph.node)
    del body.graph.node[:]
    body.graph.node.extend(gathers + nodes)

    return projection, body


def pack_dit(model: onnx.ModelProto) -> None:
    nodes = list(model.graph.node)
    byname = {n.name: n for n in nodes}
    additions = []
    # Override named boundary tensors, then retain only their live dependencies.
    for block in range(36):
        suffix = "" if block == 0 else "_" + str(block)
        prefix = "/attn" + suffix
        inp = byname["/attn/qkv" + suffix + "/Add"].output[0]
        outputs = [
            byname[prefix + "/Add"].output[0],
            byname[prefix + "/Add_1"].output[0],
            byname[prefix + "/Squeeze_2"].output[0],
        ]
        additions.append(
            helper.make_node(
                "VosrQkvPack",
                [
                    inp,
                    f"dit.blocks.{block}.attn.q_norm.weight",
                    f"dit.blocks.{block}.attn.k_norm.weight",
                    "cos",
                    "sin",
                ],
                outputs,
                name=prefix + "/NativePack",
                domain="trt.plugins",
                mode=0,
                plugin_version="1",
                plugin_namespace="",
            )
        )
    producer = {v: n for n in nodes + additions for v in n.output}
    ordered = []
    done = set()
    pending = [(o.name, False) for o in model.graph.output]
    while pending:
        value, expanded = pending.pop()
        n = producer.get(value)
        if n is None or n.name in done:
            continue
        if expanded:
            done.add(n.name)
            ordered.append(n)
        else:
            pending.append((value, True))
            pending.extend((v, False) for v in reversed(n.input))
    del model.graph.node[:]
    model.graph.node.extend(ordered)
    model.opset_import.append(helper.make_opsetid("trt.plugins", 1))
    if len(additions) != 36:
        raise ValueError("Expected 36 VOSR DiT attention blocks.")


def dino_attention(model: onnx.ModelProto) -> None:
    nodes = list(model.graph.node)
    producer = {v: n for n in nodes for v in n.output}
    users = {}
    for n in nodes:
        for v in n.input:
            users.setdefault(v, []).append(n)
    replacement = {}
    removed = set()
    count = 0
    for soft in nodes:
        if soft.op_type != "Softmax":
            continue
        first = producer[soft.input[0]]
        last = users[soft.output[0]][0]
        if first.op_type != "MatMul" or last.op_type != "MatMul":
            raise ValueError("Unexpected VOSR DINO attention boundary.")
        q, k = first.input
        v = last.input[1]
        prefix = f"NativeAttention{count}"
        new = []
        for key, value in zip(("q", "k", "v"), (q, k, v), strict=True):
            new.append(
                helper.make_node(
                    "Cast",
                    [value],
                    [prefix + key],
                    name=prefix + key,
                    to=TensorProto.BFLOAT16,
                )
            )
        new += [
            helper.make_node(
                "MatMul",
                [prefix + "q", prefix + "k"],
                [prefix + "score"],
                name=prefix + "score",
            ),
            helper.make_node(
                "Softmax",
                [prefix + "score"],
                [prefix + "prob"],
                name=prefix + "prob",
                axis=-1,
            ),
            helper.make_node(
                "MatMul",
                [prefix + "prob", prefix + "v"],
                [prefix + "value"],
                name=prefix + "value",
            ),
            helper.make_node(
                "Cast",
                [prefix + "value"],
                list(last.output),
                name=prefix + "result",
                to=TensorProto.FLOAT,
            ),
        ]
        replacement[last.name] = new
        removed.update((first.name, soft.name, last.name))
        count += 1
    if count != 18:
        raise ValueError("Expected 18 VOSR DINO attention blocks.")
    del model.graph.node[:]
    for n in nodes:
        if n.name in replacement:
            model.graph.node.extend(replacement[n.name])
        elif n.name not in removed:
            model.graph.node.append(n)


def rewrite(path: Path, name: str) -> None:
    model = onnx.load(path, load_external_data=False)
    if name == "dit_cached":
        pack_dit(model)
    elif name == "dino":
        dino_attention(model)
    else:
        raise ValueError(name)
    onnx.save(model, path)
