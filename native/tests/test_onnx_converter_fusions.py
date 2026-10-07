"""Fusion motifs and their failed guards against the independent installed code.

The historical HardSigmoid and ShuffleChannel/Split patterns cannot fuse real
TensorProto inputs because their guards contradict themselves. LayerNorm also
has a duplicate-mean edge and an axes[4] bug. Both complete attention patterns
fail late while constructing an integer attribute from a floating head count.
These are tested as observable original behavior, not repaired by the oracle.
"""

from __future__ import annotations

import ast
import copy
import json
from dataclasses import dataclass, field
from typing import Protocol

import coverage
import numpy as np
import pytest
from coverage.parser import PythonParser
from onnx import AttributeProto as A
from onnx import GraphProto, NodeProto, TensorProto, helper, numpy_helper
from test_onnx_converter import FROZEN, check, current, model, old, outcome, snapshot


@dataclass
class Motif:
    name: str
    method: str
    source: object
    expected_op: str | None = None
    expected_count: int = 0
    expected_error: str | None = None
    references: dict = field(default_factory=dict)


def n(op, inputs, output, **attributes):
    outputs = [output] if isinstance(output, str) else output
    return helper.make_node(
        op,
        inputs.split() if isinstance(inputs, str) else inputs,
        outputs,
        name=outputs[0],
        **attributes,
    )


def motif(name, method, nodes, weights=None, **expect):
    weights = weights or {}
    produced = {v for node in nodes for v in node.output}
    inputs = list(
        dict.fromkeys(
            v
            for node in nodes
            for v in node.input
            if v and v not in produced and v not in weights
        )
    )
    return Motif(name, "fuse_" + method, model(nodes, weights, inputs=inputs), **expect)


def scalar(value):
    return np.asarray(value, np.float32)


def reshape(source, output, shape, weights, tensor_shapes):
    if tensor_shapes:
        name = output + "_shape"
        weights[name] = np.asarray(shape, np.int64)
        return n("Reshape", [source, name], output)
    return n("Reshape", [source], output, shape=shape)


def attention(combined, tensor_shapes):
    """All 20/17 actual nodes, including every tested edge and head dimension."""
    weights = {
        "scale": scalar(0.5),
        "ow": np.eye(4, dtype=np.float32),
        "ob": np.zeros(4, np.float32),
    }
    if combined:
        weights.update(
            qkvw=np.arange(48, dtype=np.float32).reshape(4, 12),
            qkvb=np.arange(12, dtype=np.float32),
        )
        nodes = [
            n("MatMul", "x qkvw", "mm"),
            n("Add", "mm qkvb", "added"),
            n("Split", "added", ["q", "k", "v"], axis=2, split=[4, 4, 4]),
        ]
    else:
        nodes = []
        for key in ("q", "k", "v"):
            weights[key + "w"] = np.eye(4, dtype=np.float32)
            weights[key + "b"] = np.arange(4, dtype=np.float32)
            nodes.extend(
                [
                    n("MatMul", ["x" + key, key + "w"], key + "mm"),
                    n("Add", [key + "mm", key + "b"], key),
                ]
            )
    nodes.extend(
        [
            n("Mul", "q scale", "scaled"),
            reshape("scaled", "qr", [1, 6, 2], weights, tensor_shapes),
            n("Transpose", "qr", "qt", perm=[1, 0, 2]),
            reshape("k", "kr", [1, 6, 2], weights, tensor_shapes),
            reshape("v", "vr", [1, 6, 2], weights, tensor_shapes),
            n("Transpose", "vr", "vt", perm=[1, 0, 2]),
            n("Transpose", "kr", "kt", perm=[1, 2, 0]),
            n("MatMul", "qt kt", "scores"),
            n("Softmax", "scores", "soft", axis=2),
            n("MatMul", "soft vt", "attend"),
            n("Transpose", "attend", "trans", perm=[1, 0, 2]),
            reshape("trans", "outshape", [1, 3, 4], weights, tensor_shapes),
            n("MatMul", "outshape ow", "outmm"),
            n("Add", "outmm ob", "y"),
        ]
    )
    return motif(
        f"attention-{'qkv' if combined else 'separate'}-{'tensor' if tensor_shapes else 'attr'}",
        "multiheadattention",
        nodes,
        weights,
        expected_op="MultiHeadAttention",
        expected_error="TypeError",
    )


def motifs():
    cases = []
    for indices in ([2], [2, 3]):
        cases.append(
            motif(
                f"gather-{len(indices)}",
                "rewrite_gather",
                [n("Gather", "x indices", "y", axis=2)],
                {"indices": np.array(indices, np.int64)},
                expected_op="Crop" if len(indices) == 1 else None,
            )
        )
    for tensor_shapes in (False, True):
        weights = {"w": np.arange(6, dtype=np.float32).reshape(2, 3)}
        node = reshape("w", "y", [3, 2], weights, tensor_shapes)
        cases.append(
            motif(
                f"weight-reshape-{tensor_shapes}",
                "weight_reshape",
                [node],
                weights,
                expected_op="noop_reducedncnn",
                expected_count=1,
            )
        )
    cases.append(
        motif(
            "weight-transpose",
            "weight_transpose",
            [n("Transpose", "w", "y", perm=[1, 0])],
            {"w": np.arange(6, dtype=np.float32).reshape(2, 3)},
            expected_op="noop_reducednccn",
            expected_count=1,
        )
    )
    for kind, shape, perm, final in (
        ("shufflechannel", [1, 2, 3, 4, 5], [0, 2, 1, 3, 4], [1, 6, 4, 5]),
        ("shufflechannel", [3, 2, 20], [1, 0, 2], [2, -1, 3, 4, 5]),
        ("pixelshuffle", [-1, 3, 2, 2, 4, 5], [0, 1, 4, 2, 5, 3], [-1, 3, 8, 10]),
        ("reorg", [-1, 3, 4, 2, 5, 2], [0, 1, 3, 5, 2, 4], [-1, 12, 4, 5]),
    ):
        for tensor_shapes, constant in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            weights = {}
            nodes = [
                reshape("x", "r", shape, weights, tensor_shapes),
                n("Transpose", "r", "t", perm=perm),
            ]
            if constant:
                nodes.append(
                    n(
                        "Constant",
                        [],
                        "unused",
                        value=numpy_helper.from_array(np.array([1], np.int64)),
                    )
                )
            nodes.append(reshape("t", "y", final, weights, tensor_shapes))
            cases.append(
                motif(
                    f"{kind}-{len(shape)}-{tensor_shapes}-{constant}",
                    kind,
                    nodes,
                    weights,
                    expected_op={
                        "shufflechannel": "ShuffleChannel",
                        "pixelshuffle": "PixelShuffle",
                        "reorg": "Reorg",
                    }[kind],
                    expected_count=2,
                )
            )
    cases.append(
        motif(
            "shuffle-split-contradictory-indices",
            "shufflechannel_split",
            [
                n("ShuffleChannel", "x", "s", reverse=1),
                n("Gather", "s zero", "a", axis=0),
                n("Gather", "s one", "y", axis=0),
            ],
            {"zero": np.array([0], np.int64), "one": np.array([1], np.int64)},
        )
    )
    for hard in (False, True):
        for weighted_clip in (False, True):
            for constant in (False, True):
                weights = {"three": scalar(3), "six": scalar(6), "zero": scalar(0)}
                nodes = [n("Add", "x three", "added")]
                nodes.append(
                    n("Clip", "added zero six", "clip")
                    if weighted_clip
                    else n("Clip", "added", "clip", min=0.0, max=6.0)
                )
                if hard:
                    nodes.append(n("Mul", "x clip", "mul"))
                if constant:
                    nodes.append(
                        n(
                            "Constant",
                            [],
                            "unused",
                            value=numpy_helper.from_array(scalar(6)),
                        )
                    )
                nodes.append(n("Div", ["mul" if hard else "clip", "six"], "y"))
                method = "hardswish" if hard else "hardsigmoid"
                cases.append(
                    motif(
                        f"{method}-{weighted_clip}-{constant}",
                        method,
                        nodes,
                        weights,
                        expected_op="HardSwish" if hard else None,
                        expected_count=3 if hard else 0,
                    )
                )
    cases.append(
        motif(
            "hardswish-hardsigmoid",
            "hardswish",
            [n("HardSigmoid", "x", "sig", alpha=0.3, beta=0.7), n("Mul", "x sig", "y")],
            expected_op="HardSwish",
            expected_count=1,
        )
    )
    cases.append(
        motif(
            "swish",
            "swish",
            [n("Sigmoid", "x", "sig"), n("Mul", "x sig", "y")],
            expected_op="Swish",
            expected_count=1,
        )
    )
    cases.append(
        motif(
            "batchnorm-sandwich",
            "batchnorm1d_squeeze_unsqueeze",
            [
                n("Unsqueeze", "x", "u", axes=[2]),
                n("BatchNormalization", "u s b m v", "bn"),
                n("Squeeze", "bn", "y", axes=[2]),
            ],
            {key: np.ones(3, np.float32) for key in ("s", "b", "m", "v")},
            expected_op="noop_reducedncnn",
            expected_count=2,
        )
    )
    cases.append(
        motif(
            "prelu-weight",
            "unsqueeze_prelu",
            [n("Unsqueeze", "s", "u", axes=[1, 2]), n("PRelu", "x u", "y")],
            {"s": np.array([0.1, 0.2, 0.3], np.float32)},
            expected_op="PRelu",
            expected_count=1,
        )
    )
    for has_shape in (False, True):
        for weighted_clip in (False, True):
            nodes = [
                n("ReduceL2", "x", "l2", axes=[1]),
                n("Clip", "l2 eps", "clip")
                if weighted_clip
                else n("Clip", "l2", "clip", min=1e-5),
            ]
            if has_shape:
                nodes.append(n("Shape", "x", "shape"))
            nodes.extend(
                [
                    n("Expand", "clip shape" if has_shape else "clip", "exp"),
                    n("Div", "x exp", "y"),
                ]
            )
            cases.append(
                motif(
                    f"normalize-{has_shape}-{weighted_clip}",
                    "normalize",
                    nodes,
                    {"eps": scalar(1e-5)},
                    expected_op="Normalize",
                    expected_count=4 if has_shape else 3,
                )
            )
    for tensor_shapes in (False, True):
        weights = {
            "s": np.ones(2, np.float32),
            "b": np.zeros(2, np.float32),
            "affs": np.arange(6, dtype=np.float32),
            "affb": np.zeros(6, np.float32),
        }
        nodes = [
            reshape("x", "r", [0, 2, -1], weights, tensor_shapes),
            n("InstanceNormalization", "r s b", "norm", epsilon=1e-4),
            reshape("norm", "r2", [1, 6, 4, 5], weights, tensor_shapes),
            n("Mul", "r2 affs", "mul"),
            n("Add", "mul affb", "y"),
        ]
        cases.append(
            motif(
                f"groupnorm-{tensor_shapes}",
                "groupnorm",
                nodes,
                weights,
                expected_op="GroupNorm",
                expected_count=4,
            )
        )
    for axes in ([-1], [-2, -1]):
        for affine in (False, True):
            for coherent in (False, True):
                nodes = [
                    n("ReduceMean", "x", "mean", axes=axes),
                    n("Sub", "x mean" if coherent else "mean mean", "sub"),
                    n("Pow", "sub two", "pow"),
                    n("ReduceMean", "pow", "var", axes=axes),
                    n("Add", "var eps", "epsadd"),
                    n("Sqrt", "epsadd", "sqrt"),
                    n("Div", "sub sqrt", "div"),
                ]
                if affine:
                    nodes.extend([n("Mul", "div s", "mul"), n("Add", "mul b", "y")])
                weights = {
                    "two": scalar(2),
                    "eps": scalar(1e-5),
                    "s": np.ones(4, np.float32),
                    "b": np.zeros(4, np.float32),
                }
                cases.append(
                    motif(
                        f"layernorm-{len(axes)}-{affine}-{'real-edges' if coherent else 'original-guard-state'}",
                        "layernorm",
                        nodes,
                        weights,
                        references={} if coherent else {"mean": 1},
                        expected_op="LayerNorm"
                        if not coherent and len(axes) == 2
                        else None,
                        expected_error="IndexError"
                        if not coherent and len(axes) == 1
                        else None,
                        expected_count=(8 if affine else 6)
                        if not coherent and len(axes) == 2
                        else 0,
                    )
                )
    cases.append(
        motif(
            "flatten",
            "flatten",
            [
                n("Shape", "x", "shape"),
                n("Gather", "shape zero", "g", axis=0),
                n(
                    "Constant",
                    [],
                    "minus",
                    value=numpy_helper.from_array(np.array([-1], np.int64)),
                ),
                n("Unsqueeze", "g", "u", axes=[0]),
                n("Unsqueeze", "minus", "u2", axes=[0]),
                n("Concat", "u u2", "cat", axis=0),
                n("Reshape", "x cat", "y"),
            ],
            {"zero": np.array([0], np.int64)},
            expected_op="Flatten",
            expected_count=5,
        )
    )
    for binary in ("Add", "Sub", "Mul", "Div", "Min", "Max"):
        for side in (0, 1):
            inputs = ["x", "expanded"] if side else ["expanded", "x"]
            cases.append(
                motif(
                    f"expand-{binary}-{side}",
                    "expand_broadcast",
                    [n("Expand", "z shape", "expanded"), n(binary, inputs, "y")],
                    {"shape": np.array([1, 3, 4, 5], np.int64)},
                    expected_op=binary,
                    expected_count=1,
                )
            )
    for recurrent in ("LSTM", "GRU", "RNN"):
        for direction in ("forward", "reverse", "bidirectional"):
            for extra in (False, True):
                weights = {"shape": np.array([0, 0, -1], np.int64)}
                nodes = [n(recurrent, "x", "r", direction=direction)]
                if direction == "bidirectional":
                    nodes.extend(
                        [
                            n("Transpose", "r", "t", perm=[0, 2, 1, 3]),
                            n("Reshape", "t shape", "y"),
                        ]
                    )
                else:
                    nodes.append(n("Squeeze", "r", "y", axes=[1]))
                if extra:
                    nodes.append(n("Transpose", "y", "out", perm=[1, 0, 2]))
                cases.append(
                    motif(
                        f"rnn-{recurrent}-{direction}-{extra}",
                        "lstm_gru_rnn",
                        nodes,
                        weights,
                        expected_count=(2 if direction == "bidirectional" else 1)
                        + int(extra),
                        expected_op="noop_reducedncnn",
                    )
                )
        cases.append(
            motif(
                f"rnn-pretranspose-{recurrent}",
                "lstm_gru_rnn",
                [n("Transpose", "x", "t", perm=[1, 0, 2]), n(recurrent, "t", "y")],
                expected_op=recurrent,
                expected_count=1,
            )
        )
    cases.extend(
        attention(combined, tensors)
        for combined in (False, True)
        for tensors in (False, True)
    )
    for binary in ("Add", "Sub", "Mul", "Div", "Min", "Max", "Pow"):
        for side in (0, 1):
            cases.append(
                motif(
                    f"scalar-{binary}-{side}",
                    "binaryop_with_scalar",
                    [n(binary, ["x", "scalar"] if side else ["scalar", "x"], "y")],
                    {"scalar": scalar(0.25)},
                    expected_op={"Sub": "RSub", "Div": "RDiv"}.get(binary, binary)
                    if not side
                    else binary,
                    expected_error="IndexError"
                    if not side and binary not in ("Sub", "Div")
                    else None,
                )
            )
    return cases


MOTIFS = motifs()


class ConverterState(Protocol):
    """The converter state the passes read and edit, as the frozen converter declares
    it. The port's constructor sets it in native code, so its class declares none."""

    onnx_graph: GraphProto
    mutable_graph_nodes: list[NodeProto]
    node_count: int
    weights: dict[str, TensorProto]
    node_reference: dict[str, int]
    blob_names: dict[str, None]


def prepare(converter, source) -> ConverterState:
    value = converter(copy.deepcopy(source))
    # Same reference/blob bookkeeping as convert(), without invoking earlier
    # fusion passes that would consume the precise motif under test.
    for node in value.onnx_graph.node:
        if node.op_type == "Constant":
            value.weights[node.output[0]] = next(
                a.t for a in node.attribute if a.name == "value"
            )
        for name in node.input:
            value.blob_names[name] = None
            value.node_reference[name] = value.node_reference.get(name, 0) + 1
        for name in node.output:
            value.blob_names[name] = None
            value.node_reference[name] = 0
    return value


def check_fusion(case, edit=None, count=None):
    a = prepare(old.Onnx2NcnnConverter, case.source)
    b = prepare(current.Onnx2NcnnConverter, case.source)
    for value in (a, b):
        value.node_reference.update(case.references)
        if edit:
            edit(value)
    ca, cb = (
        copy.deepcopy([0] if count is None else count),
        copy.deepcopy([0] if count is None else count),
    )
    no_count = case.method in ("fuse_rewrite_gather", "fuse_binaryop_with_scalar")
    ea = outcome(lambda: getattr(a, case.method)(*(() if no_count else (ca,))))
    eb = outcome(lambda: getattr(b, case.method)(*(() if no_count else (cb,))))
    assert ea == eb, (case.name, ea, eb)
    assert snapshot((b, cb)) == snapshot((a, ca)), case.name
    # Preserve aliases between mutable list nodes and graph storage, not only
    # serialized bytes. Weight reshape/transpose also reuse initializer objects.
    for value in (a, b):
        for i, node in enumerate(value.mutable_graph_nodes):
            assert node is value.onnx_graph.node[i]
    return a, ca, ea


@pytest.fixture(scope="module", autouse=True)
def fusion_coverage(record_testsuite_property):
    path = str(FROZEN / "onnx_to_ncnn.py")
    cov = coverage.Coverage(data_file=None, branch=True, include=[path])
    cov.start()
    yield
    cov.stop()
    parser = PythonParser(filename=path)
    parser.parse_source()
    possible = parser.arcs()
    # Normalize traced continuation lines to their logical statement, just as
    # coverage.py's standard analysis does for multiline Boolean guards.
    observed = set(parser.translate_arcs(cov.get_data().arcs(path) or []))
    lines = set(parser.translate_lines(cov.get_data().lines(path) or []))
    counts = parser.exit_counts()
    tree = ast.parse((FROZEN / "onnx_to_ncnn.py").read_text())
    report = {}
    for method in ast.walk(tree):
        if not isinstance(method, ast.FunctionDef) or not method.name.startswith(
            "fuse_"
        ):
            continue
        end = method.end_lineno
        assert end is not None
        statements = {line for line in parser.statements if method.lineno < line <= end}
        arcs = {
            arc
            for arc in possible
            if method.lineno < arc[0] <= end and counts.get(arc[0], 0) > 1
        }
        report[method.name] = {
            "statements": len(statements),
            "executed": len(statements & lines),
            "branch_destinations": len(arcs),
            "taken": len(arcs & observed),
            "missing_lines": sorted(statements - lines),
            "missing_branch_arcs": sorted(arcs - observed),
        }
    encoded = json.dumps(report, sort_keys=True)
    record_testsuite_property("onnx_fusion_coverage", encoded)
    print("FUSION_COVERAGE=" + encoded)


@pytest.mark.parametrize("case", MOTIFS, ids=lambda case: case.name)
def test_full_motif(case):
    value, reduced, result = check_fusion(case)
    assert result[0] == (case.expected_error or "ok"), (case.name, result)
    assert reduced == [case.expected_count]
    if case.expected_op:
        assert value.mutable_graph_nodes[-1].op_type == case.expected_op
    if case.method in ("fuse_weight_reshape", "fuse_weight_transpose"):
        assert value.weights["w"] is value.weights["y"]
    if case.method == "fuse_multiheadattention":
        # Both motifs reach the actual late protobuf error after changing every
        # precursor op, references, blobs and output inputs. No no-op fixture.
        assert all(
            node.op_type == "noop_reducedncnn"
            for node in value.mutable_graph_nodes[:-1]
        )
        assert "embed_dim" in [a.name for a in value.mutable_graph_nodes[-1].attribute]
        assert "num_heads" not in [
            a.name for a in value.mutable_graph_nodes[-1].attribute
        ]


def mutations(case):
    nodes = case.source.graph.node
    for i, node in enumerate(nodes):
        yield ("operator", i, None)
        yield ("truncate", i, None)
        for j in range(len(node.input)):
            yield ("edge", i, j)
            yield ("remove-input", i, j)
        for j in range(len(node.output)):
            yield ("reference-two", i, j)
            yield ("reference-missing", i, j)
        for j, attribute in enumerate(node.attribute):
            yield ("remove-attribute", i, j)
            if attribute.type == A.INTS:
                yield ("empty-attribute", i, j)
                for k in range(len(attribute.ints)):
                    yield ("integer-element", i, (j, k))
            elif attribute.type in (A.INT, A.FLOAT, A.STRING):
                yield ("scalar-attribute", i, j)
    for name in (tensor.name for tensor in case.source.graph.initializer):
        for kind in ("missing-weight", "wrong-weight", "weight-rank", "typed-weight"):
            yield (kind, name, None)


MUTATIONS = [(case, mutation) for case in MOTIFS for mutation in mutations(case)]


def apply_mutation(value, mutation):
    kind, i, j = mutation
    if kind == "missing-weight":
        value.weights.pop(i)
        return
    if kind in ("wrong-weight", "weight-rank", "typed-weight"):
        tensor = value.weights[i]
        data = numpy_helper.to_array(tensor)
        if kind == "wrong-weight":
            data = np.full(data.shape, 7, data.dtype)
            tensor.CopyFrom(numpy_helper.from_array(data, tensor.name))
        elif kind == "weight-rank":
            tensor.dims.insert(0, 1)
        else:
            tensor.CopyFrom(
                helper.make_tensor(
                    tensor.name, tensor.data_type, data.shape, data.ravel().tolist()
                )
            )
        return
    if kind == "truncate":
        value.node_count = i
        return
    node = value.mutable_graph_nodes[i]
    if kind == "operator":
        node.op_type = "Unrelated"
    elif kind == "edge":
        node.input[j] = "unrelated-edge"
        value.node_reference["unrelated-edge"] = 1
    elif kind == "remove-input":
        del node.input[j]
    elif kind == "reference-two":
        value.node_reference[node.output[j]] = 2
    elif kind == "reference-missing":
        del value.node_reference[node.output[j]]
    elif kind == "remove-attribute":
        del node.attribute[j]
    elif kind == "empty-attribute":
        node.attribute[j].ClearField("ints")
    elif kind == "integer-element":
        index, element = j
        node.attribute[index].ints[element] += 7
    elif kind == "scalar-attribute":
        attribute = node.attribute[j]
        if attribute.type == A.INT:
            attribute.i += 7
        elif attribute.type == A.FLOAT:
            attribute.f += 0.125
        else:
            attribute.s = b"unrelated"
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize(
    "case,mutation",
    MUTATIONS,
    ids=[f"{case.name}-{kind}-{i}-{j}" for case, (kind, i, j) in MUTATIONS],
)
def test_one_guard_or_boundary_at_a_time(case, mutation):
    check_fusion(case, lambda value: apply_mutation(value, mutation))


@pytest.mark.parametrize("case", MOTIFS, ids=lambda case: case.name)
def test_reduction_counter_error_preserves_partial_mutations(case):
    check_fusion(case, count=[])


@pytest.mark.parametrize("change", ["missing", "nonnegative"])
def test_flatten_constant_weight_guards(change):
    case = next(case for case in MOTIFS if case.name == "flatten")

    def edit(value):
        if change == "missing":
            value.weights.pop("minus")
        else:
            value.weights["minus"].CopyFrom(
                numpy_helper.from_array(np.array([0], np.int64))
            )

    _, reduced, result = check_fusion(case, edit)
    assert result == ("ok", None)
    assert reduced == [0]


@pytest.mark.parametrize("combined", [False, True])
def test_attention_bias_size_guard(combined):
    case = attention(combined, True)

    def edit(value):
        key = "qkvb" if combined else "qb"
        value.weights[key].CopyFrom(numpy_helper.from_array(np.zeros(3, np.float32)))

    _, reduced, result = check_fusion(case, edit)
    assert result == ("ok", None)
    assert reduced == [0]


def test_layernorm_affine_size_mismatch_keeps_nonaffine_fusion():
    case = next(
        case for case in MOTIFS if case.name == "layernorm-2-True-original-guard-state"
    )

    def edit(value):
        value.weights["b"].CopyFrom(numpy_helper.from_array(np.zeros(3, np.float32)))

    value, reduced, result = check_fusion(case, edit)
    assert result == ("ok", None)
    assert reduced == [6]
    assert value.mutable_graph_nodes[6].op_type == "LayerNorm"
    assert value.mutable_graph_nodes[-1].op_type == "Add"


def test_weight_reshape_extra_input_preserves_original_empty_shape():
    case = next(case for case in MOTIFS if case.name == "weight-reshape-True")

    def edit(value):
        value.mutable_graph_nodes[0].input.append("unused-third-input")

    value, reduced, result = check_fusion(case, edit)
    assert result == ("ok", None)
    assert reduced == [1]
    assert list(value.weights["w"].dims) == []
    assert value.weights["y"] is value.weights["w"]


def test_recurrent_attribute_shape_and_bidirectional_squeeze_rejection():
    case = next(case for case in MOTIFS if case.name == "rnn-LSTM-bidirectional-False")

    def edit(value):
        node = value.mutable_graph_nodes[2]
        node.ClearField("input")
        node.input.append("t")
        node.attribute.append(helper.make_attribute("shape", [0, 0, -1]))

    _, reduced, result = check_fusion(case, edit)
    assert result == ("ok", None)
    assert reduced == [2]
    rejected = motif(
        "bidirectional-squeeze",
        "lstm_gru_rnn",
        [
            n("LSTM", "x", "r", direction="bidirectional"),
            n("Squeeze", "r", "y", axes=[1]),
        ],
    )
    _, reduced, result = check_fusion(rejected)
    assert result == ("ok", None)
    assert reduced == [0]


@pytest.mark.parametrize("case", MOTIFS, ids=lambda case: case.name)
def test_whole_conversion_pipeline(case):
    # Includes earlier passes, later NCNN lowering/optimizer/serialization and
    # real reference bookkeeping, retaining exact original rejected outcomes.
    check(case.source)
