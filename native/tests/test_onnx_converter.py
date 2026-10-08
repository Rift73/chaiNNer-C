"""Differential application conversion, including state after rejected graphs."""

from __future__ import annotations

import ast
import copy
import importlib
import sys
import types
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from google.protobuf.message import Message
from onnx import TensorProto as T
from onnx import helper, numpy_helper
from test_ncnn_graph import ReferenceOptimizer, reference

from nodes.impl.onnx import onnx_to_ncnn as current

FROZEN = Path(__file__).with_name("reference_onnx_graph")
PACKAGE = "nodes.impl._reference_onnx_converter"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(FROZEN)]
sys.modules[PACKAGE] = package
old = importlib.import_module(PACKAGE + ".onnx_to_ncnn")
for name in ("NcnnLayer", "NcnnModel"):
    setattr(old, name, getattr(reference, name))
old.__dict__.update(NcnnOptimizer=ReferenceOptimizer)


def snapshot(value, seen=None):
    if seen is None:
        seen = {}
    if isinstance(value, (str, bytes, int, bool, type(None))):
        return value
    if isinstance(value, (float, np.generic)):
        return (
            type(value).__name__,
            np.asarray(value).dtype.str,
            np.asarray(value).tobytes(),
        )
    if id(value) in seen:
        return ("alias", seen[id(value)])
    seen[id(value)] = len(seen)
    if isinstance(value, Message):
        return (type(value).__name__, value.SerializeToString())
    if isinstance(value, np.ndarray):
        return (
            value.dtype.str,
            value.shape,
            value.strides,
            value.flags.writeable,
            value.tobytes(),
        )
    if isinstance(value, dict):
        return [(snapshot(k, seen), snapshot(v, seen)) for k, v in value.items()]
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [snapshot(v, seen) for v in value])
    return (type(value).__name__, snapshot(vars(value), seen))


def outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    try:
        return ("ok", call())
    except Exception as error:
        return (type(error).__name__, str(error))


def model(nodes, weights=None, inputs=("x",), outputs=None):
    outputs = list(nodes[-1].output) if outputs is None and nodes else outputs or ["x"]
    initializers = [
        numpy_helper.from_array(np.asarray(v), name)
        for name, v in (weights or {}).items()
    ]
    return helper.make_model(
        helper.make_graph(
            nodes,
            "conversion",
            [helper.make_tensor_value_info(v, T.FLOAT, [1, 3, 5, 7]) for v in inputs],
            [helper.make_tensor_value_info(v, T.FLOAT, [1, 3, 5, 7]) for v in outputs],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 11)],
        ir_version=9,
    )


def check(source, fp16=False, memory=True, success=False):
    a, b = (
        old.Onnx2NcnnConverter(copy.deepcopy(source)),
        current.Onnx2NcnnConverter(copy.deepcopy(source)),
    )
    expected = outcome(lambda: a.convert(fp16, memory))
    actual = outcome(lambda: b.convert(fp16, memory))
    assert actual[0] == expected[0], (expected, actual)
    if actual[0] == "ok":
        assert snapshot(actual[1]) == snapshot(expected[1])
        assert outcome(actual[1].write_param) == outcome(expected[1].write_param)
        assert outcome(actual[1].serialize_weights) == outcome(
            expected[1].serialize_weights
        )
    else:
        assert actual == expected
    assert snapshot(b) == snapshot(a)
    if success:
        assert actual[0] == "ok", actual
    return actual


def operators():
    result = set()
    for node in ast.walk(ast.parse((FROZEN / "onnx_to_ncnn.py").read_text())):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "op"
        ):
            for part in ast.walk(node.comparators[0]):
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    result.add(part.value)
    return sorted(result)


@pytest.mark.parametrize("op", operators())
@pytest.mark.parametrize("attributes", [{}, {"axis": -1}, {"unrecognized": "é中"}])
@pytest.mark.parametrize("fp16,memory", [(False, True), (True, False)])
def test_every_operator_default_and_errors(op, attributes, fp16, memory):
    check(
        model([helper.make_node(op, ["x"], ["y"], name="layer", **attributes)]),
        fp16,
        memory,
    )


UNARY = "Abs Acos Asin Atan Ceil Cos Exp Floor Gelu HardSigmoid HardSwish Log Neg Reciprocal Relu Sigmoid Sin Softplus Sqrt Swish Tan Tanh".split()


@pytest.mark.parametrize("op", UNARY)
@pytest.mark.parametrize("fp16", [False, True])
def test_unary_success(op, fp16):
    check(model([helper.make_node(op, ["x"], ["y"], name="test")]), fp16, success=True)


@pytest.mark.parametrize("op", "Add Sub Mul Div Max Min Pow RSub RDiv Sum".split())
@pytest.mark.parametrize("weights", [False, True])
@pytest.mark.parametrize("size", [1, 3, 15])
def test_binary_weight_modes(op, weights, size):
    source = model(
        [helper.make_node(op, ["x", "b"], ["y"], name="binary")],
        {"b": np.linspace(-1, 2, size, dtype=np.float32)} if weights else None,
        inputs=("x",) if weights else ("x", "b"),
    )
    check(source, success=not weights)


@pytest.mark.parametrize("memory", [False, True])
def test_squared_deviation_converts(memory):
    """d = x - mean(x); d * d. Upstream's inverted MemoryData test in the optimizer took
    the Sub - Split - Mul chain for a MemoryData fusion and raised KeyError: 'data'."""
    nodes = [
        helper.make_node("ReduceMean", ["x"], ["m"], name="mean", axes=[1]),
        helper.make_node("Sub", ["x", "m"], ["d"], name="sub"),
        helper.make_node("Mul", ["d", "d"], ["y"], name="square"),
    ]
    check(model(nodes), memory=memory, success=True)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("op", ["Conv", "ConvTranspose"])
@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("fp16", [False, True])
def test_convolution_weights(op, group, bias, dtype, fp16):
    rng = np.random.default_rng(21)
    weights = {"w": rng.uniform(-1, 1, (4, 2, 3, 3)).astype(dtype)}
    if bias:
        weights["b"] = rng.uniform(-1, 1, 4).astype(dtype)
    source = model(
        [
            helper.make_node(
                op,
                ["x", "w"] + (["b"] if bias else []),
                ["y"],
                name="convolution",
                group=group,
                kernel_shape=[3, 3],
                strides=[2, 1],
                pads=[1, 2, 1, 2],
                dilations=[1, 1],
            )
        ],
        weights,
    )
    check(source, fp16)


@pytest.mark.parametrize("seed", range(30))
def test_branching_chain(seed):
    rng = np.random.default_rng(seed)
    nodes = []
    names = ["x"]
    for i in range(12):
        if rng.integers(0, 3) == 0:
            op = ["Add", "Mul", "Sub"][rng.integers(0, 3)]
            inputs = [names[rng.integers(len(names))], names[rng.integers(len(names))]]
        else:
            op = UNARY[rng.integers(len(UNARY))]
            inputs = [names[rng.integers(len(names))]]
        output = "v" + str(i)
        nodes.append(helper.make_node(op, inputs, [output], name="layer" + str(i)))
        names.append(output)
    # Upstream's optimizer rejected 15 of these seeds (KeyError on '0', '1' or 'data'):
    # its inverted MemoryData test (reference_ncnn CORRECTIONS) sent any layer feeding
    # a Split and a two-input BinaryOp into the MemoryData fusion. All now convert.
    check(model(nodes, outputs=names[-3:]), success=True)


@pytest.mark.parametrize(
    "op", ["BatchNormalization", "InstanceNormalization", "GroupNorm", "LayerNorm"]
)
@pytest.mark.parametrize("size", [1, 3, 8])
@pytest.mark.parametrize("identity", [False, True])
def test_normalization(op, size, identity):
    weights = {"s": np.ones(size, np.float32), "b": np.zeros(size, np.float32)}
    if not identity:
        weights["s"] += np.float32(0.3)
        weights["b"] += np.float32(0.125)
    inputs = ["x", "s", "b"]
    if op == "BatchNormalization":
        weights.update(
            m=np.linspace(-1, 1, size, dtype=np.float32),
            v=np.linspace(0.5, 2, size, dtype=np.float32),
        )
        inputs += ["m", "v"]
    check(
        model(
            [
                helper.make_node(
                    op,
                    inputs,
                    ["y"],
                    name="norm",
                    epsilon=1e-5,
                    channels=size,
                    groups=1,
                    affine=1,
                )
            ],
            weights,
        )
    )


@pytest.mark.parametrize("op", ["AveragePool", "MaxPool"])
@pytest.mark.parametrize("kernel", [[3], [3, 5], [1, 2, 3]])
@pytest.mark.parametrize("mode", ["NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"])
def test_pool(op, kernel, mode):
    check(
        model(
            [
                helper.make_node(
                    op, ["x"], ["y"], name="pool", kernel_shape=kernel, auto_pad=mode
                )
            ]
        )
    )


def test_parallel_independent_graphs():
    sources = [
        model([helper.make_node("Relu", ["x"], ["y"], name=str(i))]) for i in range(24)
    ]
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(check, sources))


@pytest.mark.parametrize("indices", [(0, 1), (-1, 0), (0, 9), (9, 0)])
def test_swap_identity_and_failure(indices):
    source = model(
        [helper.make_node("Relu", ["x"], ["a"]), helper.make_node("Neg", ["a"], ["y"])]
    )
    a, b = (
        old.Onnx2NcnnConverter(copy.deepcopy(source)),
        current.Onnx2NcnnConverter(copy.deepcopy(source)),
    )
    assert outcome(lambda: a.swap_nodes(*indices)) == outcome(
        lambda: b.swap_nodes(*indices)
    )
    assert snapshot(a) == snapshot(b)
