"""Convert To NCNN orders an ONNX graph's nodes before converting: a graph whose nodes
are out of order converts as the ordered one does, and a graph with a cycle is refused.
Before, the converter's own reordering (upstream's: it swaps a node with the one making
its missing input, counted from the wrong index) raised IndexError on both, or swapped
two nodes back and forth for ever."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest
from onnx import TensorProto, helper

# The converter is native, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from nodes.impl.onnx.load import load_onnx_model
from packages.chaiNNer_onnx.onnx.utility.convert_to_ncnn import convert_to_ncnn_node

BACKEND = Path(__file__).resolve().parents[1] / "src"


def model(nodes: list[tuple[str, str, list[str], str]]) -> bytes:
    """A graph of (name, op, inputs, output) nodes from input x to output y, with
    weights w for a 3-channel 1x1 Conv (a model NCNN can run has one)."""
    made = [
        helper.make_node(op, inputs, [output], name=name)
        for name, op, inputs, output in nodes
    ]
    for node in made:
        if node.op_type == "Conv":
            node.attribute.append(helper.make_attribute("kernel_shape", [1, 1]))
    graph = helper.make_graph(
        made,
        "order",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor("w", TensorProto.FLOAT, [3, 3, 1, 1], [0.5] * 9)],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    ).SerializeToString()


def param(data: bytes) -> str:
    """The NCNN .param text Convert To NCNN makes."""
    return convert_to_ncnn_node(load_onnx_model(data), 0)[0].model.write_param()


ORDERED = [
    ("first", "Conv", ["x", "w"], "mid"),
    ("second", "Relu", ["mid"], "y"),
]


def test_an_unordered_graph_converts_as_the_ordered_one():
    assert param(model(ORDERED[::-1])) == param(model(ORDERED))


CYCLES = {
    "a node that takes its own output": [("loop", "Relu", ["y"], "y")],
    "two nodes that take each other's output": [
        ("first", "Relu", ["back"], "mid"),
        ("second", "Sigmoid", ["mid"], "back"),
        ("last", "Relu", ["back"], "y"),
    ],
    # the converter swapped these two back and forth for ever
    "two nodes first that take their own output": [
        ("one", "Relu", ["s"], "s"),
        ("two", "Relu", ["t"], "t"),
        ("last", "Add", ["s", "t"], "y"),
    ],
}

# Converts each model it reads and prints how each ended; apart, so a hang fails the test.
CONVERT = """
import base64, json, sys
sys.path[:0] = [sys.argv[1]]
from nodes.impl.onnx.load import load_onnx_model
from packages.chaiNNer_onnx.onnx.utility.convert_to_ncnn import convert_to_ncnn_node
for data in json.load(sys.stdin):
    try:
        convert_to_ncnn_node(load_onnx_model(base64.b64decode(data)), 0)
        print("converted", flush=True)
    except ValueError as error:
        print(error, flush=True)
"""


def test_a_graph_with_a_cycle_is_refused():
    done = subprocess.run(
        [sys.executable, "-B", "-c", CONVERT, str(BACKEND)],
        input=json.dumps(
            [base64.b64encode(model(nodes)).decode() for nodes in CYCLES.values()]
        ),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        "The ONNX model's graph has a cycle: loop -> loop. The file is damaged.",
        "The ONNX model's graph has a cycle: first -> second -> first. The file is damaged.",
        "The ONNX model's graph has a cycle: one -> one. The file is damaged.",
    ]
