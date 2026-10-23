"""safely_optimize_onnx_model (Convert To NCNN, Optimize Model and Interpolate Models
of ONNX) hands onnxoptimizer only valid models: its native code ended the whole
process (an access violation) on a damaged file's graph whose output no node makes,
or whose node lacks the inputs its op needs. Such a model is returned as it is."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest
from onnx import TensorProto, helper

pytest.importorskip("onnxoptimizer")

BACKEND = Path(__file__).resolve().parents[1] / "src"


def model(nodes: list[tuple[str, list[str], str]]) -> bytes:
    """A graph of (op, inputs, output) nodes from input x to output y."""
    graph = helper.make_graph(
        [
            helper.make_node(op, inputs, [output], name=f"n{i}")
            for i, (op, inputs, output) in enumerate(nodes)
        ],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    ).SerializeToString()


CASES = {
    # valid: the optimizer drops the Identity
    "valid": [("Relu", ["x"], "mid"), ("Identity", ["mid"], "y")],
    # damaged: no node makes the graph's output y
    "no maker of the output": [("Relu", ["x"], "mid")],
    # damaged: a Reshape without its two inputs
    "a node without its inputs": [("Relu", ["x"], "mid"), ("Reshape", [], "y")],
}

# Optimizes each model it reads and prints its node count after; apart, so a crash
# fails the test instead of ending pytest.
OPTIMIZE = """
import base64, json, sys
sys.path[:0] = [sys.argv[1]]
import onnx
from nodes.impl.onnx.utils import safely_optimize_onnx_model
for data in json.load(sys.stdin):
    model = onnx.load_model_from_string(base64.b64decode(data))
    print(len(safely_optimize_onnx_model(model).graph.node), flush=True)
"""


def test_only_a_valid_model_is_optimized():
    done = subprocess.run(
        [sys.executable, "-B", "-c", OPTIMIZE, str(BACKEND)],
        input=json.dumps(
            [base64.b64encode(model(nodes)).decode() for nodes in CASES.values()]
        ),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines() == ["1", "1", "2"]
