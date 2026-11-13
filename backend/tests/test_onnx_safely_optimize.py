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

import numpy as np
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


def reshape_of_a_reshape(second_shape: list[int]) -> bytes:
    """x, 1x1x4x6, through Reshape [0, 0, 2, 2, 6] (1x1x2x2x6), then `second_shape`."""
    graph = helper.make_graph(
        [
            helper.make_node("Reshape", ["x", "first"], ["mid"], name="n0"),
            helper.make_node("Reshape", ["mid", "second"], ["y"], name="n1"),
        ],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 6])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2, 3, 2])],
        [
            helper.make_tensor("first", TensorProto.INT64, [5], [0, 0, 2, 2, 6]),
            helper.make_tensor("second", TensorProto.INT64, [6], second_shape),
        ],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    ).SerializeToString()


# PyTorch's pixel_unshuffle exports these two Reshapes; the second's 0s copy dimensions
# from the first's output, so the optimizer's elimination of the first gave it another
# shape, and the model failed in ONNX Runtime after Optimize Model and in Interpolate
# Models.
def test_a_reshape_that_copies_dimensions_keeps_the_reshape_before_it():
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    sys.path.insert(0, str(BACKEND))
    from nodes.impl.onnx.utils import safely_optimize_onnx_model

    data = reshape_of_a_reshape([0, 0, 0, 0, -1, 2])
    x = np.arange(24, dtype=np.float32).reshape(1, 1, 4, 6)

    def run(model: bytes) -> np.ndarray:
        session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        return session.run(None, {"x": x})[0]

    optimized = safely_optimize_onnx_model(onnx.load_model_from_string(data))
    assert np.array_equal(run(optimized.SerializeToString()), run(data))
    assert run(data).shape == (1, 1, 2, 2, 3, 2)


def test_a_reshape_of_a_reshape_with_a_full_shape_is_still_fused():
    onnx = pytest.importorskip("onnx")
    sys.path.insert(0, str(BACKEND))
    from nodes.impl.onnx.utils import safely_optimize_onnx_model

    data = reshape_of_a_reshape([1, 1, 2, 2, 3, 2])
    optimized = safely_optimize_onnx_model(onnx.load_model_from_string(data))
    assert [node.op_type for node in optimized.graph.node] == ["Reshape"]


# The model is returned as it is when onnxoptimizer fails, and the log says why.
def test_an_optimizer_error_is_logged(monkeypatch: pytest.MonkeyPatch):
    onnx = pytest.importorskip("onnx")
    onnxoptimizer = pytest.importorskip("onnxoptimizer")
    sys.path.insert(0, str(BACKEND))
    from nodes.impl.onnx import utils

    def fail(*_: object) -> None:
        raise RuntimeError("optimizer failed")

    warnings: list[str] = []
    monkeypatch.setattr(onnxoptimizer, "optimize", fail)
    monkeypatch.setattr(
        utils.logger, "warning", lambda message, **_: warnings.append(message)
    )
    data = model(CASES["valid"])
    optimized = utils.safely_optimize_onnx_model(onnx.load_model_from_string(data))
    assert optimized.SerializeToString() == data
    assert len(warnings) == 1
    assert "optimizer failed" in warnings[0]
