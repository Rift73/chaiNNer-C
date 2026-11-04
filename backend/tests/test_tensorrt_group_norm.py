"""Which TensorRT builds take GroupNorm written out, and that it keeps the model's values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from nodes.impl.tensorrt.group_norm import written_out_group_norm


class Net(torch.nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.head = torch.nn.Conv2d(3, channels, 3, padding=1)
        self.norm = torch.nn.GroupNorm(groups, channels)
        self.tail = torch.nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.norm(self.head(x)))


def exported(directory: Path, channels: int, groups: int, opset: int = 14) -> bytes:
    torch.manual_seed(0)
    net = Net(channels, groups).eval().requires_grad_(False)
    path = directory / "model.onnx"
    torch.onnx.export(
        net,
        (torch.rand(1, 3, 32, 32),),
        str(path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}},
        dynamo=False,
    )
    return path.read_bytes()


@pytest.mark.parametrize("opset", [14, 20])
@pytest.mark.parametrize("high", [64, 128], ids=["wrong plan", "at the limit"])
def test_a_small_range_is_written_out_with_the_same_values(
    tmp_path: Path, opset: int, high: int
):
    data = exported(tmp_path, 16, 4, opset)
    rewritten = written_out_group_norm(data, "fp32", (32, 32), (high, high), True)
    assert rewritten is not None
    ops = {node.op_type for node in onnx.load_from_string(rewritten).graph.node}
    assert "InstanceNormalization" not in ops
    x = np.random.default_rng(0).random((1, 3, 37, 51), np.float32)
    outputs = [
        ort.InferenceSession(model, providers=["CPUExecutionProvider"]).run(
            None, {"input": x}
        )[0]
        for model in (data, rewritten)
    ]
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    ("channels", "groups", "precision", "low", "high", "dynamic"),
    [
        (16, 4, "fp32", 64, 64, False),  # fixed
        (16, 4, "fp32", 64, 64, True),  # one size
        (16, 4, "fp32", 32, 129, True),  # above the limit
        (64, 4, "fp32", 32, 65, True),  # above the limit
        (16, 4, "fp16", 32, 64, True),  # written out, FP16 loses too much
        (16, 4, "bf16", 32, 64, True),
        (16, 4, "mixed", 32, 64, True),
        (16, 1, "fp32", 32, 64, True),  # one group
        (16, 16, "fp32", 32, 64, True),  # one channel per group
    ],
)
def test_other_builds_stay_as_they_are(
    tmp_path: Path,
    channels: int,
    groups: int,
    precision: str,
    low: int,
    high: int,
    dynamic: bool,
):
    data = exported(tmp_path, channels, groups)
    size = (low, low), (high, high)
    assert written_out_group_norm(data, precision, *size, dynamic) is None


def test_a_model_without_group_norm_is_not_read():
    # Not valid ONNX, so reading it would raise.
    assert written_out_group_norm(b"no", "fp32", (32, 32), (64, 64), True) is None


def test_an_opset_declared_as_ai_onnx_is_read(tmp_path: Path):
    model = onnx.load_from_string(exported(tmp_path, 16, 4))
    model.opset_import[0].domain = "ai.onnx"
    data = model.SerializeToString()
    assert written_out_group_norm(data, "fp32", (32, 32), (64, 64), True) is not None


def test_a_reshape_whose_shape_is_another_constant_form_stays_as_it_is(tmp_path: Path):
    model = onnx.load_from_string(exported(tmp_path, 16, 4))
    norm = next(n for n in model.graph.node if n.op_type == "InstanceNormalization")
    reshape = next(n for n in model.graph.node if norm.input[0] in n.output)
    name = reshape.input[1]
    kept = [t for t in model.graph.initializer if t.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    nodes = [n for n in model.graph.node if name not in n.output]
    nodes.insert(0, onnx.helper.make_node("Constant", [], [name], value_ints=[0, 4, -1]))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    data = model.SerializeToString()
    assert written_out_group_norm(data, "fp32", (32, 32), (64, 64), True) is None
