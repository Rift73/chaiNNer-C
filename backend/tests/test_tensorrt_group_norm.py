"""Which TensorRT builds take GroupNorm written out, and that it keeps the model's values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import numpy_helper

from nodes.impl.tensorrt.group_norm import written_out_group_norm


class Net(torch.nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.head = torch.nn.Conv2d(3, channels, 3, padding=1)
        self.norm = torch.nn.GroupNorm(groups, channels)
        self.tail = torch.nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.norm(self.head(x)))


def exported(
    directory: Path,
    channels: int,
    groups: int,
    opset: int = 14,
    dtype: torch.dtype = torch.float32,
) -> bytes:
    torch.manual_seed(0)
    net = Net(channels, groups).eval().requires_grad_(False).to(dtype)
    path = directory / "model.onnx"
    torch.onnx.export(
        net,
        (torch.rand(1, 3, 32, 32, dtype=dtype),),
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
    rewritten = written_out_group_norm(data, (32, 32), (high, high), True)
    assert rewritten is not None
    ops = {node.op_type for node in onnx.load_from_string(rewritten).graph.node}
    assert "InstanceNormalization" not in ops
    x = np.random.default_rng(0).random((1, 3, 37, 51), np.float32)
    outputs = [
        np.asarray(
            ort.InferenceSession(model, providers=["CPUExecutionProvider"]).run(
                None, {"input": x}
            )[0]
        )
        for model in (data, rewritten)
    ]
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=1e-5)


def fp32_arithmetic(rewritten: bytes) -> None:
    """Raises unless each GroupNorm's arithmetic reads an FP32 cast and ends in a cast."""
    graph = onnx.load_from_string(rewritten).graph
    makers = {output: node for node in graph.node for output in node.output}
    means = [n for n in graph.node if n.op_type == "ReduceMean"]
    assert means
    for mean in means[::2]:
        cast = makers[mean.input[0]]
        assert cast.op_type == "Cast"
        assert cast.attribute[0].i == onnx.TensorProto.FLOAT
    epsilons = [n for n in graph.node if n.output[0].endswith("_epsilon")]
    assert epsilons
    for epsilon in epsilons:
        assert epsilon.attribute[0].t.data_type == onnx.TensorProto.FLOAT
    divs = [n for n in graph.node if n.output[0].endswith("_written_out")]
    assert len(divs) == len(means) // 2
    users = [
        n for n in graph.node if n.input and n.input[0] in {d.output[0] for d in divs}
    ]
    assert [n.op_type for n in users] == ["Cast"] * len(divs)


def test_an_fp16_group_norm_is_written_out_in_fp32(tmp_path: Path):
    data = exported(tmp_path, 16, 4, dtype=torch.float16)
    rewritten = written_out_group_norm(data, (32, 32), (64, 64), True)
    assert rewritten is not None
    fp32_arithmetic(rewritten)
    ops = {node.op_type for node in onnx.load_from_string(rewritten).graph.node}
    assert "InstanceNormalization" not in ops
    x = np.random.default_rng(0).random((1, 3, 37, 51), np.float32).astype(np.float16)
    outputs = [
        np.asarray(
            ort.InferenceSession(model, providers=["CPUExecutionProvider"]).run(
                None, {"input": x}
            )[0]
        )
        for model in (data, rewritten)
    ]
    assert outputs[1].dtype == np.float16
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=2e-3)


def test_a_bf16_group_norm_is_written_out_in_fp32(tmp_path: Path):
    model = onnx.load_from_string(exported(tmp_path, 16, 4))
    norm = next(n for n in model.graph.node if n.op_type == "InstanceNormalization")
    tensors = {t.name: t for t in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            tensors[node.output[0]] = node.attribute[0].t
    for name in norm.input[1:]:
        tensor = tensors[name]
        bits = torch.from_numpy(numpy_helper.to_array(tensor).copy()).to(torch.bfloat16)
        tensor.data_type = onnx.TensorProto.BFLOAT16
        tensor.raw_data = bits.view(torch.uint16).numpy().tobytes()
    rewritten = written_out_group_norm(
        model.SerializeToString(), (32, 32), (64, 64), True
    )
    assert rewritten is not None
    fp32_arithmetic(rewritten)
    casts = [
        n
        for n in onnx.load_from_string(rewritten).graph.node
        if n.op_type == "Cast" and n.output[0] == norm.output[0]
    ]
    assert [c.attribute[0].i for c in casts] == [onnx.TensorProto.BFLOAT16]


@pytest.mark.parametrize(
    ("channels", "groups", "low", "high", "dynamic"),
    [
        (16, 4, 64, 64, False),  # fixed
        (16, 4, 64, 64, True),  # one size
        (16, 4, 32, 129, True),  # above the limit
        (64, 4, 32, 65, True),  # above the limit
        (16, 1, 32, 64, True),  # one group
        (16, 16, 32, 64, True),  # one channel per group
    ],
)
def test_other_builds_stay_as_they_are(
    tmp_path: Path, channels: int, groups: int, low: int, high: int, dynamic: bool
):
    data = exported(tmp_path, channels, groups)
    size = (low, low), (high, high)
    assert written_out_group_norm(data, *size, dynamic) is None


def test_a_model_without_group_norm_is_not_read():
    # Not valid ONNX, so reading it would raise.
    assert written_out_group_norm(b"no", (32, 32), (64, 64), True) is None


def test_an_opset_declared_as_ai_onnx_is_read(tmp_path: Path):
    model = onnx.load_from_string(exported(tmp_path, 16, 4))
    model.opset_import[0].domain = "ai.onnx"
    data = model.SerializeToString()
    assert written_out_group_norm(data, (32, 32), (64, 64), True) is not None


def test_a_reshape_whose_shape_is_another_constant_form_stays_as_it_is(tmp_path: Path):
    model = onnx.load_from_string(exported(tmp_path, 16, 4))
    norm = next(n for n in model.graph.node if n.op_type == "InstanceNormalization")
    reshape = next(n for n in model.graph.node if norm.input[0] in n.output)
    name = reshape.input[1]
    kept = [t for t in model.graph.initializer if t.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    nodes = [n for n in model.graph.node if name not in n.output]
    nodes.insert(
        0, onnx.helper.make_node("Constant", [], [name], value_ints=[0, 4, -1])
    )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    data = model.SerializeToString()
    assert written_out_group_norm(data, (32, 32), (64, 64), True) is None


def test_a_reshape_with_its_shape_as_an_attribute_stays_as_it_is():
    # Reshape's form below opset 5, which the rewrite does not read.
    reshape = onnx.helper.make_node("Reshape", ["x"], ["r"], shape=[0, 4, -1])
    norm = onnx.helper.make_node("InstanceNormalization", ["r", "s", "b"], ["y"])
    graph = onnx.helper.make_graph(
        [reshape, norm],
        "old",
        [
            onnx.helper.make_tensor_value_info(
                "x", onnx.TensorProto.FLOAT, [1, 16, "h", "w"]
            )
        ],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, None)],
        [
            onnx.numpy_helper.from_array(np.ones(4, np.float32), "s"),
            onnx.numpy_helper.from_array(np.zeros(4, np.float32), "b"),
        ],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 4)]
    )
    data = model.SerializeToString()
    assert written_out_group_norm(data, (32, 32), (64, 64), True) is None
