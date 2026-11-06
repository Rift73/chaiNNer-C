"""Which ONNX files Build Engine gives the nested dense blocks of a new conversion, and that
they keep their values. Nothing here needs a GPU: the builders are stand-ins."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import numpy_helper
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.ESRGAN import ESRGAN
from test_tensorrt_drft import _run_node

from nodes.impl.onnx.model import ARCH_METADATA_KEY, OnnxInfo
from nodes.impl.pytorch import esrgan_export
from nodes.impl.pytorch.convert_to_onnx_impl import convert_to_onnx_impl
from nodes.impl.tensorrt.dense_blocks import nested_dense_blocks
from packages.chaiNNer_tensorrt.tensorrt.utility import build_engine

CPU = torch.device("cpu")
# One block of three dense blocks, four joins each
PLAIN = [2, 3, 4, 5] * 3
NESTED = [2] * 12


def tiny(**kwargs: Any) -> ImageModelDescriptor:
    torch.manual_seed(0)
    settings: dict[str, Any] = {"num_filters": 8, "num_blocks": 1, "scale": 4, **kwargs}
    model = ModelLoader().load_from_state_dict(ESRGAN(**settings).state_dict())
    assert isinstance(model, ImageModelDescriptor)
    return model.eval()


def new_conversion(model: ImageModelDescriptor, opset: int = 20) -> bytes:
    return convert_to_onnx_impl(model, CPU, use_half=True, opset_version=opset)


@pytest.fixture(name="plain_conversion")
def fixture_plain_conversion(monkeypatch: pytest.MonkeyPatch):
    """The FP16 export of a version before nesting."""

    def convert(model: ImageModelDescriptor, opset: int = 20) -> bytes:
        with monkeypatch.context() as patch:
            patch.setattr(esrgan_export, "for_export", lambda m: m)
            return new_conversion(model, opset)

    return convert


def joins(onnx_bytes: bytes) -> list[int]:
    graph = onnx.load_from_string(onnx_bytes).graph
    return [len(node.input) for node in graph.node if node.op_type == "Concat"]


def edited(onnx_bytes: bytes, edit: Callable[[onnx.GraphProto], None]) -> bytes:
    model = onnx.load_from_string(onnx_bytes)
    edit(model.graph)
    return model.SerializeToString()


def first(graph: onnx.GraphProto, op_type: str) -> onnx.NodeProto:
    return next(node for node in graph.node if node.op_type == op_type)


@pytest.mark.parametrize(
    "settings",
    [
        {"scale": 1},
        {"scale": 2},
        {"scale": 4},
        {"scale": 8},
        {"scale": 2, "shuffle_factor": 2},
        {"scale": 1, "shuffle_factor": 4},
        {"scale": 4, "in_nc": 1, "out_nc": 1},
        {"scale": 4, "plus": True},
    ],
    ids=["1x", "2x", "4x", "8x", "2x-unshuffle", "1x-unshuffle", "grey", "plus"],
)
@pytest.mark.parametrize("opset", [14, 17, 20])
def test_an_earlier_fp16_file_gets_the_graph_of_a_new_conversion(
    plain_conversion: Callable[..., bytes], settings: dict[str, Any], opset: int
):
    model = tiny(**settings)
    plain = plain_conversion(model, opset)
    assert joins(plain) == PLAIN
    nested = nested_dense_blocks(plain, "fp16")
    assert nested is not None
    assert (
        onnx.load_from_string(nested).graph
        == onnx.load_from_string(new_conversion(model, opset)).graph
    )


@pytest.mark.parametrize("settings", [{}, {"plus": True}], ids=["plain", "plus"])
def test_the_values_stay_exactly(settings: dict[str, Any]):
    # In FP32: ONNX Runtime's CPU has no FP16 convolution and rounds where its casts land,
    # which the two layouts place differently.
    plain = convert_to_onnx_impl(tiny(**settings), CPU, opset_version=20)
    assert joins(plain) == PLAIN
    nested = nested_dense_blocks(plain, "fp16")
    assert nested is not None
    x = np.random.default_rng(0).random((1, 3, 37, 51), np.float32)
    before, after = (
        ort.InferenceSession(data, providers=["CPUExecutionProvider"]).run(
            None, {"input": x}
        )[0]
        for data in (plain, nested)
    )
    np.testing.assert_array_equal(before, after)


def test_the_record_and_all_else_of_the_file_stay(
    plain_conversion: Callable[..., bytes],
):
    model = onnx.load_from_string(plain_conversion(tiny()))
    model.metadata_props.add(key=ARCH_METADATA_KEY, value="ESRGAN")
    nested = nested_dense_blocks(model.SerializeToString(), "fp16")
    assert nested is not None
    after = onnx.load_from_string(nested)
    assert joins(nested) == NESTED
    assert after.metadata_props == model.metadata_props
    assert after.opset_import == model.opset_import
    assert [node.output for node in after.graph.node] == [
        node.output for node in model.graph.node
    ]
    onnx.checker.check_model(after)


def test_weights_in_initializers_and_swapped_operands_are_found(
    plain_conversion: Callable[..., bytes],
):
    """Another tool can write the same arithmetic differently."""

    def rewrite(graph: onnx.GraphProto) -> None:
        for node in list(graph.node):
            if node.op_type == "Constant":
                tensor = numpy_helper.to_array(node.attribute[0].t)
                graph.initializer.append(
                    numpy_helper.from_array(tensor, node.output[0])
                )
                graph.node.remove(node)
            elif node.op_type in ("Mul", "Add"):
                node.input.reverse()

    data = edited(plain_conversion(tiny()), rewrite)
    nested = nested_dense_blocks(data, "fp16")
    assert nested is not None
    assert joins(nested) == NESTED


@pytest.mark.parametrize("precision", ["fp32", "bf16", "mixed"])
def test_only_an_fp16_build_is_nested(
    plain_conversion: Callable[..., bytes], precision: str
):
    assert nested_dense_blocks(plain_conversion(tiny()), precision) is None


def test_a_new_conversion_stays_as_it_is():
    assert nested_dense_blocks(new_conversion(tiny()), "fp16") is None


def test_a_model_without_leaky_relu_is_not_read():
    # Not valid ONNX, so reading it would raise.
    assert nested_dense_blocks(b"no", "fp16") is None


def _other_residual_scale(graph: onnx.GraphProto) -> None:
    mul = first(graph, "Mul")
    scale = next(
        node
        for node in graph.node
        if node.op_type == "Constant" and node.output[0] in mul.input
    )
    scale.attribute[0].t.CopyFrom(
        numpy_helper.from_array(np.array(0.5, np.float16), scale.attribute[0].t.name)
    )


def _join_on_another_axis(graph: onnx.GraphProto) -> None:
    join = next(n for n in graph.node if n.op_type == "Concat" and len(n.input) == 3)
    join.attribute[0].i = 2


def _other_channel_order(graph: onnx.GraphProto) -> None:
    join = next(n for n in graph.node if n.op_type == "Concat" and len(n.input) == 5)
    join.input[0], join.input[1] = join.input[1], join.input[0]


def _another_activation(graph: onnx.GraphProto) -> None:
    activation = first(graph, "LeakyRelu")
    activation.op_type = "Relu"
    del activation.attribute[:]


def _no_residual(graph: onnx.GraphProto) -> None:
    add = first(graph, "Add")
    add.op_type = "Mul"


def _join_read_elsewhere(graph: onnx.GraphProto) -> None:
    join = next(n for n in graph.node if n.op_type == "Concat" and len(n.input) == 5)
    graph.node.append(onnx.helper.make_node("Identity", [join.output[0]], ["extra"]))


def _another_join_in_the_network(graph: onnx.GraphProto) -> None:
    """Dense blocks inside another architecture, as a TSCUNet has."""
    x = first(graph, "Concat").input[0]
    join = onnx.helper.make_node("Concat", [x, x], ["extra"], axis=1)
    graph.node.append(join)


@pytest.mark.parametrize(
    "edit",
    [
        _other_residual_scale,
        _join_on_another_axis,
        _other_channel_order,
        _another_activation,
        _no_residual,
        _join_read_elsewhere,
        _another_join_in_the_network,
    ],
)
def test_a_file_with_a_block_that_differs_stays_as_it_is(
    plain_conversion: Callable[..., bytes], edit: Callable[[onnx.GraphProto], None]
):
    assert nested_dense_blocks(edited(plain_conversion(tiny()), edit), "fp16") is None


def test_a_dense_network_that_is_not_esrgan_stays(tmp_path: Path):
    """Joins of growing width, as DenseNet's, with no scaled residual."""

    class Dense(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.convs = torch.nn.ModuleList(
                torch.nn.Conv2d(3 + 2 * i, 3 if i == 4 else 2, 3, padding=1)
                for i in range(5)
            )
            self.act = torch.nn.LeakyReLU(0.2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tensors = [self.act(self.convs[0](x)), x]
            tensors.reverse()
            for conv in self.convs[1:4]:
                tensors.append(self.act(conv(torch.cat(tensors, 1))))
            return self.convs[4](torch.cat(tensors, 1))

    path = tmp_path / "dense.onnx"
    torch.onnx.export(
        Dense().eval().half().requires_grad_(False),
        (torch.rand(1, 3, 8, 8).half(),),
        str(path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    data = path.read_bytes()
    assert joins(data) == [2, 3, 4, 5]
    assert nested_dense_blocks(data, "fp16") is None


def _run_with_key(
    monkeypatch: pytest.MonkeyPatch, data: bytes, info: OnnxInfo, precision: Any
) -> tuple[bytes, tuple[object, ...]]:
    """The ONNX Build Engine builds from and the settings of its engine's key."""
    keys: list[tuple[object, ...]] = []

    def key(_model: object, settings: tuple[object, ...], *_: object) -> str:
        keys.append(settings)
        return "key"

    monkeypatch.setattr(build_engine, "engine_key", key)
    routes = _run_node(monkeypatch, data, build_engine.ShapeMode.FIXED, info, precision)
    ((name, args, _),) = routes
    assert name == "generic"
    built = args[0]
    assert isinstance(built, bytes)
    (settings,) = keys
    return built, settings


def test_build_engine_builds_an_earlier_fp16_file_nested_under_its_own_key(
    monkeypatch: pytest.MonkeyPatch, plain_conversion: Callable[..., bytes]
):
    built, settings = _run_with_key(
        monkeypatch,
        plain_conversion(tiny()),
        OnnxInfo(opset=20, dtype="fp16"),
        build_engine.Precision.FP16,
    )
    assert joins(built) == NESTED
    assert "dense blocks nested" in settings


@pytest.mark.parametrize("half", [True, False], ids=["new FP16 conversion", "FP32"])
def test_build_engine_keeps_the_file_and_key_of_any_other_build(
    monkeypatch: pytest.MonkeyPatch, half: bool
):
    data = convert_to_onnx_impl(tiny(), CPU, use_half=half, opset_version=20)
    precision = build_engine.Precision.FP16 if half else build_engine.Precision.FP32
    built, settings = _run_with_key(
        monkeypatch, data, OnnxInfo(opset=20, dtype=precision.value), precision
    )
    assert built == data
    config = build_engine.BuildConfig(
        precision=precision.value,
        workspace_size_gb=0.0,
        min_shape=(256, 256),
        opt_shape=(256, 256),
        max_shape=(256, 256),
        use_dynamic_shapes=False,
        allow_tf32=False,
        optimization_level=3,
        sparse_weights=True,
        tiling_optimization="none",
    )
    assert settings == ("onnx", *dataclasses.astuple(config))
