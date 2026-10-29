"""VOSR's MIGraphX route (AMD GPUs, Linux) on the CPU: the graph change from the TensorRT
plugin to the custom op, the programs' fixed shapes, the op library's shape contract (when
it was built), the node's routing and checks, the session's stage dispatch and the program
cache. The GPU parts: test_vosr2_migraphx_gpu.py."""

from __future__ import annotations

import ctypes
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch

from api import NodeContext
from nodes.impl.migraphx import vosr2_migx, vosr2_migx_compile
from nodes.impl.migraphx.vosr2_migx import MigraphxSession, ProgramStages
from nodes.impl.migraphx.vosr2_migx_compile import (
    OP,
    PROGRAMS,
    fp8_products,
    migraphx_graph,
    program_inputs,
    program_name,
)
from nodes.impl.pytorch.vosr2 import (
    BundlePaths,
    VosrBackend,
    VosrBundle,
    VosrColorCorrection,
    VosrMemoryMode,
)
from nodes.impl.pytorch.vosr2_plan_file import plain
from nodes.impl.pytorch.vosr2_temporal import TemporalMode
from nodes.impl.pytorch.vosr2_trt_execution import StageExecution
from nodes.utils.seed import Seed

if TYPE_CHECKING:
    import onnx


def plugin_graph(path: Path) -> None:
    """A graph with the TensorRT plugin node as the export's rewrite places it, its
    normalization weights as external data, and its outputs consumed."""
    import onnx
    from onnx import TensorProto, helper

    def value(name: str, shape: list[int]) -> onnx.ValueInfoProto:
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    weights = [
        helper.make_tensor(name, TensorProto.FLOAT, [64], np.full(64, 1.5))
        for name in ("q_norm", "k_norm")
    ]
    nodes = [
        helper.make_node(
            "VosrQkvPack",
            ["x", "q_norm", "k_norm", "cos", "sin"],
            ["q", "k", "v"],
            name="/attn/NativePack",
            domain="trt.plugins",
            mode=0,
            plugin_version="1",
            plugin_namespace="",
        ),
        *(
            helper.make_node("Identity", [name], [f"{name}_out"], name=f"use_{name}")
            for name in ("q", "k", "v")
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "plugin",
        [value("x", [1, 5, 4608]), value("cos", [5, 64]), value("sin", [5, 64])],
        [value(f"{name}_out", [1, 24, 5, 64]) for name in ("q", "k", "v")],
        initializer=weights,
    )
    # IR 10, as ONNX Runtime of the test runtimes reads
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("trt.plugins", 1),
        ],
    )
    onnx.save(
        model,
        path,
        save_as_external_data=True,
        location=f"{path.name}.data",
        size_threshold=0,
    )


def test_the_plugin_node_becomes_the_custom_op_and_its_split(tmp_path):
    import onnx
    import onnxruntime
    from onnx import TensorProto, helper

    source = tmp_path / "dit_cached.onnx"
    plugin_graph(source)
    target = migraphx_graph(source)
    assert target == tmp_path / "dit_cached.bf16.migraphx.onnx"
    # A build's folder keeps what it converted
    assert migraphx_graph(source) == target
    # The external data stays where it was: the new graph reads the original's file.
    model = onnx.load(target)
    assert {(opset.domain, opset.version) for opset in model.opset_import} == {
        ("", 21),
        ("chainner", 1),
    }
    (op,) = [node for node in model.graph.node if node.domain]
    assert (op.domain, op.op_type) == ("chainner", OP)
    assert list(op.input) == ["x", "q_norm", "k_norm", "cos", "sin"]
    assert len(op.output) == 1 and not op.attribute
    # Q, K and V are the packed output's slices along its first axis, in that order:
    # with the op replaced by the packing of three inputs, they come back unchanged.
    packing = [
        helper.make_tensor("pack_axes", TensorProto.INT64, [1], [0]),
    ]
    stand_in = [
        *(
            helper.make_node("Unsqueeze", [name, "pack_axes"], [f"{name}_packed"])
            for name in ("a", "b", "c")
        ),
        helper.make_node(
            "Concat", ["a_packed", "b_packed", "c_packed"], list(op.output), axis=0
        ),
    ]
    nodes = [node for node in model.graph.node if node is not op]
    del model.graph.node[:]
    model.graph.node.extend(stand_in + nodes)
    model.graph.initializer.extend(packing)
    del model.graph.input[:]
    model.graph.input.extend(
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 24, 5, 64])
        for name in ("a", "b", "c")
    )
    del model.opset_import[:]
    model.opset_import.append(helper.make_opsetid("", 21))
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    inputs = {
        name: np.random.default_rng(index).standard_normal((1, 24, 5, 64), np.float32)
        for index, name in enumerate(("a", "b", "c"))
    }
    outputs = session.run(["q_out", "k_out", "v_out"], inputs)
    for actual, expected in zip(outputs, inputs.values(), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_a_graph_without_the_plugin_is_compiled_as_it_is(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    path = tmp_path / "dino.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["result"])],
        "dino",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 448, 448])],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, [1, 3, 448, 448])],
    )
    onnx.save(helper.make_model(graph), path)
    assert migraphx_graph(path) == path
    assert not list(tmp_path.glob("*.migraphx.onnx"))


def test_hidden_enum_variants_are_neither_options_nor_types():
    from enum import Enum

    from nodes.properties.inputs import EnumInput

    class Choice(Enum):
        A = "a"
        B = "b"
        C = "c"

    class Short(Enum):
        A = "a"
        C = "c"

    shown = EnumInput(Choice, "Choice", hidden=(Choice.B,), type_name="Choice")
    plain = EnumInput(Short, "Choice", type_name="Choice")
    assert shown.to_dict() == plain.to_dict()
    assert shown.type_definitions == plain.type_definitions
    assert shown.accepted_values == {"a", "c"}
    assert shown.enforce("c") is Choice.C


def test_the_node_offers_migraphx_only_on_linux_or_windows_amd_without_nvidia():
    from packages.chaiNNer_pytorch.pytorch.processing import upscale_image_vosr as node

    (schema,) = [
        data
        for data in node.processing_group.nodes
        if data.schema_id == "chainner:pytorch:upscale_image_vosr"
    ]
    (backend,) = [i for i in schema.inputs if i.label == "Backend"]
    values = [option["value"] for option in backend.to_dict()["options"]]
    if node.OFFERS_MIGRAPHX:
        assert values == [member.value for member in VosrBackend]
    else:
        # The node NVIDIA and CPU systems had: three backends, TensorRT's docs only
        assert values == ["pytorch", "tensorrt", "tensorrt_fp8"]
        assert "MIGraphX" not in str(backend.description)
        assert "migraphx" not in str(backend.type_definitions).lower()


def products_graph(stage: str) -> onnx.ModelProto:
    """A stage's products as the export has them: a BF16 weight product, an attention
    product (no weight) after a BF16 cast, a last BF16 weight product (the DiT's output
    projection) and an FP32 weight product and convolution (DINO's islands)."""
    from onnx import TensorProto, helper

    def weight(name: str, kind: int, shape: list[int]) -> onnx.TensorProto:
        values = np.ones(int(np.prod(shape)), np.float32)
        if kind == TensorProto.BFLOAT16:
            raw = (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()
            return helper.make_tensor(name, kind, shape, raw, raw=True)
        return helper.make_tensor(name, kind, shape, values)

    bf16, fp32 = TensorProto.BFLOAT16, TensorProto.FLOAT
    nodes = [
        helper.make_node("Conv", ["image", "patch"], ["patches"], name="/patch/Conv"),
        helper.make_node("Cast", ["x"], ["xb"], to=bf16, name="cast_x"),
        helper.make_node("MatMul", ["xb", "w1"], ["h1"], name="/blocks.0/fc/MatMul"),
        helper.make_node("MatMul", ["h1", "h1t"], ["scores"], name="/attn/MatMul"),
        helper.make_node("Transpose", ["h1"], ["h1t"], name="/attn/Transpose"),
        helper.make_node("Softmax", ["scores"], ["p"], name="/attn/Softmax"),
        helper.make_node("MatMul", ["x", "w3"], ["h3"], name="/blocks.0/fp32/MatMul"),
        helper.make_node("MatMul", ["p", "w2"], ["h2"], name="/linear/MatMul"),
    ]
    # Transpose before its user, as an exporter orders nodes
    nodes[3], nodes[4] = nodes[4], nodes[3]
    graph = helper.make_graph(
        nodes,
        stage,
        [
            helper.make_tensor_value_info("x", fp32, [4, 4]),
            helper.make_tensor_value_info("image", fp32, [1, 3, 4, 4]),
        ],
        [
            helper.make_tensor_value_info(name, kind, None)
            for name, kind in (("h2", bf16), ("h3", fp32), ("patches", fp32))
        ],
        initializer=[
            weight("w1", bf16, [4, 4]),
            weight("w2", bf16, [4, 4]),
            weight("w3", fp32, [4, 4]),
            weight("patch", fp32, [2, 3, 2, 2]),
        ],
    )
    return helper.make_model(
        graph, ir_version=10, opset_imports=[helper.make_opsetid("", 21)]
    )


def float_products(model: onnx.ModelProto) -> set[str]:
    """The products and convolutions whose inputs are FP32, which MIGraphX's
    quantize_fp8 quantizes."""
    from onnx import TensorProto

    types = {tensor.name: tensor.data_type for tensor in model.graph.initializer}
    types.update(
        (value.name, value.type.tensor_type.elem_type) for value in model.graph.input
    )
    for node in model.graph.node:
        if node.op_type == "Cast":
            types[node.output[0]] = node.attribute[0].i
    return {
        node.name
        for node in model.graph.node
        if node.op_type in ("MatMul", "Conv")
        and types.get(node.input[1], types.get(node.input[0])) == TensorProto.FLOAT
    }


@pytest.mark.parametrize(
    "stage,expected",
    [
        # Weight products; the cached DiT's last is its output projection
        ("dit_kv", {"/blocks.0/fc/MatMul", "/linear/MatMul", "/blocks.0/fp32/MatMul"}),
        ("dit_cached", {"/blocks.0/fc/MatMul", "/blocks.0/fp32/MatMul"}),
        # Every product, attention too; the FP32 convolution leaves FP32
        (
            "dino",
            {
                "/blocks.0/fc/MatMul",
                "/attn/MatMul",
                "/linear/MatMul",
                "/blocks.0/fp32/MatMul",
            },
        ),
    ],
)
def test_fp8_products_are_the_tensorrt_recipes(stage, expected):
    model = products_graph(stage)
    count = fp8_products(model, stage)
    assert float_products(model) == expected
    # FP32 products are counted, not changed
    assert count == len(expected)
    if stage == "dino":
        (conv,) = [node for node in model.graph.node if node.op_type == "Conv"]
        assert conv.output[0] == "patches/computed16"
    # Every changed product's result is cast back to its own type under its own name
    producers = {name: node for node in model.graph.node for name in node.output}
    for name, kind in (("h1", 16), ("h2", 16), ("p", None)):
        node = producers[name]
        if kind is not None and node.op_type == "Cast":
            assert node.attribute[0].i == kind


def test_fp16_types_keep_fp32_and_the_op_bf16(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    bf16, fp16, fp32 = TensorProto.BFLOAT16, TensorProto.FLOAT16, TensorProto.FLOAT
    raw = (np.array([1.5, -2.0, 3.25, 1e-3], np.float32).view(np.uint32) >> 16).astype(
        np.uint16
    )
    nodes = [
        helper.make_node("Cast", ["x"], ["xb"], to=bf16, name="cast"),
        helper.make_node(
            "Constant",
            [],
            ["c"],
            value=helper.make_tensor("c", bf16, [4], raw.tobytes(), raw=True),
            name="constant",
        ),
        helper.make_node("Add", ["xb", "c"], ["s"], name="add"),
        helper.make_node("MatMul", ["s", "w"], ["m"], name="matmul"),
        helper.make_node(OP, ["m"], ["packed"], name="qkv", domain="chainner"),
        helper.make_node("Cast", ["x"], ["kept"], to=fp32, name="island"),
    ]
    graph = helper.make_graph(
        nodes,
        "dit_cached",
        [helper.make_tensor_value_info("x", fp32, [1, 4])],
        [
            helper.make_tensor_value_info("packed", bf16, None),
            helper.make_tensor_value_info("kept", fp32, [1, 4]),
        ],
        initializer=[
            helper.make_tensor("w", bf16, [4, 4], np.repeat(raw, 4).tobytes(), raw=True)
        ],
    )
    source = tmp_path / "dit_cached.onnx"
    onnx.save(
        helper.make_model(graph, ir_version=10),
        source,
        save_as_external_data=True,
        location="dit_cached.onnx.data",
        size_threshold=0,
    )
    target = migraphx_graph(source, "fp16")
    assert target.name == "dit_cached.fp16.migraphx.onnx"
    model = onnx.load(target)
    (weight,) = model.graph.initializer
    assert weight.data_type == fp16
    np.testing.assert_array_equal(
        numpy_helper.to_array(weight)[0], np.full(4, 1.5, np.float16)
    )
    nodes = {node.name: node for node in model.graph.node}
    assert nodes["cast"].attribute[0].i == fp16
    assert nodes["island"].attribute[0].i == fp32
    assert nodes["constant"].attribute[0].t.data_type == fp16
    # The op computes in BF16 between casts
    op = nodes["qkv"]
    assert [nodes[name].attribute[0].i for name in op.input] == [bf16]
    assert op.output[0] == "packed/bf16" and nodes["packed"].attribute[0].i == fp16
    types = {
        value.name: value.type.tensor_type.elem_type for value in model.graph.output
    }
    assert types == {"packed": fp16, "kept": fp32}


def test_programs_are_the_stages_at_the_shapes_restoration_uses():
    assert program_inputs("dino") == {"x": (1, 3, 448, 448)}
    assert program_inputs("dit_kv") == {"conditioning": (1, 1024, 1024)}
    for width, tokens in ((64, 1024), (72, 1152)):
        name = program_name("dit_cached", width)
        assert name == f"dit_cached_{width}"
        assert program_inputs(name) == {
            "latent": (1, 32, 64, width),
            "kv": (72, 1, 1024, 1536),
            "cos": (tokens, 64),
            "sin": (tokens, 64),
        }
    assert program_name("dino", 448) == "dino"
    with pytest.raises(ValueError, match="64 or 72 latent columns"):
        program_name("dit_cached", 80)


class Shape(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int32),
        ("rank", ctypes.c_int32),
        ("dims", ctypes.c_int64 * 8),
    ]


ComputeShape = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(Shape),
    ctypes.c_int32,
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(Shape),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_char_p,
    ctypes.c_size_t,
)


class Op(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("flags", ctypes.c_uint32),
        ("compute_shape", ComputeShape),
        ("compute", ctypes.c_void_p),
    ]


BF16, FLOAT = 2, 0


def shape(dtype: int, *dims: int) -> Shape:
    value = Shape(dtype, len(dims))
    for i, dim in enumerate(dims):
        value.dims[i] = dim
    return value


@pytest.mark.skipif(sys.platform != "linux", reason="MIGraphX runs on Linux only")
def test_the_op_library_registers_the_op_with_the_plugins_shapes():
    path = Path(vosr2_migx_compile.__file__).with_name("vosr2_migx_ops.so")
    if not path.is_file():
        pytest.skip("this build has no vosr2_migx_ops.so")
    try:
        library = ctypes.CDLL(str(path))
    except OSError as error:
        # A runtime without ROCm's wheels (the CUDA one) has no HIP for it to link.
        if "libamdhip64" not in str(error):
            raise
        pytest.skip(f"no HIP runtime: {error}")
    table, count = ctypes.POINTER(Op)(), ctypes.c_int32()
    assert library.chainner_migx_ops(2, ctypes.byref(table), ctypes.byref(count)) != 0
    assert library.chainner_migx_ops(1, ctypes.byref(table), ctypes.byref(count)) == 0
    assert count.value == 1
    op = table[0]
    assert op.name == OP.encode()
    assert op.flags == 2  # CHAINNER_MIGX_FIXED_TYPES: BF16 under any precision

    def compute(*inputs: Shape, attributes: int = 0) -> tuple[int, Shape, bytes]:
        array = (Shape * len(inputs))(*inputs)
        output, workspace = Shape(), ctypes.c_size_t(0)
        error = ctypes.create_string_buffer(256)
        status = op.compute_shape(
            array,
            len(inputs),
            None,
            attributes,
            ctypes.byref(output),
            ctypes.byref(workspace),
            error,
            256,
        )
        assert workspace.value == 0
        return status, output, error.value

    def inputs(tokens: int, dtype: int = BF16) -> list[Shape]:
        return [
            shape(dtype, 1, tokens, 4608),
            shape(BF16, 64),
            shape(BF16, 64),
            shape(BF16, tokens, 64),
            shape(BF16, tokens, 64),
        ]

    for tokens in (1, 1024, 1152):
        status, output, _ = compute(*inputs(tokens))
        assert status == 0
        assert (output.type, output.rank) == (BF16, 5)
        assert list(output.dims[:5]) == [3, 1, 24, tokens, 64]
    for arguments, attributes, message in (
        (inputs(1153), 0, b"x must be BF16 (1, 1 to 1152, 4608)"),
        (inputs(8, FLOAT), 0, b"x must be BF16"),
        (inputs(8)[:4], 0, b"expects 5 inputs and no attributes"),
        (inputs(8), 1, b"expects 5 inputs and no attributes"),
        ([*inputs(8)[:3], shape(BF16, 7, 64), inputs(8)[4]], 0, b"rotary tables"),
    ):
        status, _, error = compute(*arguments, attributes=attributes)
        assert status != 0
        assert error.startswith(b"chainner_vosr2_qkv: ") and message in error


def use_device(monkeypatch, node, device, hip):
    monkeypatch.setattr(
        node,
        "get_settings",
        lambda _: SimpleNamespace(
            device=torch.device(device), budget_limit=0, force_cache_wipe=False
        ),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.version, "hip", hip)


def run_node(node, memory, backend, node_id="migraphx"):
    return node.upscale_image_vosr_node(
        cast(NodeContext, SimpleNamespace(node_id=node_id, iteration=None)),
        np.zeros((16, 16, 3), np.float32),
        4,
        Seed(0),
        VosrColorCorrection.WAVELET,
        memory,
        backend,
        TemporalMode.THREE_FRAMES,
        65,
        texture=True,
    )


def amd_gpu(gfx: str = "gfx1100") -> SimpleNamespace:
    fp8 = {"gfx942": "fnuz", "gfx950": "ocp", "gfx1201": "ocp"}.get(gfx)
    return SimpleNamespace(
        index=0,
        name="Radeon Test",
        gfx=gfx,
        fp8_format=fp8,
        supports_fp8=fp8 is not None,
    )


@pytest.mark.parametrize(
    "memory,backend,expected",
    [
        (VosrMemoryMode.DEFAULT, VosrBackend.MIGRAPHX, "bf16"),
        (VosrMemoryMode.DEFAULT, VosrBackend.MIGRAPHX_FP8, "reduced"),
        # MIGraphX cannot stream weights: Low runs on PyTorch
        (VosrMemoryMode.LOW, VosrBackend.MIGRAPHX, "pytorch"),
        (VosrMemoryMode.LOW, VosrBackend.MIGRAPHX_FP8, "pytorch"),
        (VosrMemoryMode.DEFAULT, VosrBackend.PYTORCH, "pytorch"),
    ],
)
def test_node_routes_migraphx_on_an_amd_gpu(
    monkeypatch, tmp_path, memory, backend, expected
):
    from packages.chaiNNer_pytorch.pytorch.processing import upscale_image_vosr as node

    class SelectedEngine(Exception):
        pass

    def select(name):
        def selected(*args, **kwargs):
            raise SelectedEngine(name)

        return selected

    use_device(monkeypatch, node, "cuda", "7.14.60850")
    monkeypatch.setattr(vosr2_migx, "check_usable", lambda *args, **kwargs: amd_gpu())
    monkeypatch.setattr(
        vosr2_migx,
        "ensure_migraphx_engine",
        lambda *args, reduced: select("reduced" if reduced else "bf16")(),
    )
    monkeypatch.setattr(node, "release_full_session", select("pytorch"))
    monkeypatch.setattr(node, "ensure_model", lambda _: BundlePaths(tmp_path))
    monkeypatch.setattr(node, "ensure_face_model", lambda _: tmp_path / "faces.onnx")
    with pytest.raises(SelectedEngine, match=expected):
        run_node(node, memory, backend)
    assert node._session(backend, memory) == (expected != "pytorch")


@pytest.mark.parametrize("backend", [VosrBackend.TENSORRT, VosrBackend.TENSORRT_FP8])
def test_tensorrt_on_an_amd_gpu_fails_before_any_work(monkeypatch, backend):
    from packages.chaiNNer_pytorch.pytorch.processing import upscale_image_vosr as node

    monkeypatch.setattr(node, "ensure_model", lambda _: pytest.fail("started work"))
    use_device(monkeypatch, node, "cuda", "7.14.60850")
    with pytest.raises(ValueError, match="Select the MIGraphX or the PyTorch backend"):
        run_node(node, VosrMemoryMode.DEFAULT, backend)


@pytest.mark.parametrize("backend", [VosrBackend.MIGRAPHX, VosrBackend.MIGRAPHX_FP8])
def test_unusable_migraphx_fails_before_any_work(monkeypatch, backend):
    from packages.chaiNNer_pytorch.pytorch.processing import upscale_image_vosr as node

    monkeypatch.setattr(node, "ensure_model", lambda _: pytest.fail("started work"))
    # An NVIDIA (or CPU) PyTorch, the CPU on a ROCm PyTorch
    for device, hip in (("cuda", None), ("cpu", None), ("cpu", "7.14.60850")):
        use_device(monkeypatch, node, device, hip)
        with pytest.raises(ValueError, match="requires an AMD GPU and the ROCm build"):
            run_node(node, VosrMemoryMode.DEFAULT, backend)
    use_device(monkeypatch, node, "cuda", "7.14.60850")
    monkeypatch.setattr(vosr2_migx.sys, "platform", "darwin")
    with pytest.raises(ValueError, match="runs on Linux and Windows only"):
        run_node(node, VosrMemoryMode.DEFAULT, backend)
    monkeypatch.setattr(vosr2_migx.sys, "platform", "linux")
    monkeypatch.setattr(vosr2_migx, "runner_built", lambda: False)
    with pytest.raises(ValueError, match="has no MIGraphX runner"):
        run_node(node, VosrMemoryMode.DEFAULT, backend)
    monkeypatch.setattr(vosr2_migx, "runner_built", lambda: True)
    monkeypatch.setattr(vosr2_migx, "amd", SimpleNamespace(devices=[]))
    with pytest.raises(ValueError, match="no AMD GPU with HIP index 0"):
        run_node(node, VosrMemoryMode.DEFAULT, backend)
    monkeypatch.setattr(vosr2_migx, "amd", SimpleNamespace(devices=[amd_gpu()]))
    assert vosr2_migx.check_usable(torch.device("cuda")).gfx == "gfx1100"


class FakeProgram:
    def __init__(self, inputs: dict, types: dict, output: tuple, output_type: str):
        self.input_names = list(inputs)
        self.input_shapes = [list(value) for value in inputs.values()]
        self.input_types = [types[name] for name in inputs]
        self.output_shapes = [list(output)]
        self.output_types = [output_type]
        self.calls: list[tuple[dict, list, int]] = []
        self.closed = False

    def run(self, inputs: dict, outputs: list, stream: int) -> None:
        self.calls.append((inputs, outputs, stream))

    def close(self) -> None:
        self.closed = True


def test_programs_bind_matching_inputs_and_copy_the_others(monkeypatch):
    device = torch.device("cpu")
    stream = cast(torch.cuda.Stream, SimpleNamespace(cuda_stream=7))
    stages = ProgramStages.__new__(ProgramStages)
    stages.device = device
    stages.execution = StageExecution(device, stream)
    programs = {}
    for name in PROGRAMS:
        inputs = program_inputs(name)
        programs[name] = FakeProgram(
            # MIGraphX may list a program's inputs in another order
            dict(reversed(inputs.items())),
            dict.fromkeys(inputs, "bfloat16"),
            (1, 16, 64, 64) if name.startswith("dit_cached") else (1, 1024, 1024),
            "bfloat16",
        )
    stages.programs = programs
    recorded = []
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda tensor, s: recorded.append(tensor)
    )
    latent = torch.ones(1, 32, 64, 72)  # float32: copied into a BF16 buffer
    kv = torch.zeros(72, 1, 1024, 1536, dtype=torch.bfloat16)  # bound as it is
    cos = torch.zeros(1152, 64, dtype=torch.bfloat16)
    sin = torch.zeros(1152, 64, dtype=torch.bfloat16).t().contiguous().t()
    result = stages.run("dit_cached", (latent, kv, cos, sin), stream)
    program = programs["dit_cached_72"]
    ((pointers, outputs, used),) = program.calls
    assert used == 7 and outputs == [result.data_ptr()]
    assert result.shape == (1, 16, 64, 64) and result.dtype == torch.bfloat16
    assert pointers["kv"] == kv.data_ptr() and pointers["cos"] == cos.data_ptr()
    copied = stages.execution.buffers[
        (("dit_cached_72", "latent"), latent.shape, torch.bfloat16)
    ]
    assert pointers["latent"] == copied.data_ptr() and bool((copied == 1).all())
    # A non-contiguous view is copied too
    assert pointers["sin"] != sin.data_ptr()
    assert [tensor.data_ptr() for tensor in recorded] == [kv.data_ptr(), cos.data_ptr()]
    assert not programs["dit_cached_64"].calls
    with pytest.raises(ValueError, match="takes conditioning"):
        stages.run("dit_kv", (torch.zeros(1, 1000, 1024),), stream)
    stages.close()
    assert all(program.closed for program in programs.values()) and not stages.programs


def test_the_session_runs_vae_stages_in_pytorch_and_the_rest_as_programs(monkeypatch):
    session = MigraphxSession.__new__(MigraphxSession)
    session.device = torch.device("cpu")
    session.stream = cast(torch.cuda.Stream, object())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: session.stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    calls = []

    class Module(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            calls.append(("module", x.dtype))
            return x * 2

    class Programs:
        def run(self, name, arguments, stream):
            calls.append(("program", name, stream is session.stream))
            return arguments[0]

        def close(self):
            calls.append("closed")

    modules: dict[str, torch.nn.Module] = {"encoder_pre": Module()}
    session.modules = modules
    session.programs = cast(ProgramStages, None)
    with pytest.raises(RuntimeError, match="not activated"):
        session.stage("dino", torch.zeros(1))
    session.programs = cast(ProgramStages, Programs())
    image = torch.ones(1, 3, 4, 4)
    assert session.stage("encoder_pre", image).dtype == torch.bfloat16
    session.stage("dino", image)
    assert calls == [("module", torch.bfloat16), ("program", "dino", True)]
    monkeypatch.setattr(StageExecution, "clear", lambda self: None)
    session.execution = StageExecution(session.device, session.stream)
    session.constants, session.ropes = {"noise": image}, {64: (image, image)}
    session.kv_tensor, session.geometry, session.budget = image, (512, 512, 0), 1
    session.offload()
    assert calls[-1] == "closed" and session.programs is None
    assert not session.constants and not session.ropes and session.kv_tensor is None
    with pytest.raises(ValueError, match="GPU Memory Low runs on PyTorch"):
        session.activate(lambda: None, reserve=1)


@pytest.mark.parametrize(
    "reduced,gfx,precision",
    [
        (False, "gfx1100", "bf16"),
        (False, "gfx942", "bf16"),
        # MIGraphX FP8: FP8 in the GPU's format where it has FP8 math, else BF16 where
        # it has BF16 math, FP16 on RDNA1/2 and CDNA1
        (True, "gfx942", "fp8"),
        (True, "gfx1201", "fp8"),
        (True, "gfx1100", "bf16"),
        (True, "gfx90a", "bf16"),
        (True, "gfx1030", "fp16"),
        (True, "gfx1010", "fp16"),
        (True, "gfx908", "fp16"),
    ],
)
def test_a_program_set_is_published_once_and_reused(
    monkeypatch, tmp_path, reduced, gfx, precision
):
    builds = []
    library = tmp_path / "vosr2_migx_ops.so"
    library.write_bytes(b"op library")
    monkeypatch.setattr(vosr2_migx, "read_configs", lambda paths: None)
    monkeypatch.setattr(vosr2_migx, "check_usable", lambda *args: amd_gpu(gfx))
    monkeypatch.setattr(
        vosr2_migx, "load_ops", lambda: SimpleNamespace(MIGRAPHX_VERSION="2.16.1")
    )
    monkeypatch.setattr(vosr2_migx, "ops_path", lambda: library)

    def compile_worker(
        context, worker, source, directory, device, *, label, limit, arguments
    ):
        assert worker == Path(vosr2_migx.__file__).with_name("vosr2_migx_build.py")
        assert (label, source, arguments) == (
            "MIGraphX",
            tmp_path / "model",
            (precision,),
        )
        engines = {}
        for name in PROGRAMS:
            plain(directory, name).write_bytes(name.encode())
            engines[name] = vosr2_migx.file_hash(plain(directory, name))
        (directory / "validation.json").write_text(
            json.dumps({"engines": engines, "profiles": []}), encoding="utf-8"
        )
        builds.append(directory)

    monkeypatch.setattr(vosr2_migx, "compile_worker", compile_worker)
    context = cast(
        NodeContext,
        SimpleNamespace(
            node_id="cache", storage_dir=tmp_path / "storage", suspend=lambda: None
        ),
    )
    paths = BundlePaths(tmp_path / "model")
    device = torch.device("cuda")

    def ensure() -> Path:
        return vosr2_migx.ensure_migraphx_engine(
            context, paths, device, reduced=reduced
        )

    first = ensure()
    assert len(builds) == 1 and first.parent.name == f"vosr2-migraphx-{precision}"
    # Plain programs (MIGraphX loads a file), described by the manifest
    for name in PROGRAMS:
        assert plain(first, name).read_bytes() == name.encode()
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert "compression" not in manifest and set(manifest["engines"]) == set(PROGRAMS)
    identity = manifest["identity"]
    assert (identity["precision"], identity["gfx"]) == (precision, gfx)
    assert identity["programs"]["dit_cached_72"]["cos"] == [1152, 64]
    assert ensure() == first and len(builds) == 1
    # Another MIGraphX makes another set
    monkeypatch.setattr(
        vosr2_migx, "load_ops", lambda: SimpleNamespace(MIGRAPHX_VERSION="2.16.2")
    )
    assert ensure() != first and len(builds) == 2


def test_calibration_images_are_fixed_and_varied():
    from nodes.impl.migraphx.vosr2_migx_build import calibration_images

    images = calibration_images()
    again = calibration_images()
    assert len(images) == 4
    for image, same in zip(images, again, strict=True):
        assert image.shape == (512, 576, 3) and image.dtype == np.float32
        assert 0 <= image.min() and image.max() <= 1
        np.testing.assert_array_equal(image, same)
    # Smooth blobs, sharp shapes, fine texture
    steps = [np.abs(np.diff(image, axis=1)) for image in images]
    assert steps[0].max() < 0.05 and steps[1].max() > 0.5
    assert steps[2].mean() == max(step.mean() for step in steps)


def test_calibration_inputs_are_what_restoration_feeds_each_program(monkeypatch):
    from nodes.impl.migraphx import vosr2_migx_build

    encoded = []

    class Vae(torch.nn.Module):
        def encode(self, image: torch.Tensor) -> SimpleNamespace:
            assert image.shape == (1, 3, 512, 576) and image.min() >= -1
            encoded.append(image)
            latent = torch.full((1, 16, 64, 72), 2.0)
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    class Dino(torch.nn.Module):
        def forward_intermediate_layer(self, x: torch.Tensor, layer: int):
            assert x.shape == (1, 3, 448, 448) and x.dtype == torch.float32
            assert layer == 17
            return torch.ones(1, 1024, 1024)

    class Projection(torch.nn.Module):
        def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
            assert conditioning.dtype == torch.bfloat16
            return conditioning[0, :2, :3] * 3

    monkeypatch.setattr(
        "nodes.impl.pytorch.vosr2_trt_models.export_model",
        lambda name, bundle: Projection(),
    )
    bundle = SimpleNamespace(
        vae=Vae(),
        dino=Dino(),
        mean=torch.ones(1, 16, 1, 1),
        std=torch.ones(1, 16, 1, 1),
    )
    inputs = vosr2_migx_build.calibration_inputs(
        cast(VosrBundle, bundle), torch.device("cpu")
    )
    assert len(encoded) == 4
    assert {name: len(values) for name, values in inputs.items()} == dict.fromkeys(
        PROGRAMS, 4
    )
    sample = {name: values[0] for name, values in inputs.items()}
    assert sample["dino"]["x"].dtype == np.float32
    assert str(sample["dit_kv"]["conditioning"].dtype) == "bfloat16"
    wide, narrow = sample["dit_cached_72"], sample["dit_cached_64"]
    for name in PROGRAMS:
        for key, value in sample[name].items():
            # K/V: the stand-in projection's
            if key != "kv":
                assert value.shape == program_inputs(name)[key]
    assert str(wide["latent"].dtype) == "bfloat16"
    # The latent window: normalized latent (2 - 1) / 1, then the seeded noise
    assert (wide["latent"][0, :16].astype(np.float32) == 1).all()
    np.testing.assert_array_equal(narrow["latent"], wide["latent"][..., :64])
    np.testing.assert_array_equal(wide["kv"].astype(np.float32), 3)
