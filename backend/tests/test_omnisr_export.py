from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, numpy_helper
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.OmniSR import OmniSR

from nodes.impl.pytorch import omnisr_export
from nodes.impl.pytorch.convert_to_onnx_impl import convert_to_onnx_impl

CPU = torch.device("cpu")


def tiny(window_size: int = 8, scale: int = 2) -> ImageModelDescriptor:
    torch.manual_seed(0)
    net = OmniSR(num_feat=16, window_size=window_size, up_scale=scale)
    model = ModelLoader().load_from_state_dict(net.state_dict())
    assert isinstance(model, ImageModelDescriptor)
    return model.eval()


def attentions(model: ImageModelDescriptor) -> list[torch.nn.Module]:
    kinds = (omnisr_export.Channel_Attention, omnisr_export.Channel_Attention_grid)
    return [m for m in model.model.modules() if isinstance(m, kinds)]


def with_dead_channel(model: ImageModelDescriptor) -> ImageModelDescriptor:
    """The model with one channel of every attention's q at exactly zero, as weight decay
    leaves it, so its vector has no length."""
    for attention in attentions(model):
        with torch.no_grad():
            attention.get_submodule("qkv_dwconv").weight[0] = 0
    return model


def image(size: int = 32) -> np.ndarray:
    return np.random.default_rng(0).random((1, 3, size, size), dtype=np.float32)


def run(onnx_bytes: bytes, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: x})[0]


def elem_types(model: onnx.ModelProto) -> dict[str, int]:
    graph = onnx.shape_inference.infer_shapes(model).graph
    return {v.name: v.type.tensor_type.elem_type for v in graph.value_info}


@pytest.mark.parametrize("window_size", [8, 16])
def test_the_lengths_are_fp32_in_the_fp16_export(window_size: int):
    model = onnx.load_from_string(
        convert_to_onnx_impl(tiny(window_size), CPU, use_half=True, opset_version=20)
    )
    types = elem_types(model)
    lengths = [n for n in model.graph.node if n.op_type == "ReduceL2"]
    # Two attentions, q and k of each
    assert len(lengths) == 4
    for node in lengths:
        assert types[node.input[0]] == TensorProto.FLOAT
        assert types[node.output[0]] == TensorProto.FLOAT


def test_the_image_means_are_fp32_in_the_fp16_export():
    model = onnx.load_from_string(
        convert_to_onnx_impl(tiny(), CPU, use_half=True, opset_version=20)
    )
    types = elem_types(model)
    means = [n for n in model.graph.node if n.op_type == "ReduceMean"]
    # The gate's mean, and LayerNorm2d's, which were FP32 already
    assert means
    assert all(types[node.input[0]] == TensorProto.FLOAT for node in means)


def test_the_divisor_never_rounds_to_zero_in_the_fp16_export():
    model = onnx.load_from_string(
        convert_to_onnx_impl(tiny(), CPU, use_half=True, opset_version=20)
    )
    constants = {
        node.output[0]: numpy_helper.to_array(node.attribute[0].t)
        for node in model.graph.node
        if node.op_type == "Constant"
    }
    floors = [
        constants[node.input[1]]
        for node in model.graph.node
        if node.op_type == "Clip" and node.input[1] in constants and node.input[2] == ""
    ]
    assert len(floors) == 4
    assert all(float(floor) > 0 for floor in floors)


def test_a_vector_too_small_to_square_in_fp16_keeps_unit_length():
    small = torch.full((1, 64), 1e-4).half()
    assert small.pow(2).sum() == 0
    unit = omnisr_export.unit_length(small)
    assert torch.linalg.vector_norm(unit.float()).item() == pytest.approx(1, abs=0.01)


def test_an_all_zero_vector_stays_zero():
    unit = omnisr_export.unit_length(torch.zeros(1, 64).half())
    assert torch.equal(unit, torch.zeros(1, 64).half())


def test_a_dead_channel_stays_finite_in_the_fp16_export():
    model = with_dead_channel(tiny())
    half = convert_to_onnx_impl(model, CPU, use_half=True, opset_version=20)
    output = run(half, image())
    assert np.isfinite(output).all()
    reference = run(convert_to_onnx_impl(model, CPU, opset_version=20), image())
    assert np.abs(output.astype(np.float32) - reference).max() < 0.02


@pytest.mark.parametrize("window_size", [8, 16])
@pytest.mark.parametrize("size", [32, 48, 57])
def test_the_fp16_export_stays_close_to_fp32_at_other_sizes(
    window_size: int, size: int
):
    model = tiny(window_size)
    half = convert_to_onnx_impl(model, CPU, use_half=True, opset_version=20)
    plain = convert_to_onnx_impl(model, CPU, opset_version=20)
    output = run(half, image(size))
    reference = run(plain, image(size))
    assert output.shape == reference.shape
    assert np.abs(output.astype(np.float32) - reference).max() < 0.02


def test_the_rewrite_works_on_a_copy():
    model = tiny().half()
    weights = {k: v.clone() for k, v in model.model.state_dict().items()}
    copied = omnisr_export.for_export(model)
    assert copied.model is not model.model
    assert not any(
        isinstance(
            m, (omnisr_export.ChannelAttention, omnisr_export.ChannelAttentionGrid)
        )
        for m in model.model.modules()
    )
    state = copied.model.state_dict()
    assert state.keys() == weights.keys()
    for key, value in weights.items():
        assert torch.equal(state[key], value)
        assert torch.equal(model.model.state_dict()[key], value)


def test_only_fp16_omnisr_is_rewritten():
    model = tiny()
    assert model.dtype == torch.float32
    assert omnisr_export.for_export(model) is model


def test_the_fp32_export_keeps_its_arithmetic():
    model = onnx.load_from_string(convert_to_onnx_impl(tiny(), CPU, opset_version=20))
    assert not any(
        node.op_type == "Cast" and node.attribute[0].i == TensorProto.FLOAT16
        for node in model.graph.node
    )
