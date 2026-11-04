from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.ESRGAN import ESRGAN

from nodes.impl.pytorch import esrgan_export
from nodes.impl.pytorch.convert_to_onnx_impl import convert_to_onnx_impl

CPU = torch.device("cpu")


def tiny(**kwargs) -> ImageModelDescriptor:
    torch.manual_seed(0)
    settings = {"num_filters": 8, "num_blocks": 2, "scale": 4, **kwargs}
    model = ModelLoader().load_from_state_dict(ESRGAN(**settings).state_dict())
    assert isinstance(model, ImageModelDescriptor)
    return model.eval()


def concat_inputs(onnx_bytes: bytes) -> list[int]:
    graph = onnx.load_from_string(onnx_bytes).graph
    return [len(node.input) for node in graph.node if node.op_type == "Concat"]


def image(channels: int = 3, size: int = 32) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((1, channels, size, size), dtype=np.float32)


@pytest.mark.parametrize(
    "settings",
    [
        {"scale": 1},
        {"scale": 2},
        {"scale": 4},
        {"scale": 8},
        {"scale": 2, "shuffle_factor": 2},
        {"scale": 1, "shuffle_factor": 4},
        {"scale": 4, "num_blocks": 1},
        {"scale": 4, "in_nc": 1, "out_nc": 1},
    ],
    ids=["1x", "2x", "4x", "8x", "2x-unshuffle", "1x-unshuffle", "1-block", "grey"],
)
def test_the_nested_blocks_compute_the_same_values(settings: dict[str, object]):
    model = tiny(**settings).half()
    nested = esrgan_export.for_export(model)
    x = torch.from_numpy(image(model.input_channels)).half()
    with torch.no_grad():
        assert torch.equal(nested.model(x), model.model(x))


def test_the_nested_blocks_keep_the_plus_paths():
    model = tiny(plus=True).half()
    nested = esrgan_export.for_export(model)
    blocks = [
        m
        for m in nested.model.modules()
        if isinstance(m, esrgan_export.ResidualDenseBlock_5C)
    ]
    assert blocks
    assert all(block.conv1x1 is not None for block in blocks)
    x = torch.from_numpy(image()).half()
    with torch.no_grad():
        assert torch.equal(nested.model(x), model.model(x))


def test_the_rewrite_works_on_a_copy():
    model = tiny().half()
    weights = {k: v.clone() for k, v in model.model.state_dict().items()}
    nested = esrgan_export.for_export(model)
    assert nested.model is not model.model
    assert not any(
        isinstance(m, esrgan_export._NestedDenseBlock) for m in model.model.modules()
    )
    # The same state dict under the same names
    state = nested.model.state_dict()
    assert state.keys() == weights.keys()
    for key, value in weights.items():
        assert torch.equal(state[key], value)
        assert torch.equal(model.model.state_dict()[key], value)


def test_only_fp16_esrgan_is_rewritten():
    model = tiny()
    assert model.dtype == torch.float32
    assert esrgan_export.for_export(model) is model


def test_the_fp16_export_joins_two_tensors_at_a_time():
    half = convert_to_onnx_impl(tiny(), CPU, use_half=True, opset_version=20)
    # Two blocks: three RDBs each, four joins each
    assert concat_inputs(half) == [2] * (2 * 3 * 4)


def test_the_fp32_export_keeps_the_plain_joins():
    plain = convert_to_onnx_impl(tiny(), CPU, opset_version=20)
    assert concat_inputs(plain) == [2, 3, 4, 5] * (2 * 3)


@pytest.mark.parametrize(
    "settings",
    [{"scale": 4}, {"scale": 2, "shuffle_factor": 2}, {"scale": 8}],
    ids=["4x", "2x-unshuffle", "8x"],
)
@pytest.mark.parametrize("size", [32, 47])
def test_onnx_runtime_runs_the_fp16_export(settings: dict[str, object], size: int):
    model = tiny(**settings)
    data = convert_to_onnx_impl(model, CPU, use_half=True, opset_version=20)
    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    x = image(size=size)
    half = x.astype(np.float16)
    (actual,) = session.run(None, {"input": half})
    with torch.no_grad():
        expected = model.half()(torch.from_numpy(x).half()).float().numpy()
    assert actual.dtype == np.float16
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=2e-2)
