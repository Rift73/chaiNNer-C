from __future__ import annotations

from collections import Counter

import onnx
import pytest
import torch
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.DRFT.__arch import legacy_drft_arch as legacy

from nodes.impl.tensorrt import drft, drft_export

PLUGIN_OPS = {
    "DrftIdentityBarrier_TRT",
    "DrftOCABQKVPackTokenMajor_TRT",
    "DrftOCABRelativeBiasSafe_TRT",
    "DrftWindowNeuralBias_TRT",
    "DrftFinalAffineCrop_TRT",
}


def legacy_drft(window_size: int, overlap_ratio: float) -> ImageModelDescriptor:
    """A small random legacy DRFT x2 (2 groups of 6 blocks), loaded by spandrel."""
    torch.manual_seed(0)
    network = legacy.DRFT(
        embed_dim=48,
        depths=(6, 6),
        num_heads=(6, 6),
        window_size=window_size,
        overlap_ratio=overlap_ratio,
        upscale=2,
        num_feat=16,
        rank=8,
        drop_path_rate=0.0,
    )
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.normal_(0.0, 0.2)
    descriptor = ModelLoader().load_from_state_dict(network.state_dict())
    assert isinstance(descriptor, ImageModelDescriptor)
    return descriptor


@pytest.fixture(scope="module")
def plugin_model() -> ImageModelDescriptor:
    return legacy_drft(32, 0.5)


@pytest.fixture(scope="module")
def plugin_onnx(plugin_model: ImageModelDescriptor) -> bytes:
    """The plugin model's export, once: an FP16 trace on the CPU takes about half a
    minute even for this small model (PyTorch's CPU convolutions in FP16 are slow)."""
    return drft_export.export_onnx(plugin_model)


def test_the_loaded_model_is_legacy_drft(plugin_model: ImageModelDescriptor):
    assert drft_export.is_legacy_drft(plugin_model)


@pytest.mark.parametrize(("window_size", "overlap_ratio"), [(32, 0.5), (16, 0.5)])
@pytest.mark.parametrize(("height", "width"), [(64, 64), (80, 112)])
def test_the_swapped_model_computes_the_original_in_fp32(
    window_size: int, overlap_ratio: float, height: int, width: int
):
    descriptor = legacy_drft(window_size, overlap_ratio)
    original = descriptor.model
    assert isinstance(original, legacy.DRFT)
    original = original.float().eval()
    with torch.no_grad():
        original.fold_reparam_conv()
    swapped = drft_export.deployment(original)
    image = torch.rand(1, 3, height, width)
    with torch.inference_mode():
        expected = original(image)
        actual = swapped(image)
    assert actual.dtype == torch.float32
    # The same maths as spandrel's own forward, summed in another order (its attention
    # kernel)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def plugin_counts(proto: onnx.ModelProto) -> Counter[str]:
    return Counter(node.op_type for node in proto.graph.node)


def test_the_onnx_holds_the_plugins_and_no_einsum(plugin_onnx: bytes):
    proto = onnx.load_from_string(plugin_onnx)
    counts = plugin_counts(proto)
    assert "Einsum" not in counts
    groups, blocks = 2, 6
    assert {op: counts[op] for op in PLUGIN_OPS} == {
        # blocks 2 and 5 of each group, each group, norm, body add,
        # conv_before_upsample, one pixel shuffle (x2)
        "DrftIdentityBarrier_TRT": groups * 2 + groups + 4,
        "DrftOCABQKVPackTokenMajor_TRT": groups,
        "DrftOCABRelativeBiasSafe_TRT": groups,
        "DrftWindowNeuralBias_TRT": groups * blocks,
        "DrftFinalAffineCrop_TRT": 1,
    }
    for node in proto.graph.node:
        if node.op_type in PLUGIN_OPS:
            assert node.domain == ""
    attributes = {
        node.op_type: {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        for node in proto.graph.node
        if node.op_type in PLUGIN_OPS
    }
    plugin = {"plugin_version": b"1", "plugin_namespace": b""}
    assert attributes["DrftOCABRelativeBiasSafe_TRT"] == {
        **plugin,
        "q_window_size": 32,
        "kv_window_size": 48,
    }
    assert attributes["DrftOCABQKVPackTokenMajor_TRT"] == {
        **plugin,
        "channels": 48,
        "heads": 6,
        "query_window_size": 32,
        "key_window_size": 48,
    }
    assert attributes["DrftWindowNeuralBias_TRT"] == {}
    assert attributes["DrftIdentityBarrier_TRT"] == plugin
    final = attributes["DrftFinalAffineCrop_TRT"]
    assert final.pop("scale") == 2
    assert final.pop("plugin_version") == b"1"
    assert final.pop("plugin_namespace") == b""
    assert set(final) == {"mean_r", "mean_g", "mean_b"}
    graph_input, graph_output = proto.graph.input[0], proto.graph.output[0]
    assert graph_input.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert graph_output.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    for value in (graph_input, graph_output):
        dims = value.type.tensor_type.shape.dim
        assert all(dim.dim_param for dim in dims[2:])
    types = Counter(
        onnx.TensorProto.DataType.Name(tensor.data_type)
        for tensor in proto.graph.initializer
    )
    assert types["FLOAT16"] and not types["BFLOAT16"]


def test_the_attention_logits_are_not_clamped(plugin_onnx: bytes):
    """spandrel's export-safe maths clamp the logits to +-50 (a Clip), which changes
    the softmax wherever one passes 50; the export's attention does not."""
    proto = onnx.load_from_string(plugin_onnx)
    assert "Clip" not in plugin_counts(proto)


def test_the_dense_skip_exports_one_matmul_per_block(
    plugin_onnx: bytes,
):
    """TensorRT 11 computes Concat(block outputs) @ W wrongly in this graph, so no
    MatMul takes a Concat; each group sums one MatMul per block instead."""
    proto = onnx.load_from_string(plugin_onnx)
    producers = {out: node for node in proto.graph.node for out in node.output}
    matmuls = [node for node in proto.graph.node if node.op_type == "MatMul"]
    assert not any(
        node.input[0] in producers and producers[node.input[0]].op_type == "Concat"
        for node in matmuls
    )
    groups, blocks = 2, 6
    dense = [node for node in matmuls if "/residual_group/MatMul" in node.name]
    assert len(dense) == groups * blocks


def test_other_windows_keep_the_dense_bias_as_maths():
    proto = onnx.load_from_string(drft_export.export_onnx(legacy_drft(16, 0.5)))
    counts = plugin_counts(proto)
    assert counts["DrftOCABRelativeBiasSafe_TRT"] == 0
    assert counts["DrftOCABQKVPackTokenMajor_TRT"] == 2
    assert counts["DrftWindowNeuralBias_TRT"] == 12
    assert "Einsum" not in counts


def test_the_manifest_round_trips(plugin_onnx: bytes):
    manifest = drft.export_manifest(plugin_onnx)
    assert manifest == {
        "format": 1,
        "plugins": "legacy-v1",
        "scale": 2,
        "window_size": 32,
        "min_size": 64,
        "alignment": 1,
    }


def test_other_models_are_refused():
    from spandrel.architectures.SPAN import SPAN

    descriptor = ModelLoader().load_from_state_dict(
        SPAN(num_in_ch=3, num_out_ch=3).state_dict()
    )
    assert isinstance(descriptor, ImageModelDescriptor)
    assert not drft_export.is_legacy_drft(descriptor)
    with pytest.raises(ValueError):
        drft_export.export_onnx(descriptor)


def test_chainner_loads_the_onnx(plugin_onnx: bytes):
    from nodes.impl.onnx.load import load_onnx_model

    assert load_onnx_model(plugin_onnx).sub_type == "Generic"
