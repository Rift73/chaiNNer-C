from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
import torch
from onnx import TensorProto, helper
from spandrel import ModelLoader
from spandrel.architectures.DUAL.__arch import dual_arch
from spandrel.architectures.DUAL.__arch.dual_arch import dual_xs

from nodes.impl.onnx.load import load_onnx_model
from nodes.impl.tensorrt import dual


@pytest.mark.parametrize(
    ("tags", "factory"),
    [(["Light", "128dim"], "dual_light"), (["XS"], "dual_xs"), (["XL"], "dual_xl")],
)
def test_presets_map_to_the_guides_factories(tags: list[str], factory: str):
    assert dual.factory_of(tags) == factory


@pytest.mark.parametrize("tags", [[], ["96dim", "4g"], ["DUAL3X"]])
def test_other_dual_models_are_refused(tags: list[str]):
    with pytest.raises(ValueError, match="six DUAL presets"):
        dual.factory_of(tags)


def _plain_onnx() -> bytes:
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph(
        [node],
        "plain",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])],
    )
    return helper.make_model(graph).SerializeToString()


def test_only_a_dual_export_has_a_manifest():
    assert dual.export_manifest(_plain_onnx()) is None


def test_only_the_rgb_readout_moves_ahead_of_its_pixel_shuffle():
    # Moving a 3x3 convolution ahead of a x2 shuffle keeps its result but quadruples its
    # work; for the 64-to-256 tail convolution that cost Light ~4.5 ms at 512x512.
    spec = importlib.util.spec_from_file_location(
        "dual_tensorrt_graph", dual.VENDOR / "scripts/dual_tensorrt/graph.py"
    )
    assert spec is not None and spec.loader is not None
    graph_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph_module)
    tail = torch.nn.Sequential(
        torch.nn.Conv2d(64, 256, 3, padding=1),
        torch.nn.PixelShuffle(2),
        torch.nn.Conv2d(64, 256, 3, padding=1),
        torch.nn.PixelShuffle(2),
        torch.nn.Conv2d(64, 3, 3, padding=1),
    )
    graph = graph_module.Graph(dual_arch, SimpleNamespace(channels=128, trunk=(8, 8)))
    graph.tail_sequence("x", tail)
    weights = {constant.name: tuple(constant.dims) for constant in graph.constants}
    convolutions = [node for node in graph.nodes if node.op_type == "Conv"]
    assert [weights[node.input[1]] for node in convolutions] == [
        (256, 64, 3, 3),
        (256, 64, 3, 3),
        (12, 256, 3, 3),
    ]


def test_a_folded_xs_exports_a_fixed_size_tensorrt_onnx(tmp_path: Path):
    torch.manual_seed(0)
    model = dual_xs(scale=1).eval()
    path = tmp_path / "xs.pth"
    torch.save(model.state_dict(), path)
    descriptor = ModelLoader("cpu").load_from_file(path)  # folded on load
    onnx_bytes = dual.export_onnx(
        descriptor.model.state_dict(), descriptor.tags, descriptor.scale, 32, 64
    )
    manifest = dual.export_manifest(onnx_bytes)
    assert manifest is not None
    spec = manifest["specialization"]
    assert (spec["factory"], spec["scale"], spec["height"], spec["width"]) == (
        "dual_xs",
        1,
        32,
        64,
    )
    assert spec["plugin_key"] == "c128_h32_w64"
    assert manifest["checkpoint_folded"] is True
    domains = {node.domain for node in onnx.load_from_string(onnx_bytes).graph.node}
    assert "trt" in domains
    info = load_onnx_model(onnx_bytes).info
    assert (info.fixed_input_height, info.fixed_input_width) == (32, 64)
    assert (info.scale_height, info.input_channels, info.output_channels) == (1, 3, 3)
