from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnx.external_data_helper
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from spandrel import ModelLoader
from spandrel.architectures.DUAL.__arch import dual_arch
from spandrel.architectures.DUAL.__arch.dual_arch import dual_xs

from nodes.impl.onnx.load import load_onnx_model
from nodes.impl.onnx.model import MIXED
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
    assert dual.export_manifest(onnx.load_from_string(_plain_onnx())) is None


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


def test_the_exporters_write_onnxs_files_in_one_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # ONNX's own writer opens and closes weights.bin once per tensor, and real-time
    # antivirus scanning rescans the growing file after each close: a DUAL Light export
    # (335 tensors) took about 100 s with Windows Defender on, 0.1 s with it off.
    monkeypatch.syspath_prepend(str(dual.VENDOR))
    export = importlib.import_module("scripts.dual_tensorrt.export")
    # 2048, 16, 1000 (external by ONNX's rule, which counts the bytes object's
    # overhead), 4096 and 8 bytes.
    shapes = [(16, 32), (4,), (250,), (32, 32), (2,)]
    weights = [
        numpy_helper.from_array(
            np.arange(np.prod(shape), dtype=np.float32).reshape(shape), f"w{i}"
        )
        for i, shape in enumerate(shapes)
    ]
    value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "weights",
        [value],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        initializer=weights,
    )
    model = helper.make_model(graph)
    expected, written = tmp_path / "onnx", tmp_path / "one-pass"
    expected.mkdir()
    written.mkdir()
    reference = onnx.ModelProto()
    reference.CopyFrom(model)
    onnx.save_model(
        reference,
        str(expected / "model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=1024,
    )

    def per_tensor(*_args: object):
        raise AssertionError("weights.bin was written one tensor at a time")

    monkeypatch.setattr(onnx.external_data_helper, "save_external_data", per_tensor)
    export.save_onnx(model, written)
    for name in ("model.onnx", "weights.bin"):
        assert (written / name).read_bytes() == (expected / name).read_bytes()
    assert (expected / "weights.bin").stat().st_size == 2048 + 1000 + 4096


def _folded_xs(tmp_path: Path):
    torch.manual_seed(0)
    model = dual_xs(scale=1).eval()
    path = tmp_path / "xs.pth"
    torch.save(model.state_dict(), path)
    return ModelLoader("cpu").load_from_file(path)  # folded on load


@pytest.fixture(scope="module")
def fixed_xs_onnx(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    descriptor = _folded_xs(tmp_path_factory.mktemp("xs"))
    return dual.export_onnx(
        descriptor.model.state_dict(), descriptor.tags, descriptor.scale, (32, 64)
    )


def test_a_folded_xs_exports_a_fixed_size_tensorrt_onnx(fixed_xs_onnx: bytes):
    onnx_bytes = fixed_xs_onnx
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
    model = onnx.load_from_string(onnx_bytes)
    assert "trt" in {node.domain for node in model.graph.node}
    assert dual.export_manifest(model) == manifest


def test_chainner_loads_a_dual_onnx_as_mixed_precision(fixed_xs_onnx: bytes):
    # Loading runs in the native backend, which GitHub's checks do not build.
    pytest.importorskip("nodes.impl._chainner_graph")
    info = load_onnx_model(fixed_xs_onnx).info
    assert (info.fixed_input_height, info.fixed_input_width) == (32, 64)
    assert (info.scale_height, info.input_channels, info.output_channels) == (1, 3, 3)
    assert info.dtype == MIXED
    assert load_onnx_model(_plain_onnx()).info.dtype == "fp32"


def test_a_folded_xs_exports_one_tensorrt_onnx_for_every_size(tmp_path: Path):
    descriptor = _folded_xs(tmp_path)
    onnx_bytes = dual.export_onnx(
        descriptor.model.state_dict(), descriptor.tags, descriptor.scale, None
    )
    manifest = dual.export_manifest(onnx_bytes)
    assert manifest is not None
    spec = manifest["specialization"]
    assert (spec["factory"], spec["scale"], spec["dynamic"]) == ("dual_xs", 1, True)
    assert (spec["alignment"], spec["plugin_key"]) == (dual.ALIGNMENT, "c128")
    graph = onnx.load_from_string(onnx_bytes).graph
    dims = graph.input[0].type.tensor_type.shape.dim
    assert [d.dim_value or d.dim_param for d in dims] == [1, 3, "in_height", "in_width"]
    plugins = {node.op_type for node in graph.node if node.domain == "trt"}
    assert plugins == {
        "DualNorm_c128_TRT",
        "DualCore_c128_TRT",
        "DualProject_c128_TRT",
        "DualAttentionBarrier_TRT",
    }


def test_a_dual_engines_input_name_carries_its_alignment():
    assert dual.input_alignment(dual.aligned_input_name(4)) == 4
    assert dual.input_alignment("input") is None
    assert dual.input_alignment("input_aligned_x") is None


@pytest.mark.parametrize(
    ("low", "opt", "high", "message"),
    [
        ((64, 64), (512, 512), (1088, 1920), None),
        ((64, 64), (512, 500), (1088, 1920), "dynamic DUAL engine's optimal size must"),
        ((64, 64), (2048, 512), (1088, 1920), "minimum <= optimal <= maximum"),
        # Shape Mode Fixed on a dynamic ONNX: one size, a fixed engine
        ((256, 250), (256, 250), (256, 250), r"^A fixed DUAL engine's size must be"),
    ],
)
def test_a_dynamic_engines_profile_keeps_to_the_onnxs_alignment(
    low: tuple[int, int],
    opt: tuple[int, int],
    high: tuple[int, int],
    message: str | None,
):
    manifest = {"specialization": {"dynamic": True, "alignment": 64}}
    if message is None:
        dual.check_profile(manifest, low, opt, high)
    else:
        with pytest.raises(ValueError, match=message):
            dual.check_profile(manifest, low, opt, high)
