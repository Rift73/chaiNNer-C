"""RDUAL's TensorRT ONNX parts (nodes/impl/tensorrt/rdual_export.py), exported on the CPU
as the child process does; nothing here needs a GPU, TensorRT or Triton."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from spandrel.architectures.RDUAL import RDUAL, RDUALArch

from nodes.impl.tensorrt import rdual, rdual_export, rdual_recipe

# The parts' node counts at 384x384 of the graphs whose engines were checked against
# the training source (the 2x DUAL Light layout, four experts). Local and retrieval
# neighborhood attention made them fewer (677, 1375, 1455 with local windows and 96 px
# retrieval blocks): each local layer above 16 positions per axis is one DualLocalNA
# node in place of its window partition, and each retrieval one DualXR node in place of
# its blocks; the routers' means are DualRowSum nodes. The flow is unchanged.
NODES_384 = {"prefix": 585, "revcore": 1193, "fwdcore": 1254, "flow": 976}


@pytest.fixture(scope="module")
def export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    model = RDUAL(scale=2, use_checkpoint=False)
    # Distinct flow weights, as trained ones are: the exporter shares equal
    # constants (the initial zero biases, unit scales) through Identity nodes.
    with torch.no_grad():
        for parameter in model.flow.parameters():
            parameter.normal_(0, 0.05)
    model.get_buffer("flow_loaded").fill_(True)
    folded = RDUALArch().load(model.state_dict()).model
    root = tmp_path_factory.mktemp("rdual")
    weights = rdual.write_weights(folded, root / "weights.safetensors")
    rdual_export.main(str(weights), str(root / "export"), "384", "384")
    return root / "export"


def manifest(export: Path) -> dict:
    return json.loads((export / "export.json").read_text(encoding="utf-8"))


def test_an_export_at_384_has_the_checked_graphs_node_counts(export: Path):
    parts = manifest(export)["parts"]

    assert {name: part["nodes"] for name, part in parts.items()} == NODES_384
    for name in NODES_384:
        assert (export / name / "model.onnx").is_file()
        assert (export / name / "weights.bin").is_file()


def test_the_parts_pass_the_sweeps_tensors_to_each_other(export: Path):
    m = manifest(export)
    assert (m["height"], m["width"], m["scale"]) == (384, 384, 2)
    assert m["plugin_key"] == "c128_h192_w192"
    parts = {
        name: (
            dict(part["inputs"]),
            dict(part["outputs"]),
        )
        for name, part in m["parts"].items()
    }
    trunk = "BFLOAT16, 1x192x192x128"
    cut = {"fine": trunk, "center": trunk, "key": "BFLOAT16, 1x96x96x128"}

    def donor(name: str) -> dict[str, str]:
        return {
            f"state_{name}": trunk,
            f"guidance_{name}": "FLOAT, 1x5x192x192",
            f"valid_{name}": "BOOL, 1",
        }

    image = {"input": "FLOAT, 1x3x384x384"}
    assert parts["prefix"] == (image, {"stem": trunk, **cut})
    assert parts["revcore"] == ({**donor("future"), **cut}, {"state_out": trunk})
    assert parts["fwdcore"] == (
        {**image, **donor("past"), **donor("future"), "stem": trunk, **cut},
        {"output": "FLOAT, 1x3x768x768", "state_out": trunk},
    )
    assert parts["flow"] == (
        {"frames": "FLOAT, 2x3x384x384"},
        {"guidance": "FLOAT, 2x2x192x192"},
    )


def test_the_plugin_nodes_are_rduals_and_duals(export: Path):
    import onnx

    kinds = {
        name: {
            node.op_type
            for node in onnx.load(
                str(export / name / "model.onnx"), load_external_data=False
            ).graph.node
            if node.domain == "trt"
        }
        for name in NODES_384
    }

    assert kinds["flow"] == {"RDUALCorrWindow", "RDUALCorrSample", "RDUALMatch"}
    assert {"RDUALFlowKV", "RDUALTemporal"} <= kinds["revcore"]
    assert {"RDUALFlowKV", "RDUALTemporal"} <= kinds["fwdcore"]
    assert not kinds["prefix"] & {"RDUALFlowKV", "RDUALTemporal"}


def test_local_and_retrieval_neighborhoods_run_at_every_site(export: Path):
    """At 384x384 the trunk (192) and the coarse branch (96) exceed 16 per axis and the
    coarse grid exceeds 48: every local layer is local NA, at both sizes, and both
    transfers are retrieval NA; the manifest says so."""
    import onnx

    def count(part: str, op: str) -> int:
        graph = onnx.load(str(export / part / "model.onnx"), load_external_data=False)
        return sum(n.op_type == f"{op}_c128_h192_w192_TRT" for n in graph.graph.node)

    # Group 0's local layers 0, 1, 3 at the trunk's size, its first two in the coarse
    # branch; groups 1-3 hold 2, 2 and 4 local layers.
    assert [count(p, "DualLocalNA") for p in ("prefix", "revcore", "fwdcore")] == [
        5,
        8,
        8,
    ]
    # The 2x transfer feeds the state; the 4x transfer reuses its neighborhoods.
    assert [count(p, "DualXR") for p in ("prefix", "revcore", "fwdcore")] == [0, 1, 2]
    m = manifest(export)
    assert (m["context"], m["local"], m["retrieval"]) == ("na-v1",) * 3


def test_a_model_with_its_neighborhoods_off_keeps_windows_and_blocks():
    model = SimpleNamespace(local_na=False, retrieval_na=False)
    assert rdual_export.neighborhoods(model) == (False, False)
    # spandrel's RDUAL before the flags takes the training source's defaults.
    assert rdual_export.neighborhoods(SimpleNamespace()) == (True, True)


def test_sizes_pad_as_the_model_pads():
    size = rdual_recipe.Size(837, 1488)

    assert (size.padded_height, size.padded_width) == (840, 1488)
    assert (size.trunk_height, size.trunk_width) == (420, 744)
    assert (size.flow_height, size.flow_width) == (848, 1488)
    small = rdual_recipe.Size(17, 9)
    assert (small.flow_height, small.flow_width) == (32, 32)


def test_other_context_windows_are_refused():
    model = SimpleNamespace(hyperparameters={"c_windows": (16, 32)})

    with pytest.raises(ValueError, match="context windows"):
        rdual_export.trunk_parts(model, rdual_recipe.Size(64, 64))


def test_the_flow_s_precision_is_chosen_in_one_place():
    size = rdual_recipe.Size

    assert rdual_recipe.flow_tokens(size(837, 1488)) == 4929
    assert rdual_recipe.flow_tokens(size(1080, 1920)) == 8160
    assert rdual_recipe.flow_precision(size(837, 1488)) is rdual_recipe.MIXED_FLOW
    assert rdual_recipe.flow_precision(size(720, 1280)) is rdual_recipe.MIXED_FLOW
    assert rdual_recipe.flow_tokens(size(900, 1600)) == 5700
    assert rdual_recipe.flow_precision(size(900, 1600)) is rdual_recipe.MIXED_FLOW
    assert rdual_recipe.flow_precision(size(1080, 1920)) is rdual_recipe.FULL_FLOW
    # a build may name its recipe
    assert rdual_recipe.flow_precision(size(1080, 1920), "mixed").half
    assert not rdual_recipe.flow_precision(size(384, 384), "full").half_modules
    with pytest.raises(ValueError, match="No flow recipe"):
        rdual_recipe.flow_precision(size(384, 384), "fp8")
    assert rdual.flow_recipe(1080, 1920) == "full"
    assert rdual.flow_recipe(1080, 1920, "mixed") == "mixed"


def test_the_manifest_names_the_flow_s_recipe(export: Path):
    assert manifest(export)["flow_precision"] == "mixed"
