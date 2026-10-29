from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from spandrel import ModelLoader
from spandrel.architectures.DUAL.__arch.dual_arch import dual_xs

from nodes.impl.tensorrt import dual

# Needs Triton with its AMD backend (the Build Program environment); no GPU.
triton = pytest.importorskip("triton")
dual_hip = pytest.importorskip("nodes.impl.migraphx.dual_hip")

if "amd" not in triton.backends.backends:
    pytest.skip("this Triton has no AMD backend", allow_module_level=True)

KEY = "c128"
GFX = ["gfx1100", "gfx942", "gfx1201"]


def _plugin_model(key: str = KEY, *, read_sigma: bool = False) -> onnx.ModelProto:
    """Each of the exporter's plugin nodes, as graph.py emits them."""
    attrs = {"domain": "trt", "plugin_version": "1", "plugin_namespace": ""}
    nodes = [
        helper.make_node(
            f"DualNorm_{key}_TRT",
            ["x", "gamma", "beta"],
            ["n", "n_unused_1"],
            name="norm",
            residual=0,
            epsilon=1e-4,
            **attrs,
        ),
        helper.make_node(
            f"DualCore_{key}_TRT", ["qkv", "wdw", "tau"], ["core"], name="core", **attrs
        ),
        helper.make_node(
            f"DualProject_{key}_TRT",
            ["core", "wp", "n"],
            ["y"],
            name="project",
            **attrs,
        ),
        helper.make_node(
            f"DualNA_{key}_TRT",
            ["qkv", "bq", "bk"],
            ["na"],
            name="na",
            window=32,
            **attrs,
        ),
        helper.make_node("DualAttentionBarrier_TRT", ["y"], ["b"], name="b", **attrs),
        helper.make_node("Add", ["b", "na"], ["out"], name="add"),
    ]
    outputs = ["out", "n_unused_1"] if read_sigma else ["out"]
    initializers = [
        numpy_helper.from_array(np.ones(128, np.float32), "gamma"),
        numpy_helper.from_array(np.zeros(128, np.float32), "beta"),
        numpy_helper.from_array(np.ones(4, np.float32), "tau"),
    ]
    trunk = [1, 32, 48]
    graph = helper.make_graph(
        nodes,
        "plugins",
        [
            helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [*trunk, 128]),
            helper.make_tensor_value_info("qkv", TensorProto.BFLOAT16, [*trunk, 384]),
            helper.make_tensor_value_info("wdw", TensorProto.BFLOAT16, [3, 3, 384]),
            helper.make_tensor_value_info("wp", TensorProto.BFLOAT16, [128, 128]),
            helper.make_tensor_value_info("bq", TensorProto.BFLOAT16, [4, 47 * 47, 16]),
            helper.make_tensor_value_info("bk", TensorProto.BFLOAT16, [4, 47 * 47, 16]),
        ],
        [
            helper.make_tensor_value_info(name, TensorProto.BFLOAT16, [*trunk, 128])
            for name in outputs
        ],
        initializers,
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("trt", 1)],
    )


def _attributes(node: onnx.NodeProto) -> dict:
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def test_plugin_nodes_lower_to_the_chainner_ops():
    lowered = dual_hip.lower(_plugin_model(), KEY, dual_hip.Target.amd("gfx1100"))
    nodes = list(lowered.graph.node)
    assert [(n.domain, n.op_type) for n in nodes] == [
        *(
            ("chainner", f"chainner_dual_{op}")
            for op in ("norm", "cell", "region", "apply", "project", "na", "barrier")
        ),
        ("", "Add"),
    ]
    norm, cell, region, apply, project, na, barrier = nodes[:7]
    # One output each: Norm's unread sigma is dropped.
    assert list(norm.input) == ["x", "gamma", "beta"] and list(norm.output) == ["n"]
    assert list(cell.input) == ["qkv", "wdw"]
    assert list(region.input) == [cell.output[0], "tau", "qkv"]
    assert list(apply.input) == ["qkv", "wdw", region.output[0]]
    assert list(apply.output) == ["core"]
    assert list(project.input) == ["core", "wp", "n"] and list(project.output) == ["y"]
    assert list(na.input) == ["qkv", "bq", "bk"] and list(barrier.input) == ["y"]
    assert {(o.domain, o.version) for o in lowered.opset_import} == {
        ("", 20),
        ("chainner", 1),
    }
    onnx.checker.check_model(lowered)


def test_each_op_carries_its_kernel_and_tile_sizes():
    lowered = dual_hip.lower(_plugin_model(), KEY, dual_hip.Target.amd("gfx1100"))
    attributes = {
        n.op_type.removeprefix("chainner_dual_"): _attributes(n)
        for n in lowered.graph.node
    }
    for op in ("norm", "cell", "region", "apply", "project", "na"):
        kernel = attributes[op]
        assert kernel["code"].startswith(b"\x7fELF"), op  # an HSACO code object
        assert len(kernel["code_id"]) == 64 and kernel["block"] % 32 == 0
    assert attributes["norm"]["rows"] == dual_hip.NORM_PIXELS
    assert attributes["project"]["rows"] == dual_hip.PROJECT_ROWS
    assert attributes["apply"]["parts"] == 1
    assert (attributes["na"]["window"], attributes["na"]["parts"]) == (32, 2)
    assert attributes["barrier"] == {}
    # Norm's code is the dynamic TensorRT plugin's kernel, its sigma kept.
    assert attributes["norm"]["kernel"] == b"norm"


def test_a_fixed_key_lowers_alike():
    key = "c128_h32_w48"
    lowered = dual_hip.lower(_plugin_model(key), key, dual_hip.Target.amd("gfx1100"))
    assert [n.op_type for n in lowered.graph.node][:3] == [
        "chainner_dual_norm",
        "chainner_dual_cell",
        "chainner_dual_region",
    ]


def test_a_read_sigma_is_refused():
    with pytest.raises(ValueError, match="sigma"):
        dual_hip.lower(
            _plugin_model(read_sigma=True), KEY, dual_hip.Target.amd("gfx1100")
        )


def test_the_residual_norm_variant_is_refused():
    model = _plugin_model()
    for attribute in model.graph.node[0].attribute:
        if attribute.name == "residual":
            attribute.i = 1
    with pytest.raises(ValueError, match="residual"):
        dual_hip.lower(model, KEY, dual_hip.Target.amd("gfx1100"))


@pytest.mark.parametrize("gfx", ["gfx1100", "gfx90a:sramecc+:xnack-"])
def test_target_features_are_dropped_and_wave_size_follows_the_family(gfx: str):
    target = dual_hip.Target.amd(gfx)
    assert target.arch == gfx.partition(":")[0]
    assert target.warp_size == (64 if gfx.startswith("gfx9") else 32)


@pytest.mark.parametrize("gfx", GFX)
@pytest.mark.parametrize("channels", [128, 160])
def test_every_kernel_compiles_for_amd_without_a_gpu(gfx: str, channels: int):
    # 160 is a width that is not a power of two (masked projection columns).
    target = dual_hip.Target.amd(gfx)
    for op, parameter in (
        ("norm", 1e-4),
        ("cell", 0.0),
        ("region", 0.0),
        ("apply", 0.0),
        ("project", 0.0),
        ("na", 32),
        ("na", 64),
    ):
        kernel = dual_hip.kernel(op, channels, target, parameter)
        assert kernel["code"].startswith(b"\x7fELF"), (op, parameter)
        assert kernel["shared"] <= 65536
        warps = dual_hip.dual_na_kernel.OPTIONS["num_warps"] if op == "na" else 4
        assert kernel["block"] == warps * target.warp_size


@pytest.mark.skipif(
    "nvidia" not in triton.backends.backends, reason="this Triton has no CUDA backend"
)
def test_ptx_is_nul_terminated_for_the_driver():
    kernel = dual_hip.kernel("cell", 128, dual_hip.Target("cuda", 120))
    assert kernel["code"].endswith(b"\0") and b".entry cell_stats" in kernel["code"]


@pytest.fixture(scope="module")
def dynamic_xs(tmp_path_factory: pytest.TempPathFactory) -> onnx.ModelProto:
    torch.manual_seed(0)
    path = Path(tmp_path_factory.mktemp("xs")) / "xs.pth"
    torch.save(dual_xs(scale=1).eval().state_dict(), path)
    descriptor = ModelLoader("cpu").load_from_file(path)  # folded on load
    return onnx.load_from_string(
        dual.export_onnx(
            descriptor.model.state_dict(), descriptor.tags, descriptor.scale, None
        )
    )


@pytest.mark.parametrize(
    ("size", "classes"),
    [
        # per trunk axis: 96-px blocks, and a strip if the axis is not a multiple
        ((200, 200), 4),
        ((64, 96), 1),  # no blocks down, no strip across
        ((192, 100), 2),  # no strip down
    ],
)
def test_a_dynamic_onnx_keeps_the_retrieval_classes_of_its_size(
    dynamic_xs: onnx.ModelProto, size: tuple[int, int], classes: int
):
    # MIGraphX refuses the empty tensors of absent classes (TensorRT takes them).
    lowered = dual_hip.lower(dynamic_xs, KEY, dual_hip.Target.amd("gfx1100"))
    before = len(lowered.graph.node)
    dual_hip.fix_size(lowered, *size, unshuffle=False)
    graph = lowered.graph
    dims = graph.input[0].type.tensor_type.shape.dim
    assert [d.dim_value for d in dims] == [1, 3, *size]
    # XS x1 retrieves once (XR), one attention barrier per window class
    barriers = [
        n
        for n in graph.node
        if n.op_type == "chainner_dual_barrier" and n.name.startswith("/xr/")
    ]
    assert len(barriers) == classes
    assert (len(graph.node) == before) == (classes == 4)
    assert not [n for n in graph.node if n.op_type == "Concat" and len(n.input) < 2]
    produced = {o for n in graph.node for o in n.output}
    constants = {i.name for i in graph.initializer} | {i.name for i in graph.input}
    assert {i for n in graph.node for i in n.input if i} <= produced | constants
    onnx.checker.check_model(lowered)
