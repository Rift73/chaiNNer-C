from __future__ import annotations

import re

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

# Needs TensorRT and Triton (the engine build's environment).
dual_aot = pytest.importorskip("nodes.impl.tensorrt.dual_aot")

KEY = "c128_h32_w48"


def _guide_model():
    """A graph with each of the guide's plugin nodes, as graph.py emits them."""
    attrs = {"domain": "trt", "plugin_version": "1", "plugin_namespace": ""}
    nodes = [
        helper.make_node(
            f"DualNorm_{KEY}_TRT",
            ["x", "gamma", "beta"],
            ["n", "n_unused_1"],
            name="norm",
            residual=0,
            epsilon=1e-5,
            **attrs,
        ),
        helper.make_node(
            f"DualCore_{KEY}_TRT", ["qkv", "wdw", "tau"], ["core"], name="core", **attrs
        ),
        helper.make_node(
            f"DualProject_{KEY}_TRT",
            ["core", "wp", "n"],
            ["y"],
            name="project",
            **attrs,
        ),
        helper.make_node("DualAttentionBarrier_TRT", ["y"], ["out"], name="b", **attrs),
    ]
    initializers = [
        numpy_helper.from_array(np.ones(128, np.float32), "gamma"),
        numpy_helper.from_array(np.zeros(128, np.float32), "beta"),
        numpy_helper.from_array(np.ones(4, np.float32), "tau"),
    ]
    graph = helper.make_graph(
        nodes,
        "guide",
        [
            helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [1, 32, 48, 128]),
            helper.make_tensor_value_info(
                "qkv", TensorProto.BFLOAT16, [1, 32, 48, 384]
            ),
            helper.make_tensor_value_info("wdw", TensorProto.BFLOAT16, [3, 3, 384]),
            helper.make_tensor_value_info("wp", TensorProto.BFLOAT16, [128, 128]),
        ],
        [helper.make_tensor_value_info("out", TensorProto.BFLOAT16, [1, 32, 48, 128])],
        initializers,
    )
    return helper.make_model(graph)


def test_geometry_of_a_plugin_key():
    g = dual_aot.Geometry.of(KEY)
    assert (g.channels, g.height, g.width, g.heads) == (128, 32, 48, 4)
    assert g.cells == 2 * 3
    assert g.region_counts == [1 * 2, 1 * 2, 2 * 2, 2 * 2]
    assert g.stats_size == 6 * 4 * 1088
    assert g.matrices_size == 12 * 4 * 1024


def test_lowering_replaces_every_plugin_node_with_stand_ins():
    lowered = dual_aot.lower(_guide_model(), KEY)
    ops = [p.op for p in lowered.plugins]
    assert ops == [
        f"norm_{KEY}",
        f"cell_{KEY}",
        f"region_{KEY}",
        f"apply_{KEY}",
        f"project_{KEY}",
        "barrier",
    ]
    norm, cell, region, apply, project, barrier = lowered.plugins
    assert norm.attributes == {"epsilon": pytest.approx(1e-5)}
    assert (norm.inputs, norm.outputs) == (["x", "gamma", "beta"], ["n", "n_unused_1"])
    assert cell.inputs == ["qkv", "wdw"] and region.inputs == [cell.outputs[0], "tau"]
    assert apply.inputs == ["qkv", "wdw", region.outputs[0]] and apply.outputs == [
        "core"
    ]
    assert project.inputs == ["core", "wp", "n"] and barrier.outputs == ["out"]
    graph = lowered.model.graph
    assert not [n for n in graph.node if n.op_type.startswith("Dual")]
    produced = {o for n in graph.node for o in n.output}
    for plugin in lowered.plugins:
        assert set(plugin.outputs) <= produced  # every plugin output has a stand-in
    read = {n.input[0] for n in graph.node if n.name in lowered.readers}
    assert {"gamma", "beta", "tau", "wdw", "wp", "x", "qkv"} <= read


def test_a_width_only_key_lowers_to_the_dynamic_plugins():
    dynamic = "c128"
    model = _guide_model()
    for node in model.graph.node:
        node.op_type = node.op_type.replace(KEY, dynamic)
    lowered = dual_aot.lower(model, dynamic)
    assert [p.op for p in lowered.plugins] == [
        "norm_c128",
        "cell_c128",
        "region_c128",
        "apply_c128",
        "project_c128",
        "barrier_dynamic",
    ]
    cell, region = lowered.plugins[1:3]
    # The region plugin takes qkv only to see the trunk's run-time size.
    assert region.inputs == [cell.outputs[0], "tau", "qkv"]


def _na_model(key: str):
    """A neighborhood-attention node as graph.py emits it."""
    node = helper.make_node(
        f"DualNA_{key}_TRT",
        ["qkv", "bq", "bk"],
        ["na"],
        name="na",
        domain="trt",
        plugin_version="1",
        plugin_namespace="",
        window=64,
    )
    tables = [
        numpy_helper.from_array(np.zeros((4, 79 * 79, 16), np.float32), name)
        for name in ("bq", "bk")
    ]
    graph = helper.make_graph(
        [node],
        "na",
        [helper.make_tensor_value_info("qkv", TensorProto.BFLOAT16, [1, 32, 48, 384])],
        [helper.make_tensor_value_info("na", TensorProto.BFLOAT16, [1, 32, 48, 128])],
        tables,
    )
    return helper.make_model(graph)


@pytest.mark.parametrize("key", [KEY, "c128"])
def test_the_na_node_lowers_to_a_slice_and_its_plugin(key: str):
    lowered = dual_aot.lower(_na_model(key), key)
    (plugin,) = lowered.plugins
    assert (plugin.op, plugin.inputs, plugin.outputs, plugin.attributes) == (
        f"na_{key}",
        ["qkv", "bq", "bk"],
        ["na"],
        {"window": 64},
    )
    graph = lowered.model.graph
    (stand_in,) = [n for n in graph.node if "na" in n.output]
    constants = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    assert stand_in.op_type == "Slice" and stand_in.input[0] == "qkv"
    assert [constants[name].tolist() for name in stand_in.input[1:]] == [
        [0],
        [128],
        [3],
    ]
    read = {n.input[0] for n in graph.node if n.name in lowered.readers}
    assert read == {"qkv", "bq", "bk"}


@pytest.mark.parametrize("sm", [86, 120])
@pytest.mark.parametrize("window", [32, 64])
def test_the_na_kernel_fits_the_aot_plugins_shared_memory(window: int, sm: int):
    # TensorRT's AOT plugin path launches with at most 48 KiB of shared memory.
    kernel = dual_aot.na_kernel(128, window, sm)
    assert kernel.metadata.shared <= dual_aot.MAX_SHARED


def test_kernels_load_whole_16_byte_vectors():
    # TensorRT's buffers are 16-byte aligned; unless the compile is told so, Triton loads
    # one element at a time and the core runs about 14x slower.
    kernel = dual_aot.compile_kernel(
        dual_aot.vendored_kernels().cell_stats,
        {"X": "*bf16", "WEIGHT": "*bf16", "STATS": "*fp32"},
        {"H": 32, "W": 48, "C": 128, "BT": 256},
        120,
    )
    assert re.search(r"ld\.global\S*\.v4\.", kernel.asm["ptx"])


def test_the_residual_norm_variant_is_refused():
    model = _guide_model()
    for attribute in model.graph.node[0].attribute:
        if attribute.name == "residual":
            attribute.i = 1
    with pytest.raises(ValueError, match="residual"):
        dual_aot.lower(model, KEY)
