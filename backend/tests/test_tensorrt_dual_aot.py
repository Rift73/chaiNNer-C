from __future__ import annotations

import re

import numpy as np
import onnx
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


def _router_model(residual_shape: list[int]):
    """The guide's graph followed by a dynamic router's input: the block's output plus a
    residual, then Transpose to NCHW, Cast to FP32 and ReduceMean over H and W."""
    model = _guide_model()
    graph = model.graph
    graph.input.append(
        helper.make_tensor_value_info("residual", TensorProto.BFLOAT16, residual_shape)
    )
    graph.initializer.append(
        numpy_helper.from_array(np.array([2, 3], np.int64), "mean_axes")
    )
    graph.node.extend(
        [
            helper.make_node("Add", ["out", "residual"], ["sum"], name="add"),
            helper.make_node("Transpose", ["sum"], ["nchw"], perm=[0, 3, 1, 2]),
            helper.make_node("Cast", ["nchw"], ["wide"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMean", ["wide", "mean_axes"], ["mean"]),
        ]
    )
    graph.output.append(
        helper.make_tensor_value_info("mean", TensorProto.FLOAT, [1, 128, 1, 1])
    )
    return model


def test_a_router_s_residual_addition_lowers_to_its_plugin():
    lowered = dual_aot.lower(_router_model([1, 32, 48, 128]), KEY)
    *_, barrier, router = lowered.plugins
    assert barrier.op == "barrier"
    assert (router.op, router.inputs, router.outputs) == (
        "router_add",
        ["out", "residual"],
        ["sum"],
    )
    read = {
        lowered.readers[n.name]
        for n in lowered.model.graph.node
        if "router_add" in n.name
    }
    assert read == {"out", "residual"}


def test_a_broadcast_residual_keeps_the_native_addition():
    lowered = dual_aot.lower(_router_model([1, 1, 1, 128]), KEY)
    assert "router_add" not in [p.op for p in lowered.plugins]


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
        {"H": 32, "W": 48, "C": 128, "BT": dual_aot.CELL_BT},
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


def _chain_model(key: str = KEY) -> onnx.ModelProto:
    """One trunk layer's tail as the exporter writes it: projection + bias -> residual
    add -> Norm -> expand + bias -> GELU (chain A), then contraction + bias -> residual
    add -> the next Norm (chain B); `n2` (the next Norm's output) is the graph output."""
    attrs = {"domain": "trt", "plugin_version": "1", "plugin_namespace": ""}
    c, shape = 128, [1, 32, 48]

    def weight(name: str, dims: list[int]) -> onnx.TensorProto:
        return numpy_helper.from_array(np.full(dims, 0.01, np.float32), name)

    def norm(x: str, out: str, name: str) -> onnx.NodeProto:
        return helper.make_node(
            f"DualNorm_{key}_TRT",
            [x, "gamma", "beta"],
            [out, out + "_sigma"],
            name=name,
            residual=0,
            epsilon=1e-5,
            **attrs,
        )

    nodes = [
        helper.make_node("MatMul", ["g", "w1"], ["p"]),
        helper.make_node("Add", ["p", "b1"], ["pb"]),
        helper.make_node("Add", ["r", "pb"], ["s"]),
        norm("s", "n", "norm"),
        helper.make_node("MatMul", ["n", "w2"], ["e"]),
        helper.make_node("Add", ["e", "b2"], ["h"]),
        helper.make_node("Div", ["h", "c_div"], ["hd"]),
        helper.make_node("Erf", ["hd"], ["he"]),
        helper.make_node("Add", ["he", "c_add"], ["ha"]),
        helper.make_node("Mul", ["h", "c_mul"], ["hm"]),
        helper.make_node("Mul", ["hm", "ha"], ["gelu"]),
        helper.make_node("MatMul", ["gelu", "w3"], ["q"]),
        helper.make_node("Add", ["q", "b3"], ["qb"]),
        helper.make_node("Add", ["s", "qb"], ["s2"]),
        norm("s2", "n2", "norm2"),
    ]
    initializers = [
        weight("w1", [c, c]),
        weight("b1", [c]),
        weight("w2", [c, 2 * c]),
        weight("b2", [2 * c]),
        weight("w3", [2 * c, c]),
        weight("b3", [c]),
        numpy_helper.from_array(np.ones(c, np.float32), "gamma"),
        numpy_helper.from_array(np.zeros(c, np.float32), "beta"),
        numpy_helper.from_array(np.array(1.4142135, np.float32), "c_div"),
        numpy_helper.from_array(np.array(1.0, np.float32), "c_add"),
        numpy_helper.from_array(np.array(0.5, np.float32), "c_mul"),
    ]
    graph = helper.make_graph(
        nodes,
        "chains",
        [
            helper.make_tensor_value_info("g", TensorProto.FLOAT, [*shape, c]),
            helper.make_tensor_value_info("r", TensorProto.FLOAT, [*shape, c]),
        ],
        [helper.make_tensor_value_info("n2", TensorProto.FLOAT, [*shape, c])],
        initializers,
    )
    return helper.make_model(graph)


def test_both_chains_become_one_plugin_each():
    lowered = dual_aot.lower(_chain_model(), KEY)

    assert [(p.op, p.inputs, p.outputs) for p in lowered.plugins] == [
        (
            f"chaina_{KEY}",
            ["g", "r", "w1", "b1", "gamma", "beta", "w2", "b2"],
            ["s", "gelu"],
        ),
        (f"chainb_{KEY}", ["gelu", "s", "w3", "b3", "gamma", "beta"], ["s2", "n2"]),
    ]
    chain_a = lowered.plugins[0].attributes
    assert chain_a["eps"] == pytest.approx(1e-5)
    assert (chain_a["c_div"], chain_a["c_add"], chain_a["c_mul"]) == (
        pytest.approx(1.4142135),
        1.0,
        0.5,
    )
    # Every weight a plugin reads has a named reader.
    assert {"w1", "b1", "w2", "b2", "w3", "b3"} <= set(lowered.readers.values())


def test_a_read_norm_statistic_keeps_the_chain_apart():
    model = _chain_model()
    model.graph.output.append(
        helper.make_tensor_value_info("n_sigma", TensorProto.FLOAT, [1, 32, 48, 1])
    )
    lowered = dual_aot.lower(model, KEY)

    assert [p.op for p in lowered.plugins] == [f"norm_{KEY}", f"chainb_{KEY}"]


def test_chains_stay_unfused_in_dynamic_and_other_width_models():
    none = (0, 0, 0)
    assert dual_aot.fuse_chains(dual_aot.lower(_chain_model(), KEY), "c128") == none
    other = "c160_h32_w48"
    empty = dual_aot.Lowered(onnx.ModelProto(), [], {})
    assert dual_aot.fuse_chains(empty, other) == none
    unfused = dual_aot.lower(_chain_model(), KEY, chains=False)
    assert [p.op for p in unfused.plugins] == [f"norm_{KEY}", f"norm_{KEY}"]


@pytest.mark.parametrize("sm", [86, 120])
def test_the_chain_kernels_fit_the_aot_plugins_shared_memory(sm: int):
    from nodes.impl.tensorrt import dual_chain_kernels as chains

    pointers = dict.fromkeys(
        (
            "g_ptr",
            "r_ptr",
            "w1_ptr",
            "b1_ptr",
            "w2_ptr",
            "b2_ptr",
            "res_ptr",
            "out_ptr",
        ),
        "*bf16",
    )
    norm = {"gamma_ptr": "*fp32", "beta_ptr": "*fp32"}
    gelu = {"pixels": 1536, "eps": 1e-5, "c_div": 1.4142135, "c_add": 1.0, "c_mul": 0.5}
    a, c = (
        dual_aot.compile_kernel(
            chains.chain_a,
            {**pointers, **norm},
            gelu | chains.CHAIN_A_LAUNCH | {"project": project},
            sm,
            chains.OPTIONS,
        )
        for project in (False, True)
    )
    b = dual_aot.compile_kernel(
        chains.chain_b,
        {
            **dict.fromkeys(("g_ptr", "r_ptr", "w_ptr", "b_ptr"), "*bf16"),
            **norm,
            **dict.fromkeys(("res_ptr", "y_ptr"), "*bf16"),
        },
        {"pixels": 1536, "eps": 1e-5} | chains.CHAIN_B_LAUNCH,
        sm,
        chains.OPTIONS,
    )
    names = (
        "a",
        "wg",
        "bg",
        "na",
        "wp",
        "bp",
        "r",
        "gamma",
        "beta",
        "wq",
        "res",
        "qkv",
    )
    d = dual_aot.compile_kernel(
        chains.chain_d,
        {f"{n}_ptr": "*fp32" if n in ("gamma", "beta") else "*bf16" for n in names},
        {"pixels": 1536, "eps": 1e-5} | chains.CHAIN_D_LAUNCH,
        sm,
        chains.OPTIONS,
    )
    shared = (a.metadata.shared, b.metadata.shared, c.metadata.shared)
    shared += (d.metadata.shared,)
    assert max(shared) <= dual_aot.MAX_SHARED


@pytest.mark.parametrize("window", [32, 64])
def test_the_square_block_na_kernel_fits_too(window: int):
    kernel = dual_aot.na_kernel(128, window, 120)
    assert kernel.metadata.name == "na_attention_blocks"


def _gate_model() -> onnx.ModelProto:
    """An attention layer's tail as exported: gate projection + bias -> Sigmoid -> x the
    attention output -> projection + bias -> residual add -> Norm -> the next qkv."""
    attrs = {"domain": "trt", "plugin_version": "1", "plugin_namespace": ""}
    c, shape = 128, [1, 32, 48]

    def weight(name: str, dims: list[int]) -> onnx.TensorProto:
        return numpy_helper.from_array(np.full(dims, 0.01, np.float32), name)

    nodes = [
        helper.make_node("MatMul", ["a", "wg"], ["ga"]),
        helper.make_node("Add", ["ga", "bg"], ["gb"]),
        helper.make_node("Sigmoid", ["gb"], ["gs"]),
        helper.make_node("Mul", ["attended", "gs"], ["gm"]),
        helper.make_node("MatMul", ["gm", "wp"], ["pa"]),
        helper.make_node("Add", ["pa", "bp"], ["pb"]),
        helper.make_node("Add", ["r", "pb"], ["s"]),
        helper.make_node(
            f"DualNorm_{KEY}_TRT",
            ["s", "gamma", "beta"],
            ["n", "n_sigma"],
            name="norm",
            residual=0,
            epsilon=1e-5,
            **attrs,
        ),
        helper.make_node("MatMul", ["n", "wq"], ["qkv"]),
    ]
    initializers = [
        weight("wg", [c, c]),
        weight("bg", [c]),
        weight("wp", [c, c]),
        weight("bp", [c]),
        weight("wq", [c, 3 * c]),
        numpy_helper.from_array(np.ones(c, np.float32), "gamma"),
        numpy_helper.from_array(np.zeros(c, np.float32), "beta"),
    ]
    value = TensorProto.FLOAT
    graph = helper.make_graph(
        nodes,
        "gate",
        [
            helper.make_tensor_value_info(name, value, [*shape, c])
            for name in ("a", "attended", "r")
        ],
        [
            helper.make_tensor_value_info("s", value, [*shape, c]),
            helper.make_tensor_value_info("qkv", value, [*shape, 3 * c]),
        ],
        initializers,
    )
    return helper.make_model(graph)


def test_an_attention_tail_becomes_chain_d():
    model = _gate_model()
    # `s` is the graph's output here, as the next residual is in the trunk.
    lowered = dual_aot.lower(model, KEY)

    assert [(p.op, p.inputs, p.outputs) for p in lowered.plugins] == [
        (
            f"chaind_{KEY}",
            ["a", "wg", "bg", "attended", "wp", "bp", "r", "gamma", "beta", "wq"],
            ["s", "qkv"],
        )
    ]
    assert lowered.plugins[0].attributes["eps"] == pytest.approx(1e-5)
    assert {"wg", "bg", "wp", "bp", "wq"} <= set(lowered.readers.values())


def test_a_gate_read_elsewhere_keeps_chain_d_apart():
    model = _gate_model()
    model.graph.output.append(
        helper.make_tensor_value_info("gs", TensorProto.FLOAT, [1, 32, 48, 128])
    )
    model.graph.node.append(helper.make_node("Identity", ["gs"], ["gs_copy"]))
    lowered = dual_aot.lower(model, KEY)

    assert [p.op for p in lowered.plugins] == [f"norm_{KEY}"]


def test_a_dual_project_s_ffn_becomes_chain_c():
    """The guide's DualProject (core, Wp, residual) -> Norm -> expand + GELU: one
    chain C plugin in the project plugin's place."""
    attrs = {"domain": "trt", "plugin_version": "1", "plugin_namespace": ""}
    c, shape = 128, [1, 32, 48]
    model = _chain_model()
    graph = model.graph
    # s comes from a DualProject instead of the projection + bias + residual add
    nodes = [n for n in graph.node if n.output[0] not in ("p", "pb", "s")]
    project = helper.make_node(
        f"DualProject_{KEY}_TRT", ["g", "wp", "r"], ["s"], name="project", **attrs
    )
    del graph.node[:]
    graph.node.extend([project, *nodes])
    graph.input.append(
        helper.make_tensor_value_info("wp", TensorProto.BFLOAT16, [c, c])
    )
    lowered = dual_aot.lower(model, KEY)

    chain_c = lowered.plugins[0]
    assert (chain_c.op, chain_c.inputs, chain_c.outputs) == (
        f"chainc_{KEY}",
        ["g", "r", "wp", "wp", "gamma", "beta", "w2", "b2"],
        ["s", "gelu"],
    )
    assert not any(p.op.startswith("project_") for p in lowered.plugins)
    assert [p.op for p in lowered.plugins[1:]] == [f"chainb_{KEY}"]
    assert shape == [1, 32, 48]
