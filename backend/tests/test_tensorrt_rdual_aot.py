"""RDUAL's AOT plugins' graph lowering (nodes/impl/tensorrt/rdual_aot.py); no GPU: the
plugins are registered apart from the lowering."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

# Needs TensorRT and Triton (the engine build's environment).
rdual_aot = pytest.importorskip("nodes.impl.tensorrt.rdual_aot")

H, W = 8, 12
KEY = f"c128_h{H}_w{W}"
TRT = {"domain": "trt", "plugin_version": "1", "plugin_namespace": ""}


def _value(name: str, kind: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, kind, shape)


def _fusion_model(with_norm: bool = False) -> onnx.ModelProto:
    """A key/value warp feeding the temporal attention, as rdual_export emits them,
    optionally after one of DUAL's Norm nodes."""
    n = H * W
    nodes = []
    kv_input = "kv"
    if with_norm:
        nodes.append(
            helper.make_node(
                f"DualNorm_{KEY}_TRT",
                ["x", "gamma", "beta"],
                ["normed", "normed_unused_1"],
                name="norm",
                residual=0,
                epsilon=1e-5,
                **TRT,
            )
        )
        nodes.append(helper.make_node("MatMul", ["normed", "wkv"], ["kv"]))
    nodes += [
        helper.make_node(
            "RDUALFlowKV",
            [kv_input, "coords"],
            ["warped"],
            name="warp",
            height=H,
            width=W,
            **TRT,
        ),
        helper.make_node(
            "RDUALTemporal",
            ["q", "k", "v", "table", "null_key", "null_bias", "slot_bias", "valid"],
            ["attended", "attended_unused_1"],
            name="temporal",
            height=H,
            width=W,
            slots=1,
            radius=3,
            **TRT,
        ),
    ]
    initializers = [
        numpy_helper.from_array(np.zeros((7, 7, 4), np.float32), "table"),
        numpy_helper.from_array(np.zeros((1, 4, 1), np.float32), "null_bias"),
        numpy_helper.from_array(np.zeros((1, 4), np.float32), "slot_bias"),
    ]
    inputs = [
        _value("coords", TensorProto.FLOAT, [1, 2, H, W]),
        _value("q", TensorProto.BFLOAT16, [1, 4, n, 32]),
        _value("k", TensorProto.BFLOAT16, [1, 1, 4, n, 32]),
        _value("v", TensorProto.BFLOAT16, [1, 1, 4, n, 32]),
        _value("null_key", TensorProto.BFLOAT16, [4, 32]),
        _value("valid", TensorProto.BOOL, [1, 1]),
    ]
    if with_norm:
        inputs += [
            _value("x", TensorProto.BFLOAT16, [1, H, W, 128]),
            _value("wkv", TensorProto.BFLOAT16, [128, 256]),
        ]
        initializers += [
            numpy_helper.from_array(np.ones(128, np.float32), "gamma"),
            numpy_helper.from_array(np.zeros(128, np.float32), "beta"),
        ]
    else:
        inputs.append(_value("kv", TensorProto.BFLOAT16, [1, H, W, 256]))
    graph = helper.make_graph(
        nodes,
        "fusion",
        inputs,
        [
            _value("warped", TensorProto.BFLOAT16, [2, 1, 4, n, 32]),
            _value("attended", TensorProto.FLOAT, [1, 4, n, 32]),
        ],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


def _flow_model() -> onnx.ModelProto:
    """The flow part's plugin nodes as rdual_export writes them: the global matching,
    then one correlation lookup (window, then sample)."""
    c, n = 64, H * W
    nodes = [
        helper.make_node(
            "RDUALMatch",
            ["q", "k", "g"],
            ["matched"],
            name="match",
            batch=2,
            n=n,
            channels=c,
            **TRT,
        ),
        helper.make_node(
            "RDUALCorrWindow",
            ["f0", "f1", "flow"],
            ["window"],
            name="window",
            batch=2,
            height=H,
            width=W,
            channels=c,
            half=1,
            **TRT,
        ),
        helper.make_node(
            "RDUALCorrSample",
            ["window", "flow"],
            ["corrs"],
            name="sample",
            batch=2,
            height=H,
            width=W,
            channels=c,
            half=1,
            **TRT,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "flow",
        [
            _value("q", TensorProto.FLOAT, [2, n, c]),
            _value("k", TensorProto.FLOAT, [2, n, c]),
            _value("g", TensorProto.FLOAT, [2, n, 2]),
            _value("f0", TensorProto.FLOAT16, [2, n, c]),
            _value("f1", TensorProto.FLOAT16, [2, n, c]),
            _value("flow", TensorProto.FLOAT, [2, 2, H, W]),
        ],
        [
            _value("matched", TensorProto.FLOAT, [2, n, 2]),
            _value("corrs", TensorProto.FLOAT, [2, 81, H, W]),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


def _shapes(model: onnx.ModelProto) -> dict[str, tuple[int, list[int]]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return {
        value.name: (
            value.type.tensor_type.elem_type,
            [d.dim_value for d in value.type.tensor_type.shape.dim],
        )
        for value in [*inferred.graph.value_info, *inferred.graph.output]
    }


def test_rdual_s_nodes_lower_to_stand_ins_of_their_outputs_shapes():
    lowered = rdual_aot.lower(_fusion_model(), None)

    kinds = [node.op_type for node in lowered.model.graph.node]
    assert not {"RDUALFlowKV", "RDUALTemporal"} & set(kinds)
    shapes = _shapes(lowered.model)
    n = H * W
    assert shapes["warped"] == (TensorProto.BFLOAT16, [2, 1, 4, n, 32])
    assert shapes["attended"] == (TensorProto.FLOAT, [1, 4, n, 32])
    assert shapes["attended_unused_1"] == (TensorProto.FLOAT, [1, 4, n])
    assert [(p.op, p.inputs, p.outputs) for p in lowered.plugins] == [
        (rdual_aot.flow_kv_name(H, W), ["kv", "coords"], ["warped"]),
        (
            rdual_aot.temporal_name(H, W, 1, 3),
            ["q", "k", "v", "table", "null_key", "null_bias", "slot_bias", "valid"],
            ["attended", "attended_unused_1"],
        ),
    ]


def test_every_plugin_input_has_a_named_reader():
    lowered = rdual_aot.lower(_fusion_model(), None)

    needed = {name for plugin in lowered.plugins for name in plugin.inputs}
    assert set(lowered.readers.values()) == needed
    readers = {
        node.name for node in lowered.model.graph.node if node.op_type == "Identity"
    }
    assert set(lowered.readers) <= readers


def test_the_flow_s_plugins_lower_without_a_volume():
    lowered = rdual_aot.lower(_flow_model(), None)

    shapes = _shapes(lowered.model)
    assert shapes["window"] == (TensorProto.FLOAT, [2 * H * W, 128])
    assert shapes["corrs"] == (TensorProto.FLOAT, [2, 81, H, W])
    assert shapes["matched"] == (TensorProto.FLOAT, [2, H * W, 2])
    window, sample = rdual_aot.corr_names(2, H, W, 1)
    assert (window, sample) == (
        f"corr_window_b2_h{H}_w{W}_f16",
        f"corr_sample_b2_h{H}_w{W}_f16",
    )
    assert [(p.op, p.inputs) for p in lowered.plugins] == [
        (rdual_aot.match_name(2, H * W, 64), ["q", "k", "g"]),
        (window, ["f0", "f1", "flow"]),
        (sample, ["window", "flow"]),
    ]
    kinds = {node.op_type for node in lowered.model.graph.node}
    assert not kinds & set(rdual_aot.NODES)


def test_with_a_key_dual_s_plugins_lower_too_in_graph_order():
    lowered = rdual_aot.lower(_fusion_model(with_norm=True), KEY)

    kinds = {node.op_type for node in lowered.model.graph.node}
    assert not any(kind.startswith(("Dual", "RDUAL")) for kind in kinds)
    assert [p.op for p in lowered.plugins] == [
        f"norm_{KEY}",
        rdual_aot.flow_kv_name(H, W),
        rdual_aot.temporal_name(H, W, 1, 3),
    ]
