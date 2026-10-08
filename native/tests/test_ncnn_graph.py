"""Exact NCNN graph differential tests against the frozen installed source.

The optimizer oracle is the frozen optimizer.py with the CORRECTIONS of
reference_ncnn/generate_optimizer_cpp.py, the source the native passes translate.
These tests do not import or execute the NCNN inference engine.
"""

from __future__ import annotations

import copy
import importlib
import io
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nodes.impl.ncnn import model as port
from nodes.impl.ncnn.optimizer import NcnnOptimizer

PACKAGE = "nodes.impl._reference_ncnn"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).with_name("reference_ncnn"))]
sys.modules[PACKAGE] = package
reference = importlib.import_module(PACKAGE + ".model")
FrozenOptimizer = importlib.import_module(PACKAGE + ".optimizer").NcnnOptimizer
generator = importlib.import_module(PACKAGE + ".generate_optimizer_cpp")
corrected = types.ModuleType(PACKAGE + ".corrected_optimizer")
corrected.__package__ = PACKAGE
exec(compile(generator.corrected_source(), generator.SOURCE, "exec"), vars(corrected))
ReferenceOptimizer = corrected.NcnnOptimizer


def snapshot(value, seen=None):
    """Include aliases, insertion order, exact dtype/bits, and failure mutations."""
    if seen is None:
        seen = {}
    if isinstance(value, (str, bytes, int, float, bool, type(None), np.generic)):
        if isinstance(value, (float, np.generic)):
            return (
                type(value).__name__,
                np.asarray(value).dtype.str,
                np.asarray(value).tobytes(),
            )
        return value
    if id(value) in seen:
        return ("alias", seen[id(value)])
    seen[id(value)] = len(seen)
    if isinstance(value, np.ndarray):
        return (
            "array",
            value.dtype.str,
            value.shape,
            value.strides,
            value.flags.writeable,
            value.tobytes(),
        )
    if isinstance(value, dict):
        return (
            "dict",
            [(snapshot(k, seen), snapshot(v, seen)) for k, v in value.items()],
        )
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, [snapshot(v, seen) for v in value])
    return (type(value).__name__, snapshot(vars(value), seen))


def outcome(call):
    try:
        return ("ok", call())
    except Exception as error:
        return (type(error).__name__, str(error))


def build(module, specification):
    model = module.NcnnModel(len(specification), sum(len(s[3]) for s in specification))
    for op, name, inputs, outputs, params, weights in specification:
        layer = module.NcnnLayer(
            op, name, len(inputs), len(outputs), list(inputs), list(outputs)
        )
        for key, value in params.items():
            layer.params[key] = copy.deepcopy(value)
        for key, value in weights.items():
            array = value[0].copy(order="K")
            layer.weight_data[key] = module.NcnnWeight(array, value[1])
        model.layers.append(layer)
    return model


def spec(
    op,
    name,
    inputs,
    outputs,
    params: dict[int, Any] | None = None,
    weights: dict[str, tuple[np.ndarray, bytes]] | None = None,
):
    return (op, name, inputs, outputs, params or {}, weights or {})


def weights(dtype, shape=(2, 2, 1, 1), seed=14):
    rng = np.random.default_rng(seed)
    return {
        "weight": (
            rng.uniform(-2, 2, shape).astype(dtype),
            port.DTYPE_FP16 if dtype == np.float16 else port.DTYPE_FP32,
        ),
        "bias": (rng.uniform(-1, 1, shape[0]).astype(np.float32), b""),
    }


def conv(dtype=np.float32, name="conv", inputs=("input",), outputs=("a",)):
    return spec(
        "Convolution", name, inputs, outputs, {0: 2, 1: 1, 5: 1, 6: 4}, weights(dtype)
    )


def bn(name="bn", inputs=("a",), outputs=("b",)):
    values = {
        "slope": [0.3, 1.7],
        "mean": [-0.2, 0.9],
        "variance": [0.7, 1.1],
        "bias": [0.1, -0.4],
    }
    return spec(
        "BatchNorm",
        name,
        inputs,
        outputs,
        {0: 2, 1: 0.001},
        {k: (np.array(v, np.float32), b"") for k, v in values.items()},
    )


def cases():
    memory = spec(
        "MemoryData",
        "data",
        (),
        ("data",),
        {0: 2},
        {"data": (np.array([0.3, 1.7], np.float32), b"")},
    )
    inner = spec(
        "InnerProduct",
        "ip",
        ("input",),
        ("a",),
        {0: 2, 1: 1, 2: 4},
        weights(np.float32, (2, 2)),
    )
    source = spec("Input", "input", (), ("a",))
    pooling = spec("Pooling", "pool", ("input",), ("a",), {4: 1})
    base = {
        "fuse_batchnorm_scale": [
            bn(),
            spec(
                "Scale",
                "scale",
                ("b",),
                ("out",),
                {0: 2, 1: 1},
                {
                    "scale": (np.array([0.3, 0.7], np.float32), b""),
                    "bias": (np.array([0.2, -0.1], np.float32), b""),
                },
            ),
        ],
        "fuse_x_batchnorm": [conv(), bn()],
        "fuse_x_mul": [
            memory,
            conv(),
            spec("BinaryOp", "mul", ("a", "data"), ("out",), {0: 2}),
        ],
        "fuse_x_add": [
            memory,
            conv(),
            spec("BinaryOp", "add", ("a", "data"), ("out",), {0: 0}),
        ],
        "fuse_innerproduct_dropout": [
            inner,
            spec("Dropout", "drop", ("a",), ("out",), {0: 0.7}),
        ],
        "fuse_x_activation": [conv(), spec("ReLU", "relu", ("a",), ("out",), {0: 0.1})],
        "fuse_memorydata_binaryop": [
            spec(
                "MemoryData",
                "data",
                (),
                ("a",),
                {0: 1},
                {"data": (np.array([0.5], np.float32), b"")},
            ),
            spec("BinaryOp", "add", ("a", "input"), ("out",), {0: 0}),
        ],
        "fuse_binaryop_eltwise": [
            spec("BinaryOp", "mul0", ("input0",), ("a",), {0: 2, 1: 1, 2: 0.3}),
            spec("BinaryOp", "mul1", ("input1",), ("b",), {0: 2, 1: 1, 2: 0.7}),
            spec("BinaryOp", "add", ("a", "b"), ("out",), {0: 0}),
        ],
        "eliminate_dropout": [
            source,
            spec("Dropout", "drop", ("a",), ("out",), {0: 1.0}),
        ],
        "eliminate_pooling1x1": [
            source,
            spec("Pooling", "pool", ("a",), ("out",), {1: 1, 2: 1}),
        ],
        "eliminate_noop": [source, spec("Noop", "noop", ("a",), ("out",))],
        "eliminate_split": [
            source,
            spec("Split", "split", ("a",), ("b", "c")),
            spec("ReLU", "relu", ("b",), ("out",)),
        ],
        "eliminate_flatten_after_global_pooling": [
            pooling,
            spec("Flatten", "flat", ("a",), ("out",)),
        ],
        "eliminate_reshape_after_global_pooling": [
            pooling,
            spec("Reshape", "shape", ("a",), ("out",), {0: -1, 1: -233, 2: -233, 3: 0}),
        ],
        "eliminate_reshape_before_binaryop": [
            spec("Reshape", "shape", ("input",), ("a",), {0: 1, 1: 1, 3: 1}),
            spec("BinaryOp", "add", ("a", "input1"), ("out",)),
        ],
        "replace_reduction_with_global_pooling": [
            spec("Reduction", "r0", ("input",), ("a",), {0: 3, 1: 0, 2: 1.0, 3: [2]}),
            spec("Reduction", "r1", ("a",), ("out",), {0: 3, 1: 0, 2: 1.0, 3: [2]}),
        ],
        "replace_prelu_with_leaky_relu": [
            spec(
                "PReLU",
                "prelu",
                ("input",),
                ("out",),
                {0: 1},
                {"slope": (np.array([0.1], np.float64), b"")},
            )
        ],
        "replace_convolution_with_innerproduct_after_global_pooling": [
            pooling,
            conv(inputs=("a",), outputs=("out",)),
        ],
        "replace_convolution_with_innerproduct_after_innerproduct": [
            inner,
            conv(inputs=("a",), outputs=("out",)),
        ],
        "eliminate_flatten_after_innerproduct": [
            inner,
            spec("Flatten", "flat", ("a",), ("out",)),
        ],
        "eliminate_orphaned_memorydata": [memory, source],
    }
    return base


CASES = cases()
MUTATIONS = [
    "none",
    "empty",
    "unmatched",
    "reverse",
    "missing_outputs",
    "missing_inputs",
]


def mutated(name, mutate):
    specification = copy.deepcopy(CASES[name])
    if mutate == "empty":
        specification = []
    elif mutate == "reverse":
        specification.reverse()
    elif mutate == "unmatched":
        specification = [
            spec(s[0], s[1], ["unknown"] * len(s[2]), s[3], s[4], s[5])
            for s in specification
        ]
    elif mutate == "missing_outputs":
        specification = [spec(s[0], s[1], s[2], [], s[4], s[5]) for s in specification]
    elif mutate == "missing_inputs":
        specification = [spec(s[0], s[1], [], s[3], s[4], s[5]) for s in specification]
    return specification


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("mutate", MUTATIONS)
def test_every_pass_exact_state_and_errors(name, mutate):
    specification = mutated(name, mutate)
    a, b = build(reference, specification), build(port, specification)
    ra = outcome(getattr(ReferenceOptimizer(a), "_NcnnOptimizer__" + name))
    rb = outcome(getattr(NcnnOptimizer(b), "_NcnnOptimizer__" + name))
    assert ra == rb
    assert snapshot(a) == snapshot(b)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("seed", range(12))
def test_optimizer_complete_pipeline(dtype, seed):
    cv = list(conv(dtype))
    cv[5] = weights(dtype, seed=seed)
    specification = [tuple(cv), bn(), spec("ReLU", "relu", ("b",), ("out",), {0: 0.25})]
    a, b = build(reference, specification), build(port, specification)
    ra = outcome(lambda: ReferenceOptimizer(a).optimize())
    rb = outcome(lambda: NcnnOptimizer(b).optimize())
    assert ra == rb
    assert snapshot(a) == snapshot(b)
    assert a.write_param() == b.write_param()
    assert a.serialize_weights() == b.serialize_weights()


@pytest.mark.parametrize(
    "op",
    [
        "Input",
        "Convolution",
        "Deconvolution",
        "ConvolutionDepthWise",
        "InnerProduct",
        "BatchNorm",
        "Scale",
        "PReLU",
        "Interp",
        "PixelShuffle",
        "Reshape",
    ],
)
@pytest.mark.parametrize(
    "suffix",
    [
        "",
        " 0=1",
        " 0=1.0",
        " 0=1e1",
        " 0=1E1",
        " -23300=1,2.0,3",
        " 0=999999999999999999999999999999",
        " 0=1 0=2",
        " 0=x",
        " 999=1",
        " 0=1=2",
    ],
)
def test_layer_parser(op, suffix):
    line = f"\t{op}\u2003naïve猫 1 1 in out{suffix}\n"
    a, b = reference.NcnnModel(), port.NcnnModel()
    ra, rb = (
        outcome(lambda: a.parse_param_layer(line)),
        outcome(lambda: b.parse_param_layer(line)),
    )
    if ra[0] == "ok":
        assert rb[0] == "ok"
        assert snapshot(ra[1]) == snapshot(rb[1])
    else:
        assert ra == rb


@pytest.mark.parametrize(
    "line",
    [
        "",
        "x",
        "MemoryData x 0 1 out",
        "Input x",
        "Input x 0",
        "Input x xx 1 out",
        "Unknown x 0 1 out",
        "Input x -1 0",
        "Input x 0 -1",
        "Input x 0 0 0",
        "Input x 0 0 =1",
    ],
)
def test_parser_errors(line):
    ra = outcome(lambda: reference.NcnnModel().parse_param_layer(line))
    rb = outcome(lambda: port.NcnnModel().parse_param_layer(line))
    if ra[0] == "ok":
        assert rb[0] == "ok"
        assert snapshot(ra[1]) == snapshot(rb[1])
    else:
        assert ra == rb


WEIGHT_LINES = [
    "BatchNorm bn 1 1 in out 0=2",
    "Convolution conv 1 1 in out 0=2 1=1 5=1 6=4",
    "ConvolutionDepthWise conv 1 1 in out 0=2 1=1 5=1 6=2 7=2",
    "Deconvolution conv 1 1 in out 0=2 1=1 5=1 6=4",
    "InnerProduct ip 1 1 in out 0=2 1=1 2=4",
    "PReLU prelu 1 1 in out 0=2",
    "Scale scale 1 1 in out 0=2 1=1",
    "Scale scale 1 1 in out 0=-233",
    "Input input 0 1 out",
]


@pytest.mark.parametrize("line", WEIGHT_LINES)
@pytest.mark.parametrize("tag", [port.DTYPE_FP32, port.DTYPE_FP16, b"FAIL"])
@pytest.mark.parametrize("length", [0, 1, 3, 4, 6, 8, 12, 16, 24, 32, 64])
def test_weight_decoder_and_position(line, tag, length):
    binary = (tag + np.arange(20, dtype=np.float32).tobytes())[:length]
    a, b = reference.NcnnModel(), port.NcnnModel()
    oa, la = a.parse_param_layer(line)
    ob, lb = b.parse_param_layer(line)
    sa, sb = io.BytesIO(binary), io.BytesIO(binary)
    ra = outcome(lambda: a.load_layer_weights(sa, oa, la))
    rb = outcome(lambda: b.load_layer_weights(sb, ob, lb))
    assert sa.tell() == sb.tell()
    if ra[0] == "ok":
        assert rb[0] == "ok"
        assert snapshot(ra[1]) == snapshot(rb[1])
    else:
        assert ra == rb


@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.int32, np.complex64]
)
@pytest.mark.parametrize("tag", [b"", port.DTYPE_FP16, port.DTYPE_FP32, b"tag"])
@pytest.mark.parametrize("layout", ["C", "F", "reverse", "readonly"])
def test_add_weight_cast(dtype, tag, layout):
    data = np.arange(12, dtype=np.float64).reshape(3, 4).astype(dtype)
    if layout == "F":
        data = np.asfortranarray(data)
    elif layout == "reverse":
        data = data[::-1, ::-1]
    elif layout == "readonly":
        data.flags.writeable = False
    saved = data.tobytes()
    a, b = reference.NcnnLayer("Convolution"), port.NcnnLayer("Convolution")
    with np.errstate(all="ignore"):
        ra = outcome(lambda: a.add_weight("weight", data, tag))
        rb = outcome(lambda: b.add_weight("weight", data, tag))
    assert ra == rb
    assert snapshot(a) == snapshot(b)
    assert data.tobytes() == saved


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        -0.0,
        0.123456789,
        1e-20,
        float("nan"),
        float("inf"),
        [1, 2.3456789],
        10**100,
    ],
)
def test_param_defaults_sorting_and_formats(value):
    a, b = (
        reference.NcnnParamCollection("Convolution"),
        port.NcnnParamCollection("Convolution"),
    )
    for obj in (a, b):
        obj[11] = 3
        obj[1] = 3
        obj[18] = value
        obj[0] = 2
    ra, rb = outcome(lambda: str(a)), outcome(lambda: str(b))
    assert ra == rb
    assert snapshot(a) == snapshot(b)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_model_file_roundtrip_and_concurrent_reads(dtype, tmp_path):
    model = build(reference, [conv(dtype)])
    param, binary = tmp_path / "猫.param", tmp_path / "猫.bin"
    model.write_param(param)
    model.write_bin(binary)
    expected = reference.NcnnModel.load_from_file(str(param), str(binary))

    def run(_):
        actual = port.NcnnModel.load_from_file(str(param), str(binary))
        assert snapshot(actual) == snapshot(expected)
        assert actual.write_param() == expected.write_param()
        assert actual.serialize_weights() == expected.serialize_weights()
        assert port.NcnnModelWrapper.get_broadcast_data(
            actual
        ) == reference.NcnnModelWrapper.get_broadcast_data(expected)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(16)))


def test_source_oracle_is_independent():
    assert reference.__file__ is not None
    assert "native_ncnn_graph" not in Path(reference.__file__).read_text()
    assert len(CASES) == 21


def branch_variants():
    variants = []
    for name, specifications in CASES.items():
        for index, layer in enumerate(specifications):
            for key in reference.param_schema[layer[0]]:
                if key == "weightOrder":
                    continue
                for value in (0, 1, 2, -233, 0.5):
                    changed = copy.deepcopy(specifications)
                    changed[index][4][int(key)] = value
                    variants.append(
                        pytest.param(name, changed, id=f"{name}-{index}-{key}-{value}")
                    )
    return variants


@pytest.mark.parametrize("name,specification", branch_variants())
def test_pass_parameter_branches(name, specification):
    a, b = build(reference, specification), build(port, specification)
    ra = outcome(getattr(ReferenceOptimizer(a), "_NcnnOptimizer__" + name))
    rb = outcome(getattr(NcnnOptimizer(b), "_NcnnOptimizer__" + name))
    assert ra == rb
    assert snapshot(a) == snapshot(b)


@pytest.mark.parametrize("name", CASES)
def test_pass_aliases_and_repeat(name, monkeypatch):
    a, b = build(reference, CASES[name]), build(port, CASES[name])
    for model in (a, b):
        if model.layers:
            monkeypatch.setattr(
                model.layers[0],
                "custom_metadata",
                {"untouched": ["猫", 10**100]},
                raising=False,
            )
            monkeypatch.setattr(
                model,
                "saved_aliases",
                [
                    model.layers[0],
                    model.layers[0].inputs,
                    model.layers[0].outputs,
                    model.layers[0].params.param_dict,
                ],
                raising=False,
            )
    for _ in range(2):
        ra = outcome(getattr(ReferenceOptimizer(a), "_NcnnOptimizer__" + name))
        rb = outcome(getattr(NcnnOptimizer(b), "_NcnnOptimizer__" + name))
        assert ra == rb
        assert snapshot(a) == snapshot(b)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize(
    "special", [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1e30, -1e30]
)
@pytest.mark.parametrize("policy", ["ignore", "raise"])
def test_optimizer_numerical_edges(dtype, special, policy):
    specification = [conv(dtype), bn()]
    with np.errstate(all="ignore"):
        specification[0][5]["weight"][0].flat[0] = special
        specification[1][5]["variance"][0].flat[0] = special
    a, b = build(reference, specification), build(port, specification)
    with np.errstate(all=policy):
        ra = outcome(lambda: ReferenceOptimizer(a).optimize())
        rb = outcome(lambda: NcnnOptimizer(b).optimize())
    assert ra == rb
    assert snapshot(a) == snapshot(b)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("layout", ["reverse", "F", "readonly"])
def test_optimizer_foreign_weights(dtype, layout):
    a, b = build(reference, [conv(dtype), bn()]), build(port, [conv(dtype), bn()])
    for model in (a, b):
        for layer in model.layers:
            for weight in layer.weight_data.values():
                if layout == "reverse":
                    weight.weight = weight.weight[::-1]
                elif layout == "F":
                    weight.weight = np.asfortranarray(weight.weight)
                else:
                    weight.weight.flags.writeable = False
    ra = outcome(lambda: ReferenceOptimizer(a).optimize())
    rb = outcome(lambda: NcnnOptimizer(b).optimize())
    assert ra == rb
    assert snapshot(a) == snapshot(b)


@pytest.mark.parametrize("op", ["ConvolutionDepthWise", "InnerProduct"])
def test_original_failed_batchnorm_fusion_retains_inserted_bias(op):
    if op == "InnerProduct":
        first = spec(
            op,
            "conv",
            ("input",),
            ("a",),
            {0: 2, 1: 0, 2: 4},
            weights(np.float32, (2, 2)),
        )
    else:
        first = spec(
            op,
            "conv",
            ("input",),
            ("a",),
            {0: 2, 1: 1, 5: 0, 6: 4, 7: 1},
            weights(np.float32, (1, 2, 2, 1, 1)),
        )
    del first[5]["bias"]
    a, b = build(reference, [first, bn()]), build(port, [first, bn()])
    ra = outcome(lambda: ReferenceOptimizer(a).optimize())
    rb = outcome(lambda: NcnnOptimizer(b).optimize())
    assert ra == rb
    assert ra[0] == "ValueError"
    assert (
        "axes don't match array" in ra[1]
        if op == "InnerProduct"
        else "remapped shapes" in ra[1]
    )
    assert "bias" in a.layers[0].weight_data
    assert snapshot(a) == snapshot(b)


@pytest.mark.parametrize("name", ["memorydata", "other_layer", "unmatched"])
def test_memorydata_split_fusion(name):
    """MemoryData - Split - BinaryOp fusion starts from MemoryData layers only, as in
    ncnnoptimize. Upstream's inverted test started from every other layer: this
    "data"-carrying Input fused, and one without the weight raised KeyError."""
    data = spec(
        "Input" if name == "other_layer" else "MemoryData",
        "data",
        (),
        ("a",),
        {0: 1, 1: 0, 2: 0},
        {"data": (np.array([0.3], np.float32), b"")},
    )
    split = spec("Split", "split", ("a",), ("b", "c"))
    binary = spec(
        "BinaryOp",
        "bin",
        ("b" if name != "unmatched" else "missing", "other"),
        ("out",),
        {0: 0},
    )
    a, b = build(reference, [data, split, binary]), build(port, [data, split, binary])
    before = snapshot(b)
    method = "_NcnnOptimizer__fuse_memorydata_binaryop"
    ra = outcome(getattr(ReferenceOptimizer(a), method))
    rb = outcome(getattr(NcnnOptimizer(b), method))
    assert ra == rb == ("ok", None)
    assert snapshot(a) == snapshot(b)
    if name == "memorydata":
        assert b.layers[2].inputs == ["other"]
        assert b.layers[2].params[1].value == 1
        assert b.layers[1].outputs == ["c"]
    else:
        assert snapshot(b) == before


def test_corrected_oracle_departs_from_upstream_only_as_recorded():
    """The pass corpus cases where the corrected oracle's result differs from the
    frozen one's: the MemoryData fusion changes none."""
    differ = []
    runs = [(n, m, mutated(n, m)) for n in CASES for m in MUTATIONS]
    runs += [(p.values[0], p.id, p.values[1]) for p in branch_variants()]
    for name, label, specification in runs:
        results = []
        for optimizer in (FrozenOptimizer, ReferenceOptimizer):
            model = build(reference, specification)
            result = outcome(getattr(optimizer(model), "_NcnnOptimizer__" + name))
            results.append((result, snapshot(model)))
        if results[0] != results[1]:
            differ.append((name, label))
    assert differ == []


def test_generator_reproduces_the_committed_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "OUTPUT", tmp_path / "ncnn_optimizer_passes.hpp")
    generator.main()
    committed = Path(__file__).parents[1] / "include" / "ncnn_optimizer_passes.hpp"
    assert generator.OUTPUT.read_bytes() == committed.read_bytes().replace(
        b"\r\n", b"\n"
    )


@pytest.mark.parametrize(
    "data",
    [
        0.25,
        3,
        True,
        np.float16(0.25),
        np.float32(0.25),
        np.float64(0.25),
        np.int64(3),
        [1.0],
        None,
        "1",
    ],
)
@pytest.mark.parametrize("tag", [port.DTYPE_FP16, port.DTYPE_FP32])
def test_add_weight_scalar_and_rejected_contracts(data, tag):
    a, b = reference.NcnnLayer("Convolution"), port.NcnnLayer("Convolution")
    ra, rb = (
        outcome(lambda: a.add_weight("weight", data, tag)),
        outcome(lambda: b.add_weight("weight", data, tag)),
    )
    assert ra == rb
    assert snapshot(a) == snapshot(b)
    ma, mb = reference.NcnnModel(), port.NcnnModel()
    ma.layers, mb.layers = [a], [b]
    assert outcome(ma.serialize_weights) == outcome(mb.serialize_weights)


@pytest.mark.parametrize("dtype", [">f2", ">f4", ">f8", ">i4", ">c8"])
@pytest.mark.parametrize("tag", [port.DTYPE_FP16, port.DTYPE_FP32])
def test_add_weight_foreign_endian(dtype, tag):
    data = np.array([0, 1, -3, 17], dtype=dtype)[::-1]
    a, b = reference.NcnnLayer("Convolution"), port.NcnnLayer("Convolution")
    assert outcome(lambda: a.add_weight("weight", data, tag)) == outcome(
        lambda: b.add_weight("weight", data, tag)
    )
    assert snapshot(a) == snapshot(b)


def exception_tree(call):
    try:
        call()
    except Exception as error:

        def tree(e, depth=0):
            if e is None or depth == 4:
                return None
            return (
                type(e).__name__,
                str(e),
                e.__suppress_context__,
                tree(e.__cause__, depth + 1),
                tree(e.__context__, depth + 1),
            )

        return tree(error)
    return None


@pytest.mark.parametrize("key", [-1, 999, "unknown"])
def test_missing_parameter_exception_context(key):
    a, b = (
        reference.NcnnParamCollection("Convolution"),
        port.NcnnParamCollection("Convolution"),
    )
    assert exception_tree(lambda: a[key]) == exception_tree(lambda: b[key])


def test_native_graph_does_not_call_numpy_weight_algorithms(monkeypatch):
    a = build(port, [conv(), bn()])

    class NoNumericCall:
        def __init__(self, original):
            self.resolve_dtypes = original.resolve_dtypes

        def __call__(self, *args, **kwargs):
            raise AssertionError("NumPy numeric algorithm called")

    for name in ("sqrt", "add", "subtract", "multiply", "divide"):
        monkeypatch.setattr(np, name, NoNumericCall(getattr(np, name)))
    NcnnOptimizer(a).optimize()
    assert a.layers[1].op_type == "ncnnfused"


def test_parameter_missing_alias_preserves_explicit_cause(monkeypatch):
    for module in (reference, port):
        monkeypatch.setitem(
            module.param_schema["Convolution"]["11"], "defaultValue", "missing_alias"
        )
    a, b = (
        reference.NcnnParamCollection("Convolution"),
        port.NcnnParamCollection("Convolution"),
    )
    assert exception_tree(lambda: a[11]) == exception_tree(lambda: b[11])


@pytest.mark.parametrize(
    "text",
    [
        "",
        "magic\n",
        "ignored\n1  2\n",
        "ignored\n1\n",
        "ignored\n-1 -2\n",
        "ignored\n999 999\nInput input 0 1 output\n",
        "ignored\n0 0\n\n",
        "ignored\n1 1\nMemoryData x 0 1 output\n",
        "ignored\n1 1\nConvolution c 1 1 input output 0=1 1=1 6=1\n",
    ],
)
def test_file_parser_headers_errors_and_ignored_magic(text, tmp_path):
    param, binary = tmp_path / "header.param", tmp_path / "header.bin"
    param.write_text(text, encoding="utf-8")
    binary.write_bytes(b"")
    ra = outcome(lambda: reference.NcnnModel.load_from_file(str(param), str(binary)))
    rb = outcome(lambda: port.NcnnModel.load_from_file(str(param), str(binary)))
    if ra[0] == "ok":
        assert rb[0] == "ok"
        assert snapshot(ra[1]) == snapshot(rb[1])
    else:
        assert ra == rb


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("factor", [1, 2, 3])
@pytest.mark.parametrize("before", [True, False])
def test_model_broadcast_metadata_order_and_errors(stride, factor, before):
    convolution = conv()
    convolution[4][3] = stride
    resize = spec(
        "Interp", "resize", ("a",), ("b",), {1: float(factor), 2: float(factor)}
    )
    specifications = [resize, convolution] if before else [convolution, resize]
    a, b = build(reference, specifications), build(port, specifications)
    assert (
        outcome(lambda: reference.NcnnModelWrapper(a))[:1]
        == outcome(lambda: port.NcnnModelWrapper(b))[:1]
    )
    ra = outcome(lambda: reference.NcnnModelWrapper.get_broadcast_data(a))
    rb = outcome(lambda: port.NcnnModelWrapper.get_broadcast_data(b))
    assert ra == rb
