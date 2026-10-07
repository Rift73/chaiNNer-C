"""Exact native graph decisions against frozen installed ONNX helpers."""

from __future__ import annotations

import copy
import itertools
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from bounded import run_python
from onnx import TensorProto as T
from onnx import helper
from reference_onnx_graph import load as old_load
from reference_onnx_graph import update_model_dims as old_dims
from reference_onnx_graph import utils as old

from nodes.impl.native_graph import graph
from nodes.impl.onnx import load as current_load
from nodes.impl.onnx import update_model_dims as current_dims
from nodes.impl.onnx import utils as current


def model(shape=(1, 3, "h", "w"), dtype=T.FLOAT, output_shape=None):
    output_shape = shape if output_shape is None else output_shape
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["in"], ["out"])],
            "test",
            [helper.make_tensor_value_info("in", dtype, shape)],
            [helper.make_tensor_value_info("out", dtype, output_shape)],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
    )


def outcome(call):
    try:
        return (None, call())
    except Exception as error:
        return (type(error), str(error))


@pytest.mark.parametrize(
    "second,fourth",
    itertools.product(
        [0, -1, 1, 3, 4, 5, 2**70, "dynamic", None, np.int64(3), True], repeat=2
    ),
)
@pytest.mark.parametrize("middle", [1, 5, "height", None])
def test_parse_shape(second, fourth, middle):
    source = [1, second, middle, fourth]
    assert outcome(lambda: current.parse_onnx_shape(source)) == outcome(
        lambda: old.parse_onnx_shape(source)  # pyright: ignore[reportArgumentType] -- upstream's annotation says a 4-tuple, but its callers pass ORT's list shapes; the test pins both functions on lists
    )


@pytest.mark.parametrize(
    "source", [[], [1], [1, 3], [1, 7, 9], [1, 3, 9, 11, 17], (1, "a", "b", 3)]
)
def test_parse_shape_errors(source):
    assert outcome(lambda: current.parse_onnx_shape(source)) == outcome(
        lambda: old.parse_onnx_shape(source)
    )


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (1,),
        (1, 2, 3),
        (1, 3, 5, 7),
        (1, "c", "h", "w"),
        (1, None, 0, ""),
        (1, 2, 3, 4, 5),
    ],
)
def test_tensor_shape(shape):
    value = helper.make_tensor_value_info("x", T.FLOAT, shape).type.tensor_type
    assert outcome(lambda: current.to_onnx_tensor_shape(value)) == outcome(
        lambda: old.to_onnx_tensor_shape(value)
    )


@pytest.mark.parametrize(
    "dtype",
    [
        T.FLOAT,
        T.FLOAT16,
        T.DOUBLE,
        T.BFLOAT16,
        T.INT8,
        T.INT64,
        T.BOOL,
        T.STRING,
        T.COMPLEX64,
    ],
)
@pytest.mark.parametrize("kind", ["normal", "output_only", "empty", "sequence"])
def test_model_classification(dtype, kind):
    source = model(dtype=dtype)
    if kind in ("output_only", "empty"):
        source.graph.ClearField("input")
    if kind == "empty":
        source.graph.ClearField("output")
    if kind == "sequence":
        source.graph.input[0].type.CopyFrom(
            helper.make_sequence_type_proto(helper.make_tensor_type_proto(dtype, [3]))
        )
    assert current.is_image_to_image(source) == old.is_image_to_image(source)
    assert current.get_tensor_fp_datatype(source) == old.get_tensor_fp_datatype(source)
    assert current.get_opset(source) == old.get_opset(source)


@pytest.mark.parametrize(
    "imports", [[], [("custom", 2)], [("custom", 2), ("", 14)], [("", 8), ("", 19)]]
)
def test_opset_order(imports):
    source = model()
    source.ClearField("opset_import")
    source.opset_import.extend(
        helper.make_opsetid(domain, version) for domain, version in imports
    )
    assert current.get_opset(source) == old.get_opset(source)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 3, "h", "w"),
        (1, "h", "w", 3),
        (1, 3, 5, 7),
        (1, 7, 5, 3),
        ("b", "c", "h", "w"),
        (1, 0, "h", "w"),
    ],
)
@pytest.mark.parametrize("size", [(16, 16), (7, 11), (1, 1), (0, 8), (-1, 16)])
def test_shape_inference_mutations(shape, size):
    a, b = model(shape), model(shape)
    original, actual = old.ModelShapeInference(a), current.ModelShapeInference(b)
    for attr in (
        "input_shape",
        "output_shape",
        "tensor_format",
        "input_channels",
        "fixed_input_width",
        "fixed_input_height",
        "output_channels",
    ):
        assert getattr(actual, attr) == getattr(original, attr)
    assert outcome(lambda: actual.infer_shape(size)) == outcome(
        lambda: original.infer_shape(size)
    )
    assert b.SerializeToString() == a.SerializeToString()


@pytest.mark.parametrize(
    "case",
    [
        "fixed",
        "symbol",
        "negative",
        "duplicate",
        "missing_input",
        "missing_output",
        "too_many",
        "float",
        "none",
        "huge",
        "unknown_field",
    ],
)
def test_dimension_update_error_state(case):
    source = model()
    inputs: dict[str, Sequence[str | int]] = {"in": [1, 3, 13, 17]}
    outputs: dict[str, Sequence[str | int]] = {"out": [1, 3, "h", "w"]}
    if case == "symbol":
        inputs["in"] = [1, 3, "new_h", "new_w"]
    elif case == "negative":
        inputs["in"] = [1, 3, -1, -1]
    elif case == "duplicate":
        source.graph.value_info.append(
            helper.make_tensor_value_info("extra", T.FLOAT, ["in_2"])
        )
        inputs["in"] = [1, 3, -1, 17]
    elif case == "missing_input":
        inputs = {}
    elif case == "missing_output":
        outputs = {}
    elif case == "too_many":
        outputs["out"] = [1, 3, "h", "w", 3]
    elif case == "float":
        inputs["in"] = [1, 3, 2.5, 17]  # pyright: ignore[reportArgumentType] -- deliberate: a float dim pins the error path against upstream
    elif case == "none":
        inputs["in"] = [1, 3, None, 17]  # pyright: ignore[reportArgumentType] -- deliberate: a None dim pins the error path against upstream
    elif case == "huge":
        inputs["in"] = [1, 3, 2**80, 17]
    elif case == "unknown_field":
        # Valid unknown protobuf field100 varint9 survives native object edits.
        source.ParseFromString(source.SerializeToString() + b"\xa0\x06\x09")
    a, b = copy.deepcopy(source), copy.deepcopy(source)
    expected = outcome(lambda: old_dims.update_inputs_outputs_dims(a, inputs, outputs))
    actual = outcome(
        lambda: current_dims.update_inputs_outputs_dims(b, inputs, outputs)
    )
    if expected[0] is None:
        assert actual[0] is None and actual[1] is b
    else:
        assert actual == expected
    assert a.SerializeToString() == b.SerializeToString()


PATTERNS = [
    b"1959-1960-1961-1962-1963-1964-1965",
    b"1808-1827-1828-2296-1831-1850-1958",
    b"/stage1/rebnconvin/conv_s1/Conv-/stage1/rebnconvin/relu_s1/Relu",
    b"output-d1-Concat_1876-Concat_1896-Concat_1916-Concat_1936-Concat_1956",
]


def classify_reference(data):
    if (
        old_load.U2NET_STANDARD.search(data[-1000:])
        or old_load.U2NET_SILUETA.search(data[-600:])
        or old_load.U2NET_ISNET.search(data[:10000])
    ):
        return 1
    if old_load.U2NET_CLOTH.search(data[-1000:]):
        return 2
    return 0


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("gap", [b"", b"x", b"\x00", b"\n", b"\xff", b" " * 100])
@pytest.mark.parametrize("padding", [0, 599, 600, 999, 1000, 9999, 10000])
def test_rembg_byte_signatures(pattern, gap, padding):
    data = b"-" * padding + pattern.replace(b"-", gap) + b"z" * padding
    assert graph().onnx_classify(data) == classify_reference(data)


def test_rembg_signature_random_bytes():
    rng = np.random.default_rng(149)
    for _ in range(300):
        data = rng.integers(0, 256, 12000, dtype=np.uint8).tobytes()
        pattern = PATTERNS[int(rng.integers(4))]
        at = int(rng.integers(12000))
        data = data[:at] + pattern + data[at:]
        assert graph().onnx_classify(data) == classify_reference(data)


@pytest.mark.parametrize("dtype", [T.FLOAT, T.FLOAT16, T.DOUBLE])
@pytest.mark.parametrize("shape", [(1, 3, "h", "w"), (1, "h", "w", 3), (1, 3, 5, 7)])
def test_whole_loader(dtype, shape):
    a, b = model(shape, dtype), model(shape, dtype)
    expected, actual = old_load.load_onnx_model(a), current_load.load_onnx_model(b)
    assert actual.bytes == expected.bytes
    assert actual.sub_type == expected.sub_type
    assert vars(actual.info) | {"size_req": vars(actual.info.size_req)} == vars(
        expected.info
    ) | {"size_req": vars(expected.info.size_req)}
    assert a.SerializeToString() == b.SerializeToString()


def test_independent_graphs_concurrent():
    def run(index):
        a, b = model(), model()
        actual = current.ModelShapeInference(a).infer_shape((index + 1, index + 3))
        expected = old.ModelShapeInference(b).infer_shape((index + 1, index + 3))
        assert actual == expected
        assert a.SerializeToString() == b.SerializeToString()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(24)))


# The walkers once iterated a range whose .attr() chain had already freed its
# intermediates: on CPython 3.14 with upb protobuf, a model without inputs crashed
# the process (an access violation). Each walker runs in a child process.
WALKERS = """
from onnx import TensorProto as T
from onnx import helper

from nodes.impl.native_graph import graph


def model():
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["in"], ["out"])],
            "test",
            [helper.make_tensor_value_info("in", T.FLOAT, [1, 3, "h", "w"])],
            [helper.make_tensor_value_info("out", T.FLOAT, [1, 3, "h", "w"])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
    )


native = graph()
for _ in range(200):
    source = model()
    native.onnx_tensor_shape(source.graph.input[0].type.tensor_type)
    native.onnx_update_dims(source, {"in": [1, 3, 8, 8]}, {"out": [1, 3, 8, 8]})
    source.graph.ClearField("input")
    native.onnx_fp_type(source)
print("walked")
"""


def test_graph_walkers_keep_their_ranges_alive():
    done = run_python(
        ["-c", WALKERS], Path(__file__).resolve().parents[2] / "backend" / "src", 300
    )
    assert done.returncode == 0, done.output[-4000:]
    assert done.output.splitlines()[-1] == "walked"
