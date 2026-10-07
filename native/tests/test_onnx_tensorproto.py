"""Native application helpers against independent installed TensorProto source.

ONNX's public storage decoder is deliberately retained. These tests compare the
application's traversal, copies, shape quirks, exceptions and protobuf mutations,
including behavior that is awkward but observable in the original converter.
"""

from __future__ import annotations

import struct
import warnings
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from onnx import AttributeProto as A
from onnx import NodeProto, helper, numpy_helper
from onnx import TensorProto as T
from reference_onnx_graph import tensorproto_utils as old

from nodes.impl.onnx import tensorproto_utils as current


def snapshot(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = call()
            error = None
        except Exception as exc:
            value, error = None, (type(exc), str(exc))
    return value, error, [(type(w.message), str(w.message)) for w in caught]


def equal_value(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, np.ndarray):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
        assert actual.flags.c_contiguous and actual.flags.writeable
        assert actual.flags.owndata
    elif isinstance(actual, float):
        assert struct.pack("d", actual) == struct.pack("d", expected)
    elif isinstance(actual, T):
        assert actual.SerializeToString() == expected.SerializeToString()
    else:
        assert actual == expected


def compare_calls(actual, expected):
    a, error_a, warnings_a = snapshot(actual)
    b, error_b, warnings_b = snapshot(expected)
    assert error_a == error_b
    assert warnings_a == warnings_b
    if error_a is None:
        equal_value(a, b)


def attr_node():
    first = A(name="x", i=-(2**63), f=-0.0, s=b"ascii\0text", type=A.INTS)
    first.ints.extend([-(2**63), -1, 0, 1, 2**63 - 1])
    first.floats.extend([-0.0, 0.0, np.inf, -np.inf, np.nan, 0.1])
    first.t.CopyFrom(numpy_helper.from_array(np.array([1, 2], np.int32)))
    return NodeProto(attribute=[A(name="before", i=7), first, A(name="x", i=11)])


@pytest.mark.parametrize("suffix", ["ai", "af", "i", "f", "s", "tensor"])
@pytest.mark.parametrize("key", ["x", "before", "missing", "", None, 3])
def test_attributes(suffix, key):
    node = attr_node()
    before = node.SerializeToString()
    compare_calls(
        lambda: getattr(current, f"get_node_attr_{suffix}")(node, key),
        lambda: getattr(old, f"get_node_attr_{suffix}")(node, key),
    )
    assert node.SerializeToString() == before


@pytest.mark.parametrize("suffix", ["i", "f", "s"])
def test_default_identity_and_default_type(suffix):
    node, sentinel = NodeProto(), object()
    for module in (old, current):
        function = getattr(module, f"get_node_attr_{suffix}")
        assert function(node, "missing", sentinel) is sentinel
    equal_value(
        getattr(current, f"get_node_attr_{suffix}")(node, "missing"),
        getattr(old, f"get_node_attr_{suffix}")(node, "missing"),
    )


def test_tensor_identity_and_new_empty_objects():
    node = attr_node()
    for module in (old, current):
        assert module.get_node_attr_tensor(node, "x") is node.attribute[1].t
        a = module.get_node_attr_tensor(node, "missing")
        b = module.get_node_attr_tensor(node, "missing")
        assert a is not b
        a.name = "changed"
        assert b.name == ""


@pytest.mark.parametrize("value", [b"", b"\x00", b"\x80", b"x\xffy", "色".encode()])
def test_ascii_decode(value):
    node = NodeProto(attribute=[A(name="s", s=value)])
    compare_calls(
        lambda: current.get_node_attr_s(node, "s"),
        lambda: old.get_node_attr_s(node, "s"),
    )


@pytest.mark.parametrize("integer", [-(2**80), -(2**63), -1, 0, 2**63 - 1, 2**80])
@pytest.mark.parametrize("suffix", ["i", "ai"])
def test_clamp_bounds(integer, suffix):
    node = SimpleNamespace(
        attribute=[SimpleNamespace(name="x", i=integer, ints=[integer])]
    )
    compare_calls(
        lambda: getattr(current, f"get_node_attr_{suffix}")(node, "x"),
        lambda: getattr(old, f"get_node_attr_{suffix}")(node, "x"),
    )


@pytest.mark.parametrize(
    "values",
    [
        [],
        [1, 2, -3],
        [0.0, -0.0, np.nan, np.inf, -np.inf],
        np.array([0.1, 0.2], np.float16),
        np.array([1, 2, 3], np.int64)[::-1],
        np.array([[1, 2], [3, 4]], np.float32),
        ["bad"],
        [1, object()],
        [2**200],
        None,
    ],
)
@pytest.mark.parametrize("key", ["x", "", None, 5])
def test_setter_validation_atomicity_and_wrong_field(values, key):
    a, b = attr_node(), attr_node()
    compare_calls(
        lambda: current.set_node_attr_ai(a, key, values),
        lambda: old.set_node_attr_ai(b, key, values),
    )
    assert a.SerializeToString() == b.SerializeToString()
    if len(a.attribute) == 4:
        assert a.attribute[-1].type == A.INTS
        assert list(a.attribute[-1].ints) == []


DTYPES = [
    np.float16,
    np.float32,
    np.float64,
    np.int32,
    np.int64,
    np.int8,
    np.uint8,
    np.uint32,
    np.uint64,
    np.bool_,
    np.complex64,
    np.complex128,
]
SHAPES = [
    (),
    (0,),
    (1,),
    (2,),
    (0, 3),
    (1, 0),
    (2, 0),
    (2, 1),
    (2, 2),
    (1, 1, 1),
    (2, 1, 1),
]


def make_tensor(dtype, shape, storage):
    source = np.arange(np.prod(shape, dtype=np.int64)).astype(dtype).reshape(shape)
    tensor = numpy_helper.from_array(source, name="tensor")
    if storage == "typed":
        tensor = helper.make_tensor(
            "tensor", tensor.data_type, shape, source.ravel().tolist(), raw=False
        )
    return tensor


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("storage", ["raw", "typed"])
@pytest.mark.parametrize("suffix", ["f", "ai", "af"])
def test_decoded_dtype_shape_storage(dtype, shape, storage, suffix):
    tensor = make_tensor(dtype, shape, storage)
    before = tensor.SerializeToString()
    compare_calls(
        lambda: getattr(current, f"get_node_attr_from_input_{suffix}")(tensor),
        lambda: getattr(old, f"get_node_attr_from_input_{suffix}")(tensor),
    )
    assert tensor.SerializeToString() == before


@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.int32, np.int64]
)
@pytest.mark.parametrize("shape", [(7,), (7, 1)])
@pytest.mark.parametrize("suffix", ["f", "ai", "af"])
def test_extremes(dtype, shape, suffix):
    if np.issubdtype(dtype, np.floating):
        values = [
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
            np.finfo(dtype).tiny,
            np.finfo(dtype).max,
        ]
    else:
        values = [
            np.iinfo(dtype).min,
            -1,
            0,
            1,
            7,
            np.iinfo(dtype).max - 1,
            np.iinfo(dtype).max,
        ]
    tensor = numpy_helper.from_array(np.array(values, dtype).reshape(shape))
    compare_calls(
        lambda: getattr(current, f"get_node_attr_from_input_{suffix}")(tensor),
        lambda: getattr(old, f"get_node_attr_from_input_{suffix}")(tensor),
    )


@pytest.mark.parametrize(
    "dtype,bits",
    [
        (np.float16, [0x7C01, 0x7E03, 0xFC02, 0xFE05]),
        (np.float32, [0x7F800001, 0x7FC00003, 0xFF800002, 0xFFC00005]),
        (
            np.float64,
            [
                0x7FF0000000000001,
                0x7FF8000000000003,
                0xFFF0000000000002,
                0xFFF8000000000005,
            ],
        ),
    ],
)
@pytest.mark.parametrize("suffix", ["f", "af"])
def test_nan_payloads(dtype, bits, suffix):
    values = np.array(bits, dtype=f"u{np.dtype(dtype).itemsize}").view(dtype)
    tensor = numpy_helper.from_array(values)
    compare_calls(
        lambda: getattr(current, f"get_node_attr_from_input_{suffix}")(tensor),
        lambda: getattr(old, f"get_node_attr_from_input_{suffix}")(tensor),
    )


@pytest.mark.parametrize("suffix", ["f", "ai", "af"])
@pytest.mark.parametrize(
    "case",
    [
        "undefined",
        "wrong_raw",
        "wrong_typed",
        "negative_dims",
        "segment",
        "external",
        "string",
    ],
)
def test_invalid_format_errors(suffix, case):
    tensor = T(data_type=T.INT64, dims=[2], int64_data=[1, 2])
    if case == "undefined":
        tensor.data_type = 0
    elif case == "wrong_raw":
        tensor.raw_data = b"\x01\x02\x03"
    elif case == "wrong_typed":
        tensor.ClearField("int64_data")
        tensor.int64_data.append(1)
    elif case == "negative_dims":
        tensor.dims[:] = [-2]
    elif case == "segment":
        tensor.segment.begin = 1
    elif case == "external":
        tensor.data_location = T.EXTERNAL
        tensor.external_data.add(
            key="location", value="nonexistent-tensorproto-fixture.bin"
        )
    else:
        tensor = T(data_type=T.STRING, dims=[1], string_data=[b"\xff"])
    before = tensor.SerializeToString()
    compare_calls(
        lambda: getattr(current, f"get_node_attr_from_input_{suffix}")(tensor),
        lambda: getattr(old, f"get_node_attr_from_input_{suffix}")(tensor),
    )
    assert tensor.SerializeToString() == before


@pytest.mark.parametrize("suffix", ["f", "ai", "af"])
def test_decoder_call_order_and_logged_rejection(monkeypatch, suffix):
    tensor = T(data_type=T.BOOL)
    seen = []

    def decode(value):
        assert value is tensor
        seen.append("decode")
        raise ValueError("decoder sentinel")

    def log(message, *args):
        seen.append(message % args if args else message)

    monkeypatch.setattr(numpy_helper, "to_array", decode)
    monkeypatch.setattr(old, "logger", SimpleNamespace(error=log))
    monkeypatch.setattr(current, "logger", SimpleNamespace(error=log))
    expected = snapshot(
        lambda: getattr(old, f"get_node_attr_from_input_{suffix}")(tensor)
    )
    original_seen = seen[:]
    seen.clear()
    actual = snapshot(
        lambda: getattr(current, f"get_node_attr_from_input_{suffix}")(tensor)
    )
    assert actual[1:] == expected[1:]
    assert (
        seen
        == original_seen
        == (["decode"] if suffix == "f" else ["Unknown data type 9"])
    )
    if actual[1] is None:
        equal_value(actual[0], expected[0])


@pytest.mark.parametrize(
    "suffix,dtype",
    [
        ("ai", np.int32),
        ("ai", np.int64),
        ("af", np.float16),
        ("af", np.float32),
        ("af", np.float64),
    ],
)
@pytest.mark.parametrize("layout", ["reversed", "fortran", "readonly", "unaligned"])
def test_decoder_array_copy_layout(monkeypatch, suffix, dtype, layout):
    shape = (8, 1) if dtype == np.int64 else (3, 4)
    source = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    if layout == "reversed":
        source = source[::-1, ::-1]
    elif layout == "fortran":
        source = np.asfortranarray(source)
    elif layout == "readonly":
        source.setflags(write=False)
    else:
        target = np.ndarray(shape, dtype, buffer=bytearray(source.nbytes + 1), offset=1)
        target[:] = source
        source = target
    before = source.tobytes()
    tensor = numpy_helper.from_array(source)
    monkeypatch.setattr(numpy_helper, "to_array", lambda _: source)
    compare_calls(
        lambda: getattr(current, f"get_node_attr_from_input_{suffix}")(tensor),
        lambda: getattr(old, f"get_node_attr_from_input_{suffix}")(tensor),
    )
    actual = getattr(current, f"get_node_attr_from_input_{suffix}")(tensor)
    assert not np.shares_memory(actual, source)
    actual.fill(7)
    assert source.tobytes() == before


@pytest.mark.parametrize(
    "type_id", [T.FLOAT, T.FLOAT16, T.DOUBLE, T.INT32, T.INT64, T.STRING, 0]
)
@pytest.mark.parametrize("raw_size", [0, 1, 2, 3, 4, 5, 7, 8, 17])
@pytest.mark.parametrize("mode", [T.FLOAT, T.FLOAT16, T.DOUBLE, -1, None, "10"])
def test_data_size_quirks(type_id, raw_size, mode):
    tensor = T(
        data_type=type_id,
        raw_data=b"x" * raw_size,
        float_data=[1, 2, 3],
        int32_data=[1, 2, 3, 4, 5],
    )
    assert current.get_tensor_proto_data_size(
        tensor, mode
    ) == old.get_tensor_proto_data_size(tensor, mode)


def test_concurrent_helpers():
    tensors = [
        numpy_helper.from_array(np.arange(2048, dtype=dtype))
        for dtype in (np.float16, np.float32, np.float64, np.int32, np.int64)
    ]

    def check(index):
        tensor = tensors[index % len(tensors)]
        suffix = "ai" if tensor.data_type in (T.INT32, T.INT64) else "af"
        compare_calls(
            lambda: getattr(current, f"get_node_attr_from_input_{suffix}")(tensor),
            lambda: getattr(old, f"get_node_attr_from_input_{suffix}")(tensor),
        )
        node = attr_node()
        equal_value(
            current.get_node_attr_ai(node, "x"), old.get_node_attr_ai(node, "x")
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check, range(32)))
