"""Real tiny CPU models test host ownership; ORT's inference engine is retained."""

from __future__ import annotations

import ctypes as ct
import gc
import importlib.util
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from nodes.impl import native_onnx_runtime as native
from nodes.impl import native_versions
from nodes.impl.onnx import session as integration
from nodes.impl.onnx.model import OnnxGeneric, OnnxInfo

PATH = Path(__file__).with_name("reference_onnx_session.py")
SPEC = importlib.util.spec_from_file_location(
    "nodes.impl.onnx.reference_onnx_session", PATH
)
assert SPEC is not None and SPEC.loader is not None
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


def model(kind="Identity", dtype=TensorProto.FLOAT, shape=None):
    if shape is None:
        shape = [1, 3, "height", "width"]
    x = helper.make_tensor_value_info('input "色"', dtype, shape)
    output_type = TensorProto.FLOAT16 if kind == "Cast" else dtype
    y = helper.make_tensor_value_info("output", output_type, shape)
    weights = []
    if kind == "Identity":
        nodes = [helper.make_node("Identity", [x.name], [y.name])]
    elif kind == "Add":
        weights = [numpy_helper.from_array(np.array([0.25], np.float32), "amount")]
        nodes = [helper.make_node("Add", [x.name, "amount"], [y.name])]
    elif kind == "Cast":
        nodes = [helper.make_node("Cast", [x.name], [y.name], to=TensorProto.FLOAT16)]
    else:
        weights = [
            numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), "scales")
        ]
        nodes = [
            helper.make_node(
                "Resize",
                [x.name, "", "scales"],
                [y.name],
                mode="nearest",
                coordinate_transformation_mode="asymmetric",
                nearest_mode="floor",
            )
        ]
    graph = helper.make_graph(nodes, "tiny-host-contract", [x], [y], weights)
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10
    ).SerializeToString()


def counts():
    sessions, results = ct.c_size_t(), ct.c_size_t()
    assert native._api().cn_ort_live_handles(ct.byref(sessions), ct.byref(results)) == 0
    return sessions.value, results.value


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


@pytest.mark.parametrize("kind", ["Identity", "Add", "Cast", "Resize"])
def test_real_cpu_models_and_dynamic_shapes(kind):
    serialized = model(kind)
    baseline = ort.InferenceSession(serialized, providers=["CPUExecutionProvider"])
    current = native.NativeSession(serialized)
    try:
        assert current.runtime_version == native_versions.ORT
        assert (
            current.get_providers()
            == baseline.get_providers()
            == ["CPUExecutionProvider"]
        )
        assert current.get_provider_options() == baseline.get_provider_options()
        for actual, expected in zip(
            [*current.get_inputs(), *current.get_outputs()],
            [*baseline.get_inputs(), *baseline.get_outputs()],
            strict=True,
        ):
            assert (actual.name, actual.type, actual.shape) == (
                expected.name,
                expected.type,
                expected.shape,
            )
        for shape in ((1, 3, 5, 7), (1, 3, 1, 9), (1, 3, 13, 11)):
            source = np.random.default_rng(12).random(shape, dtype=np.float32)
            source.flat[:5] = [-0.0, 0.0, 1, -1, 1e-40]
            name = current.get_inputs()[0].name
            for outputs in (None, [], ["output"]):
                exact(
                    current.run(outputs, {name: source})[0],
                    baseline.run(outputs, {name: source})[0],
                )
    finally:
        current.close()


@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
        "float16",
        "float64",
        "uint8",
        "int8",
        "uint16",
        "int16",
        "int32",
        "int64",
        "uint32",
        "uint64",
        "bool",
    ],
)
def test_numeric_identity(dtype):
    array = np.arange(24).reshape(2, 3, 4).astype(dtype)
    code = helper.np_dtype_to_tensor_dtype(array.dtype)
    serialized = model(dtype=code, shape=["a", "b", "c"])
    baseline = ort.InferenceSession(serialized, providers=["CPUExecutionProvider"])
    current = native.NativeSession(serialized)
    try:
        name = current.get_inputs()[0].name
        for source in (
            array,
            array[:, ::-1, ::-1],
            array.transpose(2, 1, 0),
            np.asfortranarray(array),
        ):
            source.flags.writeable = False
            before = source.copy()
            actual = current.run(None, {name: source})[0]
            exact(actual, baseline.run(None, {name: source})[0])
            exact(source.copy(), before)
            assert actual.flags.writeable and actual.flags.c_contiguous
            assert not np.shares_memory(actual, source)
    finally:
        current.close()


@pytest.mark.parametrize("shape", [[], [0], [2, 0, 3]])
def test_scalars_and_empty_tensors(shape):
    serialized = model(shape=shape)
    baseline = ort.InferenceSession(serialized, providers=["CPUExecutionProvider"])
    current = native.NativeSession(serialized)
    try:
        source = np.ones(shape, np.float32)
        name = current.get_inputs()[0].name
        actual = current.run(None, {name: source})[0]
        expected = baseline.run(None, {name: source})[0]
        assert isinstance(expected, np.ndarray)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
    finally:
        current.close()


def test_weak_cache_semantics_and_cleanup():
    before = counts()
    wrapper = OnnxGeneric(model(), OnnxInfo(13, "fp32"))
    first = integration.get_onnx_session(wrapper, 0, "CPUExecutionProvider", False)
    assert isinstance(first, native.NativeSession)
    # Existing model-only cache ignores subsequent provider/settings changes.
    assert (
        integration.get_onnx_session(wrapper, 9, "unavailable-provider", True) is first
    )
    del wrapper
    gc.collect()
    assert first.run(
        None, {first.get_inputs()[0].name: np.ones((1, 3, 2, 2), np.float32)}
    )
    del first
    gc.collect()
    assert counts() == before


def test_multiple_sessions_concurrent_runs():
    before = counts()
    sessions = [
        native.NativeSession(model(kind)) for kind in ("Identity", "Add", "Cast")
    ]
    references = [
        ort.InferenceSession(model(kind), providers=["CPUExecutionProvider"])
        for kind in ("Identity", "Add", "Cast")
    ]
    source = np.arange(3 * 17 * 19, dtype=np.float32).reshape(1, 3, 17, 19) / 1024
    expected = [s.run(None, {s.get_inputs()[0].name: source})[0] for s in references]

    def run(index):
        which = index % 3
        s = sessions[which]
        exact(s.run(None, {s.get_inputs()[0].name: source})[0], expected[which])

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(run, range(24)))
    finally:
        for session in sessions:
            session.close()
    assert counts() == before


@pytest.mark.parametrize("error", ["missing", "extra", "rank", "dtype", "output"])
def test_runtime_error_classes_and_messages(error):
    serialized = model()
    baseline = ort.InferenceSession(serialized, providers=["CPUExecutionProvider"])
    current = native.NativeSession(serialized)
    name = current.get_inputs()[0].name
    feed: dict[str, np.ndarray] = {name: np.ones((1, 3, 2, 2), np.float32)}
    outputs = None
    if error == "missing":
        feed = {}
    elif error == "extra":
        feed["unexpected"] = feed[name]
    elif error == "rank":
        feed[name] = np.ones((3, 2, 2), np.float32)
    elif error == "dtype":
        feed[name] = feed[name].astype(np.float64)
    else:
        outputs = ["unknown"]
    try:
        with pytest.raises(Exception) as expected:
            baseline.run(outputs, feed)
        with pytest.raises(type(expected.value)) as actual:
            current.run(outputs, feed)
        assert str(actual.value) == str(expected.value)
    finally:
        current.close()


def test_corrupt_model_and_bad_dll_do_not_fallback():
    before = counts()
    with pytest.raises(Exception) as expected:
        ort.InferenceSession(b"not an ONNX model", providers=["CPUExecutionProvider"])
    with pytest.raises(type(expected.value)):
        native.NativeSession(b"not an ONNX model")
    with pytest.raises(Exception, match="Cannot load ONNX Runtime library"):
        native.NativeSession(
            model(), library_path=Path(__file__).with_name("not_present.dll")
        )
    assert counts() == before


def test_closed_and_double_close_are_safe():
    before = counts()
    session = native.NativeSession(model())
    session.close()
    session.close()
    with pytest.raises(Exception, match="closed or invalid"):
        session.run(
            None, {session.get_inputs()[0].name: np.zeros((1, 3, 2, 2), np.float32)}
        )
    assert counts() == before


def test_raw_result_owns_noop_input_after_session_close():
    """A graph output directly aliases its input; C++ must own that storage."""
    before = counts()
    value = helper.make_tensor_value_info("shared", TensorProto.FLOAT, [2, 3])
    graph = helper.make_graph([], "aliased-buffer", [value], [value])
    serialized = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10
    ).SerializeToString()
    current = native.NativeSession(serialized)
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    expected = source.copy()
    dims = (ct.c_int64 * 2)(2, 3)
    descriptor = native._Input(b"shared", source.ctypes.data, source.nbytes, dims, 2, 1)
    outputs = (ct.c_char_p * 1)(b"shared")
    result = ct.c_uint64()
    dll = native._api()
    error = ct.create_string_buffer(1024)
    assert (
        dll.cn_ort_run(
            current._handle,
            ct.byref(descriptor),
            1,
            outputs,
            1,
            ct.byref(result),
            error,
            len(error),
        )
        == 0
    )
    try:
        current.close()
        source.fill(99)
        del source
        gc.collect()
        actual = np.empty((2, 3), np.float32)
        assert (
            dll.cn_ort_result_copy(
                result, 0, actual.ctypes.data, actual.nbytes, error, len(error)
            )
            == 0
        )
        exact(actual, expected)
    finally:
        assert dll.cn_ort_release_result(result, error, len(error)) == 0
        current.close()
    assert dll.cn_ort_release_result(result, error, len(error)) == 2
    assert counts() == before


@pytest.mark.parametrize(
    "malformed",
    [
        "negative",
        "overflow",
        "short",
        "null",
        "bad_type",
        "null_name",
        "bad_handle",
        "wrapped_data",
        "unaligned_shape",
    ],
)
def test_raw_malformed_input_does_not_leak(malformed):
    before = counts()
    current = native.NativeSession(model(shape=[2, 3]))
    source = np.ones((2, 3), np.float32)
    dims = (ct.c_int64 * 2)(2, 3)
    descriptor = native._Input(
        current.get_inputs()[0].name.encode(),
        source.ctypes.data,
        source.nbytes,
        dims,
        2,
        1,
    )
    session_id = current._handle
    if malformed == "negative":
        dims[0] = -1
    elif malformed == "overflow":
        dims[0] = 2**62
    elif malformed == "short":
        descriptor.bytes -= 1
    elif malformed == "null":
        descriptor.data = None
    elif malformed == "bad_type":
        descriptor.type = 8
    elif malformed == "null_name":
        descriptor.name = None
    elif malformed == "wrapped_data":
        descriptor.data = ct.c_size_t(-3).value
    elif malformed == "unaligned_shape":
        descriptor.shape = ct.cast(ct.c_void_p(1), ct.POINTER(ct.c_int64))
    else:
        session_id = 2**64 - 1
    outputs = (ct.c_char_p * 1)(b"output")
    result = ct.c_uint64(12345)
    dll = native._api()
    error = ct.create_string_buffer(1024)
    try:
        status = dll.cn_ort_run(
            session_id,
            ct.byref(descriptor),
            1,
            outputs,
            1,
            ct.byref(result),
            error,
            len(error),
        )
        assert status == (9 if malformed == "bad_type" else 2)
        assert result.value == 12345 and error.value
    finally:
        current.close()
    assert counts() == before


def test_close_racing_run_keeps_native_ownership_safe():
    before = counts()
    current = native.NativeSession(model("Add"))
    name = current.get_inputs()[0].name
    source = np.ones((1, 3, 127, 129), np.float32)

    def run(_):
        try:
            actual = current.run(None, {name: source})[0]
        except Exception as error:
            assert "closed or invalid" in str(error)
        else:
            exact(actual, source + np.float32(0.25))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run, i) for i in range(12)]
        current.close()
        for future in futures:
            future.result()
    assert counts() == before


def test_exceptional_float_bits_and_unaligned_input():
    serialized = model(shape=[7])
    current = native.NativeSession(serialized)
    baseline = ort.InferenceSession(serialized, providers=["CPUExecutionProvider"])
    source = np.ndarray((7,), np.float32, buffer=bytearray(29), offset=1)
    source.view(np.uint32)[:] = [
        0,
        0x80000000,
        1,
        0x7F800000,
        0xFF800000,
        0x7FC01234,
        0x7FA00001,
    ]
    try:
        feed = {current.get_inputs()[0].name: source}
        exact(current.run(None, feed)[0], baseline.run(None, feed)[0])
    finally:
        current.close()


def test_invalid_model_span_rejected_before_runtime_loading():
    error = ct.create_string_buffer(1024)
    handle = ct.c_uint64(999)
    status = native._api().cn_ort_create(
        "unused",
        ct.c_void_p(ct.c_size_t(-3).value),
        64,
        ct.byref(handle),
        error,
        len(error),
    )
    assert status == 2 and handle.value == 999


def test_numeric_factory_never_uses_python_inference_binding(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dense CPU session must use the public native API")

    monkeypatch.setattr(ort, "InferenceSession", forbidden)
    wrapper = OnnxGeneric(model(), OnnxInfo(13, "fp32"))
    session = integration.create_inference_session(wrapper, 0, "CPUExecutionProvider")
    assert isinstance(session, native.NativeSession)
    try:
        source = np.ones((1, 3, 1, 1), np.float32)
        exact(session.run(None, {session.get_inputs()[0].name: source})[0], source)
    finally:
        session.close()


@pytest.mark.parametrize("dtype", [">f4", ">i8"])
def test_original_non_native_endian_interpretation(dtype):
    source = np.array([1, 2, 3], dtype=dtype)
    code = helper.np_dtype_to_tensor_dtype(source.dtype.newbyteorder("="))
    serialized = model(dtype=code, shape=[3])
    baseline = ort.InferenceSession(serialized, providers=["CPUExecutionProvider"])
    current = native.NativeSession(serialized)
    try:
        feed = {current.get_inputs()[0].name: source}
        exact(current.run(None, feed)[0], baseline.run(None, feed)[0])
    finally:
        current.close()


def test_explicit_provider_retention_without_gpu_execution(monkeypatch):
    calls = []
    sentinel = object()

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(ort, "InferenceSession", capture)
    wrapper = OnnxGeneric(model(), OnnxInfo(13, "fp32"))
    for provider in (
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
        "DmlExecutionProvider",
    ):
        assert (
            integration.create_inference_session(
                wrapper, 2, provider, True, "example-cache"
            )
            is sentinel
        )
        actual = calls.pop()
        assert (
            REFERENCE.create_inference_session(
                wrapper, 2, provider, True, "example-cache"
            )
            is sentinel
        )
        assert calls.pop() == actual


def test_string_graph_retains_original_binding():
    serialized = model(dtype=TensorProto.STRING, shape=[1])
    assert not native.supports_model(serialized)
    wrapper = OnnxGeneric(serialized, OnnxInfo(13, "fp32"))
    session = integration.create_inference_session(wrapper, 0, "CPUExecutionProvider")
    assert isinstance(session, ort.InferenceSession)
    result = session.run(
        None, {session.get_inputs()[0].name: np.array(["text"], object)}
    )
    assert isinstance(result[0], np.ndarray)
    assert result[0].tolist() == ["text"]


def test_cpu_provider_deduplication_warning_matches_original():
    wrapper = OnnxGeneric(model(), OnnxInfo(13, "fp32"))
    observed = []
    for module in (REFERENCE, integration):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            session = module.create_inference_session(
                wrapper, 0, "CPUExecutionProvider"
            )
        observed.append([(item.category, str(item.message)) for item in caught])
        if isinstance(session, native.NativeSession):
            session.close()
    assert observed[0] == observed[1]
