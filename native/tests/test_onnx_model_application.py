"""Native ONNX loading/interpolation control against frozen installed helpers.

One test runs tiny synthetic models on CPU only. No downloaded model,
GPU provider, performance measurement or inference-engine conversion is involved.
"""

from __future__ import annotations

import ast
import copy
import sys
import types
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper
from reference_onnx_graph import load as old_load
from reference_onnx_graph import utils as old_utils

from nodes.impl.native_analysis import mean as native_mean
from nodes.impl.native_graph import graph
from nodes.impl.native_onnx_runtime import NativeSession
from nodes.impl.native_tensors import cast_numpy, interpolate_numpy
from nodes.impl.onnx import load as current_load
from nodes.impl.onnx import utils as current_utils

ROOT = Path(__file__).resolve().parents[2]
FROZEN = Path(__file__).with_name("reference_onnx_graph")
CURRENT_INTERP = (
    ROOT / "backend/src/packages/chaiNNer_onnx/onnx/utility/interpolate_models.py"
)


class NotAnException(BaseException):
    pass


def outcome(
    function: Callable[[], Any],
) -> tuple[Any, tuple[type[BaseException], str] | None, list[tuple[type, str]]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = function()
            error = None
        except BaseException as exc:
            value = None
            error = (type(exc), str(exc))
    return value, error, [(type(w.message), str(w.message)) for w in caught]


def info(value):
    return (
        value.sub_type,
        value.bytes,
        vars(value.info) | {"size_req": vars(value.info.size_req)},
    )


def tiny_model(weight=1.0, dtype=np.float32, name="weight"):
    tensor = numpy_helper.from_array(np.asarray([weight], dtype=dtype), name)
    dtype_id = tensor.data_type
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Mul", ["x", name], ["y"])],
            "tiny-interpolation",
            [helper.make_tensor_value_info("x", dtype_id, [1, 3, "h", "w"])],
            [helper.make_tensor_value_info("y", dtype_id, [1, 3, "h", "w"])],
            [tensor],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )


class Recorder:
    def __init__(self):
        self.events = []

    def error(self, message, *args, exc_info=False):
        error = sys.exc_info()[1] if exc_info else None
        self.events.append(
            (
                message % args if args else message,
                None if error is None else (type(error).__name__, str(error)),
            )
        )

    def exception(self, message, *args):
        self.error(message, *args, exc_info=True)

    def debug(self, message, *args):
        self.events.append((message % args if args else message, None))


class Probe(current_utils.ModelShapeInference, old_utils.ModelShapeInference):
    """A scripted shape inference for both loaders; it never builds a model."""

    def __init__(self, responses, width=None, height=None):
        self.fixed_input_width = width
        self.fixed_input_height = height
        self.responses = iter(responses)
        self.calls = []

    def infer_shape(self, input_size):
        self.calls.append(input_size)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


KNOWN = ((16, 16, 3), (32, 32, 3))
UNKNOWN = ((16, 16, 3), (None, 32, 3))


@pytest.mark.parametrize(
    "width,height", [(None, None), (3, None), (None, 5), (3, 5), (0, 0), (-1, 7)]
)
@pytest.mark.parametrize(
    "responses",
    [
        [KNOWN] * 4,
        [UNKNOWN] * 4,
        [ValueError("probe")] * 4,
        [UNKNOWN, RuntimeError("probe"), UNKNOWN, KNOWN],
        [None, ((), (1,)), ((), (1, 2, 3, 4)), KNOWN],
        [((), None)] * 4,
        [NotAnException("do not catch")] * 4,
    ],
)
def test_probe_order_errors_and_exception_logging(
    monkeypatch, width, height, responses
):
    a, b = Probe(responses, width, height), Probe(responses, width, height)
    log_a, log_b = Recorder(), Recorder()
    monkeypatch.setattr(old_load, "logger", log_a)
    monkeypatch.setattr(current_load, "logger", log_b)
    expected = outcome(lambda: old_load._detect_size_req(a))
    actual = outcome(lambda: current_load._detect_size_req(b))
    assert actual[1:] == expected[1:]
    if actual[1] is None:
        assert vars(actual[0][0]) == vars(expected[0][0])
        assert actual[0][1] == expected[0][1]
    assert a.calls == b.calls
    assert log_a.events == log_b.events


def test_probe_logging_preserves_and_restores_exception_context(monkeypatch):
    records = []

    class RaisingLog(Recorder):
        def error(self, message, *args, exc_info=False):
            records.append(sys.exc_info()[1])
            raise LookupError("logger failed")

    for module in (old_load, current_load):
        monkeypatch.setattr(module, "logger", RaisingLog())
        outer = RuntimeError("outer")
        try:
            raise outer
        except RuntimeError:
            try:
                module._detect_size_req(Probe([ValueError("probe")]))
            except LookupError as error:
                assert isinstance(error.__context__, ValueError)
                assert str(error.__context__) == "probe"
            else:
                raise AssertionError("logger exception was suppressed")
            assert sys.exc_info()[1] is outer
    assert [(type(error), str(error)) for error in records] == [
        (ValueError, "probe")
    ] * 2


@pytest.mark.parametrize(
    "input_hw,output_hw",
    [
        ((3, 5), (6, 15)),
        ((3, 5), (7, 16)),
        ((None, 5), (None, 10)),
        ((3, None), (6, None)),
        ((0, 5), (0, 10)),
        ((3, 0), (6, 0)),
        ((-3, 5), (6, -10)),
        ((3, 5), (0, 0)),
        ((2.5, 4.0), (5.0, 8.0)),
        ((np.int64(3), np.int64(5)), (np.int64(6), np.int64(10))),
    ],
)
def test_loader_scale_and_partial_metadata(monkeypatch, input_hw, output_hw):
    shape = ((*input_hw, 3), (*output_hw, 7))
    records = []
    results = []
    for module in (old_load, current_load):
        prototype = tiny_model()
        before = prototype.SerializeToString()

        def constructor(model):
            model.doc_string = "inference mutation after byte snapshot"
            return types.SimpleNamespace(
                fixed_input_width=11,
                fixed_input_height=13,
                input_channels=3,
                output_channels=4,
            )

        monkeypatch.setattr(module, "ModelShapeInference", constructor)
        requirement = module.SizeReq(multiple_of=64)
        monkeypatch.setattr(
            module, "_detect_size_req", lambda _, req=requirement: (req, shape)
        )
        result = module.load_onnx_model(prototype)
        assert result.bytes == before
        assert result.info.size_req is requirement
        records.append(prototype.SerializeToString())
        results.append(info(result))
    assert results[0] == results[1]
    assert records[0] == records[1]


@pytest.mark.parametrize(
    "stage",
    [
        "construct",
        "missing-height",
        "probe",
        "bad-pair",
        "bad-input",
        "bad-output",
        "uncaught",
    ],
)
def test_loader_exception_boundary_partial_updates(monkeypatch, stage):
    results = []
    for module in (old_load, current_load):

        def constructor(_):
            if stage == "construct":
                raise RuntimeError("construct")
            if stage == "uncaught":
                raise NotAnException("not an Exception")
            infer = types.SimpleNamespace(
                fixed_input_width=11, input_channels=3, output_channels=7
            )
            if stage != "missing-height":
                infer.fixed_input_height = 13
            return infer

        def detect(_, module=module):
            if stage == "probe":
                raise RuntimeError("probe")
            shapes = {
                "bad-pair": (1,),
                "bad-input": (None, (1, 2, 3)),
                "bad-output": ((1, 2, 3), (1, 2)),
            }
            return module.SizeReq(multiple_of=64), shapes.get(stage, KNOWN)

        monkeypatch.setattr(module, "ModelShapeInference", constructor)
        monkeypatch.setattr(module, "_detect_size_req", detect)
        value, error, caught = outcome(
            lambda module=module: module.load_onnx_model(tiny_model())
        )
        results.append((None if value is None else info(value), error, caught))
    assert results[0] == results[1]


def functions(path):
    tree = ast.parse(path.read_text("utf-8"))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for definition in definitions:
        definition.decorator_list = []
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *definitions,
    ]
    module = types.ModuleType("interpolation_test_" + path.parent.name)
    module.__dict__.update(
        np=np,
        onnx=onnx,
        onph=numpy_helper,
        deepcopy=copy.deepcopy,
        graph=graph,
        native_mean=native_mean,
        interpolate_numpy=interpolate_numpy,
        cast_numpy=cast_numpy,
        logger=Recorder(),
        NO_TILING=object(),
        load_onnx_model=current_load.load_onnx_model,
        safely_optimize_onnx_model=lambda value: value,
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module


def pair():
    return functions(FROZEN / "interpolate_models.py"), functions(CURRENT_INTERP)


DTYPES = [
    np.float16,
    np.float32,
    np.float64,
    np.int32,
    np.int64,
    np.uint8,
    np.bool_,
    np.complex64,
    np.complex128,
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("amount", [-10, 0, 1, 37, 50, 99, 100, 110, 0.25])
@pytest.mark.parametrize("shape", [(), (0,), (1,), (2, 3)])
def test_pair_decode_arithmetic_encode(dtype, amount, shape):
    a = np.arange(np.prod(shape, dtype=np.int64)).astype(dtype).reshape(shape)
    b = np.full(shape, 3, dtype)
    wa, wb = (
        [numpy_helper.from_array(a, "first")],
        [numpy_helper.from_array(b, "second")],
    )
    before = wa[0].SerializeToString(), wb[0].SerializeToString()
    original, native = pair()
    expected = outcome(lambda: original.perform_interp(wa, wb, amount))
    actual = outcome(lambda: native.perform_interp(wa, wb, amount))
    assert actual[1:] == expected[1:]
    if actual[1] is None:
        assert [value.SerializeToString() for value in actual[0]] == [
            value.SerializeToString() for value in expected[0]
        ]
        assert actual[0][0] is not wb[0]
        assert actual[0][0].name == "second"
    assert before == (wa[0].SerializeToString(), wb[0].SerializeToString())


@pytest.mark.parametrize(
    "length_a,length_b", [(0, 0), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2)]
)
def test_zip_iterator_advance_order(length_a, length_b):
    observations = []
    for module in pair():
        events = []

        class Weights:
            def __init__(self, label, count, events=events):
                self.label, self.count = label, count
                self.events = events

            def __iter__(self):
                self.events.append((self.label, "iter"))
                return self

            def __next__(self):
                self.events.append((self.label, self.count))
                if not self.count:
                    raise StopIteration
                self.count -= 1
                return numpy_helper.from_array(np.ones(2, np.float32), self.label)

        result = module.perform_interp(
            Weights("a", length_a), Weights("b", length_b), 50
        )
        observations.append((events, [value.SerializeToString() for value in result]))
    assert observations[0] == observations[1]


@pytest.mark.parametrize("bad", ["shape", "malformed-a", "malformed-b", "string"])
def test_pair_errors_leave_inputs_untouched(bad):
    wa = numpy_helper.from_array(np.ones(2, np.float32), "a")
    wb = numpy_helper.from_array(np.zeros(3 if bad == "shape" else 2, np.float32), "b")
    if bad == "malformed-a":
        wa.raw_data = b"x"
    elif bad == "malformed-b":
        wb.raw_data = b"x"
    elif bad == "string":
        wa = numpy_helper.from_array(np.array(["x", "y"], object), "a")
    before = wa.SerializeToString(), wb.SerializeToString()
    expected, actual = [
        outcome(lambda module=module: module.perform_interp([wa], [wb], 50))
        for module in pair()
    ]
    assert actual[1:] == expected[1:]
    assert actual[1] is not None
    assert before == (wa.SerializeToString(), wb.SerializeToString())


@pytest.mark.parametrize("amount", [0, 100])
def test_endpoint_alias_shortcuts_do_not_touch_models(amount):
    for module in pair():
        a, b = object(), object()
        result = module.interpolate_models_node(None, a, b, amount)
        assert result == ((a, 100, 0) if amount == 0 else (b, 0, 100))
        assert result[0] is (a if amount == 0 else b)


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "decode-a",
        "decode-b",
        "optimize-a",
        "optimize-b",
        "count",
        "shape",
        "deepcopy",
        "load",
        "compatibility",
        "probe",
    ],
)
def test_interpolation_orchestration_order(failure):
    observations = []
    for module in pair():
        events, copies, loaded = [], [], []
        a, b = tiny_model(1), tiny_model(3)
        a.doc_string, b.doc_string = "A", "B"
        if failure == "count":
            b.graph.initializer.append(numpy_helper.from_array(scalar(7), "extra"))
        elif failure == "shape":
            b.graph.initializer[0].CopyFrom(
                numpy_helper.from_array(np.ones(2, np.float32), "weight")
            )
        a_bytes, b_bytes = a.SerializeToString(), b.SerializeToString()
        models = {a_bytes: a, b_bytes: b}

        def decode(data, models=models, events=events):
            value = models[data]
            tag = value.doc_string.lower()
            events.append("decode-" + tag)
            if failure == "decode-" + tag:
                raise ValueError("decode " + tag)
            return value

        def optimize(value, events=events):
            tag = value.doc_string.lower()
            events.append("optimize-" + tag)
            if failure == "optimize-" + tag:
                raise ValueError("optimize " + tag)
            value.producer_name = "optimized"
            return value

        def deep(value, events=events, copies=copies):
            events.append("deepcopy")
            if failure == "deepcopy":
                raise RuntimeError("deepcopy failed")
            result = copy.deepcopy(value)
            copies.append(result)
            return result

        def load(value, events=events, loaded=loaded):
            events.append("load")
            loaded.append(value)
            if failure == "load":
                raise RuntimeError("load failed")
            return types.SimpleNamespace(sub_type="Generic", proto=value)

        def compatible(context, value, events=events, loaded=loaded):
            assert context == "context"
            assert value.proto is loaded[0]
            events.append("compatibility")
            if failure == "probe":
                raise RuntimeError("probe")
            return failure != "compatibility"

        module.__dict__.update(
            onnx=types.SimpleNamespace(load_from_string=decode),
            safely_optimize_onnx_model=optimize,
            deepcopy=deep,
            load_onnx_model=load,
            check_will_upscale=compatible,
        )
        result, error, caught = outcome(
            lambda module=module, a_bytes=a_bytes, b_bytes=b_bytes: (
                module.interpolate_models_node(
                    "context",
                    types.SimpleNamespace(bytes=a_bytes),
                    types.SimpleNamespace(bytes=b_bytes),
                    25,
                )
            )
        )
        if copies:
            assert copies[0] is not b
            assert loaded[0] is copies[0]
            assert copies[0].graph.initializer[0] is not b.graph.initializer[0]
        observations.append(
            (
                events,
                error,
                caught,
                a.SerializeToString(),
                b.SerializeToString(),
                [value.SerializeToString() for value in copies],
                None if result is None else result[1:],
                module.logger.events,
            )
        )
    assert observations[0] == observations[1]


def scalar(value):
    return np.asarray(value, np.float32)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("value", [0.0, 0.5, 0.5000001, 1.0, np.nan])
def test_compatibility_fixture_and_mean_boundary(dtype, value):
    results = []
    for module in pair():
        seen = []

        def upscale(
            context, image, model, tiling, amount, flag, module=module, seen=seen
        ):
            assert context == "context" and model.sub_type == "Generic"
            assert tiling is module.NO_TILING and amount == 0 and flag is False
            assert image.dtype == np.float32 and image.shape == (3, 3, 3)
            assert image.flags.f_contiguous and image.flags.owndata
            assert image.strides == (4, 12, 36)
            assert image.tobytes() == np.ones((3, 3, 3), np.float32).tobytes()
            seen.append(True)
            return np.full((3, 3, 3), value, dtype)

        module.__dict__.update(upscale_image_node=upscale)
        result = module.check_will_upscale(
            "context", types.SimpleNamespace(sub_type="Generic")
        )
        results.append((type(result), bool(result), seen))
        assert (
            module.check_will_upscale(None, types.SimpleNamespace(sub_type="RemBg"))
            is True
        )
    assert results[0] == results[1]


def test_real_tiny_cpu_interpolation_and_inference():
    a, b = tiny_model(0.75), tiny_model(1.5)
    original, native = pair()
    values = []
    for module in (original, native):
        module.__dict__.update(
            load_onnx_model=(
                old_load if module is original else current_load
            ).load_onnx_model,
            safely_optimize_onnx_model=(
                old_utils if module is original else current_utils
            ).safely_optimize_onnx_model,
        )

        def upscale(_context, image, model, _tiling, _amount, _flag):
            nchw = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
            with closing(NativeSession(model.bytes)) as session:
                result = session.run(None, {"x": nchw})[0]
            return np.ascontiguousarray(result[0].transpose(1, 2, 0))

        module.__dict__.update(upscale_image_node=upscale)
        result = module.interpolate_models_node(
            None,
            current_load.load_onnx_model(a.SerializeToString()),
            current_load.load_onnx_model(b.SerializeToString()),
            25,
        )
        values.append(result)
    assert values[0][0].bytes == values[1][0].bytes
    assert values[0][1:] == values[1][1:] == (75, 25)
    x = np.arange(45, dtype=np.float32).reshape(1, 3, 3, 5) / np.float32(44)
    with closing(NativeSession(values[1][0].bytes)) as session:
        actual = session.run(None, {"x": x})[0]
    baseline = ort.InferenceSession(
        values[0][0].bytes, providers=["CPUExecutionProvider"]
    )
    expected = baseline.run(None, {"x": x})[0]
    assert isinstance(expected, np.ndarray)
    assert actual.tobytes() == expected.tobytes()


def test_concurrent_independent_model_application():
    def run(index):
        original, native = pair()
        arrays = [numpy_helper.from_array(np.arange(32, dtype=np.float32), "weight")]
        a = original.perform_interp(arrays, arrays, index)
        b = native.perform_interp(arrays, arrays, index)
        assert a[0].SerializeToString() == b[0].SerializeToString()
        source = tiny_model(index)
        actual = current_load.load_onnx_model(copy.deepcopy(source))
        expected = old_load.load_onnx_model(copy.deepcopy(source))
        assert info(actual) == info(expected)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(12)))


def test_registered_node_contract_unchanged():
    def contract(path):
        tree = ast.parse(path.read_text("utf-8"))
        node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "interpolate_models_node"
        )
        assert node.returns is not None
        return (
            ast.dump(node.args),
            [ast.dump(value) for value in node.decorator_list],
            ast.dump(node.returns),
        )

    assert contract(CURRENT_INTERP) == contract(FROZEN / "interpolate_models.py")


def test_fixed_probe_property_access_order(monkeypatch):
    observations = []
    for module in (old_load, current_load):
        events = []

        class OrderedProbe:
            def __getattribute__(self, name, events=events):
                events.append(name)
                if name == "fixed_input_width":
                    return 7
                if name == "fixed_input_height":
                    return 11
                return object.__getattribute__(self, name)

            def infer_shape(self, size, events=events):
                events.append(size)
                return KNOWN

        monkeypatch.setattr(module, "logger", Recorder())
        requirement, shape = module._detect_size_req(OrderedProbe())
        observations.append((events, vars(requirement), shape))
    assert observations[0] == observations[1]


def test_probe_requirement_factory_failure_is_logged_and_retried(monkeypatch):
    observations = []
    for module in (old_load, current_load):
        events, log = [], Recorder()
        constructor = module.SizeReq

        def requirement(*, multiple_of=1, constructor=constructor, events=events):
            events.append(multiple_of)
            if multiple_of == 16:
                raise ValueError("requirement failed")
            return constructor(multiple_of=multiple_of)

        monkeypatch.setattr(module, "SizeReq", requirement)
        monkeypatch.setattr(module, "logger", log)
        probe = Probe([KNOWN, KNOWN])
        result = module._detect_size_req(probe)
        observations.append(
            (vars(result[0]), result[1], events, probe.calls, log.events)
        )
    assert observations[0] == observations[1]


def test_probe_unpack_consumes_only_one_extra_iterator_element(monkeypatch):
    observations = []
    for module in (old_load, current_load):
        events, log = [], Recorder()

        def dimensions(events=events):
            for i in range(8):
                events.append(i)
                yield i

        probe = Probe([((), dimensions()) for _ in range(4)])
        monkeypatch.setattr(module, "logger", log)
        result = module._detect_size_req(probe)
        observations.append((vars(result[0]), result[1], events, log.events))
    assert observations[0] == observations[1]
    assert observations[0][2] == [0, 1, 2, 3] * 4


@pytest.mark.parametrize("failure", ["pop", "extend", "load", None])
def test_initializer_rebuild_partial_failures(failure):
    observations = []
    for module in pair():
        events, rebuilt = [], []
        wa = [
            numpy_helper.from_array(np.array([i], np.float32), f"w{i}")
            for i in range(3)
        ]
        wb = copy.deepcopy(wa)
        a = types.SimpleNamespace(graph=types.SimpleNamespace(initializer=wa))
        b = types.SimpleNamespace(graph=types.SimpleNamespace(initializer=wb))

        class Initializers(list):
            def pop(self, events=events):
                events.append(("pop", len(self)))
                if failure == "pop" and len(self) == 2:
                    raise ValueError("pop failed")
                return super().pop()

            def extend(self, items, events=events):
                for i, item in enumerate(items):
                    events.append(("extend", item.name))
                    if failure == "extend" and i == 1:
                        raise ValueError("extend failed")
                    self.append(copy.deepcopy(item))

        def deep(value, rebuilt=rebuilt):
            result = types.SimpleNamespace(
                graph=types.SimpleNamespace(
                    initializer=Initializers(copy.deepcopy(value.graph.initializer))
                )
            )
            rebuilt.append(result)
            return result

        def load(value, events=events):
            events.append(("load", len(value.graph.initializer)))
            if failure == "load":
                raise ValueError("load failed")
            return value

        module.__dict__.update(
            onnx=types.SimpleNamespace(
                load_from_string=lambda data, a=a, b=b: a if data == b"a" else b
            ),
            deepcopy=deep,
            load_onnx_model=load,
            check_will_upscale=lambda *_: True,
        )
        value, error, caught = outcome(
            lambda module=module: module.interpolate_models_node(
                None,
                types.SimpleNamespace(bytes=b"a"),
                types.SimpleNamespace(bytes=b"b"),
                25,
            )
        )
        observations.append(
            (
                events,
                error,
                caught,
                [t.SerializeToString() for t in rebuilt[0].graph.initializer],
                [t.SerializeToString() for t in wb],
                None if value is None else value[1:],
            )
        )
        assert rebuilt[0].graph.initializer is not wb
    assert observations[0] == observations[1]
