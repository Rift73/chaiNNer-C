"""CPU-only coefficient interpolation; no model loading/inference or GPU calls.

Real ONNX TensorProto serialization and NCNN model/layer containers are exercised.
Torch tests use small CPU state dictionaries and a metadata-only tensor to verify
that unsupported devices are never transferred into the C/NumPy path.
"""

from __future__ import annotations

import ast
import ctypes as ct
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from onnx import numpy_helper as onph

from nodes.impl import native_tensors as native
from nodes.impl.native_graph import graph
from nodes.impl.ncnn import model as ncnn

ROOT = Path(__file__).resolve().parents[2]
REF = Path(__file__).with_name("reference_tensors")
ONNX_PATH = (
    ROOT / "backend/src/packages/chaiNNer_onnx/onnx/utility/interpolate_models.py"
)
TORCH_PATH = (
    ROOT / "backend/src/packages/chaiNNer_pytorch/pytorch/utility/interpolate_models.py"
)


def function(path, name):
    tree = ast.parse(path.read_text("utf-8"))
    definition = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    definition.decorator_list = []
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        definition,
    ]
    tree = ast.fix_missing_locations(tree)
    context: dict[str, Any] = {
        "np": np,
        "torch": torch,
        "onph": onph,
        "interpolate_numpy": native.interpolate_numpy,
        "interpolate_torch": native.interpolate_torch,
        "cast_numpy": native.cast_numpy,
        "graph": graph,
    }
    exec(compile(tree, str(path), "exec"), context)
    return context[name]


def ncnn_reference():
    module = types.ModuleType("nodes.impl.ncnn._reference_tensor_model")
    module.__package__ = "nodes.impl.ncnn"
    # The frozen code reads the unchanged format schema adjacent to __file__.
    module.__file__ = ncnn.__file__
    tree = ast.parse((REF / "ncnn_model.py").read_text("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "logger":
            # The frozen source logged through a module the shipped tree lacks.
            node.module = "sanic.log"
    exec(compile(tree, str(REF / "ncnn_model.py"), "exec"), module.__dict__)
    return module


NCNN_REFERENCE = ncnn_reference()
ONNX = function(ONNX_PATH, "perform_interp")
ORIGINAL_ONNX = function(REF / "onnx_interpolate.py", "perform_interp")
TORCH = function(TORCH_PATH, "perform_interp")
ORIGINAL_TORCH = function(REF / "pytorch_interpolate.py", "perform_interp")


def array(dtype, layout="plain", shape=(9, 13), seed=731):
    out = np.random.default_rng(seed).uniform(-3, 3, shape).astype(dtype)
    if layout == "strided":
        out = out[::-2, ::2]
    elif layout == "fortran":
        out = np.asfortranarray(out)
    elif layout == "readonly":
        out.flags.writeable = False
    elif layout == "unaligned":
        other = np.ndarray(
            out.shape, dtype=out.dtype, buffer=bytearray(out.nbytes + 1), offset=1
        )
        other[...] = out
        out = other
    return out


def exact(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    np.testing.assert_array_equal(a, b)
    if np.issubdtype(a.dtype, np.floating):
        # Include signed zero and all finite rounding bits in the comparison.
        finite = np.isfinite(b)
        assert (
            np.ascontiguousarray(a[finite]).tobytes()
            == np.ascontiguousarray(b[finite]).tobytes()
        )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize(
    "layout", ["plain", "strided", "fortran", "readonly", "unaligned"]
)
@pytest.mark.parametrize("amount", [0, 0.01, 0.3333, 0.5, 0.99, 1, -0.25, 1.25])
def test_tensor_interpolation(dtype, layout, amount):
    a, b = array(dtype, layout), array(dtype, layout, seed=981)
    before_a, before_b = a.copy(), b.copy()
    exact(
        native.interpolate_numpy(a, b, amount, 1 - amount),
        a * amount + b * (1 - amount),
    )
    exact(a, before_a)
    exact(b, before_b)


@pytest.mark.parametrize(
    "amount", [0, 0.1, 0.3333, 0.5, 1, 0.333251953125 + 1e-10, 0.333251953125 - 1e-10]
)
def test_all_half_patterns(amount):
    a = np.arange(65536, dtype=np.uint16).view(np.float16)
    b = np.roll(a, 1729)
    with np.errstate(all="ignore"):
        exact(
            native.interpolate_numpy(a, b, amount, 1 - amount),
            a * amount + b * (1 - amount),
        )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("amount", [0, 0.19, 0.5, 1])
def test_nan_inf_and_zero_coefficients(dtype, amount):
    a = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1, -1], dtype=dtype)
    b = a[::-1]
    with np.errstate(all="ignore"):
        exact(
            native.interpolate_numpy(a, b, amount, 1 - amount),
            a * amount + b * (1 - amount),
        )


@pytest.mark.parametrize(
    "dtype",
    [np.float16, np.float32, np.float64, np.int32, np.int64, np.uint8, np.complex64],
)
@pytest.mark.parametrize("shape", [(), (0,), (3, 0), (2, 3)])
def test_scalar_empty_and_retained_dtype_contract(dtype, shape):
    a = np.ones(shape, dtype=dtype)
    b = np.full(shape, 2, dtype=dtype)
    expected = a * 0.23 + b * 0.77
    actual = native.interpolate_numpy(a, b, 0.23, 0.77)
    assert type(actual) is type(expected)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize(
    "dtype_a,dtype_b",
    [(np.float16, np.float32), (np.float32, np.float64), (np.int64, np.float32)],
)
def test_retained_mixed_dtype_and_broadcast(dtype_a, dtype_b):
    a, b = np.ones((2, 3), dtype_a), np.arange(3, dtype=dtype_b)
    exact(native.interpolate_numpy(a, b, 0.13, 0.87), a * 0.13 + b * 0.87)


@pytest.mark.parametrize(
    "dtype",
    [np.float16, np.float32, np.float64, np.int32, np.int64, np.uint8, np.complex64],
)
@pytest.mark.parametrize("amount", [0, 17, 50, 100])
def test_real_onnx_initializers(dtype, amount):
    a = [
        onph.from_array(array(dtype), "a_kernel"),
        onph.from_array(np.array(3, dtype), "a_scalar"),
    ]
    b = [
        onph.from_array(array(dtype, seed=712), "b_kernel"),
        onph.from_array(np.array(7, dtype), "b_scalar"),
    ]
    before = [w.SerializeToString() for w in (*a, *b)]
    actual, expected = ONNX(a, b, amount), ORIGINAL_ONNX(a, b, amount)
    assert [w.SerializeToString() for w in actual] == [
        w.SerializeToString() for w in expected
    ]
    assert [w.SerializeToString() for w in (*a, *b)] == before


def layer(module, dtype, tag, *, bias=True, seed=731):
    weights = {"weight": module.NcnnWeight(array(dtype, seed=seed), tag)}
    if bias:
        weights["bias"] = module.NcnnWeight(np.arange(9, dtype=np.float32), b"")
    return module.NcnnLayer(
        "Convolution", "fixture", 1, 1, ["input"], ["output"], weight_data=weights
    )


@pytest.mark.parametrize(
    "dtype_a,dtype_b",
    [
        (np.float16, np.float16),
        (np.float32, np.float32),
        (np.float16, np.float32),
        (np.float32, np.float16),
    ],
)
@pytest.mark.parametrize("amount", [0, 0.13, 0.5, 1])
def test_ncnn_layers_bytes_and_original_mutations(dtype_a, dtype_b, amount):
    def execute(module):
        a = layer(
            module,
            dtype_a,
            module.DTYPE_FP16 if dtype_a == np.float16 else module.DTYPE_FP32,
        )
        b = layer(
            module,
            dtype_b,
            module.DTYPE_FP16 if dtype_b == np.float16 else module.DTYPE_FP32,
            seed=927,
        )
        output, encoded = module.NcnnModel.interp_layers(a, b, amount)
        return output, encoded, a, b

    actual, expected = execute(ncnn), execute(NCNN_REFERENCE)
    assert actual[1] == expected[1]
    layers = (
        (actual[0], expected[0]),
        (actual[2], expected[2]),
        (actual[3], expected[3]),
    )
    for actual_layer, expected_layer in layers:
        for key, a in actual_layer.weight_data.items():
            b = expected_layer.weight_data[key]
            assert a.quantize_tag == b.quantize_tag
            exact(a.weight, b.weight)


def test_ncnn_model_container_without_inference():
    def execute(module):
        a, b = module.NcnnModel(1, 1), module.NcnnModel(1, 1)
        a.layers = [layer(module, np.float32, module.DTYPE_FP32)]
        b.layers = [layer(module, np.float32, module.DTYPE_FP32, seed=729)]
        return a.interpolate(b, 0.29)

    actual, expected = execute(ncnn), execute(NCNN_REFERENCE)
    assert actual.serialize_weights() == expected.serialize_weights()
    assert actual.layers[0].name == expected.layers[0].name


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.int64, torch.complex64]
)
@pytest.mark.parametrize("layout", ["plain", "strided", "scalar", "empty"])
@pytest.mark.parametrize("amount", [0, 19, 50, 100])
def test_torch_cpu_state_dict(dtype, layout, amount):
    a = torch.arange(24, dtype=torch.float64).reshape(4, 6).to(dtype=dtype)
    b = torch.flip(a, (0, 1)) / 3 if dtype != torch.int64 else torch.flip(a, (0, 1))
    if layout == "strided":
        a, b = a[:, ::2], b[:, ::2]
    elif layout == "scalar":
        a, b = a[0, 0], b[0, 0]
    elif layout == "empty":
        a, b = a[:0], b[:0]
    actual, expected = (
        TORCH({"weight": a}, {"weight": b}, amount),
        ORIGINAL_TORCH({"weight": a}, {"weight": b}, amount),
    )
    torch.testing.assert_close(
        actual["weight"], expected["weight"], rtol=0, atol=0, equal_nan=True
    )
    assert actual["weight"].device == expected["weight"].device


def test_torch_retains_autograd_and_meta_device(monkeypatch):
    a = torch.arange(12, dtype=torch.float32, requires_grad=True)
    b = torch.ones(12, requires_grad=True)
    result = native.interpolate_torch(a, b, 0.25, 0.75)
    result.sum().backward()
    torch.testing.assert_close(a.grad, torch.full((12,), 0.25), rtol=0, atol=0)
    torch.testing.assert_close(b.grad, torch.full((12,), 0.75), rtol=0, atol=0)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported tensor must not enter NumPy/native code")

    monkeypatch.setattr(native, "interpolate_numpy", forbidden)
    a = torch.empty((2, 3), device="meta")
    result = native.interpolate_torch(a, a, 0.5, 0.5)
    assert result.device.type == "meta"
    result = native.interpolate_torch(
        torch.arange(6, dtype=torch.float32)[::2], torch.ones(3), 0.5, 0.5
    )
    torch.testing.assert_close(result, torch.tensor([0.5, 1.5, 2.5]), rtol=0, atol=0)


def test_concurrent_large_tensor_interpolation():
    inputs = [
        (array(dtype, shape=(257, 513)), array(dtype, shape=(257, 513), seed=981))
        for dtype in (np.float16, np.float32, np.float64)
    ]
    expected = [a * 0.37 + b * 0.63 for a, b in inputs]

    def interpolate(i: int) -> np.ndarray:
        return native.interpolate_numpy(*inputs[i % 3], 0.37, 0.63)

    with ThreadPoolExecutor(max_workers=6) as pool:
        result = list(pool.map(interpolate, range(18)))
    for i, actual in enumerate(result):
        exact(actual, expected[i % 3])


def test_unsupported_and_invalid_contracts(monkeypatch):
    dll = native._api()  # raw ABI rejection must precede writes
    assert dll.cn_tensor_interpolate(None, None, None, 1, 0, 0.5, 0.5) == 1
    data = np.zeros(4, np.float64)
    assert (
        dll.cn_tensor_interpolate(
            data.ctypes.data,
            data.ctypes.data,
            data.ctypes.data,
            (1 << (ct.sizeof(ct.c_size_t) * 8)) - 1,
            2,
            0.5,
            0.5,
        )
        == 2
    )
    assert (
        dll.cn_tensor_interpolate(
            data.ctypes.data, data.ctypes.data, data.ctypes.data, 4, 3, 0.5, 0.5
        )
        == 1
    )
    np.testing.assert_array_equal(data, 0)
    with pytest.raises(AssertionError, match="same size and shape"):
        ONNX(
            [onph.from_array(np.ones(2, np.float32))],
            [onph.from_array(np.ones(3, np.float32))],
            50,
        )
    with pytest.raises(ValueError, match="not compatible"):
        TORCH({"a": torch.ones(2)}, {"b": torch.ones(2)}, 50)

    def broken():
        raise RuntimeError("test native load failure")

    monkeypatch.setattr(native, "_api", broken)
    with pytest.raises(RuntimeError, match="test native load failure"):
        native.interpolate_numpy(
            np.ones(3, np.float32), np.ones(3, np.float32), 0.5, 0.5
        )


@pytest.mark.parametrize(
    "current,reference",
    [
        (ONNX_PATH, REF / "onnx_interpolate.py"),
        (TORCH_PATH, REF / "pytorch_interpolate.py"),
    ],
)
def test_interpolation_node_contract_unchanged(current, reference):
    def node(path):
        return next(
            n
            for n in ast.parse(path.read_text("utf-8")).body
            if isinstance(n, ast.FunctionDef) and n.name == "interpolate_models_node"
        )

    a, b = node(current), node(reference)
    assert ast.dump(a.args) == ast.dump(b.args)
    assert [ast.dump(d) for d in a.decorator_list] == [
        ast.dump(d) for d in b.decorator_list
    ]
