"""Numeric NCNN compatibility exercised through real optimizer passes."""

from __future__ import annotations

import copy
import warnings
from typing import Literal

import numpy as np
import pytest
from test_ncnn_graph import (
    CASES,
    NcnnOptimizer,
    ReferenceOptimizer,
    build,
    outcome,
    port,
    reference,
    snapshot,
)

DTYPES = [
    np.bool_,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    np.longdouble,
    np.clongdouble,
]
PASSES = [
    "fuse_batchnorm_scale",
    "fuse_x_batchnorm",
    "fuse_x_mul",
    "fuse_x_add",
    "fuse_innerproduct_dropout",
]


type ErrorPolicy = Literal["ignore", "warn", "raise", "call", "print", "log"]


def run_pair(specification, name, mode: ErrorPolicy = "ignore", transform=None):
    a, b = build(reference, specification), build(port, specification)
    if transform:
        transform(a)
        transform(b)
    with np.errstate(all=mode), warnings.catch_warnings(record=True) as wa:
        warnings.simplefilter("always")
        ra = outcome(getattr(ReferenceOptimizer(a), "_NcnnOptimizer__" + name))
    with np.errstate(all=mode), warnings.catch_warnings(record=True) as wb:
        warnings.simplefilter("always")
        rb = outcome(getattr(NcnnOptimizer(b), "_NcnnOptimizer__" + name))
    assert ra == rb
    assert [(w.category.__name__, str(w.message)) for w in wa] == [
        (w.category.__name__, str(w.message)) for w in wb
    ]
    assert snapshot(a) == snapshot(b)
    return a, b


def values(dtype, shape, seed):
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    if dtype.kind == "b":
        return rng.integers(0, 2, shape).astype(dtype)
    if dtype.kind in "ui":
        # Full-width bit patterns exercise wraparound, not only small integers.
        return (
            np.frombuffer(rng.bytes(int(np.prod(shape)) * dtype.itemsize), dtype)
            .copy()
            .reshape(shape)
        )
    if dtype.kind == "c":
        return (rng.uniform(-7, 7, shape) + 1j * rng.uniform(-7, 7, shape)).astype(
            dtype
        )
    return rng.uniform(-7, 7, shape).astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", PASSES)
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("mode", ["ignore", "raise"])
def test_numeric_optimizer_types(dtype, name, seed, mode):
    specification = copy.deepcopy(CASES[name])
    for i, layer in enumerate(specification):
        for j, (key, (array, tag)) in enumerate(list(layer[5].items())):
            layer[5][key] = (values(dtype, array.shape, seed + i * 37 + j * 101), tag)
    run_pair(specification, name, mode)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (1,), (0,), (2, 0)])
@pytest.mark.parametrize("name", ["fuse_batchnorm_scale", "fuse_innerproduct_dropout"])
def test_scalar_empty_result_identity(dtype, shape, name):
    specification = copy.deepcopy(CASES[name])
    for i, layer in enumerate(specification):
        for j, (key, (_, tag)) in enumerate(list(layer[5].items())):
            layer[5][key] = (values(dtype, shape, i + j + 2), tag)
    run_pair(specification, name)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", PASSES)
@pytest.mark.parametrize("layout", ["reverse", "fortran", "readonly", "byteswap"])
def test_numeric_weight_layouts(dtype, name, layout):
    specification = copy.deepcopy(CASES[name])
    for i, layer in enumerate(specification):
        for j, (key, (array, tag)) in enumerate(list(layer[5].items())):
            layer[5][key] = (values(dtype, array.shape, i + j + 1), tag)

    def transform(model):
        for layer in model.layers:
            for weight in layer.weight_data.values():
                data = weight.weight
                if layout == "reverse":
                    weight.weight = data[::-1]
                elif layout == "fortran":
                    weight.weight = np.asfortranarray(data)
                elif layout == "byteswap":
                    weight.weight = data.byteswap().view(data.dtype.newbyteorder())
                else:
                    data.flags.writeable = False

    run_pair(specification, name, transform=transform)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("real", [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("imag", [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("mode", ["ignore", "raise", "warn"])
def test_complex_special_values(dtype, real, imag, mode):
    specification = copy.deepcopy(CASES["fuse_x_batchnorm"])
    for layer in specification:
        for key, (array, tag) in list(layer[5].items()):
            layer[5][key] = (array.astype(dtype), tag)
    specification[1][5]["variance"][0][:] = complex(real, imag)
    specification[1][4][1] = 0.0
    run_pair(specification, "fuse_x_batchnorm", mode)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("name", PASSES)
@pytest.mark.parametrize("mode", ["ignore", "raise", "warn"])
@pytest.mark.parametrize("size", ["tiny", "huge", "largest"])
def test_complex_magnitude_boundaries(dtype, name, mode, size):
    specification = copy.deepcopy(CASES[name])
    component = np.dtype(dtype).type(0).real.dtype
    finfo = np.finfo(component)
    value = (
        finfo.tiny
        if size == "tiny"
        else finfo.max
        if size == "largest"
        else np.sqrt(finfo.max)
    )
    for layer in specification:
        for key, (array, tag) in list(layer[5].items()):
            new = np.full(array.shape, complex(value, -value), dtype)
            layer[5][key] = (new, tag)
    run_pair(specification, name, mode)


@pytest.mark.parametrize("left_dtype", DTYPES)
@pytest.mark.parametrize("right_dtype", DTYPES)
@pytest.mark.parametrize("name", ["fuse_batchnorm_scale", "fuse_x_batchnorm"])
@pytest.mark.parametrize("mode", ["ignore", "raise"])
def test_mixed_dtype_arithmetic(left_dtype, right_dtype, name, mode):
    specification = copy.deepcopy(CASES[name])
    for i, layer in enumerate(specification):
        for j, (key, (array, tag)) in enumerate(list(layer[5].items())):
            dtype = right_dtype if j % 2 else left_dtype
            layer[5][key] = (values(dtype, array.shape, i + j + 41), tag)
    run_pair(specification, name, mode)


@pytest.mark.parametrize("dtype", [np.int8, np.float32, np.complex64])
@pytest.mark.parametrize("bias_shape", [(), (1,), (1, 2), (2, 1), (3,), (0,)])
def test_inplace_add_rejection_state(dtype, bias_shape):
    specification = copy.deepcopy(CASES["fuse_batchnorm_scale"])
    for i, layer in enumerate(specification):
        for j, (key, (array, tag)) in enumerate(list(layer[5].items())):
            shape = bias_shape if i == 1 and key == "bias" else array.shape
            layer[5][key] = (values(dtype, shape, i + j + 41), tag)
    run_pair(specification, "fuse_batchnorm_scale")


@pytest.mark.parametrize("left_dtype", DTYPES)
@pytest.mark.parametrize("right_dtype", DTYPES)
@pytest.mark.parametrize("shape", [(2,), (3,)])
def test_inplace_dtype_resolution_error_order(left_dtype, right_dtype, shape):
    specification = copy.deepcopy(CASES["fuse_batchnorm_scale"])
    for i, layer in enumerate(specification):
        for j, (key, (array, tag)) in enumerate(list(layer[5].items())):
            dtype = right_dtype if i == 1 and key == "bias" else left_dtype
            outshape = shape if i == 1 and key == "bias" else array.shape
            layer[5][key] = (values(dtype, outshape, i + j + 37), tag)
    run_pair(specification, "fuse_batchnorm_scale", "raise")


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("name", PASSES)
@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("mode", ["ignore", "raise"])
def test_complex_arbitrary_component_bits(dtype, name, seed, mode):
    specification = copy.deepcopy(CASES[name])
    rng = np.random.default_rng(seed + 1038)
    for layer in specification:
        for key, (array, tag) in list(layer[5].items()):
            bits = rng.bytes(array.size * np.dtype(dtype).itemsize)
            data = np.frombuffer(bits, dtype=dtype).copy().reshape(array.shape)
            layer[5][key] = (data, tag)
    run_pair(specification, name, mode)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("name", PASSES)
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("signaling", [False, True])
@pytest.mark.parametrize("component", [0, 1])
@pytest.mark.parametrize("mode", ["ignore", "raise", "warn"])
def test_complex_nan_payloads(dtype, name, signed, signaling, component, mode):
    specification = copy.deepcopy(CASES[name])
    word_type = np.uint32 if dtype == np.complex64 else np.uint64
    bits = 32 if dtype == np.complex64 else 64
    exponent = 0x7F800000 if bits == 32 else 0x7FF0000000000000
    quiet = 0 if signaling else 1 << (bits - (10 if bits == 32 else 13))
    payload = exponent | quiet | 0x3471 | (int(signed) << (bits - 1))
    for layer in specification:
        for key, (array, tag) in list(layer[5].items()):
            data = array.astype(dtype)
            data.view(word_type).reshape(-1, 2)[:, component] = payload
            layer[5][key] = (data, tag)
    run_pair(specification, name, mode)
