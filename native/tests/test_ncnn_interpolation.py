"""NCNN native interpolation: exact data, aliases, and failure mutations."""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.typing import DTypeLike
from test_ncnn_graph import build, outcome, port, reference, snapshot, spec
from test_ncnn_graph_numeric import DTYPES, ErrorPolicy, values


def pair(
    module,
    left_dtype: DTypeLike = np.float32,
    right_dtype: DTypeLike = np.float32,
    shape=(2, 3),
    tags=(b"", b""),
):
    def one(dtype, seed, tag):
        return build(
            module,
            [
                spec(
                    "Convolution",
                    "weight",
                    ["in"],
                    ["out"],
                    {0: 2, 1: 1, 5: 1, 6: 6},
                    {
                        "weight": (values(dtype, shape, seed), tag),
                        "bias": (values(dtype, shape, seed + 1), b""),
                    },
                )
            ],
        ).layers[0]

    return one(left_dtype, 107, tags[0]), one(right_dtype, 231, tags[1])


def compare(setup, amount, mode: ErrorPolicy = "ignore", model=False):
    observations = []
    for module in (reference, port):
        a, b = setup(module)
        with np.errstate(all=mode), warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            result = outcome(
                lambda a=a, b=b, module=module: (
                    a.interpolate(b, amount)
                    if model
                    else module.NcnnModel.interp_layers(a, b, amount)
                )
            )
        observations.append(
            (
                snapshot((a, b, result)),
                [(w.category.__name__, str(w.message)) for w in recorded],
            )
        )
    assert observations[0] == observations[1]


@pytest.mark.parametrize("left_dtype", DTYPES)
@pytest.mark.parametrize("right_dtype", DTYPES)
@pytest.mark.parametrize("amount", [0, 0.0, 0.23, 1.0, 2])
def test_numeric_interpolation(left_dtype, right_dtype, amount):
    compare(lambda module: pair(module, left_dtype, right_dtype), amount)


@pytest.mark.parametrize(
    "tag_a", [b"", port.DTYPE_FP16, port.DTYPE_FP32, b"tag", b"xxxx"]
)
@pytest.mark.parametrize(
    "tag_b", [b"", port.DTYPE_FP16, port.DTYPE_FP32, b"tag", b"xxxx"]
)
@pytest.mark.parametrize("dtype", [np.int64, np.float16, np.float32, np.complex64])
@pytest.mark.parametrize("mode", ["ignore", "warn", "raise"])
def test_quantize_tag_mutations(dtype, tag_a, tag_b, mode):
    compare(lambda module: pair(module, dtype, dtype, tags=(tag_a, tag_b)), 0.13, mode)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (0,), (2, 0), (1,), (2, 3)])
@pytest.mark.parametrize("amount", [0.31, np.inf, np.nan])
def test_scalar_empty_exceptional_coefficient(dtype, shape, amount):
    compare(lambda module: pair(module, dtype, dtype, shape), amount, "warn")


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.complex128, np.uint16])
@pytest.mark.parametrize(
    "layout", ["fortran", "reverse", "readonly", "byteswap", "shared"]
)
def test_foreign_and_shared_weights(dtype, layout):
    def setup(module):
        a, b = pair(module, dtype, dtype)
        for layer in (a, b):
            for weight in layer.weight_data.values():
                data = weight.weight
                if layout == "fortran":
                    weight.weight = np.asfortranarray(data)
                elif layout == "reverse":
                    weight.weight = data[::-1, ::-1]
                elif layout == "readonly":
                    data.flags.writeable = False
                elif layout == "byteswap":
                    weight.weight = data.byteswap().view(data.dtype.newbyteorder())
        if layout == "shared":
            b.weight_data = a.weight_data
        return a, b

    compare(setup, 0.5)


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "empty_a",
        "empty_b",
        "key",
        "shape",
        "tag",
        "late_key",
        "late_shape",
        "overflow_cast",
    ],
)
def test_error_state_and_aliases(mutation):
    def setup(module):
        a, b = pair(
            module, np.float32, np.float32, tags=(module.DTYPE_FP32, module.DTYPE_FP16)
        )
        if mutation == "empty_a":
            a.weight_data.clear()
        elif mutation == "empty_b":
            b.weight_data.clear()
        elif mutation in {"key", "late_key"}:
            key = "weight" if mutation == "key" else "bias"
            b.weight_data["renamed"] = b.weight_data.pop(key)
        elif mutation in {"shape", "late_shape"}:
            key = "weight" if mutation == "shape" else "bias"
            b.weight_data[key].weight = b.weight_data[key].weight.reshape(3, 2)
        elif mutation == "tag":
            b.weight_data["weight"].quantize_tag = b""
        elif mutation == "overflow_cast":
            a.weight_data["weight"].weight[:] = 1e20
        a.extra = {"aliased": a.inputs}
        return a, b

    compare(setup, 0.5, "raise")


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "different_locations",
        "extra_unweighted",
        "extra_weighted",
        "no_weights",
        "same_model",
        "tag",
        "late_shape",
    ],
)
@pytest.mark.parametrize("amount", [0.0, 0.37, 1.0])
def test_model_pairing_copy_protocol_and_errors(mutation, amount):
    def setup(module):
        x, y = pair(
            module, np.float32, np.float16, tags=(module.DTYPE_FP32, module.DTYPE_FP16)
        )
        a, b = module.NcnnModel(17, 23), module.NcnnModel(19, 29)
        a.layers = [
            module.NcnnLayer("Input", "input"),
            x,
            module.NcnnLayer("Noop", "tail"),
        ]
        b.layers = [
            module.NcnnLayer("Input", "input"),
            y,
            module.NcnnLayer("Noop", "tail"),
        ]
        a.extra = {
            "self": a,
            "input_alias": x.inputs,
            "array_alias": x.weight_data["weight"].weight,
        }
        a.bin_length = 987
        if mutation == "different_locations":
            b.layers.reverse()
        elif mutation == "extra_unweighted":
            b.layers.append(module.NcnnLayer("Noop", "spare"))
        elif mutation == "extra_weighted":
            b.layers.append(y)
        elif mutation == "no_weights":
            a.layers.clear()
            b.layers.clear()
        elif mutation == "same_model":
            b = a
        elif mutation == "tag":
            y.weight_data["bias"].quantize_tag = b"bad"
        elif mutation == "late_shape":
            y.weight_data["bias"].weight = y.weight_data["bias"].weight.reshape(3, 2)
        return a, b

    compare(setup, amount, model=True)


def test_concurrent_repeatability():
    def run(_):
        a, b = pair(port, np.complex128, np.float32)
        result = port.NcnnModel.interp_layers(a, b, 0.73)
        return snapshot((a, b, result))

    expected = run(0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(value == expected for value in pool.map(run, range(32)))
