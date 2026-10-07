"""Independent NumPy/Torch CPU arithmetic oracles; no inference, models or GPU."""

from __future__ import annotations

import ast
import ctypes as ct
import itertools
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from nodes.impl import native_tensors as native
from nodes.impl.native_analysis import mean as native_mean
from nodes.impl.native_graph import graph

DTYPES = tuple(native._TYPES)
LAYOUTS = (
    "plain",
    "fortran",
    "strided",
    "readonly",
    "unaligned",
    "scalar",
    "empty",
    "broadcast",
)


def values(dtype, layout="plain", seed=781):
    rng = np.random.default_rng(seed)
    if dtype.kind in "iu":
        # Include large integers whose conversion to f32 must avoid a double detour.
        out = rng.integers(0, 2**64, size=(7, 9), dtype=np.uint64).astype(dtype)
    elif dtype.kind == "b":
        out = rng.integers(0, 2, (7, 9)).astype(dtype)
    elif dtype.kind == "c":
        out = (rng.normal(size=(7, 9)) + 1j * rng.normal(size=(7, 9))).astype(dtype)
    else:
        out = rng.normal(size=(7, 9)).astype(dtype)
    if layout == "fortran":
        out = np.asfortranarray(out)
    elif layout == "strided":
        out = out[::-2, ::2]
    elif layout == "readonly":
        out.flags.writeable = False
    elif layout == "unaligned":
        other = np.ndarray(out.shape, dtype, buffer=bytearray(out.nbytes + 1), offset=1)
        other[...] = out
        out = other
    elif layout == "scalar":
        out = np.asarray(out[0, 0])
    elif layout == "empty":
        out = out[:0]
    elif layout == "broadcast":
        out = np.broadcast_to(out[:1, :], (7, 9))
    return out


def exact(actual, expected):
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    a, b = np.asarray(actual), np.asarray(expected)
    if a.dtype.kind in "fc":
        if a.dtype.kind == "c":
            a, b = (
                np.ascontiguousarray(a).view(a.real.dtype),
                np.ascontiguousarray(b).view(b.real.dtype),
            )
        finite = np.isfinite(b)
        assert a[finite].tobytes() == b[finite].tobytes()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("amount", (0.0, 0.13, 0.3333, 0.5, 1.0))
def test_numpy_numeric_contract(dtype, layout, amount):
    a, b = values(dtype, layout), values(dtype, layout, 961)
    before = a.tobytes(), b.tobytes()
    with np.errstate(all="ignore"):
        actual = native.interpolate_numpy(a, b, amount, 1 - amount)
        expected = a * amount + b * (1 - amount)
    exact(actual, expected)
    assert actual.strides == expected.strides  # a NumPy scalar's strides are ()
    assert before == (a.tobytes(), b.tobytes())


@pytest.mark.parametrize("left,right", itertools.product(DTYPES, DTYPES))
@pytest.mark.parametrize("scalar_side", ("none", "left", "right"))
def test_numpy_mixed_promotion(left, right, scalar_side):
    a = values(left, "scalar" if scalar_side == "left" else "plain")
    b = values(right, "scalar" if scalar_side == "right" else "plain", 192)
    with np.errstate(all="ignore"):
        exact(native.interpolate_numpy(a, b, 0.19, 0.81), a * 0.19 + b * 0.81)


@pytest.mark.parametrize("dtype", ("float16", "float32", "float64", "complex64"))
@pytest.mark.parametrize("layout", ("plain", "fortran", "strided", "scalar"))
@pytest.mark.parametrize("amount", (2**70, -(2**80), 2**200))
def test_numpy_object_coefficient(dtype, layout, amount):
    # A Python int beyond 64 bits is an object-dtype coefficient, cast by NumPy.
    a = values(np.dtype(dtype), layout)
    b = values(np.dtype(dtype), layout, 961)
    with np.errstate(all="ignore"):
        actual = native.interpolate_numpy(a, b, amount, 0.5)
        expected = a * amount + b * 0.5
    exact(actual, expected)
    assert actual.strides == expected.strides  # () for the scalar layout's NumPy scalar


@pytest.mark.parametrize("source,target", itertools.product(DTYPES, DTYPES))
def test_casts(source, target):
    a = values(source, "strided")
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        actual, expected = native.cast_numpy(a, target), a.astype(target)
    exact(actual, expected)
    assert actual.strides == expected.strides


@pytest.mark.parametrize("source", (np.float16, np.float32, np.float64))
@pytest.mark.parametrize("target", DTYPES)
def test_exceptional_casts(source, target):
    with np.errstate(all="ignore"):
        a = np.array(
            [
                np.nan,
                np.inf,
                -np.inf,
                0.0,
                -0.0,
                -0.5,
                -1.0,
                1e100,
                -1e100,
                2**31,
                2**32,
                2**63,
                2**64,
            ],
            dtype=source,
        )
        exact(native.cast_numpy(a, target), a.astype(target))


TORCH_TYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


@pytest.mark.parametrize("dtype", TORCH_TYPES)
@pytest.mark.parametrize("layout", ("plain", "strided", "transpose", "scalar", "empty"))
@pytest.mark.parametrize("amount", (0.0, 0.13, 0.3333, 0.5, 1.0))
def test_torch_numeric_contract(dtype, layout, amount):
    a = torch.from_numpy(
        values(np.dtype("complex128" if dtype.is_complex else "float64"))
    )
    if not dtype.is_floating_point and not dtype.is_complex:
        a = a * 1000000000
    a = a.to(dtype)
    b = a.flip((0, 1))
    if layout == "strided":
        a, b = a[:, ::2], b[:, ::2]
    elif layout == "transpose":
        a, b = a.T, b.T
    elif layout == "scalar":
        a, b = a[0, 0], b[0, 0]
    elif layout == "empty":
        a, b = a[:0], b[:0]
    actual = native.interpolate_torch(a, b, amount, 1 - amount)
    expected = amount * a + (1 - amount) * b
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    if actual.is_floating_point() or actual.is_complex():
        ar = actual.resolve_conj().contiguous().reshape(-1).view(torch.uint8)
        er = expected.resolve_conj().contiguous().reshape(-1).view(torch.uint8)
        assert torch.equal(ar, er)


@pytest.mark.parametrize("left,right", itertools.product(TORCH_TYPES, TORCH_TYPES))
def test_torch_mixed_promotion(left, right):
    a = torch.tensor([[1, 2, 3], [-4, 7, 18]], dtype=left)
    b = torch.tensor([2, 8, 4], dtype=right)
    actual = native.interpolate_torch(a, b, 0.27, 0.73)
    expected = 0.27 * a + 0.73 * b
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_all_torch_half_storage(dtype):
    a = torch.from_numpy(np.arange(65536, dtype=np.uint16).view(np.int16)).view(dtype)
    b = a.roll(713)
    for amount in (0.13, 0.3333, 0.5, 1.0):
        actual = native.interpolate_torch(a, b, amount, 1 - amount)
        expected = amount * a + (1 - amount) * b
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        finite = torch.isfinite(expected)
        assert torch.equal(
            actual[finite].view(torch.int16), expected[finite].view(torch.int16)
        )


def test_concurrent_and_repeated():
    inputs = [(values(dtype), values(dtype, seed=191)) for dtype in DTYPES]
    with np.errstate(all="ignore"):
        expected = [a * 0.29 + b * 0.71 for a, b in inputs]

    def run(index: int):
        with np.errstate(all="ignore"):
            return native.interpolate_numpy(*inputs[index % len(inputs)], 0.29, 0.71)

    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(run, range(56)))
    for i, value in enumerate(actual):
        exact(value, expected[i % len(inputs)])


def test_raw_abi_guards():
    dll = native._api()
    data = np.zeros(8, np.float32)
    source = native._view(data)
    out = native._view(data.copy())
    assert dll.cn_tensor_scale_typed(None, ct.byref(out), 0.5, 0, None) == 1
    assert (
        dll.cn_tensor_scale_typed(ct.byref(source), ct.byref(source), 0.5, 0, None) == 1
    )
    out.type = 99
    assert dll.cn_tensor_cast_typed(ct.byref(source), ct.byref(out), None) == 1
    out = native._view(np.zeros(8, np.float32))
    out.count = 9
    assert (
        dll.cn_tensor_add_typed(ct.byref(source), ct.byref(source), ct.byref(out), None)
        == 1
    )
    out = native._view(np.zeros(8, np.float32))
    out.strides[0] = 0
    assert dll.cn_tensor_scale_typed(ct.byref(source), ct.byref(out), 0.5, 0, None) == 1
    np.testing.assert_array_equal(data, 0)


@pytest.mark.parametrize(
    "source,bits",
    [
        (np.float16, [0x7C01, 0xFC01, 0x7E01]),
        (np.float32, [0x7F800001, 0xFF800001, 0x7FC00001]),
        (np.float64, [0x7FF0000000000001, 0xFFF0000000000001, 0x7FF8000000000001]),
    ],
)
@pytest.mark.parametrize("target", DTYPES)
def test_signaling_nan_cast_payload_and_events(source, bits, target):
    storage = {np.float16: np.uint16, np.float32: np.uint32, np.float64: np.uint64}[
        source
    ]
    a = np.array(bits, dtype=storage).view(source)

    def invoke(function):
        with warnings.catch_warnings(record=True) as recorded, np.errstate(all="warn"):
            out = function()
        return out, [str(w.message) for w in recorded]

    actual, actual_events = invoke(lambda: native.cast_numpy(a, target))
    expected, expected_events = invoke(lambda: a.astype(target))
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    assert actual_events == expected_events


@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize("amount", [0.0, 0.13, 1.0, 2.0, 1e-100, 1e100])
def test_floating_error_stages(dtype, amount):
    real = np.empty((), dtype).real.dtype
    limits = np.finfo(real)
    a = np.array(
        [
            np.nan,
            np.inf,
            -np.inf,
            limits.max,
            limits.tiny,
            np.nextafter(real.type(0), real.type(1)),
            0.0,
            -0.0,
        ],
        dtype,
    )
    b = a[::-1]

    def invoke(function):
        with warnings.catch_warnings(record=True) as recorded, np.errstate(all="warn"):
            out = function()
        return out, [str(w.message) for w in recorded]

    actual, actual_events = invoke(
        lambda: native.interpolate_numpy(a, b, amount, 1 - amount)
    )
    expected, expected_events = invoke(lambda: a * amount + b * (1 - amount))
    exact(actual, expected)
    assert actual_events == expected_events


def test_additional_raw_metadata_guards():
    dll = native._api()
    a = np.zeros(8, np.float32)
    out = np.empty_like(a)
    av, bv = native._view(a), native._view(out)
    assert dll.cn_tensor_add_typed(ct.byref(av), None, ct.byref(bv), None) == 1
    unaligned = native._View.from_buffer(bytearray(ct.sizeof(native._View) + 1), 1)
    assert (
        dll.cn_tensor_scale_typed(ct.byref(unaligned), ct.byref(bv), 0.5, 0, None) == 1
    )
    broken = native._view(a)
    broken.shape = ct.cast(ct.addressof(av.shape.contents) + 1, ct.POINTER(ct.c_size_t))
    assert dll.cn_tensor_scale_typed(ct.byref(broken), ct.byref(bv), 0.5, 0, None) == 1
    broken = native._view(a)
    broken.dimensions = 2
    broken.shape = ct.cast(
        ct.c_void_p((1 << (8 * ct.sizeof(ct.c_size_t))) - 8), ct.POINTER(ct.c_size_t)
    )
    assert dll.cn_tensor_scale_typed(ct.byref(broken), ct.byref(bv), 0.5, 0, None) == 2
    for address in (
        a.ctypes.data,
        a.ctypes.data + 1,
        out.ctypes.data,
        ct.addressof(av),
        ct.addressof(bv.shape.contents),
    ):
        event = ct.cast(address, ct.POINTER(ct.c_int))
        assert dll.cn_tensor_scale_typed(ct.byref(av), ct.byref(bv), 0.5, 0, event) == 1
    broken = native._view(out)
    broken.data = ct.addressof(av.shape.contents)
    assert dll.cn_tensor_scale_typed(ct.byref(av), ct.byref(broken), 0.5, 0, None) == 1
    np.testing.assert_array_equal(a, 0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_torch_default_dtype_integer_products(dtype):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        for input_dtype in (torch.bool, torch.uint8, torch.int64):
            a = torch.tensor([0, 1, 123, 1000000001], dtype=torch.int64).to(input_dtype)
            b = a.flip((0,))
            expected = 0.13 * a + 0.87 * b
            actual = native.interpolate_torch(a, b, 0.13, 0.87)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finally:
        torch.set_default_dtype(previous)


def _extract(path, name, context):
    tree = ast.parse(path.read_text("utf-8-sig"))
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
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), context)
    return context[name]


@pytest.mark.parametrize("family", ("onnx", "ncnn", "pytorch"))
@pytest.mark.parametrize(
    "layout", ("plain", "fortran", "strided", "large", "above_2_24", "nonfinite")
)
def test_validation_mean_without_model_execution(family, layout):
    root = Path(__file__).resolve().parents[2]
    current = (
        root
        / f"backend/src/packages/chaiNNer_{family}/{family}/utility/interpolate_models.py"
    )
    baseline = (
        Path(__file__).with_name("reference_tensors")
        / f"installed_{family}_interpolate.py"
    )
    image = (
        np.random.default_rng(181).uniform(0.45, 0.55, (17, 19, 3)).astype(np.float32)
    )
    if layout == "fortran":
        image = np.asfortranarray(image)
    elif layout == "strided":
        image = image[::-1, ::2]
    elif layout == "large":
        image = np.random.default_rng(192).random((129, 131, 3), dtype=np.float32)
    elif layout == "above_2_24":
        # An odd count above 2**24: _mean divides by it in float64.
        image = np.random.default_rng(193).random((2367, 2365, 3), dtype=np.float32)
    elif layout == "nonfinite":
        image[0, 0] = np.nan
    means = []

    def measured_mean(value):
        result = native_mean(value)
        expected = np.mean(value)
        assert result.tobytes() == expected.tobytes()
        means.append(result)
        return result

    class Descriptor:
        input_channels = 3
        size_requirements = SimpleNamespace(minimum=3, multiple_of=1)

        def __call__(self, *args):
            return torch.empty(0)

    context: dict[str, Any] = {
        "np": np,
        "torch": torch,
        "graph": graph,
        "native_mean": measured_mean,
        "NO_TILING": object(),
        "upscale_image_node": lambda *args: image,
        "perform_interp": lambda *args: {},
        "ModelLoader": lambda *args: SimpleNamespace(
            load_from_state_dict=lambda *args: Descriptor()
        ),
        "ImageModelDescriptor": Descriptor,
        "MaskedImageModelDescriptor": type("Masked", (), {}),
        "np2tensor": lambda *args, **kwargs: torch.empty(0),
        "tensor2np": lambda *args, **kwargs: image,
        "gc": SimpleNamespace(collect=lambda: None),
    }
    name = "check_can_interp" if family == "pytorch" else "check_will_upscale"
    original = _extract(baseline, name, dict(context))
    shipped = dict(context)
    if family == "pytorch":
        # The shipped check keeps the original validation body in a module helper.
        _extract(current, "_check_interp_state", shipped)
    converted = _extract(current, name, shipped)
    args = (
        ({}, {}) if family == "pytorch" else (None, SimpleNamespace(sub_type="Generic"))
    )
    assert original(*args) == converted(*args)
    assert len(means) == 1


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.complex32,
    ],
)
@pytest.mark.parametrize(
    "layout", ["expanded", "mixed", "scalar_fortran", "empty_transpose", "singleton"]
)
def test_torch_output_layout_and_complex_half(dtype, layout):
    base = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    a, b = base.to(dtype), base.flip((0, 1)).to(dtype)
    if layout == "expanded":
        a, b = a[:1].expand(3, 4), b[:1].expand(3, 4)
    elif layout == "mixed":
        a, b = a.T, b.T.contiguous()
    elif layout == "scalar_fortran":
        a, b = a[0, 0], b.T
    elif layout == "empty_transpose":
        a, b = a[:0].T, b[:0].T
    elif layout == "singleton":
        a, b = (
            a.reshape(3, 1, 4, 1).permute(1, 3, 0, 2),
            b.reshape(3, 1, 4, 1).permute(1, 3, 0, 2),
        )
    actual = native.interpolate_torch(a, b, 0.13, 0.87)
    expected = 0.13 * a + 0.87 * b
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert actual.stride() == expected.stride()


@pytest.mark.parametrize(
    "source,bits",
    [
        (np.complex64, [0x3F800000, 0x7F800001]),
        (np.complex128, [0x3FF0000000000000, 0x7FF0000000000001]),
    ],
)
@pytest.mark.parametrize("target", [np.float16, np.float32, np.float64])
def test_discarded_imaginary_nan_does_not_raise(source, bits, target):
    integer = np.uint32 if source == np.complex64 else np.uint64
    a = np.array(bits, integer).view(source)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        with np.errstate(all="raise"):
            exact(native.cast_numpy(a, np.dtype(target)), a.astype(target))


def test_torch_shape_error_contract():
    a, b = torch.ones(2, 3), torch.ones(4, 3)
    with pytest.raises(RuntimeError) as expected:
        _ = 0.13 * a + 0.87 * b
    with pytest.raises(type(expected.value)) as actual:
        native.interpolate_torch(a, b, 0.13, 0.87)
    assert str(actual.value) == str(expected.value)
