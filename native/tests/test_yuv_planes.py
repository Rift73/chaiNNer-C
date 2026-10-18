"""cn_plane_codes_to_f32 and cn_plane_f32_to_codes, colour_info.py's code_to_float
and float_to_code in C: at every level this CPU has, every code of 8 to 16 bits and
float planes with values outside 0-1, ties, NaN, infinities and subnormals give the
NumPy forms' results bit for bit, in every layout the bridge takes."""

from __future__ import annotations

import ctypes as ct

import numpy as np
import pytest
from test_isa_dispatch import isa_get, isa_set

from nodes.impl import colour_info
from nodes.impl.colour_info import code_to_float, float_to_code
from nodes.impl.native_planes import _api

ROWS, WIDTH, STRIDE = 64, 1027, 1040  # 1027 % 8 == 3: every row has a scalar tail
DEPTHS = range(8, 17)
FORMS = [(False, False), (False, True), (True, False), (True, True)]


@pytest.fixture
def levels():
    """Every level this CPU has, scalar first; the original level afterwards."""
    effective, _, cpu = isa_get()
    try:
        yield range(cpu + 1)
    finally:
        assert isa_set(effective) == effective


def _numpy_values(codes: np.ndarray, bits: int, full: bool, chroma: bool):
    return colour_info._numpy_code_to_float(codes, bits, full, chroma)


def _numpy_codes(values: np.ndarray, bits: int, full: bool, chroma: bool):
    # NumPy warns on the NaN cast and on products beyond float32; the kernels do not.
    with np.errstate(invalid="ignore", over="ignore"):
        return colour_info._numpy_float_to_code(values, bits, full, chroma)


def _bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values).view(np.uint32)


def _cropped(values: np.ndarray) -> np.ndarray:
    """values (a whole number of times, then the first ones again) in ROWS rows of
    WIDTH elements cropped from rows of STRIDE: rows read in place, with tails."""
    buffer = np.zeros((ROWS, STRIDE), values.dtype)
    buffer[:, :WIDTH] = np.resize(values, ROWS * WIDTH).reshape(ROWS, WIDTH)
    return buffer[:, :WIDTH]


def _specials() -> np.ndarray:
    words = [0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFA00000, 0x7F800000, 0xFF800000]
    words += [0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x80000000, 0]
    words += [0x7F7FFFFF, 0xFF7FFFFF, 0x3F000000, 0xBF000000, 0x3F800000]
    return np.array(words, np.uint32).view(np.float32)


def _ties(bits: int, full: bool, chroma: bool) -> np.ndarray:
    """For every code k, the float32 values within 2 ulp of where the scaled value is
    k + 0.5, so many scaled values are exact ties."""
    offset, scale = colour_info._scale(bits, full, chroma)
    halves = np.arange(-2, (1 << bits) + 1, dtype=np.float64) + 0.5
    centres = ((halves - offset) / scale + (0.5 if chroma else 0.0)).astype(np.float32)
    words = centres.view(np.int32)[:, None] + np.arange(-2, 3, dtype=np.int32)
    return words.view(np.float32).ravel()


@pytest.mark.parametrize("bits", DEPTHS)
@pytest.mark.parametrize(("full", "chroma"), FORMS)
def test_every_code_converts_as_numpy(levels, bits: int, full: bool, chroma: bool):
    rng = np.random.default_rng(bits)
    wide = _cropped(rng.permutation(np.arange(65536, dtype=np.uint16)))
    narrow = _cropped(rng.permutation(np.arange(256, dtype=np.uint8)))
    for codes in (wide, narrow):
        expected = _bits(_numpy_values(codes, bits, full, chroma))
        for level in levels:
            assert isa_set(level) == level
            np.testing.assert_array_equal(
                _bits(code_to_float(codes, bits, full, chroma)),
                expected,
                err_msg=f"{level} {codes.dtype}",
            )


@pytest.mark.parametrize("bits", DEPTHS)
@pytest.mark.parametrize(("full", "chroma"), FORMS)
def test_float_planes_convert_as_numpy(levels, bits: int, full: bool, chroma: bool):
    rng = np.random.default_rng(100 + bits)
    codes = np.arange(1 << bits, dtype=np.uint8 if bits == 8 else np.uint16)
    random_words = rng.integers(0, 2**32, 1 << 18, dtype=np.uint64).astype(np.uint32)
    ties = _ties(bits, full, chroma)
    values = np.concatenate(
        [
            _numpy_values(codes, bits, full, chroma),
            ties,
            random_words.view(np.float32),
            rng.uniform(-0.5, 1.5, 1 << 16).astype(np.float32),
            _specials(),
        ]
    )
    offset, scale = colour_info._scale(bits, full, chroma)
    scaled = (ties - np.float32(0.5 if chroma else 0)) * np.float32(scale)
    scaled += np.float32(offset)
    assert np.count_nonzero(scaled % 1 == 0.5) >= 1 << bits  # ties are reached
    for plane in (_cropped(values), values):
        expected = _numpy_codes(plane, bits, full, chroma)
        for level in levels:
            assert isa_set(level) == level
            result = float_to_code(plane, bits, full, chroma)
            assert result.dtype == expected.dtype
            np.testing.assert_array_equal(result, expected, err_msg=str(level))


def test_nan_gives_code_zero():
    nans = _specials()[:4]  # quiet, negative quiet, signalling, negative signalling
    for bits in (8, 10, 16):
        for full, chroma in FORMS:
            expected = _numpy_codes(nans, bits, full, chroma)
            assert expected.tolist() == [0, 0, 0, 0]
            assert float_to_code(nans, bits, full, chroma).tolist() == [0, 0, 0, 0]


def test_layouts_and_dtypes_follow_numpy():
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 1024, (37, 29), dtype=np.uint16)
    values = rng.uniform(-0.2, 1.2, (37, 29)).astype(np.float32)
    code_forms = [codes, codes.T, codes[::-1], codes[:, ::2], codes[3], codes[:0]]
    code_forms += [codes.reshape(37, 29, 1), np.asarray(codes[3, 4])]
    for form in code_forms:
        result = code_to_float(form, 10, False, True)
        expected = _numpy_values(form, 10, False, True)
        assert result.shape == expected.shape
        assert result.strides == expected.strides
        np.testing.assert_array_equal(_bits(result), _bits(expected))
    value_forms = [values, values.T, values[::-1], values[:, ::3], values[5]]
    value_forms += [values[:0], values.astype(np.float64)]
    value_forms += [values.astype(">f4"), values.astype(np.float16), codes]
    for form in value_forms:
        result = float_to_code(form, 10, False, False)
        expected = _numpy_codes(form, 10, False, False)
        assert result.shape == expected.shape
        np.testing.assert_array_equal(result, expected)


def test_every_in_range_input_takes_the_kernel(monkeypatch):
    def numpy_form(*_args: object) -> np.ndarray:
        raise AssertionError("an in-range input reached the NumPy form")

    monkeypatch.setattr(colour_info, "_numpy_code_to_float", numpy_form)
    monkeypatch.setattr(colour_info, "_numpy_float_to_code", numpy_form)
    for bits in DEPTHS:
        for dtype in (np.uint8, np.uint16):
            code_to_float(np.zeros((2, 9), dtype), bits, False, False)
        for dtype in (np.float32, np.float64, np.float16):
            float_to_code(np.zeros((2, 9), dtype), bits, True, True)


def test_other_inputs_take_the_numpy_form(monkeypatch):
    calls = []

    def recorded(name: str):
        original = getattr(colour_info, name)

        def form(*args: object) -> np.ndarray:
            calls.append(name)
            return original(*args)

        return form

    for name in ("_numpy_code_to_float", "_numpy_float_to_code"):
        monkeypatch.setattr(colour_info, name, recorded(name))
    code_to_float(np.zeros(4, np.int32), 10, False, False)
    code_to_float(np.zeros(4, np.float32), 10, False, False)
    code_to_float(np.zeros(4, ">u2"), 10, False, False)
    code_to_float(np.zeros(4, np.uint8), 7, False, False)
    float_to_code(np.zeros(4, np.float32), 7, False, False)
    assert calls == ["_numpy_code_to_float"] * 4 + ["_numpy_float_to_code"]


def test_overlap_misalignment_and_bounds_are_refused():
    buffer = np.zeros(4096, np.float32)
    base = buffer.ctypes.data
    to_f32, to_codes = _api().cn_plane_codes_to_f32, _api().cn_plane_f32_to_codes
    # 8 x 8 uint16 codes; the output would start inside them
    assert to_f32(base, 1, 8, 8, 8, 16.0, 219.0, 0, base + 64) != 0
    assert to_f32(base + 1, 1, 8, 8, 8, 16.0, 219.0, 0, base + 8192) != 0
    assert to_f32(base, 1, 8, 8, 8, 16.0, 219.0, 0, base + 8194) != 0
    assert to_f32(base, 1, 8, 8, 9, 16.0, 219.0, 0, base + 8192) != 0
    assert to_f32(base, 1, 8, 8, 8, 16.0, 219.0, 0, ct.c_void_p(base + 8192)) == 0
    assert to_codes(base, 8, 8, 8, 16.0, 219.0, 0, 255, 0, base + 128) != 0
    assert to_codes(base + 2, 8, 8, 8, 16.0, 219.0, 0, 255, 0, base + 8192) != 0
    assert to_codes(base, 8, 8, 8, 16.0, 219.0, 0, 256, 0, base + 8192) != 0
    assert to_codes(base, 8, 8, 8, 16.0, 219.0, 0, 65536, 1, base + 8192) != 0
    assert to_codes(base, 8, 8, 8, 16.0, 219.0, 0, 1023, 1, base + 8193) != 0
    assert to_codes(base, 8, 8, 8, 16.0, 219.0, 0, 1023, 1, base + 8192) == 0
