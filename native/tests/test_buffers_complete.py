"""Pinned NumPy exceptional casts, including observable error policy."""

from __future__ import annotations

import ctypes as ct
import io
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from test_buffers import ORIGINAL

from nodes.impl import image_utils
from nodes.impl import native_buffers as native


def data(shape):
    values = (
        np.random.default_rng(82)
        .integers(0, 2**32, np.prod(shape), np.uint32)
        .view(np.float32)
        .reshape(shape)
    )
    return values


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize(
    "shape",
    [(1,), (7,), (8,), (31,), (32,), (33,), (8191,), (8192,), (8193,), (173, 191, 3)],
)
@pytest.mark.parametrize("layout", ["C", "F", "reversed", "readonly", "unaligned"])
def test_random_float_bitpatterns(bits, shape, layout):
    image = data(shape)
    if layout == "F":
        image = np.asfortranarray(image)
    elif layout == "reversed":
        image = image[::-1]
    elif layout == "readonly":
        image.flags.writeable = False
    elif layout == "unaligned":
        other = np.ndarray(
            shape, np.float32, buffer=bytearray(image.nbytes + 1), offset=1
        )
        other[...] = image
        image = other
    saved = image.tobytes()
    with np.errstate(all="ignore"):
        expected = getattr(ORIGINAL, f"to_uint{bits}")(image, normalized=True)
        actual = getattr(image_utils, f"to_uint{bits}")(image, normalized=True)
        assert native.converted_pixels(image, bits, True) is not None
    assert actual.tobytes() == expected.tobytes()
    assert image.tobytes() == saved


@pytest.mark.parametrize("bits", [8, 16])
def test_integer_cast_boundaries(bits):
    scale = (1 << bits) - 1
    targets = (
        np.array(
            [
                -(2**31),
                2**31,
                -65536,
                65536,
                2**24,
                -(2**24),
                2**32,
                -(2**32),
                2**63,
                2**127,
            ],
            np.float32,
        )
        / scale
    )
    image = np.concatenate(
        (
            targets,
            np.nextafter(targets, np.float32(np.inf)),
            np.nextafter(targets, np.float32(-np.inf)),
        )
    )
    with np.errstate(all="ignore"):
        assert (
            getattr(image_utils, f"to_uint{bits}")(image, normalized=True).tobytes()
            == getattr(ORIGINAL, f"to_uint{bits}")(image, normalized=True).tobytes()
        )


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("kind", ["nan", "inf", "huge", "signed_nan", "combined"])
def test_diagnostics_and_raises(bits, normalized, kind):
    values = {"nan": np.nan, "inf": np.inf, "huge": np.finfo(np.float32).max}
    if kind == "signed_nan":
        image = np.array([0x7F800123], np.uint32).view(np.float32)
    elif kind == "combined":
        image = np.array([0x7F800123, 0x7F7FFFFF], np.uint32).view(np.float32)
    else:
        image = np.array([values[kind]], np.float32)
    for policy in ("warn", "raise"):
        observations = []
        for module in (ORIGINAL, image_utils):
            with (
                warnings.catch_warnings(record=True) as captured,
                np.errstate(all=policy),
            ):
                warnings.simplefilter("always")
                try:
                    result = getattr(module, f"to_uint{bits}")(
                        image, normalized=normalized
                    )
                except FloatingPointError as error:
                    outcome = type(error), str(error)
                else:
                    outcome = result.tobytes()
                observations.append((outcome, [str(w.message) for w in captured]))
        assert observations[0] == observations[1]


@pytest.mark.parametrize("policy", ["call", "log", "print"])
def test_floating_error_handlers(policy, capfd):
    old_handler = np.geterrcall()
    image = np.array([np.inf, np.finfo(np.float32).max], np.float32)
    observed = []
    try:
        for module in (ORIGINAL, image_utils):
            calls = []
            log = io.StringIO()
            np.seterrcall(
                (lambda error, flag, calls=calls: calls.append((error, flag)))
                if policy == "call"
                else log
            )
            with np.errstate(all=policy):
                result = module.to_uint16(image, normalized=True)
            observed.append(
                (result.tobytes(), calls, log.getvalue(), capfd.readouterr().err)
            )
        assert observed[0] == observed[1]
    finally:
        np.seterrcall(old_handler)


def test_concurrent_exceptional_conversion():
    image = data((257, 263, 3))
    image.flags.writeable = False
    with np.errstate(all="ignore"):
        expected = ORIGINAL.to_uint16(image, normalized=True)

    def run(_):
        with np.errstate(all="ignore"):
            return image_utils.to_uint16(image, normalized=True)

    with ThreadPoolExecutor(max_workers=6) as executor:
        for actual in executor.map(run, range(18)):
            assert actual.tobytes() == expected.tobytes()


def test_checked_raw_cast_guards():
    dll = native._api()
    source = np.zeros(5, np.float32)
    result = np.full(5, 29, np.uint16)
    events = ct.c_int(71)
    assert (
        dll.cn_pixels_convert_checked(
            None, result.ctypes.data, 5, 0, 2, 0, ct.byref(events)
        )
        == 1
    )
    assert (
        dll.cn_pixels_convert_checked(
            source.ctypes.data,
            result.ctypes.data,
            ct.c_size_t(-1).value,
            0,
            2,
            0,
            ct.byref(events),
        )
        == 2
    )
    assert np.all(result == 29) and events.value == 71
