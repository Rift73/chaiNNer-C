"""Pinned OpenCV color arithmetic, including vector tails and foreign buffers."""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from nodes.impl import native_color_complete as native


def exact(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual.view(np.uint32)[finite], expected.view(np.uint32)[finite]
    )


def input_image(code: int, width: int, height: int = 7, seed: int = 482) -> np.ndarray:
    op = native._CODES[code][0]
    channels = (
        1
        if code in (cv2.COLOR_GRAY2BGR, cv2.COLOR_GRAY2BGRA)
        else 4
        if code
        in (
            cv2.COLOR_BGRA2BGR,
            cv2.COLOR_BGRA2RGB,
            cv2.COLOR_BGRA2RGBA,
            cv2.COLOR_BGRA2GRAY,
            cv2.COLOR_RGBA2GRAY,
        )
        else 3
    )
    image = np.random.default_rng(seed).random(
        (height, width, channels), dtype=np.float32
    )
    if op in (5, 7):
        image[:, :, 0] *= 360
    elif op == 11:
        image[:, :, 0] *= 100
        image[:, :, 1:] = image[:, :, 1:] * 256 - 128
    return image


@pytest.mark.parametrize("code", list(native._CODES))
@pytest.mark.parametrize("width", [1, 3, 4, 7, 8, 9, 16, 41])
@pytest.mark.parametrize("layout", ["plain", "strided", "readonly", "unaligned"])
def test_colors(code: int, width: int, layout: str) -> None:
    source = input_image(code, width)
    if layout == "strided":
        source = source[::-1, ::-1]
    elif layout == "readonly":
        source.flags.writeable = False
    elif layout == "unaligned":
        source = np.ndarray(
            source.shape,
            dtype=np.float32,
            buffer=bytearray(source.nbytes + 1),
            offset=1,
        )
        source[:] = input_image(code, width)
    before = source.copy()
    exact(native.cvt_color(source, code), cv2.cvtColor(source, code))
    np.testing.assert_array_equal(source.view(np.uint32), before.view(np.uint32))


@pytest.mark.parametrize("code", list(native._CODES))
@pytest.mark.parametrize(
    "optimized,ipp", [(False, False), (False, True), (True, False)]
)
def test_alternate_dispatch(code: int, optimized: bool, ipp: bool) -> None:
    original_opt, original_ipp = cv2.useOptimized(), cv2.ipp.useIPP()
    try:
        cv2.setUseOptimized(optimized)
        cv2.ipp.setUseIPP(ipp)
        source = input_image(code, 41)
        exact(native.cvt_color(source, code), cv2.cvtColor(source, code))
    finally:
        cv2.setUseOptimized(original_opt)
        cv2.ipp.setUseIPP(original_ipp)


@pytest.mark.parametrize("code", list(native._CODES))
@pytest.mark.parametrize(
    "kind", ["zero", "signed_zero", "out_of_range", "nonfinite", "gray"]
)
def test_adversarial(code: int, kind: str) -> None:
    source = input_image(code, 41)
    rng = np.random.default_rng(193)
    if kind == "zero":
        source[:] = 0
    elif kind == "signed_zero":
        source[:] = np.array([-0.0, 0.0], np.float32)[rng.integers(0, 2, source.shape)]
    elif kind == "out_of_range":
        source[:] = rng.uniform(-3, 3, source.shape)
        if native._CODES[code][0] in (5, 7):
            source[:, :, 0] *= 1000
    elif kind == "nonfinite":
        source[:] = np.array([-np.inf, -0.0, 0.0, np.inf, np.nan, 0.2], np.float32)[
            rng.integers(0, 6, source.shape)
        ]
    elif kind == "gray":
        source[:] = rng.random((*source.shape[:2], 1), dtype=np.float32)
    exact(native.cvt_color(source, code), cv2.cvtColor(source, code))


def test_concurrent() -> None:
    inputs = [(input_image(code, 257, height=129), code) for code in native._CODES]
    expected = [cv2.cvtColor(source, code) for source, code in inputs]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda item: native.cvt_color(*item), inputs))
    for actual, reference in zip(results, expected, strict=True):
        exact(actual, reference)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_other_dtype_retains_original(dtype: type) -> None:
    source = np.arange(60, dtype=dtype).reshape(4, 5, 3)
    np.testing.assert_array_equal(
        native.cvt_color(source, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(source, cv2.COLOR_BGR2GRAY),
    )


@pytest.mark.parametrize("shape", [(0, 3, 3), (2, 0, 3), (3,), (2, 3, 4, 3), (2, 3, 2)])
def test_invalid_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises((ValueError, RuntimeError)):
        native.cvt_color(np.zeros(shape, np.float32), cv2.COLOR_BGR2GRAY)


def test_raw_boundary() -> None:
    function = native._api().cn_color_convert_f32
    source, out = np.zeros(96, np.float32), np.zeros(96, np.float32)
    pointer = ct.POINTER(ct.c_float)
    args = [native.ptr(source), native.ptr(out), 2, 3, 3, 3, 2, 0, 8, 1]
    assert function(*args) == 0
    for index, value in [
        (0, pointer()),
        (1, pointer()),
        (2, 0),
        (4, 2),
        (5, 2),
        (6, 15),
        (7, 1),
        (8, 0),
        (9, 2),
    ]:
        invalid = args.copy()
        invalid[index] = value
        assert function(*invalid) == 1
    for offset in [0, 1, 4]:
        invalid = args.copy()
        invalid[1] = ct.cast(source.ctypes.data + offset, pointer)
        assert function(*invalid) == 1
    invalid = args.copy()
    invalid[2] = ct.c_size_t(-1).value
    assert function(*invalid) == 2


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 3), (3, 7, 3), (37, 45, 3), (129, 257, 3)])
@pytest.mark.parametrize("kind", ["random", "nonfinite", "signed_zero", "large"])
def test_lab_lch(inverse: bool, shape: tuple[int, ...], kind: str) -> None:
    from test_transparency import REFERENCE_DATA

    source = np.random.default_rng(391).random(shape, dtype=np.float32)
    if kind == "nonfinite":
        source[:] = np.array([-np.inf, np.inf, np.nan, 0, 0.5, 1], np.float32)[
            np.random.default_rng(19).integers(0, 6, shape)
        ]
    elif kind == "signed_zero":
        source[:] = -0.0
    elif kind == "large":
        source *= 1e20
    source = source[::-1, ::-1]
    before = source.copy()
    source.flags.writeable = False
    with np.errstate(all="ignore"):
        reference = getattr(
            REFERENCE_DATA, "__lch_to_lab" if inverse else "__lab_to_lch"
        )(source)
    exact(native.lab_lch(source, inverse=inverse), reference)
    np.testing.assert_array_equal(source.view(np.uint32), before.view(np.uint32))


@pytest.mark.parametrize("source_id", range(14))
@pytest.mark.parametrize("target_id", range(14))
@pytest.mark.parametrize("kind", ["nonfinite", "signed_zero", "out_of_range"])
def test_graph_adversarial(source_id: int, target_id: int, kind: str) -> None:
    from test_transparency import REFERENCE_CONVERT

    from nodes.impl.color import convert

    source_space = convert.color_space_from_id(source_id)
    shape = (3, 41, source_space.channels)
    rng = np.random.default_rng(17)
    if kind == "nonfinite":
        source = np.array([-np.inf, np.inf, np.nan, -0.0, 0.0, 0.5, 1], np.float32)[
            rng.integers(0, 7, shape)
        ]
    elif kind == "signed_zero":
        source = np.array([-0.0, 0.0], np.float32)[rng.integers(0, 2, shape)]
    else:
        source = rng.uniform(-2, 3, shape).astype(np.float32)
    if source_space.channels == 1:
        source = source[:, :, 0]
    with np.errstate(all="ignore"):
        expected = REFERENCE_CONVERT.convert(
            source,
            REFERENCE_CONVERT.color_space_from_id(source_id),
            REFERENCE_CONVERT.color_space_from_id(target_id),
        )
        actual = convert.convert(
            source, source_space, convert.color_space_from_id(target_id)
        )
    exact(actual, expected)


@pytest.mark.parametrize(
    "code",
    [
        cv2.COLOR_BGR2GRAY,
        cv2.COLOR_BGR2YUV,
        cv2.COLOR_BGR2YCrCb,
        cv2.COLOR_BGR2HSV,
        cv2.COLOR_BGR2HLS,
        cv2.COLOR_BGR2LAB,
    ],
)
@pytest.mark.parametrize("width", [1, 7, 8, 15, 16, 17, 33])
def test_four_channel_source(code: int, width: int) -> None:
    source = np.random.default_rng(241).random((5, width, 4), dtype=np.float32)
    exact(native.cvt_color(source, code), cv2.cvtColor(source, code))


def test_entire_lab_lattice() -> None:
    r, g, b = np.meshgrid(
        *(np.arange(33, dtype=np.float32) / 32 for _ in range(3)), indexing="ij"
    )
    source = np.stack([b, g, r], axis=-1).reshape(1089, 33, 3)
    exact(
        native.cvt_color(source, cv2.COLOR_BGR2LAB),
        cv2.cvtColor(source, cv2.COLOR_BGR2LAB),
    )


@pytest.mark.parametrize("width", [1, 7, 8, 9, 16, 31, 257])
def test_lab_inverse_piecewise_boundaries(width: int) -> None:
    values = np.array(
        [
            0,
            8,
            100,
            -128,
            127,
            np.nextafter(np.float32(8), np.float32(-np.inf)),
            np.nextafter(np.float32(8), np.float32(np.inf)),
        ],
        np.float32,
    )
    source = values[
        np.random.default_rng(111).integers(0, values.size, (131, width, 3))
    ]
    exact(
        native.cvt_color(source, cv2.COLOR_LAB2BGR),
        cv2.cvtColor(source, cv2.COLOR_LAB2BGR),
    )


@pytest.mark.parametrize("size", [1, 3, 4, 7, 8, 9, 16, 33, 127])
def test_colorwheel_node(size: int) -> None:
    from test_analysis_ops import load_body

    root = Path(__file__).resolve().parents[2]
    actual = load_body(
        root
        / "backend/src/packages/chaiNNer_standard/image/create_images/create_colorwheel.py"
    )
    reference = load_body(
        Path(__file__).with_name("reference_color_complete") / "create_colorwheel.py"
    )
    exact(actual.create_colorwheel_node(size), reference.create_colorwheel_node(size))


def test_graph_does_not_call_original_kernels(monkeypatch) -> None:
    from test_transparency import REFERENCE_CONVERT

    from nodes.impl.color import convert

    cases = []
    rng = np.random.default_rng(48)
    for source_id in range(14):
        source_space = convert.color_space_from_id(source_id)
        source = rng.random((3, 9, source_space.channels), dtype=np.float32)
        if source_space.channels == 1:
            source = source[:, :, 0]
        for target_id in range(14):
            expected = REFERENCE_CONVERT.convert(
                source,
                REFERENCE_CONVERT.color_space_from_id(source_id),
                REFERENCE_CONVERT.color_space_from_id(target_id),
            )
            cases.append(
                (source, source_space, convert.color_space_from_id(target_id), expected)
            )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Float32 graph called an original arithmetic kernel")

    for name in ("cvtColor", "merge"):
        monkeypatch.setattr(cv2, name, forbidden)
    for name in ("sin", "cos", "arctan2", "hypot", "clip", "stack", "dstack"):
        monkeypatch.setattr(np, name, forbidden)
    for source, source_space, target_space, expected in cases:
        exact(convert.convert(source, source_space, target_space), expected)
