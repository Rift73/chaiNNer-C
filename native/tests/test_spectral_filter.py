"""Bit-exact C crossCorr extraction; the original public IPP DFT is retained.

OpenCV 4.8 is the independent installed binary oracle. The constant half-gray
fixtures deliberately expose saved-pixel differences from changing its DFT.
No performance or inference tests are performed here.
"""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from nodes.impl import native_convolution
from nodes.impl import native_spectral_filter as spectral
from nodes.impl.native import lib
from nodes.impl.native_versions import CV_CN_MAX

ORIGINAL_FILTER = cv2.filter2D
ORIGINAL_DFT = cv2.dft
ORIGINAL_MULTIPLY = cv2.mulSpectrums
ORIGINAL_BORDER = cv2.copyMakeBorder


def exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    finite = ~np.isnan(expected)
    integer = np.uint32 if actual.dtype == np.float32 else np.uint64
    np.testing.assert_array_equal(
        actual[finite].view(integer), expected[finite].view(integer)
    )


@pytest.fixture(params=[False, True], ids=["open-source-dft", "installed-ipp-dft"])
def ipp(request):
    previous = cv2.ipp.useIPP()
    cv2.ipp.setUseIPP(request.param)
    try:
        yield request.param
    finally:
        cv2.ipp.setUseIPP(previous)


def baseline(image, kernel, padding=0, border=cv2.BORDER_REFLECT_101):
    if padding:
        image = ORIGINAL_BORDER(image, padding, padding, padding, padding, 0)
    return ORIGINAL_FILTER(image, -1, kernel, borderType=border)


@pytest.mark.parametrize("shape", [(1, 1), (1, 19), (17, 1), (13, 19)])
@pytest.mark.parametrize("channels", [None, 1, 2, 3, 4, 7])
@pytest.mark.parametrize("kernel_shape", [(13, 10), (11, 13), (1, 131), (131, 1)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_shapes_and_kernels(ipp, shape, channels, kernel_shape, dtype):
    rng = np.random.default_rng(291)
    full_shape = shape if channels is None else (*shape, channels)
    image = rng.random(full_shape, dtype=np.float32)
    kernel = rng.uniform(-1, 1, kernel_shape).astype(dtype)
    exact(spectral.filter2d(image, kernel), baseline(image, kernel))


@pytest.mark.parametrize("shape", [(1, 1), (1, 17, 3), (19, 1, 4), (7, 13, 2)])
@pytest.mark.parametrize("border", [0, 1, 2, 3, 4, 16, 17, 18, 19, 20])
@pytest.mark.parametrize("padding", [0, 1, 19])
def test_borders_and_outer_zero_frame(ipp, shape, border, padding):
    rng = np.random.default_rng(553)
    image = rng.random(shape, dtype=np.float32)
    kernel = rng.random((13, 11), dtype=np.float32)
    exact(
        spectral.filter2d(image, kernel, padding, border),
        baseline(image, kernel, padding, border),
    )


@pytest.mark.parametrize("shape", [(263, 271), (519, 521, 3), (259, 513, 4)])
@pytest.mark.parametrize("kernel_shape", [(12, 12), (13, 17)])
@pytest.mark.parametrize("padding", [0, 7])
def test_partial_tiles_and_interleaving(ipp, shape, kernel_shape, padding):
    rng = np.random.default_rng(748)
    image = rng.random(shape, dtype=np.float32)
    kernel = rng.random(kernel_shape, dtype=np.float32)
    kernel /= kernel.sum()
    exact(spectral.filter2d(image, kernel, padding), baseline(image, kernel, padding))


def fractional_kernel(rx, ry):
    # Frozen original Box Blur kernel construction, including float32 edge order.
    kernel = np.ones((int(np.ceil(ry)) * 2 + 1, int(np.ceil(rx)) * 2 + 1), np.float32)
    kernel /= (rx * 2 + 1) * (ry * 2 + 1)
    kernel[0, :] *= ry % 1
    kernel[-1, :] *= ry % 1
    kernel[:, 0] *= rx % 1
    kernel[:, -1] *= rx % 1
    return kernel


@pytest.mark.parametrize("radii", [(73.2, 169.7), (77.1, 102.3), (70.5, 70.5)])
@pytest.mark.parametrize("shape", [(31, 37), (17, 21, 3), (1, 1, 4)])
def test_fractional_box_rounding_ties(ipp, radii, shape):
    image = np.full(shape, 0.5, np.float32)
    kernel = fractional_kernel(*radii)
    got, expected = spectral.filter2d(image, kernel), baseline(image, kernel)
    exact(got, expected)
    np.testing.assert_array_equal(np.round(got * 255), np.round(expected * 255))


def test_retained_ipp_changes_actual_saved_pixels():
    previous = cv2.ipp.useIPP()
    image = np.full((31, 37), 0.5, np.float32)
    kernel = fractional_kernel(73.2, 169.7)
    try:
        cv2.ipp.setUseIPP(True)
        ipp = baseline(image, kernel)
        cv2.ipp.setUseIPP(False)
        open_source = baseline(image, kernel)
    finally:
        cv2.ipp.setUseIPP(previous)
    # This is the installed IPP2021.8 + AVX2 contract, not a claim about all CPUs.
    assert np.count_nonzero(np.round(ipp * 255) != np.round(open_source * 255)) > 0


@pytest.mark.parametrize("target", ["image", "kernel"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -0.0, 1e30, -1e30])
@pytest.mark.parametrize("channels", [None, 1, 3, 4])
def test_nonfinite_and_signed_zero(ipp, target, value, channels):
    rng = np.random.default_rng(381)
    shape = (9, 11) if channels is None else (9, 11, channels)
    image = rng.random(shape, dtype=np.float32)
    kernel = rng.random((11, 13), dtype=np.float32)
    array = image if target == "image" else kernel
    if value == 0:
        array.fill(value)
    else:
        array.ravel()[::7] = value
    exact(spectral.filter2d(image, kernel), baseline(image, kernel))


@pytest.mark.parametrize("value", [1e300, -1e300, 1e-300, -1e-300, np.nan, np.inf])
def test_double_coefficients_convert_to_float_before_dft(ipp, value):
    image = np.full((7, 9, 3), 0.5, np.float32)
    kernel = np.full((13, 11), value, np.float64)
    exact(spectral.filter2d(image, kernel), baseline(image, kernel))


@pytest.mark.parametrize("target", ["image", "kernel"])
@pytest.mark.parametrize(
    "layout", ["reverse", "transpose", "readonly", "broadcast", "unaligned"]
)
def test_foreign_buffers(ipp, target, layout):
    rng = np.random.default_rng(849)
    image = rng.random((13, 17, 4), dtype=np.float32)
    kernel = rng.random((13, 11)).astype(np.float64)
    array = image if target == "image" else kernel
    if layout == "reverse":
        array = array[::-1, ::-1]
    elif layout == "transpose":
        array = array.swapaxes(0, 1)
    elif layout == "readonly":
        array.setflags(write=False)
    elif layout == "broadcast":
        array = np.broadcast_to(array[:1, :1], array.shape)
    else:
        unaligned = np.ndarray(array.shape, array.dtype, bytearray(array.nbytes + 1), 1)
        unaligned[:] = array
        array = unaligned
    if target == "image":
        image = array
    else:
        kernel = array
    before = array.copy()
    got = spectral.filter2d(image, kernel)
    exact(got, baseline(image, kernel))
    exact(array, before)
    assert not np.shares_memory(got, array)
    assert got.flags.c_contiguous and got.flags.writeable


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 2), (1, 9), (2, 1), (9, 1), (3, 4), (4, 3), (256, 257)]
)
@pytest.mark.parametrize("kind", ["random", "signed_zero", "nonfinite"])
def test_packed_spectrum_all_parities(shape, kind):
    rng = np.random.default_rng(385)
    a, b = rng.uniform(-10, 10, (2, *shape))
    if kind == "signed_zero":
        a.ravel()[::2] = -0.0
        b.ravel()[1::2] = -0.0
    elif kind == "nonfinite":
        a.ravel()[::7] = np.nan
        b.ravel()[1::5] = np.inf
    expected = ORIGINAL_MULTIPLY(a, b, 0, conjB=True)
    status = lib().cn_spectral_multiply(
        a.ctypes.data_as(ct.POINTER(ct.c_double)),
        b.ctypes.data_as(ct.POINTER(ct.c_double)),
        *shape,
    )
    assert status == 0
    exact(a, expected)


def test_only_public_dft_is_retained(monkeypatch):
    image = np.full((31, 37, 4), 0.5, np.float32)
    kernel = fractional_kernel(73.2, 169.7)
    expected = baseline(image, kernel, 3)
    calls = []

    def dft(*args, **kwargs):
        calls.append(kwargs)
        return ORIGINAL_DFT(*args, **kwargs)

    def prohibited(*_args, **_kwargs):
        pytest.fail(
            "A converted buffer, multiplication, or filtering operation was called"
        )

    monkeypatch.setattr(cv2, "dft", dft)
    for name in ("filter2D", "mulSpectrums", "copyMakeBorder", "split", "merge"):
        monkeypatch.setattr(cv2, name, prohibited)
    exact(spectral.filter2d(image, kernel, 3), expected)
    assert len(calls) == 1 + 2 * 4
    assert all("nonzeroRows" in kwargs for kwargs in calls)


def test_concurrent_images_are_independent():
    rng = np.random.default_rng(329)
    inputs = [rng.random((259, 263, 3), dtype=np.float32) for _ in range(8)]
    kernel = rng.random((13, 11), dtype=np.float32)
    expected = [baseline(image, kernel) for image in inputs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda image: spectral.filter2d(image, kernel), inputs))
    for got, wanted in zip(actual, expected, strict=True):
        exact(got, wanted)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (1,), (2, 3, 0), (1, 2, 3, 4)])
def test_invalid_image_shape(shape):
    with pytest.raises(ValueError):
        spectral.filter2d(np.empty(shape, np.float32), np.ones((13, 11), np.float32))


def test_channel_ceiling():
    # OpenCV's channel ceiling: both sides raise cv2.error past it.
    image = np.ones((2, 3, CV_CN_MAX + 1), np.float32)
    kernel = np.ones((13, 11), np.float32)
    with pytest.raises(cv2.error):
        spectral.filter2d(image, kernel)
    with pytest.raises(cv2.error):
        cv2.filter2D(image, -1, kernel)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float64])
def test_invalid_image_dtype(dtype):
    with pytest.raises(TypeError):
        spectral.filter2d(np.ones((3, 5), dtype), np.ones((13, 11), np.float32))


@pytest.mark.parametrize(
    "kernel", [None, [1], np.zeros((1,)), np.zeros((0, 3)), np.zeros((3, 3, 1))]
)
def test_invalid_kernel_shape(kernel):
    with pytest.raises(ValueError):
        spectral.filter2d(np.ones((3, 5), np.float32), kernel)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"padding": -1},
        {"padding": 1.5},
        {"padding": 2**31},
        {"border": -1},
        {"border": 5},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        spectral.filter2d(
            np.ones((3, 5), np.float32), np.ones((13, 11), np.float32), **kwargs
        )


def test_native_rejects_invalid_boundaries_without_writes():
    native = lib()
    source = np.ones((2, 3), np.float32)
    scratch = np.full((7, 9), 837.0, np.float64)
    out = np.full((2, 3), 942.0, np.float32)
    fp = source.ctypes.data_as(ct.POINTER(ct.c_float))
    dp = scratch.ctypes.data_as(ct.POINTER(ct.c_double))
    op = out.ctypes.data_as(ct.POINTER(ct.c_float))
    assert native.cn_spectral_kernel(source.ctypes.data, 2, dp, 2, 3, 7, 9) == 1
    assert native.cn_spectral_kernel(source.ctypes.data, 0, dp, 2, 3, 2**31, 9) == 2
    assert (
        native.cn_spectral_tile(fp, dp, 2, 3, 1, 2**31, 0, 0, 2, 3, 3, 3, 7, 9, 0, 4)
        == 2
    )
    assert (
        native.cn_spectral_tile(fp, dp, 2, 3, 1, 0, 0, 0, 2, 3, 3, 3, 7, 9, 1, 4) == 1
    )
    assert (
        native.cn_spectral_tile(fp, dp, 2, 3, 1, 0, 2, 0, 2, 3, 3, 3, 7, 9, 0, 4) == 1
    )
    assert native.cn_spectral_store(dp, op, 2, 3, 1, 0, 0, 3, 3, 7, 9, 0) == 1
    assert native.cn_spectral_store(dp, op, 2, 3, 1, 0, 0, 2, 3, 7, 9, 1) == 1
    assert native.cn_spectral_multiply(dp, dp, 7, 9) == 1
    np.testing.assert_array_equal(scratch, 837.0)
    np.testing.assert_array_equal(out, 942.0)


def test_unrepresentable_optimal_size_is_rejected(monkeypatch):
    monkeypatch.setattr(cv2, "getOptimalDFTSize", lambda _size: -1)
    with pytest.raises(OverflowError):
        spectral.filter2d(np.ones((3, 5), np.float32), np.ones((13, 11), np.float32))


@pytest.mark.parametrize("optimized", [True, False])
@pytest.mark.parametrize("not_exact", [True, False])
@pytest.mark.parametrize("area", [49, 50, 129, 130])
@pytest.mark.parametrize("padding", [0, 3])
def test_convolve_actual_dispatch_threshold(
    ipp, optimized, not_exact, area, padding, monkeypatch
):
    original_optimized = cv2.useOptimized()
    original_not_exact = cv2.ipp.useIPP_NotExact()
    rng = np.random.default_rng(342)
    image = rng.random((7, 19, 3), dtype=np.float32)
    kernel = rng.uniform(-1, 1, (1, area))
    calls = []

    def recorded(*args, **kwargs):
        calls.append(1)
        return spectral.filter2d(*args, **kwargs)

    monkeypatch.setattr(native_convolution, "_spectral_filter", recorded)
    try:
        cv2.setUseOptimized(optimized)
        cv2.ipp.setUseIPP(ipp)
        cv2.ipp.setUseIPP_NotExact(not_exact)
        expected = baseline(image, kernel, padding)
        exact(native_convolution.convolve(image, kernel, padding), expected)
        assert len(calls) == int(area >= (130 if optimized else 50))
        assert cv2.useOptimized() == optimized
        assert cv2.ipp.useIPP() == ipp
        assert cv2.ipp.useIPP_NotExact() == not_exact
    finally:
        cv2.setUseOptimized(original_optimized)
        cv2.ipp.setUseIPP(ipp)
        cv2.ipp.setUseIPP_NotExact(original_not_exact)


def test_convolve_spectral_replaces_filter_and_padding(monkeypatch):
    image = np.full((13, 17, 4), 0.5, np.float32)
    kernel = fractional_kernel(73.2, 169.7).astype(np.float64)
    expected = baseline(image, kernel, 3)

    def prohibited(*_args, **_kwargs):
        pytest.fail("Convolve retained a converted filtering or border operation")

    monkeypatch.setattr(cv2, "filter2D", prohibited)
    monkeypatch.setattr(cv2, "copyMakeBorder", prohibited)
    exact(native_convolution.convolve(image, kernel, 3), expected)
