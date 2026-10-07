"""Full float32 CPU wavelet domain and per-call floating-point controls."""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from test_wavelet_ops import ORIGINAL, exact

from nodes.impl import native_wavelet as native


def source(shape, pattern):
    rng = np.random.default_rng(39271)
    if pattern == "subnormal":
        value = rng.integers(0, 1 << 23, shape, np.uint32).view(np.float32)
    elif pattern == "allbits":
        value = rng.integers(0, 2**32, shape, np.uint32).view(np.float32)
    elif pattern == "signedzero":
        value = np.resize(np.array([0, 0x80000000], np.uint32), shape).view(np.float32)
    elif pattern == "nonfinite":
        value = np.resize(
            np.array([0, -0.0, np.nan, np.inf, -np.inf], np.float32), shape
        )
    else:
        value = rng.uniform(-10, 10, shape).astype(np.float32)
    return torch.from_numpy(value)


@pytest.mark.parametrize(
    "shape", [(1, 3, 1, 1), (1, 3, 1, 7), (2, 3, 7, 11), (1, 3, 81, 97)]
)
@pytest.mark.parametrize(
    "pattern", ["subnormal", "allbits", "signedzero", "nonfinite", "signed"]
)
@pytest.mark.parametrize("levels", [1, 3, 5, 10])
def test_entire_cpu_float32_domain(shape, pattern, levels):
    image = source(shape, pattern)
    before = image.numpy().tobytes()
    actual = native.decomposition(image, levels)
    assert actual is not None
    for a, b in zip(actual, ORIGINAL.wavelet_decomposition(image, levels), strict=True):
        exact(a, b)
    style = torch.flip(image, (-1,))
    result = native.reconstruction(image, style, levels)
    assert result is not None
    exact(result, ORIGINAL.wavelet_reconstruction(image, style, levels))
    assert image.numpy().tobytes() == before


@pytest.mark.parametrize("pattern", ["subnormal", "allbits", "signedzero", "nonfinite"])
@pytest.mark.parametrize(
    "layout", ["strided", "channels_last", "transpose", "unaligned"]
)
def test_foreign_layout_uses_c_for_exceptional_values(pattern, layout):
    image = source((2, 3, 11, 17), pattern)
    if layout == "strided":
        image = image[:, :, ::2, ::2]
    elif layout == "transpose":
        image = image.transpose(-1, -2)
    elif layout == "channels_last":
        image = image.contiguous(memory_format=torch.channels_last)
    else:
        value = image.numpy()
        copy = np.ndarray(
            value.shape, np.float32, bytearray(value.nbytes + 1), offset=1
        )
        copy[:] = value
        image = torch.from_numpy(copy)
    actual = native.decomposition(image, 3)
    assert actual is not None
    for a, b in zip(actual, ORIGINAL.wavelet_decomposition(image, 3), strict=True):
        exact(a, b)


def test_exceptional_image_never_delegates_to_framework_convolution(monkeypatch):
    image = source((1, 3, 11, 17), "subnormal")
    expected = ORIGINAL.wavelet_reconstruction(image, image, 5)

    def forbidden(*args, **kwargs):
        raise AssertionError("Supported float32 CPU wavelets must execute C")

    monkeypatch.setattr(torch.nn.functional, "conv2d", forbidden)
    result = native.reconstruction(image, image, 5)
    assert result is not None
    exact(result, expected)


def test_pool_inherits_denormal_controls_on_each_concurrent_call():
    # An earlier pool user may have different FTZ/DAZ controls. Large inputs
    # force helper work; compare with an independent oneDNN reference for each
    # caller state. No timing or GPU measurements are involved.
    image = source((1, 3, 91, 103), "subnormal")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        expected = {}
        for flush in (False, True):
            torch.set_flush_denormal(flush)
            expected[flush] = ORIGINAL.wavelet_reconstruction(image, image, 3)
        torch.set_flush_denormal(False)

        def call(index):
            flush = bool(index % 2)
            torch.set_flush_denormal(flush)
            try:
                result = native.reconstruction(image, image, 3)
                assert result is not None
                # Turn flushing off for bit comparison of the returned array.
                torch.set_flush_denormal(False)
                exact(result, expected[flush])
                return result.numpy().tobytes()
            finally:
                torch.set_flush_denormal(False)

        for _ in range(2):
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(call, range(12)))
            assert len(set(results[::2])) == 1
            assert len(set(results[1::2])) == 1
    finally:
        torch.set_flush_denormal(False)
        torch.set_num_threads(previous_threads)


def test_c_wavelet_buffer_ranges_are_checked_before_writes():
    api = native._api()
    arrays = [np.full((1, 3, 3, 5), 123, np.float32) for _ in range(3)]
    source, high, low = arrays
    pointer = ct.POINTER(ct.c_float)

    def p(value):
        return ct.cast(value, pointer)

    for address, status in [
        (source.ctypes.data + 1, 1),
        (high.ctypes.data, 1),
        (high.ctypes.data + 4, 1),
        (ct.c_size_t(-4).value, 2),
    ]:
        assert (
            api.cn_wavelet_blur_f32(p(address), p(high.ctypes.data), 1, 3, 5, 1, 1)
            == status
        )
        assert (
            api.cn_wavelet_decompose_f32(
                p(address), p(high.ctypes.data), p(low.ctypes.data), 1, 3, 5, 3, 1
            )
            == status
        )
        assert (
            api.cn_wavelet_reconstruct_f32(
                p(address), p(source.ctypes.data), p(high.ctypes.data), 1, 3, 5, 3, 1
            )
            == status
        )
        for array in arrays:
            np.testing.assert_array_equal(array, 123)
    assert (
        api.cn_wavelet_decompose_f32(
            p(source.ctypes.data),
            p(high.ctypes.data),
            p(high.ctypes.data),
            1,
            3,
            5,
            3,
            1,
        )
        == 1
    )


def test_explicit_non_onednn_alternate_reduction_remains_compatible():
    # 1x1 convolution becomes a BLAS dot when oneDNN is explicitly disabled.
    # This retained framework setting differs from the app's default CPU path.
    image = source((1, 3, 1, 1), "signed")
    with torch.backends.mkldnn.flags(enabled=False):
        assert native.reconstruction(image, image, 3) is None
