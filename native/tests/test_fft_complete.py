"""Direct checked C FFT comparisons against the installed NumPy complex FFT."""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from nodes.impl.native import check, lib


def fft(data, inverse=False):
    data = np.require(data, dtype=np.complex128, requirements=["C", "A"])
    out = np.empty_like(data)
    function = lib().cn_fft2
    function.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_int]
    function.restype = ct.c_int
    check(function(data.ctypes.data, out.ctypes.data, *data.shape, int(inverse)))
    return out


@pytest.mark.parametrize("height", [1, 2, 3, 4, 5, 7, 8, 11, 13, 16, 31, 47, 53, 97])
@pytest.mark.parametrize("width", [1, 3, 4, 5, 7, 11, 17, 32, 53, 79])
@pytest.mark.parametrize("inverse", [False, True])
def test_fft_codelets_and_prime_reductions_exact(height, width, inverse):
    rng = np.random.default_rng(35)
    data = rng.normal(size=(height, width)) + 1j * rng.normal(size=(height, width))
    before = data.copy()
    expected = np.fft.ifftn(data) if inverse else np.fft.fftn(data)
    actual = fft(data, inverse)
    np.testing.assert_array_equal(
        actual.view(np.uint64), expected.copy().view(np.uint64)
    )
    np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize("shape", [(1, 257), (263, 1), (257, 263), (513, 257)])
def test_concurrent_prime_and_parallel_axes(shape):
    rng = np.random.default_rng(771)
    data = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    expected = np.fft.fftn(data).copy()

    def work(_):
        actual = fft(data)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        back = fft(actual, True)
        reference = np.fft.ifftn(expected).copy()
        np.testing.assert_array_equal(back.view(np.uint64), reference.view(np.uint64))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(4)))


def test_fft_invalid_abi_before_output_writes():
    image = np.zeros((2, 3), np.complex128)
    fft(image)  # assign exact ABI
    out = np.full_like(image, 123 + 45j)
    original = out.copy()
    f = lib().cn_fft2
    assert f(image.ctypes.data, out.ctypes.data, 0, 3, 0) == 1
    assert f(image.ctypes.data, out.ctypes.data, 2, 3, 2) == 1
    assert f(None, out.ctypes.data, 2, 3, 0) == 1
    assert f(image.ctypes.data + 1, out.ctypes.data, 2, 3, 0) == 1
    assert f(image.ctypes.data, out.ctypes.data, ct.c_size_t(-1).value, 3, 0) == 2
    assert f(image.ctypes.data, image.ctypes.data + 16, 1, 2, 0) == 1
    np.testing.assert_array_equal(out, original)
