"""cn_planar_to_interleaved_f32, the upscaling backends' output conversion (with its
red/blue swap and finiteness report), and cn_planes_to_interleaved_f32, Load Video's
and Clip Frames': at every level this CPU has, float16 or float32 planes (one
(C, H, W) buffer, or separate planes in any order) become NumPy's float32 (H, W, C)
image bit for bit, NaN payloads and subnormals included."""

from __future__ import annotations

import ctypes as ct

import numpy as np
import pytest
from test_isa_dispatch import isa_get, isa_set

from nodes.impl.native_framework_images import (
    _api,
    planar_to_interleaved,
    planar_to_interleaved_checked,
    planes_to_interleaved,
)

ROWS, WIDTH, STRIDE = 64, 1027, 1040  # 1027 % 8 == 3: every row has a scalar tail


@pytest.fixture
def levels():
    """Every level this CPU has, scalar first; the original level afterwards."""
    effective, _, cpu = isa_get()
    try:
        yield range(cpu + 1)
    finally:
        assert isa_set(effective) == effective


def _expected(planes: np.ndarray, reverse: bool) -> np.ndarray:
    ordered = planes[::-1] if reverse else planes
    return ordered.transpose(1, 2, 0).astype(np.float32)


def _bits(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image).view(np.uint32)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("reverse", [False, True], ids=["rgb", "reversed"])
def test_every_half_converts_as_numpy(levels, channels: int, reverse: bool):
    rng = np.random.default_rng(channels)
    patterns = (np.arange(ROWS * WIDTH) % 65536).astype(np.uint16)
    buffer = np.zeros((channels, ROWS, STRIDE), np.uint16)
    for plane in buffer:
        plane[:, :WIDTH] = rng.permutation(patterns).reshape(ROWS, WIDTH)
    planes = buffer.view(np.float16)[:, :, :WIDTH]
    assert all(np.unique(plane).size == 65536 for plane in planes.view(np.uint16))
    expected = _bits(_expected(planes, reverse))
    for level in levels:
        assert isa_set(level) == level
        np.testing.assert_array_equal(
            _bits(planar_to_interleaved(planes, reverse)), expected, err_msg=str(level)
        )


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_float32_bits_are_moved_unchanged(levels, channels: int):
    rng = np.random.default_rng(10 + channels)
    raw = rng.integers(0, 2**32, (channels, ROWS, STRIDE), dtype=np.uint64)
    planes = raw.astype(np.uint32).view(np.float32)[:, 3:, :WIDTH]
    for reverse in (False, True):
        expected = _bits(_expected(planes, reverse))
        for level in levels:
            assert isa_set(level) == level
            np.testing.assert_array_equal(
                _bits(planar_to_interleaved(planes, reverse)),
                expected,
                err_msg=f"{level} {reverse}",
            )


def test_overlap_and_misalignment_are_refused():
    buffer = np.zeros(4096, np.float32)
    convert = _api().cn_planar_to_interleaved_f32
    base = buffer.ctypes.data
    # 3 x 8 x 8 float32 planes; the output would start inside them
    assert convert(base, 0, 64, 8, 8, 8, 3, 0, base + 256, None) != 0
    assert convert(base + 2, 0, 64, 8, 8, 8, 3, 0, base + 8192, None) != 0
    assert convert(base, 0, 64, 8, 8, 8, 3, 0, base + 8192 + 2, None) != 0
    assert (
        convert(base, 0, 64, 8, 8, 9, 3, 0, base + 8192, None) != 0
    )  # width > row stride
    assert convert(base, 0, 64, 8, 8, 8, 3, 0, ct.c_void_p(base + 8192), None) == 0


@pytest.mark.parametrize("channels", [3, 4, 5])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_red_and_blue_swap_as_swap_red_blue(levels, channels: int, dtype):
    # Three or four channels as the backends' swap_red_blue orders them (BGR, BGRA);
    # any further channel stays in place.
    rng = np.random.default_rng(20 + channels)
    storage = np.uint16 if dtype == np.float16 else np.uint32
    raw = rng.integers(0, 2**32, (channels, ROWS, STRIDE), dtype=np.uint64)
    planes = raw.astype(storage).view(dtype)[:, 1:, :WIDTH]
    order = [2, 1, 0, *range(3, channels)]
    expected = _bits(planes[order].transpose(1, 2, 0).astype(np.float32))
    for level in levels:
        assert isa_set(level) == level
        np.testing.assert_array_equal(
            _bits(planar_to_interleaved(planes, swap_red_blue=True)),
            expected,
            err_msg=str(level),
        )


def test_swap_needs_three_channels_and_excludes_reverse():
    buffer = np.zeros(4096, np.float32)
    convert = _api().cn_planar_to_interleaved_f32
    base = buffer.ctypes.data
    assert convert(base, 0, 64, 8, 8, 8, 2, 2, base + 8192, None) != 0
    assert convert(base, 0, 64, 8, 8, 8, 3, 3, base + 8192, None) != 0
    assert convert(base, 0, 64, 8, 8, 8, 3, 2, base + 8192, None) == 0
    with pytest.raises(ValueError, match="either reverses or swaps"):
        planar_to_interleaved(np.zeros((3, 2, 2), np.float32), True, swap_red_blue=True)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_finiteness_is_reported_from_every_sample(levels, channels: int, dtype):
    # One Inf or NaN anywhere (an 8-pixel group or a row's scalar tail, any plane)
    # makes the image non-finite; the image itself is planar_to_interleaved's.
    planes = np.random.default_rng(30).random((channels, 9, 21)).astype(dtype)
    specials = (np.inf, -np.inf, np.nan)
    positions = [(0, 0, 0), (channels - 1, 4, 7), (channels // 2, 8, 20), (0, 3, 17)]
    for level in levels:
        assert isa_set(level) == level
        image, finite = planar_to_interleaved_checked(planes, swap_red_blue=False)
        assert finite
        assert _bits(image).tobytes() == _bits(_expected(planes, False)).tobytes()
        for position in positions:
            for special in specials:
                damaged = planes.copy()
                damaged[position] = special
                image, finite = planar_to_interleaved_checked(damaged)
                assert not finite, (level, position, special)
                assert (
                    _bits(image).tobytes() == _bits(_expected(damaged, False)).tobytes()
                )


# Each channel count's plane order: Load Video's G, B and R planes taken as B, G, R.
ORDERS = {1: (0,), 3: (1, 0, 2), 4: (2, 0, 3, 1)}


def _separate_planes(rng: np.random.Generator, channels: int, dtype: type) -> list:
    """Planes in buffers of their own, each with its own row stride and offset."""
    bits = np.uint16 if dtype is np.float16 else np.uint32
    planes = []
    for k in range(channels):
        raw = rng.integers(0, np.iinfo(bits).max + 1, (ROWS, WIDTH + 8 * k + 5))
        planes.append(raw.astype(bits).view(dtype)[:, k : k + WIDTH])
    return planes


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("dtype", [np.float16, np.float32], ids=["half", "float"])
def test_separate_planes_convert_as_numpy_in_their_order(
    levels, channels: int, dtype: type
):
    rng = np.random.default_rng(20 + channels)
    planes = _separate_planes(rng, channels, dtype)
    ordered = [planes[k] for k in ORDERS[channels]]
    expected = _bits(np.stack(ordered, axis=2).astype(np.float32))
    for level in levels:
        assert isa_set(level) == level
        np.testing.assert_array_equal(
            _bits(planes_to_interleaved(ordered)), expected, err_msg=str(level)
        )


def test_one_buffers_planes_in_another_order(levels):
    # Clip Frames: vspipe's G, B, R planes of one (3, H, W) buffer as a BGR image.
    rng = np.random.default_rng(30)
    raw = rng.integers(0, 2**32, (3, ROWS, WIDTH), dtype=np.uint64)
    planes = raw.astype(np.uint32).view(np.float32)
    expected = _bits(np.stack((planes[1], planes[0], planes[2]), axis=2))
    for level in levels:
        assert isa_set(level) == level
        image = planes_to_interleaved((planes[1], planes[0], planes[2]))
        np.testing.assert_array_equal(_bits(image), expected, err_msg=str(level))


def test_separate_planes_refused_as_one_buffer_is():
    buffer = np.zeros(4096, np.float32)
    convert = _api().cn_planes_to_interleaved_f32
    base = buffer.ctypes.data

    def call(pointers, strides, out, width=8):
        planes = (ct.c_void_p * 3)(*pointers)
        rows = (ct.c_size_t * 3)(*strides)
        return convert(planes, rows, 0, 8, width, 3, out)

    planes, strides = [base, base + 256, base + 512], [8, 8, 8]
    assert call(planes, strides, base + 8192) == 0
    assert call(planes, strides, base + 512 + 4) != 0  # inside the last plane
    assert call([base, None, base + 512], strides, base + 8192) != 0
    assert call([base, base + 258, base + 512], strides, base + 8192) != 0
    assert call(planes, strides, base + 8192 + 2) != 0
    assert call(planes, [8, 7, 8], base + 8192) != 0  # width > a row stride
    assert convert(None, (ct.c_size_t * 3)(8, 8, 8), 0, 8, 8, 3, base + 8192) != 0
    with pytest.raises(ValueError, match="rows of contiguous samples"):
        planes_to_interleaved([np.zeros((4, 4), np.float32)[:, ::2]] * 3)
    with pytest.raises(ValueError, match="one shape and dtype"):
        planes_to_interleaved(
            [np.zeros((4, 4), np.float32), np.zeros((4, 5), np.float32)]
        )
