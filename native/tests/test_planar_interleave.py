"""cn_planar_to_interleaved_f32, the upscaling backends' output conversion (with its
red/blue swap and finiteness report), and cn_planes_to_interleaved_f32, Load Video's
and Clip Frames': at every level this CPU has, float16 or float32 planes (one
(C, H, W) buffer, or separate planes in any order) become NumPy's float32 (H, W, C)
image bit for bit, NaN payloads and subnormals included. bfloat16 planes (uint16 bits)
become the float32 they are the top half of, and cn_interleaved_to_bfloat_planes,
a TensorRT engine's input, rounds to even with a NaN quieted, at every level."""

from __future__ import annotations

import ctypes as ct
import threading

import numpy as np
import pytest
from test_isa_dispatch import isa_set

from nodes.impl import native_framework_images as images
from nodes.impl.native_framework_images import (
    _api,
    interleaved_to_bfloat_planes,
    planar_to_interleaved,
    planar_to_interleaved_checked,
    planes_to_interleaved,
)

ROWS, WIDTH, STRIDE = 64, 1027, 1040  # 1027 % 8 == 3: every row has a scalar tail


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


@pytest.mark.parametrize("channels", [1, 3, 4, 5])
@pytest.mark.parametrize("swap", [False, True], ids=["rgb", "swapped"])
def test_every_bfloat_is_the_top_half_of_its_float32(levels, channels: int, swap: bool):
    if swap and channels < 3:
        pytest.skip("the swap takes three or more channels")
    rng = np.random.default_rng(40 + channels)
    patterns = (np.arange(ROWS * WIDTH) % 65536).astype(np.uint16)
    buffer = np.zeros((channels, ROWS, STRIDE), np.uint16)
    for plane in buffer:
        plane[:, :WIDTH] = rng.permutation(patterns).reshape(ROWS, WIDTH)
    planes = buffer[:, :, :WIDTH]
    order = [2, 1, 0, *range(3, channels)] if swap else list(range(channels))
    expected = (planes[order].astype(np.uint32) << 16).transpose(1, 2, 0)
    for level in levels:
        assert isa_set(level) == level
        image = planar_to_interleaved(planes, swap_red_blue=swap, bfloat16=True)
        np.testing.assert_array_equal(_bits(image), expected, err_msg=str(level))


def _bfloat_bits(values: np.ndarray) -> np.ndarray:
    """To nearest, ties to even; a NaN becomes the quiet NaN with its sign."""
    bits = np.ascontiguousarray(values).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    out = (rounded >> np.uint32(16)).astype(np.uint16)
    nan = (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    out[nan] = np.where(bits[nan] >> np.uint32(31), 0xFFC0, 0x7FC0)
    return out


@pytest.mark.parametrize("channels", [1, 3, 4, 5])
@pytest.mark.parametrize("swap", [False, True], ids=["rgb", "swapped"])
def test_interleaved_rounds_to_bfloat_planes(levels, channels: int, swap: bool):
    if swap and channels < 3:
        pytest.skip("the swap takes three or more channels")
    rng = np.random.default_rng(50 + channels)
    # A tile of a wider image: rows longer than the tile's, every row with a scalar tail.
    raw = rng.integers(0, 2**32, (ROWS, STRIDE, channels), dtype=np.uint64)
    words = raw.astype(np.uint32)
    words[0, :11, :] = np.array(
        [
            0x00000000,
            0x80000000,
            0x00000001,
            0x7F7FFFFF,
            0x7F800000,
            0xFF800000,
            0x3F808000,
            0x3F818000,
            0x7FC00000,
            0x7F800001,
            0xFFFFFFFF,
        ],
        np.uint32,
    )[:, None]
    words[1, WIDTH - 3 : WIDTH, :] = 0xFFC12345  # NaNs in a row's tail
    tile = words.view(np.float32)[:, :WIDTH]
    order = [2, 1, 0, *range(3, channels)] if swap else list(range(channels))
    expected = _bfloat_bits(tile[:, :, order].transpose(2, 0, 1))
    for level in levels:
        assert isa_set(level) == level
        out = np.empty((channels, ROWS, WIDTH), np.uint16)
        interleaved_to_bfloat_planes(tile, out, swap_red_blue=swap)
        np.testing.assert_array_equal(out, expected, err_msg=str(level))


def test_the_first_calls_set_the_signatures_one_at_a_time():
    # Items of one tile reach these conversions together, outside any session's lock.
    _api.cache_clear()
    done = threading.Event()

    def first_call():
        _api()
        done.set()

    with images._api_lock:
        thread = threading.Thread(target=first_call)
        thread.start()
        assert not done.wait(0.1)
    thread.join(10)
    assert done.is_set()


def test_bfloat_conversions_are_refused_as_the_others_are():
    buffer = np.zeros(4096, np.float32)
    convert = _api().cn_interleaved_to_bfloat_planes
    base = buffer.ctypes.data
    # 8 x 8 x 3 float32 pixels; the planes would start inside them
    assert convert(base, 24, 8, 8, 3, 0, base + 8192) == 0
    assert convert(base, 24, 8, 8, 3, 0, base + 256) != 0
    assert convert(base + 2, 24, 8, 8, 3, 0, base + 8192) != 0
    assert convert(base, 24, 8, 8, 3, 0, base + 8192 + 1) != 0
    assert convert(base, 23, 8, 8, 3, 0, base + 8192) != 0  # rows narrower than a row
    assert convert(base, 8, 8, 8, 1, 1, base + 8192) != 0  # a swap needs three channels
    assert convert(None, 24, 8, 8, 3, 0, base + 8192) != 0
    assert convert(base, 24, 8, 8, 0, 0, base + 8192) != 0
    with pytest.raises(ValueError, match="uint16 storage"):
        interleaved_to_bfloat_planes(
            np.zeros((2, 2, 3), np.float32), np.zeros((2, 2, 3), np.uint16)
        )
    with pytest.raises(ValueError, match="writable"):
        out = np.zeros((3, 2, 2), np.uint16)
        out.setflags(write=False)
        interleaved_to_bfloat_planes(np.zeros((2, 2, 3), np.float32), out)
    interleave = _api().cn_planar_to_interleaved_f32
    assert interleave(base, 2, 64, 8, 8, 8, 3, 0, base + 8192, None) == 0
    assert interleave(base, 3, 64, 8, 8, 8, 3, 0, base + 8192, None) != 0
    for planes, flag in (
        (np.zeros((3, 2, 2), np.uint16), False),
        (np.zeros((3, 2, 2), np.float16), True),
    ):
        with pytest.raises(ValueError, match="uint16 planes of bfloat16 bits"):
            planar_to_interleaved(planes, bfloat16=flag)


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
        return convert(planes, rows, 0, 8, width, 3, 1.0, 0.0, out)

    planes, strides = [base, base + 256, base + 512], [8, 8, 8]
    assert call(planes, strides, base + 8192) == 0
    assert call(planes, strides, base + 512 + 4) != 0  # inside the last plane
    assert call([base, None, base + 512], strides, base + 8192) != 0
    assert call([base, base + 258, base + 512], strides, base + 8192) != 0
    assert call(planes, strides, base + 8192 + 2) != 0
    assert call(planes, [8, 7, 8], base + 8192) != 0  # width > a row stride
    assert (
        convert(None, (ct.c_size_t * 3)(8, 8, 8), 0, 8, 8, 3, 1.0, 0.0, base + 8192) != 0
    )
    rows = (ct.c_size_t * 3)(8, 8, 8)
    pointers = (ct.c_void_p * 3)(*planes)
    assert convert(pointers, rows, 0, 8, 8, 3, 0.0, 0.0, base + 8192) != 0
    assert convert(pointers, rows, 0, 8, 8, 3, 1.0, float("nan"), base + 8192) != 0
    with pytest.raises(ValueError, match="rows of contiguous samples"):
        planes_to_interleaved([np.zeros((4, 4), np.float32)[:, ::2]] * 3)
    with pytest.raises(ValueError, match="one shape and dtype"):
        planes_to_interleaved(
            [np.zeros((4, 4), np.float32), np.zeros((4, 5), np.float32)]
        )


@pytest.mark.parametrize("width", [1, 7, 8, 9, 17, 64])
@pytest.mark.parametrize("gain", [65535 / 65280, 65535 / 65472])
def test_a_gain_multiplies_in_the_pass_and_stops_at_one(width, gain, levels):
    """Load Video's float RGB: at every level and across the 8-pixel groups' tails, each
    value is float32 value * float32 gain, at most 1."""
    rng = np.random.default_rng(width)
    planes = rng.random((3, 5, width), dtype=np.float32)
    planes[:, :, -1] = 1.0
    gain32 = np.float32(gain)
    expected = np.minimum(
        np.stack((planes[1], planes[0], planes[2]), axis=2) * gain32, np.float32(1)
    )
    for level in levels:
        assert isa_set(level) == level
        image = planes_to_interleaved((planes[1], planes[0], planes[2]), gain)
        np.testing.assert_array_equal(image, expected, err_msg=str(level))
    assert image.max() == 1.0


@pytest.mark.parametrize("width", [1, 7, 8, 9, 17, 64])
def test_a_gain_and_offset_stay_between_zero_and_one(width, levels):
    """Load Video's limited-range gray: at every level and across the 8-pixel groups' tails,
    each value is float32 value * gain + offset, in two steps, between 0 and 1."""
    gain, offset = 65535 / (219 * 256), -16 / 219
    rng = np.random.default_rng(width)
    planes = rng.random((3, 5, width), dtype=np.float32)
    planes[:, :, 0] = 0.0
    planes[:, :, -1] = 1.0
    scaled = np.stack(tuple(planes), axis=2) * np.float32(gain) + np.float32(offset)
    expected = np.maximum(np.minimum(scaled, np.float32(1)), np.float32(0))
    for level in levels:
        assert isa_set(level) == level
        image = planes_to_interleaved(tuple(planes), gain, offset)
        np.testing.assert_array_equal(image, expected, err_msg=str(level))
    assert not np.signbit(image).any()
