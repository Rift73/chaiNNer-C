"""cn_planar_to_interleaved_f32, the TensorRT node's output conversion: at every
level this CPU has, (C, H, W) float16 or float32 planes become NumPy's float32
(H, W, C) image bit for bit, NaN payloads and subnormals included."""

from __future__ import annotations

import ctypes as ct

import numpy as np
import pytest
from test_isa_dispatch import isa_get, isa_set

from nodes.impl.native_framework_images import _api, planar_to_interleaved

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
    assert convert(base, 0, 64, 8, 8, 8, 3, 0, base + 256) != 0
    assert convert(base + 2, 0, 64, 8, 8, 8, 3, 0, base + 8192) != 0
    assert convert(base, 0, 64, 8, 8, 8, 3, 0, base + 8192 + 2) != 0
    assert convert(base, 0, 64, 8, 8, 9, 3, 0, base + 8192) != 0  # width > row stride
    assert convert(base, 0, 64, 8, 8, 8, 3, 0, ct.c_void_p(base + 8192)) == 0
