"""A TensorRT engine's bfloat16 tensors, held as uint16 bits as NumPy has no such type:
both native conversions give exactly what ml_dtypes gives."""

from __future__ import annotations

import numpy as np
import pytest

from nodes.impl.native_framework_images import (
    interleaved_to_bfloat_planes,
    planar_to_interleaved,
)

ml_dtypes = pytest.importorskip("ml_dtypes")


def _reference(x: np.ndarray) -> np.ndarray:
    return x.astype(ml_dtypes.bfloat16).view(np.uint16)


def test_float32_to_bf16_matches_ml_dtypes():
    rng = np.random.default_rng(0)
    random = rng.integers(0, 2**32, 2_000_000, dtype=np.uint64).astype(np.uint32)
    special = np.array(
        [
            0x00000000,  # +0
            0x80000000,  # -0
            0x00000001,  # smallest subnormal
            0x7F7FFFFF,  # largest finite: rounds to Inf
            0x7F800000,  # +Inf
            0xFF800000,  # -Inf
            0x3F808000,  # tie, even below: stays
            0x3F818000,  # tie, odd below: rounds up
            0x7FC00000,  # quiet NaN
            0x7F800001,  # signalling NaN: rounding alone would give Inf
            0xFFFFFFFF,  # NaN: rounding alone would wrap to +0
        ],
        np.uint32,
    )
    x = np.concatenate([special, random]).view(np.float32).reshape(-1, 1, 1)
    out = np.empty((1, x.shape[0], 1), np.uint16)
    interleaved_to_bfloat_planes(x, out)
    np.testing.assert_array_equal(out, _reference(x.transpose(2, 0, 1)))


@pytest.mark.parametrize("channels", [3, 4])
def test_strided_tiles_round_as_contiguous_ones(channels: int):
    # A tile of a larger image, BGR(A) -> RGB(A), as the session uploads it.
    rng = np.random.default_rng(1)
    image = rng.uniform(-2, 2, (9, 40, channels)).astype(np.float32)
    tile = image[2:7, 3:21]
    order = [2, 1, 0, 3][:channels]
    out = np.empty((channels, 5, 18), np.uint16)
    interleaved_to_bfloat_planes(tile, out, swap_red_blue=True)
    np.testing.assert_array_equal(
        out, _reference(np.ascontiguousarray(tile[:, :, order].transpose(2, 0, 1)))
    )
    flipped = tile[:, :, ::-1]  # a layout the kernel does not take: read through a copy
    interleaved_to_bfloat_planes(flipped, out)
    np.testing.assert_array_equal(
        out, _reference(np.ascontiguousarray(flipped.transpose(2, 0, 1)))
    )


def test_bf16_to_float32_is_exact_for_every_value():
    every = np.arange(65536, dtype=np.uint32).astype(np.uint16).reshape(1, 256, 256)
    expected = every.view(ml_dtypes.bfloat16).astype(np.float32).view(np.uint32)
    image = planar_to_interleaved(every, bfloat16=True)
    assert image.shape == (256, 256, 1)
    np.testing.assert_array_equal(image.view(np.uint32)[:, :, 0], expected[0])
