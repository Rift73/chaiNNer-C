from __future__ import annotations

import numpy as np
import pytest

from nodes.impl.tensorrt.bfloat16 import bf16_bits_to_float32, float32_to_bf16_bits

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
    x = np.concatenate([special, random]).view(np.float32)
    np.testing.assert_array_equal(float32_to_bf16_bits(x), _reference(x))


def test_strided_planes_round_like_contiguous():
    rng = np.random.default_rng(1)
    hwc = rng.uniform(-2, 2, (5, 7, 3)).astype(np.float32)
    chw = hwc.transpose(2, 0, 1)[::-1]  # the node's BGR -> RGB planar view
    np.testing.assert_array_equal(
        float32_to_bf16_bits(chw), _reference(np.ascontiguousarray(chw))
    )


def test_bf16_to_float32_is_exact_for_every_value():
    every = np.arange(65536, dtype=np.uint32).astype(np.uint16)
    expected = every.view(ml_dtypes.bfloat16).astype(np.float32).view(np.uint32)
    np.testing.assert_array_equal(bf16_bits_to_float32(every).view(np.uint32), expected)
