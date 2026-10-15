"""Descale (native/src/descale_ops.c over third_party/descale): it inverts a bilinear upscale,
and layouts, chunking and the solver cache do not change results. Bit-identity with the
VapourSynth plugin was checked once at integration (8 of 9 cases; the ninth is Release 11's
Lanczos-with-blur window, which Release 12 changed)."""

import cv2
import numpy as np
import pytest

from nodes.impl import native_descale

ARGS = (3, 0.0, 0.5, 1.0)


def run(image, width, height, mode=1, border=2):
    return native_descale.descale(
        image, width, height, mode, *ARGS, (0.0, 0.0), (width, height), border
    )


def test_inverts_bilinear_upscale():
    rng = np.random.default_rng(3)
    small = rng.random((45, 67, 3), dtype=np.float32)
    large = cv2.resize(small, (100, 70), interpolation=cv2.INTER_LINEAR)
    # border 2 (extend) is OpenCV's replicate padding.
    assert np.abs(run(large, 67, 45) - small).max() < 1e-4


@pytest.mark.parametrize("shape", [(5, 3, 1), (9, 17, 4), (130, 75, 3)])
def test_channels_are_independent_planes(shape):
    rng = np.random.default_rng(4)
    image = rng.random(shape, dtype=np.float32)
    h, w = shape[:2]
    out = run(image, max(1, w * 2 // 3), max(1, h * 2 // 3), mode=5, border=0)
    for k in range(shape[2]):
        plane = run(
            np.ascontiguousarray(image[:, :, k]),
            out.shape[1],
            out.shape[0],
            mode=5,
            border=0,
        )
        assert np.array_equal(out[:, :, k], plane)


def test_same_size_is_identity_and_bad_targets_fail():
    image = np.random.default_rng(5).random((20, 30, 3), dtype=np.float32)
    assert np.array_equal(run(image, 30, 20), image)
    with pytest.raises(ValueError):
        run(image, 31, 20)
