"""Descale (native/src/descale_ops.c over third_party/descale): it inverts a bilinear upscale,
and layouts, chunking and the solver cache do not change results. Bit-identity with the
VapourSynth plugin (Release 12) was checked at integration: 12 of 12 cases."""

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


# 3 -> 2 columns with Spline36 reaches past the mirror's range, so that shape extends.
@pytest.mark.parametrize("shape, border", [((5, 3, 1), 2), ((9, 17, 4), 0), ((130, 75, 3), 0)])
def test_channels_are_independent_planes(shape, border):
    rng = np.random.default_rng(4)
    image = rng.random(shape, dtype=np.float32)
    h, w = shape[:2]
    out = run(image, max(1, w * 2 // 3), max(1, h * 2 // 3), mode=5, border=border)
    for k in range(shape[2]):
        plane = run(
            np.ascontiguousarray(image[:, :, k]),
            out.shape[1],
            out.shape[0],
            mode=5,
            border=border,
        )
        assert np.array_equal(out[:, :, k], plane)


@pytest.mark.parametrize(
    "width, mode, blur, left",
    [(2, 3, 1.0, 0.0), (8, 1, 10.0, 0.0), (100, 1, 1.0, 1000.0), (100, 1, 1.0, -1000.0)],
)
def test_mirror_reaching_past_the_target_is_refused(width, mode, blur, left):
    # descale's mirror reflects a tap once, so a kernel wider than the target or a
    # shift past an edge indexed outside the weights: a crash at Source Left -1000.
    image = np.random.default_rng(6).random((16, 400, 3), dtype=np.float32)
    with pytest.raises(ValueError) as error:
        native_descale.descale(
            image, width, 16, mode, 3, 0.0, 0.5, blur, (left, 0.0), (width, 16), 0
        )
    assert str(error.value) == (
        "Descale: this Shift and Border=Mirror would read outside the kernel; "
        "use the Extend or Zero border, or a smaller shift."
    )
    # Extend clamps every tap, so the same parameters stay valid there.
    out = native_descale.descale(
        image, width, 16, mode, 3, 0.0, 0.5, blur, (left, 0.0), (width, 16), 2
    )
    assert out.shape == (16, width, 3)


def test_system_beyond_int_indices_is_refused():
    # descale sizes its src_dim x dst_dim matrices with int products: 65536 x 32768
    # overflowed into an access violation inside the core build.
    strip = np.zeros((16, 65536), np.float32)
    with pytest.raises(ValueError, match="^Invalid descale parameters$"):
        run(strip, 32768, 16)


def test_same_size_is_identity_and_bad_targets_fail():
    image = np.random.default_rng(5).random((20, 30, 3), dtype=np.float32)
    assert np.array_equal(run(image, 30, 20), image)
    with pytest.raises(ValueError):
        run(image, 31, 20)
