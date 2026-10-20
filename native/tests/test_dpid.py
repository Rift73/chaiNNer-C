"""DPID (native/src/dpid_ops.c) against a float64 NumPy port of mergian/dpid's main.cu.

The oracle follows kernelGuidance, calcAverage and kernelDownsampling with the guidance and
the output in float (no 8-bit truncation) and the CUDA footprint edges (Matlab's dpid.m tests
the bottom edge against sy, a typo not reproduced). Its distance is the L2 norm over the colour
channels divided by the square root of their count: on 0..1 with three channels, main.cu's
division by sqrt(3 * 255^2) on 0..255. The C kernel keeps its images in float32, so results
agree within TOLERANCE (fable-decisions.md, 2026-10-21).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from native_images import seed_of

from nodes.impl import native_alpha_gamma
from nodes.impl import native_resample as native
from nodes.impl.native import ptr
from nodes.impl.resize import ResizeFilter, resize

TOLERANCE = 1e-5
LAMBDAS = [0.0, 0.5, 1.0, 2.5]
# ((height, width) in, (width, height) out): integer and non-integer ratios, a large ratio,
# single rows and columns, a growing side and a pure upscale.
GEOMETRIES = [
    ((4, 4), (2, 2)),
    ((37, 23), (10, 7)),
    ((48, 64), (17, 13)),
    ((9, 11), (4, 5)),
    ((64, 48), (3, 2)),
    ((1, 17), (5, 1)),
    ((13, 1), (1, 4)),
    ((9, 11), (1, 1)),
    ((7, 5), (3, 11)),
    ((5, 4), (13, 9)),
]


def axis(o: int, pitch: float, size: int) -> tuple[int, int, np.ndarray]:
    """main.cu's Local and contribution() along one axis: the samples [start, end) and their
    factors."""
    s = max(o * pitch, 0.0)
    e = min((o + 1) * pitch, float(size))
    start, end = math.floor(s), math.ceil(e)
    x = np.arange(start, end, dtype=np.float64)
    f = np.ones(end - start)
    f = np.where(x < s, f * (1.0 - (s - x)), f)
    f = np.where(x + 1.0 > e, f * (1.0 - ((x + 1.0) - e)), f)
    return start, end, f


def oracle_dpid(
    image: np.ndarray, out_w: int, out_h: int, lam: float, colour: int
) -> np.ndarray:
    """DPID of an (h, w, c) image whose first `colour` channels make the distance; every
    channel takes the weights."""
    src = np.asarray(image, np.float64)
    ih, iw, c = src.shape
    columns = [axis(ox, iw / out_w, iw) for ox in range(out_w)]
    rows = [axis(oy, ih / out_h, ih) for oy in range(out_h)]
    # kernelGuidance: the coverage-weighted mean of each footprint.
    guidance = np.empty((out_h, out_w, c))
    for oy, (y0, y1, fy) in enumerate(rows):
        for ox, (x0, x1, fx) in enumerate(columns):
            f = fy[:, None] * fx[None, :]
            patch = src[y0:y1, x0:x1]
            guidance[oy, ox] = (patch * f[:, :, None]).sum((0, 1)) / f.sum()
    out = np.empty((out_h, out_w, c))
    for oy, (y0, y1, fy) in enumerate(rows):
        for ox, (x0, x1, fx) in enumerate(columns):
            # calcAverage: (1 2 1) x (1 2 1), renormalised at the borders.
            average, total = np.zeros(c), 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    y, x = oy + dy, ox + dx
                    if 0 <= y < out_h and 0 <= x < out_w:
                        k = (2.0 if dy == 0 else 1.0) * (2.0 if dx == 0 else 1.0)
                        average += k * guidance[y, x]
                        total += k
            average /= total
            # kernelDownsampling: coverage x (distance / sqrt(colour))^lambda.
            patch = src[y0:y1, x0:x1]
            if lam == 0:
                weight = np.ones(patch.shape[:2])
            else:
                difference = average[:colour] - patch[:, :, :colour]
                distance = np.sqrt((difference**2).sum(-1)) / math.sqrt(colour)
                weight = distance**lam
            weight = weight * fy[:, None] * fx[None, :]
            if weight.sum() == 0:
                out[oy, ox] = average
            else:
                out[oy, ox] = (patch * weight[:, :, None]).sum((0, 1)) / weight.sum()
    return out


def oracle(
    image: np.ndarray, dimensions: tuple[int, int], lam: float, separate: bool = False
) -> np.ndarray:
    """native.dpid's channel rules: RGBA carries alpha with the colour weights, or with
    separate is a colour DPID beside an alpha DPID; other channel counts are all colour."""
    out_w, out_h = dimensions
    src = image[:, :, None] if image.ndim == 2 else image
    c = src.shape[2]
    if c == 4 and separate:
        return np.concatenate(
            [
                oracle_dpid(src[:, :, :3], out_w, out_h, lam, 3),
                oracle_dpid(src[:, :, 3:], out_w, out_h, lam, 1),
            ],
            axis=2,
        )
    return oracle_dpid(src, out_w, out_h, lam, 3 if c == 4 else c)


def noise(seed: int, shape: tuple[int, int], channels: int) -> np.ndarray:
    values = np.random.default_rng(seed).random((*shape, channels), dtype=np.float32)
    return values[:, :, 0] if channels == 1 else values


def assert_close(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    error = np.abs(actual.astype(np.float64) - expected).max()
    assert error <= TOLERANCE, f"max abs error {error:.3g}"


# channels, separate: gray, two colour channels, RGB, RGBA carried and RGBA separate.
MODES = [(1, False), (2, False), (3, False), (4, False), (4, True)]


@pytest.mark.parametrize("lam", LAMBDAS)
@pytest.mark.parametrize(("channels", "separate"), MODES)
@pytest.mark.parametrize(("shape", "dimensions"), GEOMETRIES)
def test_matches_oracle(shape, dimensions, channels, separate, lam):
    img = noise(seed_of(shape, dimensions, channels, separate, lam), shape, channels)
    before = img.copy()
    actual = native.dpid(img, dimensions, lam, separate)
    assert_close(actual, oracle(img, dimensions, lam, separate))
    np.testing.assert_array_equal(img, before)


@pytest.mark.parametrize("lam", [3.0, 10.0])
def test_large_lambdas_match_oracle(lam):
    img = noise(7, (40, 30), 3)
    assert_close(native.dpid(img, (11, 13), lam), oracle(img, (11, 13), lam))


def test_four_by_four_lambda_zero_is_the_box_average():
    img = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
    expected = np.array([[10, 18], [42, 50]], np.float32)[:, :, None] / 64
    np.testing.assert_array_equal(native.dpid(img, (2, 2), 0.0), expected)
    # Lambda 1 weighs each block's pixels by their distance to the block's tent average.
    assert_close(native.dpid(img, (2, 2), 1.0), oracle(img, (2, 2), 1.0))


@pytest.mark.parametrize("lam", LAMBDAS)
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_flat_image_weighs_zero_and_returns_the_tent_average(channels, lam):
    # Every distance is 0: the weight sum is 0 and each output is its tent average.
    img = np.full((8, 12, channels), 0.375, np.float32)
    for separate in (False, True):
        out = native.dpid(img, (6, 4), lam, separate)
        np.testing.assert_array_equal(out, np.full((4, 6, channels), 0.375, np.float32))


@pytest.mark.parametrize("lam", [0.5, 1.0])
def test_carried_alpha_takes_the_tent_average_where_colour_weighs_zero(lam):
    # Constant colour, varying alpha: the colour weights are all 0, so alpha is the tent
    # average of its guidance, not a weighted footprint mean.
    img = noise(11, (12, 16), 4)
    img[:, :, :3] = 0.5
    actual = native.dpid(img, (8, 6), lam)
    assert np.all(actual[:, :, :3] == 0.5)
    assert_close(actual, oracle(img, (8, 6), lam))


def test_separate_alpha_is_a_colour_dpid_beside_an_alpha_dpid():
    img = noise(13, (37, 23), 4)
    for lam in LAMBDAS:
        rgba = native.dpid(img, (10, 7), lam, True)
        rgb = native.dpid(np.ascontiguousarray(img[:, :, :3]), (10, 7), lam)
        alpha = native.dpid(np.ascontiguousarray(img[:, :, 3]), (10, 7), lam)
        np.testing.assert_array_equal(rgba, np.concatenate([rgb, alpha], axis=2))


@pytest.mark.parametrize("lam", [0.0, 1.0, 2.5])
@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_gamma_correction_wraps_the_kernel(channels, separate, lam):
    # As cn_resample_filtered: Gamma 2.2 (alpha kept) before, 1 / 2.2 after.
    img = noise(17, (29, 31), channels)
    linear = native_alpha_gamma.fast_gamma(img, 2.2)
    expected = oracle(linear, (12, 9), lam, separate).astype(np.float32)
    expected = native_alpha_gamma.fast_gamma(expected, 1 / 2.2)
    actual = native.dpid(img, (12, 9), lam, separate, True)
    assert_close(actual, expected.astype(np.float64))


@pytest.mark.parametrize("gamma", [False, True])
@pytest.mark.parametrize("separate", [False, True])
def test_resize_premultiplies_rgba_unless_separate(separate, gamma):
    img = noise(19, (33, 27), 4)
    actual = resize(img, (9, 14), ResizeFilter.DPID, separate, gamma, 1.5)
    if separate:
        expected = native.dpid(img, (9, 14), 1.5, True, gamma)
    else:
        premultiplied = native.dpid(native.premultiply(img), (9, 14), 1.5, False, gamma)
        expected = native.finish_alpha_inplace(premultiplied)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_resize_passes_lambda_and_channels(channels):
    img = noise(23, (21, 19), channels)
    for lam in LAMBDAS:
        actual = resize(img, (8, 5), ResizeFilter.DPID, True, dpid_lambda=lam)
        np.testing.assert_array_equal(actual, native.dpid(img, (8, 5), lam, True))
    # The default lambda is 1.
    np.testing.assert_array_equal(
        resize(img, (8, 5), ResizeFilter.DPID, True),
        native.dpid(img, (8, 5), 1.0, True),
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_same_size_is_a_copy(channels):
    img = noise(29, (6, 9), channels)
    img.setflags(write=False)
    out = resize(img, (9, 6), ResizeFilter.DPID, dpid_lambda=2.0)
    np.testing.assert_array_equal(out, img)
    assert not np.shares_memory(out, img)
    assert out.flags.writeable


def test_auto_never_picks_dpid():
    img = noise(31, (16, 16), 3)
    np.testing.assert_array_equal(
        resize(img, (5, 5), ResizeFilter.AUTO), resize(img, (5, 5), ResizeFilter.BOX)
    )
    np.testing.assert_array_equal(
        resize(img, (40, 40), ResizeFilter.AUTO),
        resize(img, (40, 40), ResizeFilter.LANCZOS),
    )


def test_strided_input_and_nonfinite_values():
    img = noise(37, (20, 18), 4)[:, ::2, :]
    np.testing.assert_array_equal(
        native.dpid(img, (4, 5), 1.0),
        native.dpid(np.ascontiguousarray(img), (4, 5), 1.0),
    )
    # A NaN reaches only the outputs whose footprint or tent reads its guidance.
    img = noise(41, (16, 16), 3)
    img[0, 0, 1] = np.nan
    out = native.dpid(img, (8, 8), 1.0)
    reached = np.zeros((8, 8, 3), bool)
    reached[:2, :2] = True
    assert np.isnan(out[reached]).all()
    assert np.isfinite(out[~reached]).all()


@pytest.mark.parametrize("lam", [-0.5, math.nan, math.inf])
def test_rejects_bad_lambda(lam):
    with pytest.raises(ValueError, match="lambda"):
        native.dpid(noise(43, (4, 4), 3), (2, 2), lam)


def test_raw_c_boundaries():
    dll = native._api()
    source = np.zeros((4, 4, 3), np.float32)
    out = np.zeros((2, 2, 3), np.float32)
    good: list[object] = [ptr(source), ptr(out), 4, 4, 3, 2, 2, 1.0, 0, 0]
    assert dll.cn_resample_dpid(*good) == 0
    for index, value in [
        (0, None),
        (1, None),
        (2, 0),
        (3, 0),
        (4, 0),
        (4, 5),
        (5, 0),
        (6, 0),
        (7, -1.0),
        (7, math.nan),
        (7, math.inf),
        (8, 2),
        (9, -1),
    ]:
        args = list(good)
        args[index] = value
        assert dll.cn_resample_dpid(*args) == 1, (index, value)
    args = list(good)
    args[2] = 2**31
    assert dll.cn_resample_dpid(*args) == 2
    args = list(good)
    args[5] = 2**32
    assert dll.cn_resample_dpid(*args) == 2


def test_dimensions_before_dispatch():
    img = noise(47, (4, 4), 3)
    assert native.dpid(img, (0, 3), 1.0).shape == (3, 0, 3)
    with pytest.raises(OverflowError):
        native.dpid(img, (-1, 3), 1.0)
    with pytest.raises(OverflowError):
        native.dpid(img, (2**32, 1), 1.0)
    with pytest.raises(ValueError, match="16 GiB"):
        native.dpid(img, (2**31 - 1, 3), 1.0)
    with pytest.raises(TypeError):
        native.dpid(img.astype(np.float64), (2, 2), 1.0)
