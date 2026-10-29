"""Descale (native/src/descale_ops.c over third_party/descale): it inverts a bilinear upscale,
and layouts, chunking and the solver cache do not change results. Bit-identity with the
VapourSynth plugin (Release 12) was checked at integration: 12 of 12 cases, at the running
ISA level only (the AVX2 unit uses FMA, exempt from cross-level identity, ARCHITECTURE
section 5); across levels the outputs agree within a tolerance. Past one reflection, where
the plugin indexed outside its weights, the Mirror border repeats the reflection
(third_party/descale/PATCHES): upscale_matrix below is that upscale in float64. Parameters
that leave a target pixel without weight (black) or give non-finite weights are refused."""

import itertools
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from bounded import run_python
from test_isa_dispatch import isa_get, isa_set

from nodes.impl import native_descale

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"

ARGS = (3, 0.0, 0.5, 1.0)


def run(image, width, height, mode=1, border=2):
    return native_descale.descale(
        image, width, height, mode, *ARGS, (0.0, 0.0), (width, height), border
    )


# Each kernel's support in Blur units (Lanczos: its Taps).
SUPPORT = {1: 1, 2: 2, 4: 2, 5: 3, 6: 4, 7: 0}
# The splines' cubic pieces on [n, n + 1), coefficients of 1, t, t^2, t^3 for t = x - n.
SPLINES = {
    4: [(1, -1 / 5, -9 / 5, 1), (0, -7 / 15, 4 / 5, -1 / 3)],
    5: [
        (1, -3 / 209, -453 / 209, 13 / 11),
        (0, -156 / 209, 270 / 209, -6 / 11),
        (0, 26 / 209, -45 / 209, 1 / 11),
    ],
    6: [
        (1, -3 / 2911, -6387 / 2911, 49 / 41),
        (0, -2328 / 2911, 4032 / 2911, -24 / 41),
        (0, 582 / 2911, -1008 / 2911, 6 / 41),
        (0, -97 / 2911, 168 / 2911, -1 / 41),
    ],
}


def kernel(mode, taps, b, c, x):
    """descale.c's calculate_weight at x = |distance| / blur (an array), in its operation
    order, so a weight is exactly zero where the plugin's is."""
    square, cube = x * x, x * x * x
    if mode == 1:
        return np.maximum(1.0 - x, 0.0)
    if mode == 2:
        near = (
            (12 - 9 * b - 6 * c) * cube + (-18 + 12 * b + 6 * c) * square + (6 - 2 * b)
        )
        far = (-b - 6 * c) * cube + (6 * b + 30 * c) * square + (-12 * b - 48 * c) * x
        return np.select(
            [x < 1, x < 2], [near / 6.0, (far + (8 * b + 24 * c)) / 6.0], 0.0
        )
    if mode == 3:
        return np.where(x < taps, np.sinc(x) * np.sinc(x / taps), 0.0)
    if mode == 7:
        return np.ones_like(x)
    pieces = np.array([*SPLINES[mode], (0, 0, 0, 0)])
    n = np.minimum(np.floor(x), len(pieces) - 1)
    k = pieces[n.astype(int)]
    t = x - n
    return k[:, 0] + k[:, 1] * t + k[:, 2] * (t * t) + k[:, 3] * (t * t * t)


def upscale_matrix(small, big, mode, taps, b, c, blur, shift, active, border):
    """descale.c's scaling_weights in float64: the big x small upscale the solve inverts, its
    Mirror border folding a tap's pixel with period 2 * small (third_party/descale/PATCHES)."""
    support = math.ceil(SUPPORT.get(mode, taps) * blur)
    size = 2 * support if support > 0 else 1
    matrix = np.zeros((big, small))
    for i in range(big):
        pos = (i + 0.5) / (big / active) + shift
        start = pos - size / 2
        x = (
            math.floor(start + (0.5 if start < 0 else 0.49999999999999994))
            + 0.5
            + np.arange(size)
        )
        weights = kernel(mode, taps, b, c, np.abs(x - pos) / blur)
        total = weights.sum() or np.finfo(np.float64).eps
        pixel = np.floor(x).astype(np.int64)
        if border == 0:
            pixel %= 2 * small
            pixel = np.where(pixel < small, pixel, 2 * small - 1 - pixel)
        elif border == 1:
            weights = np.where((pixel >= 0) & (pixel < small), weights, 0.0)
        np.add.at(matrix[i], np.clip(pixel, 0, small - 1), weights / total)
    return matrix


def test_inverts_bilinear_upscale():
    rng = np.random.default_rng(3)
    small = rng.random((45, 67, 3), dtype=np.float32)
    large = cv2.resize(small, (100, 70), interpolation=cv2.INTER_LINEAR)
    # border 2 (extend) is OpenCV's replicate padding.
    assert np.abs(run(large, 67, 45) - small).max() < 1e-4


@pytest.mark.parametrize(
    "shape, border", [((5, 3, 1), 0), ((9, 17, 4), 0), ((130, 75, 3), 2)]
)
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


# (source, target, kernel, Taps, Blur, Source Left): taps more than the target's size past an edge
OUT_OF_REACH = [
    (400, 2, 3, 3, 1.0, 0.0),
    (400, 6, 3, 3, 1.0, 0.0),
    (3, 2, 5, 3, 1.0, 0.0),
    (400, 2, 3, 3, 2.0, 0.0),
    (400, 4, 6, 3, 1.0, 0.0),
    (400, 100, 1, 3, 1.0, 1000.0),
    (400, 100, 1, 3, 1.0, -1000.0),
    (400, 100, 2, 3, 1.0, -100.0),
]


@pytest.mark.parametrize("big, small, mode, taps, blur, left", OUT_OF_REACH)
def test_mirror_past_one_reflection_inverts_its_upscale(
    big, small, mode, taps, blur, left
):
    # descale's Mirror reflected a tap once, so these indexed outside the weights (a crash
    # at Source Left -1000); repeated, the solve inverts the upscale they describe, at
    # every ISA level.
    matrix = upscale_matrix(small, big, mode, taps, 1 / 3, 1 / 3, blur, left, small, 0)
    expected = np.random.default_rng(6).random((16, small, 3))
    image = np.einsum("ij,rjc->ric", matrix, expected).astype(np.float32)
    effective, _, maximum = isa_get()
    outputs = []
    try:
        for level in range(maximum + 1):
            assert isa_set(level) == level
            outputs.append(
                native_descale.descale(
                    image,
                    small,
                    16,
                    mode,
                    taps,
                    1 / 3,
                    1 / 3,
                    blur,
                    (left, 0.0),
                    (small, 16),
                    0,
                )
            )
    finally:
        assert isa_set(effective) == effective
    for out in outputs:
        assert np.isfinite(out).all()
        assert np.abs(out - expected).max() <= 1e-3
        assert np.abs(out - outputs[0]).max() <= 2e-6


@pytest.mark.parametrize("border", [0, 1, 2])
@pytest.mark.parametrize("mode", [1, 2, 3, 4, 5, 6, 7])
def test_upscale_matrix_is_the_solved_upscale(mode, border):
    # The port above against the solver on every kernel and border, within reach.
    matrix = upscale_matrix(67, 100, mode, 3, 0.0, 0.5, 1.0, 0.25, 67, border)
    expected = np.random.default_rng(7).random((3, 67))
    image = (expected @ matrix.T).astype(np.float32)
    out = native_descale.descale(
        image, 67, 3, mode, 3, 0.0, 0.5, 1.0, (0.25, 0.0), (67, 3), border
    )
    assert np.abs(out - expected).max() <= 1e-3


# _core(16, 8, mode, taps, ...) on every border, in a child: the cases below crashed it.
REFUSED_CHILD = """
import sys
from nodes.impl import native_descale
mode, taps = int(sys.argv[1]), int(sys.argv[2])
blur, shift, active = (float(value) for value in sys.argv[3:6])
for border in (0, 1, 2):
    try:
        native_descale._core(16, 8, mode, taps, 0.0, 0.5, blur, shift, active, border)
    except ValueError as error:
        assert str(error) == "Invalid descale parameters", error
    else:
        sys.exit(f"border {border} accepted")
"""


@pytest.mark.parametrize(
    "mode, taps, blur, shift, active",
    [
        (1, 3, "nan", "0", "8"),
        (1, 3, "inf", "0", "8"),
        (1, 3, "1", "nan", "8"),
        (1, 3, "1", "inf", "8"),
        (1, 3, "1", "0", "inf"),
        (1, 3, "1", "1e308", "1e308"),
        (1, 3, "1e9", "0", "8"),
        (3, 32, "1e10", "0", "8"),
    ],
)
def test_unrepresentable_parameters_are_refused(mode, taps, blur, shift, active):
    # A NaN or infinite shift made every tap position NaN, whose pixel index read and
    # wrote outside the weights; so did a finite shift and window whose positions
    # overflow to infinity (1e308 each). An infinite Blur or window, or Blur x Taps past
    # what descale's int support and bandwidth hold, overflowed them.
    args = [str(mode), str(taps), blur, shift, active]
    done = run_python(["-c", REFUSED_CHILD, *args], BACKEND, 120)
    assert done.returncode == 0, done.output


# Every kernel (Lanczos at four Taps) x Blur x ratio x border; (source width for a target
# of 24, Source Left): the same width with a half-pixel and a whole-pixel shift, 1.5, 2
# and 3.
SWEEP_KERNELS = [
    (1, 3),
    (2, 3),
    (3, 1),
    (3, 3),
    (3, 8),
    (3, 32),
    (4, 3),
    (5, 3),
    (6, 3),
    (7, 3),
]
SWEEP_BLURS = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.6, 0.75, 1.0, 2.0, 10.0]
SWEEP_SOURCES = [(24, 0.5), (24, 1.0), (36, 0.0), (48, 0.0), (72, 0.0)]


def unweighted(matrix):
    """Whether a target pixel takes no weight in the solver's float32 weights, counting
    those below 1e-12 as none: Lanczos's sinc at a whole-number distance leaves ~1e-17."""
    return not (np.abs(matrix.astype(np.float32)) >= 1e-12).any(axis=0).all()


@pytest.mark.parametrize("mode, taps", SWEEP_KERNELS)
def test_unweighted_target_pixels_are_refused_exactly(mode, taps):
    # descale solves a target pixel no source pixel weighs to 0 (Blur 0.01 with Bilinear
    # gave a black image): exactly those parameter sets are refused, named by whether
    # Blur 1 would weigh every pixel; every other one gives a finite result.
    rng = np.random.default_rng(9)
    sweep = itertools.product(SWEEP_BLURS, SWEEP_SOURCES, (0, 1, 2))
    for blur, (big, left), border in sweep:
        args = (mode, taps, 0.0, 0.5)
        image = rng.random((2, big), dtype=np.float32)
        window = ((left, 0.0), (24, 2), border)
        if not unweighted(upscale_matrix(24, big, *args, blur, left, 24, border)):
            out = native_descale.descale(image, 24, 2, *args, blur, *window)
            assert np.isfinite(out).all()
            continue
        if unweighted(upscale_matrix(24, big, *args, 1.0, left, 24, border)):
            message = (
                r"^Descale: with this source window .* \(the result would be black\)"
            )
        else:
            message = rf"^Descale: Blur {blur:g} is too narrow for the "
        with pytest.raises(ValueError, match=message):
            native_descale.descale(image, 24, 2, *args, blur, *window)


def test_refusal_messages():
    image = np.random.default_rng(10).random((8, 1920, 3), dtype=np.float32)
    with pytest.raises(ValueError) as error:
        native_descale.descale(
            image, 1280, 8, 1, 3, 0.0, 0.5, 0.01, (0.0, 0.0), (1280, 8), 0
        )
    assert str(error.value) == (
        "Descale: Blur 0.01 is too narrow for the Bilinear kernel at this size: some "
        "target pixels get no weight (the result would be black). Use a larger Blur "
        "(1 is the kernel's own width)."
    )
    # Point ignores Blur: a half-pixel Source Left at the same width leaves the first
    # column without weight.
    strip = np.ascontiguousarray(image[:, :24])
    with pytest.raises(ValueError) as error:
        native_descale.descale(
            strip, 24, 8, 7, 3, 0.0, 0.5, 1.0, (0.5, 0.0), (24, 8), 2
        )
    assert str(error.value) == (
        "Descale: with this source window (Source Left, Top, Width, Height) some target "
        "pixels get no weight from the Point kernel (the result would be black). Use a "
        "window that covers the target."
    )
    with pytest.raises(ValueError) as error:
        native_descale.descale(
            image, 1280, 8, 2, 3, np.inf, 0.5, 1.0, (0.0, 0.0), (1280, 8), 0
        )
    assert str(error.value) == (
        "Descale: the Bicubic kernel's weights are not finite with these parameters "
        "(the result would not be finite either)."
    )


def test_system_beyond_int_indices_is_refused():
    # descale sizes its src_dim x dst_dim matrices with int products: 65536 x 32768
    # overflowed into an access violation inside the core build.
    strip = np.zeros((16, 65536), np.float32)
    with pytest.raises(ValueError, match="^Invalid descale parameters$"):
        run(strip, 32768, 16)
    # The float weights take ceil(dst / 8) * 8 rows of up to src columns.
    with pytest.raises(ValueError, match="^Invalid descale parameters$"):
        native_descale._core(300_000_000, 1, 1, 3, 0.0, 0.5, 1.0, 0.0, 1.0, 2)


def test_column_pass_beyond_int_indices_is_refused():
    # descale's column pass indexes the middle plane (h rows of a 16-float-aligned stride)
    # with int products; 2^20 rows of 2049 columns would overflow them.
    core_v = native_descale._core(1 << 20, 1, 1, 3, 0.0, 0.5, 1.0, 0.0, 1.0, 2)
    workspace = native_descale._lib.cn_descale_workspace
    assert workspace(1 << 20, 2032, 1, None, core_v.handle) > 0
    assert workspace(1 << 20, 2049, 1, None, core_v.handle) == 0


@pytest.mark.parametrize(
    "mode, b, c", [(1, 0.0, 0.5), (2, 1 / 3, 1 / 3), (3, 0.0, 0.5)]
)
def test_levels_agree_within_the_fma_exemption(mode, b, c):
    # The AVX2 unit uses FMA (exempt from cross-level identity, ARCHITECTURE section 5):
    # its 1080p -> 720p output differs from the scalar unit's by up to ~8e-7.
    image = np.random.default_rng(8).random((1080, 1920, 3), dtype=np.float32)
    effective, _, maximum = isa_get()
    outputs = {}
    try:
        for level in range(maximum + 1):
            assert isa_set(level) == level
            outputs[level] = native_descale.descale(
                image, 1280, 720, mode, 3, b, c, 1.0, (0.0, 0.0), (1280, 720), 0
            )
    finally:
        assert isa_set(effective) == effective
    for level in range(1, maximum + 1):
        assert np.abs(outputs[level] - outputs[0]).max() <= 2e-6, level


def test_same_size_is_identity_and_bad_targets_fail():
    image = np.random.default_rng(5).random((20, 30, 3), dtype=np.float32)
    assert np.array_equal(run(image, 30, 20), image)
    with pytest.raises(ValueError):
        run(image, 31, 20)
