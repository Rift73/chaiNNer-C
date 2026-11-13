"""Blend Images' Color and Luminosity: the non-separable modes of ISO 32000-2 11.3.5.3 and the W3C compositing spec.

The reference is the specs' Lum, SetLum and ClipColor with NumPy in float64, the alpha rules of Blend Images' other modes
in float32, in the kernel's operation order, so every comparison is of bits. Planted truths are worked by hand from the
specs' formulas. Images are BGR(A), so the weights run 0.11, 0.59, 0.3.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from nodes.impl.blend import (
    BlendMode,
    blend_images,
)
from nodes.impl.native import lib, ptr
from nodes.impl.native_blend import raw_blend
from nodes.impl.native_composite import canvas

MODES = [BlendMode.COLOR, BlendMode.LUMINOSITY]
PAIRS = list(itertools.product([1, 3, 4], repeat=2))
# sha256 of the fixed patterns' outputs over every channel pair, frozen from the NumPy reference below.
FROZEN = {
    BlendMode.COLOR: "062ce8f088a5aee65b447bbe462a36bdfb79327a1b21357346674f65713d1876",
    BlendMode.LUMINOSITY: "27d11255b7db83d1f75322887e3191a3dbd530c2a693f2248bf1787aefda44c2",
}


def lum(c: np.ndarray) -> np.ndarray:
    return (0.11 * c[..., 0:1]) + (0.59 * c[..., 1:2]) + (0.3 * c[..., 2:3])


def set_lum(color: np.ndarray, source: np.ndarray) -> np.ndarray:
    """SetLum(color, Lum(source)) with ClipColor, in float64, as float32."""
    c = color.astype(np.float64)
    c = c + (lum(source.astype(np.float64)) - lum(c))
    l = lum(c)
    low = c.min(axis=-1, keepdims=True)
    high = c.max(axis=-1, keepdims=True)
    with np.errstate(all="ignore"):
        c = np.where((low < 0) & (l > low), l + (((c - l) * l) / (l - low)), c)
        c = np.where(
            (high > 1) & (high > l), l + (((c - l) * (1.0 - l)) / (high - l)), c
        )
    return c.astype(np.float32)


def bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=2)
    return image[:, :, :3] if image.shape[2] == 4 else image


def channels_of(image: np.ndarray) -> int:
    return 1 if image.ndim == 2 else image.shape[2]


def expected(overlay: np.ndarray, base: np.ndarray, mode: BlendMode) -> np.ndarray:
    oc, bc = channels_of(overlay), channels_of(base)
    target = max(oc, bc)
    if target == 1:
        return np.clip(base if mode == BlendMode.COLOR else overlay, 0, 1)
    o, b = bgr(overlay), bgr(base)
    blended = set_lum(o, b) if mode == BlendMode.COLOR else set_lum(b, o)
    if target == 3:
        return np.clip(blended, 0, 1)
    if bc < 4:
        alpha = overlay[:, :, 3:]
        rgb = np.clip((alpha * blended) + ((1 - alpha) * b), 0, 1)
        return np.concatenate([rgb, np.ones_like(alpha)], axis=2)
    o_a = overlay[:, :, 3:] if oc == 4 else np.ones_like(base[:, :, 3:])
    b_a = base[:, :, 3:]
    final_a = 1 - ((1 - o_a) * (1 - b_a))
    strength = o_a * b_a
    value = (((o_a - strength) * o) + ((b_a - strength) * b)) + (strength * blended)
    rgb = np.clip(value / np.maximum(final_a, 0.0001), 0, 1)
    return np.concatenate([rgb, np.clip(final_a, 0, 1)], axis=2)


def assert_bits(actual: np.ndarray, reference: np.ndarray) -> None:
    assert actual.shape == reference.shape
    assert actual.dtype == reference.dtype == np.float32
    nan = np.isnan(actual)
    np.testing.assert_array_equal(nan, np.isnan(reference))
    np.testing.assert_array_equal(
        actual.view(np.uint32)[~nan], reference.view(np.uint32)[~nan]
    )


def image(rng, channels: int, height: int = 7, width: int = 11) -> np.ndarray:
    shape = (height, width) if channels == 1 else (height, width, channels)
    return rng.random(shape, dtype=np.float32)


def pattern(channels: int, seed: int) -> np.ndarray:
    """A fixed image without a generator's version in it: multiples of an odd step, wrapped to 0..1."""
    shape = (9, 13) if channels == 1 else (9, 13, channels)
    values = (np.arange(np.prod(shape), dtype=np.int64) * (2 * seed + 1) * 2861) % 8191
    return (values / 8190).reshape(shape).astype(np.float32)


def px(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32).reshape(1, 1, len(values))


def test_pure_red_takes_a_mid_grey_base_luminosity():
    # Lum(red) = 0.3, so SetLum adds 0.2: R 1.2, G = B = 0.2. Lum stays 0.5, and ClipColor scales toward it by 0.5 / 0.7.
    out = blend_images(px(0, 0, 1), px(0.5, 0.5, 0.5), BlendMode.COLOR)
    np.testing.assert_allclose(out[0, 0], [2 / 7, 2 / 7, 1], atol=1e-6)


def test_pure_green_takes_a_dark_base_luminosity():
    # Lum(green) = 0.59, so SetLum subtracts 0.49: B = R = -0.49, G 0.51; ClipColor scales toward 0.1 by 0.1 / 0.59.
    out = blend_images(px(0, 1, 0), px(0.1, 0.1, 0.1), BlendMode.COLOR)
    np.testing.assert_allclose(out[0, 0], [0, 10 / 59, 0], atol=1e-6)


def test_blue_base_takes_a_mid_grey_overlay_luminosity():
    # Lum(blue) = 0.11, so SetLum adds 0.39: B 1.39, G = R = 0.39; ClipColor scales toward 0.5 by 0.5 / 0.89.
    out = blend_images(px(0.5, 0.5, 0.5), px(1, 0, 0), BlendMode.LUMINOSITY)
    np.testing.assert_allclose(out[0, 0], [1, 39 / 89, 39 / 89], atol=1e-6)


def test_no_clip_needed_keeps_the_differences():
    # Lum(overlay) = 0.11 * 0.2 + 0.59 * 0.3 + 0.3 * 0.4 = 0.319, the base's 0.5: every channel gains 0.181.
    out = blend_images(px(0.2, 0.3, 0.4), px(0.5, 0.5, 0.5), BlendMode.COLOR)
    np.testing.assert_allclose(out[0, 0], [0.381, 0.481, 0.581], atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_grey_images_have_only_luminosity(mode):
    rng = np.random.default_rng(11)
    overlay, base = image(rng, 1), image(rng, 1)
    assert_bits(
        blend_images(overlay, base, mode),
        base if mode == BlendMode.COLOR else overlay,
    )
    # The same values as three equal channels come out the same, exactly: no weight's rounding shows.
    wanted = base if mode == BlendMode.COLOR else overlay
    out = blend_images(np.dstack([overlay] * 3), np.dstack([base] * 3), mode)
    assert_bits(out, np.dstack([wanted] * 3))


@pytest.mark.parametrize("mode", MODES)
def test_luminosity_lands_on_the_right_layer(mode):
    rng = np.random.default_rng(23)
    overlay, base = image(rng, 3, 64, 64), image(rng, 3, 64, 64)
    out = blend_images(overlay, base, mode)
    assert out.min() >= 0 and out.max() <= 1
    wanted = lum((base if mode == BlendMode.COLOR else overlay).astype(np.float64))
    np.testing.assert_allclose(lum(out.astype(np.float64)), wanted, atol=2e-6)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("channels", PAIRS)
@pytest.mark.parametrize("size", [(1, 1), (1, 9), (9, 1), (8, 8), (300, 300)])
def test_every_channel_pair_matches_the_reference(mode, channels, size):
    # 300 x 300 crosses the kernel's 65,536-pixel parallel chunk.
    rng = np.random.default_rng(37)
    overlay, base = image(rng, channels[0], *size), image(rng, channels[1], *size)
    before = overlay.copy(), base.copy()
    assert_bits(blend_images(overlay, base, mode), expected(overlay, base, mode))
    np.testing.assert_array_equal(overlay, before[0])
    np.testing.assert_array_equal(base, before[1])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("channels", PAIRS)
def test_alpha_edges_and_channel_limits(mode, channels):
    levels = np.array([0, 0.0001, 0.5, 0.9999, 1], dtype=np.float32)
    a, b = np.meshgrid(levels, levels)
    overlay = a if channels[0] == 1 else np.dstack(
        [np.roll(a, i, axis=0) for i in range(channels[0])]
    )
    base = b if channels[1] == 1 else np.dstack(
        [np.roll(b, i, axis=1) for i in range(channels[1])]
    )
    assert_bits(blend_images(overlay, base, mode), expected(overlay, base, mode))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("channels", PAIRS)
def test_values_outside_the_range_and_nan(mode, channels):
    levels = np.array([-0.0, 0.25, 1.0, -0.5, 1.5, np.nan], dtype=np.float32)
    a, b = np.meshgrid(levels, levels)
    overlay = a if channels[0] == 1 else np.dstack(
        [np.roll(a, i, axis=0) for i in range(channels[0])]
    )
    base = b if channels[1] == 1 else np.dstack(
        [np.roll(b, i, axis=1) for i in range(channels[1])]
    )
    with np.errstate(all="ignore"):
        reference = expected(overlay, base, mode)
    assert_bits(blend_images(overlay, base, mode), reference)


@pytest.mark.parametrize("mode", MODES)
def test_infinities_stay_inside_the_clip(mode):
    levels = np.array([0.0, 0.5, 1.0, np.inf, -np.inf], dtype=np.float32)
    a, b = np.meshgrid(levels, levels)
    out = blend_images(np.dstack([a, b, a]), np.dstack([b, a, a]), mode)
    finite = out[~np.isnan(out)]
    assert finite.min() >= 0 and finite.max() <= 1


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("channels", PAIRS)
def test_canvas_blends_the_overlap_of_a_placed_overlay(mode, channels):
    rng = np.random.default_rng(41)
    base, overlay = image(rng, channels[0], 8, 10), image(rng, channels[1], 8, 10)
    out = canvas(base, overlay, mode.value, 0, 0, True)
    assert_bits(out, expected(overlay, base, mode))
    # Moved 3 across with the crop: the columns the overlay covers are its blend.
    moved = canvas(base, overlay, mode.value, 3, 0, True)
    assert_bits(
        moved[:, 3:], expected(overlay[:, :7], base[:, 3:], mode).reshape(moved[:, 3:].shape)
    )


@pytest.mark.parametrize("mode", MODES)
def test_element_entry_refuses_a_mode_that_needs_whole_pixels(mode):
    a = np.ones((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Invalid image dimensions or arguments"):
        raw_blend(a, a, mode.value)


def test_mode_past_the_last_is_refused():
    a = np.ones(4, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    assert lib().cn_blend_images(ptr(a), ptr(a), ptr(out), 1, 4, 4, 25) == 1
    np.testing.assert_array_equal(out, 0)


def digest(mode: BlendMode) -> str:
    """The frozen outputs' hash: every channel pair over the fixed pattern."""
    h = hashlib.sha256()
    for oc, bc in PAIRS:
        h.update(blend_images(pattern(oc, 1), pattern(bc, 2), mode).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("mode", MODES)
def test_golden_output_and_reference_agree(mode):
    h = hashlib.sha256()
    for oc, bc in PAIRS:
        h.update(expected(pattern(oc, 1), pattern(bc, 2), mode).tobytes())
    assert h.hexdigest() == FROZEN[mode]
    assert digest(mode) == FROZEN[mode]


ISA_CHILD = """
import json
import sys

sys.path.insert(0, sys.argv[1])
import test_blend_non_separable as t
from nodes.impl import native

print(json.dumps({"line": native.isa_line(), "digests": [t.digest(m) for m in t.MODES]}))
"""


def test_identical_bits_at_every_isa_level():
    results = {}
    for level in ("scalar", "avx2", "avx512"):
        environment = dict(
            os.environ, PYTHONPATH=os.pathsep.join(sys.path), CHAINNER_C_ISA=level
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", ISA_CHILD, os.path.dirname(__file__)],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        results[level] = json.loads(completed.stdout.splitlines()[-1])
        assert f"requested {level}" in results[level]["line"]
    wanted = [FROZEN[mode] for mode in MODES]
    for level in results:
        assert results[level]["digests"] == wanted, level
