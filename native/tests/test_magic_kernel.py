"""Resize's Magic Kernel Sharp filters (John Costella's kernels, 2013 and 2021).

The reference is the published construction, not the kernel's piecewise form: the Magic Kernel m (the quadratic B-spline)
convolved with the discrete sharpening step, in exact fractions. Taking the kernels apart this way also gives the two
properties that define them: a downscale is m at the output's scale followed by the sharpening step on the output's
pixels, and an upscale is the sharpening step on the source's pixels followed by m.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from nodes.impl.resize import ResizeFilter, resize

KERNELS = [ResizeFilter.MKS2013, ResizeFilter.MKS2021]
# The sharpening steps, from the centre tap outward.
SHARPEN = {
    ResizeFilter.MKS2013: (Fraction(3, 2), Fraction(-1, 4)),
    ResizeFilter.MKS2021: (
        Fraction(204, 144),
        Fraction(-35, 144),
        Fraction(6, 144),
        Fraction(-1, 144),
    ),
}
SUPPORT = {ResizeFilter.MKS2013: 2.5, ResizeFilter.MKS2021: 4.5}
# sha256 of the fixed pattern's outputs (see frozen_outputs), checked against the construction by the tests below.
FROZEN = {
    ResizeFilter.MKS2013: "62199fc11ae9a84dd3fc002eea68fa0040734eb36ad2c57cdd16ff34c3a1c6cb",
    ResizeFilter.MKS2021: "6188d6974d74194f06cdba6bfa7ea9f2c553fff87974c41ab3ee1a6bff05bdbe",
}


def magic(x: Fraction) -> Fraction:
    x = abs(x)
    if x <= Fraction(1, 2):
        return Fraction(3, 4) - x * x
    if x <= Fraction(3, 2):
        return (x - Fraction(3, 2)) ** 2 / 2
    return Fraction(0)


def sharp(filter_: ResizeFilter, x: Fraction) -> Fraction:
    taps = SHARPEN[filter_]
    total = taps[0] * magic(x)
    for k, tap in enumerate(taps[1:], start=1):
        total += tap * (magic(x - k) + magic(x + k))
    return total


def tapped(values: np.ndarray, taps: tuple[Fraction, ...], axis: int) -> np.ndarray:
    """values convolved along an axis with a symmetric step, the first len(taps) - 1 and last lines dropped."""
    weights = [float(t) for t in taps]
    reach = len(weights) - 1
    total = weights[0] * np.take(values, range(reach, values.shape[axis] - reach), axis)
    for k, weight in enumerate(weights[1:], start=1):
        left = np.take(values, range(reach - k, values.shape[axis] - reach - k), axis)
        right = np.take(values, range(reach + k, values.shape[axis] - reach + k), axis)
        total = total + weight * (left + right)
    return total


def test_the_published_kernels_at_known_points():
    # Costella's piecewise forms, at their centres and joins.
    assert sharp(ResizeFilter.MKS2013, Fraction(0)) == Fraction(17, 16)
    assert sharp(ResizeFilter.MKS2013, Fraction(1)) == 0
    assert sharp(ResizeFilter.MKS2013, Fraction(2)) == Fraction(-1, 32)
    assert sharp(ResizeFilter.MKS2013, Fraction(5, 2)) == 0
    assert sharp(ResizeFilter.MKS2021, Fraction(0)) == Fraction(577, 576)
    assert sharp(ResizeFilter.MKS2021, Fraction(9, 2)) == 0


@pytest.mark.parametrize("filter_", KERNELS)
def test_impulse_response_is_the_kernel(filter_):
    # On a grey line, so the negative lobes stay inside the clip. The kernels sum to 1 on any grid, so the plan's
    # normalization leaves their values alone.
    scale = 8
    width = 21
    line = np.full((1, width), 0.5, np.float32)
    line[0, width // 2] = 0.75
    out = resize(line, (width * scale, 1), filter_)
    for x in range(width * scale):
        position = (Fraction(2 * x + 1, 2 * scale) - Fraction(1, 2)) - width // 2
        wanted = 0.5 + float(sharp(filter_, position)) / 4
        assert out[0, x, 0] == pytest.approx(wanted, abs=2e-6), x


@pytest.mark.parametrize("filter_", KERNELS)
@pytest.mark.parametrize("factor", [2, 3])
def test_downscale_is_magic_then_the_sharpening_step(filter_, factor):
    rng = np.random.default_rng(5)
    size = 96
    source = (0.4 + 0.2 * rng.random((size, 1))).astype(np.float64)
    out = resize(source.astype(np.float32), (1, size // factor), filter_)[:, :, 0]
    # m at the output's scale, over the source's pixels: normalized as the plan normalizes, m sums to 1 on the grid.
    magic_out = np.zeros(size // factor)
    for y in range(size // factor):
        centre = Fraction(2 * y + 1, 2) * factor - Fraction(1, 2)
        weights = [magic((Fraction(i) - centre) / factor) for i in range(size)]
        magic_out[y] = sum(float(w) * source[i, 0] for i, w in enumerate(weights)) / float(
            sum(weights)
        )
    taps = SHARPEN[filter_]
    margin = 2 * len(taps)
    wanted = tapped(magic_out, taps, 0)
    np.testing.assert_allclose(
        out[len(taps) - 1 :][margin : -margin - len(taps) + 1, 0],
        wanted[margin:-margin],
        atol=4e-6,
    )


@pytest.mark.parametrize("filter_", KERNELS)
def test_upscale_is_the_sharpening_step_then_magic(filter_):
    rng = np.random.default_rng(6)
    size, factor = 48, 3
    source = (0.4 + 0.2 * rng.random((size, 1))).astype(np.float64)
    out = resize(source.astype(np.float32), (1, size * factor), filter_)[:, :, 0]
    taps = SHARPEN[filter_]
    reach = len(taps) - 1
    sharpened = tapped(source, taps, 0)[:, 0]
    # The sharpened source's magic reconstruction, away from the edges where the sharpened lines run out.
    for y in range(3 * factor * reach, (size - 3 * reach) * factor):
        centre = Fraction(2 * y + 1, 2 * factor) - Fraction(1, 2)
        wanted = sum(
            float(magic(Fraction(i + reach) - centre)) * value
            for i, value in enumerate(sharpened)
        )
        assert out[y, 0] == pytest.approx(wanted, abs=4e-6), y


@pytest.mark.parametrize("filter_", KERNELS)
@pytest.mark.parametrize(
    "shape,target",
    [
        ((1, 1), (1, 1)),
        ((1, 1), (5, 3)),
        ((2, 2), (1, 1)),
        ((3, 1), (1, 7)),
        ((1, 9), (2, 1)),
        ((64, 48), (16, 12)),
        ((5, 7), (40, 56)),
    ],
)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_a_flat_image_stays_flat_at_every_size(filter_, shape, target, channels):
    image = np.full((*shape, channels), 0.25, np.float32)
    out = resize(image, (target[1], target[0]), filter_)
    assert out.shape == (*target, channels)
    np.testing.assert_allclose(out, 0.25, atol=2e-6)


def test_the_two_kernels_differ_from_each_other_and_from_lanczos():
    rng = np.random.default_rng(7)
    image = rng.random((32, 32, 3), dtype=np.float32)
    outputs = [
        resize(image, (64, 64), f)
        for f in (ResizeFilter.MKS2013, ResizeFilter.MKS2021, ResizeFilter.LANCZOS)
    ]
    assert np.abs(outputs[0] - outputs[1]).max() > 1e-3
    assert np.abs(outputs[0] - outputs[2]).max() > 1e-3


@pytest.mark.parametrize("filter_", KERNELS)
def test_channels_resize_alike_and_alpha_is_premultiplied(filter_):
    rng = np.random.default_rng(8)
    grey = rng.random((16, 24), dtype=np.float32)
    single = resize(grey, (40, 10), filter_)
    many = resize(np.dstack([grey] * 3), (40, 10), filter_)
    for c in range(3):
        np.testing.assert_array_equal(many[:, :, c], single[:, :, 0])
    # A colour under alpha 0 cannot bleed: a pixel next to it keeps its own colour once alpha is divided back out, where
    # alpha's overshoot has not been clipped.
    rgba = np.zeros((1, 16, 4), np.float32)
    rgba[0, :8] = (0.25, 0.5, 0.75, 1.0)
    rgba[0, 8:] = (1.0, 0.0, 1.0, 0.0)
    out = resize(rgba, (32, 1), filter_)
    visible = out[0, (out[0, :, 3] > 0.05) & (out[0, :, 3] < 0.999), :3]
    assert len(visible) >= 2
    np.testing.assert_allclose(visible, np.tile((0.25, 0.5, 0.75), (len(visible), 1)), atol=2e-5)
    separate = resize(rgba, (32, 1), filter_, separate_alpha=True)
    assert np.abs(separate[0, 18, 0] - out[0, 18, 0]) > 1e-3


def test_c_entry_still_refuses_dpid_and_unknown_ids():
    from nodes.impl import native_resample

    image = np.zeros((2, 2, 3), np.float32)
    for filter_id in (0, 12, 15, -1):
        with pytest.raises(ValueError, match="Unknown filtered resize method"):
            native_resample.filtered(image, (3, 3), filter_id, False)


def frozen_outputs(filter_: ResizeFilter) -> str:
    h = hashlib.sha256()
    for channels in (1, 3, 4):
        shape = (37, 53) if channels == 1 else (37, 53, channels)
        count = int(np.prod(shape))
        image = ((np.arange(count, dtype=np.int64) * 2861) % 8191) / 8190
        image = image.reshape(shape).astype(np.float32)
        for size in ((91, 113), (17, 11), (53, 37)):
            h.update(resize(image, size, filter_).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("filter_", KERNELS)
def test_frozen_output(filter_):
    assert frozen_outputs(filter_) == FROZEN[filter_]


ISA_CHILD = """
import json
import sys

sys.path.insert(0, sys.argv[1])
import test_magic_kernel as t
from nodes.impl import native

print(json.dumps({"line": native.isa_line(), "digests": [t.frozen_outputs(f) for f in t.KERNELS]}))
"""


def test_identical_bits_at_every_isa_level():
    wanted = [FROZEN[filter_] for filter_ in KERNELS]
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
        result = json.loads(completed.stdout.splitlines()[-1])
        assert f"requested {level}" in result["line"]
        assert result["digests"] == wanted, level
