"""Linear Light: the curves (native/include/linear_light.h through native/src/linear_light_ops.c and
nodes/impl/linear_light.py) and the Descale and Resize paths that use them."""

import cv2
import numpy as np
import pytest
from native_images import EDGES, image, seed_of
from test_isa_dispatch import isa_get, isa_set

from nodes.impl import linear_light, native_descale, native_resample
from nodes.impl.linear_light import LightTransfer
from nodes.impl.native import ptr
from nodes.impl.resize import ResizeFilter, resize

TRANSFERS = list(LightTransfer)
# Each curve's breakpoints: encoded and linear.
BREAKPOINTS = {
    LightTransfer.GAMMA_24: (),
    LightTransfer.SRGB: (0.04045, 0.0031308),
    LightTransfer.GAMMA_22: (),
    LightTransfer.BT709_CAMERA: (0.081, 0.018),
}


def to_linear(x: np.ndarray, transfer: LightTransfer) -> np.ndarray:
    """The curve to linear light in float64, extended as copysign(f(|x|), x)."""
    a = np.abs(x.astype(np.float64))
    with np.errstate(invalid="ignore"):
        if transfer is LightTransfer.GAMMA_24:
            v = a**2.4
        elif transfer is LightTransfer.SRGB:
            v = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
        elif transfer is LightTransfer.GAMMA_22:
            v = a**2.2
        else:
            v = np.where(a < 0.081, a / 4.5, ((a + 0.099) / 1.099) ** (1 / 0.45))
    return np.copysign(v, x)


def to_encoded(x: np.ndarray, transfer: LightTransfer) -> np.ndarray:
    a = np.abs(x.astype(np.float64))
    with np.errstate(invalid="ignore"):
        if transfer is LightTransfer.GAMMA_24:
            v = a ** (1 / 2.4)
        elif transfer is LightTransfer.SRGB:
            v = np.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055)
        elif transfer is LightTransfer.GAMMA_22:
            v = a ** (1 / 2.2)
        else:
            v = np.where(a < 0.018, 4.5 * a, 1.099 * a**0.45 - 0.099)
    return np.copysign(v, x)


def converted(values: np.ndarray, transfer: LightTransfer, encode: bool) -> np.ndarray:
    if encode:
        return linear_light.encode_inplace(values.copy(), transfer)
    return linear_light.linearize(values, transfer)


@pytest.mark.parametrize("encode", [False, True])
@pytest.mark.parametrize("transfer", TRANSFERS)
def test_curves_at_fixed_points(transfer, encode):
    points = [0.0, 0.5, 1.0, -0.25, 1.5]
    for breakpoint in BREAKPOINTS[transfer]:
        below, above = np.nextafter(
            np.float32(breakpoint), [np.float32(0), np.float32(1)]
        )
        points += [breakpoint, float(below), float(above)]
    values = np.array([points], np.float32)
    # The kernel reads float32 and evaluates in double: the reference takes the same inputs.
    expected = (to_encoded if encode else to_linear)(values, transfer).astype(
        np.float32
    )
    np.testing.assert_array_max_ulp(
        converted(values, transfer, encode), expected, maxulp=1
    )


@pytest.mark.parametrize("encode", [False, True])
@pytest.mark.parametrize("transfer", TRANSFERS)
def test_curves_within_an_ulp_of_pow(transfer, encode):
    # The kernel's polynomial power against NumPy's pow, rounded to float32 the same way.
    rng = np.random.default_rng(seed_of("light ulp", transfer.value, encode))
    magnitudes = np.exp(rng.uniform(-100.0, 88.0, 50000))
    values = np.concatenate([rng.uniform(-2.0, 2.0, 100000), magnitudes, -magnitudes])
    values = values.astype(np.float32)[np.newaxis]
    # Powers past float32's range are infinities on both sides.
    with np.errstate(over="ignore"):
        expected = (to_encoded if encode else to_linear)(values, transfer)
        expected = expected.astype(np.float32)
    np.testing.assert_array_max_ulp(
        converted(values, transfer, encode), expected, maxulp=1
    )


@pytest.mark.parametrize("transfer", TRANSFERS)
def test_round_trip(transfer):
    values = np.linspace(-0.5, 1.5, 20001, dtype=np.float32)[np.newaxis]
    if transfer is LightTransfer.BT709_CAMERA:
        # The standard's rounded constants leave a gap: 1.099 * 0.018^0.45 - 0.099 is
        # 0.0812479, not 0.081, so an encoded value in between maps below 0.018 and comes
        # back on the 4.5x segment (up to 2.5e-4 off).
        gap = (np.abs(values) >= 0.081) & (np.abs(values) < 0.0812480)
        assert (
            np.abs(
                to_encoded(to_linear(values[gap], transfer), transfer) - values[gap]
            ).max()
            > 2e-4
        )
        values = values[~gap][np.newaxis]
    back = linear_light.encode_inplace(
        linear_light.linearize(values, transfer), transfer
    )
    assert np.abs(back - values).max() <= 2e-6


@pytest.mark.parametrize("transfer", TRANSFERS)
def test_alpha_and_special_values(transfer):
    img = image(seed_of("light alpha", transfer.value), 9, 13, 4)
    for encode in (False, True):
        out = converted(img, transfer, encode)
        assert np.array_equal(
            out[:, :, 3].view(np.uint32), img[:, :, 3].view(np.uint32)
        )
        colour = img[:, :, :3]
        finite = np.isfinite(colour)
        expected = (to_encoded if encode else to_linear)(colour[finite], transfer)
        np.testing.assert_array_max_ulp(
            out[:, :, :3][finite], expected.astype(np.float32), maxulp=1
        )
        assert np.array_equal(np.isnan(out[:, :, :3]), np.isnan(colour))
        assert np.array_equal(np.signbit(out[:, :, :3]), np.signbit(colour))
        infinite = np.isinf(colour)
        assert np.array_equal(out[:, :, :3][infinite], colour[infinite])


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_rows_in_parallel_equal_one_row(channels):
    img = image(seed_of("light rows", channels), 701, 613, channels)
    whole = linear_light.linearize(img, LightTransfer.SRGB)
    row = img.reshape(1, 701 * 613, channels) if channels > 1 else img.reshape(1, -1)
    one_row = linear_light.linearize(row, LightTransfer.SRGB)
    assert np.array_equal(
        whole.reshape(-1).view(np.uint32), one_row.reshape(-1).view(np.uint32)
    )


def test_kernel_arguments():
    function = linear_light._api().cn_linear_light
    data = np.zeros((2, 3, 4), np.float32)
    assert function(ptr(data), 2, 3, 4, 3, 1, 1) == 0
    assert function(ptr(data), 2, 3, 4, 4, 0, 1) == 1  # no such transfer
    assert function(ptr(data), 2, 3, 4, -1, 0, 1) == 1
    assert function(ptr(data), 2, 3, 4, 0, 2, 0) == 1
    assert function(ptr(data), 2, 3, 4, 0, 0, 2) == 1
    assert function(ptr(data), 6, 1, 1, 0, 0, 1) == 1  # alpha: two or four channels
    assert function(ptr(data), 2, 4, 3, 0, 0, 1) == 1
    assert function(ptr(data), 2, 3, 5, 0, 0, 0) == 1
    assert function(None, 2, 3, 4, 0, 0, 0) == 1
    assert function(ptr(data), 0, 3, 4, 0, 0, 1) == 0  # empty: nothing to convert
    assert function(ptr(data), 2**62, 2**62, 4, 0, 0, 0) == 2
    with pytest.raises(ValueError, match="writable"):
        readonly = data.copy()
        readonly.setflags(write=False)
        linear_light.encode_inplace(readonly, LightTransfer.SRGB)


@pytest.mark.parametrize(
    ("channels", "skip_alpha"), [(1, 0), (2, 0), (2, 1), (3, 0), (4, 0), (4, 1)]
)
def test_levels_give_the_same_bits(channels, skip_alpha):
    # The AVX2 unit runs the scalar path's operations: every ISA level gives the same bits,
    # in groups of four and in the tails (37 pixels a row).
    rng = np.random.default_rng(seed_of("light levels", channels, skip_alpha))
    values = np.concatenate(
        [
            EDGES,
            rng.uniform(-2.0, 2.0, 30000),
            rng.uniform(0.0, 0.1, 10000),
            np.exp(rng.uniform(-100.0, 88.0, 10000)),
        ]
    ).astype(np.float32)
    rows = values.size // (37 * channels)
    img = values[: rows * 37 * channels].reshape(rows, 37, channels)
    function = linear_light._api().cn_linear_light
    effective, _, maximum = isa_get()
    outputs = {}
    try:
        for level in range(maximum + 1):
            assert isa_set(level) == level
            results = []
            for transfer in TRANSFERS:
                for encode in (0, 1):
                    out = img.copy()
                    status = function(
                        ptr(out), rows, 37, channels, transfer.value, encode, skip_alpha
                    )
                    assert status == 0
                    results.append(out.view(np.uint32))
            outputs[level] = results
    finally:
        assert isa_set(effective) == effective
    for level in range(1, maximum + 1):
        for a, b in zip(outputs[level], outputs[0], strict=True):
            assert np.array_equal(a, b), level
    if skip_alpha:
        for out in outputs[0]:
            assert np.array_equal(out[:, :, -1], img[:, :, -1].view(np.uint32))


def descale(img: np.ndarray, light: LightTransfer | None) -> np.ndarray:
    """Bilinear (1), Extend border (2: OpenCV's replicate padding) to 67 x 45."""
    return native_descale.descale(
        img, 67, 45, 1, 3, 0.0, 0.5, 1.0, (0.0, 0.0), (67, 45), 2, light
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("transfer", TRANSFERS)
def test_descale_recovers_a_linear_light_upscale(transfer, channels):
    rng = np.random.default_rng(seed_of("descale light", transfer.value, channels))
    small = rng.uniform(0.1, 0.9, (45, 67, channels)).astype(np.float32)
    small = small[:, :, 0] if channels == 1 else small
    upscaled = cv2.resize(
        linear_light.linearize(small, transfer),
        (100, 70),
        interpolation=cv2.INTER_LINEAR,
    )
    large = linear_light.encode_inplace(upscaled, transfer)
    assert np.abs(descale(large, transfer) - small).max() <= 1e-4
    assert np.abs(descale(large, None) - small).max() > 1e-3


def test_descale_keeps_ringing():
    # A Lanczos descale of a hard edge rings below 0: linear light keeps the sign.
    img = np.full((40, 60, 3), 0.02, np.float32)
    img[:, 30:] = 1.0
    out = native_descale.descale(
        img, 41, 27, 3, 3, 0.0, 0.5, 1.0, (0.0, 0.0), (41, 27), 0, LightTransfer.SRGB
    )
    assert out.min() < -1e-3 and out.max() > 1.0


FILTERS = [
    f for f in ResizeFilter if f not in (ResizeFilter.AUTO, ResizeFilter.NEAREST)
]


def halve(x: np.ndarray) -> np.ndarray:
    """Box halving in float64: the mean of each 2 x 2 block."""
    h, w = x.shape[0] // 2, x.shape[1] // 2
    return x.astype(np.float64).reshape(h, 2, w, 2, *x.shape[2:]).mean(axis=(1, 3))


def halved_in_linear_light(
    img: np.ndarray, transfer: LightTransfer, separate: bool
) -> np.ndarray:
    """Box halving in float64 as Linear Light resizes: the colour channels linearized,
    alpha-weighted unless separate, then encoded; alpha averaged as is."""
    if img.ndim == 3 and img.shape[2] == 4:
        colour, alpha = to_linear(img[:, :, :3], transfer), img[:, :, 3:]
        colour = halve(colour) if separate else halve(colour * alpha) / halve(alpha)
        return np.concatenate([to_encoded(colour, transfer), halve(alpha)], axis=2)
    return to_encoded(halve(to_linear(img, transfer)), transfer)


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("transfer", TRANSFERS)
def test_resize_matches_a_float64_reference(transfer, channels, separate):
    rng = np.random.default_rng(
        seed_of("resize light", transfer.value, channels, separate)
    )
    img = rng.random((34, 46, channels), dtype=np.float32)
    if channels == 4:
        img[:, :, 3] = rng.uniform(0.2, 1.0, (34, 46))
    img = img[:, :, 0] if channels == 1 else img
    actual = resize(img, (23, 17), ResizeFilter.BOX, separate, light=transfer)
    expected = halved_in_linear_light(img, transfer, separate)
    # The kernels return height x width x channels for a 2-D image too.
    assert np.abs(actual.reshape(expected.shape) - expected).max() <= 1e-5
    plain = resize(img, (23, 17), ResizeFilter.BOX, separate)
    assert np.abs(actual - plain).max() > 1e-3


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("filter", FILTERS)
@pytest.mark.parametrize("transfer", TRANSFERS)
def test_resize_converts_around_the_premultiplied_resize(
    transfer, filter, channels, separate
):
    # Every curve, DPID included: linearize, premultiply, resize without the kernels'
    # gamma correction, un-premultiply, encode.
    img = image(
        seed_of("resize light order", transfer.value, filter.value, channels, separate),
        23,
        29,
        channels,
        edges=False,
    )
    for dims in ((13, 17), (47, 31)):
        actual = resize(img, dims, filter, separate, light=transfer)
        source = linear_light.linearize(img, transfer)
        premultiplied = channels == 4 and not separate
        if premultiplied:
            source = native_resample.premultiply(source)
        if filter is ResizeFilter.DPID:
            out = native_resample.dpid(source, dims, 1.0, separate, False)
        else:
            out = native_resample.filtered(source, dims, filter.value, False)
        if premultiplied:
            out = native_resample.finish_alpha_inplace(out)
        expected = linear_light.encode_inplace(out, transfer)
        assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("filter", [ResizeFilter.BOX, ResizeFilter.DPID])
@pytest.mark.parametrize("transfer", TRANSFERS)
def test_resize_soft_alpha_edge(transfer, filter):
    # Opaque white beside (0.8, 0.4, 0.2) at alpha 0.2, then black at alpha 0.2, halved:
    # the edge pixel mixes white with the faint colour in linear light, weighted by
    # alpha, and stays below white (Gamma 2.2 through the kernels' own gamma
    # correction gave (1, 1, 1)).
    img = np.zeros((4, 8, 4), np.float32)
    img[:, :4] = (1, 1, 1, 1)
    img[:, 4:] = (0, 0, 0, 0.2)
    img[:, 3:5] = (0.8, 0.4, 0.2, 0.2)
    actual = resize(img, (4, 2), filter, light=transfer)
    assert np.isfinite(actual).all()
    assert (actual[:, 1, :3] < 0.99).all(), actual[0, 1]
    if filter is ResizeFilter.BOX:
        expected = halved_in_linear_light(img, transfer, False)
        assert np.abs(actual - expected).max() <= 1e-5
        if transfer is LightTransfer.GAMMA_22:
            np.testing.assert_allclose(
                actual[0, 1, :3], (0.970, 0.932, 0.923), atol=1e-3
            )


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("filter", FILTERS)
@pytest.mark.parametrize("transfer", TRANSFERS)
def test_resize_keeps_values_below_0_and_above_1(transfer, filter, channels, separate):
    # Values below 0 (Descale's ringing) and above 1 (highlights): Linear Light never
    # turns them into NaN or white. A flat field mixes nothing, so it comes out as the
    # resize gives it without Linear Light (the filters that clamp to [0, 1] clamp it).
    for value in (-0.02, -0.5, 1.3):
        img = np.full((12, 16, channels), value, np.float32)
        if channels == 4:
            img[:, :, 3] = 0.6
        img = img[:, :, 0] if channels == 1 else img
        for dims in ((8, 6), (24, 18)):
            out = resize(img, dims, filter, separate, light=transfer)
            plain = resize(img, dims, filter, separate)
            assert not np.isnan(out).any(), value
            np.testing.assert_allclose(out, plain, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("transfer", TRANSFERS)
def test_resize_ringing_never_turns_white(transfer, channels, separate):
    # Ringing pixels among mid grey through same-size Lanczos and Linear, which mix
    # nothing either: as without Linear Light (Gamma 2.2 through the kernels' own gamma
    # correction made -0.02 white, or NaN).
    img = np.full((16, 16, channels), 0.5, np.float32)
    img[8, 8, : min(channels, 3)] = -0.02
    img[2, 2, : min(channels, 3)] = -0.5
    img = img[:, :, 0] if channels == 1 else img
    for same_size in (ResizeFilter.LANCZOS, ResizeFilter.LINEAR):
        out = resize(img, (16, 16), same_size, separate, light=transfer)
        plain = resize(img, (16, 16), same_size, separate)
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out, plain, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_resize_nearest_ignores_linear_light(channels):
    img = image(seed_of("nearest light", channels), 17, 31, channels)
    plain = resize(img, (40, 9), ResizeFilter.NEAREST)
    for transfer in TRANSFERS:
        lit = resize(img, (40, 9), ResizeFilter.NEAREST, False, light=transfer)
        assert np.array_equal(lit.view(np.uint32), plain.view(np.uint32))
