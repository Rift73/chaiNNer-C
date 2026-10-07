"""Full Blend Images canvas parity and high-channel Crop Border coverage."""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest
import reference_blend
from test_resample_ops import border, border_ref, load_body, with_layout

from nodes.impl import native_composite as native
from nodes.impl.color.color import Color
from nodes.impl.native_versions import CV_CN_MAX

ROOT = Path(__file__).resolve().parents[2]
NODE = (
    ROOT
    / "backend/src/packages/chaiNNer_standard/image_utility/compositing/blend_images.py"
)
current = load_body(NODE, "complete_composite_node")
reference = load_body(
    Path(__file__).with_name("reference_composite") / "blend_images.py",
    "original_composite_node",
)
reference.__dict__.update(
    blend_images=lambda a, b, mode: reference_blend.blend_images(
        a, b, reference_blend.BlendMode[mode.name]
    )
)


def original_bgra(image, channels):
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA).copy()
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA).copy()
    assert channels == 4
    return image.copy()


reference.__dict__.update(convert_to_bgra=original_bgra)


def image(channels, shape=(7, 11), seed=4):
    return np.random.default_rng(seed).random(
        shape if channels == 1 else (*shape, channels), dtype=np.float32
    )


def compare(base, overlay, mode, position, xy, crop):
    args = (
        base,
        overlay,
        current.BlendMode[mode],
        current.BlendOverlayPosition[position],
        *xy,
        *xy,
        crop,
    )
    reference_args = (
        base,
        overlay,
        reference.BlendMode[mode],
        reference.BlendOverlayPosition[position],
        *xy,
        *xy,
        crop,
    )
    with np.errstate(all="ignore"):
        actual = current.blend_images_node(*args)
        expected = reference.blend_images_node(*reference_args)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    # Bits, so the sign of a zero counts (Consult 11 D-17.1: np.clip keeps -0).
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    return actual


@pytest.mark.parametrize("mode", [mode.name for mode in current.BlendMode])
@pytest.mark.parametrize("channels", [(a, b) for a in (1, 3, 4) for b in (1, 3, 4)])
@pytest.mark.parametrize(
    "offset,crop", [((2, 1), True), ((-4, -3), True), ((-5, 4), False)]
)
def test_every_mode_and_channel_canvas(mode, channels, offset, crop):
    base, overlay = image(channels[0]), image(channels[1], (5, 9), 91)
    saved = base.copy(), overlay.copy()
    compare(base, overlay, mode, "PIXEL_OFFSET", offset, crop)
    for actual, expected in zip((base, overlay), saved, strict=True):
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("position", [p.name for p in current.BlendOverlayPosition])
@pytest.mark.parametrize(
    "layout", ["contiguous", "strided", "transpose", "readonly", "unaligned"]
)
@pytest.mark.parametrize("crop", [False, True])
def test_positions_layouts_and_large_overlay(position, layout, crop):
    compare(
        with_layout(image(3), layout),
        with_layout(image(4, (13, 17)), layout),
        "SOFT_LIGHT",
        position,
        (33, -21),
        crop,
    )


@pytest.mark.parametrize("channels", [(a, b) for a in (1, 3, 4) for b in (1, 3, 4)])
@pytest.mark.parametrize("constant", [0, 1])
@pytest.mark.parametrize("offset", [(0, 0), (4, -3), (-40, 50)])
def test_colors_and_nonoverlap(channels, constant, offset):
    sources: list[np.ndarray | Color] = [image(channels[0]), image(channels[1], (3, 5))]
    sources[constant] = Color(tuple(0.17 + c * 0.19 for c in range(channels[constant])))
    base, overlay = sources
    compare(base, overlay, "NORMAL", "PIXEL_OFFSET", offset, True)
    compare(
        image(channels[0]),
        image(channels[1]),
        "NORMAL",
        "PIXEL_OFFSET",
        (-40, 50),
        True,
    )


@pytest.mark.parametrize("mode", [m.name for m in current.BlendMode])
def test_special_values(mode):
    base, overlay = image(4), image(3, (9, 13))
    for array in (base, overlay):
        array.flat[:9] = [np.nan, np.inf, -np.inf, -0.0, 0.0, 1.0, -1.0, 3.0, 0.5]
    compare(base, overlay, mode, "PIXEL_OFFSET", (-3, 2), False)


@pytest.mark.parametrize("channels", [5, 7, 8, 9, 16, 129, 257, 8193])
@pytest.mark.parametrize(
    "layout",
    [
        "contiguous",
        "strided",
        "transpose",
        "readonly",
        "unaligned",
        "fortran",
        "channel_strided",
    ],
)
@pytest.mark.parametrize(
    "selection", ["ALL_SECTIONS", "CENTER_SECTION", "LARGEST_SECTION"]
)
def test_border_high_channels(channels, layout, selection):
    data = image(channels, (7, 9))
    if layout == "fortran":
        data = np.asfortranarray(data)
    elif layout == "channel_strided":
        data = np.ascontiguousarray(data.transpose(0, 2, 1)).transpose(0, 2, 1)
    else:
        data = with_layout(data, layout)
    data = data.copy(order="K") if not data.flags.writeable else data
    data[:2] = 0.15
    data[-2:] = 0.15
    data[:, :2] = 0.15
    data[:, -2:] = 0.15
    expected = border_ref.crop_border_node(
        data, 17, border_ref.SelectMode[selection], 1
    )
    actual = border.crop_border_node(data, 17, border.SelectMode[selection], 1)
    np.testing.assert_array_equal(actual, expected)
    assert np.shares_memory(actual, data)


def test_concurrent_canvas():
    base, overlay = image(4, (129, 131)), image(3, (127, 129))
    base.flags.writeable = overlay.flags.writeable = False

    def run(_):
        return native.canvas(base, overlay, 17, -7, 8, False)

    expected = run(0)
    with ThreadPoolExecutor(max_workers=6) as executor:
        outputs = list(executor.map(run, range(18)))
    for output in outputs:
        assert output.tobytes() == expected.tobytes()


def test_foreign_shape_and_raw_abi():
    for shape in ((2, 3, 0), (0, 3, 3), (2, 3, 3, 2)):
        with pytest.raises(ValueError):
            native.canvas(np.empty(shape, np.float32), image(3), 0, 0, 0, True)
    with pytest.raises(ValueError, match="At least one"):
        native.canvas(Color.gray(0), Color.gray(1), 0, 0, 0, True)
    function = native._api()
    assert function(None, None, None, None, 0) == 1
    data = np.zeros(3, np.float32)
    layer = native._Layer(native.ptr(data), ct.c_size_t(-1).value, 2, 3, 0)
    assert function(ct.byref(layer), None, None, None, 0) == 2
    assert not data.any()


@pytest.mark.parametrize("channels", [5, 8, 9, 16, 129, 257, 8193])
@pytest.mark.parametrize("layout", ["C", "F", "channel_strided", "broadcast"])
def test_border_mean_threshold_rounding(channels, layout):
    data = image(channels, (5, 7), 998)
    if layout == "F":
        data = np.asfortranarray(data)
    elif layout == "channel_strided":
        data = np.ascontiguousarray(data.transpose(0, 2, 1)).transpose(0, 2, 1)
    elif layout == "broadcast":
        data = np.broadcast_to(data[:, :, :1], data.shape)
    color = border_ref.get_border_color(data)
    differences = np.abs(data - color).mean(axis=-1)
    for value in (differences[1, 1], differences[2, 3]):
        for tolerance in (
            value,
            np.nextafter(value, np.float32(0)),
            np.nextafter(value, np.float32(1)),
        ):
            expected = border_ref.crop_border_node(
                data, float(tolerance) * 100, border_ref.SelectMode.ALL_SECTIONS, 0
            )
            actual = border.crop_border_node(
                data, float(tolerance) * 100, border.SelectMode.ALL_SECTIONS, 0
            )
            np.testing.assert_array_equal(actual, expected)


# OpenCV 5.0.0 converts at most CV_CN_MAX channels to a 2-D Mat (OpenCV 4: 512).
@pytest.mark.parametrize("channels", [5, 7, 8, 9, 17, 65, CV_CN_MAX])
@pytest.mark.parametrize(
    "layout", ["contiguous", "strided", "transpose", "readonly", "unaligned"]
)
@pytest.mark.parametrize("offset", [(0, 0), (1, -1), (-2, 3), (30, 40)])
def test_shift_supported_high_channels(channels, layout, offset):
    from nodes.impl import native_channels
    from nodes.impl.image_utils import ShiftFill, shift

    source = with_layout(image(channels), layout)
    saved = source.copy()
    dx, dy = offset
    expected = shift(source, dx, dy, ShiftFill.BLACK)
    actual = native_channels.shift(source, dx, dy, ShiftFill.BLACK.value)
    assert actual.tobytes() == expected.tobytes()
    assert source.tobytes() == saved.tobytes()


@pytest.mark.parametrize("channels", [5, 8, 17, CV_CN_MAX])
@pytest.mark.parametrize("special", [np.nan, np.inf, -np.inf, -0.0])
def test_shift_high_channel_nonfinite(channels, special):
    from nodes.impl import native_channels
    from nodes.impl.image_utils import ShiftFill, shift

    source = image(channels, (3, 5))
    source[0, 0], source[1, 3], source[-1, -1] = special, special, special
    for offset in ((0, 0), (1, -1), (-2, 1)):
        expected = shift(source, *offset, ShiftFill.BLACK)
        actual = native_channels.shift(source, *offset, ShiftFill.BLACK.value)
        np.testing.assert_array_equal(actual, expected)
        if special == 0:
            assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("channels", [5, CV_CN_MAX, CV_CN_MAX + 1, 513])
@pytest.mark.parametrize("fill", [-1, 0, 1])
def test_shift_original_high_channel_errors(channels, fill):
    from nodes.impl import native_channels
    from nodes.impl.image_utils import ShiftFill, shift

    if channels <= CV_CN_MAX and fill == 0:
        return
    source = image(channels, (2, 3))
    with pytest.raises((cv2.error, AssertionError)) as expected:
        shift(source, 1, 1, ShiftFill(fill))
    with pytest.raises(type(expected.value)) as actual:
        native_channels.shift(source, 1, 1, fill)
    assert str(actual.value) == str(expected.value)
