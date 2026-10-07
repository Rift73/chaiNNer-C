"""Consult 11 D-16: the 18 producers return upstream's memory layout.

Each producer runs through the port and through its frozen upstream body (the
differential modules' loaders) on inputs in Lens Blur's planar layout (a (c, h, w)
base's transpose(1, 2, 0)), in C order and, for the nodes with more than one image
input, mixed. The inputs hold signed zeros, a NaN with a payload, infinities,
denormals and values outside [0, 1]. Stretch Contrast's and Crop to Content's
are finite (a NaN range makes a NaN image), and Color Transfer's moderate too (a
NaN, an infinity or a covariance overflow makes both trees raise). At every ISA
level the CPU has (cn_isa_set) and under the default MXCSR and DAZ|FTZ
(golden_kernels.mxcsr), the port's result equals upstream's: NaN positions and
every other bit, and upstream's layout, the strides np.clip's order K gives the
result at ImageOutput.enforce, or for an assume_normalized output the result's
own strides, which enforce passes on.

These are the goldens of D-16's planar-aware kernels: the channel runs of
cn_adjust_f32 and cn_levels_f32, the strided pixels of cn_chroma_key,
cn_matting_trimap_output and the cn_transfer_* kernels, the output rows of
cn_channels_crop, and the per-plane calls of cn_channels_assemble,
cn_channels_shift and cn_layout_paste.
"""

from __future__ import annotations

from typing import Any

import golden_kernels
import numpy as np
import pytest
from test_adjustments import MODULES as ADJUSTMENTS
from test_analysis_ops import MODULES as ANALYSIS
from test_channels import MODULES as CHANNELS
from test_channels import node_functions
from test_isa_dispatch import isa_get, isa_set
from test_layout_ops import MODULES as LAYOUT
from test_transfer_complete import NODE, ORIGINAL_NODE
from test_transparency import CHROMA, ORIGINAL_CHROMA

from nodes.impl import native
from nodes.impl.color.color import Color
from nodes.impl.image_utils import ShiftFill

NAN = np.uint32(0x7FC12345).view(np.float32)
# Color Transfer's covariance overflows past about 1e19.
MODERATE = np.array([0, -0.0, 1, 0.5, -0.25, 1.75, 1e-40, -1e-40], np.float32)
FINITE = np.array([*MODERATE, 3e38], np.float32)
SPECIALS = np.array([*FINITE, np.inf, -np.inf, NAN], np.float32)
SMALL, LARGE = (37, 61), (129, 257)
LAYOUTS = ("planar", "c")
PAIRS = [(a, b) for a in LAYOUTS for b in LAYOUTS]


@pytest.fixture(params=range(len(native.ISA_LEVELS)), ids=native.ISA_LEVELS)
def level(request):
    """Each ISA level the CPU has, set while no kernel runs."""
    effective, _, cpu = isa_get()
    if request.param > cpu:
        pytest.skip(f"the CPU's maximum level is {native.ISA_LEVELS[cpu]}")
    assert isa_set(request.param) == request.param
    yield request.param
    assert isa_set(effective) == effective


@pytest.fixture(params=golden_kernels.MXCSR_STATES)
def state(request):
    return request.param


def image(channels, layout, size=SMALL, seed=0, values=SPECIALS):
    """Noise with a fifth of its values special: a planar base's transpose(1, 2,
    0), or the same values in C order. One channel is a C-ordered plane."""
    rng = np.random.default_rng(seed)
    base = rng.random((channels, *size), dtype=np.float32)
    pick = rng.random(base.shape) < 0.2
    base[pick] = rng.choice(values, int(pick.sum()))
    if channels == 1:
        return base[0]
    planar = base.transpose(1, 2, 0)
    return planar if layout == "planar" else np.ascontiguousarray(planar)


def outcome(function, *args):
    try:
        return function(*args)
    except Exception as error:  # compared with the other tree's: type and message
        return error


def layout_of(result, exact):
    """What the next node receives: np.clip's order-K layout of result, or with
    exact (an assume_normalized output) result's own strides."""
    if exact:
        return result.strides
    return np.empty_like(result, np.float32, order="K").strides


def same(actual, expected, exact):
    if isinstance(expected, Exception):
        assert isinstance(actual, Exception), actual
        assert (type(actual), str(actual)) == (type(expected), str(expected))
        return
    assert isinstance(actual, np.ndarray), actual
    assert (actual.shape, actual.dtype) == (expected.shape, expected.dtype)
    assert layout_of(actual, exact) == layout_of(expected, exact)
    if expected.dtype == np.complex128:
        # What enforce keeps of a complex result (normalize's astype).
        actual, expected = actual.real, expected.real
    # NaN positions and every other bit (ARCHITECTURE §7): a NaN the libm makes,
    # from log10 of a negative, carries that library's payload.
    nan = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), nan)
    bits = np.uint32 if expected.dtype == np.float32 else np.uint64
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual)[~nan].view(bits),
        np.ascontiguousarray(expected)[~nan].view(bits),
    )


def check(state, port, upstream, exact=False):
    """port and upstream, each a function and its arguments, under the state."""
    with golden_kernels.mxcsr(native.lib(), state), np.errstate(all="ignore"):
        expected = outcome(*upstream)
        actual = outcome(*port)
    same(actual, expected, exact)


def both(modules, name, *args) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """(function, *args) for the port's module and for upstream's, an args item
    that is a callable taking the module (its enum)."""
    current, original = (
        (
            getattr(module, name),
            *(arg(module) if callable(arg) else arg for arg in args),
        )
        for module in modules
    )
    return current, original


@pytest.mark.parametrize("size", [SMALL, LARGE], ids=["small", "large"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize(
    ("name", "args", "exact"),
    [
        ("add", [12], False),
        ("multiply", [1.3], False),
        ("divide", [1.7], False),
        ("clamp", [0.2, 0.7], True),
        ("invert_color", [], True),
        ("brightness_and_contrast", [15, 20], True),
        ("brightness_and_contrast", [-50, 50], True),
        ("log_to_linear", [95, 685, 0.6, False], False),
        ("log_to_linear", [95, 685, 0.6, True], False),
        ("color_levels", [True, True, True, False, 0.1, 0.9, 1.4, 0.05, 0.95], False),
        ("color_levels", [False, True, False, True, 0.2, 0.8, 0.7, 0, 1], False),
    ],
)
def test_adjustments(level, state, name, args, exact, channels, layout, size):
    data = image(channels, layout, size, seed=channels)
    check(state, *both(ADJUSTMENTS[name], name + "_node", data, *args), exact)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("association", ["PREMULTIPLY_RGB", "UNPREMULTIPLY_RGB"])
def test_premultiplied_alpha(level, state, association, layout):
    data = image(4, layout, LARGE, seed=5)
    check(
        state,
        *both(
            ADJUSTMENTS["premultiplied_alpha"],
            "premultiplied_alpha_node",
            data,
            lambda module: module.AlphaAssociation[association],
        ),
    )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize(
    ("mode", "keep"),
    [("AUTO", True), ("AUTO", False), ("PERCENTILE", True), ("MANUAL", True)],
)
def test_stretch_contrast(level, state, mode, keep, channels, layout):
    data = image(channels, layout, seed=7, values=FINITE)
    check(
        state,
        *both(
            ADJUSTMENTS["stretch_contrast"],
            "stretch_contrast_node",
            data,
            lambda module: module.StretchMode[mode],
            keep,
            1.5,
            20,
            200,
        ),
    )


@pytest.mark.parametrize("layouts", PAIRS)
@pytest.mark.parametrize(
    "channels", [(3,), (4,), (3, 1), (1, 3), (3, 3), (1, 1), (4, 4, 1)]
)
def test_merge_channels(level, state, channels, layouts):
    images = [
        image(count, layouts[min(index, 1)], seed=11 + index)
        for index, count in enumerate(channels)
    ]
    images += [None] * (4 - len(images))
    current, original = node_functions("merge_channels")
    check(state, (current, *images), (original, *images))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("rgb", [3, 4, 1, "color"])
@pytest.mark.parametrize("alpha", ["image", "color"])
def test_merge_transparency(level, state, rgb, alpha, layout):
    color = Color((0.1, 0.2, 0.3)) if rgb == "color" else image(rgb, layout, seed=13)
    plane = (
        Color.gray(0.4)
        if alpha == "color" and rgb != "color"
        else image(1, layout, seed=17)
    )
    current, original = node_functions("merge_transparency")
    check(state, (current, color, plane), (original, color, plane))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("offset", [(7, -5), (-61, 3), (0, 36)])
def test_shift_wrap(level, state, offset, channels, layout):
    data = image(channels, layout, seed=19)
    check(
        state,
        *both(
            CHANNELS["shift"],
            "shift_node",
            data,
            lambda module: module.ShiftCoordMode.ABSOLUTE,
            0,
            0,
            *offset,
            ShiftFill.WRAP,
        ),
    )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("threshold", [0, 50])
@pytest.mark.parametrize("border", [(0, 0, 0, 0), (2, 5, 3, 1)])
def test_crop_to_content(level, state, border, threshold, layout):
    """A real crop: upstream returns a view into np.copy(img) (assume_normalized),
    so its strides are the whole copy's."""
    data = image(4, layout, seed=23, values=FINITE)
    top, right, bottom, left = border
    data[:top, :, 3] = data[data.shape[0] - bottom :, :, 3] = 0
    data[:, :left, 3] = data[:, data.shape[1] - right :, 3] = 0
    current, original = node_functions("crop_to_content")
    check(state, (current, data, threshold), (original, data, threshold), exact=True)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("threshold", [0.1, 0.4])
def test_chroma_binary(level, state, threshold, layout):
    data = image(3, layout, seed=29)
    key = Color((0.5, 0.25, 0.75))
    check(
        state,
        (CHROMA.binary_keying, data, key, threshold),
        (ORIGINAL_CHROMA.binary_keying, data, key, threshold),
    )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_chroma_trimap_output(level, state, layout):
    data = image(3, layout, seed=31)
    args = (data, Color((0.5, 0.5, 0.5)), 0.2, 0.35, 0, 1, True)
    check(
        state,
        (CHROMA.trimap_matting_keying, *args),
        (ORIGINAL_CHROMA.trimap_matting_keying, *args),
    )


@pytest.mark.parametrize("layouts", PAIRS)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_high_pass_custom(level, state, channels, layouts):
    data = image(channels, layouts[0], seed=37)
    blurred = image(min(channels, 3), layouts[1], seed=41)
    check(
        state,
        *both(
            ANALYSIS["high_pass"],
            "high_pass_node",
            data,
            lambda module: module.BlurMode.CUSTOM,
            1.3,
            blurred,
            2.7,
        ),
    )


@pytest.mark.parametrize("layouts", PAIRS)
@pytest.mark.parametrize("channels", [(3, 3), (4, 3), (3, 4), (4, 4)])
@pytest.mark.parametrize(
    "algorithm", ["LINEAR_HISTOGRAM", "PRINCIPAL_COLOR", "MEAN_STD"]
)
def test_color_transfer(level, state, algorithm, channels, layouts):
    data = image(channels[0], layouts[0], seed=43, values=MODERATE)
    reference = image(channels[1], layouts[1], (29, 47), seed=47, values=MODERATE)
    check(
        state,
        *both(
            (NODE, ORIGINAL_NODE),
            "color_transfer_node",
            data,
            reference,
            lambda module: module.TransferColorAlgorithm[algorithm],
            lambda module: module.TransferColorSpace.RGB,
            lambda module: module.OverflowMethod.CLIP,
            False,
        ),
    )


def spritesheet(module, tiles, grid, order):
    collector = module.merge_spritesheet_node(None, *grid, module.OrderEnum[order])
    for tile in tiles:
        collector.on_iterate(tile)
    return collector.on_complete()


@pytest.mark.parametrize("layouts", ["planar", "c", "first", "row"])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("order", ["ROW_MAJOR", "COLUMN_MAJOR"])
def test_merge_spritesheet(level, state, order, channels, layouts):
    """Two rows (columns) of two sprites; first: only the first sprite planar; row:
    the first row (column) planar, so the sheet's two concatenations disagree."""
    planar = {
        "planar": {0, 1, 2, 3},
        "c": set(),
        "first": {0},
        "row": {0, 1},
    }[layouts]
    tiles = [
        image(channels, "planar" if n in planar else "c", (7, 11), seed=53 + n)
        for n in range(4)
    ]
    check(
        state,
        (spritesheet, LAYOUT["merge_spritesheet"][0], tiles, (2, 2), order),
        (spritesheet, LAYOUT["merge_spritesheet"][1], tiles, (2, 2), order),
    )
