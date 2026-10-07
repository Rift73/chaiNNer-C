"""Frozen-Rust parity and explicit deterministic corrections for both Dither nodes.

Ordinary cases compare every float32 bit, with no tolerance. The installed
palette hash order and R-tree tie traversal do not define a canonical answer;
those cases test exact minimum-distance membership plus documented ordering.
The original one-dimensional R-tree panic is retained as regression evidence
(recorded from the real module in the conformance manifest), not reproduced in
the C implementation or in chaiNNer-C's chainner_ext.
"""

from __future__ import annotations

import ast
import ctypes as ct
import json
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from test_isa_dispatch import isa_get, isa_set

from nodes.impl import native_neighborhood as native
from nodes.impl.native import lib

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "backend/src/packages/chaiNNer_standard/image_filter/quantize"
REFERENCE = Path(__file__).with_name("reference_neighborhood") / "quantize"


def load_node(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.ImportFrom)
        or (
            node.level == 0
            and node.module not in {"api", "nodes.groups"}
            and not (node.module or "").startswith("nodes.properties")
        )
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    module = types.ModuleType(path.stem)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


UNIFORM = (load_node(SOURCE / "dither.py"), load_node(REFERENCE / "dither.py"))
PALETTE = (
    load_node(SOURCE / "dither_palette.py"),
    load_node(REFERENCE / "dither_palette.py"),
)


def bits_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def image(channels=3, height=7, width=11, seed=825):
    return np.random.default_rng(seed).random(
        (height, width, channels), dtype=np.float32
    )


def uniform(module, source, colors=7, history=16, mode="RIEMERSMA", diffusion="FS"):
    return module.dither_node(
        source,
        colors,
        module.UniformDitherAlgorithm[mode],
        module.ThresholdMap.BAYER_16,
        module.ErrorDiffusionMap(diffusion),
        history,
    )


def palette(module, source, colors, history=16, mode="RIEMERSMA", diffusion="FS"):
    return module.dither_palette_node(
        source,
        colors,
        module.PaletteDitherAlgorithm[mode],
        module.ErrorDiffusionMap(diffusion),
        history,
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (1, 31),
        (29, 1),
        (2, 3),
        (3, 2),
        (3, 3),
        (4, 4),
        (7, 8),
        (8, 7),
        (13, 17),
        (17, 13),
        (5, 131),
        (129, 6),
    ],
)
@pytest.mark.parametrize("history", [2, 3, 16, 31, 257])
@pytest.mark.parametrize("colors", [2, 3, 7, 256, 0xFFFFFFFF])
def test_uniform_riemersma_exact(channels, shape, history, colors):
    source = image(channels, *shape, seed=channels + shape[0] * shape[1])
    saved = source.copy()
    bits_equal(
        uniform(UNIFORM[0], source, colors, history),
        uniform(UNIFORM[1], source, colors, history),
    )
    bits_equal(source, saved)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("history", [2, 7, 16, 31])
@pytest.mark.parametrize("count", [1, 2, 17, 299, 300, 301, 1024, 4096])
def test_palette_riemersma_exact(channels, history, count):
    if channels == 1 and count >= 300:
        return  # This original panic has separate exact-nearest regression tests.
    source = image(channels, 5, 7, seed=channels + count)
    colors = image(channels, 2, count, seed=channels + count + 1)
    saved, saved_colors = source.copy(), colors.copy()
    bits_equal(
        palette(PALETTE[0], source, colors, history),
        palette(PALETTE[1], source, colors, history),
    )
    bits_equal(source, saved)
    bits_equal(colors, saved_colors)


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("count", [300, 301, 1024, 4096])
@pytest.mark.parametrize("algorithm", list(native.DIFFUSION_IDS))
def test_large_palette_all_diffusions_exact(channels, count, algorithm):
    source = image(channels, 17, 23, seed=channels + count)
    colors = image(channels, 1, count, seed=channels + count + 1)
    bits_equal(
        palette(PALETTE[0], source, colors, mode="DIFFUSION", diffusion=algorithm),
        palette(PALETTE[1], source, colors, mode="DIFFUSION", diffusion=algorithm),
    )


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("count", [300, 301, 1024, 4096, 16384])
def test_large_palette_quantization_exact(channels, count):
    source = image(channels, 13, 17, seed=channels + count)
    colors = image(channels, 1, count, seed=channels + count + 1)
    bits_equal(
        palette(PALETTE[0], source, colors, mode="NONE"),
        palette(PALETTE[1], source, colors, mode="NONE"),
    )


def layout(source, name):
    if name == "strided":
        return source[::-2, ::-1, :]
    if name == "transposed":
        return source.swapaxes(0, 1)
    if name == "readonly":
        source.flags.writeable = False
        return source
    if name == "unaligned":
        result = np.ndarray(
            source.shape, np.float32, buffer=bytearray(source.nbytes + 1), offset=1
        )
        result[:] = source
        return result
    if name == "broadcast":
        return np.broadcast_to(source[:1, :1, :], source.shape)
    if name == "two_dimensional" and source.shape[2] == 1:
        return source[:, :, 0]
    return source


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize(
    "name",
    ["strided", "transposed", "readonly", "unaligned", "broadcast", "two_dimensional"],
)
@pytest.mark.parametrize("count", [17, 307])
def test_riemersma_array_contracts(channels, name, count):
    source = layout(image(channels), name)
    colors = layout(image(channels, 1, count, seed=884), "unaligned")[:, ::-1]
    saved, saved_colors = source.copy(), colors.copy()
    bits_equal(uniform(UNIFORM[0], source), uniform(UNIFORM[1], source))
    if channels != 1 or count < 300:
        bits_equal(
            palette(PALETTE[0], source, colors), palette(PALETTE[1], source, colors)
        )
    else:
        result = palette(PALETTE[0], source, colors)
        bits_equal(result, palette(PALETTE[0], source, colors))
        assert result.shape == (*source.shape[:2], channels)
    bits_equal(source, saved)
    bits_equal(colors, saved_colors)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("history", [2, 3, 16, 65, 10000])
def test_uniform_rounding_signed_zero_extremes(channels, history):
    values = np.array(
        [-2, -1, -0.5, -0.25, -0.0, 0.0, 0.25, 0.5, 0.75, 1, 2], np.float32
    )
    source = np.broadcast_to(values[None, :, None], (1, len(values), channels))
    for colors in (2, 3, 7, 0xFFFFFFFF):
        bits_equal(
            uniform(UNIFORM[0], source, colors, history),
            uniform(UNIFORM[1], source, colors, history),
        )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("mode", ["NONE", "DIFFUSION", "RIEMERSMA"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_image_existing_semantics(channels, mode, value):
    source = image(channels)
    source[2, 3, 0] = value
    colors = image(channels, 1, 17, seed=881)
    # NaN payload propagation may vary across scalar math runtimes; use exact
    # array equality for NaNs, while every finite output remains exact.
    np.testing.assert_array_equal(
        uniform(UNIFORM[0], source, mode=mode),
        uniform(UNIFORM[1], source, mode=mode),
    )
    bits_equal(
        palette(PALETTE[0], source, colors, mode=mode),
        palette(PALETTE[1], source, colors, mode=mode),
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("mode", ["NONE", "DIFFUSION", "RIEMERSMA"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("location", ["image", "palette"])
def test_nonfinite_large_palette_takes_the_linear_rule(channels, mode, value, location):
    # Consult 8 D-5: with 300+ unique colors a pixel with a nonfinite channel, or
    # every pixel when the palette holds a nonfinite color, takes the linear rule of
    # smaller palettes. Upstream's R-tree traversal decides those queries; its bulk
    # load panics on a NaN color of 3 or 4 channels, and chaiNNer-C's chainner_ext
    # keeps that panic.
    source = image(channels)
    colors = image(channels, 1, 307, seed=882)
    target = source if location == "image" else colors
    target.reshape(-1)[0] = value
    saved, saved_colors = source.copy(), colors.copy()
    result = palette(PALETTE[0], source, colors, mode=mode)
    bits_equal(result, palette(PALETTE[0], source, colors, mode=mode))
    if mode == "NONE":
        bits_equal(result, linear_reference(source, colors))
    if location == "palette" and np.isnan(value) and channels != 1:
        with pytest.raises(BaseException, match="None") as raised:
            palette(PALETTE[1], source, colors, mode=mode)
        assert type(raised.value).__name__ == "PanicException"
    else:
        bits_equal(result, palette(PALETTE[1], source, colors, mode=mode))
    bits_equal(source, saved)
    bits_equal(colors, saved_colors)


@pytest.fixture(params=["scalar", "auto"])
def isa_level(request):
    """The DLL's ISA level for the test: scalar, or the CPU's (avx2 and above take
    the brute-force search of palettes without a kd tree)."""
    effective = isa_get()[0]
    if request.param == "scalar":
        assert isa_set(0) == 0
    try:
        yield request.param
    finally:
        assert isa_set(effective) == effective


def with_color(colors, index, channel, value):
    """colors with a copy of its first color, one channel set to value, at index."""
    added = colors[:, :1].copy()
    added[0, 0, channel] = value
    return np.concatenate((colors[:, :index], added, colors[:, index:]), axis=1)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("mode", ["NONE", "DIFFUSION", "RIEMERSMA"])
@pytest.mark.parametrize("count", [299, 303, 307, 1023])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
@pytest.mark.parametrize("path", [0, 1], ids=["node", "chainner_ext"])
def test_an_infinite_color_never_changes_finite_pixels(
    isa_level, channels, mode, count, value, path
):
    # Consult 8 D-5's property, true upstream too: an infinite distance never wins,
    # so a finite image dithered with a +-inf color added to its palette equals the
    # result without it, across the 300-color threshold (with the color, 300 and 304
    # colors take the brute-force search at avx2).
    source = image(channels, 9, 13, seed=count)
    colors = image(channels, 1, count, seed=count + 1)
    index = count * 7 % (count + 1)
    infinite = with_color(colors, index, count % channels, value)
    bits_equal(
        palette(PALETTE[path], source, infinite, mode=mode),
        palette(PALETTE[path], source, colors, mode=mode),
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_an_infinite_color_never_changes_finite_pixels_beside_nonfinite_ones(
    channels, value
):
    source = image(channels, 9, 13, seed=31)
    source[1, 2, 0] = np.nan
    source[4, 5, -1] = np.inf
    source[7, 0, 0] = -np.inf
    colors = image(channels, 1, 401, seed=32)
    finite = np.isfinite(source).all(axis=2)
    infinite = with_color(colors, 200, 0, value)
    for path in (0, 1):
        expected = palette(PALETTE[path], source, colors, mode="NONE")
        actual = palette(PALETTE[path], source, infinite, mode="NONE")
        bits_equal(actual[finite], expected[finite])


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("mode", ["NONE", "DIFFUSION", "RIEMERSMA"])
def test_nonfinite_duplicate_palette_keeps_small_palette_semantics(channels, mode):
    source = image(channels)
    source[2, 3, 0] = np.nan
    colors = image(channels, 1, 19, seed=889)
    colors[0, 3, 0] = np.nan
    repeated = np.tile(colors, (1, 80, 1))
    bits_equal(
        palette(PALETTE[0], source, repeated, mode=mode),
        palette(PALETTE[1], source, repeated, mode=mode),
    )


@pytest.mark.parametrize("mode", [0, 2, 3])
@pytest.mark.parametrize("flag", [True, False], ids=["compatible", "no-compatible"])
def test_nonfinite_large_palette_c_entries_dither_every_pixel(mode, flag):
    # The compatible slot stays for B3's goldens and the frozen helper: written 1
    # when given (Consult 8 D-5), optional.
    source = image(3)
    source[0, 0, 0] = np.nan
    colors = image(3, 1, 307)
    result = np.full(source.shape, -19, np.float32)
    compatible = ct.c_int(9)
    written = ct.byref(compatible) if flag else None
    p = ct.POINTER(ct.c_float)
    arguments = (source.ctypes.data_as(p), result.ctypes.data_as(p), 7, 11, 3)
    if mode == 3:
        status = lib().cn_neighborhood_palette_riemersma(
            *arguments, colors.ctypes.data_as(p), 307, 16, 1 / 16, written
        )
    else:
        status = lib().cn_neighborhood_dither(
            *arguments, 2, mode, 2, 0, colors.ctypes.data_as(p), 307, written
        )
    assert status == 0
    assert compatible.value == (1 if flag else 9)
    bits_equal(result, native.palette_dither(source, colors, mode))


def test_nonfinite_large_palette_original_panic_and_the_linear_rule():
    source = image(3, 1, 1)
    colors = image(3, 1, 307)
    colors[:, :, 0] = np.nan
    with pytest.raises(BaseException, match="None") as raised:
        palette(PALETTE[1], source, colors, mode="NONE")
    assert type(raised.value).__name__ == "PanicException"
    bits_equal(
        palette(PALETTE[0], source, colors, mode="NONE"),
        linear_reference(source, colors),
    )


def canonical_palette(colors):
    """Independent reference: explicit input order, raw-bit uniqueness, scalar key."""
    colors = colors.reshape(-1, colors.shape[-1])
    unique = {}
    for color in colors:
        unique.setdefault(color.tobytes(), color)
    values = list(unique.values())

    def key(color):
        if len(color) == 1:
            value = color[0]
        else:
            value = (
                color[0] * color[0] * np.float32(0.2126)
                + color[1] * color[1] * np.float32(0.7152)
                + color[2] * color[2] * np.float32(0.0722)
            )
            if len(color) == 4:
                value += color[3] * np.float32(10)
        bits = value.view(np.uint32).item()
        return bits ^ (0xFFFFFFFF if bits >> 31 else 0x80000000)

    return np.array(sorted(values, key=key), dtype=np.float32)


def nearest_reference(source, colors):
    ordered = canonical_palette(colors)
    output = np.empty_like(source)
    for original, target in zip(
        source.reshape(-1, source.shape[-1]),
        output.reshape(-1, source.shape[-1]),
        strict=True,
    ):
        delta = ordered - original
        distances = np.zeros(len(ordered), np.float32)
        for c in range(source.shape[-1]):
            distances += delta[:, c] * delta[:, c]
        target[:] = ordered[np.argmin(distances)]
    return output


def linear_reference(source, colors):
    """The linear scan's rule (palettes below 300 colors, and nonfinite queries since
    Consult 8 D-5) over canonical_palette's order: the first color at the least
    distance; a NaN distance never wins, and the first color stays when its own
    distance is NaN."""
    ordered = canonical_palette(colors)
    pixels = source.reshape(-1, source.shape[-1])
    with np.errstate(all="ignore"):
        distance = np.zeros((len(pixels), len(ordered)), np.float32)
        for c in range(source.shape[-1]):
            delta = ordered[None, :, c] - pixels[:, None, c]
            distance = distance + delta * delta
    unordered = np.isnan(distance)
    least = np.where(unordered, np.float32(np.inf), distance).min(axis=1)
    index = np.where(unordered[:, 0], 0, (distance == least[:, None]).argmax(axis=1))
    return ordered[index].reshape(source.shape)


@pytest.mark.parametrize("count", [300, 301, 1024, 65536])
def test_large_grayscale_now_supported(count):
    colors = np.linspace(0, 1, count, dtype=np.float32)[None, :, None]
    source = image(1, 3, 11)
    actual = palette(PALETTE[0], source, colors, mode="NONE")
    bits_equal(actual, nearest_reference(source, colors))


def test_large_grayscale_original_panic_is_regression_evidence():
    # The real module's panic, recorded in the conformance manifest's "corrected"
    # cell; chaiNNer-C's chainner_ext corrects it as the C node path does.
    manifest = json.loads(
        (Path(__file__).with_name("chainner_ext") / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert {
        (row["class"], row["outcome"]["raise"]["message"])
        for row in manifest["cases"]
        if row["cell"] == "palette/grayscale_tree"
    } == {("corrected", "Point dimension too small - must be at least 2")}
    colors = np.linspace(0, 1, 300, dtype=np.float32)[None, :, None]
    source = image(1, 2, 3)
    expected = nearest_reference(source, colors)
    bits_equal(palette(PALETTE[0], source, colors, mode="NONE"), expected)
    bits_equal(palette(PALETTE[1], source, colors, mode="NONE"), expected)


@pytest.mark.parametrize("count", [300, 1024])
@pytest.mark.parametrize("mode", ["NONE", "RIEMERSMA", *native.DIFFUSION_IDS])
def test_large_grayscale_modes_against_original_rgb_equivalent(count, mode):
    # Replicating finite gray values into RGB avoids the original 1-D R-tree
    # bug while exercising its real diffusion and Hilbert-history arithmetic.
    source = image(1, 13, 19, seed=982)
    colors = image(1, 1, count, seed=312)
    algorithm = mode if mode in native.DIFFUSION_IDS else "FS"
    mode = "DIFFUSION" if mode in native.DIFFUSION_IDS else mode
    expected = palette(
        PALETTE[1],
        np.repeat(source, 3, axis=2),
        np.repeat(colors, 3, axis=2),
        mode=mode,
        diffusion=algorithm,
    )[:, :, :1]
    bits_equal(
        palette(PALETTE[0], source, colors, mode=mode, diffusion=algorithm), expected
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("mode", ["NONE", "DIFFUSION", "RIEMERSMA"])
def test_duplicate_colors_do_not_trigger_tree_or_change_results(channels, mode):
    source = image(channels)
    colors = image(channels, 1, 19)
    duplicate = np.tile(colors, (1, 80, 1))
    bits_equal(
        palette(PALETTE[0], source, duplicate, mode=mode),
        palette(PALETTE[1], source, duplicate, mode=mode),
    )
    bits_equal(
        palette(PALETTE[0], source, duplicate, mode=mode),
        palette(PALETTE[0], source, colors, mode=mode),
    )


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("mode", ["NONE", "DIFFUSION", "RIEMERSMA"])
def test_equal_luminance_ties_have_canonical_repeatable_results(channels, mode):
    colors = np.zeros((1, 2, channels), np.float32)
    colors[0, :, 0] = [0.5, -0.5]
    source = np.zeros((7, 11, channels), np.float32)
    actual = palette(PALETTE[0], source, colors, mode=mode)
    for _ in range(8):
        bits_equal(actual, palette(PALETTE[0], source, colors, mode=mode))
    # At the initial zero query both colors have exactly equal squared distance.
    bits_equal(actual[:1, :1], colors[:, :1])
    assert np.isin(actual[:, :, 0], [-0.5, 0.5]).all()


def test_normalized_equal_luminance_tie_is_repeatable_and_exact_nearest():
    # Captured in reference-sources-dither/tie-evidence.json. All channels are
    # within [0,1]. The original returned each color on repeated identical
    # calls; both squared distances and both luminance keys are bit-identical.
    colors = np.array(
        [[[1051940858, 1011245376, 1058377322], [1055123326, 1015242208, 1051716570]]],
        np.uint32,
    ).view(np.float32)
    source = np.array([[[1053532092, 1013354112, 1055753303]]], np.uint32).view(
        np.float32
    )
    difference = colors[0] - source[0, 0]
    distance = np.zeros(2, np.float32)
    for c in range(3):
        distance += difference[:, c] * difference[:, c]
    assert distance[0].view(np.uint32) == distance[1].view(np.uint32)
    expected = nearest_reference(source, colors)
    bits_equal(expected, colors[:, :1])
    for _ in range(32):
        bits_equal(palette(PALETTE[0], source, colors, mode="NONE"), expected)
    bits_equal(palette(PALETTE[0], source, colors[:, ::-1], mode="NONE"), colors[:, 1:])


def test_hilbert_subdivision_boundaries_all_small_rectangles():
    for height in range(1, 34):
        for width in range(1, 34):
            source = image(3, height, width, seed=height * 100 + width)
            actual = uniform(UNIFORM[0], source, colors=3, history=3)
            expected = uniform(UNIFORM[1], source, colors=3, history=3)
            np.testing.assert_array_equal(
                actual.view(np.uint32),
                expected.view(np.uint32),
                err_msg=f"Hilbert rectangle {height}x{width}",
            )


@pytest.mark.parametrize("channels", [3, 4])
def test_large_tree_exact_distance_ties_choose_canonical_color(channels):
    # A regular lattice yields many equal-distance queries. It checks tree
    # bounds and tie replacement independently of traversal or the Rust tree.
    axes = np.meshgrid(*([np.linspace(0, 1, 9, dtype=np.float32)] * 3), indexing="ij")
    colors = np.stack(axes, axis=-1).reshape(1, -1, 3)
    source = (colors[:, :127] + np.float32(1 / 16)).clip(0, 1)
    if channels == 4:
        colors = np.concatenate(
            (colors, np.ones((*colors.shape[:2], 1), np.float32)), axis=2
        )
        source = np.concatenate(
            (source, np.ones((*source.shape[:2], 1), np.float32)), axis=2
        )
    result = palette(PALETTE[0], source, colors, mode="NONE")
    bits_equal(result, nearest_reference(source, colors))
    bits_equal(result, palette(PALETTE[0], source, colors[:, ::-1], mode="NONE"))


@pytest.mark.parametrize(
    "shape", [(), (3,), (0, 3), (3, 0), (3, 2, 0), (3, 2, 2), (3, 2, 5), (2, 3, 4, 1)]
)
def test_invalid_shapes_rejected_before_c(monkeypatch, shape):
    def forbidden(*args):
        raise AssertionError("Invalid buffer dispatched to C")

    monkeypatch.setattr(lib(), "cn_neighborhood_uniform_riemersma", forbidden)
    monkeypatch.setattr(lib(), "cn_neighborhood_palette_riemersma", forbidden)
    source = np.zeros(shape, np.float32)
    with pytest.raises(ValueError):
        native.uniform_dither(source, 7, 3)
    with pytest.raises(ValueError):
        native.palette_dither(source, image(3, 1, 7), 3)


@pytest.mark.parametrize(
    "shape", [(0, 3, 3), (1, 0, 3), (2, 3, 3), (1, 3, 1), (1, 3, 4), (1, 3, 3, 1)]
)
def test_invalid_palette_rejected_before_c(monkeypatch, shape):
    def forbidden(*args):
        raise AssertionError("Invalid buffer dispatched to C")

    monkeypatch.setattr(lib(), "cn_neighborhood_palette_riemersma", forbidden)
    with pytest.raises(ValueError):
        native.palette_dither(image(), np.zeros(shape, np.float32), 3)


@pytest.mark.parametrize("dtype", [np.float64, np.int32, np.uint8, np.dtype(">f4")])
def test_dtype_contract(dtype):
    with pytest.raises(TypeError):
        native.uniform_dither(image().astype(dtype), 7, 3)
    with pytest.raises(TypeError):
        native.palette_dither(image(), image(3, 1, 7).astype(dtype), 3)


@pytest.mark.parametrize("history", [-1, 0, 1, 2.5, 0x100000000])
def test_history_bounds(history):
    with pytest.raises(ValueError):
        native.uniform_dither(image(), 7, 3, history_length=history)
    with pytest.raises(ValueError):
        native.palette_dither(image(), image(3, 1, 7), 3, history_length=history)


def test_foreign_c_boundaries():
    library = lib()
    value = ct.c_float()
    p = ct.pointer(value)
    maximum = ct.c_size_t(-1).value
    compatible = ct.c_int()
    assert (
        library.cn_neighborhood_uniform_riemersma(None, p, 1, 1, 1, 7, 16, 1 / 16) == 1
    )
    assert (
        library.cn_neighborhood_uniform_riemersma(p, p, maximum, 2, 3, 7, 16, 1 / 16)
        == 2
    )
    assert library.cn_neighborhood_uniform_riemersma(p, p, 1, 1, 2, 7, 16, 1 / 16) == 1
    assert library.cn_neighborhood_uniform_riemersma(p, p, 1, 1, 1, 7, 1, 1 / 16) == 1
    for decay in (0, -1, 1, np.nan, np.inf):
        assert (
            library.cn_neighborhood_uniform_riemersma(p, p, 1, 1, 1, 7, 16, decay) == 1
        )
    assert (
        library.cn_neighborhood_palette_riemersma(
            p, p, 1, 1, 1, None, 1, 16, 1 / 16, ct.byref(compatible)
        )
        == 1
    )
    assert (
        library.cn_neighborhood_palette_riemersma(
            p, p, 1, 1, 1, p, 0, 16, 1 / 16, ct.byref(compatible)
        )
        == 1
    )
    assert (
        library.cn_neighborhood_palette_riemersma(
            p, p, 1, 1, 1, p, maximum, 16, 1 / 16, ct.byref(compatible)
        )
        == 2
    )


def test_concurrency_repeatability_and_input_immutability():
    sources = [image(c, 35, 41, seed=c) for c in (1, 3, 4)]
    colors = [image(c, 1, 1024, seed=c + 10) for c in (1, 3, 4)]
    before = [array.copy() for array in sources + colors]

    def run(index):
        source, pal = sources[index % 3], colors[index % 3]
        return (
            uniform(UNIFORM[0], source),
            palette(PALETTE[0], source, pal),
            palette(PALETTE[0], source, pal, mode="NONE"),
            palette(PALETTE[0], source, pal, mode="DIFFUSION"),
        )

    expected = [run(index) for index in range(3)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run, range(24)))
    for index, outputs in enumerate(results):
        for result, reference in zip(outputs, expected[index % 3], strict=True):
            bits_equal(result, reference)
    for result, original in zip(sources + colors, before, strict=True):
        bits_equal(result, original)


def test_node_registrations_and_signatures_unchanged():
    for filename in ("dither.py", "dither_palette.py"):
        before = ast.parse((REFERENCE / filename).read_text(encoding="utf-8"))
        after = ast.parse((SOURCE / filename).read_text(encoding="utf-8"))
        old = next(
            node
            for node in before.body
            if isinstance(node, ast.FunctionDef) and node.name.endswith("_node")
        )
        new = next(
            node
            for node in after.body
            if isinstance(node, ast.FunctionDef) and node.name.endswith("_node")
        )
        assert ast.dump(old.args) == ast.dump(new.args)
        assert [ast.dump(item) for item in old.decorator_list] == [
            ast.dump(item) for item in new.decorator_list
        ]
        # These adapters must contain no retained Rust computational dispatch.
        for node in ast.walk(after):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "chainner_ext"


# The real chainner_ext's luminance key keeps one NaN of a color with several, measured
# on every ordered pair of NaN channels (fresh processes; equal keys follow its random
# AHashSet order): alpha's, then blue's, then red's, then green's. Ranks by channel.
UPSTREAM_NAN_RANK = {3: (1, 0, 2), 4: (1, 0, 2, 3)}


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("path", ["node", "chainner_ext"])
def test_palette_key_keeps_upstreams_nan(channels, path):
    """A holds a negative NaN in channel i and a positive one in j, B only j's: A's key
    is the negative NaN (A first) where upstream keeps i's, else it ties B's and the
    first occurrence, B, stays first. Every distance is NaN, so the lookup gives the
    sorted palette's first color."""
    import chainner_ext

    rank = UPSTREAM_NAN_RANK[channels]
    pixel = np.full((1, 1, channels), 0.5, np.float32)
    for i in range(channels):
        for j in range(channels):
            if i == j:
                continue
            a = [0x3F000000] * channels
            b = [0x3E800000] * channels
            a[i], a[j], b[j] = 0xFFC0AAAA, 0x7FC0BBBB, 0x7FC0BBBB
            words = np.array([b, a], np.uint32)
            colors = words.view(np.float32).reshape(1, 2, channels)
            if path == "node":
                out = native.palette_dither(pixel, colors, 0)
            else:
                quantization = chainner_ext.PaletteQuantization(colors)
                out = chainner_ext.quantize(pixel, quantization)
            first = words[1] if rank[i] > rank[j] else words[0]
            assert (out.reshape(channels).view(np.uint32) == first).all(), (i, j)
