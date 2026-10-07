"""Exact comparisons with frozen nodes and installed OpenCV/Rust kernels."""

from __future__ import annotations

import ast
import ctypes as ct
import json
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from nodes.impl import native_neighborhood as native
from nodes.impl.native import lib
from nodes.impl.native_versions import CV_CN_MAX

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "backend/src/packages/chaiNNer_standard/image_filter"
REFERENCE = Path(__file__).with_name("reference_neighborhood")
PATHS = {
    "median_blur": "blur/median_blur.py",
    "dither": "quantize/dither.py",
    "dither_palette": "quantize/dither_palette.py",
}


def load_body(path):
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


MODULES = {
    name: (load_body(SOURCE / relative), load_body(REFERENCE / relative))
    for name, relative in PATHS.items()
}


def image(channels=3, height=17, width=23, layout="normal", seed=735):
    shape = (height, width) if channels == 1 else (height, width, channels)
    source = np.random.default_rng(seed).random(shape, dtype=np.float32)
    if layout == "strided":
        source = source[::-1, ::2]
    elif layout == "transposed":
        source = source.swapaxes(0, 1)
    elif layout == "readonly":
        source.flags.writeable = False
    elif layout == "unaligned":
        other = np.ndarray(
            shape, np.float32, buffer=bytearray(source.nbytes + 1), offset=1
        )
        other[:] = source
        source = other
    elif layout == "singleton":
        source = source.reshape(height, width, 1) if channels == 1 else source
    return source


def equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("radius", [0, 1, 2, 3, 4, 7, 19, 127, 128, 1000])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("shape", [(1, 1), (1, 8), (7, 1), (6, 9), (9, 6), (33, 37)])
def test_median(radius, channels, shape):
    source = image(channels, *shape)
    saved = source.copy()
    actual, original = MODULES["median_blur"]
    if radius == 1000:
        with pytest.raises(cv2.error, match="k < 16"):
            original.median_blur_node(source, radius)
        # Wide-window histogram overflow is an explicitly corrected defect.
        # Independent weighted-histogram regressions cover the corrected values.
        result = actual.median_blur_node(source, radius)
        assert result.dtype == np.uint8
        assert result.shape == source.shape
        equal(source, saved)
        return
    equal(
        actual.median_blur_node(source, radius),
        original.median_blur_node(source, radius),
    )
    equal(source, saved)


@pytest.mark.parametrize("radius", [1, 2, 3, 14])
@pytest.mark.parametrize(
    "layout", ["strided", "transposed", "readonly", "unaligned", "singleton"]
)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_median_foreign(radius, layout, channels):
    source = image(channels, layout=layout)
    saved = source.copy()
    actual, original = MODULES["median_blur"]
    equal(
        actual.median_blur_node(source, radius),
        original.median_blur_node(source, radius),
    )
    equal(source, saved)


@pytest.mark.parametrize(
    "pattern", ["constant", "ascending", "descending", "checkerboard", "impulse"]
)
@pytest.mark.parametrize("radius", [1, 2, 3, 15])
def test_median_adversarial(pattern, radius):
    source = image(1, 19, 71)
    if pattern == "constant":
        source.fill(0.5)
    elif pattern == "ascending":
        source = np.linspace(0, 1, source.size, dtype=np.float32).reshape(source.shape)
    elif pattern == "descending":
        source = np.linspace(1, 0, source.size, dtype=np.float32).reshape(source.shape)
    elif pattern == "checkerboard":
        source = (np.indices(source.shape).sum(axis=0) % 2).astype(np.float32)
    else:
        source.fill(0)
        source[9, 35] = 1
    actual, original = MODULES["median_blur"]
    equal(
        actual.median_blur_node(source, radius),
        original.median_blur_node(source, radius),
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("radius", [1, 2])
def test_median_nonfinite_native_parity(value, radius):
    source = image()
    source[8, 7, 1] = value
    actual, original = MODULES["median_blur"]
    equal(
        actual.median_blur_node(source, radius),
        original.median_blur_node(source, radius),
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("radius", [1, 2])
def test_median_signed_zero_bits(channels, radius):
    rng = np.random.default_rng(19)
    shape = (7, 9) if channels == 1 else (7, 9, channels)
    source = np.array([-0.0, 0.0, 0.25], np.float32)[rng.integers(0, 3, size=shape)]
    saved = source.copy()
    actual, original = MODULES["median_blur"]
    equal(
        actual.median_blur_node(source, radius).view(np.uint32),
        original.median_blur_node(source, radius).view(np.uint32),
    )
    equal(source.view(np.uint32), saved.view(np.uint32))


def call_uniform(
    module,
    source,
    colors,
    mode,
    bayer="BAYER_16",
    diffusion="FLOYD_STEINBERG",
    history=16,
):
    return module.dither_node(
        source,
        colors,
        getattr(module.UniformDitherAlgorithm, mode),
        getattr(module.ThresholdMap, bayer),
        getattr(module.ErrorDiffusionMap, diffusion),
        history,
    )


@pytest.mark.parametrize("mode", ["NONE", "ORDERED", "DIFFUSION", "RIEMERSMA"])
@pytest.mark.parametrize("colors", [2, 3, 7, 16, 256])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_uniform_modes(mode, colors, channels):
    source = image(channels)
    saved = source.copy()
    actual, original = MODULES["dither"]
    equal(
        call_uniform(actual, source, colors, mode),
        call_uniform(original, source, colors, mode),
    )
    equal(source, saved)


DIFFUSIONS = [
    "FLOYD_STEINBERG",
    "JARVIS_ET_AL",
    "STUCKI",
    "ATKINSON",
    "BURKES",
    "SIERRA",
    "TWO_ROW_SIERRA",
    "SIERRA_LITE",
]


@pytest.mark.parametrize("algorithm", DIFFUSIONS)
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1), (37, 43)])
def test_uniform_diffusion(algorithm, channels, shape):
    source = image(channels, *shape)
    actual, original = MODULES["dither"]
    equal(
        call_uniform(actual, source, 3, "DIFFUSION", diffusion=algorithm),
        call_uniform(original, source, 3, "DIFFUSION", diffusion=algorithm),
    )


@pytest.mark.parametrize("bayer", ["BAYER_2", "BAYER_4", "BAYER_8", "BAYER_16"])
@pytest.mark.parametrize("colors", [2, 3, 8, 259])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_ordered_maps(bayer, colors, channels):
    source = image(channels, 19, 33)
    actual, original = MODULES["dither"]
    equal(
        call_uniform(actual, source, colors, "ORDERED", bayer),
        call_uniform(original, source, colors, "ORDERED", bayer),
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("mode", ["NONE", "ORDERED", "DIFFUSION"])
@pytest.mark.parametrize(
    "layout", ["strided", "transposed", "readonly", "unaligned", "singleton"]
)
def test_uniform_foreign(channels, mode, layout):
    source = image(channels, layout=layout)
    saved = source.copy()
    actual, original = MODULES["dither"]
    equal(
        call_uniform(actual, source, 7, mode), call_uniform(original, source, 7, mode)
    )
    equal(source, saved)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("mode", ["NONE", "ORDERED", "DIFFUSION"])
def test_uniform_rounding_and_range(channels, mode):
    values = np.array([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5], np.float32)
    source = np.repeat(values[:, None, None], channels, axis=2)
    actual, original = MODULES["dither"]
    equal(
        call_uniform(actual, source, 3, mode), call_uniform(original, source, 3, mode)
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("mode", ["NONE", "ORDERED", "DIFFUSION"])
@pytest.mark.parametrize("colors", [2, 3, 0xFFFFFFFF])
def test_uniform_nonfinite_and_large_palette_factor(channels, mode, colors):
    values = np.array([np.nan, np.inf, -np.inf, 0, 0.25, 0.5, 0.75, 1], np.float32)
    source = np.tile(values[None, :, None], (3, 1, channels))
    actual, original = MODULES["dither"]
    equal(
        call_uniform(actual, source, colors, mode),
        call_uniform(original, source, colors, mode),
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("algorithm", DIFFUSIONS)
def test_uniform_parallel_diffusion(channels, algorithm):
    source = image(channels, 137, 131, layout="readonly", seed=883)
    actual, original = MODULES["dither"]
    equal(
        call_uniform(actual, source, 7, "DIFFUSION", diffusion=algorithm),
        call_uniform(original, source, 7, "DIFFUSION", diffusion=algorithm),
    )


def recorded_grayscale_panics():
    """The real chainner_ext 0.3.10's panics on 300+ grayscale palettes, as recorded."""
    manifest = json.loads(
        (Path(__file__).with_name("chainner_ext") / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        row["outcome"]["raise"]["message"]
        for row in manifest["cases"]
        if row["cell"] == "palette/grayscale_tree"
    }


def call_palette(
    module, source, palette, mode="DIFFUSION", diffusion="FLOYD_STEINBERG"
):
    return module.dither_palette_node(
        source,
        palette,
        getattr(module.PaletteDitherAlgorithm, mode),
        getattr(module.ErrorDiffusionMap, diffusion),
        16,
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("mode", ["NONE", "DIFFUSION", "RIEMERSMA"])
@pytest.mark.parametrize("count", [1, 2, 8, 299, 300])
def test_palette_modes(channels, mode, count):
    source = image(channels, 7, 11)
    palette = image(channels, 2, count, seed=751)
    actual, original = MODULES["dither_palette"]
    if channels == 1 and count >= 300:
        # Upstream's R-tree panics on one-dimensional palettes, as the real module's
        # record in the conformance manifest's "corrected" cell shows. The complete
        # C palette implementation and chaiNNer-C's chainner_ext both correct it.
        assert recorded_grayscale_panics() == {
            "Point dimension too small - must be at least 2"
        }
        result = call_palette(actual, source, palette, mode)
        assert result.shape == (*source.shape[:2], 1)
        assert np.isfinite(result).all()
        equal(result, call_palette(actual, source, palette, mode))
        equal(result, call_palette(original, source, palette, mode))
        return
    equal(
        call_palette(actual, source, palette, mode),
        call_palette(original, source, palette, mode),
    )


@pytest.mark.parametrize("algorithm", DIFFUSIONS)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_palette_diffusion(algorithm, channels):
    source = image(channels, 37, 43)
    palette = image(channels, 1, 13, seed=77)
    actual, original = MODULES["dither_palette"]
    equal(
        call_palette(actual, source, palette, diffusion=algorithm),
        call_palette(original, source, palette, diffusion=algorithm),
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("mode", ["NONE", "DIFFUSION"])
@pytest.mark.parametrize(
    "layout", ["strided", "transposed", "readonly", "unaligned", "singleton"]
)
def test_palette_foreign(channels, mode, layout):
    source = image(channels, layout=layout)
    palette = image(channels, 1, 19, layout="unaligned", seed=852)[:, ::-2]
    saved = source.copy()
    saved_palette = palette.copy()
    actual, original = MODULES["dither_palette"]
    equal(
        call_palette(actual, source, palette, mode),
        call_palette(original, source, palette, mode),
    )
    equal(source, saved)
    equal(palette, saved_palette)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_palette_duplicate_and_distance_ties(channels):
    source = np.full((9, 13, channels), 0.5, np.float32)
    palette = np.tile(
        np.array([0, 1, 0, 1], np.float32)[None, :, None], (1, 1, channels)
    )
    actual, original = MODULES["dither_palette"]
    equal(
        call_palette(actual, source, palette, "NONE"),
        call_palette(original, source, palette, "NONE"),
    )
    equal(
        call_palette(actual, source, palette), call_palette(original, source, palette)
    )


def test_palette_signed_zero_order_is_total_and_native():
    # Rust total_cmp distinguishes -0 from +0. The complete C implementation
    # therefore handles this palette directly and preserves the selected sign.
    palette = np.array([[0.0, -0.0]], np.float32)
    result = native.palette_dither(image(1), palette, 0)
    assert np.all(result.view(np.uint32) == np.uint32(0x80000000))


@pytest.mark.parametrize("shape", [(), (7,), (0, 4), (4, 0), (2, 3, 0), (2, 3, 4, 1)])
def test_reject_malformed_images(shape):
    source = np.zeros(shape, np.float32)
    for operation in (
        lambda: native.median(source, 1),
        lambda: native.uniform_dither(source, 2, 0),
        lambda: native.palette_dither(source, image(3, 1, 4), 0),
    ):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize("channels", [2, 5, 7, CV_CN_MAX, CV_CN_MAX + 1])
@pytest.mark.parametrize("radius", [1, 2, 3])
def test_median_unrestricted_channel_contract(channels, radius):
    source = image(channels, 5, 7)
    actual, original = MODULES["median_blur"]
    # OpenCV 5.0.0 caps a 2-D Mat at CV_CN_MAX channels (4.8: 512).
    if radius > 2 or channels > CV_CN_MAX:
        for module in (original, actual):
            with pytest.raises(cv2.error):
                module.median_blur_node(source, radius)
    else:
        equal(
            actual.median_blur_node(source, radius),
            original.median_blur_node(source, radius),
        )


@pytest.mark.parametrize("channels", [2, 5])
def test_dither_rejects_unsupported_channels(channels):
    source = image(channels)
    with pytest.raises(ValueError):
        native.uniform_dither(source, 3, 0)
    with pytest.raises(ValueError):
        native.palette_dither(source, image(channels, 1, 4), 0)


@pytest.mark.parametrize("dtype", [np.float64, np.int32, np.uint8, np.dtype(">f4")])
def test_reject_foreign_dtypes(dtype):
    source = np.zeros((2, 3, 3), dtype)
    with pytest.raises(TypeError):
        native.uniform_dither(source, 2, 0)
    with pytest.raises(TypeError):
        native.median(source, 1)


@pytest.mark.parametrize("radius", [-1, 0, 1001, 1.5])
def test_reject_bad_radius(radius):
    with pytest.raises(ValueError):
        native.median(image(), radius)


@pytest.mark.parametrize("colors", [-1, 0, 1, 0x100000000, 2.5])
def test_reject_bad_colors(colors):
    with pytest.raises(ValueError):
        native.uniform_dither(image(), colors, 0)


def test_checked_c_boundaries():
    library = lib()
    scalar = ct.c_float(0)
    p = ct.pointer(scalar)
    compatible = ct.c_int()
    maximum = ct.c_size_t(-1).value
    assert library.cn_neighborhood_median_f32(None, p, 1, 1, 1, 1, 8) == 1
    assert library.cn_neighborhood_median_f32(p, p, maximum, 2, 4, 1, 8) == 2
    assert library.cn_neighborhood_median_u8(p, p, 1, 1, 1, 1001) == 1
    assert library.cn_neighborhood_median_u8(p, p, maximum, 2, 4, 3) == 2
    assert (
        library.cn_neighborhood_dither(
            p, p, maximum, 2, 4, 3, 2, 2, 0, None, 0, ct.byref(compatible)
        )
        == 2
    )
    assert (
        library.cn_neighborhood_dither(
            p, p, 1, 1, 1, 3, 2, 2, 8, None, 0, ct.byref(compatible)
        )
        == 1
    )


def test_concurrent_requests_and_parallel_pixels():
    sources = [image(3, 181, 193, seed=seed) for seed in range(4)]
    saved = [source.copy() for source in sources]
    actual_median, original_median = MODULES["median_blur"]
    actual_dither, original_dither = MODULES["dither"]
    expected = [
        (
            original_median.median_blur_node(source, 3),
            original_median.median_blur_node(source, 2),
            call_uniform(original_dither, source, 7, "ORDERED"),
            call_uniform(original_dither, source, 7, "DIFFUSION"),
        )
        for source in sources
    ]

    def run(index):
        source = sources[index % len(sources)]
        return (
            actual_median.median_blur_node(source, 3),
            actual_median.median_blur_node(source, 2),
            call_uniform(actual_dither, source, 7, "ORDERED"),
            call_uniform(actual_dither, source, 7, "DIFFUSION"),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run, range(16)))
    for index, result in enumerate(results):
        for actual, reference in zip(
            result, expected[index % len(sources)], strict=True
        ):
            equal(actual, reference)
    for source, before in zip(sources, saved, strict=True):
        equal(source, before)
