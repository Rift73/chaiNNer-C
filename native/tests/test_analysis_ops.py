"""CPU differential tests; references are frozen, unmodified upstream bodies.

No timing measurements, GPU models, or application/profile access is needed.
"""

from __future__ import annotations

import ast
import ctypes as ct
import importlib.util
import re
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from nodes.impl import cas
from nodes.impl import native_analysis as native
from nodes.impl.color_transfer import mean_std
from nodes.impl.dithering import palette
from nodes.impl.image_utils import calculate_ssim
from nodes.impl.native import lib

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "backend/src"
REFERENCE = Path(__file__).with_name("reference_analysis")
PATHS = {
    "image_statistics": "image_utility/miscellaneous/image_statistics.py",
    "image_metrics": "image_utility/miscellaneous/image_metrics.py",
    "apply_palette": "image_utility/miscellaneous/apply_palette.py",
    "palette_from_image": "image_utility/miscellaneous/palette_from_image.py",
    "high_pass": "image_filter/miscellaneous/high_pass.py",
    "unsharp_mask": "image_filter/sharpen/unsharp_mask.py",
    "high_boost_filter": "image_filter/sharpen/high_boost_filter.py",
    "average_color_fix": "image_filter/correction/average_color_fix.py",
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


def load_helper(relative, name):
    spec = importlib.util.spec_from_file_location(name, REFERENCE / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULES = {
    name: tuple(
        load_body(root / "packages/chaiNNer_standard" / relative)
        for root in (SOURCE, REFERENCE)
    )
    for name, relative in PATHS.items()
}
SPLIT_TRANSPARENCY = load_body(
    SOURCE
    / "packages/chaiNNer_standard/image_channel/transparency/split_transparency.py"
)
CROP = load_body(SOURCE / "packages/chaiNNer_standard/image_dimension/crop/crop.py")
ORIGINAL_CAS = load_helper("nodes/impl/cas.py", "nodes.impl.reference_cas")
ORIGINAL_PALETTE = load_helper(
    "nodes/impl/dithering/palette.py", "nodes.impl.dithering.reference_palette"
)
ORIGINAL_TRANSFER = load_helper(
    "nodes/impl/color_transfer/mean_std.py",
    "nodes.impl.color_transfer.reference_mean_std",
)
# Baseline node functions must retain their baseline helpers, even after the
# production helpers have been replaced. Otherwise comparisons could test C twice.
MODULES["high_boost_filter"][1].__dict__.update(cas_mix=ORIGINAL_CAS.cas_mix)
for _helper in ("distinct_colors_palette", "kmeans_palette", "median_cut_palette"):
    setattr(
        MODULES["palette_from_image"][1], _helper, getattr(ORIGINAL_PALETTE, _helper)
    )


def source(channels=3, layout="contiguous", height=31, width=27, seed=143):
    shape = (height, width) if channels == 1 else (height, width, channels)
    image = np.random.default_rng(seed).random(shape, dtype=np.float32)
    image.flat[:4] = [0, 1, 0.5, 0.01]
    if layout == "strided":
        image = image[::-1, ::2]
    elif layout == "transposed":
        image = image.swapaxes(0, 1)
    elif layout == "readonly":
        image.flags.writeable = False
    elif layout == "unaligned":
        buffer = np.ndarray(
            shape, dtype=np.float32, buffer=bytearray(image.nbytes + 1), offset=1
        )
        buffer[:] = image
        image = buffer
    return image


def arrays(actual, expected, *, rtol=0, atol=0):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("size", [1, 2, 7, 8, 9, 127, 128, 129, 255, 1001, 65539])
@pytest.mark.parametrize("percentile", [0, 0.01, 25, 50, 93.17, 100])
def test_statistics(size, percentile):
    image = np.random.default_rng(721).random((1, size), dtype=np.float32)
    actual, original = MODULES["image_statistics"]
    assert actual.image_statistics_node(
        image, percentile
    ) == original.image_statistics_node(image, percentile)


def test_statistics_of_cropped_split_transparency_view():
    # R2 I-1: Load RGBA -> Split Transparency -> Crop -> Image Statistics hands the
    # node an (H, W', 3) view with strides (16W, 16, 4), which NumPy 2.5.3 reduces in
    # buffer blocks that restart at every row. The executor's channels=1 input check
    # rejects three channels, so the node function is called directly.
    actual, original = MODULES["image_statistics"]
    rng = np.random.default_rng(5)
    for trial in range(200):
        rgb, _ = SPLIT_TRANSPARENCY.split_transparency_node(
            rng.random((5, 4000, 4), dtype=np.float32)
        )
        view = CROP.crop_node(rgb, CROP.CropMode.EDGES, 0, 10, 0, 10, 0, 0, 0)
        assert view.strides == (64000, 16, 4)
        assert actual.image_statistics_node(
            view, 50.0
        ) == original.image_statistics_node(view, 50.0), trial


@pytest.mark.parametrize(
    "layout", ["contiguous", "strided", "transposed", "readonly", "unaligned"]
)
@pytest.mark.parametrize("percentile", [0, 33.33, 50, 100])
def test_statistics_layout(layout, percentile):
    image = source(1, layout)
    saved = image.copy()
    actual, original = MODULES["image_statistics"]
    assert actual.image_statistics_node(
        image, percentile
    ) == original.image_statistics_node(image, percentile)
    arrays(image, saved)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize(
    "layout", ["contiguous", "strided", "transposed", "readonly", "unaligned"]
)
def test_metrics(channels, layout):
    image = source(channels, layout, height=35, width=37)
    other = source(channels, layout, height=35, width=37, seed=900)
    actual, original = MODULES["image_metrics"]
    expected = original.image_metrics_node(image, other)
    result = actual.image_metrics_node(image, other)
    np.testing.assert_array_equal(result, expected)
    # No mathematical correction of the original natural-log PSNR definition.
    assert result[:2] == expected[:2]


@pytest.mark.parametrize("size", [1, 9, 10, 11, 19, 151])
def test_ssim_empty_crop_and_constants(size):
    image = np.full((size, size), 0.5, np.float32)
    with np.errstate(all="ignore"):
        expected = calculate_ssim(image, image)
        actual = native.calculate_ssim(image, image)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("levels", [1, 2, 7, 255, 256, 257, 65536, 65537])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("singleton", [False, True])
def test_apply_palette(levels, channels, singleton):
    image = source(1, "strided", height=7, width=11)
    if singleton:
        image = image[..., None]
    lookup = source(channels, "readonly", height=2, width=levels)
    saved, saved_palette = image.copy(), lookup.copy()
    actual, original = MODULES["apply_palette"]
    arrays(
        actual.apply_palette_node(image, lookup),
        original.apply_palette_node(image, lookup),
    )
    arrays(image, saved)
    arrays(lookup, saved_palette)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize(
    "layout", ["contiguous", "strided", "transposed", "readonly", "unaligned"]
)
def test_distinct_palette(channels, layout):
    image = np.round(source(channels, layout) * 4) / np.float32(4)
    if layout == "readonly":
        image.flags.writeable = False
    saved = image.copy()
    arrays(
        palette.distinct_colors_palette(image),
        ORIGINAL_PALETTE.distinct_colors_palette(image),
    )
    arrays(image, saved)


@pytest.mark.parametrize("method", ["ALL", "KMEANS", "MEDIAN_CUT"])
@pytest.mark.parametrize("size", [2, 8, 1024])
def test_palette_node_all_paths(method, size):
    image = np.round(source(3, height=12, width=9) * 3) / np.float32(3)
    actual, original = MODULES["palette_from_image"]
    arrays(
        actual.palette_from_image_node(
            image, getattr(actual.PaletteExtractionMethod, method), size
        ),
        original.palette_from_image_node(
            image, getattr(original.PaletteExtractionMethod, method), size
        ),
    )


def test_distinct_nonfinite_rows():
    image = np.array(
        [[[0, 1, np.nan], [0, 1, np.nan], [-0.0, 1, 2], [0, 1, 2], [np.inf, 0, 2]]],
        np.float32,
    )
    arrays(
        palette.distinct_colors_palette(image),
        ORIGINAL_PALETTE.distinct_colors_palette(image),
    )


# NumPy's NaN, x86's default NaN, two other quiet payloads and a signalling one.
TIE_NANS = np.array(
    [0x7FC00000, 0xFFC00000, 0x7FC12345, 0xFFEDCBA9, 0x7F800001], np.uint32
).view(np.float32)


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("pixels", [1, 2, 3, 16, 17, 18, 19, 40, 300, 5000, 60000])
@pytest.mark.parametrize("seed", range(3))
def test_distinct_rows_are_numpy_uniques_bits(channels, pixels, seed):
    # Consult 11 D-17.2: np.unique(axis=0) sorts the rows as structured voids with
    # NumPy's unstable npy_quicksort, then keeps the first row of each run, so which
    # of +0 and -0 it keeps, and the order of NaN rows that compare equal, are its
    # sort's. Few values make long runs of ties, at sizes either side of the
    # insertion-sort cutoff.
    rng = np.random.default_rng([seed, channels, pixels])
    values = np.concatenate(
        [np.array([0, -0.0, 0.5, 1, -np.inf], np.float32), TIE_NANS]
    )
    weights = np.array([6, 6, 1, 1, 1, 1, 1, 1, 1, 1], np.float64)
    image = rng.choice(values, (1, pixels, channels), p=weights / weights.sum())
    expected = np.unique(image.reshape(-1, channels), axis=0)
    actual = native.distinct_colors(image)
    assert actual.shape == (1, *expected.shape)
    assert actual.tobytes() == expected.tobytes()


def quicksort_killer(n):
    """McIlroy's adversary ("A Killer Adversary for Quicksort", 1999) run on NumPy's
    object-array sort, the generic npy_quicksort that void rows take too: each rank is
    frozen only when a comparison needs it. Returns the ranks and the compared pairs
    (each OBJECT_compare starts with one __lt__ call), in call order."""
    gas = n
    ranks = [gas] * n
    state = {"solid": 0, "candidate": -1}
    calls = []

    class Item:
        def __init__(self, tag):
            self.tag = tag

        def __lt__(self, other):
            x, y = self.tag, other.tag
            calls.append((x, y))
            if ranks[x] == gas and ranks[y] == gas:
                ranks[x if x == state["candidate"] else y] = state["solid"]
                state["solid"] += 1
            if ranks[x] == gas:
                state["candidate"] = x
            elif ranks[y] == gas:
                state["candidate"] = y
            return ranks[x] < ranks[y]

        def __gt__(self, other):
            return ranks[self.tag] > ranks[other.tag]

    items = np.empty(n, object)
    items[:] = [Item(tag) for tag in range(n)]
    items.sort()
    for tag in range(n):
        if ranks[tag] == gas:
            ranks[tag] = state["solid"]
            state["solid"] += 1
    return np.array(ranks), calls


@pytest.mark.parametrize("pixels", [2000, 20000])
def test_distinct_rows_follow_numpys_heapsort_fallback(pixels):
    # Consult 11 D-17.2: on the killer ranks every partition is degenerate, so the
    # sort runs out of its 2 log2(n) depth and heapsorts the rest. Pairs of adjacent
    # ranks whose every direct comparison had the larger rank first become ties (as
    # `< 0` was false, no comparison changes) told apart by a second channel of +0
    # and -0, or of two NaN payloads: which of each np.unique keeps, or puts first,
    # is the fallback's.
    ranks, calls = quicksort_killer(pixels)
    assert len(calls) > 3 * pixels * (pixels.bit_length() - 1)
    larger_first = {}
    for x, y in calls:
        larger_first.setdefault(frozenset((x, y)), set()).add(bool(ranks[x] > ranks[y]))
    order = np.argsort(ranks)
    rows = np.zeros((pixels, 2), np.float32)
    rows[:, 0] = ranks
    pairs = 0
    for rank in range(0, pixels - 1, 2):
        x, y = int(order[rank]), int(order[rank + 1])
        if larger_first.get(frozenset((x, y)), {True}) == {True}:
            rows[y, 0] = rows[x, 0]
            rows[[x, y], 1] = TIE_NANS[:2] if pairs % 2 else (0.0, -0.0)
            pairs += 1
    assert pairs > pixels // 10
    expected = np.unique(rows, axis=0)
    assert native.distinct_colors(rows[None]).tobytes() == expected.tobytes()


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("mode", ["CUSTOM", "GAUSSIAN"])
@pytest.mark.parametrize("contrast", [0, 1, 2.7, 100])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "readonly"])
def test_high_pass(channels, mode, contrast, layout):
    image = source(channels, layout)
    other = (
        image[..., :3] * np.float32(0.7) if channels == 4 else image * np.float32(0.7)
    )
    saved = image.copy()
    actual, original = MODULES["high_pass"]
    arrays(
        actual.high_pass_node(
            image, getattr(actual.BlurMode, mode), 1.3, other, contrast
        ),
        original.high_pass_node(
            image, getattr(original.BlurMode, mode), 1.3, other, contrast
        ),
    )
    arrays(image, saved)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("threshold", [0, 0.1, 11, 100])
@pytest.mark.parametrize("amount", [0, 1, 2.7, 100])
def test_unsharp(channels, threshold, amount):
    image = source(channels, "readonly")
    actual, original = MODULES["unsharp_mask"]
    arrays(
        actual.unsharp_mask_node(image, 1.3, amount, threshold),
        original.unsharp_mask_node(image, 1.3, amount, threshold),
    )


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("bias", [1, 1.3, 2, 3])
def test_cas_mask_mix(channels, bias):
    image = source(channels, "strided")
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    sharpened = image * np.float32(1.1)
    arrays(
        cas.create_cas_mask(image, kernel, bias),
        ORIGINAL_CAS.create_cas_mask(image, kernel, bias),
    )
    arrays(
        cas.cas_mix(image, sharpened, kernel, bias),
        ORIGINAL_CAS.cas_mix(image, sharpened, kernel, bias),
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("kernel", ["NORMAL", "STRONG"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_high_boost(channels, kernel, adaptive):
    image = source(channels, "readonly")
    actual, original = MODULES["high_boost_filter"]
    arrays(
        actual.high_boost_filter_node(
            image, getattr(actual.KernelType, kernel), 2.1, adaptive, 2.3
        ),
        original.high_boost_filter_node(
            image, getattr(original.KernelType, kernel), 2.1, adaptive, 2.3
        ),
    )


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("reference_channels", [3, 4])
@pytest.mark.parametrize("scale", [12.5, 50, 100])
def test_average_fix(channels, reference_channels, scale):
    image = source(channels, height=31, width=29)
    reference = source(reference_channels, height=17, width=13, seed=93)
    if channels == 4:
        image[:10, :10, 3] = 0
    if reference_channels == 4:
        reference[:4, :7, 3] = 0
    image.flags.writeable = False
    reference.flags.writeable = False
    saved, saved_ref = image.copy(), reference.copy()
    actual, original = MODULES["average_color_fix"]
    arrays(
        actual.average_color_fix_node(image, reference, scale),
        original.average_color_fix_node(image, reference, scale),
    )
    arrays(image, saved)
    arrays(reference, saved_ref)


@pytest.mark.parametrize(
    "size", [1, 7, 8, 127, 128, 129, 1001, 8191, 8192, 8193, 16383, 16384, 16385, 65539]
)
def test_channel_statistics(size):
    image = source(3, height=1, width=size).reshape(-1, 3)
    arrays(
        native.channel_stats(image), np.asarray(ORIGINAL_TRANSFER.image_stats(image))
    )


def test_empty_mean_and_channel_statistics_nan_bits():
    # _mean's 0/0 runs in float64 and gives x86's default NaN, 0xFFC00000 as float32.
    empty = np.empty((0, 3), np.float32)
    with np.errstate(all="ignore"):
        expected = np.asarray(ORIGINAL_TRANSFER.image_stats(empty), np.float32)
        expected_mean = np.mean(empty[:, 0])
    assert native.mean(empty[:, 0]).view(np.uint32) == expected_mean.view(np.uint32)
    np.testing.assert_array_equal(
        native.channel_stats(empty).view(np.uint32), expected.view(np.uint32)
    )


def test_mirrors_spell_out_nan_bits():
    # The Windows SDK's NAN macro is 0xFFC00000 up to 10.0.19041 and 0x7FC00000 from
    # 10.0.22621, so a mirror that returns it would follow the build's SDK.
    native_root = ROOT / "native"
    for path in [
        *(native_root / "src").iterdir(),
        *(native_root / "include").iterdir(),
    ]:
        if path.suffix in (".c", ".cpp", ".h", ".hpp"):
            code = path.read_text(encoding="utf-8", errors="replace")
            assert not re.search(r"\bNAN\b", code), path.name


@pytest.mark.parametrize("size", [127, 129, 8191, 8192, 8193, 16383, 16385, 65539])
@pytest.mark.parametrize("seed", [19, 271, 337, 1993])
def test_unrounded_mean_reduction_boundaries(size, seed):
    rng = np.random.default_rng(seed)
    image = rng.normal(size=(1, size)).astype(np.float32)
    image[:, ::7] *= np.float32(1000)
    assert native.mean(image) == np.mean(image)


def test_mean_divides_in_float64_above_2_24():
    # _methods._mean divides by an np.intp count, in float64: a float32 division
    # by the count agrees only up to 2**24 elements (it differs on this input).
    size = 2**24 + 1
    image = np.random.default_rng(size).random((1, size), dtype=np.float32)
    assert native.mean(image) == np.mean(image)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_statistics_nonfinite(value):
    image = source(1)
    image[0, 0] = value
    with np.errstate(all="ignore"):
        expected = (
            np.min(image),
            np.max(image),
            np.mean(image),
            np.percentile(image, 50),
        )
        actual = native.statistics(image, 50)
    np.testing.assert_allclose(actual, expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize(
    "pattern", ["sorted", "reversed", "equal", "duplicates", "organ_pipe", "sawtooth"]
)
@pytest.mark.parametrize("size", [1, 16, 17, 127, 8193, 65539])
@pytest.mark.parametrize("percentile", [0, 0.01, 25, 50, 93.17, 100])
def test_percentile_adversarial_partitions(pattern, size, percentile):
    ordered = np.arange(size, dtype=np.float32)
    if pattern == "reversed":
        ordered = ordered[::-1]
    elif pattern == "equal":
        ordered.fill(7)
    elif pattern == "duplicates":
        ordered %= 3
    elif pattern == "organ_pipe":
        ordered = np.minimum(ordered, ordered[::-1])
    elif pattern == "sawtooth":
        ordered %= 127
    image = ordered.reshape(1, -1)
    saved = image.copy()
    result = native.statistics(image, percentile)
    assert result[0] == np.min(image)
    assert result[1] == np.max(image)
    assert result[2] == np.mean(image)
    assert result[3] == np.percentile(image, percentile)
    arrays(image, saved)


@pytest.mark.parametrize("colorspace", ["LAB", "RGB"])
@pytest.mark.parametrize("overflow", ["CLIP", "SCALE"])
@pytest.mark.parametrize("reciprocal", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "transposed", "readonly"])
def test_color_transfer(colorspace, overflow, reciprocal, layout):
    image, reference = source(3, layout), source(3, layout, seed=993)
    valid, ref_valid = image[..., 0] > 0.2, reference[..., 0] > 0.3
    saved, saved_ref = image.copy(), reference.copy()
    results = [
        module.mean_std_transfer(
            image,
            reference,
            getattr(module.TransferColorSpace, colorspace),
            getattr(module.OverflowMethod, overflow),
            valid,
            ref_valid,
            reciprocal,
        )
        for module in (mean_std, ORIGINAL_TRANSFER)
    ]
    arrays(*results)
    arrays(image, saved)
    arrays(reference, saved_ref)


@pytest.mark.parametrize("overflow", ["CLIP", "SCALE"])
@pytest.mark.parametrize("reciprocal", [False, True])
@pytest.mark.parametrize("empty_mask", [False, True])
def test_color_transfer_zero_variance_and_empty_mask(overflow, reciprocal, empty_mask):
    image = np.full((17, 19, 3), 0.25, dtype=np.float32)
    reference = np.full_like(image, 0.75)
    valid = np.full(image.shape[:2], not empty_mask)
    results = []
    for module in (mean_std, ORIGINAL_TRANSFER):
        if empty_mask and overflow == "SCALE":
            with pytest.raises(ValueError), np.errstate(all="ignore"):
                module.mean_std_transfer(
                    image,
                    reference,
                    module.TransferColorSpace.RGB,
                    getattr(module.OverflowMethod, overflow),
                    valid,
                    valid,
                    reciprocal,
                )
        else:
            with np.errstate(all="ignore"):
                results.append(
                    module.mean_std_transfer(
                        image,
                        reference,
                        module.TransferColorSpace.RGB,
                        getattr(module.OverflowMethod, overflow),
                        valid,
                        valid,
                        reciprocal,
                    )
                )
    if results:
        arrays(*results)


@pytest.mark.parametrize("shape", [(3,), (0, 4), (2, 3, 0), (2, 3, 4, 5)])
def test_malformed_image_buffers(shape):
    image = np.empty(shape, dtype=np.float32)
    for call in (
        lambda: native.statistics(image, 50),
        lambda: native.is_grayscale(image, 0),
        lambda: native.binary(image, image, 0),
        lambda: native.distinct_colors(image),
        lambda: native.apply_palette(image, source(3)),
    ):
        with pytest.raises(ValueError):
            call()


def test_bad_secondary_buffers_and_dtype():
    image = source(3)
    with pytest.raises(ValueError):
        native.binary(image, image[:-1], 0)
    with pytest.raises(TypeError):
        native.distinct_colors(image.astype(np.float64))
    with pytest.raises(ValueError):
        native.cas_mix(image, image, np.ones((1, 1), dtype=np.float32))
    with pytest.raises(ValueError):
        native.correction(image, image, np.ones((1, 1), np.float32), None, add=False)
    with pytest.raises(ValueError):
        native.color_transfer(
            image,
            image,
            np.ones((1, 1), bool),
            np.ones(image.shape[:2], bool),
            (0, 1, 0, 1, 0, 1),
            reciprocal=True,
            scale=False,
        )
    # NumPy's nonfinite-to-uint8 cast selects index zero in the original node.
    np.testing.assert_array_equal(
        native.apply_palette(np.array([[np.nan]], np.float32), image), image[:1, :1]
    )


def test_c_invalid_counts_do_not_write():
    dll = lib()
    sentinel = ct.c_float(123)
    p = ct.pointer(sentinel)
    assert dll.cn_analysis_binary(p, p, p, ct.c_size_t(-1).value, 0, 1, 0) == 2
    assert dll.cn_analysis_palette(p, p, p, 1, 0, 3) == 1
    assert dll.cn_analysis_correction(p, p, None, None, p, p, 1, 0) == 1
    assert sentinel.value == 123


def test_concurrent_determinism_and_immutable_inputs():
    image = source(3, "readonly", height=279, width=257)
    lookup = source(3, "readonly", height=1, width=256)
    gray = source(1, "readonly", height=279, width=257)

    def operations():
        return (
            native.statistics(gray, 41.7),
            native.binary(image, image, 1, 2.1, 0.02),
            native.apply_palette(gray, lookup),
            native.distinct_colors(image[:15]),
        )

    expected = operations()
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: operations(), range(24)))
    for actual in results:
        assert actual[0] == expected[0]
        for a, b in zip(actual[1:], expected[1:], strict=True):
            arrays(a, b)
