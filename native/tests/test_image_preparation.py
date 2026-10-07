"""Bit-exact preparation reuse, ownership, mutation and bounded-cache tests.

Frozen references were copied from the coordinated pre-optimization snapshot.
These checks count preparation calls and validate outputs; they do not time work.
"""

from __future__ import annotations

import ast
import ctypes as ct
import hashlib
import importlib.util
import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import cv2
import numpy as np
import pytest
from PIL import ImageFont

from nodes.impl import caption, font_cache, native_gaussian, native_neighborhood
from nodes.impl import native_normal_kernel as normal
from nodes.impl import native_spectral_filter as spectral
from nodes.impl.native import lib
from nodes.impl.native_analysis import distinct_colors
from nodes.impl.native_image_setup import distinct_exceeds, exceptional
from nodes.impl.native_prepared import PreparedCache, float_state

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "backend/src"
REFERENCE = Path(__file__).with_name("reference_image_preparation")


def mask_bytes(font: ImageFont.FreeTypeFont, text: str) -> bytes:
    """The pixel values of a rendered glyph mask, row by row."""
    return bytes(font.getmask(text))  # pyright: ignore[reportArgumentType] -- Pillow's stub gives ImagingCore no __iter__, but the mask is a sequence of pixel values; tobytes() packs mode-1 bits and _new() is private API


def helper(name):
    spec = importlib.util.spec_from_file_location(
        "nodes.impl._prepared_reference_" + name, REFERENCE / (name + ".py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def node(path):
    tree = ast.parse(path.read_text("utf-8"))
    tree.body = [
        n
        for n in tree.body
        if not isinstance(n, ast.ImportFrom)
        or (
            n.level == 0
            and n.module != "nodes.groups"
            and not (n.module or "").startswith("nodes.properties")
        )
    ]
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            n.decorator_list = []
    result = types.ModuleType(path.stem)
    exec(compile(tree, str(path), "exec"), result.__dict__)
    return result


OLD_DITHER = helper("native_neighborhood")
OLD_NORMAL = helper("native_normal_kernel")
OLD_SPECTRAL = helper("native_spectral_filter")
LENS = node(SOURCE / "packages/chaiNNer_standard/image_filter/blur/lens_blur.py")
OLD_LENS = node(REFERENCE / "lens_blur.py")
PALETTE = node(
    SOURCE
    / "packages/chaiNNer_standard/image_utility/miscellaneous/palette_from_image.py"
)
OLD_PALETTE = node(REFERENCE / "palette_from_image.py")
SPLIT = node(SOURCE / "packages/chaiNNer_standard/image_channel/all/separate_rgba.py")
OLD_SPLIT = node(REFERENCE / "split_r_g_b_a.py")


def test_frozen_references_match_their_pinned_hashes():
    # The manifest pins each frozen file's bytes; native_normal_kernel.py records its
    # one ruled edit (Consult 8 D-6).
    manifest = json.loads((REFERENCE / "manifest.json").read_text("utf-8"))
    for name, entry in manifest.items():
        digest = hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest()
        assert digest == entry["sha256"], name
    assert [name for name, entry in manifest.items() if "edited" in entry] == [
        "native_normal_kernel.py"
    ]


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("limit", [0, 1, 2, 7, 32, 4096])
@pytest.mark.parametrize("pattern", ["random", "zero", "nan", "repeat"])
def test_bounded_distinct_agrees_with_full_ordered_pass(channels, limit, pattern):
    source = (
        np.random.default_rng(39).integers(0, 5, (9, 11, channels)).astype(np.float32)
    )
    if pattern == "zero":
        source[:] = 0
        source[::2] = -0.0
    elif pattern == "nan":
        source[::3, ::2, 0] = np.nan
    elif pattern == "repeat":
        source[1:] = source[0]
    source = source[::-1, ::-1]
    source.setflags(write=False)
    expected = distinct_colors(source).shape[1] > limit
    assert distinct_exceeds(source, limit) == expected


@pytest.mark.parametrize("method", ["ALL", "KMEANS", "MEDIAN_CUT"])
@pytest.mark.parametrize("pattern", ["random", "low", "zero"])
def test_palette_node_preserves_order_and_skips_unneeded_sort(
    monkeypatch, method, pattern
):
    image = np.random.default_rng(29).random((13, 17, 3), dtype=np.float32)
    if pattern == "low":
        image[:] = np.array([0.0, 0.5, 1.0], np.float32)
        image[0] = 0.25
    elif pattern == "zero":
        image[:] = -0.0
    expected = OLD_PALETTE.palette_from_image_node(
        image, getattr(OLD_PALETTE.PaletteExtractionMethod, method), 7
    )
    count = 0
    original = PALETTE.distinct_colors_palette

    def counted(value):
        nonlocal count
        count += 1
        return original(value)

    monkeypatch.setattr(PALETTE, "distinct_colors_palette", counted)
    exact(
        PALETTE.palette_from_image_node(
            image, getattr(PALETTE.PaletteExtractionMethod, method), 7
        ),
        expected,
    )
    assert count == (0 if method != "ALL" and pattern == "random" else 1)


@pytest.mark.parametrize(
    "shape", [(2, 3), (2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 7)]
)
def test_split_rgba_views_aliases_and_no_unused_allocation(monkeypatch, shape):
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = OLD_SPLIT.split_r_g_b_a_node(source)
    calls = 0
    original = np.ones

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(SPLIT.np, "ones", counted)
    actual = SPLIT.separate_rgba_node(source)
    for a, e in zip(actual, expected, strict=True):
        exact(a, e)
        assert np.shares_memory(a, source) == np.shares_memory(e, source)
    for i in range(4):
        for j in range(4):
            assert (actual[i] is actual[j]) == (expected[i] is expected[j])
    assert calls == (0 if len(shape) == 3 and shape[2] >= 4 else 1)


@pytest.mark.parametrize(
    "bits",
    [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC01234, 1, 0x80000001, 0x3F800000],
)
@pytest.mark.parametrize("position", [0, 1, 7, 31])
def test_fused_exception_predicate(bits, position):
    data = np.ones(32, np.float32)
    data.view(np.uint32)[position] = bits
    expected = not np.isfinite(data).all() or bool(
        np.any((data == 0) & np.signbit(data))
    )
    assert exceptional(data[::-1]) == expected


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("count", [1, 7, 299, 300, 317])
@pytest.mark.parametrize(
    "mode,algorithm", [(0, 0), *[(2, a) for a in range(8)], (3, 0)]
)
def test_palette_search_plan_exact_all_modes(channels, count, mode, algorithm):
    rng = np.random.default_rng(103)
    source = rng.random((7, 9, channels), dtype=np.float32)[::-1]
    colors = rng.random((1, count, channels), dtype=np.float32)
    colors[:, ::13] = -0.0
    expected = OLD_DITHER.palette_dither(source, colors, mode, algorithm=algorithm)
    for _ in range(2):
        exact(
            native_neighborhood.palette_dither(
                source, colors, mode, algorithm=algorithm
            ),
            expected,
        )


def test_palette_mutation_eviction_and_concurrent_leases():
    rng = np.random.default_rng(87)
    source = rng.random((17, 13, 4), dtype=np.float32)
    colors = rng.random((1, 313, 4), dtype=np.float32)
    cases = []
    for i in range(24):
        colors[0, 0] = i / 24
        cases.append((colors.copy(), OLD_DITHER.palette_dither(source, colors, 2)))
    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(
            pool.map(
                lambda pair: native_neighborhood.palette_dither(source, pair[0], 2),
                cases * 2,
            )
        )
    for actual, (_, expected) in zip(outputs, cases * 2, strict=True):
        exact(actual, expected)
    entries, size = native_neighborhood._palette_plans.retained()
    assert entries <= 16 and size <= 4 * 1024 * 1024


@pytest.mark.parametrize("count", [7, 317])
@pytest.mark.parametrize("where", ["image", "palette"])
def test_palette_nonfinite_contract(count, where):
    # Consult 8 D-5: 300+ unique colors with a nonfinite value dither by the linear
    # rule instead of refusing; the frozen helper reads the kernel's compatible
    # flag, now always 1, so it returns the same result.
    rng = np.random.default_rng(819)
    source = rng.random((3, 5, 3), dtype=np.float32)
    palette = rng.random((1, count, 3), dtype=np.float32)
    (source if where == "image" else palette).flat[0] = np.nan
    exact(
        native_neighborhood.palette_dither(source, palette, 0),
        OLD_DITHER.palette_dither(source, palette, 0),
    )


@pytest.mark.parametrize("radius,count", [(1, 1), (3, 5), (9, 6), (37, 2)])
def test_lens_preparation_per_call_and_exact(monkeypatch, radius, count):
    image = np.random.default_rng(98).random((7, 11, 3), dtype=np.float32)
    expected = OLD_LENS.lens_blur(image, radius, count, 1.7)
    calls = 0
    original = LENS.normalize_kernels

    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(LENS, "normalize_kernels", counted)
    exact(LENS.lens_blur(image, radius, count, 1.7), expected)
    exact(LENS.lens_blur(image, radius, count, 1.7), expected)
    # The shipped node normalizes its kernels on every call; it never consults
    # the module's prepared-plan cache.
    assert calls == 2


@pytest.mark.parametrize("shape", [(1, 13), (17, 1), (9, 11), (53, 61)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spectral_reuse_exact_and_mutable_coefficients(shape, dtype):
    image = np.random.default_rng(38).random((13, 17, 3), dtype=np.float32)
    kernel = np.linspace(-0.5, 0.75, np.prod(shape), dtype=dtype).reshape(shape)
    for delta in [0, 0.125, -0.125]:
        kernel.flat[0] += delta
        expected = OLD_SPECTRAL.filter2d(image, kernel)
        exact(spectral.filter2d(image, kernel), expected)
        exact(spectral.filter2d(image, kernel), expected)


def test_spectral_prepares_once_and_separates_dispatch(monkeypatch):
    spectral._spectra.clear()
    image = np.ones((13, 17, 3), np.float32)
    kernel = np.ones((1, 17), np.float32) / 17
    calls = 0
    original = spectral._lib.cn_spectral_kernel

    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(spectral._lib, "cn_spectral_kernel", counted)
    ipp = cv2.ipp.useIPP()
    try:
        spectral.filter2d(image, kernel)
        spectral.filter2d(image, kernel)
        assert calls == 1
        cv2.ipp.setUseIPP(not ipp)
        spectral.filter2d(image, kernel)
        assert calls == 2
    finally:
        cv2.ipp.setUseIPP(ipp)


@pytest.mark.parametrize("where", ["image", "kernel"])
@pytest.mark.parametrize("bits", [0x80000000, 0x7FC01234, 0x7F800000, 0xFF800000])
def test_spectral_cached_nonfinite_payload_and_zero_sign(where, bits):
    image = np.linspace(-0.25, 1, 13 * 17, dtype=np.float32).reshape(13, 17)
    kernel = np.ones((1, 13), np.float32) / 13
    (image if where == "image" else kernel).view(np.uint32).flat[3] = bits
    with np.errstate(all="ignore"):
        expected = OLD_SPECTRAL.filter2d(image, kernel)
        exact(spectral.filter2d(image, kernel), expected)
        exact(spectral.filter2d(image, kernel), expected)


def test_normal_public_arrays_are_independently_writable():
    parameters = [(1.0, 0.2), (3.0, 0.8), (7.0, 0.1)]
    expected = OLD_NORMAL.gaussian_kernel(parameters)
    a = normal.gaussian_kernel(parameters)
    exact(a, expected)
    a[:] = 12
    exact(normal.gaussian_kernel(parameters), expected)
    expected_pair = OLD_NORMAL.kernel_pair(expected, normalize=True)
    pair = normal.kernel_pair(expected, normalize=True)
    for actual, old in zip(pair, expected_pair, strict=True):
        exact(actual, old)
        actual[:] = -13
    for actual, old in zip(
        normal.kernel_pair(expected, normalize=True), expected_pair, strict=True
    ):
        exact(actual, old)


def test_normal_plans_reuse_construction_and_key_parameter_contents(monkeypatch):
    normal._gaussian_plans.clear()
    normal._pair_plans.clear()
    api = normal._api()
    counts = {"cn_normal_gaussian_kernel": 0, "cn_normal_kernel_pair": 0}

    def wrap(name, original):
        def counted(*args):
            counts[name] += 1
            return original(*args)

        return counted

    for name in counts:
        monkeypatch.setattr(api, name, wrap(name, getattr(api, name)))
    parameters = [(1.0, 0.2), (3.0, 0.8)]
    for _ in range(3):
        kernel = normal.gaussian_kernel(parameters)
        normal.kernel_pair(kernel, normalize=True)
    assert set(counts.values()) == {1}
    parameters[0] = (1.0, 0.7)
    normal.kernel_pair(normal.gaussian_kernel(parameters), normalize=True)
    assert set(counts.values()) == {2}


def test_palette_plan_reuses_setup_and_invalidates_on_mutation(monkeypatch):
    native_neighborhood._palette_plans.clear()
    calls = 0
    api = native_neighborhood._lib
    original = api.cn_neighborhood_palette_create

    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(api, "cn_neighborhood_palette_create", counted)
    image = np.ones((7, 5, 3), np.float32)
    palette = np.linspace(0, 1, 21, dtype=np.float32).reshape(1, 7, 3)
    for mode in (0, 2, 3):
        native_neighborhood.palette_dither(image, palette, mode)
    assert calls == 1
    palette[0, 0] = 0.17
    native_neighborhood.palette_dither(image, palette, 0)
    assert calls == 2


def test_gaussian_cache_hits_and_bound():
    info = lib().cn_gaussian_cache_info
    info.argtypes = [ct.POINTER(ct.c_uint64), ct.c_int]
    info.restype = ct.c_int
    counters = (ct.c_uint64 * 4)()
    assert info(counters, 1) == 0
    for _ in range(3):
        exact(
            native_gaussian.kernel(31, 1.7),
            cv2.getGaussianKernel(31, 1.7, cv2.CV_32F).ravel(),
        )
    assert info(counters, 0) == 0
    assert tuple(counters[:2]) == (2, 1)
    for size in range(1, 80):
        native_gaussian.kernel(size, 2.7)
    assert info(counters, 0) == 0
    assert counters[2] <= 32 and counters[3] <= 4 * 1024 * 1024


def test_prepared_cache_eviction_active_reference_and_oversize():
    cache = PreparedCache[bytearray](10, 2)
    a = cache.get("a", lambda: (bytearray(b"aaaa"), 4))
    assert cache.get("a", lambda: pytest.fail("cache hit rebuilt")) is a
    cache.get("b", lambda: (bytearray(b"bbbb"), 4))
    cache.get("c", lambda: (bytearray(b"cccc"), 4))
    assert a == b"aaaa" and cache.retained() == (2, 8)
    cache.get("large", lambda: (bytearray(11), 11))
    assert cache.retained() == (2, 8)
    cache.clear()
    assert cache.retained() == (0, 0)


def test_font_exclusive_leases_reuse_and_bounds(monkeypatch):
    font_cache.clear()
    path = str(SOURCE / "fonts/Roboto-Light.ttf")
    calls = 0
    original = ImageFont.truetype

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", counted)
    with font_cache.font_lease(path, 17) as first:
        expected = mask_bytes(first, "first e\u0301\nlast")
    with font_cache.font_lease(path, 17) as second:
        assert second is first
        assert mask_bytes(second, "first e\u0301\nlast") == expected
    assert calls == 1
    barrier = Barrier(4)

    def use(_):
        with font_cache.font_lease(path, 17) as font:
            barrier.wait()
            result = (id(font), mask_bytes(font, "concurrent"))
            barrier.wait()
            return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(use, range(4)))
    assert len({result[0] for result in results}) == 4
    assert len({result[1] for result in results}) == 1
    for size in range(1, 40):
        with font_cache.font_lease(path, size):
            pass
    entries, payload = font_cache.retained()
    assert entries <= 16 and payload <= 8 * 1024 * 1024
    with font_cache.font_lease(path, 513):
        pass
    assert font_cache.retained() == (entries, payload)


def test_float_key_excludes_sticky_status():
    before = float_state()
    with np.errstate(all="ignore"):
        np.divide(np.ones(4, np.float32), np.zeros(4, np.float32))
    assert float_state() == before


@pytest.mark.skipif(sys.platform != "win32", reason="Public Windows CRT controls")
def test_fp_control_keys_and_denormal_predicates():
    control = ct.CDLL("ucrtbase.dll")._controlfp_s  # public CRT API
    control.argtypes = [ct.POINTER(ct.c_uint), ct.c_uint, ct.c_uint]
    control.restype = ct.c_int
    saved, observed = ct.c_uint(), ct.c_uint()
    assert control(ct.byref(saved), 0, 0) == 0
    mask = 0x03000300
    settings = [0, 0x100, 0x200, 0x300, 0x01000000, 0x02000000, 0x03000000]
    info = lib().cn_gaussian_cache_info
    info.argtypes = [ct.POINTER(ct.c_uint64), ct.c_int]
    info.restype = ct.c_int
    counters = (ct.c_uint64 * 4)()
    expected = {}
    try:
        for setting in settings:
            assert control(ct.byref(observed), setting, mask) == 0
            assert info(counters, 1) == 0
            expected[setting] = native_gaussian.kernel(19, 2.7)
        assert info(counters, 1) == 0
        keys = set()
        for setting in settings * 2:
            assert control(ct.byref(observed), setting, mask) == 0
            keys.add(float_state())
            exact(native_gaussian.kernel(19, 2.7), expected[setting])
            values = np.array(
                [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF], np.uint32
            ).view(np.float32)
            for value in values.reshape(-1, 1):
                want = not np.isfinite(value).all() or bool(
                    np.any((value == 0) & np.signbit(value))
                )
                assert exceptional(value) == want
            image = values.reshape(2, 3, 1)
            count = distinct_colors(image).shape[1]
            for limit in range(7):
                assert distinct_exceeds(image, limit) == (count > limit)
        assert len(keys) == len(settings)
        assert info(counters, 0) == 0
        assert tuple(counters[:2]) == (len(settings), len(settings))
    finally:
        assert control(ct.byref(observed), saved.value, mask) == 0


def test_coefficient_cache_concurrent_eviction():
    cases = [(size, sigma) for size in range(3, 66, 2) for sigma in (0.3, 1.7)]
    expected = [cv2.getGaussianKernel(n, s, cv2.CV_64F).ravel() for n, s in cases]
    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(
            pool.map(lambda pair: native_gaussian.kernel(*pair, double=True), cases * 3)
        )
    for actual, old in zip(outputs, expected * 3, strict=True):
        exact(actual, old)


def test_default_caption_reuses_faces_and_keeps_exact_raster(monkeypatch):
    font_cache.clear()
    old = helper("caption")
    monkeypatch.setattr(
        sys.modules["__main__"], "__file__", str(SOURCE / "run.py"), raising=False
    )
    image = np.zeros((7, 131, 4), np.float32)
    expected = old.add_caption(image, "small\ncaption", 31, old.CaptionPosition.TOP)
    calls = 0
    original = ImageFont.truetype

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", counted)
    exact(
        caption.add_caption(image, "small\ncaption", 31, caption.CaptionPosition.TOP),
        expected,
    )
    first = calls
    assert first > 0
    exact(
        caption.add_caption(image, "small\ncaption", 31, caption.CaptionPosition.TOP),
        expected,
    )
    assert calls == first


def test_native_preparation_invalid_buffers_do_not_write():
    dll = lib()
    predicate = dll.cn_image_distinct_exceeds
    predicate.argtypes = [ct.c_void_p] + [ct.c_size_t] * 3 + [ct.POINTER(ct.c_int)]
    predicate.restype = ct.c_int
    sentinel = ct.c_int(77)
    values = np.ones(4, np.float32)
    assert predicate(None, 1, 1, 2, ct.byref(sentinel)) == 1
    assert predicate(values.ctypes.data, 1, 1, 4097, ct.byref(sentinel)) == 1
    assert (
        predicate(values.ctypes.data, ct.c_size_t(-1).value, 4, 3, ct.byref(sentinel))
        == 2
    )
    assert sentinel.value == 77


def test_font_file_change_invalidates_face(tmp_path):
    import shutil

    destination = tmp_path / "font.ttf"
    shutil.copyfile(SOURCE / "fonts/Roboto/Roboto-Regular.ttf", destination)
    with font_cache.font_lease(str(destination), 23) as before:
        old = mask_bytes(before, "Wim")
    shutil.copyfile(SOURCE / "fonts/Roboto/Roboto-Bold.ttf", destination)
    with font_cache.font_lease(str(destination), 23) as after:
        assert after is not before
        assert mask_bytes(after, "Wim") != old
