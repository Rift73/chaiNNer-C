"""Exact differential tests for format boundaries and model-independent CPU work."""

from __future__ import annotations

import ctypes as ct
import importlib.util
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from nodes.impl import image_utils, native_buffers
from nodes.impl.native_buffers import _api
from nodes.impl.upscale import convenient_upscale as upscale
from nodes.impl.upscale import exact_split as split
from nodes.impl.upscale import tile_blending as tiles

REFERENCE = Path(__file__).with_name("reference_buffers")


def load(name, package):
    spec = importlib.util.spec_from_file_location(
        package + "._reference_" + name, REFERENCE / (name + ".py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves its module while evaluating postponed annotations.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ORIGINAL = load("image_utils", "nodes.impl")
ORIGINAL_TILES = load("tile_blending", "nodes.impl.upscale")
INSTALLED_UPSCALE = load("installed_convenient_upscale", "nodes.impl.upscale")


@pytest.mark.parametrize("shape", [(7, 11, 1), (31, 53, 3), (81, 77, 4)])
@pytest.mark.parametrize("scale", [1, 2, 4])
def test_full_tiled_cpu_pipeline(shape, scale):
    original_split = types.ModuleType("nodes.impl.upscale._reference_exact_split")
    original_split.__package__ = "nodes.impl.upscale"
    import sys

    sys.modules[original_split.__name__] = original_split
    frozen_split = (
        Path(__file__).with_name("reference_tiling") / "installed/exact_split.py"
    )
    exec(frozen_split.read_text(), original_split.__dict__)
    original_split.__dict__.update(
        TileBlender=ORIGINAL_TILES.TileBlender,
        TileOverlap=ORIGINAL_TILES.TileOverlap,
        BlendDirection=ORIGINAL_TILES.BlendDirection,
    )
    image = np.random.default_rng(712).random(shape, dtype=np.float32)

    def callback(tile, _region):
        return np.repeat(np.repeat(tile, scale, axis=0), scale, axis=1)

    expected = original_split.exact_split(image, (32, 32), callback, 4)
    actual = split.exact_split(image, (32, 32), callback, 4)
    np.testing.assert_array_equal(actual, expected)


def source(dtype, layout):
    dtype = np.dtype(dtype)
    shape = (19, 23, 4)
    rng = np.random.default_rng(412)
    if dtype.kind in "ui":
        array = (
            rng.integers(0, 256, size=np.prod(shape) * dtype.itemsize, dtype=np.uint8)
            .view(dtype)
            .reshape(shape)
        )
        info = np.iinfo(dtype)
        array.flat[:6] = [info.min, info.max, 0, 1, info.max - 1, info.min + 1]
    elif dtype.kind == "b":
        array = rng.random(shape) < 0.3
    else:
        array = rng.uniform(-2, 2, shape).astype(dtype)
        array.flat[:8] = [np.nan, np.inf, -np.inf, -0.0, 0.0, 0.5, 1.0, 1e-6]
    if layout == "strided":
        array = array[::2, ::-2]
    elif layout == "transpose":
        array = array.swapaxes(0, 1)
    elif layout == "readonly":
        array.setflags(write=False)
    elif layout == "unaligned":
        a = np.ndarray(shape, dtype=dtype, buffer=bytearray(array.nbytes + 1), offset=1)
        a[:] = array
        array = a
    return array


@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
        "float64",
        "float16",
        "uint8",
        "uint16",
        "int8",
        "int16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "bool",
        ">u2",
    ],
)
@pytest.mark.parametrize(
    "layout", ["plain", "strided", "transpose", "readonly", "unaligned"]
)
@pytest.mark.parametrize(
    "name,normalized",
    [
        ("normalize", False),
        ("to_uint8", False),
        ("to_uint16", False),
        ("to_uint8", True),
        ("to_uint16", True),
    ],
)
def test_pixel_conversion(dtype, layout, name, normalized):
    image = source(dtype, layout)
    before = image.copy()
    args = () if name == "normalize" else (normalized,)
    with np.errstate(all="ignore"):
        actual = getattr(image_utils, name)(image, *args)
        expected = getattr(ORIGINAL, name)(image, *args)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, before)
    assert not np.shares_memory(actual, image)


@pytest.mark.parametrize("bits", [8, 16])
def test_quantization_rounding_and_extremes(bits):
    scale = (1 << bits) - 1
    image = (np.arange(-scale, scale * 2 + 1, dtype=np.float32) + 0.5) / scale
    variants = [
        image,
        np.nextafter(image, np.inf),
        np.nextafter(image, -np.inf),
        np.array(
            [
                np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                np.nan,
                np.inf,
                10001,
                -10001,
            ]
        ),
    ]
    for value in variants:
        with np.errstate(all="ignore"):
            expected = getattr(ORIGINAL, f"to_uint{bits}")(value, True)
            actual = getattr(image_utils, f"to_uint{bits}")(value, True)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(0, 2, 3), (2, 0), (0,), ()])
def test_empty_scalar_conversion(shape):
    value = np.zeros(shape, np.float32)
    for name in ("normalize", "to_uint8", "to_uint16"):
        actual = getattr(image_utils, name)(value)
        expected = getattr(ORIGINAL, name)(value)
        assert type(actual) is type(expected)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("direction", ["X", "Y"])
@pytest.mark.parametrize("channels", [1, 3, 4, 7])
@pytest.mark.parametrize("blend_fn", ["sin_blend_fn", "half_sin_blend_fn"])
@pytest.mark.parametrize("overlap", [0, 1, 4])
def test_tile_blending(direction, channels, blend_fn, overlap):
    image = np.random.default_rng(19).random((48, 48, channels), dtype=np.float32)

    def execute(module):
        blender = module.TileBlender(
            48,
            48,
            channels,
            module.BlendDirection[direction],
            getattr(module, blend_fn),
        )
        a, b = (
            (image[:, : 24 + overlap], image[:, 24 - overlap :])
            if direction == "X"
            else (image[: 24 + overlap], image[24 - overlap :])
        )
        blender.add_tile(a, module.TileOverlap(0, overlap))
        blender.add_tile(b, module.TileOverlap(overlap, 0))
        return blender.get_result(), blender

    expected, _ = execute(ORIGINAL_TILES)
    actual, blender = execute(tiles)
    np.testing.assert_array_equal(actual, expected)
    if overlap:
        # The cache is a view of a compact vector, not three full-image repeats.
        assert blender._last_blend is not None
        assert not blender._last_blend.flags.owndata
        assert 0 in blender._last_blend.strides


@pytest.mark.parametrize("channels", [1, 3, 4, 7])
@pytest.mark.parametrize("layout", ["plain", "strided", "readonly"])
def test_alpha_buffers(channels, layout):
    image = (
        np.random.default_rng(93)
        .uniform(-0.5, 1.5, (17, 29, channels))
        .astype(np.float32)
    )
    image.flat[:3] = [np.nan, np.inf, -np.inf]
    if layout == "strided":
        image = image[::2, ::-2]
    elif layout == "readonly":
        image.setflags(write=False)
    before = image.copy()
    with np.errstate(all="ignore"):
        np.testing.assert_array_equal(
            upscale.denoise_and_flatten_alpha(image),
            INSTALLED_UPSCALE.denoise_and_flatten_alpha(image),
        )
        if channels == 4:
            for a, b in zip(
                upscale.with_black_and_white_backgrounds(image),
                INSTALLED_UPSCALE.with_black_and_white_backgrounds(image),
                strict=True,
            ):
                np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("constant", [False, True, "nan"])
@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("model_channels", [1, 3])
def test_convenient_upscale_cpu_callback(constant, separate, model_channels):
    image = np.random.default_rng(16).random((11, 19, 4), dtype=np.float32)
    if constant:
        image[:, :, 3] = np.nan if constant == "nan" else 0.731

    def callback(value):
        return np.repeat(np.repeat(value, 2, axis=0), 2, axis=1)

    with np.errstate(all="ignore"):
        if model_channels == 1 and separate and constant is False:
            # The upstream helper itself rejects a 2D grayscale model output
            # in this branch. Preserve that failure rather than inventing shape semantics.
            for module in (INSTALLED_UPSCALE, upscale):
                with pytest.raises(np.exceptions.AxisError):
                    module.convenient_upscale(
                        image, model_channels, model_channels, callback, separate
                    )
            return
        expected = INSTALLED_UPSCALE.convenient_upscale(
            image, model_channels, model_channels, callback, separate
        )
        actual = upscale.convenient_upscale(
            image, model_channels, model_channels, callback, separate
        )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(1, 6, 3), (1, 1, 3), (1, 6, 1)])
@pytest.mark.parametrize("layout", ["broadcast", "reversed"])
def test_single_row_tile_strides(shape, layout):
    image = np.random.default_rng(11).random(shape, dtype=np.float32)
    a = np.broadcast_to(image[0], shape) if layout == "broadcast" else image[::-1]
    b = np.float32(1) - image
    weights = np.broadcast_to(np.float32(0.375), shape)
    before = image.copy()
    expected = ORIGINAL_TILES._fast_mix(a, b, weights)
    actual = tiles._fast_mix(a, b, weights)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, before)


def test_concurrent_bulk_calls():
    image = np.random.default_rng(17).random((257, 517, 4), dtype=np.float32)
    weights = np.broadcast_to(
        np.linspace(0, 1, 517, dtype=np.float32)[None, :, None], image.shape
    )
    funcs = [
        lambda: image_utils.to_uint16(image),
        lambda: native_buffers.tile_mix(image, image[::-1], weights),
        lambda: upscale.denoise_and_flatten_alpha(image),
        lambda: upscale.with_black_and_white_backgrounds(image)[0],
    ]
    expected = [f() for f in funcs]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: funcs[i % len(funcs)](), range(24)))
    for i, result in enumerate(results):
        np.testing.assert_array_equal(result, expected[i % len(funcs)])


def test_buffer_boundaries():
    image = np.zeros((2, 3, 4), np.float32)
    with pytest.raises(ValueError):
        native_buffers.converted_pixels(image, 32)
    with pytest.raises(ValueError):
        native_buffers.tile_mix(image, image, image[:, :, 0])
    with pytest.raises(ValueError):
        native_buffers.alpha_backgrounds(image[:, :, :3])
    with pytest.raises(ValueError):
        native_buffers.flatten_alpha(np.empty((2, 3, 8), np.float32))
    api = native_buffers._api()  # malformed direct ABI
    assert api.cn_pixels_convert(None, None, 0, 0, 0, 0) == 1
    assert (
        api.cn_pixels_convert(
            image.ctypes.data, image.ctypes.data, ct.c_size_t(-1), 0, 0, 0
        )
        == 2
    )
    assert (
        api.cn_tile_mix(
            native_buffers.ptr(image),
            native_buffers.ptr(image),
            native_buffers.ptr(image),
            native_buffers.ptr(image),
            2,
            3,
            4,
            3,
            12,
            12,
        )
        == 1
    )


class LibraryInFlight:
    """Stands in for the native library in _api(): every attribute asked of it (a
    getattr miss on the real CDLL) briefly counts the threads inside one, in `peak`."""

    def __init__(self):
        self.lock = threading.Lock()
        self.inside = self.peak = 0

    def __getattr__(self, name):
        with self.lock:
            self.inside += 1
            self.peak = max(self.peak, self.inside)
        time.sleep(0.01)
        with self.lock:
            self.inside -= 1
        return types.SimpleNamespace(name=name)


def test_api_configures_the_library_one_thread_at_a_time(monkeypatch):
    # SP3b audit A8: lru_cache does not serialize first calls, and two at once would
    # each set argtypes on the function objects their getattr misses create, so
    # _api()'s body runs under a lock: one caller configures the library at a time.
    library = LibraryInFlight()
    monkeypatch.setattr(native_buffers, "lib", lambda: library)
    start = threading.Barrier(2, timeout=5)

    def first_call(_):
        start.wait()
        return _api()

    _api.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(first_call, range(2)))
    finally:
        _api.cache_clear()  # the next caller configures the real library
    assert results == [library, library]
    assert library.peak == 1
