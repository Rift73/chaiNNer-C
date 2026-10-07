"""Frozen installed shared upscaler image arithmetic and host-layout oracles."""

from __future__ import annotations

import ctypes as ct
import importlib.util
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from test_buffers import ORIGINAL as OLD_IMAGE_UTILS

from nodes.impl import native_framework_shared as native
from nodes.impl.image_op import clipped
from nodes.impl.upscale import convenient_upscale as integration
from nodes.impl.upscale import tile_blending as tiles

REFERENCE = Path(__file__).with_name("reference_framework_shared")


def load(name: str, package: str):
    specification = importlib.util.spec_from_file_location(
        package + "._frozen_framework_" + name, REFERENCE / (name + ".py")
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


OLD_OP = load("image_op", "nodes.impl")
OLD = load("tile_blending", "nodes.impl.upscale")
OLD_UPSCALE = load("convenient_upscale", "nodes.impl.upscale")
OLD_UPSCALE.__dict__.update(clipped=OLD_OP.clipped)
OLD_UPSCALE.__dict__.update(as_target_channels=OLD_IMAGE_UTILS.as_target_channels)


def exact(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    good = ~np.isnan(expected)
    kind = np.uint32 if expected.itemsize == 4 else np.uint64
    np.testing.assert_array_equal(actual.view(kind)[good], expected.view(kind)[good])


def layout(image: np.ndarray, name: str) -> np.ndarray:
    if name == "fortran":
        return np.asfortranarray(image)
    if name == "reverse":
        return image[::-1, ::-1]
    if name == "unaligned":
        result = np.ndarray(image.shape, image.dtype, bytearray(image.nbytes + 1), 1)
        result[:] = image
        return result
    if name == "readonly":
        image.flags.writeable = False
    return image


def recorded(function):
    with np.errstate(all="warn"), warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        result = function()
    return result, [str(item.message) for item in messages]


@pytest.mark.parametrize("shape", [(0, 3), (1, 1), (7, 9), (1, 7, 3), (5, 7, 4)])
@pytest.mark.parametrize(
    "kind", ["ordinary", "fortran", "reverse", "unaligned", "readonly"]
)
@pytest.mark.parametrize("operation", ["clip", "sin", "half"])
def test_shared_curves(shape, kind, operation):
    source = np.random.default_rng(634).uniform(-3, 3, shape).astype(np.float32)
    if source.size:
        source.flat[0] = -0.0
        source.flat[-1] = np.nan
    source = layout(source, kind)
    original = {
        "clip": lambda image: np.clip(image, 0, 1),
        "sin": OLD.sin_blend_fn,
        "half": OLD.half_sin_blend_fn,
    }[operation]
    converted = {
        "clip": native.clip_image,
        "sin": tiles.sin_blend_fn,
        "half": tiles.half_sin_blend_fn,
    }[operation]
    before = source.copy()
    expected, original_messages = recorded(lambda: original(source))
    actual, actual_messages = recorded(lambda: converted(source))
    exact(actual, expected)
    assert actual.strides == expected.strides
    assert original_messages == actual_messages
    exact(source, before)


@pytest.mark.parametrize("sign", [0, 0x80000000])
def test_sine_curve_quadrant_rounding(sign):
    # NumPy 2.5.3's MSVC build fuses the sine quadrant's multiply and add (one
    # FMA). Every 61st float32 below 2^15 (19.5 M values) puts about 120
    # arguments where that rounding differs from a multiply, then an add.
    words = np.arange(0, 0x47000000, 61, dtype=np.uint32) | np.uint32(sign)
    source = words.view(np.float32)
    expected, original_messages = recorded(lambda: OLD.sin_blend_fn(source))
    actual, actual_messages = recorded(lambda: tiles.sin_blend_fn(source))
    exact(actual, expected)
    assert original_messages == actual_messages


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 7, 8, 9, 16, 17, 33, 128, 513, 8193])
@pytest.mark.parametrize("half", [False, True])
def test_blend_weights(size, half):
    function = OLD.half_sin_blend_fn if half else OLD.sin_blend_fn
    expected, original_messages = recorded(
        lambda: function(np.arange(size, dtype=np.float32) / (size - 1))
    )
    actual, actual_messages = recorded(lambda: native.blend_weights(size, half=half))
    exact(actual, expected)
    assert original_messages == actual_messages


@pytest.mark.parametrize(
    "function", [native.clip_image, tiles.sin_blend_fn, tiles.half_sin_blend_fn]
)
def test_exceptional_curves(function):
    image = np.array(
        [
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
            1e-40,
            -1e-40,
            1e20,
            -1e20,
            np.finfo(np.float32).max,
        ],
        np.float32,
    ).reshape(2, 5)
    old = (
        np.clip
        if function == native.clip_image
        else OLD.sin_blend_fn
        if function == tiles.sin_blend_fn
        else OLD.half_sin_blend_fn
    )
    expected, old_warnings = recorded(
        lambda: old(image, 0, 1) if old == np.clip else old(image)
    )
    actual, new_warnings = recorded(lambda: function(image))
    exact(actual, expected)
    assert new_warnings == old_warnings


@pytest.mark.parametrize("shape", [(1, 1, 3), (3, 7, 3), (7, 1, 3)])
@pytest.mark.parametrize("white_kind", ["ordinary", "fortran", "reverse"])
@pytest.mark.parametrize("black_kind", ["ordinary", "fortran", "unaligned"])
def test_recover_alpha(shape, white_kind, black_kind):
    white = layout(
        np.random.default_rng(63).random(shape, dtype=np.float32), white_kind
    )
    black = layout(
        np.random.default_rng(64).random(shape, dtype=np.float32), black_kind
    )
    expected = 1 - (white - black)
    actual = native.recover_alpha(white, black)
    exact(actual, expected)
    assert actual.strides == expected.strides


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 7, 3), (7, 1, 3), (3, 5, 3)])
@pytest.mark.parametrize("rgb_kind", ["ordinary", "fortran", "reverse"])
@pytest.mark.parametrize(
    "alpha_kind", ["scalar", "ordinary", "fortran", "unaligned", "readonly"]
)
def test_assemble_alpha(shape, rgb_kind, alpha_kind):
    rgb = layout(np.random.default_rng(2).random(shape, dtype=np.float32), rgb_kind)
    alpha = (
        -0.0
        if alpha_kind == "scalar"
        else layout(
            np.random.default_rng(3).random(shape[:2], dtype=np.float32), alpha_kind
        )
    )
    expected = np.dstack(
        (
            rgb,
            np.full(shape[:2], alpha, np.float32) if alpha_kind == "scalar" else alpha,
        )
    )
    actual = native.assemble_alpha(rgb, alpha)
    exact(actual, expected)
    assert actual.strides == expected.strides


@pytest.mark.parametrize(
    "kind", ["ordinary", "fortran", "reverse", "unaligned", "readonly"]
)
@pytest.mark.parametrize("overlap", [False, True])
def test_copy_into(kind, overlap):
    image = layout(np.arange(9 * 11 * 3, dtype=np.float32).reshape(9, 11, 3), kind)
    source = image[1:, 1:, :]
    expected = np.array(image, copy=True, order="K")
    actual = np.array(image, copy=True, order="K")
    expected[:-1, :-1] = expected[1:, 1:] if overlap else source
    native.copy_into(actual[:-1, :-1], actual[1:, 1:] if overlap else source)
    exact(actual, expected)


@pytest.mark.parametrize("direction", ["X", "Y"])
@pytest.mark.parametrize("channels", [1, 3, 4, 7])
@pytest.mark.parametrize("overlap", [1, 2, 4])
@pytest.mark.parametrize("half", [False, True])
def test_full_tile_assembly(direction, channels, overlap, half):
    def run(module):
        horizontal = direction == "X"
        blender = module.TileBlender(
            32 if horizontal else 7,
            7 if horizontal else 32,
            channels,
            getattr(module.BlendDirection, direction),
            blend_fn=module.half_sin_blend_fn if half else module.sin_blend_fn,
        )
        shape = (
            (7, 16 + overlap, channels) if horizontal else (16 + overlap, 7, channels)
        )
        first = np.random.default_rng(14).random(shape, dtype=np.float32)
        second = np.random.default_rng(15).random(shape, dtype=np.float32)
        blender.add_tile(first, module.TileOverlap(0, overlap))
        blender.add_tile(second, module.TileOverlap(overlap, 0))
        return blender.get_result()

    exact(run(tiles), run(OLD))


@pytest.mark.parametrize("in_channels", [1, 3, 4])
@pytest.mark.parametrize("model_channels", [1, 3, 4])
@pytest.mark.parametrize("alpha_mode", ["constant", "varying", "separate"])
@pytest.mark.parametrize("clip", [False, True])
def test_convenient_image_pipeline(in_channels, model_channels, alpha_mode, clip):
    image = np.random.default_rng(845).random((5, 7, in_channels), dtype=np.float32)
    if in_channels == 4 and alpha_mode == "constant":
        image[:, :, 3] = 0.6

    def process(image):
        if image.ndim == 2:
            image = image[:, :, None]
        return np.repeat(np.repeat(image, 2, 0), 2, 1)

    expected = OLD_UPSCALE.convenient_upscale(
        image,
        model_channels,
        model_channels,
        process,
        alpha_mode == "separate",
        clip,
    )
    actual = integration.convenient_upscale(
        image,
        model_channels,
        model_channels,
        process,
        alpha_mode == "separate",
        clip,
    )
    exact(actual, expected)


def test_clipped_op_called_once():
    calls = []
    image = np.array([[-1.0, -0.0, 2.0]], np.float32)

    def operation(value):
        calls.append(value)
        return value

    actual = clipped(operation)(image)
    assert len(calls) == 1 and calls[0] is image
    exact(actual, np.clip(image, 0, 1))
    assert not np.shares_memory(actual, image)


def test_concurrent_weights_and_clipping():
    image = np.random.default_rng(737).normal(size=(37, 53, 3)).astype(np.float32)
    expected = np.clip(image, 0, 1)
    with ThreadPoolExecutor(4) as executor:
        for value in executor.map(native.clip_image, [image] * 16):
            exact(value, expected)


@pytest.mark.parametrize("mode", range(8))
@pytest.mark.parametrize(
    "failure", ["output_null", "unaligned", "alias", "events_alias", "overflow"]
)
def test_abi_bounds(mode, failure):
    source = np.zeros(32, np.float32)
    output = np.ones(32, np.float32)
    events = (ct.c_int * 9)()
    count, address, flags = 32, output.ctypes.data, events
    if failure == "output_null":
        address = None
    elif failure == "unaligned":
        address += 1
    elif failure == "alias":
        if mode > 3:
            address = ct.addressof(events)
        else:
            address = source.ctypes.data
    elif failure == "events_alias":
        flags = address
    else:
        count = 2**64 - 1
    assert native._api().cn_framework_shared(
        source.ctypes.data, source.ctypes.data, address, count, mode, 1, 1.0, flags
    ) == (2 if failure == "overflow" else 1)
    np.testing.assert_array_equal(output, 1)


@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3), (1, 1, 2, 3), (3,), (1,)])
def test_copy_assignment_broadcast(shape):
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected, actual = np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32)
    expected[...] = source
    native.copy_into(actual, source)
    exact(actual, expected)


@pytest.mark.parametrize("shape", [(3, 2), (1, 3, 2), (2, 2, 3), (7,)])
def test_copy_assignment_shape_error(shape):
    source = np.zeros(shape, np.float32)
    destination = np.ones((2, 3), np.float32)
    with pytest.raises(ValueError) as expected:
        destination[...] = source
    with pytest.raises(ValueError) as actual:
        native.copy_into(destination, source)
    assert str(actual.value) == str(expected.value)
    np.testing.assert_array_equal(destination, 1)


def test_copy_assignment_readonly():
    destination = np.ones((2, 3), np.float32)
    destination.flags.writeable = False
    with pytest.raises(ValueError) as expected:
        destination[...] = np.zeros((2, 3), np.float32)
    with pytest.raises(ValueError) as actual:
        native.copy_into(destination, np.zeros((2, 3), np.float32))
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("operation", ["clip", "sin", "half"])
def test_nan_payloads(operation):
    image = np.array([0x7F800123, 0xFF800789, 0x7FC00123, 0xFFC00456], np.uint32).view(
        np.float32
    )
    original = {
        "clip": lambda image: np.clip(image, 0, 1),
        "sin": OLD.sin_blend_fn,
        "half": OLD.half_sin_blend_fn,
    }[operation]
    converted = {
        "clip": native.clip_image,
        "sin": tiles.sin_blend_fn,
        "half": tiles.half_sin_blend_fn,
    }[operation]
    expected, expected_messages = recorded(lambda: original(image))
    actual, actual_messages = recorded(lambda: converted(image))
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert actual_messages == expected_messages


@pytest.mark.parametrize("policy", ["warn", "raise", "call", "log", "ignore"])
def test_recover_alpha_error_policy(policy):
    white = np.array([np.inf, -np.inf, np.finfo(np.float32).max], np.float32)
    black = np.array([np.inf, np.inf, -np.finfo(np.float32).max], np.float32)

    def run(function):
        events = []

        class Sink:
            def write(self, message):
                events.append(message)

        handler = Sink() if policy == "log" else lambda *args: events.append(args)
        with (
            np.errstate(all=policy, call=handler),
            warnings.catch_warnings(record=True) as messages,
        ):
            warnings.simplefilter("always")
            try:
                output = function()
                failure = None
            except FloatingPointError as error:
                output, failure = None, str(error)
        return output, failure, events, [str(item.message) for item in messages]

    expected = run(lambda: 1 - (white - black))
    actual = run(lambda: native.recover_alpha(white, black))
    assert actual[1:] == expected[1:]
    if expected[0] is not None:
        assert actual[0] is not None
        exact(actual[0], expected[0])


@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.int16, np.uint8])
def test_shared_dtype_compatibility(dtype):
    image = np.arange(18, dtype=dtype).reshape(2, 3, 3)
    np.testing.assert_array_equal(native.clip_image(image), np.clip(image, 0, 1))
    np.testing.assert_array_equal(native.blend_curve(image), OLD.sin_blend_fn(image))
    np.testing.assert_array_equal(
        native.recover_alpha(image, image), 1 - (image - image)
    )
    alpha = np.ones((2, 3), dtype)
    np.testing.assert_array_equal(
        native.assemble_alpha(image, alpha), np.dstack((image, alpha))
    )
    target = np.zeros_like(image)
    native.copy_into(target, image)
    np.testing.assert_array_equal(target, image)


@pytest.mark.parametrize("shape", [(0, 3, 3), (3, 0, 3), (0, 0, 3)])
def test_empty_alpha_assembly(shape):
    image = np.empty(shape, np.float32)
    alpha = np.empty(shape[:2], np.float32)
    expected = np.dstack((image, alpha))
    actual = native.assemble_alpha(image, alpha)
    assert actual.shape == expected.shape and actual.strides == expected.strides


def test_custom_blend_callback_and_cached_weights():
    calls = []

    class Curve:
        __hash__ = object.__hash__

        def __call__(self, values):
            calls.append(values.copy())
            return values

        def __eq__(self, _other):
            raise AssertionError("weight generation must not compare a callback")

    blender = tiles.TileBlender(20, 4, 3, tiles.BlendDirection.X, Curve())
    result = blender._get_blend(4)
    assert blender._get_blend(4) is result
    assert len(calls) == 1
    exact(calls[0], np.arange(4, dtype=np.float32) / 3)


@pytest.mark.parametrize("operation", ["clip", "sin", "half", "alpha"])
def test_parallel_pool_threshold(operation):
    image = np.random.default_rng(965).uniform(-2, 2, (257, 263, 3)).astype(np.float32)
    other = np.random.default_rng(967).uniform(-2, 2, image.shape).astype(np.float32)
    original = {
        "clip": lambda: np.clip(image, 0, 1),
        "sin": lambda: OLD.sin_blend_fn(image),
        "half": lambda: OLD.half_sin_blend_fn(image),
        "alpha": lambda: 1 - (image - other),
    }[operation]
    converted = {
        "clip": lambda: native.clip_image(image),
        "sin": lambda: tiles.sin_blend_fn(image),
        "half": lambda: tiles.half_sin_blend_fn(image),
        "alpha": lambda: native.recover_alpha(image, other),
    }[operation]
    expected = original()
    with ThreadPoolExecutor(4) as executor:
        for result in executor.map(lambda _: converted(), range(8)):
            exact(result, expected)
