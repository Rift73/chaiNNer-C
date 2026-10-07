"""Exact installed Rust and NumPy oracles for completed threshold computation."""

from __future__ import annotations

import ctypes as ct
import itertools
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest
from test_color_ops import MODULES

from chainner_ext import binary_threshold as original_binary
from nodes.impl import native_threshold as native
from nodes.impl.native import lib, ptr


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[valid].view(np.uint32), expected[valid].view(np.uint32)
    )


@pytest.mark.parametrize("channels", [1, 3, 4, 7])
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1), (7, 9), (13, 17)])
@pytest.mark.parametrize("smoothness", [i / 10 for i in range(11)])
@pytest.mark.parametrize("threshold", [0, 0.01, 0.5, 0.9999, 1])
def test_antialias_options(channels, shape, smoothness, threshold):
    image = np.random.default_rng(843).random((*shape, channels), dtype=np.float32)
    exact(
        native.binary_threshold_aa(image, threshold, smoothness),
        original_binary(image, threshold, True, smoothness),
    )


@pytest.mark.parametrize("smoothness", [0, 0.3, 1])
@pytest.mark.parametrize("channels", [1, 3, 4, 7])
@pytest.mark.parametrize(
    "pattern", ["zeros", "nonfinite", "subnormal", "range", "cross"]
)
def test_antialias_numerical_boundaries(smoothness, channels, pattern):
    rng = np.random.default_rng(81)
    image = rng.random((13, 17, channels), dtype=np.float32)
    threshold = 0.5
    if pattern == "zeros":
        image[:] = rng.choice(np.array([-0.0, 0.0], np.float32), image.shape)
        threshold = 0.0
    elif pattern == "nonfinite":
        image[:] = rng.choice(
            np.array([-np.inf, np.inf, np.nan, -0.0, 0.0, 0.25, 0.75], np.float32),
            image.shape,
        )
    elif pattern == "subnormal":
        image = (image - np.float32(0.5)) * np.float32(1e-38)
        threshold = 0.0
    elif pattern == "range":
        image = np.ldexp(
            image - np.float32(0.5),
            rng.integers(-140, 127, image.shape, dtype=np.int32),
        )
    else:
        image[:] = (np.indices(image.shape[:2]).sum(axis=0) % 2)[..., None]
    exact(
        native.binary_threshold_aa(image, threshold, smoothness),
        original_binary(image, threshold, True, smoothness),
    )


@pytest.mark.parametrize("smoothness", [0, 0.3, 1])
def test_every_three_level_quadrant(smoothness):
    for values in itertools.product([0.0, 0.5, 1.0], repeat=4):
        image = np.array(values, np.float32).reshape(2, 2)
        exact(
            native.binary_threshold_aa(image, 0.5, smoothness),
            original_binary(image, 0.5, True, smoothness),
        )


def layout(image, kind):
    if kind == "reverse":
        return image[::-1, ::-1]
    if kind == "transpose":
        return image.swapaxes(0, 1)
    if kind == "channel_reverse":
        return image[..., ::-1]
    if kind == "readonly":
        image.setflags(write=False)
    elif kind == "broadcast":
        return np.broadcast_to(image[:1, :1], image.shape)
    elif kind == "unaligned":
        foreign = np.ndarray(
            image.shape, np.float32, buffer=bytearray(image.nbytes + 1), offset=1
        )
        foreign[:] = image
        return foreign
    return image


@pytest.mark.parametrize(
    "kind", ["reverse", "transpose", "readonly", "broadcast", "unaligned"]
)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_antialias_foreign_buffers(kind, channels):
    image = layout(
        np.random.default_rng(14).random((19, 23, channels), dtype=np.float32), kind
    )
    before = image.copy()
    result = native.binary_threshold_aa(image, 0.37, 0.7)
    exact(result, original_binary(image, 0.37, True, 0.7))
    exact(image, before)
    assert not np.shares_memory(result, image)
    assert result.flags.aligned and result.flags.c_contiguous and result.flags.writeable


@pytest.mark.parametrize(
    "channels", [8, 9, 15, 16, 17, 31, 32, 63, 64, 127, 128, 129, 257, 20000]
)
@pytest.mark.parametrize(
    "kind",
    ["plain", "reverse", "transpose", "readonly", "unaligned", "channel_reverse"],
)
@pytest.mark.parametrize("pattern", ["random", "nonfinite", "zeros"])
def test_high_channel_mean(channels, kind, pattern):
    rng = np.random.default_rng(22)
    image = rng.random((7, 9, channels), dtype=np.float32)
    if pattern == "nonfinite":
        image[:] = rng.choice(
            np.array([np.nan, np.inf, -np.inf, 0.25, 0.75], np.float32), image.shape
        )
    elif pattern == "zeros":
        image.fill(-0.0)
    image = layout(image, kind)
    before = image.copy()
    with np.errstate(invalid="ignore"):
        expected = np.mean(image, axis=-1)
    exact(native.channel_mean(image), expected)
    exact(image, before)


@pytest.mark.parametrize("channels", [8, 9, 16, 33, 129])
def test_planar_high_channel_mean(channels):
    # A node's CHW tensor transposed to HWC: the channel axis has the largest stride,
    # so np.mean(axis=-1) adds whole channel planes in order.
    image = (
        np.random.default_rng(23)
        .random((channels, 19, 23), dtype=np.float32)
        .transpose(1, 2, 0)
    )
    exact(native.channel_mean(image), np.mean(image, axis=-1))


def test_antialias_concurrent_calls():
    inputs = [
        np.random.default_rng(i).random((67, 71, 3), dtype=np.float32) for i in range(8)
    ]
    before = [image.copy() for image in inputs]
    expected = [original_binary(image, 0.37, True, 0.6) for image in inputs]

    def run(index):
        exact(
            native.binary_threshold_aa(inputs[index % 8], 0.37, 0.6),
            expected[index % 8],
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(24)))
    for image, saved in zip(inputs, before, strict=True):
        exact(image, saved)


@pytest.fixture
def api():
    native.binary_threshold_aa(np.zeros((1, 1), np.float32), 0.5, 0)
    return lib()


def test_checked_aa_abi(api):
    source = np.zeros((3, 7), np.float32)
    result = np.empty_like(source)
    fp = ct.POINTER(ct.c_float)
    a, b = ptr(source), ptr(result)
    assert api.cn_threshold_aa(None, b, 3, 7, 1, 0.5, 0) == 1
    assert api.cn_threshold_aa(a, a, 3, 7, 1, 0.5, 0) == 1
    assert (
        api.cn_threshold_aa(a, ct.cast(source.ctypes.data + 4, fp), 3, 7, 1, 0.5, 0)
        == 1
    )
    assert (
        api.cn_threshold_aa(ct.cast(source.ctypes.data + 1, fp), b, 3, 7, 1, 0.5, 0)
        == 1
    )
    assert api.cn_threshold_aa(a, b, ct.c_size_t(-1).value, 7, 1, 0.5, 0) == 2
    assert api.cn_threshold_channel_mean(a, a, 3, 7, 0, 0) == 1
    assert api.cn_threshold_channel_mean(a, b, ct.c_size_t(-1).value, 7, 0, 1) == 2
    assert api.cn_threshold_hard(a, a, 21, 2, 0.5, 1, 1) == 1
    assert (
        api.cn_threshold_hard(a, ct.cast(source.ctypes.data + 4, fp), 21, 2, 0.5, 1, 1)
        == 1
    )
    assert api.cn_threshold_hard(a, b, 21, 5, 0.5, 1, 1) == 1
    assert api.cn_threshold_hard(a, b, 21, 2, 0.5, 1, 2) == 1
    assert api.cn_threshold_hard(a, b, ct.c_size_t(-1).value, 2, 0.5, 1, 1) == 2


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (3,), (2, 3, 0), (2, 3, 4, 5)])
def test_invalid_image_shapes(shape):
    source = np.empty(shape, np.float32)
    with pytest.raises(ValueError):
        native.binary_threshold_aa(source, 0.5, 0)
    with pytest.raises(ValueError):
        native.channel_mean(source)


@pytest.fixture(params=[False, True])
def ipp(request):
    before = cv2.ipp.useIPP()
    cv2.ipp.setUseIPP(request.param)
    try:
        yield request.param
    finally:
        cv2.ipp.setUseIPP(before)


@pytest.mark.parametrize("kind", range(5))
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (3, 7), (17, 19), (137, 509)])
@pytest.mark.parametrize("threshold", [-np.inf, -0.0, 0.0, 0.5, np.inf, np.nan])
@pytest.mark.parametrize("maximum", [-0.0, 0.7, np.inf, np.nan])
def test_hard_threshold_full_float_contract(kind, shape, threshold, maximum, ipp):
    values = np.array([-np.inf, -1, -0.0, 0.0, 0.5, 1, np.inf, np.nan], np.float32)
    image = np.resize(values, shape)
    if kind == 2:
        # The deliberate correction is the scalar threshold contract: only a
        # strict exceedance replaces the input, preserving NaN and zero ties.
        expected = np.where(image > np.float32(threshold), np.float32(threshold), image)
    else:
        expected = cv2.threshold(image, threshold, maximum, kind)[1]
    exact(native.hard_threshold(image, threshold, maximum, kind), expected)


@pytest.mark.parametrize("channels", [8, 9, 16, 17, 127, 128, 129, 257])
@pytest.mark.parametrize("method", [0, 1])
def test_generated_threshold_high_channels(channels, method):
    current, reference = MODULES["generate_threshold"]
    source = np.random.default_rng(84).random((17, 19, channels), dtype=np.float32)
    assert current.generate_threshold_node(
        source, current.AutoThreshold(method)
    ) == reference.generate_threshold_node(source, reference.AutoThreshold(method))


def test_original_truncation_allocation_dependence_is_corrected():
    old_ipp = cv2.ipp.useIPP()
    cv2.ipp.setUseIPP(True)
    try:
        image = np.full((1, 129), np.nan, np.float32)
        outputs = []
        for alignment in (0, 16):
            backing = np.empty(145, np.float32)
            offset = ((alignment - backing.ctypes.data) % 32) // 4
            destination = backing[offset : offset + 129].reshape(1, 129)
            assert destination.ctypes.data % 32 == alignment
            outputs.append(cv2.threshold(image, 0.5, 1, 2, dst=destination)[1])
        # OpenCV 4.8's IPP TRUNC kept 128 and 112 of the 129 NaNs, by destination
        # alignment. OpenCV 5.0.0 keeps all 129 at both, which the C kernel already
        # returned: the library no longer depends on the allocation.
        assert np.isnan(outputs[0]).sum() == 129
        assert np.isnan(outputs[1]).sum() == 129
        for _ in range(8):
            actual = native.hard_threshold(image, 0.5, 1, 2)
            assert np.isnan(actual).all()
    finally:
        cv2.ipp.setUseIPP(old_ipp)


@pytest.mark.parametrize("source_zero", [-0.0, 0.0])
@pytest.mark.parametrize("threshold_zero", [-0.0, 0.0])
def test_truncation_zero_ties_preserve_input_bits(source_zero, threshold_zero):
    image = np.full((137, 509), source_zero, np.float32)
    exact(native.hard_threshold(image, threshold_zero, 1, 2), image)


def test_threshold_node_has_no_rust_or_opencv_image_call(monkeypatch):
    current, reference = MODULES["threshold"]
    source = np.random.default_rng(91).random((31, 37, 3), dtype=np.float32)
    expected = reference.threshold_node(
        source, 43.3, reference.ThresholdType.BINARY, 78.9, True, 7
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Threshold image arithmetic must execute in C")

    monkeypatch.setattr(cv2, "threshold", forbidden)
    monkeypatch.setattr(reference, "binary_threshold", forbidden)
    actual = current.threshold_node(
        source, 43.3, current.ThresholdType.BINARY, 78.9, True, 7
    )
    exact(actual, expected)


@pytest.mark.parametrize(
    "kind", ["reverse", "transpose", "readonly", "broadcast", "unaligned"]
)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_hard_threshold_foreign_buffers(kind, channels):
    source = layout(
        np.random.default_rng(25).random((19, 23, channels), dtype=np.float32), kind
    )
    before = source.copy()
    result = native.hard_threshold(source, 0.43, 0.81, 1)
    exact(result, cv2.threshold(source, 0.43, 0.81, 1)[1])
    exact(source, before)
    assert not np.shares_memory(source, result)
    assert result.flags.aligned and result.flags.c_contiguous and result.flags.writeable
