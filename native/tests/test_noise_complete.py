"""Exact NumPy 1.24.4 stream and frozen Add Noise differential checks."""

from __future__ import annotations

import ctypes as ct
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from nodes.impl import native_noise as native
from nodes.impl import noise
from nodes.impl.native import ptr

REFERENCE_PATH = Path(__file__).with_name("reference_generation") / "noise.py"
SPEC = importlib.util.spec_from_file_location(
    "nodes.impl._reference_noise_complete", REFERENCE_PATH
)
assert SPEC is not None and SPEC.loader is not None
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)
NAMES = [
    "gaussian_noise",
    "uniform_noise",
    "salt_and_pepper_noise",
    "poisson_noise",
    "speckle_noise",
]
SEEDS = [
    0,
    1,
    2**32 - 1,
    2**32,
    2**64 + 43,
    2**127,
    2**128 - 1,
    2**256 + 91,
    2**4096 + 2**513 + 13,
]


def equal_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def raw(seed, advance, count, bits):
    words = native.seed_words(seed)
    out = np.empty(count, np.uint64)
    api = native._api().cn_seeded_raw  # raw ABI regression
    api.argtypes = [
        ct.POINTER(ct.c_uint32),
        ct.c_size_t,
        ct.c_uint64,
        ct.POINTER(ct.c_uint64),
        ct.c_size_t,
        ct.c_int,
    ]
    api.restype = ct.c_int
    assert (
        api(
            words.ctypes.data_as(ct.POINTER(ct.c_uint32)),
            words.size,
            advance,
            out.ctypes.data_as(ct.POINTER(ct.c_uint64)),
            count,
            bits,
        )
        == 0
    )
    return out


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("advance", [0, 1, 2, 7, 2**32 + 5, 2**64 - 1])
@pytest.mark.parametrize("bits", [32, 64])
def test_seed_sequence_pcg64_and_advance(seed, advance, bits):
    rng = np.random.PCG64(seed)
    rng.advance(advance)
    if bits == 64:
        expected = rng.random_raw(1007)
    else:
        interface = rng.ctypes
        expected = np.array(
            [interface.next_uint32(interface.state) for _ in range(1007)], np.uint64
        )
    np.testing.assert_array_equal(raw(seed, advance, 1007, bits), expected)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "count", [0, 1, 2, 3, 4, 7, 31, 32, 33, 255, 256, 257, 1003, 65537]
)
def test_permutation_exact_masked_rejection(seed, count):
    expected = np.random.default_rng(seed).permutation(count)
    actual = native.seeded_permutation(seed, count)
    np.testing.assert_array_equal(actual, expected)


def image_fixture(channels, layout):
    shape = (19, 23) if channels == 1 else (19, 23, channels)
    image = np.random.default_rng(737).uniform(-0.5, 1.5, shape).astype(np.float32)
    if layout == "strided":
        image = image[::-2, 1::3]
    elif layout == "fortran":
        image = np.asfortranarray(image)
    elif layout == "readonly":
        image.setflags(write=False)
    elif layout == "broadcast":
        image = np.broadcast_to(image[:1, :1], shape)
    elif layout == "unaligned":
        copy = np.ndarray(
            shape, np.float32, buffer=bytearray(image.nbytes + 1), offset=1
        )
        copy[...] = image
        image = copy
    return image


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("channels", [1, 3, 4, 7])
@pytest.mark.parametrize("color", ["RGB", "GRAY"])
@pytest.mark.parametrize(
    "layout", ["plain", "strided", "fortran", "readonly", "broadcast", "unaligned"]
)
@pytest.mark.parametrize("seed", [0, 701, 2**257 + 1923])
def test_all_modes_channels_layouts_exact(name, channels, color, layout, seed):
    image = image_fixture(channels, layout)
    before = image.copy()
    expected = getattr(REFERENCE, name)(image, 0.337, REFERENCE.NoiseColor[color], seed)
    actual = getattr(noise, name)(image, 0.337, noise.NoiseColor[color], seed)
    equal_bits(actual, expected)
    equal_bits(image, before)
    assert not np.shares_memory(actual, image)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("color", ["RGB", "GRAY"])
def test_every_slider_amount(name, color):
    image = np.random.default_rng(25).uniform(-0.1, 1.1, (3, 7, 4)).astype(np.float32)
    for tick in range(1001):
        # The node gets a decimal percentage, then divides by 100.
        amount = (tick / 10) / 100
        expected = getattr(REFERENCE, name)(
            image, amount, REFERENCE.NoiseColor[color], 812 + tick
        )
        actual = getattr(noise, name)(
            image, amount, noise.NoiseColor[color], 812 + tick
        )
        equal_bits(actual, expected)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("color", ["RGB", "GRAY"])
@pytest.mark.parametrize("amount", [0, -0.0, np.nextafter(0.0, 1.0), 0.001, 0.5, 1])
def test_zero_subnormal_nonfinite_pixels_alpha_and_singleton(name, color, amount):
    source = np.array(
        [
            0,
            0x80000000,
            0x7FC00123,
            0xFFC01234,
            0x7F800000,
            0xFF800000,
            1,
            0x80000001,
            0x3F800000,
            0x3F800001,
            0xBF800000,
            0x3F000000,
        ],
        np.uint32,
    ).view(np.float32)
    for shape in [(1, 12), (12, 1, 1), (1, 4, 3), (3, 1, 4)]:
        image = source.reshape(shape)
        if np.signbit(amount) and name in (
            "gaussian_noise",
            "uniform_noise",
            "speckle_noise",
        ):
            message = "high - low < 0" if name == "uniform_noise" else "scale < 0"
            for module in [REFERENCE, noise]:
                with pytest.raises(ValueError, match=message):
                    getattr(module, name)(image, amount, module.NoiseColor[color], 415)
            continue
        with np.errstate(all="ignore"):
            expected = getattr(REFERENCE, name)(
                image, amount, REFERENCE.NoiseColor[color], 415
            )
            actual = getattr(noise, name)(image, amount, noise.NoiseColor[color], 415)
        equal_bits(actual, expected)


@pytest.mark.parametrize("name", ["gaussian_noise", "speckle_noise"])
@pytest.mark.parametrize("seed", [0, 41, 2**256 + 23])
def test_normal_tail_rejection_stream_large_exact(name, seed):
    # Enough draws to cover rare normal tails/rejections; no timing assertion.
    image = np.full((317, 337, 3), 0.5, np.float32)
    expected = getattr(REFERENCE, name)(image, 0.1, REFERENCE.NoiseColor.RGB, seed)
    actual = getattr(noise, name)(image, 0.1, noise.NoiseColor.RGB, seed)
    equal_bits(actual, expected)


def test_concurrent_independent_streams_and_repeatability():
    image = image_fixture(4, "readonly")
    jobs = [
        (name, color, seed)
        for name in NAMES
        for color in ["GRAY", "RGB"]
        for seed in [0, 812]
    ]
    expected = [
        getattr(REFERENCE, name)(image, 0.83, REFERENCE.NoiseColor[color], seed)
        for name, color, seed in jobs
    ]

    def run(job):
        name, color, seed = job
        return getattr(noise, name)(image, 0.83, noise.NoiseColor[color], seed)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in range(3):
            for actual, reference in zip(pool.map(run, jobs), expected, strict=True):
                equal_bits(actual, reference)


@pytest.mark.parametrize("name", ["uniform_noise", "salt_and_pepper_noise"])
@pytest.mark.parametrize("color", ["RGB", "GRAY"])
@pytest.mark.parametrize("count", [65535, 65536, 65537, 131071, 262147])
def test_parallel_chunk_boundaries_preserve_stream(name, color, count):
    image = np.full((1, count, 4), 0.37, np.float32)
    expected = getattr(REFERENCE, name)(
        image, 0.673, REFERENCE.NoiseColor[color], 2**512 + 49
    )
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(
                getattr(noise, name), image, 0.673, noise.NoiseColor[color], 2**512 + 49
            )
            for _ in range(3)
        ]
        for future in futures:
            equal_bits(future.result(), expected)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("seed", [-1, -(2**4096)])
def test_negative_seeds_preserve_original_rejection(name, seed):
    image = np.zeros((3, 4), np.float32)
    for module in [REFERENCE, noise]:
        with pytest.raises(ValueError, match="expected non-negative integer"):
            getattr(module, name)(image, 0.5, module.NoiseColor.GRAY, seed)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros(3, np.float32),
        np.zeros((1, 1, 1, 1), np.float32),
        np.zeros((0, 2), np.float32),
        np.zeros((2, 2, 0), np.float32),
        np.zeros((2, 2, 2), np.float32),
        np.zeros((2, 2), np.float64),
    ],
)
def test_invalid_shapes_do_not_dispatch(monkeypatch, image):
    monkeypatch.setattr(
        native, "_api", lambda: pytest.fail("invalid buffer dispatched to C")
    )
    with pytest.raises((TypeError, ValueError, AssertionError)):
        native.add_noise(image, 0.5, 3, 0, 0)


@pytest.mark.parametrize("amount", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_amounts_do_not_dispatch(monkeypatch, amount):
    monkeypatch.setattr(
        native, "_api", lambda: pytest.fail("invalid amount dispatched to C")
    )
    with pytest.raises(ValueError):
        native.add_noise(np.zeros((2, 2), np.float32), amount, 3, 0, 0)


def test_c_validation_preserves_output():
    image = np.zeros((2, 2, 3), np.float32)
    out = np.full_like(image, 19)
    words = native.seed_words(0)
    seed_ptr = words.ctypes.data_as(ct.POINTER(ct.c_uint32))
    api = native._api().cn_add_seeded_noise  # raw ABI regression
    for pixels, channels, nc, mode, amount, source, target, word_count in [
        (0, 3, 3, 0, 0.5, ptr(image), ptr(out), 1),
        (4, 2, 3, 0, 0.5, ptr(image), ptr(out), 1),
        (4, 3, 2, 0, 0.5, ptr(image), ptr(out), 1),
        (4, 3, 3, 5, 0.5, ptr(image), ptr(out), 1),
        (4, 3, 3, 0, np.nan, ptr(image), ptr(out), 1),
        (4, 3, 3, 0, 0.5, None, ptr(out), 1),
        (4, 3, 3, 0, 0.5, ptr(image), None, 1),
        (4, 3, 3, 0, 0.5, ptr(image), ptr(out), 0),
        (2**64 - 1, 3, 3, 0, 0.5, ptr(image), ptr(out), 1),
    ]:
        assert api(
            source, pixels, channels, nc, mode, amount, seed_ptr, word_count, target
        ) in (1, 2)
        np.testing.assert_array_equal(out, 19)


def test_production_noise_does_not_use_numpy_rng(monkeypatch):
    monkeypatch.setattr(
        np.random, "default_rng", lambda *args: pytest.fail("NumPy RNG used")
    )
    for name in NAMES:
        result = getattr(noise, name)(
            np.full((9, 7, 4), 0.5, np.float32), 0.5, noise.NoiseColor.RGB, 19
        )
        assert result.shape == (9, 7, 4)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("color", ["RGB", "GRAY"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint8])
@pytest.mark.parametrize("shape", [(3, 7), (4, 5, 4), (0, 7), (3, 0, 4)])
@pytest.mark.parametrize("amount", [0, 0.33, 1.5, 2, 2.01, 19, -0.2, np.inf, np.nan])
def test_public_helper_general_compatibility(name, color, dtype, shape, amount):
    image = np.ones(shape, dtype=dtype)
    before = image.copy()
    with np.errstate(all="ignore"):
        try:
            expected = getattr(REFERENCE, name)(
                image, amount, REFERENCE.NoiseColor[color], 1821
            )
        except Exception as original_error:
            with pytest.raises(type(original_error)) as current_error:
                getattr(noise, name)(image, amount, noise.NoiseColor[color], 1821)
            assert str(current_error.value) == str(original_error)
        else:
            actual = getattr(noise, name)(image, amount, noise.NoiseColor[color], 1821)
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual.tobytes(), expected.tobytes())
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "seed_type", ["sequence", "generator", "seed_sequence", "scalar_array"]
)
def test_public_helper_numpy_seed_objects(name, seed_type):
    def make_seed():
        if seed_type == "sequence":
            return [17, 81, 203]
        if seed_type == "generator":
            return np.random.default_rng(92)
        if seed_type == "seed_sequence":
            return np.random.SeedSequence(92)
        return np.array(92)

    image = np.full((3, 7, 4), 0.5, np.float32)
    expected_seed, actual_seed = make_seed(), make_seed()
    if seed_type == "scalar_array":
        for module in [REFERENCE, noise]:
            with pytest.raises(TypeError, match="len\\(\\) of unsized object"):
                getattr(module, name)(image, 0.31, module.NoiseColor.RGB, make_seed())
        return
    expected = getattr(REFERENCE, name)(
        image, 0.31, REFERENCE.NoiseColor.RGB, expected_seed
    )
    actual = getattr(noise, name)(image, 0.31, noise.NoiseColor.RGB, actual_seed)
    equal_bits(actual, expected)
    if seed_type == "generator":
        assert isinstance(actual_seed, np.random.Generator)
        assert isinstance(expected_seed, np.random.Generator)
        np.testing.assert_array_equal(actual_seed.random(31), expected_seed.random(31))
