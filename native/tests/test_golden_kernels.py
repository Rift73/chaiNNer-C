"""Golden kernel tool and the B3 manifests (SP4 spec 4.4, SP4a Task 8, SP4b D3).

The manifests under golden/ are written only by `golden_kernels.py write` from the
B3 DLL; a mismatch with the current DLL goes to the stand-in, never edited away.
`golden_kernels.py rebase` can move a conversion case whose clamp reference
differs from B3 to the oracle's np.clip, recording it: Task 3b1 moved 40 on NumPy
1.24.4, and U2 restored them, since NumPy 2.5.3's np.clip gives B3's bits. No test
reads the git-ignored reports or baselines (spec 4.4): the manifests carry B3's
outputs. A case's special set is the third-last part of its id, its group the
first; a conversion case whose src is not float32 has its mapping (bits, ramp)
there, and the ties group has "ties". A lens case's (a, b) pair is its third part;
a lens_power case's channels (cC) its second and exponent name its third. A
morphology case's form (golden_kernels.morph_label) is its third part, except group
X (cn_image_exceptional: X-position-word-mxcsr-count, the word third-last); a resample
case's fFgGvV its second and target (th x tw) its third. A palette case's capacity (kK)
is its second part and src mapping its third; a dither case's kind (apply, dither or
riemersma) its second, palette form its third, colors (nN) its fourth and mMaA (mode,
algorithm) its fifth. A composite case's kind is its second part, mode (mM) its third,
layers (bBoO, each channel count followed by k for a constant layer) its fourth,
geometry (xXyYcCROP, n for a minus sign) its fifth and sizes (BHxBWoOHxOW) its last. A
blend case's entry kind (images or mode) is its second part, mode (mM) its third,
channels (oOCbBC, or raw for cn_blend_mode) its fourth and count its last. A tail case
has six parts: group, entry kind (tile, store, multiply, normal or caption), form (tail
labels), set (bits for the store and multiply, whose double inputs hold PCG64 words),
MXCSR state and size.
"""

import ctypes
import dataclasses
import hashlib
import itertools
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import golden_kernels
import numpy as np
import pytest
import reference_blend
from golden_kernels import (
    BORDER_FORMS,
    BORDER_SETS,
    CONVERSION_KERNELS,
    CONVERT_DTYPES,
    DENORMAL_ROW,
    DILATE,
    DITHER_APPLY,
    DITHER_BENCH,
    DITHER_CHANNELS,
    DITHER_DECAY,
    DITHER_ENTRY,
    DITHER_FORMS,
    DITHER_HISTORY,
    DITHER_SETS,
    DITHER_SHAPE,
    DITHER_SIZES,
    ERODE,
    EXCEPTIONAL_COUNTS,
    EXCEPTIONAL_ENTRY,
    EXCEPTIONAL_POSITIONS,
    EXCEPTIONAL_WORDS,
    FILTER_ENTRY,
    HOT_COMBOS,
    KERNELS,
    LENS_INPUTS,
    LENS_PAIRS,
    LENS_SETS,
    MORPH_BENCH,
    MORPH_CHUNKS,
    MORPH_ENTRY,
    MORPH_FORMS,
    MORPH_SETS,
    MORPH_SHAPE,
    MXCSR_STATES,
    ORACLE,
    PALETTE_BENCH,
    PALETTE_CAPACITIES,
    PALETTE_CHANNELS,
    PALETTE_EMPTY_CHILD,
    PALETTE_ENTRY,
    PALETTE_PIXELS,
    PALETTE_SETS,
    PALETTE_TIE_PIXELS,
    PALETTE_TIES,
    PORT,
    POWER_CHANNELS,
    POWER_ENTRY,
    POWER_EXPONENTS,
    POWER_PIXELS,
    POWER_SETS,
    RESAMPLE_BENCH,
    RESAMPLE_EDGES,
    RESAMPLE_ENTRY,
    RESAMPLE_FILTERS,
    RESAMPLE_GRID,
    RESAMPLE_SETS,
    RIEMERSMA_ENTRY,
    SCHEMA,
    SP4A_SETS,
    SPECIALS,
    TILE_BENCH,
    TILE_FALLBACKS,
    TILE_FORMS,
    TILE_HEIGHTS,
    TILE_PLANE,
    TILE_SETS,
    TILE_UNTABLED,
    Case,
    clamps,
    digest,
    digest_int,
    dither_form_specials,
    load_manifest,
    make_array,
    matrix,
    morph_label,
    oracle_clip,
    rebased_ids,
    resample_payloads,
    run_case,
    sweep_cases,
)
from test_composite_complete import reference as composite_reference
from test_isa_dispatch import isa_get, isa_set

from nodes.impl import native, native_composite, native_filters
from nodes.impl.color.color import Color
from nodes.impl.native_opencv_simd import float32_lanes, target

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent / "golden"
TOOL = ROOT / "native" / "tools" / "golden_kernels.py"
CURRENT_DLL = Path(native.__file__).with_name("chainner_native.dll")  # lib()'s
B3_SHA256 = "5123f21c78305edd30893e582273a23b2e073921c5599455ac1f808e7c27504d"
# Controller ruling (2026-10-03): the matrix as briefed is ~140 kB, not tens of kB.
SIZE_LIMIT = 200_000


def special_set(case):
    return case.id.split("-")[-3]


def expected_specials(name, input_name, shape):
    """The set's values cycled over the in-range, deduplicated index rule."""
    values = SPECIALS[name]["values"]
    n = math.prod(shape)
    listed = [0, 1, 7, 8, n // 3, n // 2, n - 9, n - 8, n - 2, n - 1]
    indices = dict.fromkeys(i for i in listed if 0 <= i < n)
    pairs = [[i, values[k % len(values)]] for k, i in enumerate(indices) if values]
    if input_name == "kernel" and shape == [5, 5]:
        pairs += [[6, "0x00000000"], [18, "0x80000000"]]  # the skipped taps
    return pairs


def bare_kernel(case, name):
    """Controller ruling: group A and mixed convolution kernels hold no specials."""
    return case.id.split("-")[0] == "A" or name == "mixed"


def matching_sets(case):
    """The sets whose payloads and specials reproduce every input of the case."""
    return {
        name
        for name, special in SPECIALS.items()
        if special["payloads"] == case.payloads
        and all(
            recipe["specials"]
            == expected_specials(
                "clean" if input_name == "kernel" and bare_kernel(case, name) else name,
                input_name,
                recipe["shape"],
            )
            for input_name, recipe in case.inputs.items()
        )
    }


def all_nan_digests(case):
    """Digests of the case's output filled with any NaN the matrix's sets produce."""
    args = case.args
    pad = args.get("padding", 0)
    shape = (args["h"] + 2 * pad, args["w"] + 2 * pad, args["c"])
    return {
        digest(np.full(shape, bits, np.uint32).view(np.float32), case.payloads)
        for bits in (0x7FC12345, 0x7F812345, 0xFFC00000, 0x7FC00000)
    }


def twin_id(case_id):
    """The default-MXCSR twin of a daz_ftz matrix case."""
    return case_id.replace("-daz_ftz-", "-default-")


def seed_of(case_id):
    """SHA-256 of the id without its MXCSR part, so matrix twins share inputs."""
    key = case_id.replace("-daz_ftz-", "-").replace("-default-", "-")
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def combinations(entries, places=True):
    """Every (entry, form, [border or padding,] mirror value) of the entries."""
    found = set()
    for entry in entries:
        spec = golden_kernels.ENTRIES[entry]
        for form in spec.forms:
            for mirror in spec.mirrors:
                if not places:
                    found.add((entry, form, mirror))
                    continue
                for place in spec.places or (None,):
                    found.add((entry, form, place, mirror))
    return found


def axes_of(case, places=True):
    spec = golden_kernels.ENTRIES[case.entry]
    form = tuple(case.args[p] for p in spec.form)
    mirror = case.args[spec.mirror]
    if not places:
        return case.entry, form, mirror
    return case.entry, form, case.args[spec.place] if spec.place else None, mirror


class Stub:
    """A DLL stand-in: every entry returns 0 and writes nothing; FP state fixed."""

    def __init__(self, state=0):
        self.state = state

    def __getitem__(self, name):
        if name == "cn_image_fp_state":
            return lambda: self.state
        return lambda *args: 0


class NoIsa:
    """A DLL without the cn_isa exports (B3 and older)."""

    def __getitem__(self, name):
        raise AttributeError(name)


class Capped:
    """A DLL whose cn_isa_set always returns scalar."""

    def __getitem__(self, name):
        return lambda level: 0


def nan_bits(value):
    return (value & 0x7FFFFFFF) > 0x7F800000


def combo(case):
    """A conversion case's (type, output, normalize)."""
    return case.args["type"], case.args["output"], case.args["normalize"]


def has_positive_denormal(case):
    """Whether a conversion case's float32 src holds a positive denormal."""
    if case.inputs["src"].get("dtype", "float32") != "float32":
        return False
    bits = make_array(case.inputs["src"]).view(np.uint32)
    return bool(((bits != 0) & (bits < 0x00800000)).any())


def border_sets(case):
    """The sets that reproduce a convolution_border case's inputs (SP4b D8): src holds
    the set at the index rule; the kernel stays finite, holding the edges set or
    none, the (5,5) zero taps, and possibly DENORMAL_ROW."""
    src, kernel = case.inputs["src"], case.inputs["kernel"]
    row = [list(tap) for tap in DENORMAL_ROW]
    found = set()
    for name in BORDER_SETS:
        held = expected_specials(
            name if name == "edges" else "clean", "kernel", kernel["shape"]
        )
        if (
            SPECIALS[name]["payloads"] == case.payloads
            and src["specials"] == expected_specials(name, "src", src["shape"])
            and kernel["specials"] in (held, held + row)
        ):
            found.add(name)
    return found


def check_border_recipe(case, sweep=False):
    """A convolution_border case: src (h, w, c) on unit (tiny in group T, and either
    for a clean sweep src), the kernel (kh, kw) on signed, one seed by id; its set
    per border_sets; DENORMAL_ROW only in a (5,5) kernel, in the matrix in group Z
    alone."""
    args = case.args
    src, kernel = case.inputs["src"], case.inputs["kernel"]
    assert sorted(case.inputs) == ["kernel", "src"], case.id
    assert src["shape"] == [args["h"], args["w"], args["c"]], case.id
    assert kernel["shape"] == [args["kh"], args["kw"]], case.id
    assert src["seed"] == kernel["seed"] == seed_of(case.id), case.id
    assert kernel["mapping"] == "signed", case.id
    names = border_sets(case)
    row = [list(tap) for tap in DENORMAL_ROW]
    has_row = kernel["specials"][-len(row) :] == row
    assert not has_row or kernel["shape"] == [5, 5], case.id
    if sweep:
        assert names, case.id
        assert src["mapping"] == "unit" or (
            src["mapping"] == "tiny" and src["specials"] == []
        ), case.id
        return
    group = case.id.split("-")[0]
    assert special_set(case) in names, case.id
    assert src["mapping"] == ("tiny" if group == "T" else "unit"), case.id
    assert has_row == (group == "Z"), case.id


def lens_seed(case_id, name):
    """A lens input's seed: SHA-256 of the id without its MXCSR part, then -name."""
    key = case_id.replace("-daz_ftz-", "-").replace("-default-", "-")
    return int.from_bytes(hashlib.sha256(f"{key}-{name}".encode()).digest()[:8], "big")


def lens_specials(name, position, n):
    """The set at the index rule, rotated by the input's position when mixed (D3)."""
    values = SPECIALS[name]["values"]
    shift = position if SPECIALS[name]["payloads"] == "mixed" else 0
    listed = [0, 1, 7, 8, n // 3, n // 2, n - 9, n - 8, n - 2, n - 1]
    indices = dict.fromkeys(i for i in listed if 0 <= i < n)
    return [
        [i, values[(k + shift) % len(values)]] for k, i in enumerate(indices) if values
    ]


def lens_sets(case):
    """The sets whose payloads and specials reproduce every input of a lens case."""
    count = case.args["count"]
    return {
        name
        for name in LENS_SETS
        if SPECIALS[name]["payloads"] == case.payloads
        and all(
            case.inputs[input_name]["specials"] == lens_specials(name, position, count)
            for position, input_name in enumerate(LENS_INPUTS)
        )
    }


def check_lens_recipe(case, sweep=False):
    """A cn_lens_compose case: f1-f4 and the in-out out, each (count,) float32 on
    signed (a clean sweep case on signed or tiny), seeded by id and name; (a, b) one
    of LENS_PAIRS as float32 bits; the set at the index rule, rotated by the input's
    position when mixed."""
    args = case.args
    assert sorted(case.inputs) == sorted(LENS_INPUTS), case.id
    assert sorted(args) == ["a", "accumulate", "b", "count"], case.id
    assert (args["a"], args["b"]) in LENS_PAIRS.values(), case.id
    assert args["accumulate"] in (0, 1), case.id
    mappings = set()
    for name, recipe in case.inputs.items():
        assert recipe["shape"] == [args["count"]], case.id
        assert "dtype" not in recipe, case.id
        assert recipe["seed"] == lens_seed(case.id, name), case.id
        mappings.add(recipe["mapping"])
    names = lens_sets(case)
    if sweep:
        assert names, case.id
        assert mappings == {"signed"} or (mappings == {"tiny"} and "clean" in names), (
            case.id
        )
        return
    parts = case.id.split("-")
    assert (args["a"], args["b"]) == LENS_PAIRS[parts[2]], case.id
    assert parts[1] == f"acc{args['accumulate']}", case.id
    assert mappings == {"signed"}, case.id
    assert special_set(case) in names, case.id
    assert case.payloads == SPECIALS[special_set(case)]["payloads"], case.id


def power_sets(case):
    """The POWER_SETS whose payloads and specials reproduce a lens_power case's src."""
    recipe = case.inputs["src"]
    return {
        name
        for name in POWER_SETS
        if SPECIALS[name]["payloads"] == case.payloads
        and recipe["specials"] == expected_specials(name, "src", recipe["shape"])
    }


def check_power_recipe(case, sweep=False):
    """A cn_lens_power case: src the (channels, pixels) float32 plane on signed (a
    clean sweep case on signed or tiny), seeded by id; the exponent one of
    POWER_EXPONENTS as bits; the set at the index rule."""
    args = case.args
    recipe = case.inputs["src"]
    assert case.entry == POWER_ENTRY, case.id
    assert list(case.inputs) == ["src"], case.id
    assert sorted(args) == ["channels", "exponent", "pixels"], case.id
    assert args["channels"] in POWER_CHANNELS, case.id
    assert args["exponent"] in POWER_EXPONENTS.values(), case.id
    assert recipe["shape"] == [args["channels"], args["pixels"]], case.id
    assert "dtype" not in recipe, case.id
    assert recipe["seed"] == seed_of(case.id), case.id
    names = power_sets(case)
    if sweep:
        assert names, case.id
        assert recipe["mapping"] == "signed" or (
            recipe["mapping"] == "tiny" and "clean" in names
        ), case.id
        return
    parts = case.id.split("-")
    assert parts[1] == f"c{args['channels']}", case.id
    assert POWER_EXPONENTS[parts[2]] == args["exponent"], case.id
    assert recipe["mapping"] == "signed", case.id
    assert special_set(case) in names, case.id
    assert case.payloads == SPECIALS[special_set(case)]["payloads"], case.id


def positive_denormal(case):
    """Whether a lens_power case's src holds a positive denormal."""
    bits = make_array(case.inputs["src"]).view(np.uint32)
    return bool(((bits != 0) & (bits < 0x00800000)).any())


def bits_float32(text):
    return np.array([int(text, 16)], np.uint32).view(np.float32)[0]


def compose_reference(case):
    """cn_lens_compose's out by NumPy float32 operations in compose_range's order
    (lens_complete_ops.c), under the calling thread's MXCSR."""
    args = case.args
    f1, f2, f3, f4, out = (make_array(case.inputs[name]) for name in LENS_INPUTS)
    a, b = bits_float32(args["a"]), bits_float32(args["b"])
    zero = np.float32(0)
    with np.errstate(all="ignore"):
        real = f1 - f4
        imag = f2 + f3
        complex_real = zero * imag - zero
        complex_imag = zero + imag
        real = real + complex_real
        imag = zero + complex_imag
        value = real * a + imag * b
        if args["accumulate"]:
            value = out + value
    return value


def check_conversion_recipe(case, sweep=False):
    """A conversion case's src: its dtype by type, its shape by count, its seed by id;
    float32 holds its set at the index rule (signed or tiny mapping) or the ties;
    other dtypes hold bits or ramp and no specials. A matrix id names the set or
    mapping; a sweep id (sweep-<seed>-<n>) names neither."""
    recipe = case.inputs["src"]
    dtype = CONVERT_DTYPES[case.args["type"]]
    name = recipe["mapping"] if sweep else special_set(case)
    assert list(case.inputs) == ["src"], case.id
    assert recipe.get("dtype", "float32") == dtype, case.id
    assert "dtype" in recipe or dtype == "float32", case.id  # float32 stays implicit
    assert recipe["shape"] == [case.args["count"]], case.id
    assert recipe["seed"] == seed_of(case.id), case.id
    if dtype != "float32":
        assert recipe["mapping"] == name, case.id
        assert name in (("bits", "ramp") if dtype == "uint8" else ("bits",)), case.id
        assert recipe["specials"] == [], case.id
        assert case.payloads == "single", case.id
    elif recipe["mapping"] == "ties":
        assert name == "ties", case.id
        assert recipe["specials"] == [], case.id
        assert case.args["count"] == golden_kernels.ties().size, case.id
        assert case.payloads == "single", case.id
    else:
        # A sweep's clean float32 src also draws raw bits (every magnitude).
        mappings = ("signed", "tiny", "bits") if sweep else ("signed", "tiny")
        assert recipe["mapping"] in mappings, case.id
        names = matching_sets(case)
        if sweep:
            # Tiny arrays can fit several sets (nan and mixed both start 0x7fc12345).
            assert names, case.id
            assert recipe["mapping"] == "signed" or recipe["specials"] == [], case.id
        else:
            assert recipe["mapping"] == "signed" or name == "clean", case.id
            assert name in names, case.id
            assert case.payloads == SPECIALS[name]["payloads"], case.id


@pytest.mark.parametrize("kernel", KERNELS)
def test_pcg64_stream_matches_the_manifests(kernel):
    header, _ = load_manifest(GOLDEN / f"{kernel}.json")
    assert np.random.PCG64(0).random_raw(4).tolist() == header["pcg64_check"]


def test_recipes_are_exact_and_place_specials():
    for mapping, scale, offset in (("unit", 2**24, 0), ("signed", 2**23, -(2**23))):
        recipe = {
            "shape": [31, 17, 3],
            "seed": 12345,
            "mapping": mapping,
            "specials": [],
        }
        array = make_array(recipe)
        assert array.dtype == np.float32
        assert array.shape == (31, 17, 3)
        top = np.random.PCG64(12345).random_raw(31 * 17 * 3) >> np.uint64(40)
        # Every value is k * 2**-24 (unit) or k * 2**-23 (signed) exactly.
        assert np.array_equal(
            array.astype(np.float64).ravel() * scale,
            top.astype(np.float64) + offset,
        )
    # tiny: the 24 bits are the float's bits, denormals and the smallest normals.
    tiny = make_array({"shape": [64, 64], "seed": 5, "mapping": "tiny", "specials": []})
    top = np.random.PCG64(5).random_raw(64 * 64) >> np.uint64(40)
    assert tiny.dtype == np.float32
    assert np.array_equal(tiny.view(np.uint32).ravel(), top.astype(np.uint32))
    denormal = tiny.view(np.uint32) < 0x00800000
    assert 0.4 < denormal.mean() < 0.6
    # grid: (raw >> 62) * 0.25, with -0 for level 0 when bit 61 is set.
    grid = make_array({"shape": [64, 64], "seed": 6, "mapping": "grid", "specials": []})
    raw = np.random.PCG64(6).random_raw(64 * 64)
    level = (raw >> np.uint64(62)).astype(np.float64)
    assert grid.dtype == np.float32
    assert np.array_equal(grid.astype(np.float64).ravel(), level * 0.25)
    negative = (level == 0) & ((raw >> np.uint64(61)) & np.uint64(1) == 1)
    assert np.array_equal(np.signbit(grid).ravel(), negative)
    assert set(grid.view(np.uint32).ravel().tolist()) == {
        0x00000000,
        0x80000000,
        0x3E800000,
        0x3F000000,
        0x3F400000,
    }
    # zeros (Task 5b): the top 6 bits pick one of 64 words, each ZERO_WEIGHTS word
    # repeated by its weight, in key order (f32::total_cmp) with half the weight below +0.
    zeros = make_array(
        {"shape": [64, 64], "seed": 8, "mapping": "zeros", "specials": []}
    )
    raw = np.random.PCG64(8).random_raw(64 * 64)
    weights = golden_kernels.ZERO_WEIGHTS
    words = [int(word, 16) for word, weight in weights for _ in range(weight)]
    assert len(words) == 64 and sum(weight for _, weight in weights[:6]) == 32
    assert zeros.dtype == np.float32
    expected = np.array(words, np.uint32)[(raw >> np.uint64(58)).astype(np.intp)]
    assert np.array_equal(zeros.view(np.uint32).ravel(), expected)
    ordered = [int(word, 16) for word, _ in weights]
    keys = [w ^ 0xFFFFFFFF if w >> 31 else w ^ 0x80000000 for w in ordered]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert set(zeros.view(np.uint32).ravel().tolist()) == {
        int(word, 16) for word, _ in weights
    }
    listed = [(0, "0x7fc12345"), (35, "0x80000000"), (17, "0x00000001")]
    recipe = {"shape": [4, 9], "seed": 7, "mapping": "unit", "specials": listed}
    bits = make_array(recipe).view(np.uint32).ravel()
    plain = make_array({**recipe, "specials": []}).view(np.uint32).ravel()
    for index, value in listed:
        assert bits[index] == int(value, 16)
    untouched = np.setdiff1d(np.arange(36), [index for index, _ in listed])
    assert np.array_equal(bits[untouched], plain[untouched])
    # The matrix places its set's values at the index rule's positions, cycling,
    # and the (5,5) kernel always holds the two zero taps.
    # Group E's src is tiny (denormal range); every other src is unit, every
    # kernel signed. Conversion recipes follow check_conversion_recipe.
    for kernel in KERNELS:
        for case in matrix(kernel):
            if kernel in CONVERSION_KERNELS:
                check_conversion_recipe(case)
                continue
            if kernel == "convolution_border":
                check_border_recipe(case)
                continue
            if kernel == "lens":
                check_lens_recipe(case)
                continue
            if kernel == "lens_power":
                check_power_recipe(case)
                continue
            if kernel == "morphology":
                check_morph_recipe(case)
                continue
            if kernel == "resample":
                check_resample_recipe(case)
                continue
            if kernel == "palette":
                check_palette_recipe(case)
                continue
            if kernel == "dither":
                check_dither_recipe(case)
                continue
            if kernel == "composite":
                check_composite_recipe(case)
                continue
            if kernel == "blend":
                check_blend_recipe(case)
                continue
            if kernel == "tail":
                check_tail_recipe(case)
                continue
            assert special_set(case) in matching_sets(case), case.id
            group = case.id.split("-")[0]
            for name, recipe in case.inputs.items():
                assert recipe["seed"] == seed_of(case.id)
                assert recipe["mapping"] == (
                    "signed" if name == "kernel" else "tiny" if group == "E" else "unit"
                ), case.id


def test_recipes_cover_every_conversion_dtype():
    # bits: the PCG64 raw words' little-endian bytes, reinterpreted as the dtype.
    assert CONVERT_DTYPES == (
        "float32",
        "float64",
        "uint8",
        "uint16",
        "int8",
        "int16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "bool",
    )
    for dtype in CONVERT_DTYPES:
        recipe = {"shape": [37], "seed": 9, "mapping": "bits", "specials": []}
        if dtype != "float32":
            recipe["dtype"] = dtype
        array = make_array(recipe)
        assert array.dtype == np.dtype(dtype), dtype
        assert array.shape == (37,), dtype
        size = 37 * np.dtype(dtype).itemsize
        words = np.random.PCG64(9).random_raw(-(-size // 8))
        assert array.tobytes() == words.astype("<u8").tobytes()[:size], dtype
    # ramp: arange(n) % 256, so 300 values hold every byte.
    ramp = make_array(
        {"shape": [300], "seed": 0, "mapping": "ramp", "specials": [], "dtype": "uint8"}
    )
    assert ramp.dtype == np.uint8
    assert np.array_equal(ramp, np.arange(300) % 256)
    # ties: per k in [-260, 260], the float32 nearest (k + 0.5) / 255 and its two
    # neighbours bracket the tie; where a float within 4 ulps scales to the exact
    # tie (255.0f * v == k + 0.5), it is there too.
    ties = golden_kernels.ties()
    assert ties.dtype == np.float32
    recipe = {"shape": [ties.size], "seed": 0, "mapping": "ties", "specials": []}
    assert make_array(recipe).tobytes() == ties.tobytes()
    values = ties.astype(np.float64)
    exact = 0
    for k in range(-260, 261):
        tie = (k + 0.5) / 255
        assert (values < tie).any() and (values > tie).any(), k
        nearest = np.float32(tie)
        neighbours = np.nextafter(nearest, np.array([-np.inf, np.inf], np.float32))
        assert np.isin([nearest, *neighbours], ties).all(), k
        exact += bool((np.float32(255) * ties == np.float32(k + 0.5)).any())
    assert exact > 0
    with pytest.raises(ValueError, match="ties"):
        make_array({**recipe, "shape": [ties.size + 1]})
    with pytest.raises(ValueError, match="float32"):
        make_array(
            {
                "shape": [4],
                "seed": 0,
                "mapping": "unit",
                "specials": [],
                "dtype": "uint8",
            }
        )


def test_integer_outputs_hash_as_little_endian_int32():
    for value in (0, 1, 4, 7, -1, -(2**31), 2**31 - 1):
        expected = hashlib.sha256(struct.pack("<i", value)).hexdigest()
        assert digest_int(value) == expected
    assert digest_int(5) == hashlib.sha256(b"\x05\x00\x00\x00").hexdigest()
    with pytest.raises(OverflowError):
        digest_int(2**31)


def test_mixed_digest_compares_nan_positions_and_non_nan_bits():
    bits = np.array(
        [0x3F800000, 0x7FC12345, 0x00000001, 0xFFC54321, 0x80000000, 0x7F800000],
        np.uint32,
    )
    payloads = bits.copy()
    payloads[1], payloads[3] = 0x7F812345, 0x7FC00000
    a, b = bits.view(np.float32), payloads.view(np.float32)
    assert digest(a, "mixed") == digest(b, "mixed")
    assert digest(a, "single") != digest(b, "single")
    assert digest(a, "single") == hashlib.sha256(bits.tobytes()).hexdigest()
    moved = bits.copy()
    moved[[1, 2]] = moved[[2, 1]]
    for payload in ("single", "mixed"):
        assert digest(a, payload) != digest(moved.view(np.float32), payload)
    # Every non-NaN bit counts, the sign of a negative denormal included.
    for index, value in ((4, 0x00000000), (4, 0x80000001), (5, 0xFF800000)):
        changed = bits.copy()
        changed[index] = value
        assert digest(a, "mixed") != digest(changed.view(np.float32), "mixed")
    # float64 (the tail's planes) by the same rule, with the default NaN
    # 0x7ff8000000000000; no other dtype.
    words = np.array(
        [
            0x3FF0000000000000,
            0x7FF8000000012345,
            0x0000000000000001,
            0xFFF0000000000001,
        ],
        np.uint64,
    )
    others = words.copy()
    others[1], others[3] = 0xFFF8000000000000, 0x7FF8000000000000
    a64, b64 = words.view(np.float64), others.view(np.float64)
    assert digest(a64, "mixed") == digest(b64, "mixed")
    assert digest(a64, "single") != digest(b64, "single")
    assert digest(a64, "single") == hashlib.sha256(words.tobytes()).hexdigest()
    changed = words.copy()
    changed[2] = 0x8000000000000001
    assert digest(a64, "mixed") != digest(changed.view(np.float64), "mixed")
    with pytest.raises(TypeError, match="float32 or float64"):
        digest(np.zeros(4, np.float16), "single")


def check_twins(cases, sensitive, output=None):
    """A daz_ftz case shares its inputs with its default twin; the ones sensitive()
    selects differ from it on B3, in `output` or else in any output (MXCSR is a
    first-class input, review I1). Returns those cases."""
    by_id = {case.id: (case, outputs) for case, outputs in cases}
    found = []
    for case, outputs in cases:
        if case.mxcsr != "daz_ftz":
            continue
        twin, twin_outputs = by_id[twin_id(case.id)]
        assert twin.inputs == case.inputs, case.id
        if sensitive(case):
            if output is None:
                assert outputs != twin_outputs, case.id
            else:
                assert outputs[output] != twin_outputs[output], case.id
            found.append(case)
    return found


def check_conversion_manifest(kernel, cases):
    """D3's groups (case counts per group) and their axes; outputs out and events.

    conversion: A, every (type, output, normalize) at 196,613 (one chunk at the
    conversion grain, 262,144) at default MXCSR, plus daz_ftz twins of the six
    float32 combos; L, the hot combos at 589,824 (3 chunks of 196,608) and 2,764,800
    (11 chunks of 251,345/251,346), float32 on clean and snan, uint8 on bits and
    ramp, both MXCSR states; T, ties for (0,1,0) and (0,1,1), both states.
    conversion_small: S, the hot combos at counts 1-17 in both states, float32 on
    (edges, snan, inf)[(count + combo index) % 3] (controller ruling), uint8 on ramp
    and bits.
    """
    groups = {}
    for case, outputs in cases:
        groups.setdefault(case.id.split("-")[0], []).append(case)
        check_conversion_recipe(case)
        assert sorted(outputs) == ["events", "out"], case.id
        assert all(len(value) == 64 for value in outputs.values()), case.id
    every = {(t, o, n) for t in range(11) for o in range(3) for n in range(2)}
    states = {"default", "daz_ftz"}
    if kernel == "conversion":
        assert {group: len(found) for group, found in groups.items()} == {
            "A": 72,
            "L": 40,
            "T": 4,
        }
        wide = {case.args["count"] for case in groups["A"]}
        assert wide == {196_613}
        assert {combo(c) for c in groups["A"] if c.mxcsr == "default"} == every
        assert {combo(c) for c in groups["A"] if c.mxcsr == "daz_ftz"} == {
            (0, o, n) for o in range(3) for n in range(2)
        }
        large = {
            (combo(c), c.args["count"], special_set(c), c.mxcsr) for c in groups["L"]
        }
        assert large == {
            (hot, count, name, state)
            for hot in HOT_COMBOS
            for count in (589_824, 2_764_800)
            for name in (("bits", "ramp") if hot[0] == 2 else ("clean", "snan"))
            for state in states
        }
        assert {(combo(c), c.mxcsr) for c in groups["T"]} == {
            (hot, state) for hot in ((0, 1, 0), (0, 1, 1)) for state in states
        }
    else:
        assert {group: len(found) for group, found in groups.items()} == {"S": 204}
        small = {
            (combo(c), c.args["count"], special_set(c), c.mxcsr) for c in groups["S"]
        }
        rotation = ("edges", "snan", "inf")
        assert small == {
            (hot, count, name, state)
            for index, hot in enumerate(HOT_COMBOS)
            for count in range(1, 18)
            for name in (
                ("ramp", "bits") if hot[0] == 2 else (rotation[(count + index) % 3],)
            )
            for state in states
        }
    # The clamp (np.clip's, Task 3b1) keeps a positive denormal at the default MXCSR
    # and gives +0 for it under DAZ, so every f32->f32 clamp case holding one differs
    # from its twin in out.
    sensitive = check_twins(
        cases,
        lambda case: combo(case) == (0, 0, 1) and has_positive_denormal(case),
        "out",
    )
    assert len(sensitive) >= 2


def check_border_manifest(cases):
    """convolution_border's groups (SP4b D8) and their axes; output out.

    R (108): the 61-tap reach, (1,61) on (3, n, c) and (61,1) on (n, 5, c) for n in
    58-66, c 1 or 3 by n, paddings 0/1/2, fused 0/8. W (66): (n, 34 - n, c) for n in
    1-33 with (3,3) and (11,11), each (form, padding, fused) and both c. Both clean
    at the default MXCSR. L (40): the plane (67, 201, 1) at padding 0, (1,61) and
    (61,1), fused 0/8, every border set, both states. T (8): the plane's src in the
    denormal range (tiny), both states. Z (48): the (5,5) DAZ-row kernel on
    (9, 13, 3) and (8, 11, 1), paddings 0/1/2, fused 0/8, clean and inf, both states.
    Every output is informative (not all NaN).
    """
    groups = {}
    for case, outputs in cases:
        check_border_recipe(case)
        groups.setdefault(case.id.split("-")[0], []).append(case)
        assert sorted(outputs) == ["out"], case.id
        assert len(outputs["out"]) == 64, case.id
        assert outputs["out"] not in all_nan_digests(case), case.id
    assert {group: len(found) for group, found in groups.items()} == {
        "R": 108,
        "W": 66,
        "L": 40,
        "T": 8,
        "Z": 48,
    }

    getters = {
        "form": lambda c: (c.args["kh"], c.args["kw"]),
        "shape": lambda c: (c.args["h"], c.args["w"], c.args["c"]),
        "padding": lambda c: c.args["padding"],
        "fused": lambda c: c.args["fused"],
        "set": special_set,
        "mxcsr": lambda c: c.mxcsr,
    }

    def axes(group, *names):
        return {tuple(getters[name](c) for name in names) for c in groups[group]}

    states = {"default", "daz_ftz"}
    assert axes("R", "form", "padding", "fused", "set", "mxcsr") == {
        (form, padding, fused, "clean", "default")
        for form in ((1, 61), (61, 1))
        for padding in (0, 1, 2)
        for fused in (0, 8)
    }
    assert axes("R", "form", "shape") == {
        ((1, 61), (3, n, (1, 3)[n % 2])) for n in range(58, 67)
    } | {((61, 1), (n, 5, (1, 3)[n % 2])) for n in range(58, 67)}
    assert len(axes("R", "form", "shape", "padding", "fused")) == 108
    shapes = [(c.args["h"], c.args["w"], c.args["c"]) for c in groups["W"]]
    assert {(h + w, c) for h, w, c in shapes} == {(34, 1), (34, 3)}
    assert {h for h, _, _ in shapes} == set(range(1, 34))
    assert len(axes("W", "form", "shape")) == 66
    assert axes("W", "form", "padding", "fused") == {
        (form, padding, fused)
        for form in ((3, 3), (11, 11))
        for padding in (0, 1, 2)
        for fused in (0, 8)
    }
    assert axes("W", "set", "mxcsr") == {("clean", "default")}
    assert axes("L", "form", "shape", "padding", "fused", "set", "mxcsr") == {
        (form, (67, 201, 1), 0, fused, name, state)
        for form in ((1, 61), (61, 1))
        for fused in (0, 8)
        for name in BORDER_SETS
        for state in states
    }
    assert axes("T", "form", "shape", "padding", "fused", "set", "mxcsr") == {
        (form, (67, 201, 1), 0, fused, "clean", state)
        for form in ((1, 61), (61, 1))
        for fused in (0, 8)
        for state in states
    }
    assert axes("Z", "form", "shape", "padding", "fused", "set", "mxcsr") == {
        ((5, 5), shape, padding, fused, name, state)
        for shape in ((9, 13, 3), (8, 11, 1))
        for padding in (0, 1, 2)
        for fused in (0, 8)
        for name in ("clean", "inf")
        for state in states
    }
    # Under DAZ the tiny src reads as zero; and B3 skips the Z kernel's denormal tap,
    # so an infinity that only that tap reads reaches an output at the default
    # MXCSR alone.
    check_twins(
        cases,
        lambda case: (
            case.id.startswith("T-")
            or (case.id.startswith("Z-") and special_set(case) == "inf")
        ),
    )


def check_lens_manifest(cases):
    """lens's groups (SP4b D3) and their axes; output out.

    W (48): count 196,613 (4 chunks of 49,153/49,154 at grain 65,536), accumulate 0/1,
    every (a, b) pair; every lens set at the default MXCSR, and edges, nan and inf
    (the sets holding denormals) under DAZ|FTZ. S (68): counts 1-17, accumulate 0/1,
    both states, the set LENS_SETS[count % 5] and the pair (count + accumulate) % 3.
    At the default MXCSR every single-payload output is compose_reference's (NumPy's
    float32 evaluation in compose_range's order). In the mixed W cases, f1's and
    f4's NaN payloads meet.
    """
    groups = {}
    for case, outputs in cases:
        check_lens_recipe(case)
        groups.setdefault(case.id.split("-")[0], []).append(case)
        assert sorted(outputs) == ["out"], case.id
        assert len(outputs["out"]) == 64, case.id
        if case.mxcsr == "default" and case.payloads == "single":
            reference = compose_reference(case)
            assert digest(reference, "single") == outputs["out"], case.id
    assert {group: len(found) for group, found in groups.items()} == {"W": 48, "S": 68}
    pairs = tuple(LENS_PAIRS)

    def axes(group):
        return {
            (
                c.args["count"],
                c.args["accumulate"],
                c.id.split("-")[2],
                special_set(c),
                c.mxcsr,
            )
            for c in groups[group]
        }

    assert axes("W") == {
        (golden_kernels.WIDE_COUNT, accumulate, pair, name, state)
        for state in ("default", "daz_ftz")
        for accumulate in (0, 1)
        for pair in pairs
        for name in LENS_SETS
        if state == "default" or name in ("edges", "nan", "inf")
    }
    assert axes("S") == {
        (
            count,
            accumulate,
            pairs[(count + accumulate) % 3],
            LENS_SETS[count % 5],
            state,
        )
        for count in range(1, 18)
        for accumulate in (0, 1)
        for state in ("default", "daz_ftz")
    }
    for case in groups["W"]:
        if case.payloads == "mixed":
            f1, f4 = (
                make_array(case.inputs[name]).view(np.uint32) for name in ("f1", "f4")
            )
            meet = nan_bits(f1) & nan_bits(f4) & (f1 != f4)
            assert meet.any(), case.id
    check_twins(cases, lambda case: False)


def check_power_manifest(cases):
    """lens_power's groups (SP4b Task 3b) and their axes; output out.

    S (204): pixels 1-17, channels 1/3/4, both states, two cases each: with k = pixels
    + channel index, the exponent k % 3 and the set k % 4, then the exponent
    (k + 1) % 3 and the set (k + 2) % 4; so every (channels, exponent, set, state)
    occurs. L (12): POWER_PIXELS, channels 1/3/4, the exponents 1/5 and 5, clean and
    edges, default MXCSR. A positive denormal source is a nonzero output under the
    exponents 1/5 and 1 and +0 under DAZ, so those daz_ftz cases differ from their
    default twins on B3.
    """
    groups = {}
    for case, outputs in cases:
        check_power_recipe(case)
        groups.setdefault(case.id.split("-")[0], []).append(case)
        assert sorted(outputs) == ["out"], case.id
        assert len(outputs["out"]) == 64, case.id
    assert {group: len(found) for group, found in groups.items()} == {"S": 204, "L": 12}
    names = {bits: name for name, bits in POWER_EXPONENTS.items()}
    exponents = tuple(POWER_EXPONENTS)
    states = ("default", "daz_ftz")

    def axes(group):
        return {
            (
                c.args["pixels"],
                c.args["channels"],
                names[c.args["exponent"]],
                special_set(c),
                c.mxcsr,
            )
            for c in groups[group]
        }

    small = axes("S")
    assert small == {
        (
            pixels,
            channels,
            exponents[(pixels + index + step) % 3],
            POWER_SETS[(pixels + index + 2 * step) % 4],
            state,
        )
        for pixels in range(1, 18)
        for index, channels in enumerate(POWER_CHANNELS)
        for step in (0, 1)
        for state in states
    }
    assert {axis[1:] for axis in small} == {
        (channels, exponent, name, state)
        for channels in POWER_CHANNELS
        for exponent in exponents
        for name in POWER_SETS
        for state in states
    }
    assert axes("L") == {
        (POWER_PIXELS, channels, exponent, name, "default")
        for channels in POWER_CHANNELS
        for exponent in ("fifth", "five")
        for name in ("clean", "edges")
    }
    sensitive = check_twins(
        cases,
        lambda case: (
            names[case.args["exponent"]] in ("fifth", "one") and positive_denormal(case)
        ),
        "out",
    )
    assert len(sensitive) >= 10


def exceptional_bits(bits, daz):
    """image_setup_ops.c's predicate on one float32 word: -0, a negative denormal under
    DAZ, an infinity or a NaN."""
    return (
        bits == 0x80000000
        or (daz and 0x80000000 < bits < 0x80800000)
        or (bits & 0x7F800000) == 0x7F800000
    )


def morph_form(case):
    """A morphology case's MORPH_FORMS arguments."""
    return {name: case.args[name] for name in MORPH_FORMS[case.entry]}


def morph_sets(case):
    """The MORPH_SETS whose payloads and specials reproduce a morphology case's src."""
    recipe = case.inputs["src"]
    return {
        name
        for name in MORPH_SETS
        if SPECIALS[name]["payloads"] == case.payloads
        and recipe["specials"] == expected_specials(name, "src", recipe["shape"])
    }


def check_morph_recipe(case, sweep=False):
    """A morphology case, seeded by id, single-payload. cn_image_exceptional: src
    (count,) on unit with one EXCEPTIONAL_WORDS word at index 0 or count - 1 (the id's
    position and word). The others: MORPH_FORMS arguments, src (h, w, c) holding its set
    at the index rule, on unit (tiny in group T; a clean sweep src on either), the id's
    third part the form's label."""
    args = case.args
    recipe = case.inputs["src"]
    assert list(case.inputs) == ["src"], case.id
    assert "dtype" not in recipe, case.id
    assert recipe["seed"] == seed_of(case.id), case.id
    assert case.payloads == "single", case.id
    if case.entry == EXCEPTIONAL_ENTRY:
        count = args["count"]
        assert sorted(args) == ["count"], case.id
        assert recipe["shape"] == [count] and recipe["mapping"] == "unit", case.id
        [(index, word)] = recipe["specials"]
        assert word in EXCEPTIONAL_WORDS and index in (0, count - 1), case.id
        if not sweep:
            _, position, named, _, _ = case.id.split("-")
            assert named == word, case.id
            assert index == (0 if position == "first" else count - 1), case.id
        return
    assert case.entry in (MORPH_ENTRY, FILTER_ENTRY), case.id
    form = MORPH_FORMS[case.entry]
    assert sorted(args) == sorted(["h", "w", "c", *form]), case.id
    assert all(args[name] in values for name, values in form.items()), case.id
    assert recipe["shape"] == [args["h"], args["w"], args["c"]], case.id
    names = morph_sets(case)
    if sweep:
        assert names, case.id
        assert recipe["mapping"] == "unit" or (
            recipe["mapping"] == "tiny" and "clean" in names
        ), case.id
        return
    group = case.id.split("-")[0]
    assert recipe["mapping"] == ("tiny" if group == "T" else "unit"), case.id
    assert special_set(case) in names, case.id
    assert case.id.split("-")[2] == morph_label(case.entry, morph_form(case)), case.id


def check_morphology_manifest(cases):
    """morphology's groups (SP4b D3, golden_kernels._morph_matrix) and their axes;
    outputs out (result for cn_image_exceptional).

    A (216) and F (36): every MORPH_FORMS form of cn_morphology_complete and
    cn_filter_morphology on MORPH_SHAPE, clean, default MXCSR. S (60): the node forms
    (DILATE, ERODE) at radii 1-3, iterations 2, every MORPH_SETS set, both states. W
    (17), H (9), B (2): widths, heights and the bench shape, clean, default. C (12):
    MORPH_CHUNKS, the node forms r2 x2, clean, denorm and nan, both states. T (8):
    MORPH_CHUNKS, tiny src, the ellipse and the cross as maximum and minimum, both
    states. X (68): cn_image_exceptional's counts, positions, words and states; each
    result is the predicate (exceptional_bits) of its one word under its state. Every
    out is informative (not all NaN). Under DAZ the T minimum forms differ from their
    default twins on B3 (denormals tie, and the ellipse combine returns +0 where the
    deques and the merge keep a src element's bits), and so do Erode's S cases on
    denorm and edges (negative denormals read as -0 tie with +0).
    """
    groups = {}
    for case, outputs in cases:
        check_morph_recipe(case)
        groups.setdefault(case.id.split("-")[0], []).append(case)
        if case.entry == EXCEPTIONAL_ENTRY:
            assert sorted(outputs) == ["result"], case.id
            [(_, word)] = case.inputs["src"]["specials"]
            expected = exceptional_bits(int(word, 16), case.mxcsr == "daz_ftz")
            assert outputs["result"] == digest_int(int(expected)), case.id
        else:
            assert sorted(outputs) == ["out"], case.id
            assert outputs["out"] not in all_nan_digests(case), case.id
    assert {group: len(found) for group, found in groups.items()} == {
        "A": 216,
        "F": 36,
        "S": 60,
        "W": 17,
        "H": 9,
        "B": 2,
        "C": 12,
        "T": 8,
        "X": 68,
    }

    def axes(group):
        return {
            (
                c.entry,
                tuple(sorted(morph_form(c).items())),
                (c.args["h"], c.args["w"], c.args["c"]),
                special_set(c),
                c.mxcsr,
            )
            for c in groups[group]
        }

    def node(entry, form, radius, iterations, **changes):
        arguments = {"radius": radius, "iterations": iterations, **form, **changes}
        return entry, tuple(sorted(arguments.items()))

    def product(entry):
        values = MORPH_FORMS[entry]
        names = tuple(values)
        return {
            tuple(sorted(zip(names, combination, strict=True)))
            for combination in itertools.product(*values.values())
        }

    states = ("default", "daz_ftz")
    for group, entry in (("A", MORPH_ENTRY), ("F", FILTER_ENTRY)):
        assert axes(group) == {
            (entry, form, MORPH_SHAPE, "clean", "default") for form in product(entry)
        }
    dilate = [node(MORPH_ENTRY, DILATE, r, 2) for r in (1, 2, 3)]
    erode = [node(FILTER_ENTRY, ERODE, r, 2) for r in (1, 2, 3)]
    assert axes("S") == {
        (*form, MORPH_SHAPE, name, state)
        for form in dilate + erode
        for name in MORPH_SETS
        for state in states
    }
    assert axes("W") == {
        (
            *(dilate[1] if w % 2 else erode[1]),
            (5, w, (1, 3, 4)[w % 3]),
            "clean",
            "default",
        )
        for w in range(1, 18)
    }
    assert axes("H") == {
        (
            *(
                node(MORPH_ENTRY, DILATE, 3, 1)
                if h % 2
                else node(FILTER_ENTRY, ERODE, 2, 1)
            ),
            (h, 9, 3),
            "clean",
            "default",
        )
        for h in range(1, 10)
    }
    assert axes("B") == {
        (*node(MORPH_ENTRY, DILATE, 3, 2), MORPH_BENCH, "clean", "default"),
        (*node(FILTER_ENTRY, ERODE, 2, 2), MORPH_BENCH, "clean", "default"),
    }
    assert axes("C") == {
        (*form, MORPH_CHUNKS, name, state)
        for form in (dilate[1], erode[1])
        for name in ("clean", "denorm", "nan")
        for state in states
    }
    ties = [
        dilate[1],
        node(MORPH_ENTRY, DILATE, 2, 2, maximum=0),
        node(FILTER_ENTRY, ERODE, 2, 2, maximum=1),
        erode[1],
    ]
    assert axes("T") == {
        (*form, MORPH_CHUNKS, "clean", state) for form in ties for state in states
    }
    exceptional = {
        (c.args["count"], c.id.split("-")[1], c.id.split("-")[2], c.mxcsr)
        for c in groups["X"]
    }
    words = EXCEPTIONAL_WORDS
    assert exceptional == {
        (count, position, words[(i + 6 * p) % 12], states[(i // 12 + p) % 2])
        for i, count in enumerate(EXCEPTIONAL_COUNTS)
        for p, position in enumerate(EXCEPTIONAL_POSITIONS)
    }
    assert {(word, position) for _, position, word, _ in exceptional} == {
        (word, position) for word in words for position in EXCEPTIONAL_POSITIONS
    }
    assert {(word, state) for _, _, word, state in exceptional} == {
        (word, state) for word in words for state in states
    }
    minimum_ties = {
        c.id for c in groups["T"] if c.mxcsr == "daz_ftz" and not c.args["maximum"]
    }
    erode_denormals = {
        c.id
        for c in groups["S"]
        if c.mxcsr == "daz_ftz"
        and c.entry == FILTER_ENTRY
        and special_set(c) in ("denorm", "edges")
    }
    # Group X has no default twins (one state per count and position).
    twins = [(case, outputs) for case, outputs in cases if case.id[0] != "X"]
    sensitive = check_twins(twins, lambda c: c.id in minimum_ties | erode_denormals)
    assert len(sensitive) == 2 + 6


def resample_sets(case):
    """The sets (clean or RESAMPLE_SETS) whose specials and payload rule
    (resample_payloads) reproduce a resample case's src."""
    recipe = case.inputs["src"]
    return {
        name
        for name in ("clean", *RESAMPLE_SETS)
        if case.payloads == resample_payloads(name, case.args["gamma"])
        and recipe["specials"] == expected_specials(name, "src", recipe["shape"])
    }


def check_resample_recipe(case, sweep=False):
    """A cn_resample_filtered case: filter, gamma, vector_clip and channels in range,
    src (h, w, c) on unit (a clean sweep src on unit or tiny) holding its set, seeded by
    id; the id's second and third parts name the arguments and the target."""
    args = case.args
    recipe = case.inputs["src"]
    assert case.entry == RESAMPLE_ENTRY, case.id
    assert list(case.inputs) == ["src"], case.id
    assert sorted(args) == ["c", "filter", "gamma", "h", "th", "tw", "vector_clip", "w"]
    assert args["filter"] in RESAMPLE_FILTERS, case.id
    assert args["gamma"] in (0, 1) and args["vector_clip"] in (0, 1), case.id
    assert 1 <= args["c"] <= 4, case.id
    assert recipe["shape"] == [args["h"], args["w"], args["c"]], case.id
    assert "dtype" not in recipe, case.id
    assert recipe["seed"] == seed_of(case.id), case.id
    names = resample_sets(case)
    if sweep:
        assert names, case.id
        assert recipe["mapping"] == "unit" or (
            recipe["mapping"] == "tiny" and "clean" in names
        ), case.id
        return
    parts = case.id.split("-")
    assert parts[1] == f"f{args['filter']}g{args['gamma']}v{args['vector_clip']}"
    assert parts[2] == f"{args['th']}x{args['tw']}", case.id
    assert recipe["mapping"] == "unit", case.id
    assert special_set(case) in names, case.id


def check_resample_manifest(cases):
    """resample's groups (golden_kernels._resample_matrix) and their axes; output out.

    G (176): every filter, gamma, vector_clip and channels 1-4 on RESAMPLE_GRID, clean,
    default. E (128): each RESAMPLE_EDGES geometry, filters 2 and 5, gamma, vector_clip,
    every RESAMPLE_SETS set, both states. B (2): RESAMPLE_BENCH, Hermite without gamma,
    clean, default. With gamma the nan set's cases are mixed (resample_payloads: B3's
    RGBA outputs hold both 0x7fc12345 and 0xffc00000). Every out is informative. Under
    DAZ every RGBA upscale on denorm differs from its default twin on B3.
    """
    groups = {}
    for case, outputs in cases:
        check_resample_recipe(case)
        groups.setdefault(case.id.split("-")[0], []).append(case)
        assert sorted(outputs) == ["out"], case.id
        args = case.args
        shape = (args["th"], args["tw"], args["c"])
        nan_digests = {
            digest(np.full(shape, bits, np.uint32).view(np.float32), case.payloads)
            for bits in (0x7FC12345, 0xFFC00000, 0x7FC00000)
        }
        assert outputs["out"] not in nan_digests, case.id
    assert {group: len(found) for group, found in groups.items()} == {
        "G": 176,
        "E": 128,
        "B": 2,
    }

    def axes(group):
        return {
            (
                (c.args["h"], c.args["w"], c.args["c"]),
                (c.args["th"], c.args["tw"]),
                c.args["filter"],
                c.args["gamma"],
                c.args["vector_clip"],
                special_set(c),
                c.mxcsr,
            )
            for c in groups[group]
        }

    (h, w), target = RESAMPLE_GRID
    assert axes("G") == {
        ((h, w, c), target, f, g, v, "clean", "default")
        for f in RESAMPLE_FILTERS
        for g in (0, 1)
        for v in (0, 1)
        for c in (1, 2, 3, 4)
    }
    assert axes("E") == {
        (shape, target, f, g, v, name, state)
        for shape, target in RESAMPLE_EDGES
        for f in (2, 5)
        for g in (0, 1)
        for v in (0, 1)
        for name in RESAMPLE_SETS
        for state in ("default", "daz_ftz")
    }
    assert axes("B") == {
        (shape, target, 5, 0, 0, "clean", "default") for shape, target in RESAMPLE_BENCH
    }
    mixed = {case.id for case, _ in cases if case.payloads == "mixed"}
    assert mixed == {
        case.id
        for case in groups["E"]
        if case.args["gamma"] and special_set(case) == "nan"
    }
    assert len(mixed) == 16
    rgba = RESAMPLE_EDGES[0][0]
    sensitive = check_twins(
        cases,
        lambda case: (
            (case.args["h"], case.args["w"], case.args["c"]) == rgba
            and special_set(case) == "denorm"
        ),
    )
    assert len(sensitive) == 8


EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()


def palette_sets(case):
    """The PALETTE_SETS whose payloads and specials reproduce a palette case's src."""
    recipe = case.inputs["src"]
    return {
        name
        for name in PALETTE_SETS
        if SPECIALS[name]["payloads"] == case.payloads
        and recipe["specials"] == expected_specials(name, "src", recipe["shape"])
    }


def check_palette_recipe(case, sweep=False):
    """A cn_palette_median_cut case: src (pixels, channels) float32 holding its set at
    the index rule, seeded by id, on unit (tiny for a clean group Z src, zeros for group
    M; unit or tiny for a clean sweep src); channels and capacity from the matrix's
    values; the id names the capacity, mapping, set and shape."""
    args = case.args
    recipe = case.inputs["src"]
    assert case.entry == PALETTE_ENTRY, case.id
    assert list(case.inputs) == ["src"], case.id
    assert sorted(args) == ["capacity", "channels", "pixels"], case.id
    assert args["channels"] in PALETTE_CHANNELS, case.id
    assert args["capacity"] in PALETTE_CAPACITIES, case.id
    assert recipe["shape"] == [args["pixels"], args["channels"]], case.id
    assert "dtype" not in recipe, case.id
    assert recipe["seed"] == seed_of(case.id), case.id
    names = palette_sets(case)
    if sweep:
        assert names, case.id
        assert 1 <= args["pixels"] <= golden_kernels.SWEEP_MAX_PIXELS, case.id
        assert recipe["mapping"] == "unit" or (
            recipe["mapping"] == "tiny" and "clean" in names
        ), case.id
        return
    group, capacity, mapping = case.id.split("-")[:3]
    assert capacity == f"k{args['capacity']}", case.id
    assert case.id.split("-")[-1] == f"{args['pixels']}x{args['channels']}", case.id
    if group == "N":
        # The set's NaN word alone, at the position the id names.
        index = golden_kernels.palette_nan_index(
            mapping, args["pixels"], args["channels"]
        )
        word = SPECIALS[special_set(case)]["values"][0]
        assert recipe["specials"] == [[index, word]], case.id
        assert recipe["mapping"] == "unit" and case.payloads == "single", case.id
        return
    assert mapping == recipe["mapping"], case.id
    tiny = group == "Z" and special_set(case) == "clean"
    expected = "zeros" if group == "M" else ("tiny" if tiny else "unit")
    assert recipe["mapping"] == expected, case.id
    assert special_set(case) in names, case.id
    assert case.payloads == SPECIALS[special_set(case)]["payloads"], case.id


def check_palette_manifest(cases):
    """palette's groups (golden_kernels._palette_matrix) and their axes; outputs out,
    written and status.

    A (171): every PALETTE_PIXELS count (index i) x channels (ci) x capacity (ki) on
    unit, the set PALETTE_SETS[(i + ci + ki) % 6] under the state (i + ki) % 2; every
    (set, state) pair occurs. B (9): PALETTE_BENCH pixels, every channels and capacity,
    clean, default. Z (108): PALETTE_TIE_PIXELS x channels x capacities x PALETTE_TIES,
    both states. N (16): the nan or snan set's NaN word alone after the first pixel
    (golden_kernels.palette_nan_index), PALETTE_NAN_POSITIONS x PALETTE_NAN_PIXELS x
    PALETTE_NAN_CHANNELS, capacity 2, default. M (16, Task 5b): PALETTE_MEDIAN_PIXELS x
    PALETTE_MEDIAN_CHANNELS x PALETTE_MEDIAN_CAPACITIES on zeros, clean, both states. A
    NaN anywhere in src makes the first split's channel NaN, so every nan, snan and
    mixed case returns PALETTE_EMPTY_CHILD with no color (status 6 appears); every other
    case status 0 with at least one color. Under DAZ every clean group Z case (tiny src)
    differs from its default twin on B3.
    """
    groups = {}
    for case, outputs in cases:
        check_palette_recipe(case)
        groups.setdefault(case.id.split("-")[0], []).append(case)
        assert sorted(outputs) == ["out", "status", "written"], case.id
        if special_set(case) in ("nan", "snan", "mixed"):
            assert outputs["status"] == digest_int(PALETTE_EMPTY_CHILD), case.id
            assert outputs["written"] == digest_int(0), case.id
            assert outputs["out"] == EMPTY_DIGEST, case.id
        else:
            assert outputs["status"] == digest_int(0), case.id
            assert outputs["written"] != digest_int(0), case.id
            assert outputs["out"] != EMPTY_DIGEST, case.id
    assert {group: len(found) for group, found in groups.items()} == {
        "A": 171,
        "B": 9,
        "Z": 108,
        "N": 16,
        "M": 16,
    }

    def axes(group):
        return {
            (
                c.args["pixels"],
                c.args["channels"],
                c.args["capacity"],
                special_set(c),
                c.inputs["src"]["mapping"],
                c.mxcsr,
            )
            for c in groups[group]
        }

    nan_axes = {
        (
            c.id.split("-")[2],
            c.args["pixels"],
            c.args["channels"],
            c.args["capacity"],
            special_set(c),
            c.mxcsr,
        )
        for c in groups["N"]
    }
    assert nan_axes == {
        (position, pixels, channels, 2, name, "default")
        for position in golden_kernels.PALETTE_NAN_POSITIONS
        for pixels in golden_kernels.PALETTE_NAN_PIXELS
        for channels in golden_kernels.PALETTE_NAN_CHANNELS
        for name in golden_kernels.PALETTE_NAN_SETS
    }

    states = ("default", "daz_ftz")
    grid = axes("A")
    assert grid == {
        (
            pixels,
            channels,
            capacity,
            PALETTE_SETS[(i + ci + ki) % 6],
            "unit",
            states[(i + ki) % 2],
        )
        for i, pixels in enumerate(PALETTE_PIXELS)
        for ci, channels in enumerate(PALETTE_CHANNELS)
        for ki, capacity in enumerate(PALETTE_CAPACITIES)
    }
    assert {(axis[3], axis[5]) for axis in grid} == {
        (name, state) for name in PALETTE_SETS for state in states
    }
    assert axes("B") == {
        (PALETTE_BENCH, channels, capacity, "clean", "unit", "default")
        for channels in PALETTE_CHANNELS
        for capacity in PALETTE_CAPACITIES
    }
    assert axes("Z") == {
        (pixels, channels, capacity, name, mapping, state)
        for pixels in PALETTE_TIE_PIXELS
        for channels in PALETTE_CHANNELS
        for capacity in PALETTE_CAPACITIES
        for name, mapping in PALETTE_TIES
        for state in states
    }
    assert axes("M") == {
        (pixels, channels, capacity, "clean", "zeros", state)
        for pixels in golden_kernels.PALETTE_MEDIAN_PIXELS
        for channels in golden_kernels.PALETTE_MEDIAN_CHANNELS
        for capacity in golden_kernels.PALETTE_MEDIAN_CAPACITIES
        for state in states
    }
    twins = [(case, outputs) for case, outputs in cases if case.id[0] == "Z"]
    sensitive = check_twins(twins, lambda c: c.inputs["src"]["mapping"] == "tiny")
    assert len(sensitive) == 18


def dither_sets(case):
    """The DITHER_SETS whose payloads and specials reproduce a dither case's src."""
    recipe = case.inputs["src"]
    return {
        name
        for name in DITHER_SETS
        if SPECIALS[name]["payloads"] == case.payloads
        and recipe["specials"] == expected_specials(name, "src", recipe["shape"])
    }


def dither_forms(case):
    """The DITHER_FORMS whose palette rule (mapping, the mixed set rotated by position 1
    when src holds it, then dither_form_specials) and src mapping (grid for the grid and
    daz forms) reproduce a dither case; a unit-form src may be on grid too (a clean
    sweep src)."""
    args = case.args
    src, palette = case.inputs["src"], case.inputs["palette"]
    count, c = args["count"], args["c"]
    found = set()
    for form in DITHER_FORMS:
        held = []
        if "mixed" in dither_sets(case):
            held = lens_specials("mixed", 1, count * c)
        mapping = "grid" if form == "grid" else "unit"
        mappings = ("grid",) if form in ("grid", "daz") else ("unit", "grid")
        if (
            palette["mapping"] == mapping
            and palette["specials"] == held + dither_form_specials(form, count, c)
            and src["mapping"] in mappings
        ):
            found.add(form)
    return found


def check_dither_recipe(case, sweep=False):
    """A dither case: src (h, w, c) and palette (1, count, c), each seeded by SHA-256 of
    the id without its MXCSR part and -name; src holds its set at the index rule; the
    palette and the src mapping follow the case's form (dither_forms); the entry's
    arguments: mode and algorithm (not riemersma's: apply 0, 2 or 3, dither 0 or 2, an
    algorithm only with mode 2) and history 2 and decay 0.5 (not cn_neighborhood_dither's).
    A matrix id names the kind, form, colors and mode."""
    args = case.args
    src, palette = case.inputs["src"], case.inputs["palette"]
    assert sorted(case.inputs) == ["palette", "src"], case.id
    keys = {"c", "count", "h", "w"}
    if case.entry != RIEMERSMA_ENTRY:
        keys |= {"mode", "algorithm"}
        modes = (0, 2, 3) if case.entry == DITHER_APPLY else (0, 2)
        assert args["mode"] in modes, case.id
        assert args["algorithm"] in (range(8) if args["mode"] == 2 else (0,)), case.id
    if case.entry != DITHER_ENTRY:
        keys |= {"history", "decay"}
        assert (args["history"], args["decay"]) == (DITHER_HISTORY, DITHER_DECAY)
    assert set(args) == keys, case.id
    assert args["c"] in DITHER_CHANNELS, case.id
    assert src["shape"] == [args["h"], args["w"], args["c"]], case.id
    assert palette["shape"] == [1, args["count"], args["c"]], case.id
    assert "dtype" not in src and "dtype" not in palette, case.id
    assert src["seed"] == lens_seed(case.id, "src"), case.id
    assert palette["seed"] == lens_seed(case.id, "palette"), case.id
    names, forms = dither_sets(case), dither_forms(case)
    assert names and forms, case.id
    if sweep:
        assert 1 <= args["count"] <= golden_kernels.SWEEP_MAX_COLORS, case.id
        return
    parts = case.id.split("-")
    kinds = {
        DITHER_APPLY: "apply",
        DITHER_ENTRY: "dither",
        RIEMERSMA_ENTRY: "riemersma",
    }
    assert parts[1] == kinds[case.entry], case.id
    assert parts[2] in forms, case.id
    assert parts[3] == f"n{args['count']}", case.id
    mode = args.get("mode", 3)
    assert parts[4] == f"m{mode}a{args.get('algorithm', 0)}", case.id
    assert parts[-1] == f"{args['h']}x{args['w']}x{args['c']}", case.id
    assert special_set(case) in names, case.id
    assert case.payloads == SPECIALS[special_set(case)]["payloads"], case.id
    unit_src = parts[2] in ("unit", "nonfinite", "nanfirst")
    assert src["mapping"] == ("unit" if unit_src else "grid"), case.id


def check_dither_manifest(cases):
    """dither's groups (golden_kernels._dither_matrix) and their axes; outputs out,
    compatible and status.

    On (*DITHER_SHAPE, c) unless named. P (60): apply, every DITHER_SIZES count (index i)
    x channels (ci) x modes 0 and 3 (mi), unit palette, the set ("clean", "edges",
    "nan")[(i + ci + mi) % 3] under the state (i + mi) % 2. D (24): apply, mode 2 x
    algorithms 0-7 (a) x channels at 16 colors, the set (a + ci) % 3 under the state
    a % 2. T (72): apply, the forms grid (40 colors), daz, nonfinite and nanfirst (16) x
    channels x modes 0, 2 and 3, clean, both states. S (24): apply, every DITHER_SETS set x
    channels x both states, mode 2, 16 colors. E (24): cn_neighborhood_dither (modes 0 and
    2) and riemersma at 16 and 300 colors on clean and nan, both states, 3 channels. B
    (1): apply on DITHER_BENCH, Floyd-Steinberg, 16 colors, clean, default. Every status
    is 0; compatible is 0 exactly for 300 colors with a NaN src (300 unique finite colors
    take the kd tree, which needs finite values), and then out is empty. A nanfirst
    palette's first sorted entry has a NaN distance to every color, so every pixel takes
    it. Under DAZ the daz form's mode 0 cases, and its one-channel cases in every mode,
    differ from their default twins on B3.
    """
    groups = {}
    for case, outputs in cases:
        check_dither_recipe(case)
        groups.setdefault(case.id.split("-")[0], []).append(case)
        assert sorted(outputs) == ["compatible", "out", "status"], case.id
        assert outputs["status"] == digest_int(0), case.id
        args = case.args
        incompatible = args["count"] == 300 and special_set(case) == "nan"
        assert outputs["compatible"] == digest_int(int(not incompatible)), case.id
        assert (outputs["out"] == EMPTY_DIGEST) == incompatible, case.id
        if case.id.split("-")[2] == "nanfirst":
            palette = make_array(case.inputs["palette"])
            shape = (args["h"], args["w"], args["c"])
            first = np.broadcast_to(palette[0, 0], shape)
            assert outputs["out"] == digest(first, case.payloads), case.id
    assert {group: len(found) for group, found in groups.items()} == {
        "P": 60,
        "D": 24,
        "T": 72,
        "S": 24,
        "E": 24,
        "B": 1,
    }

    def axes(group):
        return {
            (
                c.entry,
                c.id.split("-")[2],
                c.args["count"],
                c.args.get("mode", 3),
                c.args.get("algorithm", 0),
                special_set(c),
                c.mxcsr,
                (c.args["h"], c.args["w"], c.args["c"]),
            )
            for c in groups[group]
        }

    states = ("default", "daz_ftz")
    sets = ("clean", "edges", "nan")
    assert axes("P") == {
        (
            DITHER_APPLY,
            "unit",
            count,
            mode,
            0,
            sets[(i + ci + mi) % 3],
            states[(i + mi) % 2],
            (*DITHER_SHAPE, c),
        )
        for i, count in enumerate(DITHER_SIZES)
        for ci, c in enumerate(DITHER_CHANNELS)
        for mi, mode in enumerate((0, 3))
    }
    assert axes("D") == {
        (
            DITHER_APPLY,
            "unit",
            16,
            2,
            a,
            sets[(a + ci) % 3],
            states[a % 2],
            (*DITHER_SHAPE, c),
        )
        for a in range(8)
        for ci, c in enumerate(DITHER_CHANNELS)
    }
    assert axes("T") == {
        (
            DITHER_APPLY,
            form,
            40 if form == "grid" else 16,
            mode,
            0,
            "clean",
            state,
            (*DITHER_SHAPE, c),
        )
        for form in DITHER_FORMS[1:]
        for c in DITHER_CHANNELS
        for mode in (0, 2, 3)
        for state in states
    }
    assert axes("S") == {
        (DITHER_APPLY, "unit", 16, 2, 0, name, state, (*DITHER_SHAPE, c))
        for name in DITHER_SETS
        for c in DITHER_CHANNELS
        for state in states
    }
    assert axes("E") == {
        (entry, "unit", count, mode, 0, name, state, (*DITHER_SHAPE, 3))
        for entry, mode in ((DITHER_ENTRY, 0), (DITHER_ENTRY, 2), (RIEMERSMA_ENTRY, 3))
        for count in (16, 300)
        for name in ("clean", "nan")
        for state in states
    }
    assert axes("B") == {
        (DITHER_APPLY, "unit", 16, 2, 0, "clean", "default", DITHER_BENCH)
    }
    twins = [(case, outputs) for case, outputs in cases if case.id[0] in "TSE"]
    sensitive = check_twins(
        twins,
        lambda c: (
            c.id.split("-")[2] == "daz" and (c.args["mode"] == 0 or c.args["c"] == 1)
        ),
    )
    assert len(sensitive) == 5


COMPOSITE_INPUTS = golden_kernels.COMPOSITE_INPUTS  # base, overlay
COMPOSITE_ARGS = sorted(
    "bh bw bc base_constant oh ow oc overlay_constant mode x y crop".split()
)


def composite_sets(case):
    """The COMPOSITE_SETS whose payloads and specials reproduce both inputs of a
    composite case: the set at the index rule, the mixed set rotated by the input's
    position (base 0, overlay 1)."""
    return {
        name
        for name in golden_kernels.COMPOSITE_SETS
        if SPECIALS[name]["payloads"] == case.payloads
        and all(
            case.inputs[input_name]["specials"]
            == lens_specials(
                name, position, math.prod(case.inputs[input_name]["shape"])
            )
            for position, input_name in enumerate(COMPOSITE_INPUTS)
        )
    }


def composite_axes(case):
    """A composite case's (kind, mode, base (h, w, c, constant), overlay (h, w, c,
    constant), x, y, crop, set, MXCSR state, mapping)."""
    args = case.args
    return (
        case.id.split("-")[1],
        args["mode"],
        (args["bh"], args["bw"], args["bc"], args["base_constant"]),
        (args["oh"], args["ow"], args["oc"], args["overlay_constant"]),
        args["x"],
        args["y"],
        args["crop"],
        special_set(case),
        case.mxcsr,
        case.inputs["base"]["mapping"],
    )


def signed_label(value):
    """An offset in a composite id: n for a minus sign."""
    return f"n{-value}" if value < 0 else str(value)


def check_composite_recipe(case, sweep=False):
    """A cn_composite_canvas case: native_composite.canvas()'s arguments (each layer's
    h, w and c and whether it is a constant, mode 0-22, x, y, crop); not both layers
    constant, and a constant layer has the other's h and w (dimensions()). Each input is
    an image (h, w, c) or a constant's c floats, seeded by SHA-256 of the id without its
    MXCSR part and -name, holding its set (composite_sets), both on unit (on tiny in
    group T; a clean sweep case on unit or tiny). A matrix id names the mode, layers,
    geometry and sizes."""
    args = case.args
    assert case.entry == golden_kernels.COMPOSITE_ENTRY, case.id
    assert sorted(case.inputs) == list(COMPOSITE_INPUTS), case.id
    assert sorted(args) == COMPOSITE_ARGS, case.id
    assert args["bc"] in (1, 3, 4) and args["oc"] in (1, 3, 4), case.id
    assert args["mode"] in range(23) and args["crop"] in (0, 1), case.id
    assert args["base_constant"] in (0, 1) and args["overlay_constant"] in (0, 1)
    assert not (args["base_constant"] and args["overlay_constant"]), case.id
    if args["base_constant"] or args["overlay_constant"]:
        assert (args["bh"], args["bw"]) == (args["oh"], args["ow"]), case.id
    mappings = set()
    for name, prefix in zip(COMPOSITE_INPUTS, "bo", strict=True):
        recipe = case.inputs[name]
        shape = [args[f"{prefix}h"], args[f"{prefix}w"], args[f"{prefix}c"]]
        assert recipe["shape"] == (shape[2:] if args[f"{name}_constant"] else shape)
        assert "dtype" not in recipe, case.id
        assert recipe["seed"] == lens_seed(case.id, name), case.id
        mappings.add(recipe["mapping"])
    names = composite_sets(case)
    assert names, case.id
    if sweep:
        side = golden_kernels.SWEEP_MAX_SIDE
        assert all(1 <= args[key] <= side for key in ("bh", "bw", "oh", "ow")), case.id
        assert mappings == {"unit"} or (mappings == {"tiny"} and "clean" in names)
        return
    parts = case.id.split("-")
    marks = ["k" if args[f"{name}_constant"] else "" for name in COMPOSITE_INPUTS]
    assert parts[2] == f"m{args['mode']}", case.id
    assert parts[3] == f"b{args['bc']}{marks[0]}o{args['oc']}{marks[1]}", case.id
    geometry = f"x{signed_label(args['x'])}y{signed_label(args['y'])}c{args['crop']}"
    assert parts[4] == geometry, case.id
    sizes = f"{args['bh']}x{args['bw']}o{args['oh']}x{args['ow']}"
    assert parts[-1] == sizes, case.id
    assert special_set(case) in names, case.id
    assert case.payloads == SPECIALS[special_set(case)]["payloads"], case.id
    assert mappings == ({"tiny"} if parts[0] == "T" else {"unit"}), case.id


def shortcut_clauses(args):
    """D13's shortcut predicate (full_overlap in composite_complete_ops.c) on the
    geometry golden_kernels.composite_geometry gives a composite case's args, clause by
    clause: the paste at (0, 0) over the whole canvas; the base at (0, 0) with the
    canvas's size; the base not constant; the base's channels the canvas's (4 when
    padded); the overlay not constant; the overlay region the whole overlay at (0, 0).
    The shortcut runs when every clause holds."""
    h, w, _, top, left, pt, pl, ot, ol, rh, rw, padded = (
        golden_kernels.composite_geometry(args)
    )
    return {
        "paste": (pt, pl, rh, rw) == (0, 0, h, w),
        "base": (top, left, args["bh"], args["bw"]) == (0, 0, h, w),
        "base_constant": not args["base_constant"],
        "channels": args["bc"] == (4 if padded else args["bc"]),
        "overlay_constant": not args["overlay_constant"],
        "overlay": (ot, ol, args["oh"], args["ow"]) == (0, 0, rh, rw),
    }


def rotated(k):
    """A composite or blend group's k-th case's set and MXCSR state:
    COMPOSITE_SETS[k % 4] (BLEND_SETS is the same tuple) under the state (k // 4) % 2,
    so every (set, state) pair occurs once in each 8 cases."""
    return golden_kernels.COMPOSITE_SETS[k % 4], MXCSR_STATES[k // 4 % 2]


def check_composite_manifest(cases):
    """composite's groups (golden_kernels._composite_matrix) and their axes; output out.

    Layers are two COMPOSITE_SHAPE images unless named; rotated(k) gives the k-th case's
    set and state in each group but T and B. S (72): D13's shortcut, every
    COMPOSITE_PAIRS (base, overlay) channel pair p x SHORTCUT_MODES m (k = 8 p + m) at
    a full overlap (x = y = 0), crop (p + m) % 2. N (40): its near misses, every
    NEAR_MISSES kind x NEAR_PAIRS x NEAR_MODES. A (69): every mode x the blend's three
    paths, (3, 3), (3, 4) and (4, 4), at a full overlap. G (40): the overlay
    COMPOSITE_SMALL at every COMPOSITE_OFFSETS (x, y) x crop, then at the DISJOINT
    offsets with crop (no region), x GEOMETRY_PAIRS; mode k % 23. C (12): a constant
    base or overlay x COLOR_PAIRS x COLOR_OFFSETS, mode COLOR_MODES[k % 4], crop k % 2
    (which canvas() ignores with a constant). T (8): DAZ_PAIRS x DAZ_MODES x both
    states, clean on tiny, at a full overlap. B (1): COMPOSITE_BENCH, mode 1, a full
    overlap, clean, default. The shortcut (shortcut_clauses) runs exactly in S, A, T and
    B. Under DAZ every T case differs from its default twin on B3.
    """
    groups = {}
    for case, outputs in cases:
        check_composite_recipe(case)
        groups.setdefault(case.id.split("-")[0], []).append(case)
        assert sorted(outputs) == ["out"], case.id
        assert len(outputs["out"]) == 64, case.id
        takes = all(shortcut_clauses(case.args).values())
        assert takes == (case.id.split("-")[0] in ("S", "A", "T", "B")), case.id
    assert {group: len(found) for group, found in groups.items()} == {
        "S": 72,
        "N": 40,
        "A": 69,
        "G": 40,
        "C": 12,
        "T": 8,
        "B": 1,
    }

    def axes(group):
        return {composite_axes(case) for case in groups[group]}

    h, w = golden_kernels.COMPOSITE_SHAPE

    def image(c, size=(h, w)):
        return (*size, c, 0)

    pairs = golden_kernels.COMPOSITE_PAIRS
    modes = golden_kernels.SHORTCUT_MODES
    expected = set()
    for p, (bc, oc) in enumerate(pairs):
        for m, mode in enumerate(modes):
            k = len(modes) * p + m
            full = (image(bc), image(oc), 0, 0, (p + m) % 2)
            expected.add(("full", mode, *full, *rotated(k), "unit"))
    assert axes("S") == expected
    assert {axis[7:9] for axis in expected} == {
        (name, state)
        for name in golden_kernels.COMPOSITE_SETS
        for state in MXCSR_STATES
    }
    expected = set()
    near = itertools.product(
        golden_kernels.NEAR_MISSES.items(),
        golden_kernels.NEAR_PAIRS,
        golden_kernels.NEAR_MODES,
    )
    for k, ((kind, form), (bc, oc), mode) in enumerate(near):
        size, x, y, crop, base_constant, overlay_constant = form
        base = (h, w, bc, base_constant)
        overlay = (*size, oc, overlay_constant)
        expected.add((kind, mode, base, overlay, x, y, crop, *rotated(k), "unit"))
    assert axes("N") == expected
    expected = set()
    paths = itertools.product(range(23), golden_kernels.BLEND_PATHS)
    for k, (mode, (bc, oc)) in enumerate(paths):
        expected.add(("full", mode, image(bc), image(oc), 0, 0, 0, *rotated(k), "unit"))
    assert axes("A") == expected
    offsets = golden_kernels.COMPOSITE_OFFSETS
    places = [(x, y, crop) for x in offsets for y in offsets for crop in (0, 1)]
    places += [(x, y, 1) for x, y in golden_kernels.DISJOINT]
    small = golden_kernels.COMPOSITE_SMALL
    expected = set()
    geometries = itertools.product(golden_kernels.GEOMETRY_PAIRS, places)
    for k, ((bc, oc), place) in enumerate(geometries):
        layers = (image(bc), image(oc, small))
        expected.add(("geo", k % 23, *layers, *place, *rotated(k), "unit"))
    assert axes("G") == expected
    expected = set()
    colors = itertools.product(
        COMPOSITE_INPUTS, golden_kernels.COLOR_PAIRS, golden_kernels.COLOR_OFFSETS
    )
    for k, (constant, (bc, oc), (x, y)) in enumerate(colors):
        base = (h, w, bc, int(constant == "base"))
        overlay = (h, w, oc, int(constant == "overlay"))
        mode = golden_kernels.COLOR_MODES[k % 4]
        expected.add(("color", mode, base, overlay, x, y, k % 2, *rotated(k), "unit"))
    assert axes("C") == expected
    assert axes("T") == {
        ("tiny", mode, image(bc), image(oc), 0, 0, 0, "clean", state, "tiny")
        for bc, oc in golden_kernels.DAZ_PAIRS
        for mode in golden_kernels.DAZ_MODES
        for state in MXCSR_STATES
    }
    bh, bw, bc = golden_kernels.COMPOSITE_BENCH
    bench = (bh, bw, bc, 0)
    assert axes("B") == {
        ("bench", 1, bench, bench, 0, 0, 0, "clean", "default", "unit")
    }
    twins = [(case, outputs) for case, outputs in cases if case.id[0] == "T"]
    assert len(check_twins(twins, lambda case: True)) == 4


def blend_sets(case):
    """The BLEND_SETS whose payloads and specials reproduce both inputs of a blend
    case: the set at the index rule, the mixed set rotated by the input's position
    (overlay 0, base 1)."""
    return {
        name
        for name in golden_kernels.BLEND_SETS
        if SPECIALS[name]["payloads"] == case.payloads
        and all(
            case.inputs[input_name]["specials"]
            == lens_specials(
                name, position, math.prod(case.inputs[input_name]["shape"])
            )
            for position, input_name in enumerate(golden_kernels.BLEND_INPUTS)
        )
    }


def check_blend_recipe(case, sweep=False):
    """A blend case: cn_blend_images (args pixels, oc, bc, mode; overlay (pixels, oc)
    and base (pixels, bc)) or cn_blend_mode (args count, mode; both inputs (count,)),
    mode 0-22. Each input is seeded by SHA-256 of the id without its MXCSR part and
    -name and holds its set (blend_sets), on unit (group T: the overlay on tiny; a
    clean sweep case on unit or tiny, both inputs alike). A matrix id names the kind,
    mode, channels and count."""
    args = case.args
    images = case.entry == golden_kernels.BLEND_IMAGES
    assert case.entry in golden_kernels.KERNEL_ENTRIES["blend"], case.id
    assert sorted(case.inputs) == sorted(golden_kernels.BLEND_INPUTS), case.id
    keys = ["bc", "mode", "oc", "pixels"] if images else ["count", "mode"]
    assert sorted(args) == keys, case.id
    assert args["mode"] in range(23), case.id
    count = args["pixels"] if images else args["count"]
    mappings = []
    for name in golden_kernels.BLEND_INPUTS:
        recipe = case.inputs[name]
        if images:
            channels = args["oc" if name == "overlay" else "bc"]
            assert channels in (1, 3, 4), case.id
            assert recipe["shape"] == [count, channels], case.id
        else:
            assert recipe["shape"] == [count], case.id
        assert "dtype" not in recipe, case.id
        assert recipe["seed"] == lens_seed(case.id, name), case.id
        mappings.append(recipe["mapping"])
    names = blend_sets(case)
    assert names, case.id
    if sweep:
        target = max(args["oc"], args["bc"]) if images else 1
        assert 1 <= count <= golden_kernels.SWEEP_MAX_COUNT // target, case.id
        assert mappings in (["unit"] * 2, ["tiny"] * 2), case.id
        assert mappings[0] == "unit" or "clean" in names, case.id
        return
    parts = case.id.split("-")
    assert parts[1] == ("images" if images else "mode"), case.id
    assert parts[2] == f"m{args['mode']}", case.id
    label = f"o{args['oc']}b{args['bc']}" if images else "raw"
    assert parts[3] == label and parts[-1] == str(count), case.id
    assert special_set(case) in names, case.id
    assert case.payloads == SPECIALS[special_set(case)]["payloads"], case.id
    expected = ["tiny", "unit"] if parts[0] == "T" else ["unit", "unit"]
    assert mappings == expected, case.id


def check_blend_manifest(cases):
    """blend's groups (golden_kernels._blend_matrix) and their axes; output out.

    rotated(k) gives the k-th case's set and state in each group but T. M (207):
    cn_blend_images, every mode x BLEND_PAIRS (overlay, base) channel pair, k-th at
    SMALL_COUNTS[k % 17] pixels. S (68): SIMD_MODES x SIMD_PAIRS (the AVX2 unit's
    8-lane form) x pixels 1-17. W (4): SIMD_MODES x SIMD_PAIRS at WIDE_COUNT pixels.
    T (8): SIMD_MODES x SIMD_PAIRS x both states at BLEND_DAZ_PIXELS, clean, the
    overlay on tiny and the base on unit. R (25): cn_blend_mode, every mode, k-th at
    SMALL_COUNTS[k % 17] elements, then modes 0 and 1 at WIDE_COUNT. Under DAZ the T
    MULTIPLY cases differ from their default twins on B3 (the products of denormals are
    flushed) and the NORMAL ones equal them (a copy of the overlay).
    """
    groups = {}
    for case, outputs in cases:
        check_blend_recipe(case)
        groups.setdefault(case.id.split("-")[0], []).append(case)
        assert sorted(outputs) == ["out"], case.id
        assert len(outputs["out"]) == 64, case.id
    assert {group: len(found) for group, found in groups.items()} == {
        "M": 207,
        "S": 68,
        "W": 4,
        "T": 8,
        "R": 25,
    }

    def axes(group):
        return {
            (
                c.entry,
                c.args["mode"],
                c.args.get("oc"),
                c.args.get("bc"),
                c.args.get("pixels", c.args.get("count")),
                special_set(c),
                c.mxcsr,
            )
            for c in groups[group]
        }

    images, raw = golden_kernels.BLEND_IMAGES, golden_kernels.BLEND_MODE
    small = golden_kernels.SMALL_COUNTS
    wide = golden_kernels.WIDE_COUNT
    simd = list(itertools.product(golden_kernels.SIMD_MODES, golden_kernels.SIMD_PAIRS))
    pairs = itertools.product(range(23), golden_kernels.BLEND_PAIRS)
    assert axes("M") == {
        (images, mode, oc, bc, small[k % 17], *rotated(k))
        for k, (mode, (oc, bc)) in enumerate(pairs)
    }
    forms = itertools.product(simd, small)
    assert axes("S") == {
        (images, mode, oc, bc, pixels, *rotated(k))
        for k, ((mode, (oc, bc)), pixels) in enumerate(forms)
    }
    assert axes("W") == {
        (images, mode, oc, bc, wide, *rotated(k))
        for k, (mode, (oc, bc)) in enumerate(simd)
    }
    assert axes("T") == {
        (images, mode, oc, bc, golden_kernels.BLEND_DAZ_PIXELS, "clean", state)
        for mode, (oc, bc) in simd
        for state in MXCSR_STATES
    }
    modes = [(mode, small[k % 17]) for k, mode in enumerate(range(23))]
    modes += [(mode, wide) for mode in golden_kernels.SIMD_MODES]
    assert axes("R") == {
        (raw, mode, None, None, count, *rotated(k))
        for k, (mode, count) in enumerate(modes)
    }
    twins = [(case, outputs) for case, outputs in cases if case.id[0] == "T"]
    sensitive = check_twins(twins, lambda case: case.args["mode"] == 1)
    assert len(sensitive) == 2
    by_id = {case.id: outputs for case, outputs in twins}
    for case, outputs in twins:
        if case.mxcsr == "daz_ftz" and case.args["mode"] == 0:
            assert outputs == by_id[twin_id(case.id)], case.id


TILE = golden_kernels.TILE_ENTRY
STORE = golden_kernels.STORE_ENTRY
MULTIPLY = golden_kernels.MULTIPLY_ENTRY
NORMAL = golden_kernels.NORMAL_ENTRY
CAPTION = golden_kernels.CAPTION_ENTRY
TAIL_OUTPUTS = {TILE: "dst", STORE: "dst", MULTIPLY: "a", NORMAL: "dst", CAPTION: "out"}


def tail_inputs(case):
    """A tail case's inputs in call order: name -> (shape, dtype, mapping). The double
    inputs and the caption raster hold PCG64 words (bits); store's dst is in-out."""
    a = case.args
    if case.entry == TILE:
        return {"src": ([a["h"], a["w"], a["c"]], "float32", "unit")}
    if case.entry == STORE:
        return {
            "src": ([a["dh"], a["dw"]], "float64", "bits"),
            "dst": ([a["oh"], a["ow"], a["c"]], "float32", "unit"),
        }
    if case.entry == MULTIPLY:
        return {name: ([a["h"], a["w"]], "float64", "bits") for name in ("a", "b")}
    if case.entry == NORMAL:
        found = {name: ([a["pixels"]], "float32", "signed") for name in ("dx", "dy")}
        if a["alpha"]:
            found["alpha"] = ([a["pixels"]], "float32", "unit")
        return found
    return {
        "image": ([a["height"], a["width"], a["channels"]], "float32", "unit"),
        "caption": ([a["caption_height"], a["width"]], "uint8", "bits"),
    }


def tail_sets(case):
    """The sets reproducing a tail case's float32 inputs: each holds the set at the index
    rule, a mixed set rotated by its call position; bits when the case is the store's or
    the multiply's and its float32 input (store's dst) holds none."""
    if case.entry in (STORE, MULTIPLY):
        held = any(recipe["specials"] for recipe in case.inputs.values())
        return set() if held else {"bits"}
    position = {name: i for i, name in enumerate(tail_inputs(case))}
    floats = {
        name: recipe
        for name, recipe in case.inputs.items()
        if recipe.get("dtype", "float32") == "float32"
    }
    return {
        name
        for name in SPECIALS
        if SPECIALS[name]["payloads"] == case.payloads
        and all(
            recipe["specials"]
            == lens_specials(name, position[input_name], math.prod(recipe["shape"]))
            for input_name, recipe in floats.items()
        )
    }


def tail_label(case):
    """A tail case's form and size id parts (golden_kernels.tail_label's format)."""
    a = case.args
    if case.entry == TILE:
        form = f"b{a['border']}p{a['padding']}y{a['y']}x{a['x']}t{a['th']}x{a['tw']}"
        size = f"{a['h']}x{a['w']}x{a['c']}k{a['kh']}x{a['kw']}d{a['dh']}x{a['dw']}"
        return form + f"c{a['channel']}", size
    if case.entry == STORE:
        form = f"y{a['y']}x{a['x']}t{a['th']}x{a['tw']}c{a['channel']}"
        return form, f"{a['oh']}x{a['ow']}x{a['c']}d{a['dh']}x{a['dw']}"
    if case.entry == MULTIPLY:
        return "plane", f"{a['h']}x{a['w']}"
    if case.entry == NORMAL:
        form = f"c{a['channels']}r{a['invert_r']}g{a['invert_g']}a{a['alpha']}"
        return form, str(a["pixels"])
    form = f"t{a['top']}c{a['channels']}k{a['caption_height']}"
    return form, f"{a['height']}x{a['width']}"


def check_tail_args(case):
    """A tail case's args: the entry's parameters (golden_kernels.TAIL_PARAMS, and alpha
    for the normal output), each in the range its entry accepts with status 0."""
    a = case.args
    params = golden_kernels.TAIL_PARAMS[case.entry]
    assert sorted(a) == sorted([*params, "alpha"] if case.entry == NORMAL else params)
    if case.entry == TILE:
        oh, ow = a["h"] + 2 * a["padding"], a["w"] + 2 * a["padding"]
        assert 1 <= a["c"] <= 512 and a["channel"] < a["c"], case.id
        assert a["padding"] in (0, 2) and a["border"] in range(5), case.id
        assert 0 <= a["y"] < oh and 1 <= a["th"] <= oh - a["y"], case.id
        assert 0 <= a["x"] < ow and 1 <= a["tw"] <= ow - a["x"], case.id
        assert a["dh"] >= a["th"] + a["kh"] - 1 and a["dw"] >= a["tw"] + a["kw"] - 1
    elif case.entry == STORE:
        assert 1 <= a["c"] <= 512 and a["channel"] < a["c"], case.id
        assert 0 <= a["y"] < a["oh"] and 1 <= a["th"] <= a["oh"] - a["y"], case.id
        assert 0 <= a["x"] < a["ow"] and 1 <= a["tw"] <= a["ow"] - a["x"], case.id
        assert a["th"] <= a["dh"] and a["tw"] <= a["dw"], case.id
    elif case.entry == MULTIPLY:
        assert a["h"] >= 1 and a["w"] >= 1, case.id
    elif case.entry == NORMAL:
        assert a["pixels"] >= 1 and a["channels"] in (3, 4), case.id
        assert {a["invert_r"], a["invert_g"], a["alpha"]} <= {0, 1}, case.id
    else:
        assert a["channels"] in (1, 3, 4) and a["top"] in (0, 1), case.id
        assert min(a["height"], a["width"], a["caption_height"]) >= 1, case.id


def check_tail_recipe(case, sweep=False):
    """A tail case: its args (check_tail_args); its inputs (tail_inputs), each seeded by
    SHA-256 of the id without its MXCSR part and -name; a double input or the caption
    raster on bits with no specials and its dtype named; every float32 input holding the
    set (tail_sets) on its mapping, or every one on tiny in a clean case (tile group T;
    a clean sweep case on either). Payloads: the set's; single for the store and mixed
    for the multiply. A matrix id names the kind, form, set and size."""
    assert case.entry in (TILE, STORE, MULTIPLY, NORMAL, CAPTION), case.id
    check_tail_args(case)
    expected = tail_inputs(case)
    assert sorted(case.inputs) == sorted(expected), case.id
    mappings = set()
    for name, (shape, dtype, mapping) in expected.items():
        recipe = case.inputs[name]
        assert recipe["shape"] == shape, case.id
        assert recipe.get("dtype", "float32") == dtype, case.id
        assert ("dtype" in recipe) == (dtype != "float32"), case.id
        assert recipe["seed"] == lens_seed(case.id, name), case.id
        if dtype != "float32":
            assert recipe["mapping"] == "bits" and recipe["specials"] == [], case.id
        else:
            assert recipe["mapping"] in (mapping, "tiny"), case.id
            mappings.add(recipe["mapping"] == "tiny")
    names = tail_sets(case)
    assert names, case.id
    assert mappings <= {False} or (mappings == {True} and "clean" in names), case.id
    if case.entry in (STORE, MULTIPLY):
        assert case.payloads == ("single" if case.entry == STORE else "mixed"), case.id
    if sweep:
        return
    parts = case.id.split("-")
    assert len(parts) == 6 and parts[1] == golden_kernels.TAIL_KINDS[case.entry]
    assert (parts[2], parts[-1]) == tail_label(case), case.id
    assert special_set(case) in names, case.id
    if special_set(case) != "bits":
        assert case.payloads == SPECIALS[special_set(case)]["payloads"], case.id
    assert mappings <= {(parts[0], case.entry) == ("T", TILE)}, case.id


def filter2d_plane(kh, kw, oh, ow):
    """native_spectral_filter.filter2d's tile (bh, bw) and plane (dh, dw) for a (kh, kw)
    kernel over an (oh, ow) padded image."""
    bw = min(max(round(kw * 4.5), 256 - kw + 1), ow)
    bh = min(max(round(kh * 4.5), 256 - kh + 1), oh)
    dw = max(cv2.getOptimalDFTSize(bw + kw - 1), 2)
    dh = cv2.getOptimalDFTSize(bh + kh - 1)
    return (min(dh - kh + 1, oh), min(dw - kw + 1, ow)), (dh, dw)


def tile_case_args(shape, padding, kernel, origin, size, plane, channel, border):
    return {
        "h": shape[0],
        "w": shape[1],
        "c": shape[2],
        "padding": padding,
        "y": origin[0],
        "x": origin[1],
        "th": size[0],
        "tw": size[1],
        "kh": kernel[0],
        "kw": kernel[1],
        "dh": plane[0],
        "dw": plane[1],
        "channel": channel,
        "border": border,
    }


def check_tail_manifest(cases):
    """tail's groups (golden_kernels._tail_matrix), keyed (kind, group), and their axes;
    one output per entry (TAIL_OUTPUTS).

    cn_spectral_tile on SPECTRAL_SHAPE (13, 17, 3) and a (5, 7) kernel, tiles of (6, 8)
    in (11, 15) planes, rotated(k)'s set and state: G (30) every padding (0, 2) x border
    (0-4) x the first, inner and last tile, channel k % 3. R (20): (1, 1, 1) and (2, 3, 2)
    x every border x padding, one tile of the whole padded image under a (9, 11) kernel,
    the last channel. D (4): the first tile, reflect101, channel 1, edges, both paddings
    and states. T (2): as D at padding 0, clean on tiny. W (3): TILE_WIDE. B (1): the
    bench's tile, filter2d's own (filter2d_plane). cn_spectral_store, its plane on bits:
    G (12) channels 1 and 3 x the three tiles of a (13, 17) result from (11, 15) planes x
    both states; D (4) STORE_DAZ whole, both states; W (1); B (1). cn_spectral_multiply
    on bits: S (16) heights 1, 2, 7, 8 x widths 1, 2, 9, 10, the state k % 2; D (4); W
    (1); B (1) the bench plane. cn_normal_output: F (48) channels x invert_r x invert_g
    x alpha x NORMAL_SETS (edges, nan, inf) at SMALL_COUNTS[k % 17] pixels, the state
    (k // 3) % 2; D (4); W (2); B (1). cn_caption_compose: F (18) top x channels 1/3/4 x
    CAPTION_SIZES, rotated(k); D (4); W (1); B (1). Under DAZ the tile's D and T, the
    store's D and the multiply's D cases differ from their default twins on B3 (a
    denormal source reads as zero, a narrowed or multiplied denormal is flushed); the
    normal output's and the caption's D cases equal theirs (the normal's + 1 absorbs a
    flushed quotient, the caption copies the image).
    """
    groups = {}
    for case, outputs in cases:
        check_tail_recipe(case)
        group, kind = case.id.split("-")[:2]
        groups.setdefault((kind, group), []).append(case)
        assert sorted(outputs) == [TAIL_OUTPUTS[case.entry]], case.id
        assert len(outputs[TAIL_OUTPUTS[case.entry]]) == 64, case.id
    counts = {
        "tile": {"G": 30, "R": 20, "D": 4, "T": 2, "W": 3, "B": 1},
        "store": {"G": 12, "D": 4, "W": 1, "B": 1},
        "multiply": {"S": 16, "D": 4, "W": 1, "B": 1},
        "normal": {"F": 48, "D": 4, "W": 2, "B": 1},
        "caption": {"F": 18, "D": 4, "W": 1, "B": 1},
    }
    assert {key: len(found) for key, found in groups.items()} == {
        (kind, group): count
        for kind, found in counts.items()
        for group, count in found.items()
    }

    def axes(kind, group):
        return {
            (
                tuple(sorted(c.args.items())),
                special_set(c),
                c.mxcsr,
                tuple(sorted(r["mapping"] for r in c.inputs.values())),
            )
            for c in groups[(kind, group)]
        }

    def frozen(args, *rest):
        return (tuple(sorted(args.items())), *rest)

    shape, kernel, plane = (13, 17, 3), (5, 7), (11, 15)
    origins = {
        0: [((0, 0), (6, 8)), ((6, 8), (6, 8)), ((12, 16), (1, 1))],
        2: [((0, 0), (6, 8)), ((6, 8), (6, 8)), ((12, 16), (5, 5))],
    }
    tiles = [
        (padding, border, origin)
        for padding in (0, 2)
        for border in range(5)
        for origin in origins[padding]
    ]
    assert axes("tile", "G") == {
        frozen(
            tile_case_args(shape, p, kernel, *origin, plane, k % 3, border),
            *rotated(k),
            ("unit",),
        )
        for k, (p, border, origin) in enumerate(tiles)
    }
    reach = [
        (small, border, p)
        for small in ((1, 1, 1), (2, 3, 2))
        for border in range(5)
        for p in (0, 2)
    ]
    expected = set()
    for k, (small, border, p) in enumerate(reach):
        whole = (small[0] + 2 * p, small[1] + 2 * p)
        span = (whole[0] + 9, whole[1] + 11)
        args = tile_case_args(
            small, p, (9, 11), (0, 0), whole, span, small[2] - 1, border
        )
        expected.add(frozen(args, *rotated(k), ("unit",)))
    assert axes("tile", "R") == expected
    for group, special, mapping, paddings in (
        ("D", "edges", "unit", (0, 2)),
        ("T", "clean", "tiny", (0,)),
    ):
        assert axes("tile", group) == {
            frozen(
                tile_case_args(shape, p, kernel, (0, 0), (6, 8), plane, 1, 4),
                special,
                s,
                (mapping,),
            )
            for p in paddings
            for s in MXCSR_STATES
        }
    expected = set()
    for wide, p, wide_kernel, border, channel in golden_kernels.TILE_WIDE:
        whole = (wide[0] + 2 * p, wide[1] + 2 * p)
        span = (whole[0] + wide_kernel[0], whole[1] + wide_kernel[1])
        args = tile_case_args(
            wide, p, wide_kernel, (0, 0), whole, span, channel, border
        )
        expected.add(frozen(args, "clean", "default", ("unit",)))
    assert axes("tile", "W") == expected
    bench, bench_kernel, bench_plane = golden_kernels.SPECTRAL_BENCH
    assert (bench_kernel, bench[2]) == ((131, 131), 1)
    block, filtered = filter2d_plane(*bench_kernel, *bench[:2])
    assert (block, filtered) == (bench[:2], bench_plane)
    args = tile_case_args(bench, 0, bench_kernel, (0, 0), block, filtered, 0, 4)
    assert axes("tile", "B") == {frozen(args, "clean", "default", ("unit",))}

    def store(result, origin, size, splane, channel):
        names = ("oh", "ow", "c", "y", "x", "th", "tw", "dh", "dw", "channel")
        return dict(
            zip(names, (*result, *origin, *size, *splane, channel), strict=True)
        )

    stores = [
        (c, origin, s) for c in (1, 3) for origin in origins[0] for s in MXCSR_STATES
    ]
    assert axes("store", "G") == {
        frozen(store((13, 17, c), *origin, plane, k % c), "bits", s, ("bits", "unit"))
        for k, (c, origin, s) in enumerate(stores)
    }
    for group, results, states in (
        ("D", golden_kernels.STORE_DAZ, MXCSR_STATES),
        ("W", (golden_kernels.STORE_WIDE,), ("default",)),
    ):
        assert axes("store", group) == {
            frozen(
                store(r, (0, 0), r[:2], (r[0] + 5, r[1] + 7), r[2] - 1),
                "bits",
                s,
                ("bits", "unit"),
            )
            for r in results
            for s in states
        }
    args = store(bench, (0, 0), bench[:2], bench_plane, 0)
    assert axes("store", "B") == {frozen(args, "bits", "default", ("bits", "unit"))}
    planes = [(h, w) for h in (1, 2, 7, 8) for w in (1, 2, 9, 10)]
    bits2 = ("bits", "bits")
    assert axes("multiply", "S") == {
        frozen({"h": h, "w": w}, "bits", MXCSR_STATES[k % 2], bits2)
        for k, (h, w) in enumerate(planes)
    }
    assert axes("multiply", "D") == {
        frozen({"h": h, "w": w}, "bits", s, bits2)
        for h, w in golden_kernels.MULTIPLY_DAZ
        for s in MXCSR_STATES
    }
    for group, (h, w) in (("W", golden_kernels.MULTIPLY_WIDE), ("B", bench_plane)):
        expected = {frozen({"h": h, "w": w}, "bits", "default", bits2)}
        assert axes("multiply", group) == expected

    def normal(pixels, channels, invert_r, invert_g, alpha):
        args = {"pixels": pixels, "invert_r": invert_r, "invert_g": invert_g}
        return {**args, "channels": channels, "alpha": alpha}

    def normal_maps(alpha):
        return ("signed", "signed", "unit") if alpha else ("signed", "signed")

    forms = [
        (channels, r, g, alpha, name)
        for channels in (3, 4)
        for r in (0, 1)
        for g in (0, 1)
        for alpha in (0, 1)
        for name in ("edges", "nan", "inf")
    ]
    small = golden_kernels.SMALL_COUNTS
    assert axes("normal", "F") == {
        frozen(
            normal(small[k % 17], *form[:4]),
            form[4],
            MXCSR_STATES[k // 3 % 2],
            normal_maps(form[3]),
        )
        for k, form in enumerate(forms)
    }
    plain = ((3, 0, 0, 0), (4, 0, 0, 1))
    assert axes("normal", "D") == {
        frozen(normal(17, *form), "edges", s, normal_maps(form[3]))
        for form in plain
        for s in MXCSR_STATES
    }
    assert axes("normal", "W") == {
        frozen(
            normal(golden_kernels.WIDE_COUNT, *f), "edges", "default", normal_maps(f[3])
        )
        for f in plain
    }
    assert axes("normal", "B") == {
        frozen(normal(196_608, 3, 0, 0, 0), "clean", "default", normal_maps(0))
    }

    def caption(size, channels, top):
        names = ("height", "width", "caption_height")
        return {**dict(zip(names, size, strict=True)), "channels": channels, "top": top}

    sizes = golden_kernels.CAPTION_SIZES
    assert sizes == ((1, 1, 1), (5, 7, 3), (9, 13, 2))
    captions = [(top, c, size) for top in (0, 1) for c in (1, 3, 4) for size in sizes]
    assert axes("caption", "F") == {
        frozen(caption(size, c, top), *rotated(k), ("bits", "unit"))
        for k, (top, c, size) in enumerate(captions)
    }
    assert axes("caption", "D") == {
        frozen(caption(sizes[2], c, 0), "edges", s, ("bits", "unit"))
        for c in (3, 4)
        for s in MXCSR_STATES
    }
    for group, size, top in (
        ("W", golden_kernels.CAPTION_WIDE, 1),
        ("B", golden_kernels.CAPTION_BENCH, 0),
    ):
        expected = {frozen(caption(size, 3, top), "clean", "default", ("bits", "unit"))}
        assert axes("caption", group) == expected
    assert golden_kernels.CAPTION_BENCH[:2] == bench[:2]
    twins = [
        (case, outputs)
        for case, outputs in cases
        if case.id.split("-")[0] in ("D", "T")
    ]
    sensitive = check_twins(twins, lambda case: case.entry in (TILE, STORE, MULTIPLY))
    assert len(sensitive) == 7
    by_id = {case.id: outputs for case, outputs in twins}
    for case, outputs in twins:
        if case.mxcsr == "daz_ftz" and case.entry in (NORMAL, CAPTION):
            assert outputs == by_id[twin_id(case.id)], case.id


def tile_form(case):
    """A separable case's (entry, form, border, lanes); border None for the box."""
    spec = golden_kernels.ENTRIES[case.entry]
    form = tuple(case.args[p] for p in spec.form)
    border = case.args[spec.place] if spec.place else None
    return case.entry, form, border, case.args[spec.mirror]


def tile_axes(case):
    """A separable case's (entry, form, border, lanes, set, MXCSR state, shape)."""
    shape = (case.args["h"], case.args["w"], case.args["c"])
    return *tile_form(case), special_set(case), case.mxcsr, shape


def tile_forms(gaussian, box, borders=(1, 2, 4), lanes=(4, 8)):
    """(entry, form, border, lanes): the Gaussian form at each border and the box form,
    at each lanes value."""
    found = {("cn_gaussian_f32", gaussian, b, m) for b in borders for m in lanes}
    return found | {("cn_box_separable", box, None, m) for m in lanes}


def check_tiles_manifest(cases):
    """separable_tiles' groups (SP4b D9) and their axes; output dst.

    H (128): every TILE_HEIGHTS height at (h, 13, 3), the Gaussian (21, 11) at
    borders 1/2/4 and the box (3.25, 4), lanes 4/8. G (14): TILE_PLANE with those and
    the Gaussian (49, 49) at borders 1/2/4. F (16): each TILE_FALLBACKS shape with its
    forms. N (6): each TILE_UNTABLED pair at borders 1/2/4, lanes 8. W (96): (7, w, c)
    for widths 18-33 and channels 1/3/4, lanes 8, the Gaussian (21, 11) at border
    (1, 2, 4)[w % 3] and the box (1.5, 2.5). P (3): TILE_BENCH with the bench's three
    Gaussians at border 2, lanes 8. Those are clean at the default MXCSR. S (16):
    (65, 41, 3), the Gaussian (21, 11) at border 4 and the box (3.25, 4), lanes 8, every
    TILE_SETS set, both states. E (16): (65, 41, 3), src in the denormal range, the H
    forms, both states; each daz_ftz case differs from its default twin on B3. Every
    output is informative (not all NaN).
    """
    groups = {}
    for case, outputs in cases:
        groups.setdefault(case.id.split("-")[0], []).append(case)
        assert sorted(outputs) == ["dst"], case.id
        assert len(outputs["dst"]) == 64, case.id
        assert outputs["dst"] not in all_nan_digests(case), case.id
        assert case.payloads == SPECIALS[special_set(case)]["payloads"], case.id
    assert {group: len(found) for group, found in groups.items()} == {
        "H": 128,
        "G": 14,
        "F": 16,
        "N": 6,
        "W": 96,
        "P": 3,
        "S": 16,
        "E": 16,
    }

    def axes(group):
        return {tile_axes(case) for case in groups[group]}

    def clean(forms, shapes, states=("default",)):
        return {
            (*form, "clean", state, shape)
            for form in forms
            for state in states
            for shape in shapes
        }

    h_forms = tile_forms((21, 11), (3.25, 4.0))
    assert axes("H") == clean(h_forms, [(h, 13, 3) for h in TILE_HEIGHTS])
    plane = h_forms | {
        ("cn_gaussian_f32", (49, 49), b, m) for b in (1, 2, 4) for m in (4, 8)
    }
    assert axes("G") == clean(plane, [TILE_PLANE])
    assert axes("F") == set().union(
        *(clean(tile_forms(g, b), [shape]) for shape, g, b in TILE_FALLBACKS)
    )
    assert axes("N") == set().union(
        *(
            clean({("cn_gaussian_f32", form, b, 8) for b in (1, 2, 4)}, [shape])
            for shape, form in TILE_UNTABLED
        )
    )
    assert axes("W") == {
        (*form, "clean", "default", (7, w, c))
        for w in range(18, 34)
        for c in (1, 3, 4)
        for form in (
            ("cn_gaussian_f32", (21, 11), (1, 2, 4)[w % 3], 8),
            ("cn_box_separable", (1.5, 2.5), None, 8),
        )
    }
    bench = {("cn_gaussian_f32", form, 2, 8) for form in ((21, 11), (25, 25), (49, 49))}
    assert axes("P") == clean(bench, [TILE_BENCH])
    states = ("default", "daz_ftz")
    assert axes("S") == {
        (*form, name, state, (65, 41, 3))
        for form in tile_forms((21, 11), (3.25, 4.0), borders=(4,), lanes=(8,))
        for name in TILE_SETS
        for state in states
    }
    assert axes("E") == clean(h_forms, [(65, 41, 3)], states)
    assert {case.inputs["src"]["mapping"] for case in groups["E"]} == {"tiny"}
    # Every form the groups use is a tile form or a fallback's or untabled one.
    forms = {(case.entry, tile_axes(case)[1]) for case, _ in cases}
    extra = {("cn_gaussian_f32", g) for _, g, _ in TILE_FALLBACKS}
    extra |= {("cn_box_separable", b) for _, _, b in TILE_FALLBACKS}
    extra |= {("cn_gaussian_f32", form) for _, form in TILE_UNTABLED}
    tiled = {(entry, form) for entry, found in TILE_FORMS.items() for form in found}
    assert forms <= tiled | extra
    # Under DAZ the tiny src reads as zero (MXCSR is a first-class input, review I1).
    sensitive = check_twins(cases, lambda case: case.id.startswith("E-"))
    assert len(sensitive) == 8


@pytest.mark.parametrize("kernel", KERNELS)
def test_manifests_were_generated_from_b3_and_cover_the_matrix(kernel):
    path = GOLDEN / f"{kernel}.json"
    # Committed LF (git ls-files --eol: i/lf); core.autocrlf checkouts write CRLF.
    text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert len(text.encode("utf-8")) < SIZE_LIMIT
    assert text == golden_kernels.dumps(json.loads(text))
    header, cases = load_manifest(path)
    assert header["schema"] == SCHEMA
    assert header["kernel"] == kernel
    # The NumPy version is recorded only; pcg64_check guards the stream.
    assert header["generated_with"]["dll_sha256"] == B3_SHA256
    assert "numpy" in header["generated_with"]
    # A manifest records the sets defined when it was written: SP4a's, or every set
    # once denorm exists (SP4b D3).
    sets = header["special_sets"]
    assert set(sets) in (set(SP4A_SETS), set(SPECIALS))
    assert sets == {name: SPECIALS[name] for name in sets}
    assert [case for case, _ in cases] == matrix(kernel)
    ids = [case.id for case, _ in cases]
    assert len(set(ids)) == len(ids)
    entries = golden_kernels.KERNEL_ENTRIES[kernel]
    assert {case.entry for case, _ in cases} == set(entries)
    if kernel in CONVERSION_KERNELS:
        check_conversion_manifest(kernel, cases)
        return
    if kernel == "convolution_border":
        check_border_manifest(cases)
        return
    if kernel == "lens":
        check_lens_manifest(cases)
        return
    if kernel == "separable_tiles":
        check_tiles_manifest(cases)
        return
    if kernel == "lens_power":
        assert set(sets) == set(SPECIALS)
        check_power_manifest(cases)
        return
    if kernel == "morphology":
        assert set(sets) == set(SPECIALS)
        check_morphology_manifest(cases)
        return
    if kernel == "resample":
        assert set(sets) == set(SPECIALS)
        check_resample_manifest(cases)
        return
    if kernel == "palette":
        assert set(sets) == set(SPECIALS)
        check_palette_manifest(cases)
        return
    if kernel == "dither":
        assert set(sets) == set(SPECIALS)
        check_dither_manifest(cases)
        return
    if kernel == "composite":
        assert set(sets) == set(SPECIALS)
        check_composite_manifest(cases)
        return
    if kernel == "blend":
        assert set(sets) == set(SPECIALS)
        check_blend_manifest(cases)
        return
    if kernel == "tail":
        assert set(sets) == set(SPECIALS)
        check_tail_manifest(cases)
        return
    for entry in entries:
        spec = golden_kernels.ENTRIES[entry]
        mine = [case for case, _ in cases if case.entry == entry]
        assert {tuple(case.args[p] for p in spec.form) for case in mine} == set(
            spec.forms
        )
        assert {case.args[spec.mirror] for case in mine} == set(spec.mirrors)
        if spec.place:
            assert {case.args[spec.place] for case in mine} == set(spec.places)
        assert {special_set(case) for case in mine} == set(SP4A_SETS)
        assert {case.mxcsr for case in mine} == {"default", "daz_ftz"}
    for case, outputs in cases:
        payloads = {
            value
            for recipe in case.inputs.values()
            for _, value in recipe["specials"]
            if nan_bits(int(value, 16))
        }
        assert case.payloads == ("mixed" if len(payloads) > 1 else "single"), case.id
        assert case.payloads == SPECIALS[special_set(case)]["payloads"], case.id
        assert sorted(outputs) == [golden_kernels.ENTRIES[case.entry].output]
        assert all(len(value) == 64 for value in outputs.values())
    # Controller ruling: every (form, border or padding, mirror value) of every
    # entry has a case whose output is not all NaN, so no matrix change can void
    # a combination's coverage unnoticed.
    informative = {
        axes_of(case)
        for case, outputs in cases
        if not set(outputs.values()) <= all_nan_digests(case)
    }
    assert informative == combinations(entries)
    # Controller ruling (review I1): MXCSR is a first-class input. A daz_ftz case
    # shares its inputs with its default twin, and every group E daz_ftz case
    # (denormal-range src) differs from its twin on B3, so each (entry, form,
    # mirror value) has a DAZ-sensitive golden.
    sensitive = check_twins(cases, lambda case: case.id.startswith("E-"))
    assert {axes_of(case, places=False) for case in sensitive} == combinations(
        entries, places=False
    )


@pytest.mark.parametrize("kernel", KERNELS)
def test_special_sets_reach_their_inputs(kernel):
    # Controller rulings: spec 4.4 puts the set's NaN (or inf) in every input of
    # group B; group A and mixed cases put it in src, and their convolution kernel
    # holds no NaN or inf, since one such tap reaches every output pixel; so does
    # every convolution_border kernel. A conversion's float32 src holds its set's
    # NaN or inf, down to count 1; so does every lens input, a mixed one some value
    # of its rotated set. A dither palette holds the src's set only when it is mixed
    # (where src and palette NaNs meet); otherwise its form's values alone. An input of
    # another dtype (a double plane, a uint8 raster) holds PCG64 words and no set.
    defining = {
        "nan": [0x7FC12345],
        "snan": [0x7F812345],
        "inf": [0x7F800000, 0xFF800000],
        "mixed": [0x7FC12345, 0xFFC54321, 0x7F812345],
    }
    rotated = [int(value, 16) for value in SPECIALS["mixed"]["values"]]
    for case in matrix(kernel):
        name = special_set(case)
        wanted = defining.get(name)
        if kernel == "lens" and name == "mixed":
            wanted = rotated
        for input_name, recipe in case.inputs.items():
            if (input_name == "palette" and name != "mixed") or "dtype" in recipe:
                continue
            array = make_array(recipe)
            if input_name == "kernel" and (
                kernel == "convolution_border" or bare_kernel(case, name)
            ):
                assert np.isfinite(array).all(), case.id
            elif wanted:
                bits = array.view(np.uint32)
                assert np.isin(bits, wanted).any(), (case.id, input_name)


def test_daz_ftz_cases_set_mxcsr_on_the_calling_thread():
    state = native.lib()["cn_image_fp_state"]
    state.argtypes = []
    state.restype = ctypes.c_uint64
    assert state() & 0x8040 == 0
    with golden_kernels.mxcsr(native.lib(), "daz_ftz"):
        assert state() & 0x8040 == 0x8040
    assert state() & 0x8040 == 0
    # A denormal through the vertical radius-0 form: kept by default, flushed
    # (read as zero) under DAZ|FTZ.
    denormals = [[i, "0x00000001"] for i in range(9)]
    case = Case(
        id="denormal",
        entry="cn_box_separable",
        args={"h": 1, "w": 9, "c": 1, "rx": 0.0, "ry": 0.0, "lanes": 4},
        inputs={
            "src": {
                "shape": [1, 9, 1],
                "seed": 0,
                "mapping": "unit",
                "specials": denormals,
            }
        },
        mxcsr="default",
        payloads="single",
    )
    kept = np.full(9, 1, np.uint32).view(np.float32)
    flushed = np.zeros(9, np.float32)
    assert run_case(native.lib(), case) == {"dst": digest(kept, "single")}
    daz = dataclasses.replace(case, mxcsr="daz_ftz")
    assert run_case(native.lib(), daz) == {"dst": digest(flushed, "single")}
    assert state() & 0x8040 == 0


def test_mxcsr_requires_the_exact_fp_controls():
    # cn_image_fp_state: fegetround() << 32 | MXCSR & 0xe040 (rounding, FTZ, DAZ).
    with golden_kernels.mxcsr(Stub(0), "default"):
        pass
    for other in (0x40, 0x8000, 0x2000, 0x6000, 0x100 << 32):
        with pytest.raises(RuntimeError), golden_kernels.mxcsr(Stub(other), "default"):
            pass
    with golden_kernels.mxcsr(Stub(0x8040), "daz_ftz"):
        pass
    for other in (0, 0x8000, 0x8040 | 0x2000, 0x8040 | 0x100 << 32):
        with pytest.raises(RuntimeError), golden_kernels.mxcsr(Stub(other), "daz_ftz"):
            pass
    state = native.lib()["cn_image_fp_state"]
    state.argtypes = []
    state.restype = ctypes.c_uint64
    assert state() == 0


def test_run_case_refuses_an_unwritten_output():
    case = next(c for c in matrix("separable") if c.id.startswith("C-"))
    with pytest.raises(RuntimeError, match="output not fully written"):
        run_case(Stub(), case)
    # A conversion runs twice, into buffers pre-filled 0x00 and 0xff: any byte of
    # out (or the events value) it leaves alone shows as a difference.
    defaults = [c for c in matrix("conversion_small") if c.mxcsr == "default"]
    for case in [defaults[0], next(c for c in defaults if c.args["output"] == 1)]:
        with pytest.raises(RuntimeError, match="output not fully written"):
            run_case(Stub(), case)
    # A lens case without accumulate runs twice, into out from its recipe and into
    # out pre-filled 0xffffffff: an element it leaves alone shows as a difference.
    case = next(c for c in matrix("lens") if c.args["accumulate"] == 0)
    with pytest.raises(RuntimeError, match="output not fully written"):
        run_case(Stub(), case)
    # lens_power's out is pre-filled 0xffffffff, which the entry overwrites.
    case = next(c for c in matrix("lens_power") if c.mxcsr == "default")
    with pytest.raises(RuntimeError, match="output not fully written"):
        run_case(Stub(), case)
    # Morphology's out likewise; cn_image_exceptional's result starts at -1; resample
    # runs twice, into out pre-filled 0x00000000 and 0xffffffff.
    morphology = [c for c in matrix("morphology") if c.mxcsr == "default"]
    for entry in (MORPH_ENTRY, FILTER_ENTRY):
        case = next(c for c in morphology if c.entry == entry)
        with pytest.raises(RuntimeError, match="output not fully written"):
            run_case(Stub(), case)
    case = next(c for c in morphology if c.entry == EXCEPTIONAL_ENTRY)
    with pytest.raises(RuntimeError, match="result not written"):
        run_case(Stub(), case)
    case = next(c for c in matrix("resample") if c.mxcsr == "default")
    with pytest.raises(RuntimeError, match="output not fully written"):
        run_case(Stub(), case)
    # Median cut's written starts at 0, and a status 0 needs at least one color; a
    # dither entry's compatible starts at -1.
    case = next(c for c in matrix("palette") if c.mxcsr == "default")
    with pytest.raises(RuntimeError, match="output not fully written"):
        run_case(Stub(), case)
    dither = [c for c in matrix("dither") if c.mxcsr == "default"]
    for entry in golden_kernels.KERNEL_ENTRIES["dither"]:
        case = next(c for c in dither if c.entry == entry)
        with pytest.raises(RuntimeError, match="compatible not written"):
            run_case(Stub(), case)
    # The composite canvas's and both blend entries' out likewise (pre-filled
    # 0xffffffff).
    case = next(c for c in matrix("composite") if c.mxcsr == "default")
    with pytest.raises(RuntimeError, match="output not fully written"):
        run_case(Stub(), case)
    blend = [c for c in matrix("blend") if c.mxcsr == "default"]
    for entry in golden_kernels.KERNEL_ENTRIES["blend"]:
        case = next(c for c in blend if c.entry == entry)
        with pytest.raises(RuntimeError, match="output not fully written"):
            run_case(Stub(), case)
    # The tail's outputs likewise (the tile's float64 plane pre-filled with all-ones
    # words); the store runs twice, into dst as made and with its tile's channel
    # pre-filled 0xffffffff. The multiply's in-out plane cannot show an element left
    # alone: each becomes a function of its own value.
    tail = [c for c in matrix("tail") if c.mxcsr == "default"]
    for entry in (TILE, STORE, NORMAL, CAPTION):
        case = next(c for c in tail if c.entry == entry)
        with pytest.raises(RuntimeError, match="output not fully written"):
            run_case(Stub(), case)


def writes_past(src, out, count, kind, output, normalize, events):
    """A conversion that writes every element and the 4 bytes after them."""
    width = np.dtype(golden_kernels.CONVERT_OUTPUTS[output]).itemsize
    ctypes.memset(out, 0x11, count * width + 4)
    return 0


def composes_past(f1, f2, f3, f4, out, count, a, b, accumulate):
    """A lens compose that writes every element and the 4 bytes after them."""
    ctypes.memset(out, 0x11, count * 4 + 4)
    return 0


def powers_past(src, out, count, exponent, finish):
    """A lens power that writes every element and the 4 bytes after them."""
    ctypes.memset(out, 0x11, count * 4 + 4)
    return 0


def morphs_past(src, out, height, width, channels, *form):
    """A morphology entry that writes every element and the 4 bytes after them."""
    ctypes.memset(out, 0x11, height * width * channels * 4 + 4)
    return 0


def resamples_past(src, out, height, width, channels, th, tw, *options):
    """A resample that writes every element and the 4 bytes after them."""
    ctypes.memset(out, 0x11, th * tw * channels * 4 + 4)
    return 0


def palettes_past(
    src, out, pixels, channels, capacity, root_block, root_period, child_block, written
):
    """A median cut that writes min(capacity, pixels) colors and the 4 bytes after."""
    rows = min(capacity, pixels)
    written[0] = rows
    ctypes.memset(out, 0x11, rows * channels * 4 + 4)
    return 0


def dithers_past(*args):
    """Every dither call: palette creation (5 arguments) and free (1) do nothing; apply
    (11), cn_neighborhood_dither (12) and riemersma (10) write every element and the 4
    bytes after them and set compatible (the last argument) to 1."""
    if len(args) in (10, 11, 12):
        _, out, height, width, channels = args[:5]
        ctypes.memset(out, 0x11, height * width * channels * 4 + 4)
        args[-1][0] = 1
    return 0


def composites_past(base, overlay, geometry, out, mode):
    """A composite canvas that writes every element of the geometry's canvas and the 4
    bytes after them."""
    canvas = geometry.contents
    ctypes.memset(out, 0x11, canvas.height * canvas.width * canvas.channels * 4 + 4)
    return 0


def blends_past(overlay, base, out, count, *channels_and_mode):
    """A blend entry that writes every element and the 4 bytes after them:
    cn_blend_images (overlay and base channels, then the mode) writes count pixels of
    the larger channel count, cn_blend_mode (the mode alone) count elements."""
    channels = max(channels_and_mode[:2]) if len(channels_and_mode) == 3 else 1
    ctypes.memset(out, 0x11, count * channels * 4 + 4)
    return 0


def tiles_past(src, dst, *args):
    """A spectral tile that writes its (dh, dw) float64 plane and the 4 bytes after."""
    dh, dw = args[10:12]
    ctypes.memset(dst, 0x11, dh * dw * 8 + 4)
    return 0


def stores_past(src, dst, oh, ow, c, *tile):
    """A spectral store that writes all of dst and the 4 bytes after it."""
    ctypes.memset(dst, 0x11, oh * ow * c * 4 + 4)
    return 0


def multiplies_past(a, b, h, w):
    """A spectral multiply that writes all of a and the 4 bytes after it."""
    ctypes.memset(a, 0x11, h * w * 8 + 4)
    return 0


def normals_past(dx, dy, alpha, dst, pixels, invert_r, invert_g, channels):
    """A normal output that writes every element and the 4 bytes after them."""
    ctypes.memset(dst, 0x11, pixels * channels * 4 + 4)
    return 0


def captions_past(image, caption, out, height, width, channels, rows, top):
    """A caption compose that writes every element and the 4 bytes after them."""
    ctypes.memset(out, 0x11, (height + rows) * width * channels * 4 + 4)
    return 0


class WritesPast:
    """A DLL stand-in whose every entry is `function`; FP state 0."""

    def __init__(self, function):
        self.function = function

    def __getitem__(self, name):
        if name == "cn_image_fp_state":
            return lambda: 0
        return self.function


def test_adapters_refuse_a_write_past_count():
    # Review M3: the exact-size output could not show a group written past the
    # last chunk's end; the adapters' 32-byte tail does.
    case = next(c for c in matrix("conversion_small") if c.mxcsr == "default")
    with pytest.raises(RuntimeError, match="written past count"):
        run_case(WritesPast(writes_past), case)
    for accumulate in (0, 1):
        case = next(
            c
            for c in matrix("lens")
            if c.mxcsr == "default" and c.args["accumulate"] == accumulate
        )
        with pytest.raises(RuntimeError, match="written past count"):
            run_case(WritesPast(composes_past), case)
    case = next(c for c in matrix("lens_power") if c.mxcsr == "default")
    with pytest.raises(RuntimeError, match="written past count"):
        run_case(WritesPast(powers_past), case)
    for entry in (MORPH_ENTRY, FILTER_ENTRY):
        case = next(c for c in matrix("morphology") if c.entry == entry)
        with pytest.raises(RuntimeError, match="written past count"):
            run_case(WritesPast(morphs_past), case)
    case = next(c for c in matrix("resample") if c.mxcsr == "default")
    with pytest.raises(RuntimeError, match="written past count"):
        run_case(WritesPast(resamples_past), case)
    case = next(c for c in matrix("palette") if c.mxcsr == "default")
    with pytest.raises(RuntimeError, match="written past count"):
        run_case(WritesPast(palettes_past), case)
    dither = [c for c in matrix("dither") if c.mxcsr == "default"]
    for entry in golden_kernels.KERNEL_ENTRIES["dither"]:
        case = next(c for c in dither if c.entry == entry)
        with pytest.raises(RuntimeError, match="written past count"):
            run_case(WritesPast(dithers_past), case)
    case = next(c for c in matrix("composite") if c.mxcsr == "default")
    with pytest.raises(RuntimeError, match="written past count"):
        run_case(WritesPast(composites_past), case)
    blend = [c for c in matrix("blend") if c.mxcsr == "default"]
    for entry in golden_kernels.KERNEL_ENTRIES["blend"]:
        case = next(c for c in blend if c.entry == entry)
        with pytest.raises(RuntimeError, match="written past count"):
            run_case(WritesPast(blends_past), case)
    tail = [c for c in matrix("tail") if c.mxcsr == "default"]
    writers = (tiles_past, stores_past, multiplies_past, normals_past, captions_past)
    for entry, writer in zip(
        golden_kernels.KERNEL_ENTRIES["tail"], writers, strict=True
    ):
        case = next(c for c in tail if c.entry == entry)
        with pytest.raises(RuntimeError, match="written past count"):
            run_case(WritesPast(writer), case)


def d5_corrected(case):
    """Consult 8 D-5: a dither case B3 refused (300 unique colors and a NaN src:
    compatible 0, out unwritten), which the current DLL dithers, its NaN pixels by
    the linear rule of smaller palettes. The manifest keeps B3's record; exactly
    these cases differ from it."""
    return (
        case.entry in golden_kernels.KERNEL_ENTRIES["dither"]
        and case.args["count"] == 300
        and special_set(case) == "nan"
    )


def upstream_image(array):
    """An (h, w, c) array as upstream's image: (h, w) for one channel."""
    return array[..., 0] if array.shape[2] == 1 else array


def upstream_blend(case):
    """Upstream's blend_images (reference_blend.py) of a cn_blend_images case, on
    NumPy 2.5.3 under the case's MXCSR state, as a (pixels, channels) array."""
    pixels = case.args["pixels"]
    overlay, base = (
        upstream_image(make_array(case.inputs[name]).reshape(1, pixels, -1))
        for name in golden_kernels.BLEND_INPUTS
    )
    mode = reference_blend.BlendMode(case.args["mode"])
    with golden_kernels.mxcsr(native.lib(), case.mxcsr), np.errstate(all="ignore"):
        out = reference_blend.blend_images(overlay, base, mode)
    return np.ascontiguousarray(out).reshape(pixels, -1)


def upstream_composite(case):
    """Upstream's Blend Images node (test_composite_complete's reference, which
    blends with reference_blend) of a cn_composite_canvas case: its layers, mode,
    pixel offsets and crop, on NumPy 2.5.3 under the case's MXCSR state. A constant
    layer is the image of its float32 values at its h and w, as Color.to_image gives
    it, and turns crop off, as a Color does; its bits are kept, which a Color's
    Python floats can't do for a denormal under DAZ|FTZ."""
    args = case.args
    layers = []
    for name, prefix in zip(COMPOSITE_INPUTS, "bo", strict=True):
        array = make_array(case.inputs[name])
        if args[f"{name}_constant"]:
            shape = (args[f"{prefix}h"], args[f"{prefix}w"], args[f"{prefix}c"])
            array = np.broadcast_to(array, shape).copy()
        layers.append(upstream_image(array))
    base, overlay = layers
    x, y = args["x"], args["y"]
    constant = args["base_constant"] or args["overlay_constant"]
    with golden_kernels.mxcsr(native.lib(), case.mxcsr), np.errstate(all="ignore"):
        out = composite_reference.blend_images_node(
            base,
            overlay,
            composite_reference.BlendMode(args["mode"]),
            composite_reference.BlendOverlayPosition.PIXEL_OFFSET,
            x,
            y,
            x,
            y,
            bool(args["crop"]) and not constant,
        )
    return np.ascontiguousarray(out)


def d17_outputs(case, outputs):
    """Consult 11 D-17.1: upstream's outputs (upstream_blend, upstream_composite) of
    a cn_blend_images or cn_composite_canvas case whose B3 record is upstream's with
    every -0 made +0, and so differs from it; None for every other case. B3's blend
    clip, minimum(maximum(x, 0), 1), gave +0 for -0 (and, under DAZ, for a negative
    denormal), where np.clip keeps -0 (gives -0). The manifests keep B3's records;
    exactly these cases differ from them, and give upstream's outputs."""
    if case.entry == golden_kernels.BLEND_IMAGES:
        out = upstream_blend(case)
    elif case.entry == golden_kernels.COMPOSITE_ENTRY:
        out = upstream_composite(case)
    else:
        return None
    upstream = {"out": digest(out, case.payloads)}
    bits = out.view(np.uint32)
    plus_zero = np.where(bits == 0x80000000, 0, bits).view(np.float32)
    if upstream == outputs or {"out": digest(plus_zero, case.payloads)} != outputs:
        return None
    return upstream


def current_outputs(case, outputs):
    """A manifest case's outputs on the current DLL: B3's record (outputs), except
    upstream's for a D-17.1 case (d17_outputs)."""
    return d17_outputs(case, outputs) or outputs


@pytest.mark.parametrize("level", range(len(native.ISA_LEVELS)), ids=native.ISA_LEVELS)
@pytest.mark.parametrize("kernel", KERNELS)
def test_current_dll_matches_the_b3_manifest(kernel, level):
    # Every level the CPU has runs the manifest (cn_isa_set while no kernel runs).
    effective, _, cpu = isa_get()
    if level > cpu:
        pytest.skip(f"the CPU's maximum level is {native.ISA_LEVELS[cpu]}")
    _, cases = load_manifest(GOLDEN / f"{kernel}.json")
    try:
        assert isa_set(level) == level
        mismatched = [
            case.id
            for case, outputs in cases
            if run_case(native.lib(), case) != outputs
        ]
    finally:
        assert isa_set(effective) == effective
    assert mismatched == [
        case.id
        for case, outputs in cases
        if d5_corrected(case) or d17_outputs(case, outputs) is not None
    ]


@pytest.mark.parametrize("level", range(len(native.ISA_LEVELS)), ids=native.ISA_LEVELS)
@pytest.mark.parametrize("kernel,count", [("blend", 20), ("composite", 16)])
def test_d17_cases_give_upstreams_clip(kernel, count, level):
    effective, _, cpu = isa_get()
    if level > cpu:
        pytest.skip(f"the CPU's maximum level is {native.ISA_LEVELS[cpu]}")
    _, cases = load_manifest(GOLDEN / f"{kernel}.json")
    corrected = [
        (case, upstream)
        for case, outputs in cases
        if (upstream := d17_outputs(case, outputs)) is not None
    ]
    assert len(corrected) == count
    try:
        assert isa_set(level) == level
        current = [run_case(native.lib(), case) for case, _ in corrected]
    finally:
        assert isa_set(effective) == effective
    for (case, upstream), outputs in zip(corrected, current, strict=True):
        assert outputs == upstream, case.id


def test_d5_cases_are_dithered_where_b3_refused():
    # run_case checks that every element of out is written; compatible is written 1.
    _, cases = load_manifest(GOLDEN / "dither.json")
    corrected = [(case, outputs) for case, outputs in cases if d5_corrected(case)]
    assert len(corrected) == 8
    for case, outputs in corrected:
        assert outputs["compatible"] == digest_int(0), case.id
        assert outputs["out"] == EMPTY_DIGEST, case.id
        current = run_case(native.lib(), case)
        assert current["status"] == digest_int(0), case.id
        assert current["compatible"] == digest_int(1), case.id
        assert current == run_case(native.lib(), case), case.id


@pytest.mark.parametrize("kernel", CONVERSION_KERNELS)
def test_rebased_conversion_cases_are_the_clamps_blast_radius(kernel):
    # B3's clamp, maxss(+0, v), keeps -0 and, under DAZ, makes -0 of a negative
    # denormal. Task 3b1 re-based exactly the clamping cases with float32 output
    # (golden_kernels.clamps) whose float32 values (src.astype(np.float32) under the
    # case's MXCSR state) hold one, because NumPy 1.24.4's np.clip gave +0 there.
    # NumPy 2.5.3's np.clip (clip.cpp, built as maxps(0, x) then minps(1, x)) gives
    # B3's bits, so U2 restored those cases to B3's outputs and none records ORACLE.
    # Every clamping float32 output, those cases included, equals np.clip's, computed
    # here; the integer outputs cannot move (+-0 both scale to 0).
    path = GOLDEN / f"{kernel}.json"
    _, cases = load_manifest(path)
    lib = native.lib()
    moved = set()
    for case, outputs in cases:
        if not clamps(case.args) or case.args["output"] != 0:
            continue
        src = make_array(case.inputs["src"])
        with golden_kernels.mxcsr(lib, case.mxcsr), np.errstate(all="ignore"):
            bits = src.astype(np.float32).view(np.uint32)
        negative = bits == 0x80000000
        if case.mxcsr == "daz_ftz":
            negative |= (bits > 0x80000000) & (bits < 0x80800000)
        if negative.any():
            moved.add(case.id)
        assert outputs["out"] == digest(oracle_clip(lib, case), case.payloads), case.id
    assert moved
    assert rebased_ids(path) == []


def test_rebased_dither_cases_are_the_nan_key_order():
    # B3 (MSVC) gave the luminance key of a palette color with NaNs in red and green the
    # green NaN; the real chainner_ext gives the red one (numeric.h's cn_palette_key,
    # measured on every channel pair), which ties that color with a red-NaN one, kept in
    # input order. Only these cases' outputs moved, at every ISA level, and they hold
    # chaiNNer-C's outputs under that order (SP5a-c, Consult 9 P3 item 3), marked PORT:
    # upstream orders the tied keys by its random AHashSet, so no upstream output exists
    # to record as ORACLE (Consult 12 R-q).
    assert rebased_ids(GOLDEN / "dither.json") == []
    assert rebased_ids(GOLDEN / "dither.json", PORT) == [
        f"S-apply-unit-n16-m2a0-mixed-{state}-17x23x{c}"
        for state in ("default", "daz_ftz")
        for c in (3, 4)
    ]


def opencv_lanes():
    """native_filters.morphology's lanes: OpenCV's morph dispatch width."""
    return float32_lanes(target("morph"))


def node_call(case, src):
    """Whether native_filters.morphology calls the case's entry with its arguments: an
    ellipse, or any src exceptional under the case's state (exceptional_bits), runs
    cn_morphology_complete with opencv_lanes(); a rectangle or cross with a
    non-exceptional src runs cn_filter_morphology."""
    words = src.view(np.uint32).ravel().astype(np.int64)
    daz = case.mxcsr == "daz_ftz"
    exceptional = bool(
        (
            (words == 0x80000000)
            | (daz & (words > 0x80000000) & (words < 0x80800000))
            | ((words & 0x7F800000) == 0x7F800000)
        ).any()
    )
    if case.entry == FILTER_ENTRY:
        return not exceptional
    return case.args["lanes"] == opencv_lanes() and (
        case.args["shape"] == cv2.MORPH_ELLIPSE or exceptional
    )


def zero_class_differences(a, b):
    """Elements where the bits differ other than by one zero-class value (+-0 or a
    denormal) for another of the same sign."""
    x, y = a.view(np.uint32).ravel(), b.view(np.uint32).ravel()
    zero_x = (x & np.uint32(0x7FFFFFFF)) < np.uint32(0x00800000)
    zero_y = (y & np.uint32(0x7FFFFFFF)) < np.uint32(0x00800000)
    same_class = zero_x & zero_y & ((x >> np.uint32(31)) == (y >> np.uint32(31)))
    return int(((x != y) & ~same_class).sum())


def test_morphology_node_calls_equal_opencv():
    """Parity at design time (owner, 2026-10-04; task4-verdict.md): every morphology
    manifest case that is a node call (node_call) equals the installed chaiNNer's node
    code, cv2.dilate or cv2.erode with getStructuringElement(shape, (2 r + 1,) * 2) and
    the iterations, run in this venv (OpenCV 4.8.0, as installed) under the case's MXCSR
    state. At the default MXCSR the B3 digest is OpenCV's bit for bit, +-0 and NaN
    payloads included. Under DAZ the outputs differ from OpenCV's only by one zero-class
    value for another of the same sign (the ellipse combine's minss returns a tied
    denormal as +0, the deques keep the src element's bits, and OpenCV picks its own
    representative), which enforce's np.clip turns into +0 on both sides."""
    lib = native.lib()
    calls = {"default": 0, "daz_ftz": 0}
    for case, outputs in load_manifest(GOLDEN / "morphology.json")[1]:
        if case.entry == EXCEPTIONAL_ENTRY:
            continue
        src = make_array(case.inputs["src"])
        if not node_call(case, src):
            continue
        args = case.args
        if case.entry == MORPH_ENTRY:
            shape = args["shape"]
        else:
            shape = cv2.MORPH_CROSS if args["cross"] else cv2.MORPH_RECT
        size = 2 * args["radius"] + 1
        element = cv2.getStructuringElement(shape, (size, size))
        operation = cv2.dilate if args["maximum"] else cv2.erode
        with golden_kernels.mxcsr(lib, case.mxcsr), np.errstate(all="ignore"):
            expected = operation(src, element, iterations=args["iterations"])
            clipped = np.clip(expected, 0, 1)
            out = native_filters.morphology(
                src,
                shape,
                args["radius"],
                args["iterations"],
                maximum=bool(args["maximum"]),
            )
            enforced = np.clip(out, 0, 1)
        expected, out = expected.reshape(src.shape), out.reshape(src.shape)
        calls[case.mxcsr] += 1
        # The node path runs the case's entry: it gives the manifest's (B3's) outputs.
        assert digest(out, case.payloads) == outputs["out"], case.id
        if case.mxcsr == "default":
            assert digest(expected, case.payloads) == outputs["out"], case.id
            continue
        assert zero_class_differences(out, expected) == 0, case.id
        assert enforced.tobytes() == clipped.tobytes(), case.id
    # The DAZ cases show that the path runs; how many of them differ from OpenCV is
    # not pinned: the zero-class representative is a don't-care (task4-verdict.md
    # section 2), so a change that made them all equal would be no failure.
    assert calls["default"] >= 100 and calls["daz_ftz"] >= 20, calls


def test_composite_adapter_is_the_bridges_call():
    """The composite adapter's call (golden_kernels.composite_geometry and its layers)
    is native_composite.canvas's: on the current DLL the bridge, given each composite
    manifest case's layers (an image as its array, a constant as a Color of its floats),
    mode, x, y and crop under the case's MXCSR state, gives the case's outputs (B3's,
    or upstream's for a D-17.1 case: current_outputs).
    A Color holds Python floats, which canvas() turns back into float32 under that
    state: a signalling NaN comes back quiet, and under DAZ|FTZ a denormal comes back
    as zero. A constant those change cannot be a Color: such a case (the mixed set, or a
    denormal under DAZ|FTZ) is skipped, and at least 20 constant cases run."""
    lib = native.lib()
    colors = 0
    for case, outputs in load_manifest(GOLDEN / "composite.json")[1]:
        args = case.args
        layers = []
        for name in COMPOSITE_INPUTS:
            array = make_array(case.inputs[name])
            if not args[f"{name}_constant"]:
                layers.append(array)
                continue
            color = Color(tuple(float(value) for value in array))
            with golden_kernels.mxcsr(lib, case.mxcsr):
                carried = np.array(color.value, np.float32)
            if carried.tobytes() != array.tobytes():
                bits = array.view(np.uint32) & np.uint32(0x7FFFFFFF)
                denormal = ((bits != 0) & (bits < 0x00800000)).any()
                daz = case.mxcsr == "daz_ftz" and denormal
                assert special_set(case) == "mixed" or daz, case.id
                break
            layers.append(color)
        else:
            colors += args["base_constant"] or args["overlay_constant"]
            base, overlay = layers
            with golden_kernels.mxcsr(lib, case.mxcsr), np.errstate(all="ignore"):
                out = native_composite.canvas(
                    base,
                    overlay,
                    args["mode"],
                    args["x"],
                    args["y"],
                    bool(args["crop"]),
                )
            expected = current_outputs(case, outputs)["out"]
            assert digest(out, case.payloads) == expected, case.id
    assert colors >= 20, colors


def test_golden_tool_imports_neither_pillow_cv2_nor_torch():
    code = (
        "import sys; sys.path.insert(0, 'native/tools'); import golden_kernels; "
        "print(sorted(m for m in ('PIL', 'cv2', 'torch') if m in sys.modules))"
    )
    done = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def check_conversion_sweep(kernel):
    """D4: each argument uniformly from the kernel's manifest values; count in 1-140,000."""
    cases = sweep_cases(kernel, 1000, 11)
    assert cases == sweep_cases(kernel, 1000, 11)
    assert cases != sweep_cases(kernel, 1000, 12)
    assert [case.id for case in cases] == [f"sweep-11-{n}" for n in range(1000)]
    combos = {combo(case) for case, _ in load_manifest(GOLDEN / f"{kernel}.json")[1]}
    seen = {"combo": set(), "set": set(), "mapping": set(), "mxcsr": set()}
    for case in cases:
        check_conversion_recipe(case, sweep=True)
        assert combo(case) in combos, case.id
        assert 1 <= case.args["count"] <= 140_000, case.id
        seen["combo"].add(combo(case))
        seen["mxcsr"].add(case.mxcsr)
        recipe = case.inputs["src"]
        seen["mapping"].add((recipe.get("dtype", "float32"), recipe["mapping"]))
        names = (
            matching_sets(case) if recipe["mapping"] in ("signed", "tiny") else set()
        )
        if len(names) == 1:
            seen["set"] |= names
    assert seen["combo"] == combos
    assert seen["mxcsr"] == {"default", "daz_ftz"}
    # Every set reaches a float32 src; clean ones also draw the denormal range.
    assert seen["set"] == set(SP4A_SETS)
    assert ("float32", "tiny") in seen["mapping"]
    assert ("uint8", "ramp") in seen["mapping"] and ("uint8", "bits") in seen["mapping"]
    # Review M2, N2: raw float32 bits reach scaled values beyond int32 (bit 2 for
    # finite values) in a sweep case of (0,1,0), f32 -> u8 without the clamp, the
    # AVX2 path (with the clamp, scaled stays in [0, 255]).
    beyond = []
    for case in cases:
        if case.inputs["src"]["mapping"] != "bits" or combo(case) != (0, 1, 0):
            continue
        source = make_array(case.inputs["src"])
        magnitude = source.view(np.uint32) & np.uint32(0x7FFFFFFF)
        finite = source[magnitude < 0x7F800000].astype(np.float64)
        if (np.abs(finite) * 255 >= 2.0**31).any():
            beyond.append(case.id)
    assert beyond
    # Some counts exceed 65,536 and would split into unequal chunks at grain 65,536.
    # The conversion grain is 262,144, above every sweep count (1-140,000), so each
    # conversion sweep case is one chunk; the matrix's 589,824 and 2,764,800 are the
    # multi-chunk conversion cases.
    long = [case.args["count"] for case in cases if case.args["count"] > 65_536]
    assert any(count % (1 + (count - 1) // 65_536) for count in long)


def check_border_sweep():
    """D4 for convolution_border: each argument from the manifest's values; h and w
    in 1-64, and one case in eight with h in 65-300; c 1 or 3; a clean src on unit
    or tiny; some (5,5) kernels with DENORMAL_ROW."""
    cases = sweep_cases("convolution_border", 600, 11)
    assert cases == sweep_cases("convolution_border", 600, 11)
    assert cases != sweep_cases("convolution_border", 600, 12)
    assert [case.id for case in cases] == [f"sweep-11-{n}" for n in range(600)]
    seen = {key: set() for key in ("form", "padding", "fused", "set", "c", "mxcsr")}
    seen |= {"mapping": set(), "row": set()}
    tall = 0
    row = [list(tap) for tap in DENORMAL_ROW]
    for case in cases:
        check_border_recipe(case, sweep=True)
        args = case.args
        assert case.entry == "cn_convolution_spatial"
        assert 1 <= args["w"] <= 64 and 1 <= args["h"] <= 300, case.id
        tall += args["h"] > 64
        seen["form"].add((args["kh"], args["kw"]))
        for key in ("padding", "fused", "c"):
            seen[key].add(args[key])
        seen["mxcsr"].add(case.mxcsr)
        seen["mapping"].add(case.inputs["src"]["mapping"])
        seen["row"].add(case.inputs["kernel"]["specials"][-len(row) :] == row)
        names = border_sets(case)
        if len(names) == 1:
            seen["set"] |= names
    assert seen == {
        "form": set(BORDER_FORMS),
        "padding": {0, 1, 2},
        "fused": {0, 8},
        "set": set(BORDER_SETS),
        "c": {1, 3},
        "mxcsr": {"default", "daz_ftz"},
        "mapping": {"unit", "tiny"},
        "row": {False, True},
    }
    assert 0 < tall < len(cases) / 4


def check_lens_sweep():
    """D4 for lens: each argument from the manifest's values, count in 1-140,000; a
    clean case on signed or tiny inputs."""
    cases = sweep_cases("lens", 1000, 11)
    assert cases == sweep_cases("lens", 1000, 11)
    assert cases != sweep_cases("lens", 1000, 12)
    assert [case.id for case in cases] == [f"sweep-11-{n}" for n in range(1000)]
    seen = {key: set() for key in ("pair", "accumulate", "set", "mxcsr", "mapping")}
    for case in cases:
        check_lens_recipe(case, sweep=True)
        args = case.args
        assert case.entry == "cn_lens_compose"
        assert 1 <= args["count"] <= 140_000, case.id
        seen["pair"].add((args["a"], args["b"]))
        seen["accumulate"].add(args["accumulate"])
        seen["mxcsr"].add(case.mxcsr)
        seen["mapping"].add(case.inputs["f1"]["mapping"])
        names = lens_sets(case)
        if len(names) == 1:
            seen["set"] |= names
    assert seen == {
        "pair": set(LENS_PAIRS.values()),
        "accumulate": {0, 1},
        "set": set(LENS_SETS),
        "mxcsr": {"default", "daz_ftz"},
        "mapping": {"signed", "tiny"},
    }
    # Some cases are multi-chunk (count > 65,536, grain 65,536) with odd chunk lengths.
    long = [case.args["count"] for case in cases if case.args["count"] > 65_536]
    assert any(count % (1 + (count - 1) // 65_536) for count in long)


def check_power_sweep():
    """D4 for lens_power: channels, exponent, set and MXCSR state from the manifest's
    values, pixels * channels in 1-140,000; a clean case on signed or tiny src."""
    cases = sweep_cases("lens_power", 1000, 11)
    assert cases == sweep_cases("lens_power", 1000, 11)
    assert cases != sweep_cases("lens_power", 1000, 12)
    assert [case.id for case in cases] == [f"sweep-11-{n}" for n in range(1000)]
    seen = {key: set() for key in ("channels", "exponent", "set", "mxcsr", "mapping")}
    for case in cases:
        check_power_recipe(case, sweep=True)
        args = case.args
        assert 1 <= args["pixels"] * args["channels"] <= 140_000, case.id
        seen["channels"].add(args["channels"])
        seen["exponent"].add(args["exponent"])
        seen["mxcsr"].add(case.mxcsr)
        seen["mapping"].add(case.inputs["src"]["mapping"])
        names = power_sets(case)
        if len(names) == 1:
            seen["set"] |= names
    assert seen == {
        "channels": set(POWER_CHANNELS),
        "exponent": set(POWER_EXPONENTS.values()),
        "set": set(POWER_SETS),
        "mxcsr": {"default", "daz_ftz"},
        "mapping": {"signed", "tiny"},
    }
    # Some cases are multi-chunk (grain 65,536 // channels pixels) with odd chunk
    # lengths.
    long = [
        (case.args["pixels"], 65_536 // case.args["channels"])
        for case in cases
        if case.args["pixels"] > 65_536 // case.args["channels"]
    ]
    assert any(pixels % (1 + (pixels - 1) // grain) for pixels, grain in long)


def check_tiles_sweep():
    """D4 for separable_tiles: the entry, its TILE_FORMS form, lanes, border, set and
    MXCSR state from the manifest's values; h and w in 1-64, and one case in eight with
    h in 65-300; c 1, 3 or 4; a clean src on unit or tiny."""
    cases = sweep_cases("separable_tiles", 600, 11)
    assert cases == sweep_cases("separable_tiles", 600, 11)
    assert cases != sweep_cases("separable_tiles", 600, 12)
    assert [case.id for case in cases] == [f"sweep-11-{n}" for n in range(600)]
    seen = {key: set() for key in ("axes", "set", "c", "mxcsr", "mapping")}
    tall = 0
    for case in cases:
        args = case.args
        entry, form, border, lanes = tile_form(case)
        assert form in TILE_FORMS[entry], case.id
        assert 1 <= args["w"] <= 64 and 1 <= args["h"] <= 300, case.id
        tall += args["h"] > 64
        recipe = case.inputs["src"]
        assert recipe["shape"] == [args["h"], args["w"], args["c"]], case.id
        assert recipe["seed"] == seed_of(case.id), case.id
        for name, value in golden_kernels.ENTRIES[entry].fixed.items():
            assert args[name] == value, case.id
        # Tiny arrays can fit several sets (nan and mixed both start 0x7fc12345).
        names = matching_sets(case)
        assert names, case.id
        if len(names) == 1:
            seen["set"] |= names
        seen["axes"].add((entry, form, border, lanes))
        seen["c"].add(args["c"])
        seen["mxcsr"].add(case.mxcsr)
        seen["mapping"].add(recipe["mapping"])
    assert seen == {
        "axes": {
            (entry, form, border, lanes)
            for entry, forms in TILE_FORMS.items()
            for form in forms
            for border in golden_kernels.ENTRIES[entry].places or (None,)
            for lanes in (4, 8)
        },
        "set": set(SP4A_SETS),
        "c": {1, 3, 4},
        "mxcsr": {"default", "daz_ftz"},
        "mapping": {"unit", "tiny"},
    }
    assert 0 < tall < len(cases) / 4


def check_morph_sweep():
    """D4 for morphology: the entry; for cn_image_exceptional the count in 1-140,000,
    word, position and state; otherwise the entry's MORPH_FORMS arguments, set and
    state from the manifest's values, h and w in 1-64 and one case in eight with h in
    65-300, c 1, 3 or 4, and a clean src on unit or tiny."""
    cases = sweep_cases("morphology", 900, 11)
    assert cases == sweep_cases("morphology", 900, 11)
    assert cases != sweep_cases("morphology", 900, 12)
    assert [case.id for case in cases] == [f"sweep-11-{n}" for n in range(900)]
    seen = {key: set() for key in ("entry", "set", "c", "mxcsr", "mapping", "word")}
    seen["position"] = set()
    forms = {
        entry: {name: set() for name in values} for entry, values in MORPH_FORMS.items()
    }
    tall = images = 0
    for case in cases:
        check_morph_recipe(case, sweep=True)
        args = case.args
        seen["entry"].add(case.entry)
        seen["mxcsr"].add(case.mxcsr)
        if case.entry == EXCEPTIONAL_ENTRY:
            assert 1 <= args["count"] <= 140_000, case.id
            [(index, word)] = case.inputs["src"]["specials"]
            seen["word"].add(word)
            if args["count"] > 1:
                seen["position"].add("first" if index == 0 else "last")
            continue
        images += 1
        assert 1 <= args["w"] <= 64 and 1 <= args["h"] <= 300, case.id
        tall += args["h"] > 64
        for name in MORPH_FORMS[case.entry]:
            forms[case.entry][name].add(args[name])
        seen["c"].add(args["c"])
        seen["mapping"].add(case.inputs["src"]["mapping"])
        names = morph_sets(case)
        if len(names) == 1:
            seen["set"] |= names
    assert seen == {
        "entry": {MORPH_ENTRY, FILTER_ENTRY, EXCEPTIONAL_ENTRY},
        "set": set(MORPH_SETS),
        "c": {1, 3, 4},
        "mxcsr": {"default", "daz_ftz"},
        "mapping": {"unit", "tiny"},
        "word": set(EXCEPTIONAL_WORDS),
        "position": set(EXCEPTIONAL_POSITIONS),
    }
    assert forms == {
        entry: {name: set(found) for name, found in values.items()}
        for entry, values in MORPH_FORMS.items()
    }
    assert 0 < tall < images / 4


def check_resample_sweep():
    """D4 for resample: filter, gamma, vector_clip, channels, set and state from the
    manifest's values, h and w in 1-64 and one case in eight with h in 65-300, th and tw
    in 1-64, and a clean src on unit or tiny."""
    cases = sweep_cases("resample", 600, 11)
    assert cases == sweep_cases("resample", 600, 11)
    assert cases != sweep_cases("resample", 600, 12)
    assert [case.id for case in cases] == [f"sweep-11-{n}" for n in range(600)]
    keys = ("filter", "gamma", "vector_clip", "c", "set", "mxcsr", "mapping")
    seen = {key: set() for key in keys}
    tall = 0
    for case in cases:
        check_resample_recipe(case, sweep=True)
        args = case.args
        assert 1 <= args["w"] <= 64 and 1 <= args["h"] <= 300, case.id
        assert 1 <= args["th"] <= 64 and 1 <= args["tw"] <= 64, case.id
        tall += args["h"] > 64
        for key in ("filter", "gamma", "vector_clip", "c"):
            seen[key].add(args[key])
        seen["mxcsr"].add(case.mxcsr)
        seen["mapping"].add(case.inputs["src"]["mapping"])
        names = resample_sets(case)
        if len(names) == 1:
            seen["set"] |= names
    assert seen == {
        "filter": set(RESAMPLE_FILTERS),
        "gamma": {0, 1},
        "vector_clip": {0, 1},
        "c": {1, 2, 3, 4},
        "set": {"clean", *RESAMPLE_SETS},
        "mxcsr": {"default", "daz_ftz"},
        "mapping": {"unit", "tiny"},
    }
    assert 0 < tall < len(cases) / 4


def check_palette_sweep():
    """D4 for palette: pixels in 1-300, channels, capacity, set and state from the
    manifest's values, and a clean src on unit or tiny."""
    cases = sweep_cases("palette", 600, 11)
    assert cases == sweep_cases("palette", 600, 11)
    assert cases != sweep_cases("palette", 600, 12)
    assert [case.id for case in cases] == [f"sweep-11-{n}" for n in range(600)]
    keys = ("channels", "capacity", "set", "mxcsr", "mapping")
    seen = {key: set() for key in keys}
    pixels = set()
    for case in cases:
        check_palette_recipe(case, sweep=True)
        pixels.add(case.args["pixels"])
        seen["channels"].add(case.args["channels"])
        seen["capacity"].add(case.args["capacity"])
        seen["mxcsr"].add(case.mxcsr)
        seen["mapping"].add(case.inputs["src"]["mapping"])
        names = palette_sets(case)
        if len(names) == 1:
            seen["set"] |= names
    assert seen == {
        "channels": set(PALETTE_CHANNELS),
        "capacity": set(PALETTE_CAPACITIES),
        "set": set(PALETTE_SETS),
        "mxcsr": {"default", "daz_ftz"},
        "mapping": {"unit", "tiny"},
    }
    # Few and many pixels: below the capacity 16, and above 256.
    assert min(pixels) < 16 and max(pixels) > 256


def check_dither_sweep():
    """D4 for dither: the entry, form, colors in 1-40, channels, mode (and algorithm with
    mode 2), set and state from the manifest's values; h and w in 1-64, and one case in
    eight with h in 65-300; a clean unit-form src on unit or grid; a mixed case on the
    unit form."""
    cases = sweep_cases("dither", 900, 11)
    assert cases == sweep_cases("dither", 900, 11)
    assert cases != sweep_cases("dither", 900, 12)
    assert [case.id for case in cases] == [f"sweep-11-{n}" for n in range(900)]
    keys = ("entry", "form", "c", "mode", "algorithm", "set", "mxcsr", "mapping")
    seen = {key: set() for key in keys}
    counts = set()
    tall = 0
    for case in cases:
        check_dither_recipe(case, sweep=True)
        args = case.args
        assert 1 <= args["w"] <= 64 and 1 <= args["h"] <= 300, case.id
        tall += args["h"] > 64
        counts.add(args["count"])
        seen["entry"].add(case.entry)
        seen["c"].add(args["c"])
        seen["mxcsr"].add(case.mxcsr)
        seen["mapping"].add(case.inputs["src"]["mapping"])
        if case.entry != RIEMERSMA_ENTRY:
            seen["mode"].add((case.entry, args["mode"]))
            seen["algorithm"].add(args["algorithm"])
        forms, names = dither_forms(case), dither_sets(case)
        if "mixed" in names:
            assert "unit" in forms, case.id
        if len(forms) == 1:
            seen["form"] |= forms
        if len(names) == 1:
            seen["set"] |= names
    assert seen == {
        "entry": set(golden_kernels.KERNEL_ENTRIES["dither"]),
        "form": set(DITHER_FORMS),
        "c": set(DITHER_CHANNELS),
        "mode": {(DITHER_APPLY, m) for m in (0, 2, 3)}
        | {(DITHER_ENTRY, m) for m in (0, 2)},
        "algorithm": set(range(8)),
        "set": set(DITHER_SETS),
        "mxcsr": {"default", "daz_ftz"},
        "mapping": {"unit", "grid"},
    }
    assert counts == set(range(1, golden_kernels.SWEEP_MAX_COLORS + 1))
    assert 0 < tall < len(cases) / 4


def check_composite_sweep():
    """D4 for composite: a geometry by canvas()'s rules with sizes 1-40. One case in
    eight has a constant base and one in eight a constant overlay; one overlay in four
    overlaps fully (the base's size at x = y = 0), the others take their own size and an
    offset that can miss the base; every mode, channel count, crop, set and state; a
    clean case's inputs on unit or tiny. Some cases take D13's shortcut, some miss it,
    and some have no region (pixels 0)."""
    cases = sweep_cases("composite", 900, 11)
    assert cases == sweep_cases("composite", 900, 11)
    assert cases != sweep_cases("composite", 900, 12)
    assert [case.id for case in cases] == [f"sweep-11-{n}" for n in range(900)]
    keys = ("mode", "bc", "oc", "crop", "constant", "set", "mxcsr", "mapping")
    seen = {key: set() for key in keys}
    sides, shortcut, empty = set(), 0, 0
    for case in cases:
        check_composite_recipe(case, sweep=True)
        args = case.args
        sides |= {args[key] for key in ("bh", "bw", "oh", "ow")}
        for key in ("mode", "bc", "oc", "crop"):
            seen[key].add(args[key])
        seen["constant"].add((args["base_constant"], args["overlay_constant"]))
        seen["mxcsr"].add(case.mxcsr)
        seen["mapping"].add(case.inputs["base"]["mapping"])
        names = composite_sets(case)
        if len(names) == 1:
            seen["set"] |= names
        shortcut += all(shortcut_clauses(args).values())
        empty += golden_kernels.composite_geometry(args)[9] == 0
    assert seen == {
        "mode": set(range(23)),
        "bc": {1, 3, 4},
        "oc": {1, 3, 4},
        "crop": {0, 1},
        "constant": {(0, 0), (1, 0), (0, 1)},
        "set": set(golden_kernels.COMPOSITE_SETS),
        "mxcsr": {"default", "daz_ftz"},
        "mapping": {"unit", "tiny"},
    }
    assert sides == set(range(1, golden_kernels.SWEEP_MAX_SIDE + 1))
    assert 0 < shortcut < len(cases) / 2 and 0 < empty < len(cases) / 2


def check_blend_sweep():
    """D4 for blend: the entry, then one cn_blend_images case in four in the AVX2 unit's
    8-lane form (SIMD_MODES x SIMD_PAIRS), the others any mode and channel pair; the
    count (pixels or elements) in 1-SWEEP_MAX_COUNT elements, so some calls run several
    chunks with odd chunk lengths; every set and state; a clean case on unit or tiny."""
    cases = sweep_cases("blend", 900, 11)
    assert cases == sweep_cases("blend", 900, 11)
    assert cases != sweep_cases("blend", 900, 12)
    assert [case.id for case in cases] == [f"sweep-11-{n}" for n in range(900)]
    keys = ("entry", "mode", "pair", "set", "mxcsr", "mapping")
    seen = {key: set() for key in keys}
    simd = chunked = 0
    for case in cases:
        check_blend_recipe(case, sweep=True)
        args = case.args
        seen["entry"].add(case.entry)
        seen["mode"].add(args["mode"])
        seen["mxcsr"].add(case.mxcsr)
        seen["mapping"].add(case.inputs["overlay"]["mapping"])
        names = blend_sets(case)
        if len(names) == 1:
            seen["set"] |= names
        images = case.entry == golden_kernels.BLEND_IMAGES
        chunked += args["pixels" if images else "count"] > 65536
        if images:
            pair = (args["oc"], args["bc"])
            seen["pair"].add(pair)
            simd += args["mode"] in (0, 1) and pair in golden_kernels.SIMD_PAIRS
    assert seen == {
        "entry": set(golden_kernels.KERNEL_ENTRIES["blend"]),
        "mode": set(range(23)),
        "pair": set(golden_kernels.BLEND_PAIRS),
        "set": set(golden_kernels.BLEND_SETS),
        "mxcsr": {"default", "daz_ftz"},
        "mapping": {"unit", "tiny"},
    }
    assert len(cases) / 16 < simd < len(cases) / 4
    assert 0 < chunked < len(cases) / 2


def check_tail_sweep():
    """D4 for tail: the entry, then its arguments from the manifest's axes; image heights
    in 1-64 and one in eight in 65-300, widths in 1-64, the tile's kernel sides in 1-33
    (some reach past the image, folding a border over several periods) and any tile of
    the padded image, a plane up to 2 (tile) or 3 (store) rows and columns larger than it
    needs; the normal output's pixels in 1-140,000, so some calls run several chunks with
    odd chunk lengths (the other entries' multi-chunk calls are the manifest's groups W
    and B); every set and state, a clean case's float32 inputs on unit or tiny."""
    cases = sweep_cases("tail", 1000, 11)
    assert cases == sweep_cases("tail", 1000, 11)
    assert cases != sweep_cases("tail", 1000, 12)
    assert [case.id for case in cases] == [f"sweep-11-{n}" for n in range(1000)]
    keys = ("entry", "border", "padding", "c", "normal", "caption", "set", "mxcsr")
    seen = {key: set() for key in (*keys, "mapping")}
    tall, reach, extra, long = set(), 0, set(), []
    for case in cases:
        check_tail_recipe(case, sweep=True)
        a = case.args
        seen["entry"].add(case.entry)
        seen["mxcsr"].add(case.mxcsr)
        names = tail_sets(case)
        if len(names) == 1:
            seen["set"].add((case.entry, *names))
        height = a.get("h", a.get("oh", a.get("height")))
        if height is not None and height > 64:
            tall.add(case.entry)
        if case.entry == TILE:
            seen["border"].add(a["border"])
            seen["padding"].add(a["padding"])
            seen["c"].add(a["c"])
            seen["mapping"].add(case.inputs["src"]["mapping"])
            oh, ow = a["h"] + 2 * a["padding"], a["w"] + 2 * a["padding"]
            reach += a["kh"] // 2 > oh or a["kw"] // 2 > ow
            extra.add(
                (a["dh"] - a["th"] - a["kh"] + 1, a["dw"] - a["tw"] - a["kw"] + 1)
            )
            assert max(a["kh"], a["kw"]) <= 33 and a["w"] <= 64, case.id
        elif case.entry == NORMAL:
            form = (a["channels"], a["invert_r"], a["invert_g"], a["alpha"])
            seen["normal"].add(form)
            assert a["pixels"] <= golden_kernels.SWEEP_MAX_COUNT, case.id
            if a["pixels"] > 65_536:
                long.append(a["pixels"])
        elif case.entry == CAPTION:
            seen["caption"].add((a["channels"], a["top"]))
            seen["mapping"].add(case.inputs["image"]["mapping"])
        if height is not None:
            assert height <= 300, case.id
    sets = {
        TILE: golden_kernels.COMPOSITE_SETS,
        STORE: ("bits",),
        MULTIPLY: ("bits",),
        NORMAL: golden_kernels.NORMAL_SETS,
        CAPTION: golden_kernels.COMPOSITE_SETS,
    }
    assert seen == {
        "entry": {TILE, STORE, MULTIPLY, NORMAL, CAPTION},
        "border": set(range(5)),
        "padding": {0, 2},
        "c": {1, 3, 4},
        "normal": set(itertools.product((3, 4), (0, 1), (0, 1), (0, 1))),
        "caption": set(itertools.product((1, 3, 4), (0, 1))),
        "set": {(entry, name) for entry, found in sets.items() for name in found},
        "mxcsr": {"default", "daz_ftz"},
        "mapping": {"unit", "tiny"},
    }
    assert tall == {TILE, STORE, MULTIPLY, CAPTION}
    assert reach > 0 and extra == set(itertools.product(range(3), range(3)))
    assert any(pixels % (1 + (pixels - 1) // 65_536) for pixels in long)


@pytest.mark.parametrize("kernel", KERNELS)
def test_sweep_cases_draw_every_axis_from_the_matrix(kernel):
    if kernel in CONVERSION_KERNELS:
        check_conversion_sweep(kernel)
        return
    if kernel == "convolution_border":
        check_border_sweep()
        return
    if kernel == "lens":
        check_lens_sweep()
        return
    if kernel == "separable_tiles":
        check_tiles_sweep()
        return
    if kernel == "lens_power":
        check_power_sweep()
        return
    if kernel == "morphology":
        check_morph_sweep()
        return
    if kernel == "resample":
        check_resample_sweep()
        return
    if kernel == "palette":
        check_palette_sweep()
        return
    if kernel == "dither":
        check_dither_sweep()
        return
    if kernel == "composite":
        check_composite_sweep()
        return
    if kernel == "blend":
        check_blend_sweep()
        return
    if kernel == "tail":
        check_tail_sweep()
        return
    cases = sweep_cases(kernel, 600, 11)
    assert cases == sweep_cases(kernel, 600, 11)
    assert cases != sweep_cases(kernel, 600, 12)
    assert [case.id for case in cases] == [f"sweep-11-{n}" for n in range(600)]
    seen = {"set": set(), "c": set(), "mxcsr": set()}
    axes = {}
    for case in cases:
        spec = golden_kernels.ENTRIES[case.entry]
        form = tuple(case.args[p] for p in spec.form)
        place = case.args[spec.place] if spec.place else None
        assert form in spec.forms
        assert case.args[spec.mirror] in spec.mirrors
        if spec.place:
            assert place in spec.places
        for name, value in spec.fixed.items():
            assert case.args[name] == value
        assert 1 <= case.args["h"] <= 64
        assert 1 <= case.args["w"] <= 64
        assert case.args["c"] in (1, 3, 4)
        assert case.inputs["src"]["shape"] == [
            case.args["h"],
            case.args["w"],
            case.args["c"],
        ]
        for recipe in case.inputs.values():
            assert recipe["seed"] == seed_of(case.id)
        # Tiny arrays can fit several sets (nan and mixed both start 0x7fc12345).
        names = matching_sets(case)
        assert names, case.id
        if len(names) == 1:
            seen["set"] |= names
        seen["c"].add(case.args["c"])
        seen["mxcsr"].add(case.mxcsr)
        axes.setdefault(case.entry, set()).add((form, case.args[spec.mirror], place))
    assert seen == {
        "set": set(SP4A_SETS),
        "c": {1, 3, 4},
        "mxcsr": {"default", "daz_ftz"},
    }
    # Only the clean set draws its src mapping, so a sweep reaches DAZ|FTZ: a
    # clean src has no specials, and some of them are denormal-range.
    tiny = [case for case in cases if case.inputs["src"]["mapping"] == "tiny"]
    clean = [case for case in cases if not case.inputs["src"]["specials"]]
    assert 0 < len(tiny) < len(clean)
    assert all(case in clean for case in tiny)
    for entry in golden_kernels.KERNEL_ENTRIES[kernel]:
        spec = golden_kernels.ENTRIES[entry]
        assert axes[entry] == {
            (form, mirror, place)
            for form in spec.forms
            for mirror in spec.mirrors
            for place in spec.places or (None,)
        }


def test_diff_lists_the_mismatched_ids(tmp_path, capsys):
    path = GOLDEN / "convolution.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    same = tmp_path / "same.json"
    same.write_text(golden_kernels.dumps(data), encoding="utf-8", newline="\n")
    assert golden_kernels.main(["diff", str(path), str(same)]) == 0
    changed_output = data["cases"][3]["id"]
    removed = data["cases"][5]["id"]
    data["cases"][3]["outputs"]["out"] = "0" * 64
    del data["cases"][5]
    changed = tmp_path / "changed.json"
    changed.write_text(golden_kernels.dumps(data), encoding="utf-8", newline="\n")
    capsys.readouterr()
    assert golden_kernels.main(["diff", str(path), str(changed)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert [line for line in lines if line.startswith("mismatch ")] == [
        f"mismatch {changed_output}",
        f"mismatch {removed}",
    ]


def test_rebase_moves_only_the_cases_whose_reference_moved(
    tmp_path, monkeypatch, capsys
):
    # Task 3b1 (plan, Protected): rebase runs every case again and moves one only from
    # the DLL's outputs to its reference (oracle_outputs), recording ORACLE; the
    # header and every other line stay byte for byte. Stubs stand in for B3 and the
    # oracle, which moves the second case.
    dll = tmp_path / "b3.dll"
    dll.write_bytes(b"stub")
    own = {"events": "1" * 64, "out": "2" * 64}
    moved = {"events": "1" * 64, "out": "3" * 64}
    cases = matrix("conversion_small")[:3]
    data = {
        "schema": SCHEMA,
        "kernel": "conversion_small",
        "generated_with": {
            "dll_sha256": hashlib.sha256(b"stub").hexdigest(),
            "numpy": np.__version__,
        },
        "pcg64_check": np.random.PCG64(0).random_raw(4).tolist(),
        "special_sets": {name: SPECIALS[name] for name in SP4A_SETS},
        "cases": [{**dataclasses.asdict(case), "outputs": own} for case in cases],
    }
    path = tmp_path / "conversion_small.json"
    path.write_text(golden_kernels.dumps(data), encoding="utf-8", newline="\n")
    before = path.read_text(encoding="utf-8").splitlines()
    monkeypatch.setattr(golden_kernels, "_load_dll", lambda path: Stub())
    monkeypatch.setattr(golden_kernels, "run_case", lambda dll, case: own)
    monkeypatch.setattr(
        golden_kernels,
        "oracle_outputs",
        lambda dll, case, outputs: moved if case.id == cases[1].id else outputs,
    )
    arguments = ["rebase", "--dll", str(dll), "--kernel", "conversion_small"]
    arguments += ["--manifest", str(path)]
    assert golden_kernels.main(arguments) == 0
    assert f"rebased {cases[1].id}" in capsys.readouterr().out
    after = path.read_text(encoding="utf-8").splitlines()
    assert [
        i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b
    ] == [2]
    assert rebased_ids(path) == [cases[1].id]
    assert load_manifest(path)[1] == [
        (cases[0], own),
        (cases[1], moved),
        (cases[2], own),
    ]
    # Again: nothing moves.
    assert golden_kernels.main(arguments) == 0
    assert path.read_text(encoding="utf-8").splitlines() == after
    # Refused: a case holding neither output set, a manifest of another DLL, and a
    # case naming another provenance.
    other = tmp_path / "refused.json"
    for index, outputs in ((0, moved), (1, {"events": "1" * 64, "out": "4" * 64})):
        refused = json.loads(path.read_text(encoding="utf-8"))
        refused["cases"][index]["outputs"] = outputs
        other.write_text(golden_kernels.dumps(refused), encoding="utf-8", newline="\n")
        with pytest.raises(RuntimeError, match=cases[index].id):
            golden_kernels.main([*arguments[:-1], str(other)])
    refused = json.loads(path.read_text(encoding="utf-8"))
    refused["generated_with"]["dll_sha256"] = "0" * 64
    other.write_text(golden_kernels.dumps(refused), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="not a conversion_small manifest written"):
        golden_kernels.main([*arguments[:-1], str(other)])
    refused = json.loads(path.read_text(encoding="utf-8"))
    refused["cases"][0]["generated_with"] = {"oracle": "elsewhere"}
    other.write_text(golden_kernels.dumps(refused), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="unknown generated_with"):
        load_manifest(other)
    # PORT is the other known provenance: load_manifest accepts it, rebase keeps it
    # on a case holding the DLL's outputs, and rebased_ids names it apart.
    kept = json.loads(path.read_text(encoding="utf-8"))
    kept["cases"][2]["generated_with"] = PORT
    other.write_text(golden_kernels.dumps(kept), encoding="utf-8", newline="\n")
    assert load_manifest(other)[1][2] == (cases[2], own)
    assert golden_kernels.main([*arguments[:-1], str(other)]) == 0
    assert rebased_ids(other) == [cases[1].id]
    assert rebased_ids(other, PORT) == [cases[2].id]
    assert ORACLE == {"oracle": "upstream chaiNNer, CPython 3.14.8, NumPy 2.5.3"}


def test_sweep_isa_requires_the_requested_level(tmp_path, monkeypatch, capsys):
    # Both exit-2 paths through stubs: a DLL without cn_isa_set (B3 and older)
    # and a level the DLL caps below the request.
    assert not golden_kernels.set_isa(NoIsa(), Path("stub"), "scalar")
    assert "stub: no cn_isa_set export" in capsys.readouterr().err
    assert not golden_kernels.set_isa(Capped(), Path("stub"), "avx2")
    assert "cn_isa_set(avx2) returned 0" in capsys.readouterr().err
    assert golden_kernels.set_isa(Capped(), Path("stub"), "scalar")
    # Default-MXCSR cases only, so the child below never imports torch.
    seed = next(
        s
        for s in range(100)
        if all(case.mxcsr == "default" for case in sweep_cases("convolution", 2, s))
    )
    arguments = ["sweep", "--dll", "stub", "--kernel", "convolution", "--count", "2"]
    arguments += ["--seed", str(seed), "--isa", "scalar"]
    refused = tmp_path / "refused.json"
    monkeypatch.setattr(golden_kernels, "_load_dll", lambda path: NoIsa())
    assert golden_kernels.main([*arguments, "--out", str(refused)]) == 2
    assert not refused.exists()
    # The real export, in a child process: the current DLL (lib()'s) at scalar.
    out = tmp_path / "current.json"
    arguments[2] = str(CURRENT_DLL)
    done = subprocess.run(
        [sys.executable, "-B", str(TOOL), *arguments, "--out", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    header, cases = load_manifest(out)
    assert header["generated_with"]["isa"] == "scalar"
    assert [case for case, _ in cases] == sweep_cases("convolution", 2, seed)
