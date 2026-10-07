"""Forced paths: every ISA level gives the scalar level's outputs (SP4 spec 4.4).

The level decides only which implementation runs an output's operation sequence;
the mirror parameters (lanes, fused) decide the sequence. Each test runs its cases
through the C entries at every level this CPU has (cn_isa_set, called while no
kernel runs) and asserts that each level's digests equal the scalar level's under
the case's payload rule (golden_kernels.digest: a mixed-payload case compares NaN
positions and every non-NaN bit, every other case every bit).

Shapes: S1 is multi-chunk for the convolution, and some chunk ends inside an 8-lane
group of its row (the separable filters run S1 in D9's row tiles since SP4b Task 3);
S2 has height 7, widths 1 to 2R + 1 (R = 8, the AVX2 lanes) and 1, 3 or 4
channels; S3 (Gaussian only) has heights 1 to 17 at width 9. Inputs: the clean,
edges, nan and mixed sets at golden_kernels' listed indices, plus a clean src in
the denormal range; each under the default MXCSR and DAZ|FTZ, which run_case sets
on this thread with torch.set_flush_denormal.
"""

import ctypes
import dataclasses
import math
import re
import sys
from pathlib import Path

import golden_kernels
import numpy as np
import pytest
import torch
from golden_kernels import (
    MXCSR_STATES,
    digest,
    load_manifest,
    make_array,
    matrix,
    run_case,
    sweep_cases,
)
from test_golden_kernels import (
    check_border_manifest,
    check_tiles_manifest,
    compose_reference,
    current_outputs,
    d5_corrected,
    d17_outputs,
    shortcut_clauses,
    special_set,
)
from test_isa_dispatch import isa_get, isa_set, straddling_cases
from test_parallel import partition

from nodes.impl import native, native_composite
from nodes.impl.color.color import Color

GOLDEN = Path(__file__).resolve().parent / "golden"
SEPARABLE_SOURCE = Path(__file__).resolve().parents[1] / "src" / "box_complete_ops.c"
# The ISA units' shared helpers, which define the border runs' copy bounds.
SEPARABLE_ISA = (
    Path(__file__).resolve().parents[1] / "include" / "separable_isa_shared.h"
)
CONVOLUTION_ISA = (
    Path(__file__).resolve().parents[1] / "include" / "convolution_isa_shared.h"
)
CONVOLUTION_AVX512 = (
    Path(__file__).resolve().parents[1] / "src" / "convolution_avx512.c"
)
# SP4b (D4): each task appends its kernels; their manifests (B3) and a sweep run
# at every level.
SP4B_KERNELS = (
    "conversion",
    "conversion_small",
    "convolution_border",
    "lens",
    "separable_tiles",
    "lens_power",
    "morphology",
    "resample",
    "palette",
    "dither",
    "composite",
    "blend",
    "tail",
)
# The kernels whose scratch is leased from the shared pool (SP4b D10; palette since
# Task 5, coordinator ruling; composite since Task 6).
POOLED_KERNELS = ("morphology", "resample", "palette", "composite")
SCRATCH_SLOTS = 4
SCRATCH_RETAIN_BYTES = 16 << 20
# The pooled-scratch test's fill words: a NaN, +inf and -inf.
SCRATCH_FILLS = (0x7FC0DEAD, 0x7F800000, 0xFF800000)
SP4B_SWEEP = (200, 20261004)  # cases per kernel, seed
CONVERT_GRAIN = 262144  # cn_pixels_convert_checked's partition grain
CONVOLUTION_GRAIN = 16384  # cn_convolution_spatial's partition grain
B3_CONVOLUTION_GRAIN = 4096  # B3's, at which the convolution manifests were written
COMPOSE_GRAIN = 65536  # cn_lens_compose's partition grain
MORPH_COMBINE_GRAIN = 65536  # cn_morphology_complete's ellipse combine grain
B3_MORPH_COMBINE_GRAIN = 16384  # B3's, at which the morphology manifest was written
# The separable filters' row tiles (D9, box_complete_ops.c's #defines, which
# test_separable_tile_edges_match_b3_at_every_level reads back): the stack buffer's
# floats, B0 (a band's fewest rows), the largest radius with coordinate tables, and
# the fallback's grain.
SEPARABLE_TILE_FLOATS = 32768
SEPARABLE_MIN_BAND = 32
SEPARABLE_TABLE_RADIUS = 4096
SEPARABLE_GRAIN = 1024
# The CRT's rounding-control values (_RC_DOWN, _RC_UP, _RC_CHOP) and mask (_MCW_RC).
DIRECTED = {"down": 0x100, "up": 0x200, "chop": 0x300}
RC_MASK = 0x300
GAUSSIAN_FORMS = golden_kernels.ENTRIES["cn_gaussian_f32"].forms
BOX_FORMS = golden_kernels.ENTRIES["cn_box_separable"].forms
CONVOLUTION_FORMS = golden_kernels.ENTRIES["cn_convolution_spatial"].forms
# (111, 50, 3) is the smallest h giving an off-grid chunk start at paddings 0-2 at
# CONVOLUTION_GRAIN; golden_kernels._S1 keeps (45, 50, 3) for the manifests.
S1 = {"separable": (37, 41, 3), "convolution": (111, 50, 3)}
S2 = [(7, w, c) for w in range(1, 18) for c in (1, 3, 4)]
S3 = [(h, 9, 3) for h in range(1, 18)]
# (set, src mapping): the tiny mapping puts every src value in the denormal range,
# so DAZ|FTZ changes most outputs.
SOURCES = (
    ("clean", "unit"),
    ("clean", "tiny"),
    ("edges", "unit"),
    ("nan", "unit"),
    ("mixed", "unit"),
)


def form_id(form):
    return f"{form[0]:g}x{form[1]:g}"


@pytest.fixture
def levels():
    """Every level this CPU has, scalar first; the original level afterwards."""
    effective, _, cpu = isa_get()
    # Every test starts at the default MXCSR; torch can set DAZ|FTZ here.
    assert torch.set_flush_denormal(False)
    try:
        yield range(cpu + 1)
    finally:
        assert isa_set(effective) == effective


def cases(entry, form, place, mirror, shapes):
    """The entry's cases over the shapes, sources and MXCSR states.

    golden_kernels.case builds each one, so its input rules are the manifests'.
    Default and daz_ftz twins share their inputs: the seed key is the case id
    without its MXCSR state. A convolution kernel holds the src's set, so a NaN is
    in every input; the nan set also runs with a finite kernel, since one
    non-finite tap reaches every output and would hide the src's NaN payloads and
    positions.
    """
    has_kernel = "kernel" in golden_kernels.ENTRIES[entry].params
    found = []
    for shape in shapes:
        for special, mapping in SOURCES:
            for holds in (True, False) if has_kernel and special == "nan" else (True,):
                parts = (entry, form_id(form), place, mirror, special, mapping, holds)
                key = "-".join(map(str, (*parts, "x".join(map(str, shape)))))
                found += [
                    golden_kernels.case(
                        f"{key}-{mxcsr}",
                        entry,
                        form,
                        place,
                        mirror,
                        shape,
                        special,
                        mxcsr,
                        seed_key=key,
                        src_mapping=mapping,
                        kernel_holds_set=holds,
                    )
                    for mxcsr in MXCSR_STATES
                ]
    return found


def assert_levels_equal_scalar(found, levels):
    digests = {}
    for level in levels:
        assert isa_set(level) == level
        digests[level] = [run_case(native.lib(), case) for case in found]
    scalar = digests[0]
    for level in levels:
        mismatched = [
            case.id
            for case, a, b in zip(found, scalar, digests[level], strict=True)
            if a != b
        ]
        assert mismatched == [], native.ISA_LEVELS[level]


def test_s1_ends_a_chunk_inside_a_vector_group():
    """Precondition: in every convolution S1 call some chunk ends inside an 8-lane
    group of its row, which then runs scalar (vector groups never cross a chunk; spec
    4.3). The separable S1 runs in row tiles (D9), whose blocks start on the lane grid;
    the separable fallback's chunk starts: test_separable_tile_edges_match_b3_at_every_level."""
    calls = []
    h, w, c = S1["convolution"]
    for padding in (0, 1, 2):
        row = (w + 2 * padding) * c
        calls.append(((h + 2 * padding) * row, CONVOLUTION_GRAIN, row))
    for count, grain, row in calls:
        starts = [begin for begin, _ in partition(count, grain)[1:]]
        assert any(begin % row % 8 for begin in starts), (count, grain)


@pytest.mark.parametrize("form", GAUSSIAN_FORMS, ids=form_id)
@pytest.mark.parametrize("border", [1, 2, 4])
@pytest.mark.parametrize("lanes", [4, 8])
def test_gaussian_levels_equal_scalar(form, border, lanes, levels):
    shapes = [S1["separable"], *S2, *S3]
    found = cases("cn_gaussian_f32", form, border, lanes, shapes)
    assert_levels_equal_scalar(found, levels)


@pytest.mark.parametrize("form", BOX_FORMS, ids=form_id)
@pytest.mark.parametrize("lanes", [4, 8])
def test_box_levels_equal_scalar(form, lanes, levels):
    found = cases("cn_box_separable", form, None, lanes, [S1["separable"], *S2])
    assert_levels_equal_scalar(found, levels)


@pytest.mark.parametrize("form", CONVOLUTION_FORMS, ids=form_id)
@pytest.mark.parametrize("padding", [0, 1, 2])
@pytest.mark.parametrize("fused", [0, 8])
def test_convolution_levels_equal_scalar(form, padding, fused, levels):
    shapes = [S1["convolution"], *S2]
    found = cases("cn_convolution_spatial", form, padding, fused, shapes)
    assert_levels_equal_scalar(found, levels)


def sp4b_cases():
    """(kernel, case): each SP4b kernel's matrix, then its sweep (D4)."""
    return [
        pytest.param(kernel, case, id=f"{kernel}-{case.id}")
        for kernel in SP4B_KERNELS
        for case in matrix(kernel) + sweep_cases(kernel, *SP4B_SWEEP)
    ]


@pytest.mark.parametrize(("kernel", "case"), sp4b_cases())
def test_sp4b_levels_equal_scalar(kernel, case, levels):
    assert kernel in golden_kernels.KERNELS
    assert_levels_equal_scalar([case], levels)


def manifest_cases(kernel):
    return load_manifest(GOLDEN / f"{kernel}.json")[1]


def holds_denormal(case):
    """Whether an input of the case is a float32 array holding a denormal."""
    for recipe in case.inputs.values():
        if recipe.get("dtype", "float32") != "float32":
            continue
        bits = golden_kernels.make_array(recipe).view(np.uint32) & np.uint32(0x7FFFFFFF)
        if ((bits != 0) & (bits < 0x00800000)).any():
            return True
    return False


def assert_levels_match_b3(found, levels):
    """Each (case, manifest outputs) gives those outputs at every level: B3's (no
    conversion case is re-based on the oracle under NumPy 2.5.3), except exactly the
    cases B3 refused and Consult 8 D-5 dithers (d5_corrected) and the blend clip's
    signed zeros Consult 11 D-17.1 corrects (d17_outputs)."""
    corrected = [
        case.id
        for case, outputs in found
        if d5_corrected(case) or d17_outputs(case, outputs) is not None
    ]
    for level in levels:
        assert isa_set(level) == level
        mismatched = [
            case.id
            for case, outputs in found
            if run_case(native.lib(), case) != outputs
        ]
        assert mismatched == corrected, native.ISA_LEVELS[level]


@pytest.mark.parametrize("kernel", SP4B_KERNELS)
def test_daz_cases_match_b3(kernel, levels):
    """D4: at least 10 DAZ|FTZ cases on denormal-bearing inputs, each equal to its
    manifest outputs (assert_levels_match_b3) at every level (review focus 3: compares
    honour DAZ, bit tests do not)."""
    found = [
        (case, outputs)
        for case, outputs in manifest_cases(kernel)
        if case.mxcsr == "daz_ftz" and holds_denormal(case)
    ]
    assert len(found) >= 10
    assert_levels_match_b3(found, levels)


def scratch_fill():
    """cn_scratch_fill (test-only, SP4b D10): grows every idle pool slot to
    min(bytes, 16 MiB), fills it wholly with the word and returns the slots filled."""
    fill = native.lib()["cn_scratch_fill"]
    fill.argtypes = [ctypes.c_uint32, ctypes.c_size_t]
    fill.restype = ctypes.c_int
    return fill


@pytest.mark.parametrize("kernel", POOLED_KERNELS)
def test_pooled_kernels_ignore_nan_filled_scratch(kernel, levels):
    """Review focus 4 (spec 4.5): B3's fresh large allocations arrived zeroed, a pooled
    lease does not. Each case of the kernel's manifest runs, at every level, once after
    each SCRATCH_FILLS word fills every slot of the pool as a 16 MiB block, which is
    larger than any case's lease; each run still gives the case's outputs
    (test_golden_kernels.current_outputs), so no output reads scratch it did not
    write first. A NaN alone could miss a stale read
    at the start of a max/min chain (maxss/minss return their second operand on NaN, so
    the next update absorbs it); +inf absorbs every max and -inf every min."""
    fill = scratch_fill()
    found = [
        (case, current_outputs(case, outputs))
        for case, outputs in manifest_cases(kernel)
    ]
    for level in levels:
        assert isa_set(level) == level
        mismatched = []
        for case, outputs in found:
            for word in SCRATCH_FILLS:
                assert fill(word, SCRATCH_RETAIN_BYTES) == SCRATCH_SLOTS
                if run_case(native.lib(), case) != outputs:
                    mismatched.append((case.id, hex(word)))
        assert mismatched == [], native.ISA_LEVELS[level]


def sorted_palette(palette, state):
    """prepare_palette's colors (neighborhood_ops.c): deduplicated bit for bit (the
    first occurrence kept), then sorted by the luminance key in Rust's f32::total_cmp
    order, then by first occurrence. The key is the C's float32 expression (the first
    channel alone for one channel; c0 c0 0.2126 + c1 c1 0.7152 + c2 c2 0.0722, plus
    c3 10 for four), evaluated under the case's MXCSR state."""
    lib = native.lib()
    rows = palette.reshape(-1, palette.shape[-1])
    c = rows.shape[1]
    padded = np.zeros((rows.shape[0], 4), np.float32)
    padded[:, :c] = rows
    first = {}
    unique = [
        i
        for i, row in enumerate(padded.view(np.uint32))
        if first.setdefault(row.tobytes(), i) == i
    ]
    colors = rows[unique]
    with golden_kernels.mxcsr(lib, state), np.errstate(all="ignore"):
        if c == 1:
            key = colors[:, 0].copy()
        else:
            key = colors[:, 0] * colors[:, 0] * np.float32(0.2126)
            key = key + colors[:, 1] * colors[:, 1] * np.float32(0.7152)
            key = key + colors[:, 2] * colors[:, 2] * np.float32(0.0722)
            if c == 4:
                key = key + colors[:, 3] * np.float32(10)
    bits = key.view(np.uint32)
    total = np.where(bits >> np.uint32(31) == 1, ~bits, bits ^ np.uint32(0x80000000))
    order = sorted(range(len(unique)), key=lambda j: (int(total[j]), j))
    return colors[order]


def brute_force(colors, src, state):
    """D12's rule for every pixel of a mode 0 call on the sorted colors: each entry's
    distance as palette_distance computes it (0 + d0 d0, then + dk dk in channel order,
    float32, under the state); index 0 when the first distance is NaN, else the first
    index whose distance compares equal to the minimum of the non-NaN ones. Returns the
    indices and, per pixel, how many entries reach that minimum."""
    c = colors.shape[1]
    pixels = src.reshape(-1, c)
    with golden_kernels.mxcsr(native.lib(), state), np.errstate(all="ignore"):
        distance = np.zeros((pixels.shape[0], colors.shape[0]), np.float32)
        for k in range(c):
            delta = colors[None, :, k] - pixels[:, None, k]
            distance = distance + delta * delta
        unordered = np.isnan(distance)
        minimum = np.where(unordered, np.float32(np.inf), distance).min(axis=1)
        equal = distance == minimum[:, None]
        index = np.where(unordered[:, 0], 0, equal.argmax(axis=1))
        reach = equal.sum(axis=1)
    return index, reach


def test_d5_cases_follow_the_linear_rule_at_every_level(levels):
    """Consult 8 D-5: the dither cases B3 refused (test_golden_kernels.d5_corrected:
    300 unique colors, a NaN src) now dither. In mode 0 each gives brute_force's rule
    at every level: the kd tree's answer for a finite pixel, the linear scan's for a
    NaN one. Every case gives the scalar level's outputs at every level."""
    found = [case for case, _ in manifest_cases("dither") if d5_corrected(case)]
    assert len(found) == 8
    scalar = {}
    ruled = 0
    for level in levels:
        assert isa_set(level) == level
        for case in found:
            current = run_case(native.lib(), case)
            assert scalar.setdefault(case.id, current) == current, case.id
            if case.args.get("mode") != 0:
                continue
            src = make_array(case.inputs["src"])
            colors = sorted_palette(make_array(case.inputs["palette"]), case.mxcsr)
            index, _ = brute_force(colors, src, case.mxcsr)
            expected = colors[index].reshape(src.shape)
            assert current["out"] == digest(expected, case.payloads), case.id
            ruled += 1
    assert ruled == 3 * len(levels)


def test_dither_ties_resolve_to_the_lowest_palette_index(levels):
    """D12 (Step 2: palettes below 300 unique colors take nearest_palette's linear scan,
    whose strict < keeps the first index; the brute-force search must keep it): every
    mode 0 cn_neighborhood_palette_apply case of the dither manifest below 300 unique
    colors equals, on B3, brute_force's rule on sorted_palette's colors; and group T
    (grid: duplicate colors, +-0 twins and exact equidistant entries; daz: ties only
    under DAZ|FTZ; nonfinite and nanfirst: infinite and NaN distances, in modes 0, 2 and
    3) equals B3 at every level. Preconditions: the grid cases hold pixels that two or
    more entries reach, and palettes with a duplicate color and a +-0 twin (equal colors
    with different bits); the daz cases hold pixels tied under DAZ|FTZ and not at the
    default MXCSR."""
    found = manifest_cases("dither")
    ruled = grid_ties = daz_only = 0
    duplicate = twin = False
    for case, outputs in found:
        args, form = case.args, case.id.split("-")[2]
        if case.entry != golden_kernels.DITHER_APPLY or args["mode"] != 0:
            continue
        src = make_array(case.inputs["src"])
        palette = make_array(case.inputs["palette"])
        colors = sorted_palette(palette, case.mxcsr)
        if len(colors) >= 300:  # the kd tree, which needs finite values
            continue
        index, reach = brute_force(colors, src, case.mxcsr)
        expected = colors[index].reshape(src.shape)
        assert digest(expected, case.payloads) == outputs["out"], case.id
        ruled += 1
        if form == "grid":
            grid_ties += int((reach > 1).sum())
            rows = palette.reshape(-1, args["c"])
            bits = {row.tobytes() for row in rows.view(np.uint32)}
            values = {tuple(row) for row in rows.astype(np.float64).tolist()}
            duplicate |= len(bits) < len(rows)
            twin |= len(values) < len(bits)
        elif form == "daz" and case.mxcsr == "daz_ftz":
            _, plain = brute_force(sorted_palette(palette, "default"), src, "default")
            daz_only += int(((reach > 1) & (plain == 1)).sum())
    assert ruled >= 50 and grid_ties > 0 and daz_only > 0
    assert duplicate and twin
    ties = [(case, outputs) for case, outputs in found if case.id.startswith("T-")]
    assert len(ties) == 72
    assert_levels_match_b3(ties, levels)


def zero_class_ties(case):
    """Whether a channel of a palette case's src holds two zero-class values with
    different bits that compare equal under the case's state: +0 and -0, and under DAZ
    any two of them and the denormals."""
    bits = make_array(case.inputs["src"]).view(np.uint32)
    magnitude = bits & np.uint32(0x7FFFFFFF)
    limit = 0x00800000 if case.mxcsr == "daz_ftz" else 1
    zero = magnitude < np.uint32(limit)
    return any(len(set(bits[zero[:, k], k].tolist())) > 1 for k in range(bits.shape[1]))


def test_median_cut_zero_ties_match_b3_at_every_level(levels):
    """D12 as built (Task 5, coordinator ruling): where a channel's extremum compares
    equal to zero (+-0, and under DAZ the denormals), the AVX2 extrema may keep another
    tied value than B3's first-seen one; the extrema reach only range = high - low and
    its compares, which that choice leaves unchanged (palette_shared.h). Group Z of the
    palette manifest (edges and denorm on unit, clean on tiny; both MXCSR states) equals
    B3 at every level, so the channels, medians and counts the extrema lead to are B3's.
    Preconditions: every daz_ftz case, and some default case, holds zero-class ties
    (zero_class_ties)."""
    found = [
        (case, outputs)
        for case, outputs in manifest_cases("palette")
        if case.id.startswith("Z-")
    ]
    assert len(found) == 108
    assert all(zero_class_ties(case) for case, _ in found if case.mxcsr == "daz_ftz")
    assert any(zero_class_ties(case) for case, _ in found if case.mxcsr == "default")
    assert_levels_match_b3(found, levels)


def zero_class(bits, state):
    """Whether float32 bits compare equal to zero under the state: +-0, and under DAZ
    the denormals too."""
    limit = 0x00800000 if state == "daz_ftz" else 1
    return (int(bits) & 0x7FFFFFFF) < limit


def root_middle(case):
    """The root split of a group M case: describe takes channel 0 (every channel holds
    -1 and +1, asserted, so every range is 2). Returns the bits of ranks middle - 1 and
    middle (middle = pixels // 2) of channel 0 in key order (f32::total_cmp, the radix
    select's order) and the set of that channel's zero-class bits under the case's
    state."""
    src = make_array(case.inputs["src"])
    assert (src.min(axis=0) == -1).all() and (src.max(axis=0) == 1).all(), case.id
    bits = src[:, 0].copy().view(np.uint32)
    keys = np.where(bits >> np.uint32(31) == 1, ~bits, bits ^ np.uint32(0x80000000))
    ranked = bits[np.argsort(keys, kind="stable")]
    middle = len(bits) // 2
    zeros = {int(b) for b in bits if zero_class(b, case.mxcsr)}
    return int(ranked[middle - 1]), int(ranked[middle]), zeros


def test_median_cut_selection_tie_classes_match_b3_at_every_level(levels):
    """Task 5b: the median cut takes ranks middle (and, for even counts, middle - 1) by
    a radix select over order-preserving keys (palette_ops.c), where B3 ran a quickselect
    and a maxss fold over compares. The two can choose different bits only inside a
    compare class that holds several (+-0, and under DAZ the denormals and +-0); the
    median reaches only > compares and the midpoint, so the outputs are B3's. Group M of
    the palette manifest (the zeros mapping; odd and even counts; capacity 2, the root
    split alone, and 256; both MXCSR states) equals B3 at every level. Preconditions
    (root_middle): in every daz_ftz case the root's middle ranks lie in the zero class,
    which holds several bit patterns there; in some default case of each parity they lie
    in {-0, +0} with both present; and rank middle's value is -0 in some case, +0 in
    another and a denormal in a third."""
    found = [
        (case, outputs)
        for case, outputs in manifest_cases("palette")
        if case.id.startswith("M-")
    ]
    assert len(found) == 16
    parities, picks = set(), set()
    for case, _ in found:
        lower, upper, zeros = root_middle(case)
        ranks = (upper,) if case.args["pixels"] % 2 else (lower, upper)
        tied = all(zero_class(b, case.mxcsr) for b in ranks) and len(zeros) > 1
        if case.mxcsr == "daz_ftz":
            assert tied, case.id
        elif tied:
            parities.add(case.args["pixels"] % 2)
        picks.add(upper)
    assert parities == {0, 1}
    assert {0x00000000, 0x80000000} <= picks
    assert any(zero_class(b, "daz_ftz") and b & 0x7FFFFFFF for b in picks)
    assert_levels_match_b3(found, levels)


def test_composite_shortcut_geometries_match_b3(levels):
    """D13 (Task 6): cn_composite_canvas blends a full overlap straight into out, with
    no fill, gather or paste (shortcut_clauses holds every clause). Group S of the
    composite manifest takes that shortcut: base and overlay channels {1, 3, 4}^2 at a
    full overlap, modes 0, 1, 2, 4, 9, 13, 17 and 22. Group N holds its near misses (the
    paste one column or row off, a padded canvas, a constant base or overlay, an overlay
    larger than the region): none takes it, and together they fail every clause. Both
    groups equal B3 at every level."""
    found = [
        (case, outputs)
        for case, outputs in manifest_cases("composite")
        if case.id.split("-")[0] in ("S", "N")
    ]
    shortcut = [case for case, _ in found if case.id.startswith("S-")]
    near = [case for case, _ in found if case.id.startswith("N-")]
    assert len(shortcut) == 72 and len(near) == 40
    channels = {(case.args["bc"], case.args["oc"]) for case in shortcut}
    assert channels == {(b, o) for b in (1, 3, 4) for o in (1, 3, 4)}
    assert {case.args["mode"] for case in shortcut} == {0, 1, 2, 4, 9, 13, 17, 22}
    assert all(all(shortcut_clauses(case.args).values()) for case in shortcut)
    failed = set()
    for case in near:
        missed = {
            name for name, held in shortcut_clauses(case.args).items() if not held
        }
        assert missed, case.id
        failed |= missed
    assert failed == set(shortcut_clauses(shortcut[0].args))
    assert_levels_match_b3(found, levels)


BLEND_GRAIN = 65536  # cn_blend_images' partition grain, in pixels


def test_blend_chunks_start_off_the_lane_grid(levels):
    """Review focus 2 (Task 6 Step 6): the blend manifest's group W runs modes 0 and 1
    with (overlay, base) channels (1, 1) and (3, 3), the AVX2 unit's 8-lane form, on
    196,613 pixels: 4 chunks of 49,154/49,153 pixels at BLEND_GRAIN, whose element
    starts lie off the 8-lane grid at both channel counts. A vector group never crosses
    a chunk (each chunk's tail pixels run image_pixels), so each case equals B3 at
    every level."""
    found = [
        (case, outputs)
        for case, outputs in manifest_cases("blend")
        if case.id.startswith("W-")
    ]
    assert {
        (case.args["mode"], case.args["oc"], case.args["bc"]) for case, _ in found
    } == {(mode, c, c) for mode in (0, 1) for c in (1, 3)}
    for case, _ in found:
        pixels, channels = case.args["pixels"], case.args["oc"]
        starts = [begin for begin, _ in partition(pixels, BLEND_GRAIN)[1:]]
        assert len(starts) == 3 and all(begin * channels % 8 for begin in starts)
    assert_levels_match_b3(found, levels)


TILE = golden_kernels.TILE_ENTRY
STORE = golden_kernels.STORE_ENTRY
MULTIPLY = golden_kernels.MULTIPLY_ENTRY
NORMAL = golden_kernels.NORMAL_ENTRY


def tail_partition(case):
    """A tail case's cn_parallel_for count and grain, and the row length its chunks cut
    (None for the multiply's row jobs and the normal output's pixels): the tile over its
    dh x dw plane and the store over its th x tw tile at 65,536; the multiply over h + 1
    jobs at 65,536 // w + 1 (a single column runs as one row of h); the normal output
    over its pixels at 65,536; the caption over (height + caption_height) x width at
    16,384."""
    a = case.args
    if case.entry == TILE:
        return a["dh"] * a["dw"], 65536, a["dw"]
    if case.entry == STORE:
        return a["th"] * a["tw"], 65536, a["tw"]
    if case.entry == MULTIPLY:
        h, w = (1, a["h"]) if a["w"] == 1 else (a["h"], a["w"])
        return h + 1, 65536 // w + 1, None
    if case.entry == NORMAL:
        return a["pixels"], 65536, None
    return (a["height"] + a["caption_height"]) * a["width"], 16384, a["width"]


def tile_span(case, column):
    """The span of a tile row that a plane column lies in, by its virtual column: the
    left or right border edge (outside the padded image), the zero frame, the image, or
    the plane's zero columns."""
    a = case.args
    ow = a["w"] + 2 * a["padding"]
    virtual = a["x"] + column - a["kw"] // 2
    if column >= a["tw"] + a["kw"] - 1:
        return "zero"
    if virtual < 0:
        return "left edge"
    if virtual >= ow:
        return "right edge"
    return "image" if a["padding"] <= virtual < a["padding"] + a["w"] else "frame"


def test_tail_chunks_start_mid_row_with_odd_lengths(levels):
    """Review focus 2 (Task 8): every tail entry's group W case runs several chunks, one
    of odd length (tail_partition); the tile's, the store's and the caption's chunks
    start mid-row, where the row loops begin with a partial row: the tile's in its
    image, in both border edges and in its frame. Each equals B3 at every level."""
    found = [
        (case, outputs)
        for case, outputs in manifest_cases("tail")
        if case.id.startswith("W-")
    ]
    assert len(found) == 8
    spans = set()
    for case, _ in found:
        count, grain, row = tail_partition(case)
        chunks = partition(count, grain)
        assert len(chunks) > 1, case.id
        assert any((end - begin) % 2 for begin, end in chunks), case.id
        starts = [begin for begin, _ in chunks[1:]]
        if row is not None:
            assert all(begin % row for begin in starts), case.id
        if case.entry == TILE:
            spans |= {tile_span(case, begin % case.args["dw"]) for begin in starts}
    assert spans == {"image", "left edge", "right edge", "frame"}
    assert_levels_match_b3(found, levels)


# Task 6 Step 6: MULTIPLY's operands, every ordered pair of these words: quiet and
# signalling NaNs of both signs, the default NaN, +-inf, -0 and finite values (-0 x inf
# generates the default NaN).
MULTIPLY_WORDS = (
    0x7FC12345,
    0xFFC54321,
    0x7F812345,
    0xFF812345,
    0xFFC00000,
    0x7F800000,
    0x80000000,
    0x3F000000,
    0x40000000,
)


def first_source_product(overlay, base):
    """B3's MULTIPLY on float32 bits (apply_mode's mulss with the overlay as the first
    source): a NaN overlay gives itself with the quiet bit set; else a NaN base gives
    itself quieted; else the float32 product."""
    quiet = np.uint32(0x00400000)
    a, b = overlay.view(np.uint32), base.view(np.uint32)
    a_nan = (a & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    b_nan = (b & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    with np.errstate(all="ignore"):
        product = (overlay * base).view(np.uint32)
    return np.where(a_nan, a | quiet, np.where(b_nan, b | quiet, product))


@pytest.mark.parametrize("channels", [1, 3])
def test_blend_multiply_keeps_the_overlays_nan_at_every_level(channels, levels):
    """Task 6 Step 6: where an overlay NaN meets a base NaN, B3's MULTIPLY keeps the
    overlay's, quieted (its mulss takes the overlay as the first source). The AVX2
    unit's 8-pixel blocks select it explicitly (the compiler may commute vmulps), and
    its tail pixels run image_pixels, so every level gives first_source_product's bits
    on every ordered pair of MULTIPLY_WORDS, in blocks and in tails, at both channel
    counts of the unit's form (61 pixels at 3 channels: 7 blocks and 5 tail pixels; 183
    at 1: 22 blocks and 7)."""
    count = 183
    pixels = count // channels
    words = np.array(MULTIPLY_WORDS, np.uint32)
    n = words.size
    index = np.arange(count)
    overlay = words[index % n].view(np.float32)
    base = words[(index // n + index) % n].view(np.float32)
    picks = zip(index % n, (index // n + index) % n, strict=True)
    pairs = {(int(a), int(b)) for a, b in picks}
    assert len(pairs) == n * n
    meets = np.isnan(overlay) & np.isnan(base)
    blocks = pixels // 8 * 8 * channels  # the elements the 8-pixel blocks cover
    assert meets[:blocks].any() and meets[blocks:].any()
    expected = first_source_product(overlay, base)
    blend = native.lib()["cn_blend_images"]
    blend.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t] + [ctypes.c_int] * 3
    blend.restype = ctypes.c_int
    for level in levels:
        assert isa_set(level) == level
        out = np.full(count, 0xFFFFFFFF, np.uint32)
        status = blend(
            overlay.ctypes.data,
            base.ctypes.data,
            out.ctypes.data,
            pixels,
            channels,
            channels,
            1,
        )
        assert status == 0
        mismatched = (out != expected).nonzero()[0]
        assert mismatched.tolist() == [], native.ISA_LEVELS[level]


# Task 8: the normal output's inputs, every ordered (dx, dy) pair of these words but
# two NaNs: quiet and signalling NaNs of both signs, +-inf, -0, finite values and a
# denormal.
NORMAL_WORDS = (
    0x7FC12345,
    0xFFC54321,
    0x7F812345,
    0xFF812345,
    0x7F800000,
    0xFF800000,
    0x80000000,
    0x3F000000,
    0xBF400000,
    0x00000001,
)


def normal_model(dx, dy, alpha, invert_r, invert_g, channels):
    """B3's normal_output_range (0x180008530, VERIFIED(dumpbin) in task8-verdict.md) in
    NumPy float32 operations in its order: sqrtf is the CRT's, sqrtss on every
    non-negative, +inf or NaN input (its NaN path sets the quiet bit), and fabsf is
    (float)fabs((double)z), the double's sign cleared. With at most one NaN per pixel
    no two payloads meet, so the bits are determined."""
    with np.errstate(all="ignore"):
        x, y = dx.copy(), dy.copy()
        square = x * x
        square = square + y * y
        square = square + np.float32(4)
        length = np.sqrt(square)
        x, y = x / length, y / length
        z = np.float32(2) / length
        if invert_r:
            x = -x
        if invert_g:
            y = -y
        x, y = x + np.float32(1), y + np.float32(1)
        columns = [np.abs(z), y * np.float32(0.5), x * np.float32(0.5)]
    if channels == 4:
        columns.append(alpha)
    return np.stack(columns, axis=1).view(np.uint32)


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("invert", [0, 1])
def test_normal_output_nan_and_sign_forms_at_every_level(channels, invert, levels):
    """Task 8: on every ordered pair of NORMAL_WORDS with at most one NaN (84 pixels: 10
    groups of 8 and 4 tail pixels), each level gives normal_model's bits: a NaN's payload
    quieted, its sign flipped by an inversion and cleared in z, the default NaN of
    inf / inf, and alpha's words (signalling NaNs included) copied."""
    words = np.array(NORMAL_WORDS, np.uint32)
    nan = np.isnan(words.view(np.float32))
    pairs = [
        (i, j)
        for i in range(words.size)
        for j in range(words.size)
        if not (nan[i] and nan[j])
    ]
    assert len(pairs) == 84
    dx = words[[i for i, _ in pairs]].view(np.float32)
    dy = words[[j for _, j in pairs]].view(np.float32)
    alpha = words[np.arange(84) % words.size].view(np.float32)
    expected = normal_model(dx, dy, alpha, invert, invert, channels)
    assert not (expected[:, 0] >> 31).any()  # fabsf clears every sign, NaNs' included
    generated = expected[:, 1:3] & np.uint32(0x7FFFFFFF) == np.uint32(0x7FC00000)
    assert generated.any()  # inf / inf
    assert ((expected[:, 0] & np.uint32(0x7FFFFFFF)) > np.uint32(0x7FC00000)).any()
    normal = native.lib()["cn_normal_output"]
    normal.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_size_t] + [ctypes.c_int] * 3
    normal.restype = ctypes.c_int
    for level in levels:
        assert isa_set(level) == level
        out = np.full((84, channels), 0xFFFFFFFF, np.uint32)
        status = normal(
            dx.ctypes.data,
            dy.ctypes.data,
            alpha.ctypes.data if channels == 4 else None,
            out.ctypes.data,
            84,
            invert,
            invert,
            channels,
        )
        assert status == 0
        mismatched = sorted({int(i) for i in (out != expected).nonzero()[0]})
        assert mismatched == [], native.ISA_LEVELS[level]


# Task 8 fix round 1: the multiply's operand words, each placed in turn in ar, ai, br and
# b[stride] of a pair whose other three operands are PAIR_FILLERS: quiet and
# signalling NaNs of both signs, +-inf, +-0 and finite values.
PAIR_WORDS = (
    0x7FF8000000012345,
    0xFFF8000000054321,
    0x7FF0000000012345,
    0xFFF0000000054321,
    0x7FF0000000000000,
    0xFFF0000000000000,
    0x0000000000000000,
    0x8000000000000000,
    0x3FF8000000000000,  # 1.5
    0xBFE8000000000000,  # -0.75
)
PAIR_FILLERS = (0.5, -1.25, 3.0)


def multiply_model(ar, ai, br, bim):
    """multiply_pair (spectral_shared.h) in NumPy float64 operations in its order:
    bi = -b[stride], real = ar * br - ai * bi, imag = ar * bi + ai * br."""
    with np.errstate(all="ignore"):
        bi = -bim
        return ar * br - ai * bi, ar * bi + ai * br


def test_spectral_multiply_keeps_single_nan_payloads_at_every_level(levels):
    """Task 8 fix round 1 (review Minor 1): where at most one NaN meets an operation the
    payload is kept bit for bit, so each level gives multiply_model's bits on one row of
    43 pairs (an odd count: at avx2 21 two-pair vectors and the last pair through
    multiply_pair): every PAIR_WORDS word in each operand slot beside fillers, two pairs
    whose inf x 0 makes the default NaN, and one filler pair. A NaN in b[stride] comes
    out with its sign flipped (bi = -b[stride] is a sign flip)."""
    fillers = np.array(PAIR_FILLERS, np.float64).view(np.uint64)
    rows = []
    for slot in range(4):
        for word in PAIR_WORDS:
            pair = [int(fillers[(slot + k) % 3]) for k in range(4)]
            pair[slot] = word
            rows.append(pair)
    extra = [
        [np.inf, 0.5, 0.0, -1.25],
        [0.0, -np.inf, 0.5, 0.0],
        [0.5, -1.25, 3.0, 0.5],
    ]
    extra = np.array(extra, np.float64).view(np.uint64)
    pairs = np.concatenate([np.array(rows, np.uint64), extra]).view(np.float64)
    assert len(pairs) == 43
    ar, ai, br, bim = (np.ascontiguousarray(column) for column in pairs.T)
    with np.errstate(all="ignore"):
        products = (ar * br, ai * -bim, ar * -bim, ai * br)
    assert not (np.isnan(products[0]) & np.isnan(products[1])).any()
    assert not (np.isnan(products[2]) & np.isnan(products[3])).any()
    width = 2 * len(pairs) + 1  # odd: no Nyquist column; job 0 runs a[0] *= b[0]
    a = np.empty(width, np.float64)
    b = np.empty(width, np.float64)
    a[0], b[0] = 2.0, 0.25
    a[1::2], a[2::2], b[1::2], b[2::2] = ar, ai, br, bim
    expected = a.copy()
    expected[0] = a[0] * b[0]
    expected[1::2], expected[2::2] = multiply_model(ar, ai, br, bim)
    expected_bits = expected.view(np.uint64)
    nan = np.isnan(expected)
    assert nan.any() and (expected_bits[nan] >> np.uint64(63)).any()
    assert not (expected_bits[nan] >> np.uint64(63)).all()
    multiply = native.lib()["cn_spectral_multiply"]
    multiply.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 2
    multiply.restype = ctypes.c_int
    for level in levels:
        assert isa_set(level) == level
        out = a.copy()
        assert multiply(out.ctypes.data, b.ctypes.data, 1, width) == 0
        mismatched = (out.view(np.uint64) != expected_bits).nonzero()[0]
        assert mismatched.tolist() == [], native.ISA_LEVELS[level]


CAPTION_GRAIN = 16384  # cn_caption_compose's partition grain, in output pixels
# Task 8 fix round 1 (review Minor 2): (height, width, caption_height, top). Top 1: 310 x
# 64 pixels run 2 chunks of 9,920; the second starts inside the caption's 19,200 pixels and
# the caption/image boundary falls inside it. Top 0: 340 x 64 run 2 chunks of 10,880; the
# image's 19,200 pixels end inside the second.
CAPTION_STRADDLES = ((10, 64, 300, 1), (300, 64, 40, 0))


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize(
    "form", CAPTION_STRADDLES, ids=lambda form: f"top{form[3]}-{form[0]}x{form[1]}"
)
def test_caption_chunks_straddle_its_runs_at_every_level(form, channels, levels):
    """Task 8 fix round 1 (review Minor 2): caption_range's runs where a chunk starts
    inside the caption (top 1) and where the caption/image boundary falls inside a chunk
    (both forms) give, at every level, the model's bits: the image copied (NaN payloads,
    a signalling NaN, -0 and a denormal kept) beside the raster's np.float32(k) /
    np.float32(255), the same correctly rounded quotient, in every channel but channel 3,
    which is 1.0."""
    height, width, rows, top = form
    image_pixels, caption_pixels = height * width, rows * width
    chunks = partition(image_pixels + caption_pixels, CAPTION_GRAIN)
    boundary = caption_pixels if top else image_pixels
    assert len(chunks) == 2
    assert any(begin < boundary < end for begin, end in chunks)
    if top:
        assert 0 < chunks[1][0] < caption_pixels
    rng = np.random.Generator(np.random.PCG64(20261005))
    image = rng.random((height, width, channels), dtype=np.float32)
    words = image.view(np.uint32).reshape(-1)
    for index, word in enumerate((0x7FC12345, 0xFF812345, 0x80000000, 0x00000001)):
        words[index * (words.size // 4) + 3] = word
    caption = (np.arange(caption_pixels) % 256).astype(np.uint8).reshape(rows, width)
    values = caption.astype(np.float32) / np.float32(255)
    raster = np.repeat(values[..., None], channels, axis=2)
    if channels == 4:
        raster[..., 3] = np.float32(1)
    parts = (raster, image) if top else (image, raster)
    expected = np.concatenate(parts).view(np.uint32)
    compose = native.lib()["cn_caption_compose"]
    compose.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t] * 4 + [ctypes.c_int]
    compose.restype = ctypes.c_int
    for level in levels:
        assert isa_set(level) == level
        out = np.full(expected.shape, 0xFFFFFFFF, np.uint32)
        status = compose(
            image.ctypes.data,
            caption.ctypes.data,
            out.ctypes.data,
            height,
            width,
            channels,
            rows,
            top,
        )
        assert status == 0
        mismatched = sorted({int(i) for i in (out != expected).nonzero()[0]})
        assert mismatched == [], native.ISA_LEVELS[level]


COMPOSITE_GRAIN = 65536  # canvas_fill's and canvas_gather's partition grain
# Task 6: canvases off D13's shortcut whose fill (height x width pixels) and gather
# (region pixels) run several chunks, some starting inside a row: (base (h, w, c),
# base constant, overlay (h, w, c), overlay constant, x, y, crop, mode). A constant
# layer takes the image's h and w. In order: a padded canvas, its 3-channel base
# expanded to 4; a cropped gray base under a 3-channel overlay (the base region's
# canvas_channels taken from 3-channel canvas pixels); a constant base; a constant
# overlay beside a padded base (crop ignored); a constant overlay over the whole base.
CANVAS_FORMS = (
    ((300, 307, 3), 0, (281, 290, 4), 0, 9, -7, 0, 9),
    ((301, 307, 1), 0, (310, 320, 3), 0, -5, 6, 1, 17),
    ((281, 300, 4), 1, (281, 300, 3), 0, 3, 2, 0, 1),
    ((301, 300, 4), 0, (301, 300, 3), 1, -2, 0, 1, 1),
    ((291, 301, 3), 0, (291, 301, 3), 1, 0, 0, 0, 4),
)


def layer_pixels(array, constant, shape, channels):
    """A composite layer's (h, w, channels) values as layer_value gives them: an image's
    channels (its gray value in each colour channel, alpha 1 where it has none), or a
    constant's one pixel at every position."""
    h, w, c = shape
    values = array.reshape((1, 1, c) if constant else (h, w, c))
    picked = [
        values[..., 0 if c == 1 else k]
        if k < 3 or c == 4
        else np.ones_like(values[..., 0])
        for k in range(channels)
    ]
    return np.broadcast_to(np.stack(picked, axis=-1), (h, w, channels)).copy()


def numpy_canvas(arrays, form):
    """cn_composite_canvas's out built from copies in NumPy: zeros, the base at its
    place, then the region blended by the DLL's cn_blend_images (the per-element blend
    B3 runs) from the overlay's and the canvas's pixels, pasted back."""
    base_shape, base_constant, overlay_shape, overlay_constant, x, y, crop, mode = form
    args = {
        "bh": base_shape[0],
        "bw": base_shape[1],
        "bc": base_shape[2],
        "base_constant": base_constant,
        "oh": overlay_shape[0],
        "ow": overlay_shape[1],
        "oc": overlay_shape[2],
        "overlay_constant": overlay_constant,
        "x": x,
        "y": y,
        "crop": crop,
    }
    h, w, channels, top, left, pt, pl, ot, ol, rh, rw, padded = (
        golden_kernels.composite_geometry(args)
    )
    canvas = np.zeros((h, w, channels), np.float32)
    bh, bw, _ = base_shape
    canvas[top : top + bh, left : left + bw] = layer_pixels(
        arrays[0], base_constant, base_shape, channels
    )
    canvas_channels = 4 if padded else base_shape[2]
    oc = overlay_shape[2]
    region = (slice(pt, pt + rh), slice(pl, pl + rw))
    base = np.ascontiguousarray(canvas[(*region, slice(0, canvas_channels))])
    pixels = layer_pixels(arrays[1], overlay_constant, overlay_shape, oc)
    overlay = np.ascontiguousarray(pixels[ot : ot + rh, ol : ol + rw])
    blended = np.empty((rh, rw, channels), np.float32)
    blend = native.lib()["cn_blend_images"]
    blend.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t] + [ctypes.c_int] * 3
    blend.restype = ctypes.c_int
    status = blend(
        overlay.ctypes.data,
        base.ctypes.data,
        blended.ctypes.data,
        rh * rw,
        oc,
        canvas_channels,
        mode,
    )
    assert status == 0
    canvas[region] = blended
    return canvas, args, (h, w, rh, rw)


def test_composite_row_spans_across_chunks_equal_a_numpy_canvas(levels):
    """Task 6 (D13's row loops): canvas_fill and canvas_gather walk row spans of each
    chunk, so a chunk may start or end inside a row; the composite manifest's and
    sweep's canvases (sizes up to 40) are one chunk each. On each CANVAS_FORMS canvas
    (off the shortcut; some fill and some gather chunk starts inside a row at
    COMPOSITE_GRAIN), with a NaN, a signalling NaN, -0 and +inf in each image, the
    bridge's out equals numpy_canvas bit for bit at every level, after each
    SCRATCH_FILLS word fills the pool (the pooled region buffers are written before they
    are read)."""
    fill = scratch_fill()
    rng = np.random.Generator(np.random.PCG64(SP4B_SWEEP[1]))
    specials = np.array([0x7FC12345, 0x7F812345, 0x80000000, 0x7F800000], np.uint32)
    for form in CANVAS_FORMS:
        base_shape, base_constant, overlay_shape, overlay_constant = form[:4]
        layers, arrays = [], []
        for shape, constant in (
            (base_shape, base_constant),
            (overlay_shape, overlay_constant),
        ):
            if constant:
                array = rng.random(shape[2], dtype=np.float32)
                layers.append(Color(tuple(float(value) for value in array)))
            else:
                array = rng.random(shape, dtype=np.float32)
                spots = rng.choice(array.size, size=specials.size * 3, replace=False)
                array.reshape(-1).view(np.uint32)[spots] = np.resize(
                    specials, spots.size
                )
                layers.append(array)
            arrays.append(array)
        expected, args, (h, w, rh, rw) = numpy_canvas(arrays, form)
        assert not all(shortcut_clauses(args).values()), form
        fills = [begin for begin, _ in partition(h * w, COMPOSITE_GRAIN)[1:]]
        gathers = [begin for begin, _ in partition(rh * rw, COMPOSITE_GRAIN)[1:]]
        assert any(begin % w for begin in fills), form
        assert any(begin % rw for begin in gathers), form
        x, y, crop, mode = form[4:]
        base, overlay = layers
        for level in levels:
            assert isa_set(level) == level
            for word in SCRATCH_FILLS:
                assert fill(word, SCRATCH_RETAIN_BYTES) == SCRATCH_SLOTS
                with np.errstate(all="ignore"):
                    out = native_composite.canvas(base, overlay, mode, x, y, bool(crop))
                where = (form, native.ISA_LEVELS[level], hex(word))
                assert out.reshape(expected.shape).tobytes() == expected.tobytes(), (
                    where
                )


# The resample sets whose values tell the vertical pass's clip forms apart (NaN, -0, the
# denormals, values just outside [0, 1] and the largest finite ones), and the floats of
# one block of the AVX2 vertical pass (8 groups of 8, SP4b Task 7).
RESAMPLE_CLIP_SETS = ("nan", "edges", "denorm")
RESAMPLE_VERTICAL_BLOCK = 64


def resample_clip(args):
    """cn_resample_filtered's vertical clip form: 0 (none) for the triangle filter (2),
    2 (Vec4's clamp) without gamma and with vector_clip, else 1."""
    if args["filter"] == 2:
        return 0
    return 2 if not args["gamma"] and args["vector_clip"] else 1


def test_resample_vertical_clip_forms_match_b3(levels):
    """SP4b Task 7 (D14): the AVX2 vertical pass clips with B3's instructions, clip 1 as
    min(1, max(0, acc)) and clip 2 as min(max(acc, 0), 1), whose unordered selections
    differ (a NaN acc stays NaN under clip 1 and becomes +0 under clip 2). The resample
    manifest holds at least 4 cases of each clip form on each of RESAMPLE_CLIP_SETS, in
    both MXCSR states, and each form's cases reach the unit's 64-float blocks, its single
    8-float groups and its scalar tail (an output row of tw * c floats); each equals B3
    at every level."""
    found = [
        (case, outputs)
        for case, outputs in manifest_cases("resample")
        if special_set(case) in RESAMPLE_CLIP_SETS
    ]
    for clip in (0, 1, 2):
        for name in RESAMPLE_CLIP_SETS:
            held = [
                case
                for case, _ in found
                if resample_clip(case.args) == clip and special_set(case) == name
            ]
            assert len(held) >= 4, (clip, name)
            assert {case.mxcsr for case in held} == set(MXCSR_STATES), (clip, name)
            strides = {case.args["tw"] * case.args["c"] for case in held}
            assert any(
                stride >= RESAMPLE_VERTICAL_BLOCK
                and stride % RESAMPLE_VERTICAL_BLOCK >= 8
                and stride % 8
                for stride in strides
            ), (clip, name)
    assert_levels_match_b3(found, levels)


# Task 7: a (3, 50, 3) src of one row each of +inf, -inf and this NaN, resampled to
# (1, 25), with (filter, vector_clip) and B3's output word for every element.
RESAMPLE_MEET_NAN = 0x7FC12345
RESAMPLE_MEET_FORMS = ((2, 0, 0xFFC00000), (11, 0, 0xFFC00000), (11, 1, 0x00000000))


def test_resample_vertical_nan_meets_keep_b3s_payload_at_every_level(levels):
    """SP4b Task 7: where the accumulated NaN meets a tap's NaN of another payload, B3's
    addss keeps the accumulated one (its first source), and the AVX2 unit's vaddps may
    take its sources in either order. Every tap weight of the triangle (filter 2, clip 0)
    and Gaussian (11: clip 1, and clip 2 with vector_clip) filters is positive here, so
    each output's horizontal pass gives +inf, -inf and the NaN in its three source rows,
    and its vertical sum becomes the default NaN 0xffc00000 at the second tap, which the
    payload then meets. Every level gives B3's bits: the default NaN under clips 0 and
    1, +0 under clip 2 (which maps any NaN to +0). An output row is 75 floats: one
    64-float block, one group of 8 and a scalar tail of the unit."""
    function = native.lib()["cn_resample_filtered"]
    function.argtypes = (
        [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 5 + [ctypes.c_int] * 3
    )
    function.restype = ctypes.c_int
    words = np.empty((3, 50, 3), np.uint32)
    words[0], words[1], words[2] = 0x7F800000, 0xFF800000, RESAMPLE_MEET_NAN
    src = words.view(np.float32)
    sizes = (*src.shape, 1, 25)  # h, w, c, th, tw
    clips = set()
    for filter_id, vector_clip, word in RESAMPLE_MEET_FORMS:
        args = {"filter": filter_id, "gamma": 0, "vector_clip": vector_clip}
        clips.add(resample_clip(args))
        for level in levels:
            assert isa_set(level) == level
            out = np.full((1, 25, 3), 0x12345678, np.uint32)
            pointers = (src.ctypes.data, out.ctypes.data)
            assert function(*pointers, *sizes, filter_id, 0, vector_clip) == 0
            found = sorted({hex(int(value)) for value in out.reshape(-1)})
            assert found == [hex(word)], (args, native.ISA_LEVELS[level])
    assert clips == {0, 1, 2}


def test_filter_morphology_rejects_dimensions_beyond_int_max():
    """D11: cn_filter_morphology refuses a height or width above INT_MAX (OpenCV's int
    dimensions) with CN_SIZE_OVERFLOW (2), after its argument checks and before any
    allocation or read: here on 1-float buffers that no call may read. (B3 had no such
    bound: its 2**62-byte malloc failed first and returned CN_ALLOCATION_FAILED.)"""
    function = native.lib()["cn_filter_morphology"]
    function.argtypes = (
        [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 5 + [ctypes.c_int] * 2
    )
    function.restype = ctypes.c_int
    src = np.zeros(1, np.float32)
    out = np.full(1, 0x7FC0DEAD, np.uint32)
    for height, width in ((2**31, 2**29), (2**29, 2**31)):
        for cross in (0, 1):
            status = function(
                src.ctypes.data, out.ctypes.data, height, width, 1, 1, 2, cross, 1
            )
            assert status == 2, (height, width, cross)
    assert out[0] == 0x7FC0DEAD


def test_morphology_combine_chunks_start_off_the_lane_grid():
    """Precondition (SP4b Task 4b): the ellipse combine on golden_kernels.MORPH_CHUNKS
    (the morphology manifest's ellipse cases of groups C and T whose src is not
    exceptional) ran 8 chunks at B3's grain, at which the manifest was written, and runs
    2 at MORPH_COMBINE_GRAIN; at both some chunk starts off the 8-lane grid of its row,
    where the AVX2 combine's first group of the chunk starts. So those cases' B3 equality
    at every level (test_golden_kernels, test_daz_cases_match_b3,
    test_pooled_kernels_ignore_nan_filled_scratch) pins that the outputs do not depend on
    the partition."""
    h, w, c = golden_kernels.MORPH_CHUNKS
    row = w * c
    for grain, chunks in ((B3_MORPH_COMBINE_GRAIN, 8), (MORPH_COMBINE_GRAIN, 2)):
        found = partition(h * row, grain)
        assert len(found) == chunks, grain
        assert any(begin % row % 8 for begin, _ in found[1:]), grain


@pytest.mark.skipif(sys.platform != "win32", reason="Public Windows CRT controls")
@pytest.mark.parametrize("mode", DIRECTED)
def test_u8_f32_follows_the_callers_rounding_mode(mode, levels):
    """Review M1, stand-in (iv): u8 -> f32 is B3's divss (float)k / 255.0f, which
    rounds in the caller's MXCSR mode; the table holds its round-to-nearest
    quotients only. Every level, scalar heads and tails included, gives NumPy's
    float32 division under the same mode (count 7: a tail only; 17: two groups and
    a one-element tail; 196,613: 4 chunks whose heads start off the lane grid)."""
    control = ctypes.CDLL("ucrtbase.dll")["_controlfp_s"]
    control.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
    control.restype = ctypes.c_int
    convert = native.lib()["cn_pixels_convert_checked"]
    convert.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    convert.argtypes += [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    convert.restype = ctypes.c_int
    values = np.arange(256, dtype=np.float32)
    nearest = values / np.float32(255)
    saved = ctypes.c_uint()
    assert control(ctypes.byref(saved), 0, 0) == 0
    try:
        observed = ctypes.c_uint()
        assert control(ctypes.byref(observed), DIRECTED[mode], RC_MASK) == 0
        quotients = values / np.float32(255)  # NumPy divides under the same MXCSR
        # down and chop round 254 of the 256 quotients below the nearest ones.
        assert (quotients.tobytes() == nearest.tobytes()) == (mode == "up")
        for count in (7, 17, 196_613):
            source = (np.arange(count) % 256).astype(np.uint8)
            for level in levels:
                assert isa_set(level) == level
                out = np.empty(count, np.float32)
                events = ctypes.c_int(-1)
                status = convert(
                    source.ctypes.data,
                    out.ctypes.data,
                    count,
                    2,
                    0,
                    1,
                    ctypes.byref(events),
                )
                assert (status, events.value) == (0, 0)
                assert out.tobytes() == quotients[source].tobytes(), (count, level)
    finally:
        restored = ctypes.c_uint()
        assert control(ctypes.byref(restored), saved.value, RC_MASK) == 0


def test_conversion_chunk_starts_off_the_lane_grid(levels):
    """Review focus 2: 720p conversions run 11 chunks of 251,345/251,346, so chunks
    start off the 8-lane grid; a vector group must never cross a chunk boundary.
    Each such case equals its manifest outputs at every level."""
    count = 2_764_800
    found = [
        (case, outputs)
        for kernel in SP4B_KERNELS
        if kernel in golden_kernels.CONVERSION_KERNELS
        for case, outputs in manifest_cases(kernel)
        if case.args["count"] == count
    ]
    assert found
    starts = [begin for begin, _ in partition(count, CONVERT_GRAIN)[1:]]
    assert any(begin % 8 for begin in starts), starts
    assert_levels_match_b3(found, levels)


def test_sp4b_sweep_reaches_scaled_values_beyond_int32():
    """Review M2/N2: the in-suite sweep (SP4B_SWEEP) holds an f32 -> u8 case without
    the clamp, (0,1,0), whose AVX2 path sets bit 2 per lane, on raw float32 bits with
    a finite |v| * 255 >= 2^31; so test_sp4b_levels_equal_scalar compares those
    lanes with the scalar level."""
    found = []
    for kernel in SP4B_KERNELS:
        if kernel not in golden_kernels.CONVERSION_KERNELS:
            continue
        for case in sweep_cases(kernel, *SP4B_SWEEP):
            args = case.args
            combo = (args["type"], args["output"], args["normalize"])
            if combo != (0, 1, 0) or case.inputs["src"]["mapping"] != "bits":
                continue
            source = make_array(case.inputs["src"])
            magnitude = source.view(np.uint32) & np.uint32(0x7FFFFFFF)
            finite = source[magnitude < 0x7F800000].astype(np.float64)
            if (np.abs(finite) * 255 >= 2.0**31).any():
                found.append(case.id)
    assert found


# The enforce clamp is np.clip(x, 0, 1) on the reference stack (NumPy 2.5.3, this
# venv's; it was NumPy 1.24.4's until U2). NaN passes with its payload, quiet or
# signalling; -0 stays -0; v < 0 gives +0 (-inf included); v > 1 gives 1 (+inf
# included); otherwise v. NumPy's maxps/minps take a denormal as a zero of its sign
# under DAZ, so a positive one gives +0 and a negative one -0; at the default MXCSR a
# positive denormal stays and a negative one gives +0. (float32 word, default, daz_ftz)
CLAMP_TABLE = (
    (0x80000000, 0x80000000, 0x80000000),
    (0x00000000, 0x00000000, 0x00000000),
    (0x80000001, 0x00000000, 0x80000000),
    (0x807FFFFF, 0x00000000, 0x80000000),
    (0x00000001, 0x00000001, 0x00000000),
    (0x007FFFFF, 0x007FFFFF, 0x00000000),
    (0x00800000, 0x00800000, 0x00800000),
    (0x80800000, 0x00000000, 0x00000000),
    (0xBF800000, 0x00000000, 0x00000000),
    (0xFF7FFFFF, 0x00000000, 0x00000000),  # -FLT_MAX
    (0xFF800000, 0x00000000, 0x00000000),
    (0x3F000000, 0x3F000000, 0x3F000000),
    (0x3F7FFFFF, 0x3F7FFFFF, 0x3F7FFFFF),
    (0x3F800000, 0x3F800000, 0x3F800000),
    (0x3F800001, 0x3F800000, 0x3F800000),
    (0x7F7FFFFF, 0x3F800000, 0x3F800000),  # FLT_MAX
    (0x7F800000, 0x3F800000, 0x3F800000),
    (0x7FC00000, 0x7FC00000, 0x7FC00000),
    (0xFFC00000, 0xFFC00000, 0xFFC00000),  # x86's default NaN (inf - inf)
    (0x7FC12345, 0x7FC12345, 0x7FC12345),
    (0xFFC54321, 0xFFC54321, 0xFFC54321),
    (0x7F812345, 0x7F812345, 0x7F812345),
    (0xFF812345, 0xFF812345, 0xFF812345),  # a negative signalling NaN
)
# float64 sources (D7: a float64 src clamps at either normalize) whose float32 casts
# include -0, denormals, NaN and values out of [0, 1].
CLAMP_DOUBLES = (
    -0.0,
    -1e-300,
    1e-300,
    -1e-40,
    1e-40,
    -5e-324,
    math.nan,
    -1.5,
    2.0,
    0.25,
)


@pytest.mark.parametrize("count", [1, 7, 8, 17, 155])
def test_clamp_is_np_clip_at_every_level(count, levels):
    """Task 3b1: in both MXCSR states and at every level, f32 -> f32 (0,0,1) gives
    CLAMP_TABLE's words, which np.clip gives too; f64 -> f32 gives
    np.clip(x.astype(np.float32), 0, 1) at either normalize; and every clamping
    integer output (whose bytes +-0 cannot change) equals the unclamped tail (type 0,
    normalize 0) on np.clip's values. The sources tile the table (23 words, coprime to
    8) or the doubles: the scalar level runs every word through the scalar helper; at
    count 1 each word and each double runs alone, so at avx2 and avx512, where a
    one-element call is all tail, the AVX2 unit's own (VEX) copy of clamp_unit runs
    every word (Task 3b2); and at 155 (19 groups of 8, then 3) every word reaches the
    8-lane form in several lanes."""
    lib = native.lib()
    words = np.array([row[0] for row in CLAMP_TABLE], np.uint32)
    with np.errstate(all="ignore"):
        doubles = np.concatenate(
            [CLAMP_DOUBLES, words.view(np.float32).astype(np.float64)]
        )
    if count == 1:
        word_runs = [[index] for index in range(words.size)]
        double_runs = [[index] for index in range(doubles.size)]
    else:
        word_runs = [np.resize(np.arange(words.size), count)]
        double_runs = [np.resize(np.arange(doubles.size), count)]
    for column, state in enumerate(MXCSR_STATES, start=1):
        table = np.array([row[column] for row in CLAMP_TABLE], np.uint32)
        with golden_kernels.mxcsr(lib, state), np.errstate(all="ignore"):
            clipped = np.clip(words.view(np.float32), 0, 1)
            clipped_doubles = np.clip(doubles.astype(np.float32), 0, 1)
        assert clipped.view(np.uint32).tobytes() == table.tobytes(), state
        # (kind, source, expected out): the table's words or the doubles.
        runs = [(0, words.view(np.float32)[run], clipped[run]) for run in word_runs]
        runs += [(1, doubles[run], clipped_doubles[run]) for run in double_runs]
        for kind, source, expected in runs:
            tails = {
                output: golden_kernels.convert(
                    lib,
                    expected,
                    {"type": 0, "output": output, "normalize": 0},
                    state,
                    "tail",
                )[0].tobytes()
                for output in (1, 2)
            }
            for level in levels:
                assert isa_set(level) == level
                where = (state, native.ISA_LEVELS[level], source[:1])
                for output in range(3):
                    for normalize in (1,) if kind == 0 else (0, 1):
                        args = {"type": kind, "output": output, "normalize": normalize}
                        out, events = golden_kernels.convert(
                            lib, source, args, state, "clamp"
                        )
                        if output == 0:
                            assert out.tobytes() == expected.tobytes(), (*where, args)
                            assert events == 0, (*where, args)
                        else:
                            assert out.tobytes() == tails[output], (*where, args)


@pytest.mark.parametrize("kernel", golden_kernels.CONVERSION_KERNELS)
def test_clamping_conversions_equal_the_oracle_at_every_level(kernel, levels):
    """Task 3b1 (P-4): every clamping conversion (golden_kernels.clamps) of the kernel's
    manifest and in-suite sweep gives the oracle's outputs at every level, +-0 and NaN
    positions included (golden_kernels.oracle_outputs: np.clip, upstream's
    normalize, under the case's MXCSR state; an integer output must equal
    the unclamped tail on np.clip's values)."""
    lib = native.lib()
    every = [case for case, _ in manifest_cases(kernel)]
    every += sweep_cases(kernel, *SP4B_SWEEP)
    found = [case for case in every if golden_kernels.clamps(case.args)]
    assert {case.mxcsr for case in found} == set(MXCSR_STATES)
    assert {case.args["output"] for case in found} >= {0, 1}
    for level in levels:
        assert isa_set(level) == level
        mismatched = []
        for case in found:
            own = run_case(lib, case)
            if golden_kernels.oracle_outputs(lib, case, own) != own:
                mismatched.append(case.id)
        assert mismatched == [], native.ISA_LEVELS[level]


# Task 3b2: freeze_normalized(clamp=True) runs the enforce's (0, 0, 1) with src == out.
IN_PLACE_COUNTS = (*range(1, 18), 196_613, 589_824, 2_764_800)
IN_PLACE_SETS = ("clean", "edges", "nan", "denorm")


def test_conversion_in_place_matches_out_of_place(levels):
    """Task 3b2: cn_pixels_convert_checked with src == out at (type 0, output 0,
    normalize 1) gives the same build's out-of-place bytes and events, at every
    level, in both MXCSR states, on the clean, edges, nan and denorm sets (signed
    values, so about half clamp to +0), at counts 1-17 (a tail only, one group and a
    tail, two groups), 196,613 (one chunk, a 5-element tail), 589,824 (the bench
    output: 3 chunks on the 8-lane grid) and 2,764,800 (11 chunks of the 262,144
    grain, which start off the 8-lane grid). Nothing past count is written."""
    lib = native.lib()
    convert = lib["cn_pixels_convert_checked"]
    convert.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    convert.argtypes += [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    convert.restype = ctypes.c_int
    args = {"type": 0, "output": 0, "normalize": 1}
    starts = [begin for begin, _ in partition(2_764_800, CONVERT_GRAIN)[1:]]
    assert any(begin % 8 for begin in starts), starts
    cases = [
        golden_kernels.conversion_case(
            f"in-place-{label}-{count}", (0, 0, 1), count, label, "default"
        )
        for count in IN_PLACE_COUNTS
        for label in IN_PLACE_SETS
    ]
    canary = 64
    for case in cases:
        source = make_array(case.inputs["src"])
        count = source.size
        for state in MXCSR_STATES:
            for level in levels:
                assert isa_set(level) == level
                where = (case.id, state, native.ISA_LEVELS[level])
                expected, expected_events = golden_kernels.convert(
                    lib, source, args, state, case.id
                )
                buffer = np.full(count + canary, 0x7FC0DEAD, np.uint32)
                buffer[:count] = source.view(np.uint32)
                events = ctypes.c_int(-1)
                with golden_kernels.mxcsr(lib, state):
                    status = convert(
                        buffer.ctypes.data,
                        buffer.ctypes.data,
                        count,
                        0,
                        0,
                        1,
                        ctypes.byref(events),
                    )
                assert status == 0, where
                assert events.value == expected_events, where
                assert buffer[:count].tobytes() == expected.tobytes(), where
                assert (buffer[count:] == 0x7FC0DEAD).all(), where


def interior_size(case, axis):
    """Rows (axis 0) or pixel columns (axis 1) of a convolution case's interior (D8):
    the outputs where every non-zero tap (at the default MXCSR) maps into the image
    without reflection."""
    args = case.args
    size, length = (args["h"], args["kh"]) if axis == 0 else (args["w"], args["kw"])
    offsets = np.nonzero(make_array(case.inputs["kernel"]))[axis] - length // 2
    padding = args["padding"]
    first = max(0, padding - int(offsets.min()))
    last = min(size + 2 * padding, padding + size - int(offsets.max()))
    return max(0, last - first)


def test_convolution_regions_match_b3_at_every_level(levels):
    """Review focus 1 (D8): each convolution_border case equals B3 at every level.
    check_border_manifest pins the groups: the 61-tap reach (R), widths and heights
    1-33 (W), the lens-like plane (L, T) and the (5,5) DAZ-row kernel (Z). This test
    adds its own preconditions: the R interiors are 0-6 pixels wide (empty up to 60,
    then 1-6); at B3_CONVOLUTION_GRAIN, where the manifest was written, the plane
    PLANE ran 4 chunks of 3367/3366 outputs starting off the 8-lane grid of its
    201-wide rows, and at CONVOLUTION_GRAIN it is one chunk, so its equality with B3
    at every level pins that the outputs do not depend on the partition
    (test_s1_ends_a_chunk_inside_a_vector_group asserts chunk starts off the lane
    grid at CONVOLUTION_GRAIN)."""
    found = manifest_cases("convolution_border")
    check_border_manifest(found)
    widths = set()
    for case, _ in found:
        form = (case.args["kh"], case.args["kw"])
        if case.id.startswith("R-"):
            axis = 1 if form == (1, 61) else 0
            width = interior_size(case, axis)
            assert width == max(0, case.args["w" if axis else "h"] - 60), case.id
            widths.add(width)
    assert widths == set(range(7))
    h, w, _ = golden_kernels.PLANE
    chunks = partition(h * w, B3_CONVOLUTION_GRAIN)
    assert {end - begin for begin, end in chunks} == {3366, 3367}
    assert all(begin % w % 8 for begin, _ in chunks[1:])
    assert partition(h * w, CONVOLUTION_GRAIN) == [(0, h * w)]
    assert_levels_match_b3(found, levels)


def separable_radii(case):
    """(rx, ry) as the entry derives them: the Gaussian's nx // 2 and ny // 2 (n is 1
    on a width or height of 1), the box's ceil(radius)."""
    args = case.args
    if case.entry == "cn_gaussian_f32":
        nx = 1 if args["w"] == 1 else args["nx"]
        ny = 1 if args["h"] == 1 else args["ny"]
        return nx // 2, ny // 2
    return math.ceil(args["rx"]), math.ceil(args["ry"])


def band_rows(ry):
    """D9's band rule before the clamp to the height: max(B0, 4 ry)."""
    return max(SEPARABLE_MIN_BAND, 4 * ry)


def tile_geometry(case):
    """D9's (B, R, W, bands, blocks) of a separable case, or None for the fallback
    (R > SEPARABLE_TILE_FLOATS / 16, or a radius above SEPARABLE_TABLE_RADIUS):
    B = min(h, max(B0, 4 ry)), R = min(h, B + 2 ry) and
    W = min(16 ceil(row_width / 16), 16 floor(SEPARABLE_TILE_FLOATS / (16 R)))."""
    h, w, c = (case.args[name] for name in ("h", "w", "c"))
    rx, ry = separable_radii(case)
    band = min(h, band_rows(ry))
    rows = min(h, band + 2 * ry)
    if rows > SEPARABLE_TILE_FLOATS // 16 or max(rx, ry) > SEPARABLE_TABLE_RADIUS:
        return None
    row_width = w * c
    most = 16 * (SEPARABLE_TILE_FLOATS // (16 * rows))
    block = min(16 * math.ceil(row_width / 16), most)
    return band, rows, block, math.ceil(h / band), math.ceil(row_width / block)


def test_separable_tile_edges_match_b3_at_every_level(levels):
    """Review focus 1 (D9): each separable_tiles case equals B3 at every level, on the
    chosen constants (this module's, read back from box_complete_ops.c).
    check_tiles_manifest pins the groups; this test adds the geometry they rely on:
    group H holds the heights B - 1, B, B + 1, 2B and 2B + 1 of each of its forms'
    bands; on TILE_PLANE the ny 11 cases' last band and last block are partial, and
    some plane case has several bands and blocks; groups H, G, W, P, S and E run in
    tiles; the second TILE_FALLBACKS shape is a fallback whose chunks at grain 1024
    start off the 8-lane grid, and the first has R = 600; the untabled cases have a
    radius above SEPARABLE_TABLE_RADIUS (both fallbacks); and some width-group row
    has row_width / 8 * 8 = 8 (mod 16) (Task 9's 16-lane groups)."""
    for name, value in (
        ("SEPARABLE_TILE_FLOATS", SEPARABLE_TILE_FLOATS),
        ("SEPARABLE_MIN_BAND", SEPARABLE_MIN_BAND),
        ("SEPARABLE_TABLE_RADIUS", SEPARABLE_TABLE_RADIUS),
        ("SEPARABLE_GRAIN", SEPARABLE_GRAIN),
    ):
        assert defined(SEPARABLE_SOURCE, name) == value, name
    found = manifest_cases("separable_tiles")
    check_tiles_manifest(found)
    groups = {}
    for case, _ in found:
        groups.setdefault(case.id.split("-")[0], []).append(case)
    heights = {}
    for case in groups["H"]:
        band = band_rows(separable_radii(case)[1])
        heights.setdefault(band, set()).add(case.args["h"])
    for band, held in heights.items():
        assert {band - 1, band, band + 1, 2 * band, 2 * band + 1} <= held, band
    for group in "HGWPSE":
        assert all(tile_geometry(case) is not None for case in groups[group]), group
    h, w, c = golden_kernels.TILE_PLANE
    several = False
    for case in groups["G"]:
        geometry = tile_geometry(case)
        assert geometry is not None, case.id
        band, _, block, bands, blocks = geometry
        if case.args.get("ny") == 11:
            assert h % band and w * c % block, case.id
        several |= bands > 1 and blocks > 1
    assert several
    second = golden_kernels.TILE_FALLBACKS[1][0]
    for case in groups["F"]:
        shape = (case.args["h"], case.args["w"], case.args["c"])
        if shape == second:
            assert tile_geometry(case) is None, case.id
        else:
            ry = separable_radii(case)[1]
            assert min(shape[0], band_rows(ry) + 2 * ry) == 600, case.id
    h, w, c = second
    starts = [begin for begin, _ in partition(h * w * c, SEPARABLE_GRAIN)[1:]]
    assert any(begin % (w * c) % 8 for begin in starts)
    for case in groups["N"]:
        assert max(separable_radii(case)) > SEPARABLE_TABLE_RADIUS, case.id
        assert tile_geometry(case) is None, case.id
    rows = {case.args["w"] * case.args["c"] for case in groups["W"]}
    assert any(row // 8 * 8 % 16 == 8 for row in rows)
    assert_levels_match_b3(found, levels)


# The separable R fallback (D9: R > SEPARABLE_TILE_FLOATS / 16) runs two passes in
# chunks of SEPARABLE_GRAIN whose row segments start off the 8-lane grid, so their
# fused heads and tails run the scalar spans beside the vector groups. The frozen
# separable_tiles manifest holds these shapes clean only (Task 3's review): here
# TILE_FALLBACKS' second shape and 70 x 3 rows (26 groups: an 8-group block, a
# 4-group tail and single groups) take NaN, edges and denormal inputs.
R_FALLBACK_SHAPES = (golden_kernels.TILE_FALLBACKS[1][0], (2100, 70, 3))


@pytest.mark.parametrize(
    "shape", R_FALLBACK_SHAPES, ids=lambda s: "x".join(map(str, s))
)
def test_separable_r_fallback_levels_equal_scalar(shape, levels):
    """TILE_FALLBACKS' second Gaussian form at borders 1/2/4 and its box form on each
    R_FALLBACK_SHAPES shape, lanes 8, over SOURCES (nan, edges and a src in the
    denormal range among them) and both MXCSR states: every level equals scalar.
    Preconditions: every case runs the fallback, and some chunk starts off the
    8-lane grid of its rows."""
    _, gaussian, box = golden_kernels.TILE_FALLBACKS[1]
    found = []
    for border in (1, 2, 4):
        found += cases("cn_gaussian_f32", gaussian, border, 8, [shape])
    found += cases("cn_box_separable", box, None, 8, [shape])
    assert all(tile_geometry(case) is None for case in found)
    h, w, c = shape
    starts = [begin for begin, _ in partition(h * w * c, SEPARABLE_GRAIN)[1:]]
    assert any(begin % (w * c) % 8 for begin in starts)
    assert_levels_equal_scalar(found, levels)


def defined(source, name):
    """The value of `#define name <integer>` in a native source file."""
    text = source.read_text(encoding="utf-8")
    values = re.findall(rf"^#define {name} (\d+)\b", text, re.MULTILINE)
    assert len(values) == 1, name
    return int(values[0])


def border_runs(row_width, inner, outer, lanes=8, block=None):
    """The ISA row functions' border runs (border_runs_shared.h) on a whole row
    segment [0, row_width) at `lanes` lanes: the vector outputs end at the fused
    boundary end = row_width // 8 * 8; the AVX2 groups (8 lanes) stop at end, the
    AVX-512 groups (16) at end rounded up, the last one masked to its lanes below end.
    The runs [0, head) and [tail, stop) are widened into the interior [inner, outer)
    (inner rounded up and outer down to the lane grid, clamped as the row functions
    clamp them) to `block` elements (eight groups unless given) where the segment
    holds that many. Returns (end, stop, head, tail)."""
    end = row_width // 8 * 8
    stop = end if lanes == 8 else -(-end // lanes) * lanes
    block = 8 * lanes if block is None else block
    inner = min(-(-inner // lanes) * lanes, stop)
    outer = min(max(outer // lanes * lanes, inner), stop)
    head = max(inner, min(block, outer)) if inner > 0 else inner
    tail = outer
    if stop > outer:
        tail = head if stop - head < block else min(stop - block, outer)
    return end, stop, head, tail


# Task 3c: a horizontal border run's taps read one reflected copy of the row's
# elements they reach (run + 2 rx c floats), and a convolution run's taps one copy
# per tap row (run + (high - low) c floats each); a run whose copies exceed the
# unit's bound takes the per-group path. Each pair of shapes straddles its bound, at
# both units' geometries (an AVX-512 run's copy ends at its last unmasked lane).
EDGE_SHAPES = ((9, 20, 4), (9, 30, 4))  # left runs of 80 and 120 floats at rx 500
EDGE_FORMS = {"cn_gaussian_f32": (1001, 3), "cn_box_separable": (500.0, 1.0)}
# The Gaussian's horizontal sigma there: at the manifests' 2.5 only 71 of its 1001 taps
# are nonzero, so reflected values more than 35 columns from the edge weigh nothing; at
# 250 the outermost tap is exp(-2), so every reflected value in the copy is checked.
EDGE_SIGMA = 250.0
PATCH_PADDINGS = (20, 30)  # (61, 1) on (8, 12, 3): runs of 64 and 96 floats, 61 rows


def avx512_run_block():
    """convolution_avx512.c's CONVOLUTION_RUN_BLOCK: it widens a border run to that
    many elements only where the widened run's copies fit the bound."""
    return defined(CONVOLUTION_AVX512, "CONVOLUTION_RUN_BLOCK")


def test_separable_border_copy_bound_levels_equal_scalar(levels):
    """EDGE_FORMS (rx 500; the Gaussian's sigma_x EDGE_SIGMA) at borders 1/2/4 (the
    Gaussian) on EDGE_SHAPES, lanes 8, over SOURCES and both MXCSR states: every level
    equals scalar. Preconditions: at 8 and 16 lanes, the first shape's left run copy
    fits SEPARABLE_EDGE_FLOATS and the second's does not (one tile block holds each
    whole row; at 16 lanes its last group is masked); the Gaussian's outermost tap, so
    every tap, is a normal float32."""
    bound = defined(SEPARABLE_ISA, "SEPARABLE_EDGE_FLOATS")
    rx, copies = 500, {}
    for _, w, c in EDGE_SHAPES:
        for lanes in (8, 16):
            end, _, head, _ = border_runs(w * c, rx * c, 0, lanes)
            copies.setdefault(lanes, []).append(min(head, end) + 2 * rx * c)
    for fits, exceeds in copies.values():
        assert fits <= bound < exceeds, copies
    outermost = np.exp(np.float32(-(rx**2) / (2 * EDGE_SIGMA**2)), dtype=np.float32)
    assert outermost > np.finfo(np.float32).tiny
    found = []
    for entry, form in EDGE_FORMS.items():
        for border in (1, 2, 4) if entry == "cn_gaussian_f32" else (None,):
            found += cases(entry, form, border, 8, EDGE_SHAPES)
    found = [
        dataclasses.replace(case, args={**case.args, "sigma_x": EDGE_SIGMA})
        if case.entry == "cn_gaussian_f32"
        else case
        for case in found
    ]
    assert all(separable_radii(case)[0] == rx for case in found)
    geometries = [tile_geometry(case) for case in found]
    assert all(geometry is not None and geometry[4] == 1 for geometry in geometries)
    assert_levels_equal_scalar(found, levels)


def test_convolution_border_copy_bound_levels_equal_scalar(levels):
    """(61, 1) at PATCH_PADDINGS on (8, 12, 3), fused 8, over SOURCES and both MXCSR
    states: every level equals scalar. Preconditions: every chunk starts a row, so
    each run is a whole row's; each tap is its own tap row (one column), and a zero
    tap, or a denormal one under DAZ, is skipped; the AVX-512 runs are not widened
    (eight groups' copies exceed the bound); so at 8 and 16 lanes the first padding's
    left and right runs fit CONVOLUTION_PATCH_FLOATS with every tap kept, and the
    second's exceed it with only the normal and non-finite taps."""
    bound = defined(CONVOLUTION_ISA, "CONVOLUTION_PATCH_FLOATS")
    tiny = np.finfo(np.float32).tiny
    h, w, c = shape = (8, 12, 3)
    found = []
    for padding in PATCH_PADDINGS:
        row = (w + 2 * padding) * c
        chunks = partition((h + 2 * padding) * row, CONVOLUTION_GRAIN)
        assert all(begin % row == 0 for begin, _ in chunks), padding
        widths = []
        for lanes, block in ((8, 64), (16, 16)):
            end, stop, head, tail = border_runs(
                row, padding * c, (padding + w) * c, lanes, block
            )
            widths.append((min(head, end), min(stop, end) - tail))
        for case in cases("cn_convolution_spatial", (61, 1), padding, 8, [shape]):
            kernel = make_array(case.inputs["kernel"])
            kept = np.count_nonzero(kernel != 0)
            normal = np.count_nonzero(~(np.abs(kernel) < tiny))
            assert normal * avx512_run_block() > bound, case.id
            for runs in widths:
                if padding == PATCH_PADDINGS[0]:
                    assert kept * max(runs) <= bound, (case.id, runs)
                else:
                    assert normal * min(runs) > bound, (case.id, runs)
            found.append(case)
    assert_levels_equal_scalar(found, levels)


# The per-group path's direct loads: (11, 11) at WIDE_PADDING on (4, 12, 3) has two
# border runs of 352 and 344 floats per tap row (the whole row), whose 11 rows of
# copies exceed the bound at both units' geometries; next to the image, some of their
# groups have taps whose lanes all map into it without reflection.
WIDE_PADDING = 110


def test_convolution_per_group_direct_loads_levels_equal_scalar(levels):
    """(11, 11) at WIDE_PADDING on (4, 12, 3), fused 8, over SOURCES and both MXCSR
    states: every level equals scalar. Preconditions, at 8 and 16 lanes: every tap row
    holds a normal or non-finite tap (so none is skipped), and both runs' copies
    exceed CONVOLUTION_PATCH_FLOATS, so the per-group path runs; some of their groups
    have a tap whose lanes all map into the image (the direct load), and the others
    gather."""
    bound = defined(CONVOLUTION_ISA, "CONVOLUTION_PATCH_FLOATS")
    tiny = np.finfo(np.float32).tiny
    _, w, c = shape = (4, 12, 3)
    padding, reach = WIDE_PADDING, 5
    row = (w + 2 * padding) * c
    # convolution_avx512.c's widening rule with all 11 tap rows.
    run_block = avx512_run_block()
    wide = run_block if 11 * (run_block + 2 * reach * c) <= bound else 16
    found = cases("cn_convolution_spatial", (11, 11), padding, 8, [shape])
    for case in found:
        kept = ~(np.abs(make_array(case.inputs["kernel"])) < tiny)
        assert kept.any(axis=1).all(), case.id
        for lanes, block in ((8, 64), (16, wide)):
            end, stop, head, tail = border_runs(
                row, (padding + reach) * c, (padding + w - reach) * c, lanes, block
            )
            runs = ((0, min(head, end)), (tail, min(stop, end)))
            assert all(
                11 * (last - first + 2 * reach * c) > bound for first, last in runs
            ), (case.id, lanes, runs)
            maps = [
                all(0 <= (g + lane) // c + x - padding < w for lane in range(lanes))
                for first, last in runs
                for g in range(first, last, lanes)
                for x in range(-reach, reach + 1)
                if kept[:, x + reach].any()
            ]
            assert any(maps) and not all(maps), (case.id, lanes)
    assert_levels_equal_scalar(found, levels)


def off_grid_starts(total, grain, row):
    """The chunk starts of partition(total, grain) that fall off the 16-lane grid of
    their row (a chunk whose first group is masked below its start)."""
    return [begin for begin, _ in partition(total, grain)[1:] if begin % row % 16]


# Straddling cases with chunk starts off the 16-lane grid. No B3 manifest holds one:
# every straddling manifest case is a single chunk. Rows here have their fused boundary
# at 8 (mod 16), and some chunk starts mid-row off the grid. Convolution: (101, 67, 3)
# at paddings 0 and 1 (201- and 207-wide rows; two chunks at CONVOLUTION_GRAIN, the
# second starting at column 101 or 104). Separable: the R fallback (D9) on 27-wide rows
# at SEPARABLE_GRAIN, with Gaussian horizontal radii 1 and 4 and the box's 2.
STRADDLE_CONVOLUTION = ((101, 67, 3), ((3, 3), (1, 61), (11, 11)), (0, 1))
STRADDLE_SEPARABLE = ((2100, 9, 3), (((3, 685), 4), ((9, 685), 1)), (1.5, 342.5))


def test_avx512_groups_straddling_the_fused_boundary(levels):
    """Review focus 2 (D15). The straddling manifest cases (straddling_cases) equal B3
    at every level: a row's last 16-lane group holds 8 fused outputs, its other lanes
    are masked off, and the outputs from the boundary on are unfused. Preconditions:
    both kernels have such cases, some on the nan and mixed sets and some under
    DAZ|FTZ. Since all of them are single chunks, the STRADDLE_CONVOLUTION and
    STRADDLE_SEPARABLE cases (over SOURCES, both MXCSR states, fused 8 / lanes 8)
    equal scalar at every level too, with these preconditions asserted: fused boundary
    at 8 (mod 16); some chunk starting off the 16-lane grid of its row, so the head
    mask (lanes below the start) runs; and every separable case on the fallback."""
    straddling = straddling_cases()
    assert set(straddling) == {"convolution_border", "separable_tiles"}
    for kernel, found in straddling.items():
        sets = {case.id.split("-")[-3] for case, _ in found}
        assert {"nan", "mixed"} <= sets, (kernel, sets)
        assert {case.mxcsr for case, _ in found} == set(MXCSR_STATES), kernel
    assert_levels_match_b3(
        [pair for found in straddling.values() for pair in found], levels
    )
    generated = []
    (h, w, c), forms, paddings = STRADDLE_CONVOLUTION
    for padding in paddings:
        row = (w + 2 * padding) * c
        assert row // 8 * 8 % 16 == 8, padding
        total = (h + 2 * padding) * row
        assert off_grid_starts(total, CONVOLUTION_GRAIN, row), padding
        for form in forms:
            generated += cases("cn_convolution_spatial", form, padding, 8, [(h, w, c)])
    (h, w, c), gaussians, box = STRADDLE_SEPARABLE
    row = w * c
    assert row // 8 * 8 % 16 == 8
    assert off_grid_starts(h * row, SEPARABLE_GRAIN, row)
    separable = cases("cn_box_separable", box, None, 8, [(h, w, c)])
    for form, border in gaussians:
        separable += cases("cn_gaussian_f32", form, border, 8, [(h, w, c)])
    assert all(tile_geometry(case) is None for case in separable)
    assert_levels_equal_scalar(generated + separable, levels)


def test_lens_compose_keeps_signed_zero_and_infinities(levels):
    """The lens manifest's cases with (a, b) = (0, -0) or (inf, 1) on the inf and nan
    sets, both accumulate values and both MXCSR states, equal B3 at every level.
    Their outputs hold -0 and infinities: at the default MXCSR, B3's outputs are
    NumPy's float32 evaluation of compose_range's expression in its order
    (compose_reference), and those hold both."""
    found = [
        (case, outputs)
        for case, outputs in manifest_cases("lens")
        if case.id.split("-")[2] in ("zeros", "infinite")
        and case.id.split("-")[-3] in ("inf", "nan")
    ]
    wide = {
        (
            case.id.split("-")[2],
            case.id.split("-")[-3],
            case.args["accumulate"],
            case.mxcsr,
        )
        for case, _ in found
        if case.args["count"] == golden_kernels.WIDE_COUNT
    }
    assert wide == {
        (pair, name, accumulate, mxcsr)
        for pair in ("zeros", "infinite")
        for name in ("inf", "nan")
        for accumulate in (0, 1)
        for mxcsr in MXCSR_STATES
    }
    negative_zero = infinite = False
    for case, outputs in found:
        if case.mxcsr != "default":
            continue
        value = compose_reference(case)
        assert digest(value, case.payloads) == outputs["out"], case.id
        negative_zero |= bool((value.view(np.uint32) == 0x80000000).any())
        infinite |= bool(np.isinf(value).any())
    assert negative_zero and infinite
    assert_levels_match_b3(found, levels)


# B3's cn_lens_power with finish clamps its input as maxss(+0, x) (B3 0x180041141),
# not as its C text (x <= 0 -> 0): -0 stays -0, and under DAZ a denormal comes back as
# the zero of its sign. Only an exponent for which powf(-0, e) and powf(+0, e) differ
# after the output clamp shows it: at e = -1, -inf -> +0 against +inf -> 1. Outputs
# recorded from the B3 DLL (task3b-verdict.md), per MXCSR state.
CLAMP_WORDS = (
    0x80000000,
    0x00000000,
    0x80000001,
    0x00000001,
    0x807FFFFF,
    0x7FC12345,
    0xBF800000,
    0x3F000000,
    0x40000000,
)
CLAMP_OUTPUTS = {
    "default": (
        0x00000000,
        0x3F800000,
        0x3F800000,
        0x3F800000,
        0x3F800000,
        0x7FC12345,
        0x3F800000,
        0x3F800000,
        0x3F000000,
    ),
    "daz_ftz": (
        0x00000000,
        0x3F800000,
        0x00000000,
        0x3F800000,
        0x00000000,
        0x7FC12345,
        0x3F800000,
        0x3F800000,
        0x3F000000,
    ),
}


def test_lens_power_keeps_b3s_input_clamp(levels):
    """Task 3b: cn_lens_power's power_one pins B3's maxss(+0, x) input clamp; at every
    level and in both MXCSR states the finishing power gives B3's outputs at
    exponent -1."""
    dll = native.lib()
    power = dll["cn_lens_power"]
    power.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    power.argtypes += [ctypes.c_float, ctypes.c_int]
    power.restype = ctypes.c_int
    source = np.array(CLAMP_WORDS, np.uint32).view(np.float32)
    count = source.size
    for level in levels:
        assert isa_set(level) == level
        for state, expected in CLAMP_OUTPUTS.items():
            out = np.full(count, 0xFFFFFFFF, np.uint32)
            with golden_kernels.mxcsr(dll, state):
                assert power(source.ctypes.data, out.ctypes.data, count, -1, 1) == 0
            assert out.tolist() == list(expected), (level, state)


def generated_nan_case(count, accumulate):
    """A lens case where an input payload meets a NaN the operation generates (Task 2
    review, scenarios A and D), (a, b) = (0.5, -1.25). Index % 5 == 0 holds A: f1 =
    0x7fc12345, f2 = +inf, f3 = f4 = 1, so real = p meets 0 * inf - 0. Index % 5 == 2
    holds D: f1 = f4 = +inf, f2 = p, f3 = 1, so inf - inf meets 0 * p - 0. Every other
    element is a signed value, out included."""
    one, inf, payload = "0x3f800000", "0x7f800000", "0x7fc12345"
    values = {
        "f1": (payload, inf),
        "f2": (inf, payload),
        "f3": (one, one),
        "f4": (one, inf),
    }
    inputs = {}
    for position, name in enumerate(golden_kernels.LENS_INPUTS):
        a_value, d_value = values.get(name, (None, None))
        specials = [
            [i, a_value if i % 5 == 0 else d_value]
            for i in range(count)
            if name in values and i % 5 in (0, 2)
        ]
        inputs[name] = {
            "shape": [count],
            "seed": SP4B_SWEEP[1] + position,
            "mapping": "signed",
            "specials": specials,
        }
    a, b = golden_kernels.LENS_PAIRS["scaled"]
    args = {"count": count, "a": a, "b": b, "accumulate": accumulate}
    case_id = f"generated-nan-{count}-{accumulate}"
    return golden_kernels.Case(
        case_id, golden_kernels.LENS_ENTRY, args, inputs, "default", "mixed"
    )


def compose_tail(count):
    """The elements cn_lens_compose_avx2 runs through compose_one instead of an 8-lane
    group: per chunk of partition(count, COMPOSE_GRAIN), the groups start at the
    chunk's begin and the (end - begin) % 8 elements left at its end are the tail."""
    tail = np.zeros(count, bool)
    for begin, end in partition(count, COMPOSE_GRAIN):
        tail[end - (end - begin) % 8 : end] = True
    return tail


@pytest.mark.parametrize(("count", "a_in_tail"), [(21, True), (196_613, False)])
@pytest.mark.parametrize("accumulate", [0, 1])
def test_lens_compose_generated_nan_meets_payload(count, a_in_tail, accumulate, levels):
    """Stand-in ruling (spec 4.4, cfbd52f2): a NaN the operation generates (0 * inf,
    inf - inf) counts as a payload, so where one meets an input payload only the NaN
    positions and every non-NaN bit are pinned (the "mixed" digest), not the NaN bits.
    Each level equals the scalar level and B3 under that rule. B3's positions and
    non-NaN bits are compose_reference's: it equals B3 on every single-payload lens
    case (check_lens_manifest) and on these inputs (task2-verdict.md). At the AVX2
    levels the lanes split into 8-lane groups and a scalar tail per chunk
    (compose_tail), and the test asserts the split it relies on. Count 21 is one chunk
    of two groups (elements 0-15) and a 5-element tail (16-20) that holds a scenario D
    lane (17) and a scenario A lane (20). Count 196,613 is 4 chunks with tails of 2, 1,
    1 and 1 elements, which hold D lanes but no A lane. Both counts keep meet lanes
    in groups. Generated NaNs alone (the inf set) stay bit for bit:
    test_current_dll_matches_the_b3_manifest[lens] and
    test_lens_compose_keeps_signed_zero_and_infinities."""
    case = generated_nan_case(count, accumulate)
    reference = compose_reference(case)
    index = np.arange(count) % 5
    meet = np.isin(index, (0, 2))
    assert np.isnan(reference[meet]).all() and np.isfinite(reference[~meet]).all()
    tail = compose_tail(count)
    assert (meet & ~tail).any() and ((index == 2) & tail).any()
    assert ((index == 0) & tail).any() == a_in_tail
    expected = {"out": digest(reference, "mixed")}
    for level in levels:
        assert isa_set(level) == level
        assert run_case(native.lib(), case) == expected, native.ISA_LEVELS[level]
