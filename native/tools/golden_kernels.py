"""Golden manifests of the SP4 kernels, run through a DLL's C entries (SP4 spec 4.4).

Usage:
  python native/tools/golden_kernels.py write --dll DLL --kernel K --out FILE
  python native/tools/golden_kernels.py sweep --dll DLL --kernel K --count N --seed S
      [--isa scalar|avx2|avx512] [--oracle] --out FILE
  python native/tools/golden_kernels.py rebase --dll DLL --kernel K --manifest FILE
  python native/tools/golden_kernels.py diff FILE1 FILE2

write runs the kernel's fixed matrix (matrix()); the tracked manifests under
native/tests/golden/ are written only this way, from the B3 DLL. sweep runs
`count` cases drawn from the matrix's axes with random shapes (sweep_cases());
--isa first sets the DLL's level with cn_isa_set and exits 2 unless the DLL
returns that level; --oracle makes the oracle the reference of the clamping
conversions (oracle_outputs()). rebase re-bases a conversion manifest written from
the DLL on the oracle, only the cases whose reference moved (rebase()). diff
compares two manifests case by case and exits 1, listing the ids, when a case
differs or is in one file only.

No arrays are stored. An input is a recipe: PCG64 raw bits mapped to its dtype
(float32 unless the recipe names another), then float32 special values written as
bits at listed flat indices (make_array()); a mixed set's values are rotated by
the input's position, so the payloads of different inputs meet. An output is the
SHA-256 of its bytes;
a mixed-payload case first maps every NaN to 0x7fc00000 (a float64 output's to
0x7ff8000000000000), since x86 keeps the payload of whichever operand the compiler
placed first (digest()). An entry whose
outputs are not one float32 array runs through ADAPTERS[entry], which hashes an
int output as its little-endian int32 (digest_int()). DAZ|FTZ cases set the
calling thread's MXCSR with torch.set_flush_denormal, the tool's one use of
torch, imported there; Pillow is never imported.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import dataclasses
import hashlib
import itertools
import json
import math
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

import numpy as np

SCHEMA = "chaiNNer-C/golden/v1"
CONVERSION_KERNELS = ("conversion", "conversion_small")
KERNELS = (
    "separable",
    "convolution",
    *CONVERSION_KERNELS,
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
ISA_LEVELS = ("scalar", "avx2", "avx512")
Mxcsr = Literal["default", "daz_ftz"]


Payloads = Literal["single", "mixed"]
MXCSR_STATES: tuple[Mxcsr, ...] = ("default", "daz_ftz")
DAZ_FTZ = 0x8040  # MXCSR flush-to-zero (bit 15) and denormals-are-zero (bit 6)
DEFAULT_NAN = 0x7FC00000

_EDGES = (
    "0x00000000",
    "0x80000000",
    "0x00000001",
    "0x807fffff",
    "0x3f7fffff",
    "0x3f800001",
    "0x7f7fffff",
    "0xff7fffff",
)
_NO_FLT_MAX = _EDGES[:6]


class Library(Protocol):
    """What the tool uses of a ctypes.CDLL: a fresh function object per name."""

    def __getitem__(self, name: str, /) -> Any: ...


class SpecialSet(TypedDict):
    payloads: Payloads
    values: list[str]


# A case's set goes into src, and into the convolution kernel as case() says.
# The single-payload sets list their defining non-finite values first, so the
# smallest arrays (one or six special positions) hold them too (controller
# ruling, 2026-10-03).
SPECIALS: dict[str, SpecialSet] = {
    "clean": {"payloads": "single", "values": []},
    "edges": {"payloads": "single", "values": list(_EDGES)},
    "nan": {"payloads": "single", "values": ["0x7fc12345", *_NO_FLT_MAX]},
    "snan": {"payloads": "single", "values": ["0x7f812345", *_NO_FLT_MAX]},
    # Only the default NaN can arise (inf - inf, inf * 0).
    "inf": {"payloads": "single", "values": ["0x7f800000", "0xff800000", *_EDGES]},
    "mixed": {
        "payloads": "mixed",
        "values": [
            "0x7fc12345",
            "0xffc54321",
            "0x7f812345",
            "0x7f800000",
            "0xff800000",
        ],
    },
    # SP4b D3 (first used by lens_power, Task 3b): denormals only, no +-0, NaN or inf.
    "denorm": {
        "payloads": "single",
        "values": ["0x00000001", "0x007fffff", "0x80000001", "0x807fffff"],
    },
}
# SP4a's sets. The matrices and sweeps of the kernels written before denorm draw from
# these, so their manifests and sweeps stay as written.
SP4A_SETS = ("clean", "edges", "nan", "snan", "inf", "mixed")


@dataclass(frozen=True)
class Entry:
    """A C entry: its parameters in call order and the axes of its matrix."""

    params: tuple[str, ...]
    output: str
    form: tuple[str, str]
    forms: tuple[tuple[float, float], ...]
    mirror: str
    mirrors: tuple[int, int]  # (narrow, wide)
    place: str | None  # border or padding
    places: tuple[int, ...]
    place_b: int  # the border or padding of groups B and C
    fixed: dict[str, float]
    label: str


ENTRIES = {
    "cn_gaussian_f32": Entry(
        params=(
            "src",
            "dst",
            "h",
            "w",
            "c",
            "nx",
            "ny",
            "sigma_x",
            "sigma_y",
            "border",
            "lanes",
        ),
        output="dst",
        form=("nx", "ny"),
        # horizontal radius 0, <= 2 and general; vertical radius 1 and general
        forms=((1, 3), (3, 11), (5, 3), (21, 11)),
        mirror="lanes",
        mirrors=(4, 8),
        place="border",
        places=(1, 2, 4),
        place_b=4,
        fixed={"sigma_x": 2.5, "sigma_y": 2.5},
        label="n",
    ),
    "cn_box_separable": Entry(
        params=("src", "dst", "h", "w", "c", "rx", "ry", "lanes"),
        output="dst",
        form=("rx", "ry"),
        forms=((0.0, 1.0), (1.5, 2.5), (2.0, 0.5), (3.25, 4.0)),
        mirror="lanes",
        mirrors=(4, 8),
        place=None,
        places=(),
        place_b=0,
        fixed={},
        label="r",
    ),
    "cn_convolution_spatial": Entry(
        params=("src", "out", "h", "w", "c", "kernel", "kh", "kw", "padding", "fused"),
        output="out",
        form=("kh", "kw"),
        forms=((1, 1), (3, 3), (5, 5), (1, 61), (61, 1), (11, 11)),
        mirror="fused",
        mirrors=(0, 8),
        place="padding",
        places=(0, 1, 2),
        place_b=1,
        fixed={},
        label="k",
    ),
}
CONVERT_ENTRY = "cn_pixels_convert_checked"
LENS_ENTRY = "cn_lens_compose"
POWER_ENTRY = "cn_lens_power"
MORPH_ENTRY = "cn_morphology_complete"
FILTER_ENTRY = "cn_filter_morphology"
EXCEPTIONAL_ENTRY = "cn_image_exceptional"
RESAMPLE_ENTRY = "cn_resample_filtered"
PALETTE_ENTRY = "cn_palette_median_cut"
DITHER_APPLY = "cn_neighborhood_palette_apply"
DITHER_ENTRY = "cn_neighborhood_dither"
RIEMERSMA_ENTRY = "cn_neighborhood_palette_riemersma"
COMPOSITE_ENTRY = "cn_composite_canvas"
BLEND_IMAGES = "cn_blend_images"
BLEND_MODE = "cn_blend_mode"
TILE_ENTRY = "cn_spectral_tile"
STORE_ENTRY = "cn_spectral_store"
MULTIPLY_ENTRY = "cn_spectral_multiply"
NORMAL_ENTRY = "cn_normal_output"
CAPTION_ENTRY = "cn_caption_compose"
KERNEL_ENTRIES = {
    "separable": ("cn_gaussian_f32", "cn_box_separable"),
    "convolution": ("cn_convolution_spatial",),
    "conversion": (CONVERT_ENTRY,),
    "conversion_small": (CONVERT_ENTRY,),
    "convolution_border": ("cn_convolution_spatial",),
    "lens": (LENS_ENTRY,),
    "separable_tiles": ("cn_gaussian_f32", "cn_box_separable"),
    "lens_power": (POWER_ENTRY,),
    "morphology": (MORPH_ENTRY, FILTER_ENTRY, EXCEPTIONAL_ENTRY),
    "resample": (RESAMPLE_ENTRY,),
    "palette": (PALETTE_ENTRY,),
    "dither": (DITHER_APPLY, DITHER_ENTRY, RIEMERSMA_ENTRY),
    "composite": (COMPOSITE_ENTRY,),
    "blend": (BLEND_IMAGES, BLEND_MODE),
    "tail": (TILE_ENTRY, STORE_ENTRY, MULTIPLY_ENTRY, NORMAL_ENTRY, CAPTION_ENTRY),
}
# convolution_border (SP4b D8): the hoisted borders' shapes. The 61-tap forms at
# widths (heights) 58-66 have interiors 0, 0, 0, 1, ..., 6 pixels wide; the plane
# has 13,467 outputs at padding 0: 4 chunks of 3367/3366 at B3's grain 4096 (the
# manifest's), whose starts are off the 8-lane grid of its 201-wide rows, and one
# chunk at the convolution grain 16384 (SP4b Task 3e).
BORDER_FORMS = ((1, 61), (61, 1), (3, 3), (11, 11), (5, 5))
BORDER_SETS = ("clean", "nan", "inf", "edges", "mixed")
REACH_SIZES = tuple(range(58, 67))
PLANE = (67, 201, 1)
# Group Z's (5,5) kernel row -2 (indices 0-4): +-0 and one denormal, so only
# under DAZ is the row skipped whole and the interior a row taller.
DENORMAL_ROW = (
    (0, "0x00000000"),
    (1, "0x80000000"),
    (2, "0x00000001"),
    (3, "0x80000000"),
    (4, "0x00000000"),
)
# lens (cn_lens_compose): the inputs in call order (out is in-out), the (a, b)
# pairs as float32 bits, and the sets.
LENS_INPUTS = ("f1", "f2", "f3", "f4", "out")
LENS_PAIRS = {
    "scaled": ("0x3f000000", "0xbfa00000"),  # (0.5, -1.25)
    "zeros": ("0x00000000", "0x80000000"),  # (0, -0)
    "infinite": ("0x7f800000", "0x3f800000"),  # (inf, 1)
}
LENS_SETS = ("clean", "edges", "nan", "inf", "mixed")
# lens_power (cn_lens_power with finish 1): Lens Blur's finishing power of a (channels,
# pixels) CHW plane, digested in the order of the node's transpose(1, 2, 0) (HWC). The
# exponents as float32 bits (1/5, 5 and 1), the sets, and the bench's 512 x 384 pixels:
# at grain 65,536 elements that is 3 chunks at c 1, 9 at c 3 and 12 at c 4.
POWER_EXPONENTS = {"fifth": "0x3e4ccccd", "five": "0x40a00000", "one": "0x3f800000"}
POWER_SETS = ("clean", "edges", "nan", "denorm")
POWER_CHANNELS = (1, 3, 4)
POWER_PIXELS = 196_608
# morphology (SP4b Task 4): each entry's arguments after (src, out, h, w, c), in call
# order, and their manifest values. cn_morphology_complete's shape is 0 (rectangle),
# 1 (cross) or 2 (ellipse), lanes OpenCV's vector width mirror; cn_filter_morphology's
# cross is 0 (rectangle) or 1. The node forms: Dilate's ellipse through
# cn_morphology_complete (lanes 8: AVX2 OpenCV) and Erode's cross through
# cn_filter_morphology (morphology(), native_filters.py, with a non-exceptional src).
MORPH_FORMS = {
    MORPH_ENTRY: {
        "radius": (1, 2, 3),
        "iterations": (1, 2, 3),
        "shape": (0, 1, 2),
        "maximum": (0, 1),
        "lanes": (0, 4, 8, 16),
    },
    FILTER_ENTRY: {
        "radius": (1, 2, 3),
        "iterations": (1, 2, 3),
        "cross": (0, 1),
        "maximum": (0, 1),
    },
}
DILATE = {"shape": 2, "maximum": 1, "lanes": 8}
ERODE = {"cross": 1, "maximum": 0}
MORPH_SETS = ("clean", "denorm", "edges", "nan", "inf")
MORPH_SHAPE = (37, 41, 3)
MORPH_BENCH = (384, 512, 3)  # the bench's 512x384 images
# 125,967 elements: the ellipse combine ran 8 chunks (7 of 15,746, 1 of 15,745) at
# B3's grain 16,384 (the manifest's) and runs 2 (62,984 and 62,983) at the combine
# grain 65,536 (SP4b Task 4b); the cross merge (grain 65,536) runs 2. No chunk
# boundary falls on a row start of the 633-wide rows.
MORPH_CHUNKS = (199, 211, 3)
# cn_image_exceptional: one word at index 0 or count - 1 of a unit src (no other
# exceptional value), counts 1-33 (tails and 1-4 groups of 8) and the bench's 589,824.
# The words: -0, negative denormals (exceptional under DAZ only), +-inf, NaNs of both
# signs and kinds, and words just outside each class (+0, a positive denormal, FLT_MAX,
# the negative normal next to the denormals), which are never exceptional.
EXCEPTIONAL_WORDS = (
    "0x80000000",
    "0x80000001",
    "0x00000000",
    "0x7f800000",
    "0x807fffff",
    "0x7fc12345",
    "0x00000001",
    "0xff800000",
    "0x7f812345",
    "0x7f7fffff",
    "0xffc54321",
    "0x80800000",
)
EXCEPTIONAL_COUNTS = (*range(1, 34), 589_824)
EXCEPTIONAL_POSITIONS = ("first", "last")
# resample (SP4b Task 4; Task 7 reuses it): cn_resample_filtered's filters 1-11
# (native_resample's ResizeFilter values), the (h, w) -> (th, tw) geometries and sets.
RESAMPLE_FILTERS = tuple(range(1, 12))
RESAMPLE_GRID = ((29, 31), (43, 37))
# An RGBA upscale (the bridge's vector_clip form) and an RGB downscale.
RESAMPLE_EDGES = (((11, 13, 4), (71, 67)), ((37, 50, 3), (19, 25)))
RESAMPLE_SETS = ("nan", "edges", "inf", "denorm")
# The bench's Hermite resizes (bench-data/resize-real-256.json; video-resize-h264).
RESAMPLE_BENCH = (((375, 500, 3), (192, 256)), ((720, 1280, 3), (360, 640)))
# palette (SP4b Task 5): cn_palette_median_cut on a (pixels, channels) src. Any NaN in
# the src makes its first split's channel NaN and returns status 6 (empty child), as
# the original median cut fails at np.min; so the NaN sets' cases are status 6.
PALETTE_PIXELS = (*range(1, 18), 64, 1000)
PALETTE_BENCH = 196_608  # the bench's 512 x 384 pixels
PALETTE_CHANNELS = (1, 3, 4)
PALETTE_CAPACITIES = (2, 16, 256)
PALETTE_SETS = ("clean", "edges", "denorm", "nan", "snan", "mixed")
# Group Z's sources (set, mapping): zero-class ties. -0 and +0, and under DAZ the
# denormals too, compare equal, so the extrema keep the first one seen.
PALETTE_TIES = (("edges", "unit"), ("denorm", "unit"), ("clean", "tiny"))
PALETTE_TIE_PIXELS = (64, 1000)
# Group N: the set's NaN word alone, after the first pixel (where the index rule puts
# every other case's NaN): at pixel n // 2, channel c // 2 ("middle"), or at the last
# element ("last"; after the last 24-float block except at 64 x 3). The channel's range
# is then NaN only through describe's NaN rule. Capacity 2: the root's one split must
# take that channel (status 6); with more colors, a split on another channel could end
# in the same status later.
PALETTE_NAN_PIXELS = (64, 1001)
PALETTE_NAN_CHANNELS = (3, 4)
PALETTE_NAN_POSITIONS = ("middle", "last")
PALETTE_NAN_SETS = ("nan", "snan")
# Group M (SP4b Task 5b): medians inside a tie class, on the zeros mapping; an even and
# an odd count, so the even counts' rank middle - 1 is selected too.
PALETTE_MEDIAN_PIXELS = (1000, 1001)
PALETTE_MEDIAN_CHANNELS = (1, 3)
PALETTE_MEDIAN_CAPACITIES = (2, 256)
# The zeros mapping's words in key order (f32::total_cmp) with their weights out of 64:
# half the weight lies below +0, so the median of a channel falls near the -0/+0
# boundary (on the smallest denormals, -0 or +0, by sampling): inside the zero class,
# whose values compare equal (+-0, and under DAZ every denormal too). +-1 in every
# channel give every channel the range 2. The normals next to the denormals are +-0.25,
# not +-FLT_MIN: a bucket of zero-class values and -FLT_MIN has a mean that FTZ flushes
# to -0, and B3 then ends every daz_ftz case at capacity 256 with status 6 (no color).
ZERO_WEIGHTS = (
    ("0xbf800000", 6),  # -1
    ("0xbf000000", 6),  # -0.5
    ("0xbe800000", 6),  # -0.25
    ("0x807fffff", 6),  # the negative denormal of largest magnitude
    ("0x80000001", 7),  # the negative denormal of smallest magnitude
    ("0x80000000", 1),  # -0
    ("0x00000000", 1),  # +0
    ("0x00000001", 7),
    ("0x007fffff", 6),
    ("0x3e800000", 6),
    ("0x3f000000", 6),
    ("0x3f800000", 6),
)
SWEEP_MAX_PIXELS = 300
# dither (SP4b Task 5): a prepared palette (cn_neighborhood_palette_create, _apply and
# _free), cn_neighborhood_dither with a palette (colors 2 and map_size 2, which the
# palette path ignores) and cn_neighborhood_palette_riemersma. A palette of 300 or more
# unique finite colors is searched through a kd tree, a smaller one linearly (D12).
DITHER_SIZES = (1, 2, 7, 8, 9, 16, 17, 40, 299, 300)
DITHER_CHANNELS = (1, 3, 4)
DITHER_SHAPE = (17, 23)  # (h, w)
DITHER_BENCH = (384, 512, 3)  # the bench's Floyd-Steinberg form, 16 colors
DITHER_SETS = ("clean", "edges", "nan", "mixed")
DITHER_HISTORY = 2
DITHER_DECAY = "0x3f000000"  # 0.5, as float32 bits
SWEEP_MAX_COLORS = 40
# The palette forms: unit (a random palette); grid (src and palette on the grid
# mapping: duplicate colors, +-0 twins and exactly equidistant entries); daz (entries 0
# and 1 differ only in channel 0, -2**-70 against -2**-72, so on a src pixel equal to
# them elsewhere and +-0 in channel 0 their distances are the denormals 2**-140 and
# 2**-144 by default and both 0 under DAZ|FTZ: a tie only under DAZ); nonfinite (entry
# 1 holds +inf and entry 2 a NaN in channel 0); nanfirst (entry 0 holds a negative NaN,
# whose luminance key sorts first, so every distance to the first entry is NaN).
DITHER_FORMS = ("unit", "grid", "daz", "nonfinite", "nanfirst")
_DITHER_FORM_ENTRIES: dict[str, dict[int, tuple[str, ...]]] = {
    "daz": {
        0: ("0x9c800000", "0x3e800000", "0x3f000000", "0x3f400000"),
        1: ("0x9b800000", "0x3e800000", "0x3f000000", "0x3f400000"),
    },
    "nonfinite": {1: ("0x7f800000",), 2: ("0x7fc12345",)},
    "nanfirst": {0: ("0xffc54321",)},
}
# composite (SP4b Task 6): cn_composite_canvas called as native_composite.canvas calls
# it (composite_geometry()). A case's args are canvas()'s: each layer's h, w and c (bh,
# bw, bc; oh, ow, oc) and whether it is a constant (a Color: c floats, whose h and w
# are the other layer's), the mode (0-22), x, y and crop. D13's shortcut blends a full
# overlap straight into out: two images of one size at x = y = 0.
COMPOSITE_SHAPE = (9, 13)  # (h, w)
COMPOSITE_CHANNELS = (1, 3, 4)
COMPOSITE_PAIRS = tuple((b, o) for b in COMPOSITE_CHANNELS for o in COMPOSITE_CHANNELS)
SHORTCUT_MODES = (0, 1, 2, 4, 9, 13, 17, 22)
COMPOSITE_SETS = ("clean", "nan", "edges", "mixed")
# Group N, the shortcut's near misses: kind -> (overlay (h, w), x, y, crop, base
# constant, overlay constant) beside a COMPOSITE_SHAPE base. offset: the paste one pixel
# off (0, 0), over a smaller region; padded: an overlay one column wider pads the
# canvas, so the base is narrower than it (and has fewer channels unless it has 4);
# color and flat: a constant base or overlay; larger: an overlay larger than the
# region, cropped to it at (0, 0).
NEAR_MISSES = {
    "offset": ((9, 13), 1, 1, 1, 0, 0),
    "padded": ((9, 14), 0, 0, 0, 0, 0),
    "color": ((9, 13), 0, 0, 0, 1, 0),
    "flat": ((9, 13), 0, 0, 0, 0, 1),
    "larger": ((11, 15), 0, 0, 1, 0, 0),
}
NEAR_PAIRS = ((3, 3), (4, 4), (1, 3), (4, 1))
NEAR_MODES = (1, 9)
# Group A: the blend's three paths (image_range: target < 4; a 4-channel overlay on a
# base without alpha; a 4-channel base).
BLEND_PATHS = ((3, 3), (3, 4), (4, 4))
# Group G: canvas()'s offset and crop rules on a smaller overlay; the disjoint offsets
# with crop leave no region (pixels 0).
COMPOSITE_SMALL = (6, 8)
COMPOSITE_OFFSETS = (-3, 0, 5)
DISJOINT = ((20, 0), (0, -20))
GEOMETRY_PAIRS = ((3, 4), (4, 1))
# Group C: constant layers.
COLOR_PAIRS = ((3, 3), (1, 4), (4, 3))
COLOR_OFFSETS = ((0, 0), (2, -3))
COLOR_MODES = (0, 1, 9, 17)
# Group T: DAZ twins on the tiny mapping. In (3, 3) modes 4 and 14 add and subtract
# denormals; with a 4-channel base under a 3-channel overlay the colour channels are
# a + ..., which the clip's compare reads as 0 under DAZ.
DAZ_PAIRS = ((3, 3), (4, 3))
DAZ_MODES = (4, 14)
COMPOSITE_BENCH = (384, 512, 3)  # parallel-branches: the two blurs, mode 1
SWEEP_MAX_SIDE = 40
# blend (SP4b Task 6 Step 6): cn_blend_images(overlay, base, out, pixels, overlay
# channels, base channels, mode) and cn_blend_mode(overlay, base, out, count, mode),
# both partitioned at 65,536 (pixels or elements). The AVX2 unit's 8-lane form: modes 0
# (NORMAL) and 1 (MULTIPLY), equal channel counts below 4.
BLEND_INPUTS = ("overlay", "base")  # call order
BLEND_PAIRS = COMPOSITE_PAIRS  # (overlay, base) channels
BLEND_SETS = COMPOSITE_SETS
SIMD_MODES = (0, 1)
SIMD_PAIRS = ((1, 1), (3, 3))
BLEND_DAZ_PIXELS = 17  # group T: 2 groups and a tail at 1 channel, 6 and a tail at 3
# tail (SP4b Task 8): the spectral filter's C stages around cv2.dft
# (native_spectral_filter.filter2d: cn_spectral_tile stages one channel of a tile of the
# padded image, with its border, in the float64 transform plane; cn_spectral_multiply
# multiplies the plane's packed spectrum by the kernel's in place; cn_spectral_store
# narrows the plane's tile into the float32 result), cn_normal_output
# (native_color_ops.normal_output) and cn_caption_compose
# (native_buffers.compose_caption). Each entry's scalar parameters in call order; a
# normal case's args also hold alpha, 1 when the alpha plane is given and 0 for NULL.
TAIL_PARAMS = {
    TILE_ENTRY: (
        "h",
        "w",
        "c",
        "padding",
        "y",
        "x",
        "th",
        "tw",
        "kh",
        "kw",
        "dh",
        "dw",
        "channel",
        "border",
    ),
    STORE_ENTRY: ("oh", "ow", "c", "y", "x", "th", "tw", "dh", "dw", "channel"),
    MULTIPLY_ENTRY: ("h", "w"),
    NORMAL_ENTRY: ("pixels", "invert_r", "invert_g", "channels"),
    CAPTION_ENTRY: ("height", "width", "channels", "caption_height", "top"),
}
TAIL_KINDS = {
    TILE_ENTRY: "tile",
    STORE_ENTRY: "store",
    MULTIPLY_ENTRY: "multiply",
    NORMAL_ENTRY: "normal",
    CAPTION_ENTRY: "caption",
}
# The spectral entries' image: filter2d's tiles of SPECTRAL_BLOCK pixels (3 x 3 tiles at
# both paddings) under a SPECTRAL_KERNEL kernel. A tile's plane is (bh + kh, bw + kw)
# for blocks of (bh, bw), one row and column more than filter2d's least, so every plane
# has zero rows and columns. Group R: images smaller than REACH_KERNEL's reach, so a
# border folds over several periods (and a 1 x 1 image at padding 0 has length 1).
SPECTRAL_SHAPE = (13, 17, 3)
SPECTRAL_KERNEL = (5, 7)
SPECTRAL_BLOCK = (6, 8)
SPECTRAL_PADDINGS = (0, 2)
SPECTRAL_BORDERS = (0, 1, 2, 3, 4)  # constant, replicate, reflect, wrap, reflect101
SPECTRAL_TILES = ("first", "inner", "last")
SPECTRAL_REACH = ((1, 1, 1), (2, 3, 2))
REACH_KERNEL = (9, 11)
# Group W of cn_spectral_tile: (shape, padding, kernel, border, channel), the whole
# padded image as one tile. Each plane is above the grain (65,536) with an odd chunk
# length, and its chunks start mid-row: in the image (row 130, column 132); in the left
# and the right border edge (178, 82 and 356, 164); in the frame and the right edge
# (193, 76 and 386, 151).
TILE_WIDE = (
    ((250, 250, 1), 0, (11, 13), 4, 0),
    ((494, 44, 3), 0, (41, 201), 3, 2),
    ((571, 71, 4), 2, (5, 151), 2, 3),
)
# The bench's normal-map: a 512 x 384 one-channel image under the multi-Gaussian 131 x
# 131 kernel (reflect101), one tile in filter2d's optimal (540, 648) plane.
SPECTRAL_BENCH = ((384, 512, 1), (131, 131), (540, 648))
STORE_CHANNELS = (1, 3)
# Group D of cn_spectral_store: whole-image tiles of 2,000 and 2,091 doubles, enough
# bits words to narrow into the float32 denormal range, which FTZ flushes.
STORE_DAZ = ((40, 50, 1), (41, 51, 3))
STORE_WIDE = (251, 263, 2)  # 66,013 pixels: 2 chunks of 33,007/33,006, mid-row
MULTIPLY_HEIGHTS = (1, 2, 7, 8)
MULTIPLY_WIDTHS = (1, 2, 9, 10)
MULTIPLY_DAZ = ((31, 32), (32, 33))  # products of bits words in the denormal range
MULTIPLY_WIDE = (301, 250)  # 302 row jobs at grain 65,536 // 250 + 1: 2 chunks of 151
NORMAL_SETS = ("edges", "nan", "inf")
NORMAL_BENCH = 196_608  # the bench's 512 x 384 pixels
CAPTION_SIZES = ((1, 1, 1), (5, 7, 3), (9, 13, 2))  # (height, width, caption_height)
# 65,549 pixels at grain 16,384: 5 chunks of 13,110/13,109, starting mid-row.
CAPTION_WIDE = (630, 101, 19)
CAPTION_BENCH = (384, 512, 32)  # the bench's 512 x 384 images under a 32-row caption
# cn_pixels_convert_checked's `type` values (native_buffers._TYPES order) and
# `output` values.
CONVERT_DTYPES = (
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
CONVERT_OUTPUTS = ("float32", "uint8", "uint16")
# The bridges' (type, output, normalize) combos with their own loops (SP4b D7):
# u8 -> f32 normalized, f32 clamp, f32 copy, f32 -> u8 without and with the clamp.
HOT_COMBOS = ((2, 0, 1), (0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 1, 1))
# The conversion grain is 262,144: WIDE_COUNT is one chunk, 589,824 is 3 chunks of
# 196,608 and 2,764,800 is 11 chunks of 251,345/251,346. The lens grain 65,536 splits
# WIDE_COUNT into 4 chunks of 49,153/49,154.
WIDE_COUNT = 196_613
LARGE_COUNTS = (589_824, 2_764_800)
SMALL_COUNTS = tuple(range(1, 18))
SWEEP_MAX_COUNT = 140_000
# conversion_small's float32 sets, rotated over the counts (controller ruling).
_SMALL_SETS = ("edges", "snan", "inf")
_ARRAYS = ("src", "kernel", "dst", "out")
_CTYPES = {
    "sigma_x": ctypes.c_double,
    "sigma_y": ctypes.c_double,
    "rx": ctypes.c_double,
    "ry": ctypes.c_double,
    "border": ctypes.c_int,
}
# S1: separable 4551 values, 5 chunks of 911/910 at grain 1024 (first boundary at
# column 50 of a 123-wide row); convolution 6750/7332/7938 values at padding
# 0/1/2, 2 chunks at B3's grain 4096 (the manifest's) and one at the convolution
# grain 16384 (SP4b Task 3e).
_S1 = {"separable": (37, 41, 3), "convolution": (45, 50, 3)}
# S2: widths 1 to 2R + 1 with R = 8 (AVX2 lanes); group C's form per entry.
_C_FORMS = {
    "cn_gaussian_f32": (21, 11),
    "cn_box_separable": (1.5, 2.5),
    "cn_convolution_spatial": (3, 3),
}
# separable_tiles (SP4b D9): the row tiles' shapes. The manifest is written from B3
# before Task 3's geometry sweep chooses B0 (16, 32 or 64), so it holds every
# candidate's band edges: a band holds B = max(B0, 4 * ry) rows, which is 20, 32 or 64
# at ry 5 (ny 11) and B0 at ry <= 4 (every box form). Group H's heights are B - 1, B,
# B + 1, 2B and 2B + 1 of each.
TILE_BANDS = (16, 20, 32, 64)
TILE_HEIGHTS = tuple(
    sorted({n for b in TILE_BANDS for n in (b - 1, b, b + 1, 2 * b, 2 * b + 1)})
)
# The forms the tiles serve, per entry: SP4a's and the bench's Gaussians.
TILE_FORMS = {
    "cn_gaussian_f32": (*ENTRIES["cn_gaussian_f32"].forms, (25, 25), (49, 49)),
    "cn_box_separable": ENTRIES["cn_box_separable"].forms,
}
TILE_BENCH = (384, 512, 3)  # the bench's 512x384 images
TILE_PLANE = (200, 300, 3)  # row_width 900 = 4 (mod 8)
TILE_SETS = ("clean", "nan", "edges", "mixed")
# The fallback candidates (D9: R > SEPARABLE_TILE_FLOATS / 16), chosen before Task 3's
# geometry sweep: (shape, Gaussian form, box form). The first shape's R is 600, above
# 512 (the 8192-float candidate's limit) but not 2048; at the chosen 32768 it runs in
# tiles (B 400 for the Gaussian, 404 for the box; R 600, W 32, two bands). The
# second's R is above 2048 (every candidate's), so it runs the fallback, where at
# grain 1024 its 44 chunks of 1002/1003 outputs start off the 8-lane grid of its
# 21-wide rows.
TILE_FALLBACKS = (
    ((600, 20, 1), (3, 201), (1.5, 100.5)),
    ((2100, 21, 1), (3, 685), (1.5, 342.5)),
)
# No tables (D9: a radius above 4096): (shape, Gaussian form).
TILE_UNTABLED = (((3, 40, 1), (8195, 3)), ((40, 3, 1), (3, 8195)))
# Indices 6 and 18 of the (5,5) kernel: zero taps, which the convolution skips.
_ZERO_TAPS = [[6, "0x00000000"], [18, "0x80000000"]]
# Outputs are pre-filled with this pattern; run_case refuses one that keeps it. A float64
# output's fill is the all-ones word.
_UNWRITTEN = 0xFFFFFFFF
_UNWRITTEN_DOUBLE = 0xFFFFFFFFFFFFFFFF
# Bytes after a 1-D adapter's output that must stay as filled (one AVX2 register).
_CANARY = 32
# The enforce clamp's reference is upstream's normalize on the reference stack (spec
# section 3), which this venv evaluates (oracle_clip()). A conversion case re-based on
# it (rebase()) records this. Task 3b1 re-based 40 cases on NumPy 1.24.4's np.clip;
# NumPy 2.5.3's gives B3's bits there, so U2 restored them to B3's outputs.
ORACLE = {"oracle": "upstream chaiNNer, CPython 3.14.8, NumPy 2.5.3"}
# A case whose outputs chaiNNer-C defines where upstream's are not one value: the four
# dither cases re-based in SP5a-c (Consult 12 R-q), whose palettes hold NaN keys that
# upstream ties and orders by its random AHashSet. They hold the DLL's outputs.
PORT = {
    "port": "chaiNNer-C under upstream's measured NaN key order; equal keys are an "
    "upstream AHashSet tie, first occurrence kept"
}


@dataclass(frozen=True)
class Case:
    id: str
    entry: str
    args: dict
    inputs: dict[str, dict]
    mxcsr: Mxcsr
    payloads: Payloads


def _special_pairs(n: int, special: str, shift: int = 0) -> list[list]:
    """[index, bits] at the in-range, deduplicated listed indices, cycling the set
    from its value `shift`."""
    values = SPECIALS[special]["values"]
    if not values:
        return []
    listed = (0, 1, 7, 8, n // 3, n // 2, n - 9, n - 8, n - 2, n - 1)
    indices = dict.fromkeys(i for i in listed if 0 <= i < n)
    return [[i, values[(k + shift) % len(values)]] for k, i in enumerate(indices)]


def _recipe(
    shape: tuple[int, ...],
    seed: int,
    mapping: str,
    special: str,
    dtype: str = "float32",
    position: int = 0,
) -> dict:
    """A recipe; its dtype is named only when not float32 (the SP4a manifests').
    A mixed set is rotated by the input's position (D3)."""
    shift = position if SPECIALS[special]["payloads"] == "mixed" else 0
    recipe = {
        "shape": list(shape),
        "seed": seed,
        "mapping": mapping,
        "specials": _special_pairs(math.prod(shape), special, shift),
    }
    if dtype != "float32":
        recipe["dtype"] = dtype
    return recipe


def _seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def case(
    case_id: str,
    entry: str,
    form: tuple[float, float],
    place: int | None,
    mirror: int,
    shape: tuple[int, int, int],
    special: str,
    mxcsr: Mxcsr,
    *,
    seed_key: str | None = None,
    src_mapping: str = "unit",
    kernel_holds_set: bool = True,
    kernel_specials: tuple[tuple[int, str], ...] = (),
) -> Case:
    """One case; every input's seed is the first 8 bytes of SHA-256(seed_key or
    the id), big-endian.

    The convolution kernel holds the case's set unless kernel_holds_set is false
    or the set is mixed: one non-finite tap reaches every output pixel, so src's
    NaN propagation and NaN positions are visible only with a finite kernel
    (controller ruling, 2026-10-03). The (5,5) kernel always holds two zero taps,
    and then kernel_specials.
    """
    spec = ENTRIES[entry]
    h, w, c = shape
    args: dict = {"h": h, "w": w, "c": c, **dict(zip(spec.form, form, strict=True))}
    args.update(spec.fixed)
    if spec.place:
        args[spec.place] = place
    args[spec.mirror] = mirror
    seed = _seed(case_id if seed_key is None else seed_key)
    inputs = {"src": _recipe(shape, seed, src_mapping, special)}
    if entry == "cn_convolution_spatial":
        holds = kernel_holds_set and special != "mixed"
        kernel_set = special if holds else "clean"
        kernel = _recipe((args["kh"], args["kw"]), seed, "signed", kernel_set)
        if form == (5, 5):
            kernel["specials"] += [list(tap) for tap in _ZERO_TAPS]
        kernel["specials"] += [list(tap) for tap in kernel_specials]
        inputs["kernel"] = kernel
    return Case(case_id, entry, args, inputs, mxcsr, SPECIALS[special]["payloads"])


def _matrix_case(
    group: str,
    entry: str,
    form: tuple[float, float],
    place: int | None,
    mirror: int,
    shape: tuple[int, int, int],
    special: str,
    mxcsr: Mxcsr,
) -> Case:
    spec = ENTRIES[entry]
    parts = [group, entry, f"{spec.label}{form[0]:g}x{form[1]:g}"]
    if spec.place:
        parts.append(f"{spec.place}{place}")
    parts += [f"{spec.mirror}{mirror}", special, mxcsr, "x".join(map(str, shape))]
    # The seed key leaves out the MXCSR state, so default and daz_ftz twins share
    # their inputs (review I1).
    return case(
        "-".join(parts),
        entry,
        form,
        place,
        mirror,
        shape,
        special,
        mxcsr,
        seed_key="-".join(part for part in parts if part != mxcsr),
        src_mapping="tiny" if group == "E" else "unit",
        kernel_holds_set=group != "A",
    )


def conversion_case(
    case_id: str,
    combo: tuple[int, int, int],
    count: int,
    label: str,
    mxcsr: Mxcsr,
    *,
    seed_key: str | None = None,
    mapping: str = "signed",
) -> Case:
    """One cn_pixels_convert_checked case of `count` elements; seeds as case().

    label is the float32 src's special set (on `mapping`, signed in [-1, 1) unless
    given), "ties" (the float32 ties()), or for another dtype its mapping (bits,
    or ramp for uint8).
    """
    kind, output, normalize = combo
    dtype = CONVERT_DTYPES[kind]
    seed = _seed(case_id if seed_key is None else seed_key)
    if label == "ties":
        recipe = _recipe((count,), seed, "ties", "clean")
    elif dtype == "float32":
        recipe = _recipe((count,), seed, mapping, label)
    else:
        recipe = _recipe((count,), seed, label, "clean", dtype)
    args = {"count": count, "type": kind, "output": output, "normalize": normalize}
    payloads = SPECIALS[label]["payloads"] if label in SPECIALS else "single"
    return Case(case_id, CONVERT_ENTRY, args, {"src": recipe}, mxcsr, payloads)


def _conversion_matrix_case(
    group: str, combo: tuple[int, int, int], label: str, mxcsr: Mxcsr, count: int
) -> Case:
    kind, output, normalize = combo
    parts = [group, f"t{kind}o{output}n{normalize}", label, mxcsr, str(count)]
    return conversion_case(
        "-".join(parts),
        combo,
        count,
        label,
        mxcsr,
        seed_key="-".join(part for part in parts if part != mxcsr),
    )


def _conversion_matrix(kernel: str) -> list[Case]:
    """D3's groups for cn_pixels_convert_checked (ids group-tToOnN-set-mxcsr-count).

    conversion: A (72), every (type, output, normalize) at WIDE_COUNT, float32 on
    nan and every other dtype on bits, default MXCSR, plus daz_ftz twins of the six
    float32 combos (D4's denormal-bearing DAZ cases); L (40), each hot combo at the
    LARGE_COUNTS, float32 on clean and snan, uint8 on bits and ramp, both states;
    T (4), ties for (0,1,0) and (0,1,1), both states.
    conversion_small: S (204), each hot combo at counts 1-17 in both states;
    float32 on (edges, snan, inf)[(count + combo index) % 3], uint8 on ramp and bits.
    """
    cases = []
    if kernel == "conversion":
        combos = [
            (t, o, n)
            for t in range(len(CONVERT_DTYPES))
            for o in range(3)
            for n in range(2)
        ]
        for mxcsr in MXCSR_STATES:
            for combo in combos:
                if mxcsr == "default" or combo[0] == 0:
                    label = "nan" if combo[0] == 0 else "bits"
                    cases.append(
                        _conversion_matrix_case("A", combo, label, mxcsr, WIDE_COUNT)
                    )
        for combo in HOT_COMBOS:
            labels = ("bits", "ramp") if combo[0] == 2 else ("clean", "snan")
            for count in LARGE_COUNTS:
                for label in labels:
                    for mxcsr in MXCSR_STATES:
                        cases.append(
                            _conversion_matrix_case("L", combo, label, mxcsr, count)
                        )
        for combo in ((0, 1, 0), (0, 1, 1)):
            for mxcsr in MXCSR_STATES:
                cases.append(
                    _conversion_matrix_case("T", combo, "ties", mxcsr, ties().size)
                )
        return cases
    for index, combo in enumerate(HOT_COMBOS):
        for count in SMALL_COUNTS:
            if combo[0] == 2:
                labels = ("ramp", "bits")
            else:
                labels = (_SMALL_SETS[(count + index) % len(_SMALL_SETS)],)
            for label in labels:
                for mxcsr in MXCSR_STATES:
                    cases.append(
                        _conversion_matrix_case("S", combo, label, mxcsr, count)
                    )
    return cases


def _conversion_sweep(kernel: str, count: int, seed: int) -> list[Case]:
    """`count` cases drawn from Generator(PCG64(seed)), ids sweep-<seed>-<n> (D4).

    Per case, in this order: the combo (from the kernel's matrix), special set,
    MXCSR state, element count in 1-SWEEP_MAX_COUNT and a variant in 0-5: a
    float32 src takes the set, and for the clean set the variant picks the signed,
    tiny (denormal) or bits mapping (raw bits: every magnitude, so scaled values
    beyond int32 and at its bounds; review M2); a uint8 src takes bits or ramp by
    the variant, another dtype bits.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    combos = list(
        dict.fromkeys(
            (c.args["type"], c.args["output"], c.args["normalize"])
            for c in _conversion_matrix(kernel)
        )
    )
    specials = SP4A_SETS
    cases = []
    for n in range(count):
        combo = combos[int(rng.integers(len(combos)))]
        special = specials[int(rng.integers(len(specials)))]
        mxcsr = MXCSR_STATES[int(rng.integers(len(MXCSR_STATES)))]
        size = int(rng.integers(1, SWEEP_MAX_COUNT + 1))
        variant = int(rng.integers(6))
        dtype = CONVERT_DTYPES[combo[0]]
        mapping = "signed"
        if dtype == "float32":
            label = special
            if special == "clean":
                mapping = ("signed", "tiny", "bits")[variant % 3]
        else:
            label = ("bits", "ramp")[variant % 2] if dtype == "uint8" else "bits"
        cases.append(
            conversion_case(
                f"sweep-{seed}-{n}", combo, size, label, mxcsr, mapping=mapping
            )
        )
    return cases


def _border_case(
    group: str,
    form: tuple[int, int],
    padding: int,
    fused: int,
    shape: tuple[int, int, int],
    special: str,
    mxcsr: Mxcsr,
    *,
    mapping: str = "unit",
    denormal_row: bool = False,
) -> Case:
    """A convolution_border case (ids as _matrix_case's). Its kernel stays finite:
    it holds the edges set (zero, denormal and FLT_MAX taps) and no other."""
    entry = "cn_convolution_spatial"
    parts = [group, entry, f"k{form[0]}x{form[1]}", f"padding{padding}"]
    parts += [f"fused{fused}", special, mxcsr, "x".join(map(str, shape))]
    return case(
        "-".join(parts),
        entry,
        form,
        padding,
        fused,
        shape,
        special,
        mxcsr,
        seed_key="-".join(part for part in parts if part != mxcsr),
        src_mapping=mapping,
        kernel_holds_set=special == "edges",
        kernel_specials=DENORMAL_ROW if denormal_row else (),
    )


def _border_matrix() -> list[Case]:
    """convolution_border's groups (SP4b D8, review focus 1).

    R (108): the 61-tap reach, (1,61) on (3, n, c) and (61,1) on (n, 5, c) for n in
    REACH_SIZES, c 1 or 3 by n, paddings 0/1/2, fused 0/8, clean, default MXCSR.
    W (66): (n, 34 - n, c) for n in 1-33, so every width and height 1-33, with (3,3)
    and (11,11): padding n % 3, fused 8 when n + form index is odd, c 1 or 3 by n // 2;
    clean, default. L (40): PLANE at padding 0 with (1,61) and (61,1), fused 0/8,
    every border set, both MXCSR states. T (8): PLANE's src in the denormal range
    (tiny), both states. Z (48): the (5,5) kernel with DENORMAL_ROW on (9, 13, 3) and
    (8, 11, 1), paddings 0/1/2, fused 0/8, clean and inf, both states.
    """
    cases = []
    for form in ((1, 61), (61, 1)):
        for n in REACH_SIZES:
            c = (1, 3)[n % 2]
            shape = (3, n, c) if form == (1, 61) else (n, 5, c)
            for padding in (0, 1, 2):
                for fused in (0, 8):
                    cases.append(
                        _border_case(
                            "R", form, padding, fused, shape, "clean", "default"
                        )
                    )
    for n in range(1, 34):
        for index, form in enumerate(((3, 3), (11, 11))):
            fused = 8 if (n + index) % 2 else 0
            shape = (n, 34 - n, (1, 3)[n // 2 % 2])
            cases.append(
                _border_case("W", form, n % 3, fused, shape, "clean", "default")
            )
    for mxcsr in MXCSR_STATES:
        for form in ((1, 61), (61, 1)):
            for fused in (0, 8):
                for special in BORDER_SETS:
                    cases.append(
                        _border_case("L", form, 0, fused, PLANE, special, mxcsr)
                    )
                cases.append(
                    _border_case(
                        "T", form, 0, fused, PLANE, "clean", mxcsr, mapping="tiny"
                    )
                )
    for mxcsr in MXCSR_STATES:
        for shape in ((9, 13, 3), (8, 11, 1)):
            for padding in (0, 1, 2):
                for fused in (0, 8):
                    for special in ("clean", "inf"):
                        cases.append(
                            _border_case(
                                "Z",
                                (5, 5),
                                padding,
                                fused,
                                shape,
                                special,
                                mxcsr,
                                denormal_row=True,
                            )
                        )
    return cases


def _border_sweep(count: int, seed: int) -> list[Case]:
    """`count` cases drawn from Generator(PCG64(seed)), ids sweep-<seed>-<n> (D4).

    Per case, in this order: the form, padding, fused value, set and MXCSR state
    from the manifest's values; h and w in 1-64; one case in eight then draws h in
    65-300; c 1 or 3; a clean src's mapping, unit or tiny; and for (5,5) whether its
    kernel holds DENORMAL_ROW. The kernel holds the edges set and no other.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    cases = []
    for n in range(count):
        form = BORDER_FORMS[int(rng.integers(len(BORDER_FORMS)))]
        padding = int(rng.integers(3))
        fused = (0, 8)[int(rng.integers(2))]
        special = BORDER_SETS[int(rng.integers(len(BORDER_SETS)))]
        mxcsr = MXCSR_STATES[int(rng.integers(len(MXCSR_STATES)))]
        h, w = (int(size) for size in rng.integers(1, 65, size=2))
        if int(rng.integers(8)) == 0:
            h = int(rng.integers(65, 301))
        c = (1, 3)[int(rng.integers(2))]
        mapping = (
            ("unit", "tiny")[int(rng.integers(2))] if special == "clean" else "unit"
        )
        row = form == (5, 5) and bool(rng.integers(2))
        cases.append(
            case(
                f"sweep-{seed}-{n}",
                "cn_convolution_spatial",
                form,
                padding,
                fused,
                (h, w, c),
                special,
                mxcsr,
                src_mapping=mapping,
                kernel_holds_set=special == "edges",
                kernel_specials=DENORMAL_ROW if row else (),
            )
        )
    return cases


def lens_case(
    case_id: str,
    count: int,
    accumulate: int,
    pair: str,
    special: str,
    mxcsr: Mxcsr,
    *,
    seed_key: str | None = None,
    mapping: str = "signed",
) -> Case:
    """One cn_lens_compose case of `count` elements: inputs LENS_INPUTS, each seeded
    by SHA-256(<seed_key or id>-<name>) as case() seeds, holding the set (rotated
    by the input's position when mixed); (a, b) = LENS_PAIRS[pair] as bits."""
    key = case_id if seed_key is None else seed_key
    inputs = {
        name: _recipe((count,), _seed(f"{key}-{name}"), mapping, special, position=i)
        for i, name in enumerate(LENS_INPUTS)
    }
    a, b = LENS_PAIRS[pair]
    args = {"count": count, "a": a, "b": b, "accumulate": accumulate}
    return Case(case_id, LENS_ENTRY, args, inputs, mxcsr, SPECIALS[special]["payloads"])


def _lens_matrix_case(
    group: str, accumulate: int, pair: str, special: str, mxcsr: Mxcsr, count: int
) -> Case:
    parts = [group, f"acc{accumulate}", pair, special, mxcsr, str(count)]
    return lens_case(
        "-".join(parts),
        count,
        accumulate,
        pair,
        special,
        mxcsr,
        seed_key="-".join(part for part in parts if part != mxcsr),
    )


def _lens_matrix() -> list[Case]:
    """lens's groups (ids group-accA-pair-set-mxcsr-count).

    W (48): WIDE_COUNT, accumulate 0/1, every pair; every lens set at the default
    MXCSR, and edges, nan and inf (the sets holding denormals) under DAZ|FTZ.
    S (68): counts 1-17, accumulate 0/1, both states; the set LENS_SETS[count % 5]
    and the pair (count + accumulate) % 3.
    """
    cases = []
    for mxcsr in MXCSR_STATES:
        for accumulate in (0, 1):
            for pair in LENS_PAIRS:
                for special in LENS_SETS:
                    if mxcsr == "default" or special in ("edges", "nan", "inf"):
                        cases.append(
                            _lens_matrix_case(
                                "W", accumulate, pair, special, mxcsr, WIDE_COUNT
                            )
                        )
    pairs = tuple(LENS_PAIRS)
    for count in SMALL_COUNTS:
        for accumulate in (0, 1):
            pair = pairs[(count + accumulate) % len(pairs)]
            special = LENS_SETS[count % len(LENS_SETS)]
            for mxcsr in MXCSR_STATES:
                cases.append(
                    _lens_matrix_case("S", accumulate, pair, special, mxcsr, count)
                )
    return cases


def _lens_sweep(count: int, seed: int) -> list[Case]:
    """`count` cases drawn from Generator(PCG64(seed)), ids sweep-<seed>-<n> (D4).

    Per case, in this order: accumulate, pair, set and MXCSR state from the
    manifest's values, the element count in 1-SWEEP_MAX_COUNT, and a clean case's
    mapping, signed or tiny (denormal range).
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    pairs = tuple(LENS_PAIRS)
    cases = []
    for n in range(count):
        accumulate = int(rng.integers(2))
        pair = pairs[int(rng.integers(len(pairs)))]
        special = LENS_SETS[int(rng.integers(len(LENS_SETS)))]
        mxcsr = MXCSR_STATES[int(rng.integers(len(MXCSR_STATES)))]
        size = int(rng.integers(1, SWEEP_MAX_COUNT + 1))
        mapping = (
            ("signed", "tiny")[int(rng.integers(2))] if special == "clean" else "signed"
        )
        cases.append(
            lens_case(
                f"sweep-{seed}-{n}",
                size,
                accumulate,
                pair,
                special,
                mxcsr,
                mapping=mapping,
            )
        )
    return cases


def power_case(
    case_id: str,
    pixels: int,
    channels: int,
    exponent: str,
    special: str,
    mxcsr: Mxcsr,
    *,
    seed_key: str | None = None,
    mapping: str = "signed",
) -> Case:
    """One cn_lens_power case: src is the (channels, pixels) CHW plane on `mapping`
    (signed in [-1, 1) unless given) holding the set; the exponent is
    POWER_EXPONENTS[exponent] as bits; seeds as case()."""
    seed = _seed(case_id if seed_key is None else seed_key)
    src = _recipe((channels, pixels), seed, mapping, special)
    args = {
        "pixels": pixels,
        "channels": channels,
        "exponent": POWER_EXPONENTS[exponent],
    }
    return Case(
        case_id, POWER_ENTRY, args, {"src": src}, mxcsr, SPECIALS[special]["payloads"]
    )


def _power_matrix_case(
    group: str, channels: int, exponent: str, special: str, mxcsr: Mxcsr, pixels: int
) -> Case:
    parts = [group, f"c{channels}", exponent, special, mxcsr, str(pixels)]
    return power_case(
        "-".join(parts),
        pixels,
        channels,
        exponent,
        special,
        mxcsr,
        seed_key="-".join(part for part in parts if part != mxcsr),
    )


def _power_matrix() -> list[Case]:
    """lens_power's groups (ids group-cC-exponent-set-mxcsr-pixels).

    S (204): pixels 1-17, channels 1/3/4, both MXCSR states, two cases each: with
    k = pixels + channel index, the exponent k % 3 and the set k % 4, then the
    exponent (k + 1) % 3 and the set (k + 2) % 4 (indices into POWER_EXPONENTS and
    POWER_SETS). k takes 17 consecutive values per channel count, so each (channels,
    exponent, set) occurs. L (12): POWER_PIXELS, channels 1/3/4, the exponents 1/5
    and 5, clean and edges, default MXCSR. (The full sets x exponents product at
    pixels 1-17 would exceed the manifest's 100 kB.)
    """
    cases = []
    exponents = tuple(POWER_EXPONENTS)
    for pixels in SMALL_COUNTS:
        for c_index, channels in enumerate(POWER_CHANNELS):
            k = pixels + c_index
            for step in (0, 1):
                exponent = exponents[(k + step) % len(exponents)]
                special = POWER_SETS[(k + 2 * step) % len(POWER_SETS)]
                for mxcsr in MXCSR_STATES:
                    cases.append(
                        _power_matrix_case(
                            "S", channels, exponent, special, mxcsr, pixels
                        )
                    )
    for channels in POWER_CHANNELS:
        for exponent in ("fifth", "five"):
            for special in ("clean", "edges"):
                cases.append(
                    _power_matrix_case(
                        "L", channels, exponent, special, "default", POWER_PIXELS
                    )
                )
    return cases


def _power_sweep(count: int, seed: int) -> list[Case]:
    """`count` cases drawn from Generator(PCG64(seed)), ids sweep-<seed>-<n> (D4).

    Per case, in this order: channels, exponent, set and MXCSR state from the
    manifest's values, pixels in 1-(SWEEP_MAX_COUNT // channels) (at most
    SWEEP_MAX_COUNT elements), and a clean case's mapping, signed or tiny (denormal
    range).
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    exponents = tuple(POWER_EXPONENTS)
    cases = []
    for n in range(count):
        channels = POWER_CHANNELS[int(rng.integers(len(POWER_CHANNELS)))]
        exponent = exponents[int(rng.integers(len(exponents)))]
        special = POWER_SETS[int(rng.integers(len(POWER_SETS)))]
        mxcsr = MXCSR_STATES[int(rng.integers(len(MXCSR_STATES)))]
        pixels = int(rng.integers(1, SWEEP_MAX_COUNT // channels + 1))
        mapping = (
            ("signed", "tiny")[int(rng.integers(2))] if special == "clean" else "signed"
        )
        cases.append(
            power_case(
                f"sweep-{seed}-{n}",
                pixels,
                channels,
                exponent,
                special,
                mxcsr,
                mapping=mapping,
            )
        )
    return cases


def morph_label(entry: str, form: dict) -> str:
    """A morphology form's id part: s<shape>r<radius>i<iterations>m<maximum>l<lanes>
    (cn_morphology_complete) or x<cross>r<radius>i<iterations>m<maximum>."""
    if entry == MORPH_ENTRY:
        keys = (("s", "shape"), ("r", "radius"), ("i", "iterations"))
        keys += (("m", "maximum"), ("l", "lanes"))
    else:
        keys = (("x", "cross"), ("r", "radius"), ("i", "iterations"), ("m", "maximum"))
    return "".join(f"{letter}{form[name]}" for letter, name in keys)


def morph_case(
    case_id: str,
    entry: str,
    form: dict,
    shape: tuple[int, int, int],
    special: str,
    mxcsr: Mxcsr,
    *,
    seed_key: str | None = None,
    mapping: str = "unit",
) -> Case:
    """One cn_morphology_complete or cn_filter_morphology case: src (h, w, c) on
    `mapping` (unit in [0, 1) unless given) holding the set; form, the entry's other
    arguments (MORPH_FORMS' names); seeds as case()."""
    h, w, c = shape
    seed = _seed(case_id if seed_key is None else seed_key)
    args = {"h": h, "w": w, "c": c, **form}
    src = _recipe(shape, seed, mapping, special)
    return Case(
        case_id, entry, args, {"src": src}, mxcsr, SPECIALS[special]["payloads"]
    )


def _morph_matrix_case(
    group: str,
    entry: str,
    form: dict,
    shape: tuple[int, int, int],
    special: str,
    mxcsr: Mxcsr,
    mapping: str = "unit",
) -> Case:
    kind = "complete" if entry == MORPH_ENTRY else "filter"
    parts = [group, kind, morph_label(entry, form), special, mxcsr]
    parts.append("x".join(map(str, shape)))
    return morph_case(
        "-".join(parts),
        entry,
        form,
        shape,
        special,
        mxcsr,
        seed_key="-".join(part for part in parts if part != mxcsr),
        mapping=mapping,
    )


def exceptional_case(
    case_id: str,
    count: int,
    word: str,
    position: str,
    mxcsr: Mxcsr,
    *,
    seed_key: str | None = None,
) -> Case:
    """One cn_image_exceptional case: src (count,) on unit (finite, nonnegative and
    never -0, so not exceptional), with `word` at index 0 (position "first") or count -
    1 ("last"); seeds as case()."""
    seed = _seed(case_id if seed_key is None else seed_key)
    index = 0 if position == "first" else count - 1
    src = {
        "shape": [count],
        "seed": seed,
        "mapping": "unit",
        "specials": [[index, word]],
    }
    return Case(
        case_id, EXCEPTIONAL_ENTRY, {"count": count}, {"src": src}, mxcsr, "single"
    )


def _exceptional_matrix_case(
    count: int, word: str, position: str, mxcsr: Mxcsr
) -> Case:
    parts = ["X", position, word, mxcsr, str(count)]
    return exceptional_case(
        "-".join(parts),
        count,
        word,
        position,
        mxcsr,
        seed_key="-".join(part for part in parts if part != mxcsr),
    )


def _morph_matrix() -> list[Case]:
    """morphology's groups (SP4b D3; ids group-entry-form-set-mxcsr-shape, and for X
    X-position-word-mxcsr-count).

    A (216): cn_morphology_complete, every MORPH_FORMS form (shape x radius x
    iterations x maximum x lanes) on MORPH_SHAPE, clean, default MXCSR. F (36):
    cn_filter_morphology, every form on MORPH_SHAPE, clean, default. S (60): Dilate's
    ellipse (DILATE) and Erode's cross (ERODE) at radii 1-3 and iterations 2 on
    MORPH_SHAPE, every MORPH_SETS set, both states. W (17): widths 1-17 at height 5
    with 1, 3 or 4 channels (w % 3), Dilate's ellipse r2 x2 at odd widths and Erode's
    cross r2 x2 at even ones, clean, default. H (9): heights 1-9 at width 9 with 3
    channels, Dilate's ellipse r3 x1 at odd heights and Erode's cross r2 x1 at even
    ones, clean, default. B (2): MORPH_BENCH with the bench's forms (Dilate's ellipse
    r3 x2, Erode's cross r2 x2), clean, default. C (12): MORPH_CHUNKS with Dilate's
    ellipse r2 x2 and Erode's cross r2 x2 on clean, denorm and nan, both states. T (8):
    MORPH_CHUNKS with src in the denormal range (tiny: half denormals, half the
    smallest normals, all positive, so not exceptional), the ellipse r2 x2 lanes 8 and
    the cross r2 x2 (cn_filter_morphology), each as maximum and minimum, both states:
    under DAZ denormals tie with each other, and the ellipse combine's maxss/minss
    return them as +0, where the deques and the cross merge keep a src element's bits.
    X (68): cn_image_exceptional at each EXCEPTIONAL_COUNTS count (index i), the word
    EXCEPTIONAL_WORDS[(i + 6 p) % 12] at position p (0 first, 1 last), under
    MXCSR_STATES[(i // 12 + p) % 2]: each word at both positions, and under both
    states.
    """
    cases = []
    names = tuple(MORPH_FORMS[MORPH_ENTRY])
    values = MORPH_FORMS[MORPH_ENTRY]
    for shape in values["shape"]:
        for radius in values["radius"]:
            for iterations in values["iterations"]:
                for maximum in values["maximum"]:
                    for lanes in values["lanes"]:
                        form = dict(
                            zip(
                                names,
                                (radius, iterations, shape, maximum, lanes),
                                strict=True,
                            )
                        )
                        cases.append(
                            _morph_matrix_case(
                                "A", MORPH_ENTRY, form, MORPH_SHAPE, "clean", "default"
                            )
                        )
    values = MORPH_FORMS[FILTER_ENTRY]
    for cross in values["cross"]:
        for radius in values["radius"]:
            for iterations in values["iterations"]:
                for maximum in values["maximum"]:
                    form = {
                        "radius": radius,
                        "iterations": iterations,
                        "cross": cross,
                        "maximum": maximum,
                    }
                    cases.append(
                        _morph_matrix_case(
                            "F", FILTER_ENTRY, form, MORPH_SHAPE, "clean", "default"
                        )
                    )

    def node_forms(radius: int, iterations: int) -> list[tuple[str, dict]]:
        return [
            (MORPH_ENTRY, {"radius": radius, "iterations": iterations, **DILATE}),
            (FILTER_ENTRY, {"radius": radius, "iterations": iterations, **ERODE}),
        ]

    for mxcsr in MXCSR_STATES:
        for radius in (1, 2, 3):
            for entry, form in node_forms(radius, 2):
                for special in MORPH_SETS:
                    cases.append(
                        _morph_matrix_case(
                            "S", entry, form, MORPH_SHAPE, special, mxcsr
                        )
                    )
    for w in range(1, 18):
        entry, form = node_forms(2, 2)[1 - w % 2]
        shape = (5, w, (1, 3, 4)[w % 3])
        cases.append(_morph_matrix_case("W", entry, form, shape, "clean", "default"))
    for h in range(1, 10):
        entry, form = node_forms(3, 1)[0] if h % 2 else node_forms(2, 1)[1]
        cases.append(
            _morph_matrix_case("H", entry, form, (h, 9, 3), "clean", "default")
        )
    for entry, form in (node_forms(3, 2)[0], node_forms(2, 2)[1]):
        cases.append(
            _morph_matrix_case("B", entry, form, MORPH_BENCH, "clean", "default")
        )
    for mxcsr in MXCSR_STATES:
        for entry, form in node_forms(2, 2):
            for special in ("clean", "denorm", "nan"):
                cases.append(
                    _morph_matrix_case("C", entry, form, MORPH_CHUNKS, special, mxcsr)
                )
    ellipse, cross = node_forms(2, 2)
    tie_forms = (
        ellipse,
        (MORPH_ENTRY, {**ellipse[1], "maximum": 0}),
        (FILTER_ENTRY, {**cross[1], "maximum": 1}),
        cross,
    )
    for mxcsr in MXCSR_STATES:
        for entry, form in tie_forms:
            cases.append(
                _morph_matrix_case(
                    "T", entry, form, MORPH_CHUNKS, "clean", mxcsr, mapping="tiny"
                )
            )
    words = EXCEPTIONAL_WORDS
    for i, count in enumerate(EXCEPTIONAL_COUNTS):
        for p, position in enumerate(EXCEPTIONAL_POSITIONS):
            word = words[(i + 6 * p) % len(words)]
            mxcsr = MXCSR_STATES[(i // 12 + p) % 2]
            cases.append(_exceptional_matrix_case(count, word, position, mxcsr))
    return cases


def _morph_sweep(count: int, seed: int) -> list[Case]:
    """`count` cases drawn from Generator(PCG64(seed)), ids sweep-<seed>-<n> (D4).

    Per case, in this order: the entry; for cn_image_exceptional the element count in
    1-SWEEP_MAX_COUNT, the word, position and MXCSR state; otherwise each MORPH_FORMS
    argument of the entry in its order, the set (MORPH_SETS) and MXCSR state, h and w in
    1-64, then one case in eight draws h in 65-300, c 1, 3 or 4, and a clean src's
    mapping, unit or tiny (denormal range: ties under DAZ).
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    entries = KERNEL_ENTRIES["morphology"]
    cases = []
    for n in range(count):
        entry = entries[int(rng.integers(len(entries)))]
        case_id = f"sweep-{seed}-{n}"
        if entry == EXCEPTIONAL_ENTRY:
            size = int(rng.integers(1, SWEEP_MAX_COUNT + 1))
            word = EXCEPTIONAL_WORDS[int(rng.integers(len(EXCEPTIONAL_WORDS)))]
            position = EXCEPTIONAL_POSITIONS[int(rng.integers(2))]
            mxcsr = MXCSR_STATES[int(rng.integers(len(MXCSR_STATES)))]
            cases.append(exceptional_case(case_id, size, word, position, mxcsr))
            continue
        form = {
            name: values[int(rng.integers(len(values)))]
            for name, values in MORPH_FORMS[entry].items()
        }
        special = MORPH_SETS[int(rng.integers(len(MORPH_SETS)))]
        mxcsr = MXCSR_STATES[int(rng.integers(len(MXCSR_STATES)))]
        h, w = (int(size) for size in rng.integers(1, 65, size=2))
        if int(rng.integers(8)) == 0:
            h = int(rng.integers(65, 301))
        c = (1, 3, 4)[int(rng.integers(3))]
        mapping = (
            ("unit", "tiny")[int(rng.integers(2))] if special == "clean" else "unit"
        )
        cases.append(
            morph_case(case_id, entry, form, (h, w, c), special, mxcsr, mapping=mapping)
        )
    return cases


def resample_case(
    case_id: str,
    shape: tuple[int, int, int],
    target: tuple[int, int],
    filter_id: int,
    gamma: int,
    vector_clip: int,
    special: str,
    mxcsr: Mxcsr,
    *,
    seed_key: str | None = None,
    mapping: str = "unit",
) -> Case:
    """One cn_resample_filtered case: src (h, w, c) on `mapping` (unit in [0, 1) unless
    given) holding the set, resampled to target (th, tw); seeds as case().

    With gamma the nan set is mixed (resample_payloads()), every other case its set's
    payloads."""
    h, w, c = shape
    th, tw = target
    seed = _seed(case_id if seed_key is None else seed_key)
    args = {"h": h, "w": w, "c": c, "th": th, "tw": tw}
    args |= {"filter": filter_id, "gamma": gamma, "vector_clip": vector_clip}
    src = _recipe(shape, seed, mapping, special)
    payloads = resample_payloads(special, gamma)
    return Case(case_id, RESAMPLE_ENTRY, args, {"src": src}, mxcsr, payloads)


def resample_payloads(special: str, gamma: int) -> Payloads:
    """A resample case's payload rule. With gamma, the nan set's negative denormal
    (0x807fffff) goes through cn_alpha_gamma's powf (every element of an RGBA image's
    colour channels, and a scalar tail otherwise), which makes the default NaN
    0xffc00000; it can meet the set's 0x7fc12345 in a filter's sum, so that case is
    mixed (B3's outputs hold both patterns on the RGBA cases). Every other set's NaNs
    are one payload: the set's own, or only the default NaN (inf - inf, inf * 0, powf
    of a negative value)."""
    return "mixed" if gamma and special == "nan" else SPECIALS[special]["payloads"]


def _resample_matrix_case(
    group: str,
    shape: tuple[int, int, int],
    target: tuple[int, int],
    filter_id: int,
    gamma: int,
    vector_clip: int,
    special: str,
    mxcsr: Mxcsr,
) -> Case:
    parts = [group, f"f{filter_id}g{gamma}v{vector_clip}", "x".join(map(str, target))]
    parts += [special, mxcsr, "x".join(map(str, shape))]
    return resample_case(
        "-".join(parts),
        shape,
        target,
        filter_id,
        gamma,
        vector_clip,
        special,
        mxcsr,
        seed_key="-".join(part for part in parts if part != mxcsr),
    )


def _resample_matrix() -> list[Case]:
    """resample's groups (ids group-fFgGvV-target-set-mxcsr-shape).

    G (176): every filter, gamma 0/1, vector_clip 0/1 and channels 1-4 on RESAMPLE_GRID,
    clean, default MXCSR. E (128): each RESAMPLE_EDGES geometry with filters 2 (the
    unclipped triangle) and 5 (Hermite), gamma 0/1, vector_clip 0/1 (every clip form),
    every RESAMPLE_SETS set, both states. B (2): RESAMPLE_BENCH with filter 5, gamma 0,
    vector_clip 0, clean, default.
    """
    cases = []
    (h, w), target = RESAMPLE_GRID
    for filter_id in RESAMPLE_FILTERS:
        for gamma in (0, 1):
            for clip in (0, 1):
                for c in (1, 2, 3, 4):
                    cases.append(
                        _resample_matrix_case(
                            "G",
                            (h, w, c),
                            target,
                            filter_id,
                            gamma,
                            clip,
                            "clean",
                            "default",
                        )
                    )
    for mxcsr in MXCSR_STATES:
        for shape, target in RESAMPLE_EDGES:
            for filter_id in (2, 5):
                for gamma in (0, 1):
                    for clip in (0, 1):
                        for special in RESAMPLE_SETS:
                            cases.append(
                                _resample_matrix_case(
                                    "E",
                                    shape,
                                    target,
                                    filter_id,
                                    gamma,
                                    clip,
                                    special,
                                    mxcsr,
                                )
                            )
    for shape, target in RESAMPLE_BENCH:
        cases.append(
            _resample_matrix_case("B", shape, target, 5, 0, 0, "clean", "default")
        )
    return cases


def _resample_sweep(count: int, seed: int) -> list[Case]:
    """`count` cases drawn from Generator(PCG64(seed)), ids sweep-<seed>-<n> (D4).

    Per case, in this order: the filter, gamma, vector_clip, channels (1-4), set (clean
    or RESAMPLE_SETS) and MXCSR state; h and w in 1-64, then one case in eight draws h in
    65-300; th and tw in 1-64; and a clean src's mapping, unit or tiny.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    specials = ("clean", *RESAMPLE_SETS)
    cases = []
    for n in range(count):
        filter_id = RESAMPLE_FILTERS[int(rng.integers(len(RESAMPLE_FILTERS)))]
        gamma = int(rng.integers(2))
        clip = int(rng.integers(2))
        c = int(rng.integers(1, 5))
        special = specials[int(rng.integers(len(specials)))]
        mxcsr = MXCSR_STATES[int(rng.integers(len(MXCSR_STATES)))]
        h, w = (int(size) for size in rng.integers(1, 65, size=2))
        if int(rng.integers(8)) == 0:
            h = int(rng.integers(65, 301))
        th, tw = (int(size) for size in rng.integers(1, 65, size=2))
        mapping = (
            ("unit", "tiny")[int(rng.integers(2))] if special == "clean" else "unit"
        )
        cases.append(
            resample_case(
                f"sweep-{seed}-{n}",
                (h, w, c),
                (th, tw),
                filter_id,
                gamma,
                clip,
                special,
                mxcsr,
                mapping=mapping,
            )
        )
    return cases


def palette_case(
    case_id: str,
    pixels: int,
    channels: int,
    capacity: int,
    special: str,
    mxcsr: Mxcsr,
    *,
    seed_key: str | None = None,
    mapping: str = "unit",
) -> Case:
    """One cn_palette_median_cut case: src (pixels, channels) on `mapping` (unit in
    [0, 1) unless given) holding the set, at most `capacity` colors; seeds as case()."""
    seed = _seed(case_id if seed_key is None else seed_key)
    args = {"pixels": pixels, "channels": channels, "capacity": capacity}
    src = _recipe((pixels, channels), seed, mapping, special)
    return Case(
        case_id, PALETTE_ENTRY, args, {"src": src}, mxcsr, SPECIALS[special]["payloads"]
    )


def _palette_matrix_case(
    group: str,
    pixels: int,
    channels: int,
    capacity: int,
    mapping: str,
    special: str,
    mxcsr: Mxcsr,
) -> Case:
    parts = [group, f"k{capacity}", mapping, special, mxcsr, f"{pixels}x{channels}"]
    return palette_case(
        "-".join(parts),
        pixels,
        channels,
        capacity,
        special,
        mxcsr,
        seed_key="-".join(part for part in parts if part != mxcsr),
        mapping=mapping,
    )


def palette_nan_index(position: str, pixels: int, channels: int) -> int:
    """Group N's NaN element: pixel pixels // 2, channel channels // 2 ("middle"), or
    the last element ("last")."""
    if position == "middle":
        return pixels // 2 * channels + channels // 2
    return pixels * channels - 1


def _palette_nan_case(position: str, pixels: int, channels: int, special: str) -> Case:
    """A group N case: src on unit with the set's NaN word alone at palette_nan_index,
    capacity 2, default MXCSR (ids N-k2-position-set-default-PIXELSxCHANNELS)."""
    parts = ["N", "k2", position, special, "default", f"{pixels}x{channels}"]
    case_id = "-".join(parts)
    src = {
        "shape": [pixels, channels],
        "seed": _seed("-".join(part for part in parts if part != "default")),
        "mapping": "unit",
        "specials": [
            [
                palette_nan_index(position, pixels, channels),
                SPECIALS[special]["values"][0],
            ]
        ],
    }
    args = {"pixels": pixels, "channels": channels, "capacity": 2}
    return Case(case_id, PALETTE_ENTRY, args, {"src": src}, "default", "single")


def _palette_matrix() -> list[Case]:
    """palette's groups (SP4b D3; ids group-kCAPACITY-mapping-set-mxcsr-PIXELSxCHANNELS,
    group N's with the position for the mapping).

    A (171): every PALETTE_PIXELS count (index i) x PALETTE_CHANNELS (index ci) x
    PALETTE_CAPACITIES (index ki) on unit, the set PALETTE_SETS[(i + ci + ki) % 6] under
    MXCSR_STATES[(i + ki) % 2], so every (set, state) pair occurs. B (9): PALETTE_BENCH
    pixels, every channels and capacity, clean, default. Z (108): PALETTE_TIE_PIXELS x
    channels x capacities x PALETTE_TIES (edges and denorm on unit, clean on tiny), both
    states: zero-class ties in the extrema, and under DAZ in the median and the counts.
    N (16): PALETTE_NAN_POSITIONS x PALETTE_NAN_PIXELS x PALETTE_NAN_CHANNELS x
    PALETTE_NAN_SETS (_palette_nan_case). M (16, SP4b Task 5b): PALETTE_MEDIAN_PIXELS x
    PALETTE_MEDIAN_CHANNELS x PALETTE_MEDIAN_CAPACITIES on zeros, clean, both states:
    medians whose ranks fall inside a tie class.
    """
    cases = []
    for i, pixels in enumerate(PALETTE_PIXELS):
        for ci, channels in enumerate(PALETTE_CHANNELS):
            for ki, capacity in enumerate(PALETTE_CAPACITIES):
                special = PALETTE_SETS[(i + ci + ki) % len(PALETTE_SETS)]
                mxcsr = MXCSR_STATES[(i + ki) % 2]
                cases.append(
                    _palette_matrix_case(
                        "A", pixels, channels, capacity, "unit", special, mxcsr
                    )
                )
    for channels in PALETTE_CHANNELS:
        for capacity in PALETTE_CAPACITIES:
            cases.append(
                _palette_matrix_case(
                    "B", PALETTE_BENCH, channels, capacity, "unit", "clean", "default"
                )
            )
    for mxcsr in MXCSR_STATES:
        for pixels in PALETTE_TIE_PIXELS:
            for channels in PALETTE_CHANNELS:
                for capacity in PALETTE_CAPACITIES:
                    for special, mapping in PALETTE_TIES:
                        cases.append(
                            _palette_matrix_case(
                                "Z", pixels, channels, capacity, mapping, special, mxcsr
                            )
                        )
    for position in PALETTE_NAN_POSITIONS:
        for pixels in PALETTE_NAN_PIXELS:
            for channels in PALETTE_NAN_CHANNELS:
                for special in PALETTE_NAN_SETS:
                    cases.append(_palette_nan_case(position, pixels, channels, special))
    for mxcsr in MXCSR_STATES:
        for pixels in PALETTE_MEDIAN_PIXELS:
            for channels in PALETTE_MEDIAN_CHANNELS:
                for capacity in PALETTE_MEDIAN_CAPACITIES:
                    cases.append(
                        _palette_matrix_case(
                            "M", pixels, channels, capacity, "zeros", "clean", mxcsr
                        )
                    )
    return cases


def _palette_sweep(count: int, seed: int) -> list[Case]:
    """`count` cases drawn from Generator(PCG64(seed)), ids sweep-<seed>-<n> (D4).

    Per case, in this order: pixels in 1-SWEEP_MAX_PIXELS, channels, capacity, set
    (PALETTE_SETS) and MXCSR state, and a clean src's mapping, unit or tiny.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    cases = []
    for n in range(count):
        pixels = int(rng.integers(1, SWEEP_MAX_PIXELS + 1))
        channels = PALETTE_CHANNELS[int(rng.integers(len(PALETTE_CHANNELS)))]
        capacity = PALETTE_CAPACITIES[int(rng.integers(len(PALETTE_CAPACITIES)))]
        special = PALETTE_SETS[int(rng.integers(len(PALETTE_SETS)))]
        mxcsr = MXCSR_STATES[int(rng.integers(len(MXCSR_STATES)))]
        mapping = (
            ("unit", "tiny")[int(rng.integers(2))] if special == "clean" else "unit"
        )
        cases.append(
            palette_case(
                f"sweep-{seed}-{n}",
                pixels,
                channels,
                capacity,
                special,
                mxcsr,
                mapping=mapping,
            )
        )
    return cases


def dither_form_specials(form: str, count: int, channels: int) -> list[list]:
    """[index, bits] of the form's fixed entries (_DITHER_FORM_ENTRIES) in a (1, count,
    channels) palette, for the entries below count, each value up to the channels."""
    entries = _DITHER_FORM_ENTRIES.get(form, {})
    return [
        [entry * channels + k, bits]
        for entry, values in entries.items()
        if entry < count
        for k, bits in enumerate(values[:channels])
    ]


def dither_case(
    case_id: str,
    entry: str,
    form: str,
    count: int,
    mode: int,
    algorithm: int,
    special: str,
    mxcsr: Mxcsr,
    shape: tuple[int, int, int],
    *,
    seed_key: str | None = None,
    mapping: str = "unit",
) -> Case:
    """One dither case: src (h, w, c) holding the set, on the grid mapping for the grid
    and daz forms and on `mapping` (unit unless given) otherwise; the palette (1, count,
    c) on grid (the grid form) or unit, holding the mixed set (rotated by position 1)
    when src does, then the form's entries (dither_form_specials). Each input is seeded
    by SHA-256(<seed_key or id>-<name>). The arguments: h, w, c and count; mode and
    algorithm (not cn_neighborhood_palette_riemersma's); history and decay (float32
    bits; not cn_neighborhood_dither's)."""
    h, w, c = shape
    key = case_id if seed_key is None else seed_key
    src_mapping = "grid" if form in ("grid", "daz") else mapping
    src = _recipe(shape, _seed(f"{key}-src"), src_mapping, special)
    palette = _recipe(
        (1, count, c),
        _seed(f"{key}-palette"),
        "grid" if form == "grid" else "unit",
        "mixed" if special == "mixed" else "clean",
        position=1,
    )
    palette["specials"] += dither_form_specials(form, count, c)
    args: dict = {"h": h, "w": w, "c": c, "count": count}
    if entry != RIEMERSMA_ENTRY:
        args |= {"mode": mode, "algorithm": algorithm}
    if entry != DITHER_ENTRY:
        args |= {"history": DITHER_HISTORY, "decay": DITHER_DECAY}
    payloads = SPECIALS[special]["payloads"]
    return Case(case_id, entry, args, {"src": src, "palette": palette}, mxcsr, payloads)


_DITHER_KINDS = {
    DITHER_APPLY: "apply",
    DITHER_ENTRY: "dither",
    RIEMERSMA_ENTRY: "riemersma",
}


def _dither_matrix_case(
    group: str,
    entry: str,
    form: str,
    count: int,
    mode: int,
    algorithm: int,
    special: str,
    mxcsr: Mxcsr,
    shape: tuple[int, int, int],
) -> Case:
    parts = [group, _DITHER_KINDS[entry], form, f"n{count}", f"m{mode}a{algorithm}"]
    parts += [special, mxcsr, "x".join(map(str, shape))]
    return dither_case(
        "-".join(parts),
        entry,
        form,
        count,
        mode,
        algorithm,
        special,
        mxcsr,
        shape,
        seed_key="-".join(part for part in parts if part != mxcsr),
    )


def _dither_matrix() -> list[Case]:
    """dither's groups (SP4b D3; ids group-kind-form-nCOUNT-mMODEaALGORITHM-set-mxcsr-
    shape; kind apply, dither or riemersma, whose mode is labelled m3a0).

    On (*DITHER_SHAPE, c) unless named. P (60): cn_neighborhood_palette_apply, every
    DITHER_SIZES count (index i) x DITHER_CHANNELS (index ci) x modes 0 and 3 (index mi;
    mode 3 is history DITHER_HISTORY, decay 0.5), unit palette, the src set ("clean",
    "edges", "nan")[(i + ci + mi) % 3] under MXCSR_STATES[(i + mi) % 2]. D (24): apply,
    mode 2 x algorithms 0-7 (a) x channels (ci) at 16 colors, the set (a + ci) % 3 of
    those under the state a % 2. T (72): apply, the forms grid (40 colors), daz,
    nonfinite and nanfirst (16 colors) x channels x modes 0, 2 (algorithm 0) and 3, clean,
    both states. S (24): apply, every DITHER_SETS set x channels x both states, mode 2
    algorithm 0, 16 colors (the mixed set's palette and src NaNs meet). E (24):
    cn_neighborhood_dither, modes 0 and 2 (algorithm 0) x 16 and 300 colors, and
    cn_neighborhood_palette_riemersma, 16 and 300 colors, each on clean and nan, both
    states, 3 channels. B (1): apply on DITHER_BENCH, mode 2 algorithm 0 (Floyd-
    Steinberg), 16 colors, clean, default.
    """
    cases = []
    sets = ("clean", "edges", "nan")
    for i, count in enumerate(DITHER_SIZES):
        for ci, c in enumerate(DITHER_CHANNELS):
            for mi, mode in enumerate((0, 3)):
                cases.append(
                    _dither_matrix_case(
                        "P",
                        DITHER_APPLY,
                        "unit",
                        count,
                        mode,
                        0,
                        sets[(i + ci + mi) % 3],
                        MXCSR_STATES[(i + mi) % 2],
                        (*DITHER_SHAPE, c),
                    )
                )
    for algorithm in range(8):
        for ci, c in enumerate(DITHER_CHANNELS):
            cases.append(
                _dither_matrix_case(
                    "D",
                    DITHER_APPLY,
                    "unit",
                    16,
                    2,
                    algorithm,
                    sets[(algorithm + ci) % 3],
                    MXCSR_STATES[algorithm % 2],
                    (*DITHER_SHAPE, c),
                )
            )
    for mxcsr in MXCSR_STATES:
        for form in DITHER_FORMS[1:]:
            for c in DITHER_CHANNELS:
                for mode in (0, 2, 3):
                    count = 40 if form == "grid" else 16
                    shape = (*DITHER_SHAPE, c)
                    cases.append(
                        _dither_matrix_case(
                            "T",
                            DITHER_APPLY,
                            form,
                            count,
                            mode,
                            0,
                            "clean",
                            mxcsr,
                            shape,
                        )
                    )
        for special in DITHER_SETS:
            for c in DITHER_CHANNELS:
                shape = (*DITHER_SHAPE, c)
                cases.append(
                    _dither_matrix_case(
                        "S", DITHER_APPLY, "unit", 16, 2, 0, special, mxcsr, shape
                    )
                )
        shape = (*DITHER_SHAPE, 3)
        for special in ("clean", "nan"):
            for count in (16, 300):
                for mode in (0, 2):
                    cases.append(
                        _dither_matrix_case(
                            "E",
                            DITHER_ENTRY,
                            "unit",
                            count,
                            mode,
                            0,
                            special,
                            mxcsr,
                            shape,
                        )
                    )
                cases.append(
                    _dither_matrix_case(
                        "E", RIEMERSMA_ENTRY, "unit", count, 3, 0, special, mxcsr, shape
                    )
                )
    cases.append(
        _dither_matrix_case(
            "B", DITHER_APPLY, "unit", 16, 2, 0, "clean", "default", DITHER_BENCH
        )
    )
    return cases


def _dither_sweep(count: int, seed: int) -> list[Case]:
    """`count` cases drawn from Generator(PCG64(seed)), ids sweep-<seed>-<n> (D4).

    Per case, in this order: the entry; the form (DITHER_FORMS); colors in
    1-SWEEP_MAX_COLORS; channels; the mode (apply: 0, 2 or 3; cn_neighborhood_dither: 0
    or 2), and with mode 2 the algorithm (0-7); the set (DITHER_SETS) and MXCSR state; h
    and w in 1-64, then one case in eight draws h in 65-300; and a clean src's mapping,
    unit or grid (the unit, nonfinite and nanfirst forms). A mixed case takes the unit
    form (the matrix's): its palette holds the set at the index rule.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    entries = KERNEL_ENTRIES["dither"]
    cases = []
    for n in range(count):
        entry = entries[int(rng.integers(len(entries)))]
        form = DITHER_FORMS[int(rng.integers(len(DITHER_FORMS)))]
        colors = int(rng.integers(1, SWEEP_MAX_COLORS + 1))
        c = DITHER_CHANNELS[int(rng.integers(len(DITHER_CHANNELS)))]
        if entry == RIEMERSMA_ENTRY:
            mode = 3
        else:
            modes = (0, 2, 3) if entry == DITHER_APPLY else (0, 2)
            mode = modes[int(rng.integers(len(modes)))]
        algorithm = int(rng.integers(8)) if mode == 2 else 0
        special = DITHER_SETS[int(rng.integers(len(DITHER_SETS)))]
        mxcsr = MXCSR_STATES[int(rng.integers(len(MXCSR_STATES)))]
        h, w = (int(size) for size in rng.integers(1, 65, size=2))
        if int(rng.integers(8)) == 0:
            h = int(rng.integers(65, 301))
        mapping = (
            ("unit", "grid")[int(rng.integers(2))] if special == "clean" else "unit"
        )
        cases.append(
            dither_case(
                f"sweep-{seed}-{n}",
                entry,
                "unit" if special == "mixed" else form,
                colors,
                mode,
                algorithm,
                special,
                mxcsr,
                (h, w, c),
                mapping=mapping,
            )
        )
    return cases


Layer = tuple[tuple[int, int, int], int]  # ((h, w, c), constant)
COMPOSITE_INPUTS = ("base", "overlay")


def _rotated(k: int) -> tuple[str, Mxcsr]:
    """The k-th case's set and MXCSR state in a composite, blend or tail group:
    COMPOSITE_SETS[k % 4] under the state (k // 4) % 2."""
    return COMPOSITE_SETS[k % 4], MXCSR_STATES[k // 4 % 2]


def composite_case(
    case_id: str,
    layers: tuple[Layer, Layer],
    mode: int,
    x: int,
    y: int,
    crop: int,
    special: str,
    mxcsr: Mxcsr,
    *,
    seed_key: str | None = None,
    mapping: str = "unit",
) -> Case:
    """One cn_composite_canvas case: layers are the base's and the overlay's ((h, w, c),
    constant); a constant layer (a Color) must have the other's h and w, as
    dimensions() gives it, and not both are constant. Each input is an image (h, w, c),
    or a constant's c floats, on `mapping` (unit unless given) holding the set, rotated
    by its position (base 0, overlay 1) when mixed, seeded by SHA-256(<seed_key or
    id>-<name>)."""
    (base, base_constant), (overlay, overlay_constant) = layers
    if base_constant and overlay_constant:
        raise ValueError(f"{case_id}: at least one layer must be an image")
    if (base_constant or overlay_constant) and base[:2] != overlay[:2]:
        raise ValueError(f"{case_id}: a constant layer takes the image's h and w")
    key = case_id if seed_key is None else seed_key
    args = {
        "bh": base[0],
        "bw": base[1],
        "bc": base[2],
        "base_constant": base_constant,
        "oh": overlay[0],
        "ow": overlay[1],
        "oc": overlay[2],
        "overlay_constant": overlay_constant,
        "mode": mode,
        "x": x,
        "y": y,
        "crop": crop,
    }
    inputs = {}
    for position, (name, (shape, constant)) in enumerate(
        zip(COMPOSITE_INPUTS, layers, strict=True)
    ):
        held = shape[2:] if constant else shape
        seed = _seed(f"{key}-{name}")
        inputs[name] = _recipe(held, seed, mapping, special, position=position)
    payloads = SPECIALS[special]["payloads"]
    return Case(case_id, COMPOSITE_ENTRY, args, inputs, mxcsr, payloads)


def _signed(value: int) -> str:
    """An offset in a composite id: n for a minus sign (ids split on -)."""
    return f"n{-value}" if value < 0 else str(value)


def _composite_matrix_case(
    group: str,
    kind: str,
    layers: tuple[Layer, Layer],
    mode: int,
    place: tuple[int, int, int],
    special: str,
    mxcsr: Mxcsr,
    mapping: str = "unit",
) -> Case:
    """A matrix case, id group-kind-mMODE-bBCoOC-xXyYcCROP-set-mxcsr-BHxBWoOHxOW (k after
    a constant layer's channels)."""
    (base, base_constant), (overlay, overlay_constant) = layers
    x, y, crop = place
    marks = ["k" if constant else "" for constant in (base_constant, overlay_constant)]
    parts = [group, kind, f"m{mode}", f"b{base[2]}{marks[0]}o{overlay[2]}{marks[1]}"]
    parts += [f"x{_signed(x)}y{_signed(y)}c{crop}", special, mxcsr]
    parts.append(f"{base[0]}x{base[1]}o{overlay[0]}x{overlay[1]}")
    return composite_case(
        "-".join(parts),
        layers,
        mode,
        x,
        y,
        crop,
        special,
        mxcsr,
        seed_key="-".join(part for part in parts if part != mxcsr),
        mapping=mapping,
    )


def _composite_matrix() -> list[Case]:
    """composite's groups (SP4b D3, Task 6; ids as _composite_matrix_case's).

    Layers are two COMPOSITE_SHAPE images unless named. In each group but T and B the
    k-th case takes the set COMPOSITE_SETS[k % 4] under the state (k // 4) % 2. S (72):
    D13's shortcut, every COMPOSITE_PAIRS channel pair p x SHORTCUT_MODES m (k = 8 p +
    m) at a full overlap (x = y = 0), crop (p + m) % 2. N (40): the near misses, every
    NEAR_MISSES kind x NEAR_PAIRS x NEAR_MODES. A (69): every mode x BLEND_PATHS at a
    full overlap. G (40): GEOMETRY_PAIRS x the overlay COMPOSITE_SMALL at every
    COMPOSITE_OFFSETS (x, y) x crop, then at the DISJOINT offsets with crop; mode k % 23.
    C (12): a constant base or overlay x COLOR_PAIRS x COLOR_OFFSETS, mode
    COLOR_MODES[k % 4], crop k % 2 (canvas() ignores crop beside a constant). T (8):
    DAZ_PAIRS x DAZ_MODES x both states at a full overlap, clean on tiny. B (1):
    COMPOSITE_BENCH, the parallel-branches blend (mode 1, a full overlap), clean,
    default.
    """
    cases = []
    h, w = COMPOSITE_SHAPE

    def image(c: int, size: tuple[int, int] = (h, w)) -> Layer:
        return (*size, c), 0

    shortcut = itertools.product(enumerate(COMPOSITE_PAIRS), enumerate(SHORTCUT_MODES))
    for (p, (bc, oc)), (m, mode) in shortcut:
        k = len(SHORTCUT_MODES) * p + m
        layers = (image(bc), image(oc))
        place = (0, 0, (p + m) % 2)
        cases.append(
            _composite_matrix_case("S", "full", layers, mode, place, *_rotated(k))
        )
    near = itertools.product(NEAR_MISSES.items(), NEAR_PAIRS, NEAR_MODES)
    for k, ((kind, form), (bc, oc), mode) in enumerate(near):
        size, x, y, crop, base_constant, overlay_constant = form
        layers = (((h, w, bc), base_constant), ((*size, oc), overlay_constant))
        cases.append(
            _composite_matrix_case("N", kind, layers, mode, (x, y, crop), *_rotated(k))
        )
    paths = itertools.product(range(23), BLEND_PATHS)
    for k, (mode, (bc, oc)) in enumerate(paths):
        layers = (image(bc), image(oc))
        cases.append(
            _composite_matrix_case("A", "full", layers, mode, (0, 0, 0), *_rotated(k))
        )
    places = list(itertools.product(COMPOSITE_OFFSETS, COMPOSITE_OFFSETS, (0, 1)))
    places += [(x, y, 1) for x, y in DISJOINT]
    for k, ((bc, oc), place) in enumerate(itertools.product(GEOMETRY_PAIRS, places)):
        layers = (image(bc), image(oc, COMPOSITE_SMALL))
        cases.append(
            _composite_matrix_case("G", "geo", layers, k % 23, place, *_rotated(k))
        )
    colors = itertools.product(COMPOSITE_INPUTS, COLOR_PAIRS, COLOR_OFFSETS)
    for k, (constant, (bc, oc), (x, y)) in enumerate(colors):
        layers = (
            ((h, w, bc), int(constant == "base")),
            ((h, w, oc), int(constant == "overlay")),
        )
        mode = COLOR_MODES[k % len(COLOR_MODES)]
        cases.append(
            _composite_matrix_case(
                "C", "color", layers, mode, (x, y, k % 2), *_rotated(k)
            )
        )
    for (bc, oc), mode, mxcsr in itertools.product(DAZ_PAIRS, DAZ_MODES, MXCSR_STATES):
        layers = (image(bc), image(oc))
        cases.append(
            _composite_matrix_case(
                "T", "tiny", layers, mode, (0, 0, 0), "clean", mxcsr, mapping="tiny"
            )
        )
    bench = (COMPOSITE_BENCH, 0)
    cases.append(
        _composite_matrix_case(
            "B", "bench", (bench, bench), 1, (0, 0, 0), "clean", "default"
        )
    )
    return cases


def _composite_sweep(count: int, seed: int) -> list[Case]:
    """`count` cases drawn from Generator(PCG64(seed)), ids sweep-<seed>-<n> (D4: a
    geometry by canvas()'s rules with sizes 1-SWEEP_MAX_SIDE).

    Per case, in this order: the layers' kind (0 a constant base, 1 a constant overlay,
    2-7 two images); the base's h and w in 1-SWEEP_MAX_SIDE; whether the overlay
    overlaps fully (one in four); the overlay's h and w in 1-SWEEP_MAX_SIDE, replaced by
    the base's at a full overlap or beside a constant (which takes the image's h and w);
    x in [-ow - 2, bw + 2] and y in [-oh - 2, bh + 2], so an offset can miss the base,
    and both 0 at a full overlap; crop; bc and oc; the mode (0-22); the set
    (COMPOSITE_SETS) and MXCSR state; and a clean case's mapping, unit or tiny.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    cases = []
    for n in range(count):
        kind = int(rng.integers(8))
        bh, bw = (int(size) for size in rng.integers(1, SWEEP_MAX_SIDE + 1, size=2))
        full = int(rng.integers(4)) == 0
        oh, ow = (int(size) for size in rng.integers(1, SWEEP_MAX_SIDE + 1, size=2))
        if full or kind < 2:
            oh, ow = bh, bw
        x = int(rng.integers(-ow - 2, bw + 3))
        y = int(rng.integers(-oh - 2, bh + 3))
        if full:
            x = y = 0
        crop = int(rng.integers(2))
        bc, oc = (COMPOSITE_CHANNELS[int(i)] for i in rng.integers(3, size=2))
        mode = int(rng.integers(23))
        special = COMPOSITE_SETS[int(rng.integers(len(COMPOSITE_SETS)))]
        mxcsr = MXCSR_STATES[int(rng.integers(len(MXCSR_STATES)))]
        mapping = (
            ("unit", "tiny")[int(rng.integers(2))] if special == "clean" else "unit"
        )
        layers = (((bh, bw, bc), int(kind == 0)), ((oh, ow, oc), int(kind == 1)))
        cases.append(
            composite_case(
                f"sweep-{seed}-{n}",
                layers,
                mode,
                x,
                y,
                crop,
                special,
                mxcsr,
                mapping=mapping,
            )
        )
    return cases


def composite_geometry(args: dict) -> tuple[int, ...]:
    """native_composite.canvas()'s geometry for a composite case's args: (h, w,
    channels, base_top, base_left, paste_top, paste_left, overlay_top, overlay_left,
    region_height, region_width, padded), canvas_geometry's fields in order. crop is
    ignored beside a constant layer; a region with no rows or columns is empty at
    (0, 0)."""
    bh, bw, bc = args["bh"], args["bw"], args["bc"]
    oh, ow, oc = args["oh"], args["ow"], args["oc"]
    x, y = args["x"], args["y"]
    crop = bool(args["crop"]) and not (
        args["base_constant"] or args["overlay_constant"]
    )
    left, top = (0, 0) if crop else (max(0, -x), max(0, -y))
    right, bottom = (0, 0) if crop else (max(0, x + ow - bw), max(0, y + oh - bh))
    padded = int(any((left, top, right, bottom)))
    h, w = bh + top + bottom, bw + left + right
    pt, pl = (max(0, y), max(0, x)) if crop else (y + top, x + left)
    if crop:
        rh, rw = max(0, min(y + oh, bh) - pt), max(0, min(x + ow, bw) - pl)
    else:
        rh, rw = oh, ow
    if not rh or not rw:
        rh = rw = pt = pl = 0
    ot, ol = (max(0, -y), max(0, -x)) if crop and rh else (0, 0)
    canvas_channels = 4 if padded else bc
    channels = max(canvas_channels, oc) if rh else canvas_channels
    return (h, w, channels, top, left, pt, pl, ot, ol, rh, rw, padded)


def blend_case(
    case_id: str,
    entry: str,
    mode: int,
    channels: tuple[int, int] | None,
    count: int,
    special: str,
    mxcsr: Mxcsr,
    *,
    seed_key: str | None = None,
    mappings: tuple[str, str] = ("unit", "unit"),
) -> Case:
    """One blend case: cn_blend_images on `count` pixels with (overlay, base) channels
    (inputs (count, c)), or cn_blend_mode on `count` elements (channels None; inputs
    (count,)). Each input holds the set, rotated by its position (overlay 0, base 1)
    when mixed, on its mapping, seeded by SHA-256(<seed_key or id>-<name>)."""
    key = case_id if seed_key is None else seed_key
    if (channels is None) != (entry == BLEND_MODE):
        raise ValueError(f"{case_id}: channels go with {BLEND_IMAGES} alone")
    if channels is None:
        args = {"count": count, "mode": mode}
        shapes = ((count,), (count,))
    else:
        args = {"pixels": count, "oc": channels[0], "bc": channels[1], "mode": mode}
        shapes = ((count, channels[0]), (count, channels[1]))
    inputs = {
        name: _recipe(
            shape, _seed(f"{key}-{name}"), mapping, special, position=position
        )
        for position, (name, shape, mapping) in enumerate(
            zip(BLEND_INPUTS, shapes, mappings, strict=True)
        )
    }
    payloads = SPECIALS[special]["payloads"]
    return Case(case_id, entry, args, inputs, mxcsr, payloads)


def _blend_matrix_case(
    group: str,
    mode: int,
    channels: tuple[int, int] | None,
    count: int,
    special: str,
    mxcsr: Mxcsr,
    mappings: tuple[str, str] = ("unit", "unit"),
) -> Case:
    """A matrix case, id group-kind-mMODE-oOCbBC-set-mxcsr-count (kind images; mode,
    with raw for the channels, for cn_blend_mode)."""
    entry = BLEND_MODE if channels is None else BLEND_IMAGES
    label = "raw" if channels is None else f"o{channels[0]}b{channels[1]}"
    kind = "mode" if channels is None else "images"
    parts = [group, kind, f"m{mode}", label, special, mxcsr, str(count)]
    return blend_case(
        "-".join(parts),
        entry,
        mode,
        channels,
        count,
        special,
        mxcsr,
        seed_key="-".join(part for part in parts if part != mxcsr),
        mappings=mappings,
    )


def _blend_matrix() -> list[Case]:
    """blend's groups (SP4b D3, Task 6 Step 6; ids as _blend_matrix_case's).

    In each group but T the k-th case takes _rotated(k)'s set and state. M (207):
    cn_blend_images, every mode x BLEND_PAIRS, the k-th at SMALL_COUNTS[k % 17] pixels.
    S (68): SIMD_MODES x SIMD_PAIRS x pixels 1-17 (tails alone, one or two 8-pixel
    blocks and a tail). W (4): SIMD_MODES x SIMD_PAIRS at WIDE_COUNT pixels (4 chunks
    whose element starts lie off the 8-lane grid). T (8): SIMD_MODES x SIMD_PAIRS x both
    states at BLEND_DAZ_PIXELS, clean, the overlay on tiny (denormals and the smallest
    normals) and the base on unit. R (25): cn_blend_mode, every mode, the k-th at
    SMALL_COUNTS[k % 17] elements, then SIMD_MODES at WIDE_COUNT.
    """
    cases = []
    pairs = itertools.product(range(23), BLEND_PAIRS)
    for k, (mode, channels) in enumerate(pairs):
        count = SMALL_COUNTS[k % len(SMALL_COUNTS)]
        cases.append(_blend_matrix_case("M", mode, channels, count, *_rotated(k)))
    simd = list(itertools.product(SIMD_MODES, SIMD_PAIRS))
    for k, ((mode, channels), count) in enumerate(
        itertools.product(simd, SMALL_COUNTS)
    ):
        cases.append(_blend_matrix_case("S", mode, channels, count, *_rotated(k)))
    for k, (mode, channels) in enumerate(simd):
        cases.append(_blend_matrix_case("W", mode, channels, WIDE_COUNT, *_rotated(k)))
    for (mode, channels), mxcsr in itertools.product(simd, MXCSR_STATES):
        cases.append(
            _blend_matrix_case(
                "T",
                mode,
                channels,
                BLEND_DAZ_PIXELS,
                "clean",
                mxcsr,
                mappings=("tiny", "unit"),
            )
        )
    modes = [
        (mode, SMALL_COUNTS[k % len(SMALL_COUNTS)]) for k, mode in enumerate(range(23))
    ]
    modes += [(mode, WIDE_COUNT) for mode in SIMD_MODES]
    for k, (mode, count) in enumerate(modes):
        cases.append(_blend_matrix_case("R", mode, None, count, *_rotated(k)))
    return cases


def _blend_sweep(count: int, seed: int) -> list[Case]:
    """`count` cases drawn from Generator(PCG64(seed)), ids sweep-<seed>-<n> (D4).

    Per case, in this order: the entry; for cn_blend_images whether it takes the AVX2
    unit's 8-lane form (one in four), then the mode and (overlay, base) channels from
    SIMD_MODES and SIMD_PAIRS, else from 0-22 and BLEND_PAIRS, and pixels in
    1-(SWEEP_MAX_COUNT // the larger channel count); for cn_blend_mode the mode (0-22)
    and elements in 1-SWEEP_MAX_COUNT; then the set (BLEND_SETS) and MXCSR state, and a
    clean case's mapping, unit or tiny (both inputs).
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    entries = KERNEL_ENTRIES["blend"]
    cases = []
    for n in range(count):
        entry = entries[int(rng.integers(len(entries)))]
        channels = None
        if entry == BLEND_IMAGES:
            if int(rng.integers(4)) == 0:
                mode = SIMD_MODES[int(rng.integers(len(SIMD_MODES)))]
                channels = SIMD_PAIRS[int(rng.integers(len(SIMD_PAIRS)))]
            else:
                mode = int(rng.integers(23))
                channels = BLEND_PAIRS[int(rng.integers(len(BLEND_PAIRS)))]
            size = int(rng.integers(1, SWEEP_MAX_COUNT // max(channels) + 1))
        else:
            mode = int(rng.integers(23))
            size = int(rng.integers(1, SWEEP_MAX_COUNT + 1))
        special = BLEND_SETS[int(rng.integers(len(BLEND_SETS)))]
        mxcsr = MXCSR_STATES[int(rng.integers(len(MXCSR_STATES)))]
        mapping = (
            ("unit", "tiny")[int(rng.integers(2))] if special == "clean" else "unit"
        )
        cases.append(
            blend_case(
                f"sweep-{seed}-{n}",
                entry,
                mode,
                channels,
                size,
                special,
                mxcsr,
                mappings=(mapping, mapping),
            )
        )
    return cases


def tail_inputs(entry: str, args: dict) -> list[tuple[str, tuple[int, ...], str, str]]:
    """A tail entry's input arrays in call order: (name, shape, dtype, mapping). Every
    double input and the caption raster are bits; store's dst is in-out."""
    if entry == TILE_ENTRY:
        return [("src", (args["h"], args["w"], args["c"]), "float32", "unit")]
    if entry == STORE_ENTRY:
        dst = (args["oh"], args["ow"], args["c"])
        return [
            ("src", (args["dh"], args["dw"]), "float64", "bits"),
            ("dst", dst, "float32", "unit"),
        ]
    if entry == MULTIPLY_ENTRY:
        return [(name, (args["h"], args["w"]), "float64", "bits") for name in "ab"]
    if entry == NORMAL_ENTRY:
        names = ("dx", "dy", "alpha")[: 2 + args["alpha"]]
        return [
            (
                name,
                (args["pixels"],),
                "float32",
                "unit" if name == "alpha" else "signed",
            )
            for name in names
        ]
    image = (args["height"], args["width"], args["channels"])
    return [
        ("image", image, "float32", "unit"),
        ("caption", (args["caption_height"], args["width"]), "uint8", "bits"),
    ]


def tail_case(
    case_id: str,
    entry: str,
    args: dict,
    special: str,
    mxcsr: Mxcsr,
    *,
    seed_key: str | None = None,
    mapping: str | None = None,
) -> Case:
    """One tail case: tail_inputs()'s arrays, each seeded by SHA-256(<seed_key or
    id>-<name>). special is a set, which every float32 input holds (rotated by its
    position when mixed) on its mapping, or on `mapping` when given (a clean case's
    tiny); or bits, the label of the double entries, whose float32 input (store's dst)
    holds no set. Payloads: the set's; bits: single for the store (each output narrows
    one input) and mixed for the multiply (two NaN inputs can meet in a pair)."""
    key = case_id if seed_key is None else seed_key
    inputs = {}
    for position, (name, shape, dtype, own) in enumerate(tail_inputs(entry, args)):
        floats = dtype == "float32"
        held = special if floats and special in SPECIALS else "clean"
        chosen = mapping if floats and mapping is not None else own
        seed = _seed(f"{key}-{name}")
        inputs[name] = _recipe(shape, seed, chosen, held, dtype, position)
    if special in SPECIALS:
        payloads = SPECIALS[special]["payloads"]
    else:
        payloads = "mixed" if entry == MULTIPLY_ENTRY else "single"
    return Case(case_id, entry, args, inputs, mxcsr, payloads)


def tail_label(entry: str, args: dict) -> tuple[str, str]:
    """A tail case's form and size id parts."""
    a = args
    if entry == TILE_ENTRY:
        form = f"b{a['border']}p{a['padding']}y{a['y']}x{a['x']}t{a['th']}x{a['tw']}"
        size = f"{a['h']}x{a['w']}x{a['c']}k{a['kh']}x{a['kw']}d{a['dh']}x{a['dw']}"
        return f"{form}c{a['channel']}", size
    if entry == STORE_ENTRY:
        form = f"y{a['y']}x{a['x']}t{a['th']}x{a['tw']}c{a['channel']}"
        return form, f"{a['oh']}x{a['ow']}x{a['c']}d{a['dh']}x{a['dw']}"
    if entry == MULTIPLY_ENTRY:
        return "plane", f"{a['h']}x{a['w']}"
    if entry == NORMAL_ENTRY:
        form = f"c{a['channels']}r{a['invert_r']}g{a['invert_g']}a{a['alpha']}"
        return form, str(a["pixels"])
    form = f"t{a['top']}c{a['channels']}k{a['caption_height']}"
    return form, f"{a['height']}x{a['width']}"


def tile_origin(oh: int, ow: int, block: tuple[int, int], tile: str) -> tuple[int, int]:
    """filter2d's first, second (inner) or last tile origin (y, x) in an oh x ow image
    cut in blocks (bh, bw)."""
    bh, bw = block
    if tile == "first":
        return 0, 0
    if tile == "inner":
        return bh, bw
    return (oh - 1) // bh * bh, (ow - 1) // bw * bw


def tile_args(
    shape: tuple[int, int, int],
    padding: int,
    kernel: tuple[int, int],
    block: tuple[int, int],
    tile: str,
    channel: int,
    border: int,
    plane: tuple[int, int] | None = None,
) -> dict:
    """cn_spectral_tile's args for filter2d's tile (tile_origin()) of the padded image in
    blocks (bh, bw), cut at the image's end, in the plane (dh, dw), by default (bh + kh,
    bw + kw)."""
    h, w, c = shape
    kh, kw = kernel
    oh, ow = h + 2 * padding, w + 2 * padding
    y, x = tile_origin(oh, ow, block, tile)
    dh, dw = plane or (block[0] + kh, block[1] + kw)
    return {
        "h": h,
        "w": w,
        "c": c,
        "padding": padding,
        "y": y,
        "x": x,
        "th": min(block[0], oh - y),
        "tw": min(block[1], ow - x),
        "kh": kh,
        "kw": kw,
        "dh": dh,
        "dw": dw,
        "channel": channel,
        "border": border,
    }


def store_args(
    shape: tuple[int, int, int],
    block: tuple[int, int],
    tile: str,
    channel: int,
    plane: tuple[int, int],
) -> dict:
    """cn_spectral_store's args for filter2d's tile (tile_origin()) of an (oh, ow, c)
    result in blocks (bh, bw), from the plane (dh, dw)."""
    oh, ow, c = shape
    y, x = tile_origin(oh, ow, block, tile)
    th, tw = min(block[0], oh - y), min(block[1], ow - x)
    dh, dw = plane
    return {
        "oh": oh,
        "ow": ow,
        "c": c,
        "y": y,
        "x": x,
        "th": th,
        "tw": tw,
        "dh": dh,
        "dw": dw,
        "channel": channel,
    }


def _tail_matrix_case(
    group: str,
    entry: str,
    args: dict,
    special: str,
    mxcsr: Mxcsr,
    mapping: str | None = None,
) -> Case:
    """A matrix case, id group-kind-form-set-mxcsr-size (tail_label())."""
    form, size = tail_label(entry, args)
    parts = [group, TAIL_KINDS[entry], form, special, mxcsr, size]
    return tail_case(
        "-".join(parts),
        entry,
        args,
        special,
        mxcsr,
        seed_key="-".join(part for part in parts if part != mxcsr),
        mapping=mapping,
    )


def _tail_matrix() -> list[Case]:
    """tail's groups (SP4b D3, Task 8; ids as _tail_matrix_case's). _rotated(k) gives
    the k-th case's set and state where named.

    cn_spectral_tile. G (30): SPECTRAL_SHAPE, every padding x border x SPECTRAL_TILES
    tile, channel k % 3, _rotated(k). R (20): SPECTRAL_REACH x every border x padding,
    the whole padded image as one tile under REACH_KERNEL, the last channel,
    _rotated(k). D (4): G's first tile at both paddings, reflect101, channel 1 (which
    holds the edges set's denormal 0x00000001), edges, both states. T (2): as D at
    padding 0, clean on tiny, both states. W (3): TILE_WIDE. B (1): SPECTRAL_BENCH.
    cn_spectral_store, the plane on bits. G (12): SPECTRAL_SHAPE's height and width x
    STORE_CHANNELS channels x SPECTRAL_TILES from SPECTRAL_BLOCK's plane, channel
    k % c, both states. D (4): STORE_DAZ, whole tiles from a plane 5 rows and 7 columns
    larger, the last channel, both states. W (1): STORE_WIDE likewise. B (1):
    SPECTRAL_BENCH's tile.
    cn_spectral_multiply, a and b on bits. S (16): MULTIPLY_HEIGHTS x MULTIPLY_WIDTHS,
    the state k % 2. D (4): MULTIPLY_DAZ, both states. W (1): MULTIPLY_WIDE. B (1):
    SPECTRAL_BENCH's plane.
    cn_normal_output. F (48): channels 3/4 x invert_r x invert_g x alpha x NORMAL_SETS
    (fastest), at SMALL_COUNTS[k % 17] pixels, the state (k // 3) % 2. D (4): channels
    3 without and 4 with alpha, 17 pixels, edges, both states. W (2): the same forms at
    WIDE_COUNT pixels, edges. B (1): NORMAL_BENCH pixels, 3 channels, no alpha or
    inversion, clean.
    cn_caption_compose, the raster on bits. F (18): top x channels 1/3/4 x
    CAPTION_SIZES, _rotated(k). D (4): top 0, channels 3 and 4, CAPTION_SIZES[2],
    edges, both states. W (1): CAPTION_WIDE, top 1, 3 channels. B (1): CAPTION_BENCH,
    top 0, 3 channels. W and B are clean at the default MXCSR.
    """
    cases = []
    tiles = itertools.product(SPECTRAL_PADDINGS, SPECTRAL_BORDERS, SPECTRAL_TILES)
    for k, (padding, border, tile) in enumerate(tiles):
        args = tile_args(
            SPECTRAL_SHAPE,
            padding,
            SPECTRAL_KERNEL,
            SPECTRAL_BLOCK,
            tile,
            k % 3,
            border,
        )
        cases.append(_tail_matrix_case("G", TILE_ENTRY, args, *_rotated(k)))
    reach = itertools.product(SPECTRAL_REACH, SPECTRAL_BORDERS, SPECTRAL_PADDINGS)
    for k, (shape, border, padding) in enumerate(reach):
        h, w, c = shape
        block = (h + 2 * padding, w + 2 * padding)
        args = tile_args(shape, padding, REACH_KERNEL, block, "first", c - 1, border)
        cases.append(_tail_matrix_case("R", TILE_ENTRY, args, *_rotated(k)))
    for group, special, mapping, paddings in (
        ("D", "edges", None, SPECTRAL_PADDINGS),
        ("T", "clean", "tiny", (0,)),
    ):
        for padding, mxcsr in itertools.product(paddings, MXCSR_STATES):
            args = tile_args(
                SPECTRAL_SHAPE, padding, SPECTRAL_KERNEL, SPECTRAL_BLOCK, "first", 1, 4
            )
            cases.append(
                _tail_matrix_case(group, TILE_ENTRY, args, special, mxcsr, mapping)
            )
    for shape, padding, kernel, border, channel in TILE_WIDE:
        block = (shape[0] + 2 * padding, shape[1] + 2 * padding)
        args = tile_args(shape, padding, kernel, block, "first", channel, border)
        cases.append(_tail_matrix_case("W", TILE_ENTRY, args, "clean", "default"))
    shape, kernel, plane = SPECTRAL_BENCH
    args = tile_args(shape, 0, kernel, shape[:2], "first", 0, 4, plane)
    cases.append(_tail_matrix_case("B", TILE_ENTRY, args, "clean", "default"))
    h, w, _ = SPECTRAL_SHAPE
    store_plane = (
        SPECTRAL_BLOCK[0] + SPECTRAL_KERNEL[0],
        SPECTRAL_BLOCK[1] + SPECTRAL_KERNEL[1],
    )
    stores = itertools.product(STORE_CHANNELS, SPECTRAL_TILES, MXCSR_STATES)
    for k, (c, tile, mxcsr) in enumerate(stores):
        args = store_args((h, w, c), SPECTRAL_BLOCK, tile, k % c, store_plane)
        cases.append(_tail_matrix_case("G", STORE_ENTRY, args, "bits", mxcsr))
    wholes: list[tuple[str, tuple[int, int, int], Mxcsr]] = [
        ("D", d, mxcsr) for d in STORE_DAZ for mxcsr in MXCSR_STATES
    ]
    wholes.append(("W", STORE_WIDE, "default"))
    for group, shape, mxcsr in wholes:
        plane = (shape[0] + SPECTRAL_KERNEL[0], shape[1] + SPECTRAL_KERNEL[1])
        args = store_args(shape, shape[:2], "first", shape[2] - 1, plane)
        cases.append(_tail_matrix_case(group, STORE_ENTRY, args, "bits", mxcsr))
    shape, _, plane = SPECTRAL_BENCH
    args = store_args(shape, shape[:2], "first", 0, plane)
    cases.append(_tail_matrix_case("B", STORE_ENTRY, args, "bits", "default"))
    sizes = itertools.product(MULTIPLY_HEIGHTS, MULTIPLY_WIDTHS)
    for k, (h, w) in enumerate(sizes):
        args = {"h": h, "w": w}
        cases.append(
            _tail_matrix_case("S", MULTIPLY_ENTRY, args, "bits", MXCSR_STATES[k % 2])
        )
    for (h, w), mxcsr in itertools.product(MULTIPLY_DAZ, MXCSR_STATES):
        args = {"h": h, "w": w}
        cases.append(_tail_matrix_case("D", MULTIPLY_ENTRY, args, "bits", mxcsr))
    for group, (h, w) in (("W", MULTIPLY_WIDE), ("B", SPECTRAL_BENCH[2])):
        args = {"h": h, "w": w}
        cases.append(_tail_matrix_case(group, MULTIPLY_ENTRY, args, "bits", "default"))

    def normal(pixels: int, form: tuple[int, int, int, int]) -> dict:
        channels, invert_r, invert_g, alpha = form
        return {
            "pixels": pixels,
            "invert_r": invert_r,
            "invert_g": invert_g,
            "channels": channels,
            "alpha": alpha,
        }

    forms = itertools.product((3, 4), (0, 1), (0, 1), (0, 1), NORMAL_SETS)
    for k, (channels, invert_r, invert_g, alpha, special) in enumerate(forms):
        form = (channels, invert_r, invert_g, alpha)
        args = normal(SMALL_COUNTS[k % len(SMALL_COUNTS)], form)
        mxcsr = MXCSR_STATES[k // len(NORMAL_SETS) % 2]
        cases.append(_tail_matrix_case("F", NORMAL_ENTRY, args, special, mxcsr))
    alpha_forms = ((3, 0, 0, 0), (4, 0, 0, 1))
    for form, mxcsr in itertools.product(alpha_forms, MXCSR_STATES):
        args = normal(SMALL_COUNTS[-1], form)
        cases.append(_tail_matrix_case("D", NORMAL_ENTRY, args, "edges", mxcsr))
    for form in alpha_forms:
        args = normal(WIDE_COUNT, form)
        cases.append(_tail_matrix_case("W", NORMAL_ENTRY, args, "edges", "default"))
    args = normal(NORMAL_BENCH, (3, 0, 0, 0))
    cases.append(_tail_matrix_case("B", NORMAL_ENTRY, args, "clean", "default"))

    def caption(size: tuple[int, int, int], channels: int, top: int) -> dict:
        height, width, caption_height = size
        return {
            "height": height,
            "width": width,
            "channels": channels,
            "caption_height": caption_height,
            "top": top,
        }

    captions = itertools.product((0, 1), (1, 3, 4), CAPTION_SIZES)
    for k, (top, channels, size) in enumerate(captions):
        args = caption(size, channels, top)
        cases.append(_tail_matrix_case("F", CAPTION_ENTRY, args, *_rotated(k)))
    for channels, mxcsr in itertools.product((3, 4), MXCSR_STATES):
        args = caption(CAPTION_SIZES[2], channels, 0)
        cases.append(_tail_matrix_case("D", CAPTION_ENTRY, args, "edges", mxcsr))
    for group, size, top in (("W", CAPTION_WIDE, 1), ("B", CAPTION_BENCH, 0)):
        args = caption(size, 3, top)
        cases.append(_tail_matrix_case(group, CAPTION_ENTRY, args, "clean", "default"))
    return cases


def _sweep_height(rng: np.random.Generator) -> int:
    """An image height by D4: 1-64, then one case in eight redraws it in 65-300."""
    height = int(rng.integers(1, 65))
    if int(rng.integers(8)) == 0:
        height = int(rng.integers(65, 301))
    return height


def _tail_sweep(count: int, seed: int) -> list[Case]:
    """`count` cases drawn from Generator(PCG64(seed)), ids sweep-<seed>-<n> (D4).

    Per case, in this order: the entry; then cn_spectral_tile: h (_sweep_height), w in
    1-64, c (1/3/4), padding (0/2), kh and kw in 1-33, the tile's y in [0, oh) and x in
    [0, ow), th in [1, oh - y] and tw in [1, ow - x], the plane's extra rows and columns
    in 0-2 beyond (th + kh - 1, tw + kw - 1), the channel, border (0-4) and set
    (COMPOSITE_SETS); cn_spectral_store: oh (_sweep_height), ow in 1-64, c, the tile as
    the tile's, the plane's extra rows and columns in 0-3 beyond (th, tw) and the
    channel; cn_spectral_multiply: h (_sweep_height) and w in 1-64; cn_normal_output:
    pixels in 1-140,000, channels (3/4), invert_r, invert_g, alpha and set
    (NORMAL_SETS); cn_caption_compose: height (_sweep_height), width and caption_height
    in 1-64, channels (1/3/4), top and set (COMPOSITE_SETS). Then the MXCSR state, and a
    clean case's mapping, unit or tiny.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    entries = KERNEL_ENTRIES["tail"]
    cases = []

    def draw(values: tuple) -> Any:
        return values[int(rng.integers(len(values)))]

    def tile(oh: int, ow: int) -> tuple[int, int, int, int]:
        y, x = int(rng.integers(oh)), int(rng.integers(ow))
        return y, x, int(rng.integers(1, oh - y + 1)), int(rng.integers(1, ow - x + 1))

    for n in range(count):
        entry = draw(entries)
        special = "bits"
        if entry == TILE_ENTRY:
            h, w = _sweep_height(rng), int(rng.integers(1, 65))
            c, padding = draw((1, 3, 4)), draw(SPECTRAL_PADDINGS)
            kh, kw = (int(size) for size in rng.integers(1, 34, size=2))
            y, x, th, tw = tile(h + 2 * padding, w + 2 * padding)
            extra = rng.integers(3, size=2)
            args = {
                "h": h,
                "w": w,
                "c": c,
                "padding": padding,
                "y": y,
                "x": x,
                "th": th,
                "tw": tw,
                "kh": kh,
                "kw": kw,
                "dh": th + kh - 1 + int(extra[0]),
                "dw": tw + kw - 1 + int(extra[1]),
                "channel": int(rng.integers(c)),
                "border": draw(SPECTRAL_BORDERS),
            }
            special = draw(COMPOSITE_SETS)
        elif entry == STORE_ENTRY:
            oh, ow, c = _sweep_height(rng), int(rng.integers(1, 65)), draw((1, 3, 4))
            y, x, th, tw = tile(oh, ow)
            extra = rng.integers(4, size=2)
            args = {
                "oh": oh,
                "ow": ow,
                "c": c,
                "y": y,
                "x": x,
                "th": th,
                "tw": tw,
                "dh": th + int(extra[0]),
                "dw": tw + int(extra[1]),
                "channel": int(rng.integers(c)),
            }
        elif entry == MULTIPLY_ENTRY:
            args = {"h": _sweep_height(rng), "w": int(rng.integers(1, 65))}
        elif entry == NORMAL_ENTRY:
            pixels, channels = int(rng.integers(1, SWEEP_MAX_COUNT + 1)), draw((3, 4))
            invert_r, invert_g, alpha = (int(flag) for flag in rng.integers(2, size=3))
            args = {
                "pixels": pixels,
                "invert_r": invert_r,
                "invert_g": invert_g,
                "channels": channels,
                "alpha": alpha,
            }
            special = draw(NORMAL_SETS)
        else:
            height = _sweep_height(rng)
            width, caption_height = (int(size) for size in rng.integers(1, 65, size=2))
            args = {
                "height": height,
                "width": width,
                "channels": draw((1, 3, 4)),
                "caption_height": caption_height,
                "top": int(rng.integers(2)),
            }
            special = draw(COMPOSITE_SETS)
        mxcsr = draw(MXCSR_STATES)
        mapping = draw(("unit", "tiny")) if special == "clean" else None
        cases.append(
            tail_case(f"sweep-{seed}-{n}", entry, args, special, mxcsr, mapping=mapping)
        )
    return cases


def _tile_forms(
    gaussian: tuple[float, float],
    box: tuple[float, float],
    borders: tuple[int, ...] = (1, 2, 4),
    lanes: tuple[int, ...] = (4, 8),
) -> list[tuple[str, tuple[float, float], int | None, int]]:
    """(entry, form, border, lanes): per lanes value, the Gaussian form at each
    border, then the box form."""
    found: list[tuple[str, tuple[float, float], int | None, int]] = []
    for mirror in lanes:
        found += [("cn_gaussian_f32", gaussian, border, mirror) for border in borders]
        found.append(("cn_box_separable", box, None, mirror))
    return found


def _tiles_matrix() -> list[Case]:
    """separable_tiles' groups (SP4b D9, review focus 1; ids as matrix()'s).

    H (128): each TILE_HEIGHTS height at width 13 with 3 channels, the Gaussian
    (21, 11) at borders 1/2/4 and the box (3.25, 4), lanes 4/8. G (14): TILE_PLANE,
    the same forms plus the Gaussian (49, 49) at borders 1/2/4, lanes 4/8. F (16):
    each TILE_FALLBACKS shape with its Gaussian form at borders 1/2/4 and its box
    form, lanes 4/8. N (6): each TILE_UNTABLED pair at borders 1/2/4, lanes 8. W (96):
    height 7, widths 18-33 (Task 9's 16-lane groups), channels 1/3/4, lanes 8, the
    Gaussian (21, 11) at border (1, 2, 4)[w % 3] and the box (1.5, 2.5). P (3):
    TILE_BENCH with the bench's Gaussians (21, 11), (25, 25) and (49, 49) at border 2
    (native_gaussian's cv2.BORDER_REFLECT), lanes 8. All of these clean at the default
    MXCSR. S (16): (65, 41, 3), the Gaussian (21, 11) at border 4 and the box
    (3.25, 4), lanes 8, every TILE_SETS set, both MXCSR states. E (16): (65, 41, 3)
    with src in the denormal range (tiny), the H forms at lanes 4/8, both states.
    """
    cases = []
    for h in TILE_HEIGHTS:
        for entry, form, place, mirror in _tile_forms((21, 11), (3.25, 4.0)):
            cases.append(
                _matrix_case(
                    "H", entry, form, place, mirror, (h, 13, 3), "clean", "default"
                )
            )
    plane = _tile_forms((21, 11), (3.25, 4.0))
    plane += [
        ("cn_gaussian_f32", (49, 49), border, mirror)
        for mirror in (4, 8)
        for border in (1, 2, 4)
    ]
    for entry, form, place, mirror in plane:
        cases.append(
            _matrix_case(
                "G", entry, form, place, mirror, TILE_PLANE, "clean", "default"
            )
        )
    for shape, gaussian, box in TILE_FALLBACKS:
        for entry, form, place, mirror in _tile_forms(gaussian, box):
            cases.append(
                _matrix_case("F", entry, form, place, mirror, shape, "clean", "default")
            )
    for shape, form in TILE_UNTABLED:
        for border in (1, 2, 4):
            cases.append(
                _matrix_case(
                    "N", "cn_gaussian_f32", form, border, 8, shape, "clean", "default"
                )
            )
    for w in range(18, 34):
        for c in (1, 3, 4):
            for entry, form, place, mirror in (
                ("cn_gaussian_f32", (21, 11), (1, 2, 4)[w % 3], 8),
                ("cn_box_separable", (1.5, 2.5), None, 8),
            ):
                cases.append(
                    _matrix_case(
                        "W", entry, form, place, mirror, (7, w, c), "clean", "default"
                    )
                )
    for form in ((21, 11), (25, 25), (49, 49)):
        cases.append(
            _matrix_case(
                "P", "cn_gaussian_f32", form, 2, 8, TILE_BENCH, "clean", "default"
            )
        )
    for mxcsr in MXCSR_STATES:
        for entry, form, place, mirror in _tile_forms(
            (21, 11), (3.25, 4.0), borders=(4,), lanes=(8,)
        ):
            for special in TILE_SETS:
                cases.append(
                    _matrix_case(
                        "S", entry, form, place, mirror, (65, 41, 3), special, mxcsr
                    )
                )
        for entry, form, place, mirror in _tile_forms((21, 11), (3.25, 4.0)):
            cases.append(
                _matrix_case(
                    "E", entry, form, place, mirror, (65, 41, 3), "clean", mxcsr
                )
            )
    return cases


def _tiles_sweep(count: int, seed: int) -> list[Case]:
    """`count` cases drawn from Generator(PCG64(seed)), ids sweep-<seed>-<n> (D4).

    Per case, in this order: the entry, its form from TILE_FORMS, lanes, border (the
    Gaussian), set and MXCSR state; h and w in 1-64, then one case in eight draws h in
    65-300; c 1, 3 or 4; a clean src's mapping, unit or tiny. The fallback and
    untabled forms are not drawn: their radii are set by their groups' shapes.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    entries = KERNEL_ENTRIES["separable_tiles"]
    specials = SP4A_SETS
    cases = []
    for n in range(count):
        entry = entries[int(rng.integers(len(entries)))]
        spec = ENTRIES[entry]
        forms = TILE_FORMS[entry]
        form = forms[int(rng.integers(len(forms)))]
        mirror = spec.mirrors[int(rng.integers(len(spec.mirrors)))]
        place = spec.places[int(rng.integers(len(spec.places)))] if spec.place else None
        special = specials[int(rng.integers(len(specials)))]
        mxcsr = MXCSR_STATES[int(rng.integers(len(MXCSR_STATES)))]
        h, w = (int(size) for size in rng.integers(1, 65, size=2))
        if int(rng.integers(8)) == 0:
            h = int(rng.integers(65, 301))
        c = (1, 3, 4)[int(rng.integers(3))]
        mapping = (
            ("unit", "tiny")[int(rng.integers(2))] if special == "clean" else "unit"
        )
        cases.append(
            case(
                f"sweep-{seed}-{n}",
                entry,
                form,
                place,
                mirror,
                (h, w, c),
                special,
                mxcsr,
                src_mapping=mapping,
            )
        )
    return cases


def matrix(kernel: str) -> list[Case]:
    """The kernel's fixed cases, groups A to E in order (conversions, lens,
    convolution_border, separable_tiles, lens_power, morphology, resample, palette,
    dither, composite, blend and tail: see _conversion_matrix, _lens_matrix,
    _border_matrix, _tiles_matrix, _power_matrix, _morph_matrix, _resample_matrix,
    _palette_matrix, _dither_matrix, _composite_matrix, _blend_matrix and _tail_matrix).

    A: S1, every form, border or padding and mirror value, set nan, default MXCSR;
    a convolution kernel holds no specials but the (5,5) zero taps, so every
    output shows src's NaN propagation. B: S1, every form at border 4 / padding 1,
    both MXCSR states: the wide mirror value with every set, the narrow one with
    clean and mixed. C: S2 (height 7, widths 1-17, channels 1/3/4), one form,
    wide mirror, clean. D (separable): S3 (heights 1-17 at 9x3), Gaussian (1,3),
    both lanes, clean. E: S1, every form at border 4 / padding 1, both mirror
    values, clean, both MXCSR states, src in the denormal range (mapping tiny), so
    every (entry, form, mirror value) has a DAZ-sensitive golden (review I1).
    """
    if kernel in CONVERSION_KERNELS:
        return _conversion_matrix(kernel)
    if kernel == "convolution_border":
        return _border_matrix()
    if kernel == "lens":
        return _lens_matrix()
    if kernel == "separable_tiles":
        return _tiles_matrix()
    if kernel == "lens_power":
        return _power_matrix()
    if kernel == "morphology":
        return _morph_matrix()
    if kernel == "resample":
        return _resample_matrix()
    if kernel == "palette":
        return _palette_matrix()
    if kernel == "dither":
        return _dither_matrix()
    if kernel == "composite":
        return _composite_matrix()
    if kernel == "blend":
        return _blend_matrix()
    if kernel == "tail":
        return _tail_matrix()
    entries = KERNEL_ENTRIES[kernel]
    s1 = _S1[kernel]
    cases = []
    for entry in entries:
        spec = ENTRIES[entry]
        for form in spec.forms:
            for place in spec.places or (None,):
                for mirror in spec.mirrors:
                    cases.append(
                        _matrix_case(
                            "A", entry, form, place, mirror, s1, "nan", "default"
                        )
                    )
    for mxcsr in MXCSR_STATES:
        for entry in entries:
            spec = ENTRIES[entry]
            place = spec.place_b if spec.place else None
            narrow, wide = spec.mirrors
            for form in spec.forms:
                for special in SP4A_SETS:
                    cases.append(
                        _matrix_case("B", entry, form, place, wide, s1, special, mxcsr)
                    )
                for special in ("clean", "mixed"):
                    cases.append(
                        _matrix_case(
                            "B", entry, form, place, narrow, s1, special, mxcsr
                        )
                    )
    for entry in entries:
        spec = ENTRIES[entry]
        place = spec.place_b if spec.place else None
        for w in range(1, 18):
            for c in (1, 3, 4):
                cases.append(
                    _matrix_case(
                        "C",
                        entry,
                        _C_FORMS[entry],
                        place,
                        spec.mirrors[1],
                        (7, w, c),
                        "clean",
                        "default",
                    )
                )
    if kernel == "separable":
        for h in range(1, 18):
            for lanes in (4, 8):
                cases.append(
                    _matrix_case(
                        "D",
                        "cn_gaussian_f32",
                        (1, 3),
                        4,
                        lanes,
                        (h, 9, 3),
                        "clean",
                        "default",
                    )
                )
    for entry in entries:
        spec = ENTRIES[entry]
        place = spec.place_b if spec.place else None
        for form in spec.forms:
            for mirror in spec.mirrors:
                for mxcsr in MXCSR_STATES:
                    cases.append(
                        _matrix_case(
                            "E", entry, form, place, mirror, s1, "clean", mxcsr
                        )
                    )
    return cases


def sweep_cases(kernel: str, count: int, seed: int) -> list[Case]:
    """`count` cases drawn from Generator(PCG64(seed)), ids sweep-<seed>-<n>.

    Per case, in this order: the entry, form, mirror value, border or padding (if
    the entry has one), special set and MXCSR state, each uniformly from the
    kernel's matrix; h and w uniformly in 1-64; c from 1/3/4; for the clean set,
    the src mapping, unit or tiny (denormal range), so a sweep reaches DAZ|FTZ.
    Conversion kernels, convolution_border, lens, separable_tiles, lens_power,
    morphology, resample, palette, dither, composite, blend and tail: _conversion_sweep,
    _border_sweep, _lens_sweep, _tiles_sweep, _power_sweep, _morph_sweep,
    _resample_sweep, _palette_sweep, _dither_sweep, _composite_sweep, _blend_sweep and
    _tail_sweep.
    """
    if kernel in CONVERSION_KERNELS:
        return _conversion_sweep(kernel, count, seed)
    if kernel == "convolution_border":
        return _border_sweep(count, seed)
    if kernel == "lens":
        return _lens_sweep(count, seed)
    if kernel == "separable_tiles":
        return _tiles_sweep(count, seed)
    if kernel == "lens_power":
        return _power_sweep(count, seed)
    if kernel == "morphology":
        return _morph_sweep(count, seed)
    if kernel == "resample":
        return _resample_sweep(count, seed)
    if kernel == "palette":
        return _palette_sweep(count, seed)
    if kernel == "dither":
        return _dither_sweep(count, seed)
    if kernel == "composite":
        return _composite_sweep(count, seed)
    if kernel == "blend":
        return _blend_sweep(count, seed)
    if kernel == "tail":
        return _tail_sweep(count, seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    entries = KERNEL_ENTRIES[kernel]
    specials = SP4A_SETS
    cases = []
    for n in range(count):
        entry = entries[int(rng.integers(len(entries)))]
        spec = ENTRIES[entry]
        form = spec.forms[int(rng.integers(len(spec.forms)))]
        mirror = spec.mirrors[int(rng.integers(len(spec.mirrors)))]
        place = spec.places[int(rng.integers(len(spec.places)))] if spec.place else None
        special = specials[int(rng.integers(len(specials)))]
        mxcsr = MXCSR_STATES[int(rng.integers(len(MXCSR_STATES)))]
        h, w = (int(size) for size in rng.integers(1, 65, size=2))
        c = (1, 3, 4)[int(rng.integers(3))]
        mapping = (
            ("unit", "tiny")[int(rng.integers(2))] if special == "clean" else "unit"
        )
        cases.append(
            case(
                f"sweep-{seed}-{n}",
                entry,
                form,
                place,
                mirror,
                (h, w, c),
                special,
                mxcsr,
                src_mapping=mapping,
            )
        )
    return cases


@lru_cache(maxsize=1)
def ties() -> np.ndarray:
    """float32 values at every rounding tie of 255.0f * v (D3), read-only.

    Per k in [-260, 260]: the float32 nearest (k + 0.5) / 255 and its two
    neighbours, then, when one exists and is not among them, the float within 4
    ulps of the nearest, closest to the tie, for which 255.0f * v == k + 0.5.
    """
    values: list[np.float32] = []
    down, up = np.float32(-np.inf), np.float32(np.inf)
    for k in range(-260, 261):
        tie = (k + 0.5) / 255
        nearest = np.float32(tie)
        found = [np.nextafter(nearest, down), nearest, np.nextafter(nearest, up)]
        window = [nearest]
        for direction in (down, up):
            value = nearest
            for _ in range(4):
                value = np.nextafter(value, direction)
                window.append(value)
        exact = [v for v in window if np.float32(255) * v == np.float32(k + 0.5)]
        if exact:
            best = min(exact, key=lambda v, tie=tie: abs(float(v) - tie))
            if best not in found:
                found.append(best)
        values += found
    array = np.array(values, np.float32)
    array.setflags(write=False)
    return array


def make_array(recipe: dict) -> np.ndarray:
    """The recipe's array: PCG64(seed) raw bits, mapped to its dtype, then specials.

    float32 (the default dtype): unit: (raw >> 40) * 2**-24 in [0, 1); signed:
    ((raw >> 40) - 2**23) * 2**-23 in [-1, 1); both exact. tiny: the 24 bits
    (raw >> 40) as the float's bits, half denormals and half the smallest normals,
    with no float operation. grid (SP4b Task 5): (raw >> 62) * 0.25, so 0, 0.25, 0.5
    or 0.75, with -0 for 0 when bit 61 is set: every difference, square and sum of
    three squares is exact, so equal distances are exact ties. zeros (SP4b Task 5b):
    (raw >> 58) indexes the 64 words of ZERO_WEIGHTS (each word repeated by its weight,
    in order), with no float operation. ties: ties(). Any dtype: bits: the raw words'
    little-endian bytes reinterpreted as the dtype; ramp: arange(n) % 256.
    Specials are float32 bits written as uint32, so the MXCSR state cannot change
    them.
    """
    shape = tuple(recipe["shape"])
    count = math.prod(shape)
    dtype = np.dtype(recipe.get("dtype", "float32"))
    mapping = recipe["mapping"]
    if mapping == "bits":
        size = count * dtype.itemsize
        words = np.random.PCG64(recipe["seed"]).random_raw(-(-size // 8))
        array = np.frombuffer(words.astype("<u8").tobytes()[:size], dtype=dtype).copy()
    elif mapping == "ramp":
        array = (np.arange(count) % 256).astype(dtype)
    elif dtype != np.float32:
        raise ValueError(f"mapping {mapping!r} makes float32, not {dtype}")
    elif mapping == "ties":
        if count != ties().size:
            raise ValueError(f"the ties mapping has {ties().size} values, not {count}")
        array = ties().copy()
    elif mapping == "grid":
        raw = np.random.PCG64(recipe["seed"]).random_raw(count)
        level = (raw >> np.uint64(62)).astype(np.uint32)
        sign = ((raw >> np.uint64(61)) & np.uint64(1)).astype(np.uint32)
        array = level.astype(np.float32) * np.float32(0.25)
        array.view(np.uint32)[(level == 0) & (sign == 1)] = 0x80000000
    elif mapping == "zeros":
        raw = np.random.PCG64(recipe["seed"]).random_raw(count)
        words = [int(word, 16) for word, weight in ZERO_WEIGHTS for _ in range(weight)]
        index = (raw >> np.uint64(58)).astype(np.intp)
        array = np.array(words, np.uint32)[index].view(np.float32)
    else:
        top = np.random.PCG64(recipe["seed"]).random_raw(count) >> np.uint64(40)
        if mapping == "unit":
            array = top.astype(np.float32) * np.float32(2.0**-24)
        elif mapping == "signed":
            array = (top.astype(np.int64) - 2**23).astype(np.float32) * np.float32(
                2.0**-23
            )
        elif mapping == "tiny":
            array = top.astype(np.uint32).view(np.float32)
        else:
            raise ValueError(f"unknown mapping {mapping!r}")
    if recipe["specials"]:
        if dtype != np.float32:
            raise ValueError(f"specials are float32 bits, not {dtype}")
        bits = array.view(np.uint32)
        for index, value in recipe["specials"]:
            bits[index] = int(value, 16)
    return array.reshape(shape)


# Per float dtype digest() hashes: its word, the bits of +inf and its default NaN.
_FLOAT_WORDS = {
    np.dtype(np.float32): (np.uint32, 0x7F800000, DEFAULT_NAN),
    np.dtype(np.float64): (np.uint64, 0x7FF0000000000000, 0x7FF8000000000000),
}


def digest(array: np.ndarray, payloads: str) -> str:
    """SHA-256 of the float32 (or float64) bytes; "mixed" first maps every NaN to the
    default NaN, 0x7fc00000 (or 0x7ff8000000000000)."""
    if array.dtype not in _FLOAT_WORDS:
        raise TypeError(f"digest needs float32 or float64, not {array.dtype}")
    word, infinity, default = _FLOAT_WORDS[array.dtype]
    bits = np.ascontiguousarray(array).view(word)
    if payloads == "mixed":
        magnitude = bits & word(np.iinfo(word).max >> 1)
        bits = np.where(magnitude > word(infinity), word(default), bits)
    elif payloads != "single":
        raise ValueError(f"unknown payloads {payloads!r}")
    return hashlib.sha256(bits.tobytes()).hexdigest()


def digest_int(value: int) -> str:
    """SHA-256 of an int output as its little-endian int32 (D3)."""
    return hashlib.sha256(int(value).to_bytes(4, "little", signed=True)).hexdigest()


@contextlib.contextmanager
def mxcsr(dll: Library, state: str) -> Iterator[None]:
    """Run the body under the MXCSR state on the calling thread.

    The DLL's helpers take the caller's controls. daz_ftz uses
    torch.set_flush_denormal(True) and always resets it. cn_image_fp_state (the
    C rounding mode << 32, MXCSR rounding, FTZ and DAZ bits) must read exactly 0
    for default and DAZ_FTZ for daz_ftz: round to nearest (FE_TONEAREST is 0).
    """
    controls = dll["cn_image_fp_state"]
    controls.argtypes = []
    controls.restype = ctypes.c_uint64
    if state == "default":
        if controls() != 0:
            raise RuntimeError(
                f"FP controls {controls():#x} on the calling thread, not 0"
            )
        yield
        return
    if state != "daz_ftz":
        raise ValueError(f"unknown MXCSR state {state!r}")
    import torch

    torch.set_flush_denormal(True)
    try:
        if controls() != DAZ_FTZ:
            raise RuntimeError(
                f"FP controls {controls():#x} under set_flush_denormal(True), "
                f"not {DAZ_FTZ:#x}"
            )
        yield
    finally:
        torch.set_flush_denormal(False)


def run_case(dll: Library, case: Case) -> dict[str, str]:
    """Output name -> SHA-256 of one call of the case's C entry (ADAPTERS' own)."""
    adapter = ADAPTERS.get(case.entry)
    if adapter is not None:
        return adapter(dll, case)
    spec = ENTRIES[case.entry]
    args = case.args
    arrays = {name: make_array(recipe) for name, recipe in case.inputs.items()}
    expected: dict[str, tuple[int, ...]] = {"src": (args["h"], args["w"], args["c"])}
    if "kernel" in spec.params:
        expected["kernel"] = (args["kh"], args["kw"])
    if {name: array.shape for name, array in arrays.items()} != expected:
        raise ValueError(f"{case.id}: input shapes do not match the arguments")
    pad = args.get("padding", 0)
    shape = (args["h"] + 2 * pad, args["w"] + 2 * pad, args["c"])
    arrays[spec.output] = np.full(shape, _UNWRITTEN, np.uint32).view(np.float32)
    function = dll[case.entry]
    function.argtypes = [
        ctypes.c_void_p if p in _ARRAYS else _CTYPES.get(p, ctypes.c_size_t)
        for p in spec.params
    ]
    function.restype = ctypes.c_int
    values = [arrays[p].ctypes.data if p in _ARRAYS else args[p] for p in spec.params]
    with mxcsr(dll, case.mxcsr):
        status = function(*values)
    if status != 0:
        raise RuntimeError(f"{case.id}: {case.entry} returned status {status}")
    # No set carries 0xffffffff and x86 makes only input payloads or 0xffc00000.
    if (arrays[spec.output].view(np.uint32) == _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: output not fully written")
    return {spec.output: digest(arrays[spec.output], case.payloads)}


def convert(
    dll: Library, src: np.ndarray, args: dict, state: Mxcsr, name: str
) -> tuple[np.ndarray, int]:
    """cn_pixels_convert_checked of the 1-D src with args' type, output and normalize
    under the MXCSR state: (out, events); errors name `name`.

    Every byte value can be a uint8 output, so no fill pattern marks an unwritten
    element: the call runs twice, into out pre-filled 0x00 with events 0 and into
    out pre-filled 0xff with events -1, and both runs must agree. out is followed
    by a _CANARY-byte tail of the same fill, which must stay unchanged: a write past
    count (a vector group past the last chunk's end) fails the call (review M3).
    """
    if src.ndim != 1 or not src.flags.c_contiguous:
        raise ValueError(f"{name}: src must be 1-D and C-contiguous")
    count = src.size
    function = dll[CONVERT_ENTRY]
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    function.restype = ctypes.c_int
    kind = np.dtype(CONVERT_OUTPUTS[args["output"]])
    runs = []
    size = count * kind.itemsize
    for fill, initial in ((0x00, 0), (0xFF, -1)):
        buffer = np.full(size + _CANARY, fill, np.uint8)
        out = buffer[:size].view(kind)
        events = ctypes.c_int(initial)
        with mxcsr(dll, state):
            status = function(
                src.ctypes.data,
                out.ctypes.data,
                count,
                args["type"],
                args["output"],
                args["normalize"],
                ctypes.byref(events),
            )
        if status != 0:
            raise RuntimeError(f"{name}: {CONVERT_ENTRY} returned status {status}")
        if (buffer[size:] != fill).any():
            raise RuntimeError(f"{name}: written past count")
        runs.append((out, events.value))
    (out, events), (other, other_events) = runs
    if events != other_events or out.tobytes() != other.tobytes():
        raise RuntimeError(f"{name}: output not fully written")
    return out, events


def _run_conversion(dll: Library, case: Case) -> dict[str, str]:
    """cn_pixels_convert_checked (convert()): out (float32, uint8 or uint16) and
    events."""
    args = case.args
    src = make_array(case.inputs["src"])
    if src.shape != (args["count"],) or src.dtype != np.dtype(
        CONVERT_DTYPES[args["type"]]
    ):
        raise ValueError(f"{case.id}: src does not match count and type")
    out, events = convert(dll, src, args, case.mxcsr, case.id)
    if out.dtype == np.float32:
        written = digest(out, case.payloads)
    else:
        written = hashlib.sha256(out.tobytes()).hexdigest()
    return {"out": written, "events": digest_int(events)}


def clamps(args: dict) -> bool:
    """Whether a conversion runs the enforce clamp (D7: normalize, or any type but
    float32) on a float32 or float64 src, the combos whose reference is the oracle."""
    return args["type"] == 1 or (args["type"] == 0 and bool(args["normalize"]))


def oracle_clip(dll: Library, case: Case) -> np.ndarray:
    """The oracle's values for a conversion case: upstream's normalize of a float32
    or float64 image, np.clip(src.astype(np.float32), 0, 1)
    (nodes/impl/image_utils.py), under the case's MXCSR state (NumPy's loop runs on
    the calling thread). Cast warnings are silenced; they change no value."""
    src = make_array(case.inputs["src"])
    with mxcsr(dll, case.mxcsr), np.errstate(all="ignore"):
        return np.clip(src.astype(np.float32), 0, 1)


def oracle_outputs(dll: Library, case: Case, own: dict[str, str]) -> dict[str, str]:
    """A case's reference outputs, given the DLL's own (run_case()): the DLL's, except
    that a clamping conversion's (clamps()) float32 out is the oracle's (oracle_clip()).
    Its uint8 or uint16 out cannot move (+-0 both scale to 0), which this asserts:
    the DLL's bytes must equal the DLL's unclamped tail (type 0, normalize 0) run on
    the oracle's values. Events stay the DLL's."""
    if case.entry != CONVERT_ENTRY or not clamps(case.args):
        return own
    clipped = oracle_clip(dll, case)
    if case.args["output"] == 0:
        return {**own, "out": digest(clipped, case.payloads)}
    tail = {**case.args, "type": 0, "normalize": 0}
    out, _ = convert(dll, clipped, tail, case.mxcsr, case.id)
    if hashlib.sha256(out.tobytes()).hexdigest() != own["out"]:
        raise RuntimeError(f"{case.id}: the integer output is not np.clip's, scaled")
    return own


def _float32(bits: str) -> float:
    """The float32 whose bits the hex string gives, as a Python float (exact)."""
    return float(np.array([int(bits, 16)], np.uint32).view(np.float32)[0])


def _run_lens(dll: Library, case: Case) -> dict[str, str]:
    """cn_lens_compose: out is in-out, its initial values from its recipe.

    out is followed by a _CANARY-byte tail of 0xff, which must stay unchanged (a
    write past count fails the case). Without accumulate the output must not depend
    on out's initial values: the call runs again into out pre-filled 0xffffffff, and
    both runs must agree, so an element left unwritten fails the case.
    """
    args = case.args
    count = args["count"]
    arrays = {name: make_array(case.inputs[name]) for name in LENS_INPUTS}
    if set(case.inputs) != set(LENS_INPUTS) or any(
        array.shape != (count,) or array.dtype != np.float32
        for array in arrays.values()
    ):
        raise ValueError(f"{case.id}: inputs do not match count")
    function = dll[case.entry]
    function.argtypes = [ctypes.c_void_p] * 5 + [
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    initial = arrays["out"].view(np.uint32)
    fills = [initial] if args["accumulate"] else [initial, _UNWRITTEN]
    runs = []
    for fill in fills:
        buffer = np.full(count + _CANARY // 4, _UNWRITTEN, np.uint32)
        buffer[:count] = fill
        out = buffer[:count].view(np.float32)
        with mxcsr(dll, case.mxcsr):
            status = function(
                *(arrays[name].ctypes.data for name in LENS_INPUTS[:4]),
                out.ctypes.data,
                count,
                _float32(args["a"]),
                _float32(args["b"]),
                args["accumulate"],
            )
        if status != 0:
            raise RuntimeError(f"{case.id}: {case.entry} returned status {status}")
        if (buffer[count:] != _UNWRITTEN).any():
            raise RuntimeError(f"{case.id}: written past count")
        runs.append(out)
    if any(run.tobytes() != runs[0].tobytes() for run in runs):
        raise RuntimeError(f"{case.id}: output not fully written")
    return {"out": digest(runs[0], case.payloads)}


def _run_lens_power(dll: Library, case: Case) -> dict[str, str]:
    """cn_lens_power with finish 1 over the CHW plane: out, digested in the order of
    Lens Blur's transpose(1, 2, 0) of the (channels, h, w) result, (pixels, channels),
    as B3 recorded it. out is pre-filled 0xffffffff, which no set carries and the
    clamp never makes, and is followed by a _CANARY-byte tail of the same fill that
    must stay unchanged.
    """
    args = case.args
    pixels, channels = args["pixels"], args["channels"]
    src = make_array(case.inputs["src"])
    if set(case.inputs) != {"src"} or src.shape != (channels, pixels):
        raise ValueError(f"{case.id}: src does not match pixels and channels")
    count = pixels * channels
    exponent = _float32(args["exponent"])
    buffer = np.full(count + _CANARY // 4, _UNWRITTEN, np.uint32)
    out = buffer[:count].view(np.float32)
    power = dll[case.entry]
    power.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_int,
    ]
    power.restype = ctypes.c_int
    with mxcsr(dll, case.mxcsr):
        status = power(src.ctypes.data, out.ctypes.data, count, exponent, 1)
    stored = out.reshape(channels, pixels).T
    if status != 0:
        raise RuntimeError(f"{case.id}: {case.entry} returned status {status}")
    if (buffer[count:] != _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: written past count")
    if (buffer[:count] == _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: output not fully written")
    return {"out": digest(stored, case.payloads)}


def _run_morphology(dll: Library, case: Case) -> dict[str, str]:
    """cn_morphology_complete or cn_filter_morphology: out (h, w, c), pre-filled
    0xffffffff, which no set carries (every output element is a src element or the
    border value +-FLT_MAX), followed by a _CANARY-byte tail of the same fill that must
    stay unchanged."""
    args = case.args
    shape = (args["h"], args["w"], args["c"])
    src = make_array(case.inputs["src"])
    if set(case.inputs) != {"src"} or src.shape != shape:
        raise ValueError(f"{case.id}: src does not match h, w and c")
    count = src.size
    buffer = np.full(count + _CANARY // 4, _UNWRITTEN, np.uint32)
    out = buffer[:count].view(np.float32)
    names = tuple(MORPH_FORMS[case.entry])
    sizes = ("radius", "iterations", "lanes")
    function = dll[case.entry]
    function.argtypes = (
        [ctypes.c_void_p] * 2
        + [ctypes.c_size_t] * 3
        + [ctypes.c_size_t if name in sizes else ctypes.c_int for name in names]
    )
    function.restype = ctypes.c_int
    with mxcsr(dll, case.mxcsr):
        status = function(
            src.ctypes.data, out.ctypes.data, *shape, *(args[name] for name in names)
        )
    if status != 0:
        raise RuntimeError(f"{case.id}: {case.entry} returned status {status}")
    if (buffer[count:] != _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: written past count")
    if (buffer[:count] == _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: output not fully written")
    return {"out": digest(out.reshape(shape), case.payloads)}


def _run_exceptional(dll: Library, case: Case) -> dict[str, str]:
    """cn_image_exceptional: result, an int (0 or 1), which the entry writes; it starts
    at -1, so a call that leaves it fails the case."""
    count = case.args["count"]
    src = make_array(case.inputs["src"])
    if set(case.inputs) != {"src"} or src.shape != (count,):
        raise ValueError(f"{case.id}: src does not match count")
    function = dll[case.entry]
    function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    result = ctypes.c_int(-1)
    with mxcsr(dll, case.mxcsr):
        status = function(src.ctypes.data, count, ctypes.byref(result))
    if status != 0:
        raise RuntimeError(f"{case.id}: {case.entry} returned status {status}")
    if result.value not in (0, 1):
        raise RuntimeError(f"{case.id}: result not written ({result.value})")
    return {"result": digest_int(result.value)}


def _run_resample(dll: Library, case: Case) -> dict[str, str]:
    """cn_resample_filtered: out (th, tw, c). The call runs twice, into out pre-filled
    0x00000000 and 0xffffffff, and both runs must agree, so an element left unwritten
    fails the case without assuming which bits the gamma paths can make; out is followed
    by a _CANARY-byte tail of the fill, which must stay unchanged."""
    args = case.args
    shape = (args["h"], args["w"], args["c"])
    src = make_array(case.inputs["src"])
    if set(case.inputs) != {"src"} or src.shape != shape:
        raise ValueError(f"{case.id}: src does not match h, w and c")
    target = (args["th"], args["tw"], args["c"])
    count = math.prod(target)
    function = dll[case.entry]
    function.argtypes = (
        [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 5 + [ctypes.c_int] * 3
    )
    function.restype = ctypes.c_int
    runs = []
    for fill in (0x00000000, _UNWRITTEN):
        buffer = np.full(count + _CANARY // 4, fill, np.uint32)
        out = buffer[:count].view(np.float32)
        with mxcsr(dll, case.mxcsr):
            status = function(
                src.ctypes.data,
                out.ctypes.data,
                *shape,
                args["th"],
                args["tw"],
                args["filter"],
                args["gamma"],
                args["vector_clip"],
            )
        if status != 0:
            raise RuntimeError(f"{case.id}: {case.entry} returned status {status}")
        if (buffer[count:] != fill).any():
            raise RuntimeError(f"{case.id}: written past count")
        runs.append(out)
    if runs[0].tobytes() != runs[1].tobytes():
        raise RuntimeError(f"{case.id}: output not fully written")
    return {"out": digest(runs[0].reshape(target), case.payloads)}


# cn_palette_median_cut's status when a split would leave a child empty (the original
# median cut fails at np.min of an empty bucket).
PALETTE_EMPTY_CHILD = 6
# The sum block the B3 manifests pinned for every bucket, root and children (NumPy 1.24's
# fixed 8192, with no period restart); the node passes NumPy 2.5.3's (numpy_sum_block for
# the root, 0 after).
PALETTE_B3_BLOCK = 8192


def _run_palette(dll: Library, case: Case) -> dict[str, str]:
    """cn_palette_median_cut: out over the written colors (written x channels), written
    and the status. out holds min(capacity, pixels) colors, pre-filled 0xffffffff, which
    no set carries (an average of finite values is finite; any NaN in src returns status
    PALETTE_EMPTY_CHILD), followed by a _CANARY-byte tail of the same fill; the rows past
    written, and with that status every row and written (from 0), must stay unwritten."""
    args = case.args
    pixels, channels, capacity = args["pixels"], args["channels"], args["capacity"]
    src = make_array(case.inputs["src"])
    if set(case.inputs) != {"src"} or src.shape != (pixels, channels):
        raise ValueError(f"{case.id}: src does not match pixels and channels")
    rows = min(capacity, pixels)
    count = rows * channels
    buffer = np.full(count + _CANARY // 4, _UNWRITTEN, np.uint32)
    out = buffer[:count].view(np.float32)
    written = ctypes.c_size_t(0)
    function = dll[case.entry]
    function.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 6
    function.argtypes += [ctypes.POINTER(ctypes.c_size_t)]
    function.restype = ctypes.c_int
    with mxcsr(dll, case.mxcsr):
        status = function(
            src.ctypes.data,
            out.ctypes.data,
            pixels,
            channels,
            capacity,
            PALETTE_B3_BLOCK,
            0,
            PALETTE_B3_BLOCK,
            ctypes.pointer(written),
        )
    if status not in (0, PALETTE_EMPTY_CHILD):
        raise RuntimeError(f"{case.id}: {case.entry} returned status {status}")
    if (buffer[count:] != _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: written past count")
    colors = written.value
    if status == PALETTE_EMPTY_CHILD:
        if colors or (buffer[:count] != _UNWRITTEN).any():
            raise RuntimeError(f"{case.id}: status {status} wrote an output")
    elif not 1 <= colors <= rows or (buffer[: colors * channels] == _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: output not fully written")
    elif (buffer[colors * channels : count] != _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: written past count")
    stored = out[: colors * channels].reshape(colors, channels)
    return {
        "out": digest(stored, case.payloads),
        "written": digest_int(colors),
        "status": digest_int(status),
    }


def _bits_float(bits: str) -> ctypes.c_float:
    """The float32 whose bits the hex string gives, as a ctypes float (exact)."""
    return ctypes.c_float(_float32(bits))


def _run_dither(dll: Library, case: Case) -> dict[str, str]:
    """A dither entry: out (h, w, c), compatible and the status, all under the case's
    MXCSR state. cn_neighborhood_palette_apply runs cn_neighborhood_palette_create on the
    palette, apply, then cn_neighborhood_palette_free; cn_neighborhood_dither gets colors
    2 and map_size 2 (the palette path ignores both). out is pre-filled 0xffffffff, which
    no set or form carries (every output element is a palette color's bits), followed by
    a _CANARY-byte tail of the same fill; compatible starts at -1. With compatible 0 (300
    or more unique colors and a nonfinite value) out must stay unwritten, and its digest
    is the empty array's."""
    args = case.args
    h, w, c, count = args["h"], args["w"], args["c"], args["count"]
    src = make_array(case.inputs["src"])
    palette = make_array(case.inputs["palette"])
    if (
        set(case.inputs) != {"src", "palette"}
        or src.shape != (h, w, c)
        or palette.shape != (1, count, c)
    ):
        raise ValueError(f"{case.id}: inputs do not match h, w, c and count")
    size = h * w * c
    buffer = np.full(size + _CANARY // 4, _UNWRITTEN, np.uint32)
    out = buffer[:size].view(np.float32)
    compatible = ctypes.c_int(-1)
    pointers = [src.ctypes.data, out.ctypes.data, h, w, c]
    sizes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 3
    flag = ctypes.POINTER(ctypes.c_int)
    with mxcsr(dll, case.mxcsr):
        if case.entry == DITHER_APPLY:
            create = dll["cn_neighborhood_palette_create"]
            create.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
            create.argtypes += [ctypes.POINTER(ctypes.c_void_p)]
            create.argtypes += [ctypes.POINTER(ctypes.c_size_t)]
            create.restype = ctypes.c_int
            apply = dll[case.entry]
            apply.argtypes = [*sizes, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            apply.argtypes += [ctypes.c_uint32, ctypes.c_float, flag]
            apply.restype = ctypes.c_int
            release = dll["cn_neighborhood_palette_free"]
            release.argtypes = [ctypes.c_void_p]
            release.restype = None
            handle, held = ctypes.c_void_p(), ctypes.c_size_t()
            created = create(
                palette.ctypes.data,
                count,
                c,
                ctypes.pointer(handle),
                ctypes.pointer(held),
            )
            if created != 0:
                raise RuntimeError(f"{case.id}: palette creation returned {created}")
            try:
                status = apply(
                    *pointers,
                    handle,
                    args["mode"],
                    args["algorithm"],
                    args["history"],
                    _bits_float(args["decay"]),
                    ctypes.pointer(compatible),
                )
            finally:
                release(handle)
        elif case.entry == DITHER_ENTRY:
            function = dll[case.entry]
            function.argtypes = [*sizes, ctypes.c_uint32, ctypes.c_int, ctypes.c_size_t]
            function.argtypes += [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, flag]
            function.restype = ctypes.c_int
            status = function(
                *pointers,
                2,
                args["mode"],
                2,
                args["algorithm"],
                palette.ctypes.data,
                count,
                ctypes.pointer(compatible),
            )
        else:
            function = dll[case.entry]
            function.argtypes = [*sizes, ctypes.c_void_p, ctypes.c_size_t]
            function.argtypes += [ctypes.c_uint32, ctypes.c_float, flag]
            function.restype = ctypes.c_int
            status = function(
                *pointers,
                palette.ctypes.data,
                count,
                args["history"],
                _bits_float(args["decay"]),
                ctypes.pointer(compatible),
            )
    if status != 0:
        raise RuntimeError(f"{case.id}: {case.entry} returned status {status}")
    if compatible.value not in (0, 1):
        raise RuntimeError(f"{case.id}: compatible not written ({compatible.value})")
    if (buffer[size:] != _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: written past count")
    if compatible.value == 0:
        if (buffer[:size] != _UNWRITTEN).any():
            raise RuntimeError(f"{case.id}: an incompatible call wrote an output")
        stored = out[:0]
    elif (buffer[:size] == _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: output not fully written")
    else:
        stored = out.reshape(h, w, c)
    return {
        "out": digest(stored, case.payloads),
        "compatible": digest_int(compatible.value),
        "status": digest_int(status),
    }


class CanvasLayer(ctypes.Structure):
    """composite_complete_ops.c's canvas_layer (native_composite._Layer)."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("height", ctypes.c_size_t),
        ("width", ctypes.c_size_t),
        ("channels", ctypes.c_size_t),
        ("constant", ctypes.c_int),
    ]


class CanvasGeometry(ctypes.Structure):
    """composite_complete_ops.c's canvas_geometry (native_composite._Geometry), in
    composite_geometry()'s order."""

    _fields_ = [
        (name, ctypes.c_size_t)
        for name in (
            "height",
            "width",
            "channels",
            "base_top",
            "base_left",
            "paste_top",
            "paste_left",
            "overlay_top",
            "overlay_left",
            "region_height",
            "region_width",
        )
    ] + [("padded", ctypes.c_int)]


def _run_composite(dll: Library, case: Case) -> dict[str, str]:
    """cn_composite_canvas with native_composite.canvas()'s layers and geometry
    (composite_geometry()): out (h, w, channels), pre-filled 0xffffffff, which no set
    carries and no operation makes (x86's generated NaN is 0xffc00000; NaN inputs keep
    their payloads), followed by a _CANARY-byte tail of the same fill. Every element must
    be written (the fill covers the canvas) and the status must be 0."""
    args = case.args
    if set(case.inputs) != set(COMPOSITE_INPUTS):
        raise ValueError(f"{case.id}: inputs are not base and overlay")
    arrays = {name: make_array(case.inputs[name]) for name in COMPOSITE_INPUTS}
    shapes = {}
    for name, prefix in zip(COMPOSITE_INPUTS, "bo", strict=True):
        shape = (args[f"{prefix}h"], args[f"{prefix}w"], args[f"{prefix}c"])
        shapes[name] = shape
        held = shape[2:] if args[f"{name}_constant"] else shape
        if arrays[name].shape != held:
            raise ValueError(f"{case.id}: {name} does not match its h, w and c")
    geometry = composite_geometry(args)
    h, w, channels = geometry[:3]
    count = h * w * channels
    buffer = np.full(count + _CANARY // 4, _UNWRITTEN, np.uint32)
    out = buffer[:count].view(np.float32)
    layers = [
        CanvasLayer(arrays[name].ctypes.data, *shapes[name], args[f"{name}_constant"])
        for name in COMPOSITE_INPUTS
    ]
    canvas = CanvasGeometry(*geometry)
    function = dll[case.entry]
    function.argtypes = [ctypes.POINTER(CanvasLayer)] * 2 + [
        ctypes.POINTER(CanvasGeometry),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    with mxcsr(dll, case.mxcsr):
        status = function(
            ctypes.pointer(layers[0]),
            ctypes.pointer(layers[1]),
            ctypes.pointer(canvas),
            out.ctypes.data,
            args["mode"],
        )
    if status != 0:
        raise RuntimeError(f"{case.id}: {case.entry} returned status {status}")
    if (buffer[count:] != _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: written past count")
    if (buffer[:count] == _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: output not fully written")
    return {"out": digest(out.reshape(h, w, channels), case.payloads)}


def _run_blend(dll: Library, case: Case) -> dict[str, str]:
    """cn_blend_images (out: pixels x the larger channel count) or cn_blend_mode (out:
    count elements), pre-filled 0xffffffff, which no set carries and no mode makes
    (x86's generated NaN is 0xffc00000; NaN inputs keep their payloads), followed by a
    _CANARY-byte tail of the same fill. Every element must be written and the status
    must be 0."""
    args = case.args
    if set(case.inputs) != set(BLEND_INPUTS):
        raise ValueError(f"{case.id}: inputs are not overlay and base")
    arrays = [make_array(case.inputs[name]) for name in BLEND_INPUTS]
    function = dll[case.entry]
    if case.entry == BLEND_IMAGES:
        pixels, channels = args["pixels"], (args["oc"], args["bc"])
        shapes = [(pixels, channels[0]), (pixels, channels[1])]
        shape = (pixels, max(channels))
        sizes = [pixels, *channels, args["mode"]]
        function.argtypes = (
            [ctypes.c_void_p] * 3 + [ctypes.c_size_t] + [ctypes.c_int] * 3
        )
    else:
        shapes = [(args["count"],)] * 2
        shape = (args["count"],)
        sizes = [args["count"], args["mode"]]
        function.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t, ctypes.c_int]
    if [array.shape for array in arrays] != shapes:
        raise ValueError(f"{case.id}: inputs do not match the arguments")
    function.restype = ctypes.c_int
    count = math.prod(shape)
    buffer = np.full(count + _CANARY // 4, _UNWRITTEN, np.uint32)
    out = buffer[:count].view(np.float32)
    with mxcsr(dll, case.mxcsr):
        status = function(
            arrays[0].ctypes.data, arrays[1].ctypes.data, out.ctypes.data, *sizes
        )
    if status != 0:
        raise RuntimeError(f"{case.id}: {case.entry} returned status {status}")
    if (buffer[count:] != _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: written past count")
    if (buffer[:count] == _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: output not fully written")
    return {"out": digest(out.reshape(shape), case.payloads)}


def _tail_arrays(case: Case) -> dict[str, np.ndarray]:
    """A tail case's input arrays, which must be tail_inputs()'s, by name."""
    expected = {
        name: (shape, np.dtype(dtype))
        for name, shape, dtype, _ in tail_inputs(case.entry, case.args)
    }
    arrays = {name: make_array(recipe) for name, recipe in case.inputs.items()}
    if {name: (a.shape, a.dtype) for name, a in arrays.items()} != expected:
        raise ValueError(f"{case.id}: inputs do not match the arguments")
    return arrays


def _tail_call(dll: Library, case: Case, argtypes: list, *arrays: Any) -> None:
    """Call the case's entry with the arrays (data pointers, or None for NULL) and its
    TAIL_PARAMS under its MXCSR state; any status but 0 fails the case."""
    function = dll[case.entry]
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    pointers = [None if array is None else array.ctypes.data for array in arrays]
    with mxcsr(dll, case.mxcsr):
        status = function(*pointers, *(case.args[p] for p in TAIL_PARAMS[case.entry]))
    if status != 0:
        raise RuntimeError(f"{case.id}: {case.entry} returned status {status}")


def _run_spectral_tile(dll: Library, case: Case) -> dict[str, str]:
    """cn_spectral_tile: dst, the (dh, dw) float64 plane, pre-filled with the all-ones
    word, which no float32 src value widens to (a widened NaN's low 29 bits are zero),
    followed by a _CANARY-byte tail of the same fill that must stay unchanged."""
    src = _tail_arrays(case)["src"]
    shape = (case.args["dh"], case.args["dw"])
    count = math.prod(shape)
    fill = np.uint64(_UNWRITTEN_DOUBLE)
    buffer = np.full(count + _CANARY // 8, fill, np.uint64)
    dst = buffer[:count].view(np.float64)
    argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 13 + [ctypes.c_int]
    _tail_call(dll, case, argtypes, src, dst)
    if (buffer[count:] != fill).any():
        raise RuntimeError(f"{case.id}: written past count")
    if (buffer[:count] == fill).any():
        raise RuntimeError(f"{case.id}: output not fully written")
    return {"dst": digest(dst.reshape(shape), case.payloads)}


def _run_spectral_store(dll: Library, case: Case) -> dict[str, str]:
    """cn_spectral_store: dst (oh, ow, c) is in-out, its values from its recipe; the
    call narrows src's (th, tw) corner into channel `channel` of the th x tw pixels at
    (y, x). It runs twice, into dst as made and into dst with those elements pre-filled
    0xffffffff, and both runs must agree, so an element left unwritten fails the case.
    dst is followed by a _CANARY-byte tail of 0xff, which must stay unchanged."""
    args = case.args
    arrays = _tail_arrays(case)
    initial = arrays["dst"]
    count = initial.size
    y, x, th, tw = (args[name] for name in ("y", "x", "th", "tw"))
    argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 10
    runs = []
    for marked in (False, True):
        buffer = np.full(count + _CANARY // 4, _UNWRITTEN, np.uint32)
        buffer[:count] = initial.view(np.uint32).ravel()
        dst = buffer[:count].view(np.float32).reshape(initial.shape)
        if marked:
            dst.view(np.uint32)[y : y + th, x : x + tw, args["channel"]] = _UNWRITTEN
        _tail_call(dll, case, argtypes, arrays["src"], dst)
        if (buffer[count:] != _UNWRITTEN).any():
            raise RuntimeError(f"{case.id}: written past count")
        runs.append(dst)
    if runs[0].tobytes() != runs[1].tobytes():
        raise RuntimeError(f"{case.id}: output not fully written")
    return {"dst": digest(runs[0], case.payloads)}


def _run_spectral_multiply(dll: Library, case: Case) -> dict[str, str]:
    """cn_spectral_multiply: a (h, w) float64 is in-out, its values from its recipe;
    every element becomes a function of its own value, so no fill can show one left
    alone. a is followed by a _CANARY-byte tail of the all-ones word, which must stay
    unchanged."""
    arrays = _tail_arrays(case)
    initial = arrays["a"]
    count = initial.size
    fill = np.uint64(_UNWRITTEN_DOUBLE)
    buffer = np.full(count + _CANARY // 8, fill, np.uint64)
    buffer[:count] = initial.view(np.uint64).ravel()
    a = buffer[:count].view(np.float64).reshape(initial.shape)
    _tail_call(dll, case, [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 2, a, arrays["b"])
    if (buffer[count:] != fill).any():
        raise RuntimeError(f"{case.id}: written past count")
    return {"a": digest(a, case.payloads)}


def _run_normal_output(dll: Library, case: Case) -> dict[str, str]:
    """cn_normal_output: dst (pixels, channels), pre-filled 0xffffffff, which no set
    carries and no operation makes (x86's generated NaN is 0xffc00000; a NaN input keeps
    its payload, its sign flipped by an inversion or cleared by fabsf), followed by a
    _CANARY-byte tail of the same fill; alpha is NULL unless args alpha is 1."""
    args = case.args
    arrays = _tail_arrays(case)
    shape = (args["pixels"], args["channels"])
    count = math.prod(shape)
    buffer = np.full(count + _CANARY // 4, _UNWRITTEN, np.uint32)
    dst = buffer[:count].view(np.float32)
    argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_size_t] + [ctypes.c_int] * 3
    alpha = arrays.get("alpha")
    _tail_call(dll, case, argtypes, arrays["dx"], arrays["dy"], alpha, dst)
    if (buffer[count:] != _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: written past count")
    if (buffer[:count] == _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: output not fully written")
    return {"dst": digest(dst.reshape(shape), case.payloads)}


def _run_caption(dll: Library, case: Case) -> dict[str, str]:
    """cn_caption_compose: out (height + caption_height, width, channels), pre-filled
    0xffffffff, which no set carries and no quotient k / 255 makes, followed by a
    _CANARY-byte tail of the same fill."""
    args = case.args
    arrays = _tail_arrays(case)
    rows = args["height"] + args["caption_height"]
    shape = (rows, args["width"], args["channels"])
    count = math.prod(shape)
    buffer = np.full(count + _CANARY // 4, _UNWRITTEN, np.uint32)
    out = buffer[:count].view(np.float32)
    argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t] * 4 + [ctypes.c_int]
    _tail_call(dll, case, argtypes, arrays["image"], arrays["caption"], out)
    if (buffer[count:] != _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: written past count")
    if (buffer[:count] == _UNWRITTEN).any():
        raise RuntimeError(f"{case.id}: output not fully written")
    return {"out": digest(out.reshape(shape), case.payloads)}


# Entries whose outputs are not one float32 array, or whose B3 form differs (D3).
ADAPTERS: dict[str, Callable[[Library, Case], dict[str, str]]] = {
    CONVERT_ENTRY: _run_conversion,
    LENS_ENTRY: _run_lens,
    POWER_ENTRY: _run_lens_power,
    MORPH_ENTRY: _run_morphology,
    FILTER_ENTRY: _run_morphology,
    EXCEPTIONAL_ENTRY: _run_exceptional,
    RESAMPLE_ENTRY: _run_resample,
    PALETTE_ENTRY: _run_palette,
    DITHER_APPLY: _run_dither,
    DITHER_ENTRY: _run_dither,
    RIEMERSMA_ENTRY: _run_dither,
    COMPOSITE_ENTRY: _run_composite,
    BLEND_IMAGES: _run_blend,
    BLEND_MODE: _run_blend,
    TILE_ENTRY: _run_spectral_tile,
    STORE_ENTRY: _run_spectral_store,
    MULTIPLY_ENTRY: _run_spectral_multiply,
    NORMAL_ENTRY: _run_normal_output,
    CAPTION_ENTRY: _run_caption,
}


def dumps(manifest: dict) -> str:
    """Sorted-key JSON, one case per line, LF-terminated."""

    def compact(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    if min(manifest) != "cases":
        raise ValueError("the layout needs cases to sort before the other keys")
    rest = {key: value for key, value in manifest.items() if key != "cases"}
    lines = ",\n".join(compact(case) for case in manifest["cases"])
    return '{"cases":[\n' + lines + "\n]," + compact(rest)[1:] + "\n"


def _load_dll(path: Path) -> ctypes.CDLL:
    dll = ctypes.CDLL(str(path))
    version = dll["cn_abi_version"]
    version.argtypes = []
    version.restype = ctypes.c_int
    if version() != 2:
        raise RuntimeError(f"{path}: C ABI {version()}, expected 2")
    return dll


def _manifest(
    dll_path: Path, dll: Library, kernel: str, cases: list[Case], oracle: bool = False
) -> dict:
    """The cases run through the DLL; with oracle, a case whose reference
    (oracle_outputs()) differs from the DLL's outputs holds the reference and records
    ORACLE."""
    items = []
    for case in cases:
        outputs = run_case(dll, case)
        item = {**dataclasses.asdict(case), "outputs": outputs}
        if oracle:
            reference = oracle_outputs(dll, case, outputs)
            if reference != outputs:
                item.update(outputs=reference, generated_with=ORACLE)
        items.append(item)
    return {
        "schema": SCHEMA,
        "kernel": kernel,
        "generated_with": {
            "dll_sha256": hashlib.sha256(dll_path.read_bytes()).hexdigest(),
            "numpy": np.__version__,
        },
        "pcg64_check": np.random.PCG64(0).random_raw(4).tolist(),
        "special_sets": SPECIALS,
        "cases": items,
    }


def _write(out: Path, manifest: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(manifest), encoding="utf-8", newline="\n")
    print(
        f"{out}: {manifest['kernel']}, {len(manifest['cases'])} cases, "
        f"{out.stat().st_size} bytes, "
        f"dll sha256 {manifest['generated_with']['dll_sha256']}"
    )


def write_manifest(dll_path: Path, kernel: str, out: Path) -> None:
    _write(out, _manifest(dll_path, _load_dll(dll_path), kernel, matrix(kernel)))


def load_manifest(path: Path) -> tuple[dict, list[tuple[Case, dict[str, str]]]]:
    """(every key but cases, [(case, outputs)]) of a manifest of this schema. A case's
    own generated_with can only be ORACLE or PORT (rebased_ids() lists either)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {data.get('schema')!r}, expected {SCHEMA}")
    cases = []
    for item in data.pop("cases"):
        outputs = item.pop("outputs")
        if item.pop("generated_with", ORACLE) not in (ORACLE, PORT):
            raise ValueError(f"{path}: {item['id']} has an unknown generated_with")
        cases.append((Case(**item), outputs))
    return data, cases


def rebased_ids(path: Path, marker: dict[str, str] = ORACLE) -> list[str]:
    """The ids of a manifest's cases that record marker: ORACLE, re-based on the oracle
    (rebase()), or PORT, chaiNNer-C's outputs where upstream's are not one value."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        item["id"] for item in data["cases"] if item.get("generated_with") == marker
    ]


def rebase(dll_path: Path, kernel: str, path: Path) -> None:
    """Re-base the conversion manifest at path, written from dll_path, on the oracle
    (Task 3b1; plan, Protected: only the cases whose behavior the parity fix moves).

    Every case runs again through the DLL. A case whose reference (oracle_outputs())
    differs from the DLL's outputs takes the reference and records ORACLE; it must
    hold one of the two. Every other case must hold the DLL's outputs and no
    generated_with but PORT, which stays. The header and every other line stay as
    written.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    written = hashlib.sha256(dll_path.read_bytes()).hexdigest()
    if (
        data.get("schema") != SCHEMA
        or data["kernel"] != kernel
        or data["generated_with"]["dll_sha256"] != written
    ):
        raise ValueError(f"{path}: not a {kernel} manifest written from {dll_path}")
    dll = _load_dll(dll_path)
    moved = []
    for item in data["cases"]:
        fields = {
            k: v for k, v in item.items() if k not in ("outputs", "generated_with")
        }
        case = Case(**fields)
        own = run_case(dll, case)
        reference = oracle_outputs(dll, case, own)
        if reference == own:
            if item["outputs"] != own or item.get("generated_with", PORT) != PORT:
                raise RuntimeError(
                    f"{case.id}: the manifest does not hold the DLL's outputs"
                )
            continue
        if item["outputs"] not in (own, reference):
            raise RuntimeError(f"{case.id}: neither the DLL's nor the oracle's outputs")
        item.update(outputs=reference, generated_with=ORACLE)
        moved.append(case.id)
    for case_id in moved:
        print(f"rebased {case_id}")
    print(f"{len(moved)} of {len(data['cases'])} cases hold the oracle's outputs")
    _write(path, data)


def diff(a: Path, b: Path) -> int:
    head_a, cases_a = load_manifest(a)
    head_b, cases_b = load_manifest(b)
    if head_a["kernel"] != head_b["kernel"]:
        print(f"different kernels: {head_a['kernel']} and {head_b['kernel']}")
        return 1
    left = {case.id: (case, outputs) for case, outputs in cases_a}
    right = {case.id: (case, outputs) for case, outputs in cases_b}
    ids = list(dict.fromkeys([*left, *right]))
    mismatched = [i for i in ids if left.get(i) != right.get(i)]
    for case_id in mismatched:
        print(f"mismatch {case_id}")
    print(f"{len(mismatched)} of {len(ids)} cases differ")
    return 1 if mismatched else 0


def set_isa(dll: Library, dll_path: Path, level: str) -> bool:
    """cn_isa_set(level); false, with the reason on stderr, unless it returns level."""
    try:
        isa_set = dll["cn_isa_set"]
    except AttributeError:
        print(f"{dll_path}: no cn_isa_set export", file=sys.stderr)
        return False
    isa_set.argtypes = [ctypes.c_int]
    isa_set.restype = ctypes.c_int
    returned = isa_set(ISA_LEVELS.index(level))
    if returned != ISA_LEVELS.index(level):
        print(f"cn_isa_set({level}) returned {returned}", file=sys.stderr)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write", help="run the kernel's fixed matrix")
    sweep = commands.add_parser("sweep", help="run cases drawn from the matrix's axes")
    for command in (write, sweep):
        command.add_argument("--dll", type=Path, required=True)
        command.add_argument("--kernel", choices=KERNELS, required=True)
        command.add_argument("--out", type=Path, required=True)
    sweep.add_argument("--count", type=int, required=True)
    sweep.add_argument("--seed", type=int, required=True)
    sweep.add_argument("--isa", choices=ISA_LEVELS)
    sweep.add_argument(
        "--oracle", action="store_true", help="the clamping conversions' reference"
    )
    rebased = commands.add_parser("rebase", help="re-base on the oracle (rebase())")
    rebased.add_argument("--dll", type=Path, required=True)
    rebased.add_argument("--kernel", choices=CONVERSION_KERNELS, required=True)
    rebased.add_argument("--manifest", type=Path, required=True)
    compare = commands.add_parser("diff", help="compare two manifests case by case")
    compare.add_argument("a", type=Path)
    compare.add_argument("b", type=Path)
    args = parser.parse_args(argv)
    if args.command == "diff":
        return diff(args.a, args.b)
    if args.command == "write":
        write_manifest(args.dll, args.kernel, args.out)
        return 0
    if args.command == "rebase":
        rebase(args.dll, args.kernel, args.manifest)
        return 0
    dll = _load_dll(args.dll)
    if args.isa is not None and not set_isa(dll, args.dll, args.isa):
        return 2
    cases = sweep_cases(args.kernel, args.count, args.seed)
    manifest = _manifest(args.dll, dll, args.kernel, cases, args.oracle)
    manifest["generated_with"]["isa"] = args.isa  # None: the DLL's own level
    if args.oracle:
        manifest["generated_with"].update(ORACLE)
    _write(args.out, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
