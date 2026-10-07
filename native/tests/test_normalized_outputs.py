"""Lens, Gaussian, Box Blur, Dilate and Erode outputs borrowed at the output enforce
(SP4b Tasks 3b, 3b2 and 4, A3).

Lens Blur freezes and registers its fresh CHW finishing power (native_buffers.
freeze_normalized) and returns its transpose(1, 2, 0), as upstream returns that view
(D-7). ImageOutput.enforce borrows the gapless permuted view through
normalized_readonly's scan (cn_pixels_normalized_f32) with its strides, the ones
upstream's np.clip (order K) gives, instead of converting it. Bits never change: a
borrow needs the scan's proof that enforce's clip leaves every bit as it is, and a
refused borrow converts exactly as before, in the same order-K layout. B3's path is
the same node unregistered: power(finish=True), then transpose(1, 2, 0).

Gaussian and Box Blur (Task 3b2) first clamp their fresh float32 result in place with
enforce's own conversion (freeze_normalized(clamp=True)), which is idempotent, so
the user receives the bits of the raw path: the node before the opt-in, whose raw
result is unregistered and converted by the enforce. fast_gaussian_blur's other
callers keep its raw, writable result. Dilate and Erode (Task 4) do the same with
native_filters.morphology's fresh result, which owns its data in every channel count.

This module imports only the modules it needs: the node bodies and numpy_outputs
are loaded by path (the nodes.properties.outputs package would import Pillow and
torch), and DAZ|FTZ is set through the CRT's _controlfp_s, not torch.

It also holds what the Gaussian and Box Blur parity tests share (Task 3b2; owner,
2026-10-04: the contract is the schema and the output as the user receives it, not a
node's raw pre-enforce return): assert_enforced_equal, node_schema and RAW_BOX.
"""

import ast
import contextlib
import ctypes
import importlib.util
import types
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_isa_dispatch import isa_get, isa_set
from test_native_profile import child

from nodes.impl import image_utils, native, native_buffers, native_filters

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend" / "src"
PACKAGES = BACKEND / "packages/chaiNNer_standard"
LENS_PATH = PACKAGES / "image_filter/blur/lens_blur.py"
GAUSSIAN_PATH = PACKAGES / "image_filter/blur/gaussian_blur.py"
BOX_PATH = PACKAGES / "image_filter/blur/box_blur.py"
DILATE_PATH = PACKAGES / "image_filter/miscellaneous/dilate.py"
ERODE_PATH = PACKAGES / "image_filter/miscellaneous/erode.py"
# fast_gaussian_blur's callers that use its raw values in arithmetic.
RAW_CALLER_PATHS = {
    "unsharp_mask": PACKAGES / "image_filter/sharpen/unsharp_mask.py",
    "high_pass": PACKAGES / "image_filter/miscellaneous/high_pass.py",
    "edge_detection": PACKAGES / "image_filter/miscellaneous/edge_detection.py",
    "normal_map_generator": PACKAGES
    / "material_textures/normal_map/normal_map_generator.py",
}
OUTPUTS_PATH = BACKEND / "nodes/properties/outputs/numpy_outputs.py"
# The CRT's denormal control (_MCW_DN) and its flush value (_DN_FLUSH: MXCSR DAZ and
# FTZ, cn_image_fp_state 0x8040).
DENORMAL_MASK = 0x03000000
DENORMAL_FLUSH = 0x01000000
DAZ_FTZ = 0x8040
MXCSR_STATES = ("default", "daz_ftz")
# cn_pixels_normalized_f32's boundaries: +0 and normal values in (0, 1] pass.
BOUNDARY_WORDS = (
    0x00000000,
    0x80000000,
    0x00000001,
    0x007FFFFF,
    0x00800000,
    0x3F800000,
    0x3F800001,
    0x7F800000,
    0x7FC12345,  # quiet NaN
    0x7F812345,  # signalling NaN
)
SCAN_COUNTS = (*range(18), 589_824)
FILL = 0x3F000000  # 0.5, which passes
BENCH = (30, 3, 3)  # lens-spectral's radius, components and exposure gamma


def load_node(path):
    """A node module's body: imports of api, nodes.groups, nodes.properties and
    relative imports dropped, decorators removed (as the node tests load bodies)."""
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


LENS = load_node(LENS_PATH)
# B3's Lens Blur: the CHW finishing power's transposed view, unregistered.
B3_LENS = load_node(LENS_PATH)
vars(B3_LENS).update(freeze_normalized=lambda image: image)
_spec = importlib.util.spec_from_file_location(
    "nodes.properties.outputs._normalized_outputs_probe", OUTPUTS_PATH
)
assert _spec is not None and _spec.loader is not None
OUTPUTS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OUTPUTS)


def raw_fast_gaussian_blur(img, sigma_x, sigma_y, *, normalized=False):
    """fast_gaussian_blur before the opt-in: its raw result, whatever normalized says
    (the node passes True)."""
    return image_utils.fast_gaussian_blur(img, sigma_x, sigma_y)


# Gaussian and Box Blur (Task 3b2), and each before its opt-in (the raw path): the
# Gaussian body with fast_gaussian_blur's raw result, the Box body with
# freeze_normalized as the identity, so each return is its helper's raw result
# (box_blur, separable_box or filter2d), unregistered. The Box parity tests compare
# RAW_BOX with the reference's OpenCV results (kernel parity).
GAUSSIAN = load_node(GAUSSIAN_PATH)
RAW_GAUSSIAN = load_node(GAUSSIAN_PATH)
vars(RAW_GAUSSIAN).update(fast_gaussian_blur=raw_fast_gaussian_blur)
BOX = load_node(BOX_PATH)
RAW_BOX = load_node(BOX_PATH)
vars(RAW_BOX).update(freeze_normalized=lambda image, *, clamp: image)
# Dilate and Erode (Task 4), and each before its opt-in: freeze_normalized as the
# identity, so the return is native_filters.morphology's raw result, unregistered.
DILATE = load_node(DILATE_PATH)
RAW_DILATE = load_node(DILATE_PATH)
vars(RAW_DILATE).update(freeze_normalized=lambda image, *, clamp: image)
ERODE = load_node(ERODE_PATH)
RAW_ERODE = load_node(ERODE_PATH)
vars(RAW_ERODE).update(freeze_normalized=lambda image, *, clamp: image)


def upstream_enforce(value):
    """A node return as the installed chaiNNer's ImageOutput.enforce hands it to the
    user (numpy_outputs.py enforce, then image_utils.normalize), with NumPy alone and
    none of the port's code: a trailing single channel squeezed, then float32 clipped
    to [0, 1]; any other dtype becomes float32, an integer is divided by its maximum
    and, if signed, clipped."""
    if value.ndim == 3 and value.shape[2] == 1:
        value = value[:, :, 0]
    if value.dtype == np.float32:
        return np.clip(value, 0, 1)
    info = np.iinfo(value.dtype) if np.issubdtype(value.dtype, np.integer) else None
    value = value.astype(np.float32)
    if info is not None:
        value /= info.max
        if info.min == 0:
            return value
    return np.clip(value, 0, 1, out=value)


def assert_enforced_equal(actual, expected):
    """Two node returns as the user receives them are equal bitwise: shape, dtype, NaN
    positions and every other word, +0 and -0 distinct. The actual side is the port's
    ImageOutput.enforce (the executor's enforce_output); the expected side is what the
    installed chaiNNer's enforce gives (upstream_enforce), not the port's."""
    actual = OUTPUTS.ImageOutput().enforce(actual)
    expected = upstream_enforce(expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    nan = np.isnan(expected)
    assert np.array_equal(np.isnan(actual), nan)
    assert np.array_equal(actual.view(np.uint32)[~nan], expected.view(np.uint32)[~nan])


def node_schema(path):
    """A node's schema as the UI receives it: its register(...) call (the decorators)
    and its function's name and signature."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    (function,) = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.decorator_list
    ]
    returns = ast.dump(function.returns) if function.returns else None
    return (
        [ast.dump(decorator) for decorator in function.decorator_list],
        function.name,
        ast.dump(function.args),
        returns,
    )


def export(name, argtypes):
    """A fresh function object of the DLL's export (never the bridges' shared one),
    returning int."""
    function = native.lib()[name]
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    return function


def fp_state():
    """cn_image_fp_state: the C rounding mode << 32 and MXCSR's rounding, FTZ and DAZ
    bits."""
    function = native.lib()["cn_image_fp_state"]
    function.argtypes = []
    function.restype = ctypes.c_uint64
    return function()


@contextlib.contextmanager
def mxcsr(state):
    """Run the body under the MXCSR state on the calling thread (the DLL's helpers
    take the caller's controls); always restores the default."""
    control = ctypes.CDLL("ucrtbase.dll")["_controlfp_s"]
    control.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
    control.restype = ctypes.c_int
    word = ctypes.c_uint()
    assert fp_state() == 0
    if state == "daz_ftz":
        assert control(ctypes.byref(word), DENORMAL_FLUSH, DENORMAL_MASK) == 0
    try:
        assert fp_state() == (DAZ_FTZ if state == "daz_ftz" else 0)
        yield
    finally:
        assert control(ctypes.byref(word), 0, DENORMAL_MASK) == 0
        assert fp_state() == 0


@pytest.fixture
def levels():
    """Every level this CPU has, scalar first; the original level afterwards."""
    effective, _, cpu = isa_get()
    try:
        yield range(cpu + 1)
    finally:
        assert isa_set(effective) == effective


def passes(bits):
    """The scan's rule: every word is +0 or a normal value in (0, 1]."""
    bits = np.asarray(bits, np.uint32)
    return bool(((bits == 0) | ((bits >= 0x00800000) & (bits <= 0x3F800000))).all())


def scan(array):
    same = ctypes.c_int(-1)
    function = export(
        "cn_pixels_normalized_f32",
        [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)],
    )
    assert function(array.ctypes.data, array.size, ctypes.byref(same)) == 0
    return same.value


def test_normalized_scan_matches_scalar_at_every_level(levels):
    # Each boundary word at every position of counts 0-17 (tails of every length,
    # every position mod 8, from a start 0-7 elements past a 32-byte boundary) and at
    # positions 0-39, around the middle and in the last 40 of 589,824 elements, in a
    # field of 0.5: every level returns the scalar loop's `same`, which is the rule.
    storage = np.full(589_824 + 8, FILL, np.uint32)
    found = []
    for count in SCAN_COUNTS:
        if count < 18:
            starts = range(8)
            positions = range(count)
        else:
            starts = (0,)
            middle = count // 2 - 20
            positions = [*range(40), *range(middle, middle + 40)]
            positions += range(count - 40, count)
        for start in starts:
            words = storage[start : start + count]
            found.append((words, None, None))
            for position in positions:
                for word in BOUNDARY_WORDS:
                    found.append((words, position, word))
    results = {}
    for level in levels:
        assert isa_set(level) == level
        row = []
        for words, position, word in found:
            if position is None:
                row.append(scan(words.view(np.float32)))
                continue
            words[position] = word
            row.append(scan(words.view(np.float32)))
            words[position] = FILL
        results[level] = row
    expected = [int(position is None or passes([word])) for _, position, word in found]
    assert results[0] == expected
    for level in levels:
        assert results[level] == results[0], native.ISA_LEVELS[level]
    # Fields of each passing word pass at every level.
    for word in (0x00000000, 0x00800000, 0x3F000000, 0x3F800000):
        field = np.full(589_824 + 13, word, np.uint32).view(np.float32)
        for level in levels:
            assert isa_set(level) == level
            assert scan(field) == 1, (hex(word), native.ISA_LEVELS[level])


def lens_images():
    """(name, image, (radius, components, gamma)): the bench case, and inputs whose
    outputs hold 1.0, NaN or denormals, in each channel layout."""
    rng = np.random.default_rng(20261004)
    bench = rng.random((384, 512, 3), dtype=np.float32)
    nan = rng.random((19, 23, 3), dtype=np.float32)
    nan[5, 7, 1] = np.nan
    tiny = np.full((17, 21, 4), np.float32(3e-39), np.float32)
    return [
        ("bench", bench, BENCH),
        # Above 1, so the final clip stores exact 1.0 (NumPy 2.5's float32 kernel
        # normalization leaves a blurred all-ones image at 0x3F7FFFFF).
        ("saturated", np.full((19, 23, 3), 2, np.float32), (3, 2, 5)),
        ("nan", nan, (3, 2, 5)),
        ("denormal", tiny, (2, 1, 1)),
        ("gray", rng.random((17, 21), dtype=np.float32), (3, 3, 0.5)),
        ("rgba", rng.random((13, 11, 4), dtype=np.float32), (4, 5, 2.2)),
    ]


@pytest.mark.parametrize(
    ("name", "image", "parameters"),
    lens_images(),
    ids=[name for name, _, _ in lens_images()],
)
def test_lens_output_borrowed_at_output_enforce(name, image, parameters, levels):
    # At every level and in both MXCSR states, ImageOutput.enforce returns the node's
    # own transposed view of its frozen plane (or, for one channel, its 2-D view)
    # exactly when the scan passes, and in every case the bytes and the strides of
    # B3's path.
    seen = set()
    channels = 1 if image.ndim == 2 else image.shape[2]
    height, width = image.shape[:2]
    for level in levels:
        assert isa_set(level) == level
        for state in MXCSR_STATES:
            with mxcsr(state):
                node = LENS.lens_blur_node(image, *parameters)
                actual = OUTPUTS.ImageOutput().enforce(node)
                expected = OUTPUTS.ImageOutput().enforce(
                    B3_LENS.lens_blur_node(image, *parameters)
                )
            assert node.shape == (height, width, channels)
            assert node.dtype == np.float32 and node.strides == (
                4 * width,
                4,
                4 * height * width,
            )
            plane = node.base
            assert plane is not None and plane.base is None and plane.flags.owndata
            assert plane.shape == (channels, height, width) and plane.flags.c_contiguous
            assert not node.flags.writeable and not plane.flags.writeable
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert actual.strides == expected.strides, (level, state)
            assert actual.tobytes() == expected.tobytes(), (level, state)
            borrowed = passes(node.view(np.uint32))
            assert np.shares_memory(actual, node) == borrowed, (level, state)
            if borrowed and node.shape[2] > 1:
                assert actual is node
            bits = node.view(np.uint32)
            seen.add((state, borrowed))
            if name == "saturated":
                assert (bits == 0x3F800000).any()
            elif name == "nan":
                assert np.isnan(node).any()
            elif name == "denormal" and state == "default":
                assert ((bits != 0) & (bits < 0x00800000)).any()
    if name in ("nan", "denormal"):
        # NaN is refused in both states; a denormal is refused by default, and
        # under DAZ the lens arithmetic reads it as zero, so the output borrows.
        assert ("default", False) in seen
    else:
        assert seen == {("default", True), ("daz_ftz", True)}


def signed_image(rng, shape):
    """Values in [-0.5, 1.5), lowered by 2 in the left half and raised by 2 in the
    right, so a blur's outputs lie below 0 and above 1 away from the middle."""
    image = rng.uniform(-0.5, 1.5, shape).astype(np.float32)
    middle = shape[1] // 2
    image[:, :middle] -= 2
    image[:, middle:] += 2
    return image


def opt_in_cases():
    """(name, node, raw node, image, parameters) for Gaussian and Box Blur (Task
    3b2) and Dilate and Erode (Task 4). Gaussian: the gaussian and parallel-branches
    bench sigmas on the bench image, a large sigma on the downsample path (its last
    operation is still the blur), and sigma 0 on one axis (the copy path). Box:
    integer radii (box_blur), fractional (separable_box), an integer axis then a
    fractional one (box_blur, then separable_box) and large fractional radii
    (filter2d). Dilate and Erode: the morphology bench's forms (an ellipse through
    cn_morphology_complete, a cross through cn_filter_morphology), a rectangle and a
    cross on the signed image, a NaN (the exceptional path), denormals, four channels,
    and one channel as a 2-D image and as a trailing channel of 1. The CLAMPED cases' raw outputs leave [0, 1] (signed: below 0 on the
    left, above 1 on the right; bright: mostly above 1), so the clamp moves them;
    others hold a NaN or denormals, or have one or four channels."""
    rng = np.random.default_rng(20261004)
    bench = rng.random((384, 512, 3), dtype=np.float32)
    signed = signed_image(rng, (29, 31, 3))
    bright = rng.uniform(0.5, 2.5, (29, 31, 3)).astype(np.float32)
    nan = rng.random((19, 23, 3), dtype=np.float32)
    nan[5, 7, 1] = np.nan
    tiny = np.full((17, 21, 4), np.float32(3e-39), np.float32)
    gray = rng.random((17, 21), dtype=np.float32)
    rgba = rng.random((13, 11, 4), dtype=np.float32)
    gaussian = (GAUSSIAN.gaussian_blur_node, RAW_GAUSSIAN.gaussian_blur_node)
    box = (BOX.box_blur_node, RAW_BOX.box_blur_node)
    dilate = (DILATE.dilate_node, RAW_DILATE.dilate_node)
    erode = (ERODE.erode_node, RAW_ERODE.erode_node)
    shape = DILATE.MorphShape
    return [
        ("gaussian-bench", *gaussian, bench, (2.5, 1.25)),
        ("gaussian-branch-3", *gaussian, bench, (3, 3)),
        ("gaussian-branch-6", *gaussian, bench, (6, 6)),
        ("gaussian-downsample", *gaussian, bench, (30, 40)),
        ("gaussian-copy", *gaussian, signed, (0, 5)),
        ("gaussian-signed", *gaussian, signed, (1.5, 2.3)),
        ("gaussian-nan", *gaussian, nan, (2, 3)),
        ("gaussian-denormal", *gaussian, tiny, (1, 1)),
        ("gaussian-gray", *gaussian, gray, (2.5, 1.25)),
        ("gaussian-rgba", *gaussian, rgba, (11, 17)),
        ("box-integer", *box, bench, (3, 5)),
        ("box-fractional", *box, signed, (1.5, 3.2)),
        ("box-integer-fractional", *box, bright, (15, 1.2)),
        ("box-filter2d", *box, bright, (73.2, 169.7)),
        ("box-nan", *box, nan, (2, 3)),
        ("box-denormal", *box, tiny, (1.5, 1.5)),
        ("box-gray", *box, gray, (2, 2)),
        ("box-rgba", *box, rgba, (1.5, 2.5)),
        ("dilate-bench", *dilate, bench, (shape.ELLIPSE, 3, 2)),
        ("erode-bench", *erode, bench, (shape.CROSS, 2, 2)),
        ("dilate-signed", *dilate, signed, (shape.RECTANGLE, 1, 1)),
        ("erode-signed", *erode, signed, (shape.CROSS, 2, 1)),
        ("dilate-nan", *dilate, nan, (shape.ELLIPSE, 2, 1)),
        ("erode-denormal", *erode, tiny, (shape.RECTANGLE, 1, 1)),
        ("dilate-rgba", *dilate, rgba, (shape.CROSS, 1, 2)),
        ("dilate-gray", *dilate, gray, (shape.ELLIPSE, 2, 2)),
        ("erode-gray", *erode, gray[:, :, None], (shape.RECTANGLE, 1, 2)),
    ]


CLAMPED = (
    "gaussian-copy",
    "gaussian-signed",
    "box-fractional",
    "box-integer-fractional",
    "box-filter2d",
    "dilate-signed",
    "erode-signed",
)


@pytest.mark.parametrize(
    ("name", "node", "raw", "image", "parameters"),
    opt_in_cases(),
    ids=[case[0] for case in opt_in_cases()],
)
def test_opt_in_nodes_borrow_at_output_enforce(
    name, node, raw, image, parameters, levels
):
    # At every level and in both MXCSR states: the node returns its own frozen,
    # registered float32 array; ImageOutput.enforce returns that array exactly when
    # the scan passes; and in every case the enforced bytes are the raw path's (the
    # node before the opt-in, its raw result enforced).
    seen = set()
    for level in levels:
        assert isa_set(level) == level
        for state in MXCSR_STATES:
            with mxcsr(state):
                out = node(image, *parameters)
                actual = OUTPUTS.ImageOutput().enforce(out)
                raw_out = raw(image, *parameters)
                expected = OUTPUTS.ImageOutput().enforce(raw_out)
            assert out.dtype == np.float32 and out.base is None
            assert out.flags.owndata and out.flags.c_contiguous
            assert not out.flags.writeable
            assert out is not image and raw_out.flags.writeable
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes(), (level, state)
            borrowed = passes(out.view(np.uint32))
            assert np.shares_memory(actual, out) == borrowed, (level, state)
            if borrowed:
                assert actual is out
            seen.add((state, borrowed))
            raw_bits = raw_out.view(np.uint32)
            if name in CLAMPED:
                assert ((raw_out < 0) | (raw_out > 1)).any()
            elif name.endswith("nan"):
                assert np.isnan(out).any()
            elif name.endswith("denormal") and state == "default":
                assert ((raw_bits != 0) & (raw_bits < 0x00800000)).any()
    if name.endswith("nan"):
        assert seen == {("default", False), ("daz_ftz", False)}
    elif name.endswith("denormal"):
        # A denormal is refused by default; under DAZ the blur reads its input as
        # zero, so the output borrows.
        assert seen == {("default", False), ("daz_ftz", True)}
    else:
        assert seen == {("default", True), ("daz_ftz", True)}


def warning_image(signalling):
    """NaN, +-inf and -0 inputs, and with signalling a signalling NaN."""
    image = np.random.default_rng(31).random((13, 17, 3), dtype=np.float32)
    image[2, 3, 0] = np.nan
    image[5, 6, 1] = np.inf
    image[7, 8, 2] = -np.inf
    image[9, 1, 0] = np.float32(-0.0)
    if signalling:
        image.view(np.uint32)[11, 4, 1] = 0x7F812345
    return image


def warned(call):
    """The floating-point warnings of call() and of its output's enforce, and the
    enforced output."""
    with warnings.catch_warnings(record=True) as caught, np.errstate(all="warn"):
        warnings.simplefilter("always")
        output = OUTPUTS.ImageOutput().enforce(call())
    return [(item.category, str(item.message)) for item in caught], output


WARNING_CASES = [
    ("lens", LENS.lens_blur_node, B3_LENS.lens_blur_node, (3, 2, 1.7)),
    *[
        (name, node, raw, parameters)
        for name, node, raw, _, parameters in opt_in_cases()
        if name
        in (
            "gaussian-bench",
            "gaussian-downsample",
            "gaussian-copy",
            "box-integer",
            "box-fractional",
            "box-filter2d",
            "dilate-bench",
            "erode-bench",
        )
    ],
]


@pytest.mark.parametrize("state", MXCSR_STATES)
@pytest.mark.parametrize(
    ("name", "node", "raw", "parameters"),
    WARNING_CASES,
    ids=[case[0] for case in WARNING_CASES],
)
def test_no_float_warning_repeats_or_appears(name, node, raw, parameters, state):
    # The node with the enforce raises exactly the floating-point warnings of the
    # raw path (B3's store for lens; the node before its opt-in for Gaussian and Box).
    image = warning_image(signalling=name != "lens")
    with mxcsr(state):
        actual, output = warned(lambda: node(image, *parameters))
        expected, raw_output = warned(lambda: raw(image, *parameters))
    if name == "lens":
        # None can occur on either path: the enforce's float32 clip
        # (cn_pixels_convert_checked (0, 0, 1)) reports no event (its events come
        # only from integer outputs), lens's kernels report none, and its power
        # quiets a signalling NaN. A new warning must fail here, not match.
        assert expected == []
    # Gaussian and Box: the two-step path (freeze_normalized's in-place clip, then
    # the enforce) against the raw path's one conversion; on the copy path (sigma 0
    # on one axis) the signalling NaN reaches both conversions unquieted.
    assert actual == expected
    assert output.tobytes() == raw_output.tobytes()


def test_raw_callers_receive_unclamped_writable_values():
    # Unsharp mask, high pass, edge detection and the normal-map generator use
    # fast_gaussian_blur's raw values in arithmetic: each receives a writable array
    # that the enforce never borrows, with the bytes of normalized=False and values
    # outside [0, 1], which a clamp would have moved.
    image = signed_image(np.random.default_rng(613), (23, 29, 3))
    received = {}

    def spy(name):
        def fast_gaussian_blur(*args, **kwargs):
            result = image_utils.fast_gaussian_blur(*args, **kwargs)
            received.setdefault(name, []).append((args, kwargs, result))
            return result

        return fast_gaussian_blur

    nodes = {name: load_node(path) for name, path in RAW_CALLER_PATHS.items()}
    for name, module in nodes.items():
        vars(module).update(fast_gaussian_blur=spy(name))
    nodes["unsharp_mask"].unsharp_mask_node(image, 2.5, 1.3, 0)
    high_pass = nodes["high_pass"]
    high_pass.high_pass_node(image, high_pass.BlurMode.GAUSSIAN, 2.5, None, 0.7)
    edge = nodes["edge_detection"]
    edge.edge_detection_node(
        image,
        1.3,
        edge.Algorithm.DIFFERENCE_OF_GAUSSIAN,
        edge.GradientComponent.MAGNITUDE,
        1.5,
        3.2,
    )
    normal = nodes["normal_map_generator"]
    for blur_sharp in (-1.7, 1.7):  # blur, then sharpen
        normal.normal_map_generator_node(
            image,
            False,
            normal.HeightSource.AVERAGE_RGB,
            blur_sharp,
            0.2,
            1.7,
            normal.EdgeFilter.SOBEL,
            0.25,
            0.5,
            0.3,
            0.25,
            0,
            0,
            0,
            0,
            False,
            False,
            normal.AlphaOutput.NONE,
        )
    assert {name: len(calls) for name, calls in received.items()} == {
        "unsharp_mask": 1,
        "high_pass": 1,
        "edge_detection": 2,
        "normal_map_generator": 2,
    }
    for name, calls in received.items():
        for args, kwargs, result in calls:
            assert kwargs == {}, name
            assert result.dtype == np.float32 and result.flags.writeable, name
            assert native_buffers.normalized_readonly(result) is None, name
            raw = image_utils.fast_gaussian_blur(*args, normalized=False)
            assert result.tobytes() == raw.tobytes(), name
        assert any(((r < 0) | (r > 1)).any() for _, _, r in calls), name


def test_morphology_helper_returns_owned_one_channel_results():
    # Task 4 (controller ruling): native_filters.morphology allocates a one-channel
    # result (a 2-D image, or a trailing channel of 1) in its returned (h, w) shape
    # instead of reshaping an (h, w, 1) array, so the result owns its data and Dilate
    # and Erode can freeze it. Its bits are the installed chaiNNer's: cv2.dilate or
    # cv2.erode with the node's structuring element (values in [0, 1), both paths: an
    # ellipse, and a rectangle or cross through cn_filter_morphology).
    image = np.random.default_rng(97).random((19, 23), dtype=np.float32)
    for source in (image, image[:, :, None]):
        for shape in (cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE):
            for maximum in (False, True):
                out = native_filters.morphology(source, shape, 2, 2, maximum=maximum)
                assert out.shape == (19, 23) and out.dtype == np.float32
                assert out.base is None and out.flags.owndata and out.flags.writeable
                element = cv2.getStructuringElement(shape, (5, 5))
                operation = cv2.dilate if maximum else cv2.erode
                expected = operation(image, element, iterations=2)
                assert out.tobytes() == expected.tobytes(), (
                    source.ndim,
                    shape,
                    maximum,
                )


def test_morphology_helper_keeps_raw_writable_values():
    # CAS, background removal and edge detection call native_filters.morphology and
    # use its raw values: the opt-in is in the Dilate and Erode nodes only, so the
    # helper's result stays writable and unregistered, with its values outside [0, 1].
    image = signed_image(np.random.default_rng(101), (21, 25, 3))
    for shape in (cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE):
        for maximum in (False, True):
            out = native_filters.morphology(image, shape, 2, 1, maximum=maximum)
            assert out.flags.writeable and out.flags.owndata
            assert native_buffers.normalized_readonly(out) is None
            assert ((out < 0) | (out > 1)).any()


@pytest.mark.parametrize("state", MXCSR_STATES)
def test_freeze_normalized_clamp_is_the_enforce_conversion(state):
    # freeze_normalized(clamp=True) leaves the bytes converted_pixels (the enforce's
    # normalize) gives the same values, frozen and registered.
    words = (*BOUNDARY_WORDS, 0x80000001, 0x807FFFFF, 0xBF800000, 0xFF800000)
    words += (0xFFC54321, 0xFF812345, 0x7F7FFFFF, 0xFF7FFFFF, 0x3F7FFFFF, FILL)
    for count in (1, 7, 8, 17, 155):
        values = np.resize(np.array(words, np.uint32), count).view(np.float32)
        with mxcsr(state):
            expected = native_buffers.converted_pixels(values)
            out = native_buffers.freeze_normalized(values.copy(), clamp=True)
        assert expected is not None
        assert out.tobytes() == expected.tobytes(), count
        assert not out.flags.writeable
        borrowed = native_buffers.normalized_readonly(out)
        assert (borrowed is out) == passes(out.view(np.uint32)), count


def test_freeze_normalized_registers_only_a_fresh_writable_owner():
    fresh = np.full((4, 5, 3), 0.5, np.float32)
    assert native_buffers.freeze_normalized(fresh) is fresh
    assert not fresh.flags.writeable
    assert native_buffers.normalized_readonly(fresh) is fresh
    # A gapless permuted owner (F order, or converted_pixels' order K) is an owner too.
    fortran = np.asfortranarray(np.full((3, 4, 2), 0.5, np.float32))
    assert native_buffers.freeze_normalized(fortran) is fortran
    assert native_buffers.normalized_readonly(fortran) is fortran

    class Subclass(np.ndarray):
        pass

    # Every value is -1, which the clamp would make +0.
    owner = np.full((4, 6), -1, np.float32)
    subclass = Subclass((2, 2), np.float32)
    subclass[:] = -1
    unaligned = np.ndarray((4,), np.float32, bytearray(17), 1)
    unaligned[:] = -1
    readonly = np.full((2, 3), -1, np.float32)
    readonly.flags.writeable = False
    refused = [
        owner[1:],  # a view
        np.full((2, 3), -1, np.float64),
        subclass,
        # Refused by `base is not None` (a view of the bytearray), before the
        # alignment test: NumPy gives no base array that owns unaligned data.
        unaligned,
        readonly,
    ]
    for array in refused:
        writeable = array.flags.writeable
        for clamp in (False, True):
            with pytest.raises(ValueError, match="freeze_normalized"):
                native_buffers.freeze_normalized(array, clamp=clamp)
            assert array.flags.writeable == writeable
        assert (array == -1).all()  # refused before any clamp


def test_freeze_normalized_is_timed_under_the_profile():
    source = """
import json, sys
import numpy as np
from nodes.impl import native_buffers, native_profile

native_profile.reset()
out = native_buffers.freeze_normalized(np.full((4, 5, 3), 0.5, np.float32))
borrowed = native_buffers.normalized_readonly(out) is out
# Task 3b2: with clamp, the in-place conversion is timed under its export's name.
clamped = native_buffers.freeze_normalized(
    np.full((4, 5, 3), 1.5, np.float32), clamp=True
)
names = ("freeze_normalized", "cn_pixels_normalized_f32", "cn_pixels_convert_checked")
table = native_profile.snapshot()
print(json.dumps({
    "calls": {name: table[name]["calls"] for name in sorted(names) if name in table},
    "wrapped": hasattr(native_buffers.freeze_normalized, "__wrapped__"),
    "borrowed": borrowed,
    "clamped": clamped.tobytes() == np.ones((4, 5, 3), np.float32).tobytes(),
    "modules": [m for m in ("PIL", "torch") if m in sys.modules],
}))
"""
    on, off = child(source, profile=True), child(source)
    assert on == {
        "calls": {
            "cn_pixels_convert_checked": 1,
            "cn_pixels_normalized_f32": 1,
            "freeze_normalized": 2,
        },
        "wrapped": True,
        "borrowed": True,
        "clamped": True,
        "modules": [],
    }
    assert off == {
        "calls": {},
        "wrapped": False,
        "borrowed": True,
        "clamped": True,
        "modules": [],
    }


def layout_results():
    """(name, a node's float32 or uint8 return): transposed, flipped and planar
    results, as the layouts nodes return (D-7)."""
    rng = np.random.default_rng(907)
    plane = rng.uniform(-0.5, 1.5, (3, 7, 5)).astype(np.float32)
    plane4 = rng.uniform(-0.5, 1.5, (4, 6, 9)).astype(np.float32)
    hwc = rng.uniform(-0.5, 1.5, (6, 9, 3)).astype(np.float32)
    gray = rng.uniform(-0.5, 1.5, (6, 9)).astype(np.float32)
    return [
        ("planar3", plane.transpose(1, 2, 0)),
        ("planar4", plane4.transpose(1, 2, 0)),
        ("planar1", plane[:1].transpose(1, 2, 0)),
        ("transposed", hwc.swapaxes(0, 1)),
        ("transposed_gray", gray.T),
        ("flipped", hwc[::-1, ::-1]),
        ("flipped_channels", hwc[:, :, ::-1]),
        ("fortran", np.asfortranarray(hwc)),
        ("strided", hwc[::2, 1::2]),
        ("uint8_planar", (plane * 100).astype(np.uint8).transpose(1, 2, 0)),
        ("uint16_transposed", (hwc * 1000).astype(np.uint16).swapaxes(0, 1)),
    ]


@pytest.mark.parametrize(
    ("name", "value"), layout_results(), ids=[name for name, _ in layout_results()]
)
def test_enforced_layout_is_np_clips(name, value):
    # D-7: ImageOutput.enforce hands on np.clip's order-K layout, upstream's (its
    # normalize clips and converts with NumPy, which keeps the input's axis order);
    # values bit for bit.
    expected = upstream_enforce(value)
    actual = OUTPUTS.ImageOutput().enforce(value)
    assert actual.strides == expected.strides, name
    assert_enforced_equal(value, value)
    # The converted output is registered, so a later enforce of it borrows it.
    assert native_buffers.normalized_readonly(actual) is actual


def test_borrowed_permuted_views_alias_only_known_owners():
    # A gapless permuted view is borrowed only from a registered frozen owner; a view
    # of an unknown array, or a writable one, is converted as before.
    plane = np.full((3, 4, 5), 0.5, np.float32)
    native_buffers.freeze_normalized(plane)
    view = plane.transpose(1, 2, 0)
    assert native_buffers.normalized_readonly(view) is view
    assert OUTPUTS.ImageOutput().enforce(view) is view
    unknown = np.full((3, 4, 5), 0.5, np.float32)
    unknown.flags.writeable = False
    enforced = OUTPUTS.ImageOutput().enforce(unknown.transpose(1, 2, 0))
    assert not np.shares_memory(enforced, unknown)
    assert enforced.strides == view.strides
    writable = np.full((3, 4, 5), 0.5, np.float32)
    assert native_buffers.normalized_readonly(writable.transpose(1, 2, 0)) is None
    # A view with gaps, even of a registered owner, is converted.
    assert native_buffers.normalized_readonly(plane[:, ::2].transpose(1, 2, 0)) is None
