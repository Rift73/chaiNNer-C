"""Blurs, sharpening and rank filters on RGBA with transparency: no fringe from the colour under transparent pixels, on
both executors; images without transparency run as they always did, to the bit."""

import ctypes as ct

import numpy as np
import pytest
from native_differential import assert_native_matches, node_data
from native_images import image, seed_of

import api
import process
from nodes.impl import native_resample
from nodes.impl.image_utils import fast_gaussian_blur
from nodes.impl.native import lib, ptr
from nodes.impl.native_bilateral import bilateral, surface
from nodes.impl.native_convolution import box_blur
from nodes.impl.native_filters import morphology
from nodes.impl.native_neighborhood import median
from nodes.impl.native_resample import finish_alpha, has_transparency
from nodes.impl.native_unsharp import weighted

GAUSSIAN = "chainner:image:gaussian_blur"
BOX = "chainner:image:blur"
LENS = "chainner:image:lens_blur"
MEDIAN = "chainner:image:median_blur"
SURFACE = "chainner:image:bilateral_blur"
UNSHARP = "chainner:image:sharpen"
DILATE = "chainner:image:dilate"
ERODE = "chainner:image:erode"
RECT, CROSS, ELLIPSE = 0, 1, 2
LINEAR = [
    (GAUSSIAN, [1, 1]),
    (GAUSSIAN, [2.5, 0]),
    (GAUSSIAN, [12, 12]),
    (BOX, [1, 1]),
    (BOX, [2.5, 1.5]),
    (BOX, [16, 2.5]),
    (LENS, [3, 5, 5.0]),
    (LENS, [2, 1, 1.0]),
]
ALL = [
    *LINEAR,
    (MEDIAN, [1]),
    (MEDIAN, [2]),
    (MEDIAN, [4]),
    (SURFACE, [4, 25, 25]),
    (SURFACE, [3, 1000, 25]),
    (UNSHARP, [3, 1, 0]),
    (UNSHARP, [1.5, 2.5, 5]),
    (DILATE, [RECT, 2, 1]),
    (DILATE, [ELLIPSE, 3, 2]),
    (ERODE, [RECT, 2, 1]),
    (ERODE, [CROSS, 2, 2]),
]


def run_body(schema, inputs):
    """The node as the executors run it, the Python body."""
    node = node_data(schema)
    node_id = api.NodeId("x")
    values = process.enforce_inputs(list(inputs), node, node_id, [])
    return process.enforce_output(node.run(*values), node).output[0]


def result_of(schema, inputs):
    """The result, both executors agreeing to the bit where the node has a native entry that takes these inputs."""
    if schema != SURFACE and not (schema == GAUSSIAN and min(inputs[1:]) >= 11):
        assert_native_matches(schema, inputs)
    return np.asarray(run_body(schema, inputs))


def disc(height=48, width=64, soft=True):
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    distance = 16 - np.hypot(xx - width / 2, yy - height / 2)
    alpha = np.clip(distance / 3 + 0.5, 0, 1) if soft else (distance > 0).astype(np.float32)
    return alpha.astype(np.float32)


def rgba(colour, alpha, under=None):
    """colour where alpha > 0, under (black by default) where it is 0."""
    h, w = alpha.shape
    base = np.broadcast_to(np.asarray(colour, np.float32), (h, w, 3))
    hidden = np.zeros((h, w, 3), np.float32) if under is None else np.broadcast_to(under, (h, w, 3))
    return np.dstack([np.where(alpha[:, :, None] > 0, base, hidden), alpha]).astype(np.float32)


def noise(h, w, c, seed=9):
    return np.random.default_rng(seed).random((h, w, c), dtype=np.float32)


def test_gaussian_on_white_next_to_transparent_black_has_no_dark_fringe():
    row = np.zeros((1, 9, 4), np.float32)
    row[:, 4:] = 1
    out = result_of(GAUSSIAN, [row, 1, 0])[0]
    np.testing.assert_allclose(out[2:7, 3], [0.059, 0.301, 0.699, 0.941, 0.995], atol=2e-3)
    np.testing.assert_allclose(out[out[:, 3] > 0, :3], 1, atol=1e-6)


@pytest.mark.parametrize(("schema", "arguments"), ALL)
@pytest.mark.parametrize("soft", [True, False])
def test_white_stays_white_where_anything_is_seen(schema, arguments, soft):
    out = result_of(schema, [rgba((1, 1, 1), disc(soft=soft)), *arguments])
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[out[:, :, 3] > 0][:, :3], 1, atol=2e-5)


@pytest.mark.parametrize(("schema", "arguments"), ALL)
@pytest.mark.parametrize("soft", [True, False])
@pytest.mark.parametrize("hidden", ["black", "white", "noise"])
def test_flat_colour_stays_flat_whatever_the_hidden_colour(schema, arguments, soft, hidden):
    alpha = disc(soft=soft)
    under = {"black": np.float32(0), "white": np.float32(1), "noise": noise(*alpha.shape, 3)}[hidden]
    out = result_of(schema, [rgba((0.2, 0.6, 0.9), alpha, under), *arguments])
    assert np.isfinite(out).all()
    tolerance = 1 / 255 if schema == MEDIAN and arguments[0] > 2 else 2e-5
    seen = out[out[:, :, 3] > 0][:, :3]
    np.testing.assert_allclose(seen, np.broadcast_to((0.2, 0.6, 0.9), seen.shape), atol=tolerance)


@pytest.mark.parametrize(("schema", "arguments"), [(GAUSSIAN, [1, 1]), (BOX, [1, 1]), (LENS, [3, 5, 5.0]), (DILATE, [RECT, 1, 1])])
def test_results_stay_frozen_for_the_output_enforce(schema, arguments):
    node = node_data(schema)
    values = process.enforce_inputs([rgba((1, 1, 1), disc()), *arguments], node, api.NodeId("x"), [])
    assert not node.run(*values).flags.writeable


@pytest.mark.parametrize(("schema", "arguments"), ALL)
def test_alpha_zero_holds_colour_zero_and_the_hidden_colour_takes_no_part(schema, arguments):
    alpha = disc()
    colour = noise(*alpha.shape, 3, seed=11)
    black = result_of(schema, [rgba(colour, alpha), *arguments])
    white = result_of(schema, [rgba(colour, alpha, np.float32(1)), *arguments])
    assert np.isfinite(black).all()
    assert (black[black[:, :, 3] == 0][:, :3] == 0).all()
    assert black.tobytes() == white.tobytes()


@pytest.mark.parametrize(("schema", "arguments"), [a for a in LINEAR if a[0] != LENS])
def test_linear_blurs_composite_as_the_blur_of_the_composite(schema, arguments):
    # Over black, the result is the same filter on the premultiplied colours: no colour gained or lost.
    source = rgba(noise(48, 64, 3), disc() * 0.75)
    out = result_of(schema, [source, *arguments])
    colours = np.array(result_of(schema, [(source[:, :, :3] * source[:, :, 3:]).astype(np.float32), *arguments]))
    seen = out[:, :, 3] > 1e-3
    np.testing.assert_allclose((out[:, :, :3] * out[:, :, 3:])[seen], colours[seen], atol=1e-4)


@pytest.mark.parametrize(("schema", "arguments"), [a for a in ALL if a[0] not in (SURFACE, UNSHARP)])
def test_alpha_is_the_filter_of_alpha(schema, arguments):
    source = rgba(noise(48, 64, 3), disc())
    expected = result_of(schema, [source[:, :, 3], *arguments])
    tolerance = 1e-6 if schema != LENS else 1e-4
    np.testing.assert_allclose(result_of(schema, [source, *arguments])[:, :, 3], expected, atol=tolerance)


def test_gaussian_tail_keeps_its_colour():
    row = np.zeros((1, 201, 4), np.float32)
    row[:, 100:] = (0.25, 0.5, 0.75, 1)
    out = result_of(GAUSSIAN, [row, 7, 0])[0]
    assert ((out[:, 3] > 0) & (out[:, 3] < 1e-4)).any()
    seen = out[out[:, 3] > 1e-30, :3]
    np.testing.assert_allclose(seen, np.broadcast_to((0.25, 0.5, 0.75), seen.shape), atol=1e-3)


@pytest.mark.parametrize("threshold", [0, 5])
def test_unsharp_output_alpha_and_inner_colour_edges(threshold):
    # Alpha is sharpened as it always was: the mask's blur of alpha is the same. Over flat alpha a colour edge sharpens
    # as on RGB.
    colour = noise(48, 64, 3)
    alpha = disc()
    source = rgba(colour, alpha)
    out = result_of(UNSHARP, [source, 3, 1.5, threshold])
    expected = result_of(UNSHARP, [alpha, 3, 1.5, threshold])
    np.testing.assert_allclose(out[:, :, 3], expected, atol=1e-6)
    half = np.dstack([colour, np.full(alpha.shape, 0.5, np.float32)])
    np.testing.assert_allclose(result_of(UNSHARP, [half, 3, 1.5, threshold])[:, :, :3], result_of(UNSHARP, [colour, 3, 1.5, threshold]), atol=2e-5)


def test_surface_keeps_edges_inside_the_visible_colour():
    # Black and white halves under one alpha: the colour sigma still stops the blur at the edge.
    colour = np.zeros((32, 32, 3), np.float32)
    colour[:, 16:] = 1
    source = np.dstack([colour, np.full((32, 32), 0.5, np.float32)])
    out = run_body(SURFACE, [source, 4, 25, 25])
    np.testing.assert_allclose(out[:, :, :3], colour, atol=1e-6)


def test_surface_matches_bilateral_on_a_barely_transparent_image():
    # The same algorithm as the RGB route, off by the interpolated table's rounding only.
    colour = noise(48, 64, 3)
    alpha = np.full((48, 64), 1.0, np.float32)
    alpha[0, 0] = np.nextafter(np.float32(1), np.float32(0))
    out = surface(np.dstack([colour, alpha]), 3, 50, 5)
    np.testing.assert_allclose(out[:, :, :3], bilateral(colour, 3, 50 / 255, 5), atol=1e-3)
    # Black against white, the table's top: a large colour sigma still mixes them.
    colour = (noise(48, 64, 3) > 0.5).astype(np.float32)
    out = surface(np.dstack([colour, alpha]), 3, 1000, 5)
    np.testing.assert_allclose(out[:, :, :3], bilateral(colour, 3, 1000 / 255, 5), atol=1e-3)


def test_dilate_grows_the_visible_colour_not_the_hidden_one():
    # Black text on hidden white: it used to turn its own border white.
    alpha = disc(soft=False)
    out = result_of(DILATE, [rgba((0, 0, 0), alpha, np.float32(1)), RECT, 2, 1])
    assert (out[alpha > 0][:, :3] == 0).all()
    assert (out[:, :, 3] > alpha).any()


# Images without transparency: the same arrays the unchanged helpers give.
PLAIN = {
    GAUSSIAN: lambda i: fast_gaussian_blur(i, 2.5, 1.5),
    BOX: lambda i: box_blur(i, 2, 3),
    MEDIAN: lambda i: median(i, 2),
    DILATE: lambda i: morphology(i, RECT, 2, 1, maximum=True),
    ERODE: lambda i: morphology(i, RECT, 2, 1, maximum=False),
    UNSHARP: lambda i: weighted(i, fast_gaussian_blur(i, 3), 1.5),
}
ARGUMENTS = {GAUSSIAN: [2.5, 1.5], BOX: [2, 3], MEDIAN: [2], DILATE: [RECT, 2, 1], ERODE: [RECT, 2, 1], UNSHARP: [3, 1.5, 0]}


@pytest.mark.parametrize("schema", list(PLAIN))
@pytest.mark.parametrize("kind", ["rgb", "gray", "opaque"])
def test_no_transparency_is_bit_identical(schema, kind):
    colour = noise(48, 64, 3, seed=seed_of(schema, kind))
    source = {
        "rgb": colour,
        "gray": colour[:, :, 0],
        "opaque": np.dstack([colour, np.ones((48, 64), np.float32)]),
    }[kind]
    expected = np.clip(PLAIN[schema](source), 0, 1)
    assert result_of(schema, [source, *ARGUMENTS[schema]]).tobytes() == expected.tobytes()


def test_has_transparency():
    assert not has_transparency(np.zeros((4, 4, 3), np.float32))
    assert not has_transparency(np.zeros((4, 4), np.float32))
    assert not has_transparency(np.zeros((0, 4, 4), np.float32))
    assert not has_transparency(np.ones((4, 4, 4), np.float64))
    opaque = np.ones((300, 300, 4), np.float32)
    assert not has_transparency(opaque)
    for y, x in [(0, 0), (299, 299), (150, 7), (13, 200)]:
        for value in (0.0, np.nextafter(np.float32(1), np.float32(0)), 1.5, np.nan):
            single = opaque.copy()
            single[y, x, 3] = value
            assert has_transparency(single)
    colour_only = opaque.copy()
    colour_only[:, :, :3] = 0.5
    assert not has_transparency(colour_only)
    assert has_transparency(np.dstack([np.ones((3, 3, 3), np.float32), np.zeros((3, 3), np.float32)])[::2, ::-1])


def test_finish_alpha():
    source = np.array([[[0.25, 0.5, 0.75, 1], [0.1, 0.2, 0.3, 0.5], [0.7, 0.7, 0.7, 0], [0.3, 0.3, 0.3, -0.5]]], np.float32)
    divided = finish_alpha(source, divide=True)
    np.testing.assert_allclose(divided[0, 0], (0.25, 0.5, 0.75, 1))
    np.testing.assert_allclose(divided[0, 1], (0.2, 0.4, 0.6, 0.5))
    assert (divided[0, 2] == (0, 0, 0, 0)).all() and (divided[0, 3] == (0, 0, 0, -0.5)).all()
    kept = finish_alpha(source, divide=False)
    assert (kept[0, :2] == source[0, :2]).all() and (kept[0, 2:, :3] == 0).all()
    tiny = np.array([[[1e-40, 1e-40, 1e-40, 1e-40], [1, 0, 0, 1e-45], [1e-30, 0, 0, 1e-30]]], np.float32)
    out = finish_alpha(tiny, divide=True)
    assert np.isfinite(out).all() and out.max() <= 1
    np.testing.assert_allclose(out[0, 2, 0], 1, atol=1e-6)


def test_kernels_refuse_bad_arguments():
    source = np.ones((2, 2, 4), np.float32)
    out = np.empty_like(source)
    finish = native_resample._api().cn_alpha_finish
    assert finish(ptr(source), ptr(out), 0, 1) == 1
    assert finish(ptr(source), ptr(out), 1, 2) == 1
    assert finish(None, ptr(out), 1, 1) == 1
    assert finish(ptr(source), None, 1, 1) == 1
    assert finish(ptr(source), ptr(out), ct.c_size_t(-1).value, 1) == 2
    scan = native_resample._api().cn_alpha_transparent
    flag = ct.c_int()
    assert scan(None, 1, ct.byref(flag)) == 1
    assert scan(ptr(source), 1, None) == 1
    assert scan(ptr(source), ct.c_size_t(-1).value, ct.byref(flag)) == 2
    straight = lib().cn_bilateral_straight_f32
    straight.argtypes = [ct.c_void_p, ct.c_void_p, *[ct.c_size_t] * 3, ct.c_double, ct.c_double]
    straight.restype = ct.c_int
    rgb = np.empty((2, 2, 3), np.float32)
    for a, b, h, w, radius, color, space in [
        (None, rgb, 2, 2, 1, 1, 1),
        (source, None, 2, 2, 1, 1, 1),
        (source, rgb, 0, 2, 1, 1, 1),
        (source, rgb, 2, 0, 1, 1, 1),
        (source, rgb, 2, 2, 0, 1, 1),
        (source, rgb, 2, 2, 101, 1, 1),
        (source, rgb, 2, 2, 1, 0, 1),
        (source, rgb, 2, 2, 1, 1, 0),
        (source, rgb, 2, 2, 1, np.nan, 1),
        (source, rgb, 2, 2, 1, 1, np.inf),
    ]:
        pointers = [None if array is None else array.ctypes.data for array in (a, b)]
        assert straight(*pointers, h, w, radius, color, space) == 1


@pytest.mark.parametrize(
    ("schema", "arguments"),
    [(GAUSSIAN, [1.5, 1.5]), (BOX, [1.5, 1]), (LENS, [2, 2, 3.0]), (MEDIAN, [1]), (MEDIAN, [3]), (UNSHARP, [1, 1, 0]), (UNSHARP, [1, 1, 4]), (DILATE, [RECT, 1, 1]), (ERODE, [ELLIPSE, 2, 1])],
)
@pytest.mark.parametrize("size", [(1, 1), (3, 5), (17, 31), (64, 257)])
def test_executors_agree_on_noise_alpha(schema, arguments, size):
    # Every pixel partly transparent: one Python body and one native entry, to the bit, tiny sizes included.
    assert_native_matches(schema, [image(seed_of("alpha", schema, size), *size, 4, edges=False), *arguments])
