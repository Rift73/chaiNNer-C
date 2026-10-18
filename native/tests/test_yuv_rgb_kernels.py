"""cn_plane_resample_axis, cn_yuv_to_bgr and cn_bgr_to_yuv: plane_resample.py's
resampler and yuv_rgb.py's matrix step in C. At every level this CPU has, they give
the frozen NumPy forms (reference_yuv_planes, lane B's modules before the port) bit
for bit, NaN positions included (LLVM leaves a NaN's payload unspecified), for every
kernel, geometry, matrix, subsampling and siting, in every layout the bridge takes."""

from __future__ import annotations

import ast
import ctypes as ct
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_isa_dispatch import isa_get, isa_set

from nodes.impl import plane_resample, yuv_rgb
from nodes.impl.colour_info import ChromaLocation, Matrix, chroma_size_for
from nodes.impl.native_planes import _api, bgr_to_yuv, yuv_to_bgr
from nodes.impl.plane_resample import Kernel, resample_plane, resample_plane_biased

REF = Path(__file__).with_name("reference_yuv_planes")


def load_body(path: Path, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__package__ = "nodes.impl"
    sys.modules[name] = module
    tree = ast.parse(path.read_text(encoding="utf-8"))
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


reference_resample = load_body(REF / "plane_resample.py", "reference_plane_resample")
reference_yuv = load_body(REF / "yuv_rgb.py", "reference_yuv_rgb")
# The frozen yuv_rgb resamples with the frozen resampler, not the ported one.
reference_yuv.__dict__.update(
    resample_plane=reference_resample.resample_plane, Kernel=reference_resample.Kernel
)

# (input width, input height, resample_plane's keyword arguments): lane B's zimg cases,
# then widths with 16-, 8- and scalar column groups, one-pixel planes and far
# downscales whose kernels span many taps.
CASES: dict[str, tuple[int, int, dict[str, Any]]] = {
    "up": (37, 23, {"width": 80, "height": 50}),
    "down": (37, 23, {"width": 15, "height": 9}),
    "down_far": (64, 61, {"width": 5, "height": 4}),
    "tiny_up": (3, 2, {"width": 16, "height": 11}),
    "shifted": (37, 23, {"width": 37, "height": 23, "src_left": 0.25, "src_top": -0.5}),
    "chroma_left": (19, 12, {"width": 37, "height": 23, "src_left": 0.25}),
    "windowed": (
        37,
        23,
        {
            "width": 41,
            "height": 30,
            "src_left": 3.3,
            "src_top": 1.7,
            "src_width": 20.5,
            "src_height": 12.25,
        },
    ),
    "windowed_down": (
        40,
        30,
        {"width": 9, "height": 7, "src_left": -2.5, "src_width": 44.0},
    ),
    "one_axis": (37, 23, {"width": 52, "height": 23}),
    "columns_only": (37, 23, {"width": 37, "height": 61}),
    "groups": (61, 13, {"width": 91, "height": 29}),
    "single": (1, 1, {"width": 7, "height": 5}),
    "one_column": (1, 40, {"width": 1, "height": 17}),
    "far": (300, 90, {"width": 13, "height": 7, "src_left": 1.5}),
}


@pytest.fixture
def levels():
    """Every level this CPU has, scalar first; the original level afterwards."""
    effective, _, cpu = isa_get()
    try:
        yield range(cpu + 1)
    finally:
        assert isa_set(effective) == effective


def random_plane(width: int, height: int, specials: bool = True) -> np.ndarray:
    """Values well outside 0-1, and (specials) a NaN, infinities, zeros of both signs
    and a subnormal."""
    rng = np.random.default_rng(width * 1000 + height)
    plane = (rng.standard_normal((height, width)) * 2 + 0.5).astype(np.float32)
    if specials and plane.size > 16:
        flat = plane.reshape(-1)
        flat[rng.choice(flat.size, 6, replace=False)] = [
            np.nan,
            np.inf,
            -np.inf,
            -0.0,
            0.0,
            np.float32(1e-40),
        ]
    return plane


def assert_same(actual: np.ndarray, expected: np.ndarray, message: str = "") -> None:
    """Equal bits wherever expected is not NaN (signed zeros included), NaN where it is."""
    assert actual.shape == expected.shape, message
    assert actual.dtype == expected.dtype == np.float32, message
    nan = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), nan, err_msg=message)
    np.testing.assert_array_equal(
        np.where(nan, 0, actual).view(np.uint32),
        np.where(nan, 0, expected).view(np.uint32),
        err_msg=message,
    )


def reference_kernel(kernel: Kernel):
    return reference_resample.Kernel[kernel.name]


@pytest.mark.parametrize("kernel", list(Kernel))
@pytest.mark.parametrize("case", list(CASES))
def test_resample_matches_numpy(levels, kernel: Kernel, case: str):
    w, h, args = CASES[case]
    plane = random_plane(w, h)
    with np.errstate(invalid="ignore", over="ignore"):
        expected = reference_resample.resample_plane(
            plane, kernel=reference_kernel(kernel), **args
        )
    buffer = np.full((h, w + 5), np.nan, np.float32)
    buffer[:, :w] = plane
    for level in levels:
        assert isa_set(level) == level
        for form in (plane, buffer[:, :w], np.asfortranarray(plane)):
            assert_same(
                resample_plane(form, kernel=kernel, **args), expected, str(level)
            )


@pytest.mark.parametrize("kernel", [Kernel.BILINEAR, Kernel.SPLINE36])
def test_biases_equal_separate_operations(levels, kernel: Kernel):
    plane = random_plane(37, 23)
    for w, h, args in (CASES["up"], CASES["one_axis"], CASES["columns_only"]):
        arguments = (args["width"], args["height"], kernel, 0.25, 0.0, None, None)
        frozen = (args["width"], args["height"], reference_kernel(kernel), 0.25)
        for bias_in, bias_out in ((0.5, 0), (0, 0.5), (0.5, 0.5)):
            with np.errstate(invalid="ignore", over="ignore"):
                expected = reference_resample.resample_plane(
                    plane[:h, :w] - np.float32(bias_in), *frozen
                ) + np.float32(bias_out)
            for level in levels:
                assert isa_set(level) == level
                assert_same(
                    resample_plane_biased(plane[:h, :w], *arguments, bias_in, bias_out),
                    expected,
                    f"{level} {bias_in} {bias_out}",
                )


def test_taps_are_the_numpy_taps_cached_and_read_only():
    for kernel in Kernel:
        for args in ((37, 80, 0.25, 37.0), (64, 5, 0.0, 64.0), (40, 9, -2.5, 44.0)):
            index, weights = plane_resample._taps(kernel, *args)
            frozen_index, frozen_weights = reference_resample._taps(
                reference_kernel(kernel), *args
            )
            np.testing.assert_array_equal(index, frozen_index.T)
            np.testing.assert_array_equal(
                weights.view(np.uint32), frozen_weights.T.view(np.uint32)
            )
            assert index.dtype == np.int32 and index.flags.c_contiguous
            assert not index.flags.writeable and not weights.flags.writeable
            assert plane_resample._taps(kernel, *args)[0] is index


def test_matrix_steps_match_numpy(levels):
    rng = np.random.default_rng(3)
    matrix = rng.standard_normal((3, 3)).astype(np.float32)
    for width in (1, 7, 8, 29, 64):
        y, u, v = (random_plane(width, 11 + k)[:11] for k in range(3))
        wide = np.zeros((11, width + 3), np.float32)
        wide[:, :width] = y
        # pixels 4 values apart: the bridge copies them
        buffer = np.zeros((11, width, 4), np.float32)
        buffer[..., :3] = rng.standard_normal((11, width, 3))
        image = buffer[..., :3]
        image[0, 0] = [np.nan, np.inf, -np.inf]
        image[-1, -1, :2] = [-0.0, 1e-40]
        for centre in (0, 0.5):
            with np.errstate(invalid="ignore", over="ignore"):
                cb, cr = (p - np.float32(centre) if centre else p for p in (u, v))
                expected = np.empty((11, width, 3), np.float32)
                for channel, row in zip((2, 1, 0), matrix, strict=True):
                    expected[..., channel] = row[0] * y + row[1] * cb + row[2] * cr
                b, g, r = (image[..., i].astype(np.float32) for i in range(3))
                planes = [row[0] * r + row[1] * g + row[2] * b for row in matrix]
                planes[1:] = [
                    p + np.float32(centre) if centre else p for p in planes[1:]
                ]
            for level in levels:
                assert isa_set(level) == level
                message = f"{level} {width} {centre}"
                assert_same(yuv_to_bgr(y, u, v, matrix, centre), expected, message)
                cropped = yuv_to_bgr(
                    wide[:, :width], np.asfortranarray(u), v, matrix, centre
                )
                assert_same(cropped, expected, message)
                for actual, plane in zip(
                    bgr_to_yuv(image, matrix, centre), planes, strict=True
                ):
                    assert_same(actual, plane, message)
                for actual, plane in zip(
                    bgr_to_yuv(np.ascontiguousarray(image), matrix, centre),
                    planes,
                    strict=True,
                ):
                    assert_same(actual, plane, message)


MATRICES = [m for m in Matrix if m not in (Matrix.AUTO, Matrix.BT2020_CL, Matrix.ICTCP)]
SUBSAMPLINGS = [(0, 0), (1, 0), (1, 1), (0, 1), (2, 0), (2, 2)]


def _planes(luma: tuple[int, int], subsampling: tuple[int, int]):
    chroma = chroma_size_for(luma, subsampling)
    return (
        random_plane(*luma, specials=False),
        random_plane(*chroma, specials=False) * 0.2 + 0.4,
        random_plane(chroma[0], chroma[1] + 1, specials=False)[1:] * 0.2 + 0.4,
    )


@pytest.mark.parametrize(
    ("matrix", "subsampling"),
    [
        (matrix, subsampling)
        for matrix in MATRICES
        for subsampling in SUBSAMPLINGS
        if matrix is not Matrix.RGB or subsampling == (0, 0)
    ],
)
def test_conversions_match_numpy(levels, matrix: Matrix, subsampling: tuple[int, int]):
    for luma in ((37, 23), (64, 18)):
        y, u, v = _planes(luma, subsampling)
        image = np.random.default_rng(luma[0]).uniform(-0.2, 1.2, (luma[1], luma[0], 3))
        image = image.astype(np.float32)
        for location in ChromaLocation:
            if location is ChromaLocation.AUTO:
                continue
            for kernel in (Kernel.SPLINE36, Kernel.POINT, Kernel.LANCZOS):
                frozen = reference_kernel(kernel)
                expected_rgb = reference_yuv.planes_to_rgb(
                    y, u, v, matrix, location, frozen
                )
                expected_planes = reference_yuv.rgb_to_planes(
                    image, matrix, location, subsampling, frozen
                )
                for level in levels:
                    assert isa_set(level) == level
                    message = f"{level} {luma} {location} {kernel}"
                    assert_same(
                        yuv_rgb.planes_to_rgb(y, u, v, matrix, location, kernel),
                        expected_rgb,
                        message,
                    )
                    actual = yuv_rgb.rgb_to_planes(
                        image, matrix, location, subsampling, kernel
                    )
                    for plane, expected in zip(actual, expected_planes, strict=True):
                        assert_same(plane, expected, message)


@pytest.mark.parametrize("kernel", list(Kernel))
def test_every_kernel_converts_as_numpy(kernel: Kernel):
    y, u, v = _planes((61, 29), (1, 1))
    image = np.random.default_rng(5).uniform(0, 1, (29, 61, 3)).astype(np.float32)
    location = ChromaLocation.TOP_LEFT
    frozen = reference_kernel(kernel)
    assert_same(
        yuv_rgb.planes_to_rgb(y, u, v, Matrix.BT709, location, kernel),
        reference_yuv.planes_to_rgb(y, u, v, Matrix.BT709, location, frozen),
    )
    actual = yuv_rgb.rgb_to_planes(image, Matrix.BT2020_NCL, location, (1, 1), kernel)
    expected = reference_yuv.rgb_to_planes(
        image, Matrix.BT2020_NCL, location, (1, 1), frozen
    )
    for plane, reference in zip(actual, expected, strict=True):
        assert_same(plane, reference)


def test_refusals():
    buffer = np.zeros(8192, np.float32)
    base = buffer.ctypes.data
    index = np.zeros((2, 4), np.int32)
    weights = np.full((2, 4), 0.5, np.float32)
    resample = _api().cn_plane_resample_axis
    i, w = index.ctypes.data, weights.ctypes.data
    # 8 x 8 plane, rows resampled to 4: out 8 x 4
    assert resample(base, 8, 8, 8, 0, i, w, 2, 4, 0.0, 0.0, base + 16384) == 0
    assert resample(base, 8, 8, 8, 0, i, w, 2, 4, 0.0, 0.0, base + 128) != 0
    assert resample(base, 8, 8, 9, 0, i, w, 2, 4, 0.0, 0.0, base + 16384) != 0
    assert resample(base + 2, 8, 8, 8, 0, i, w, 2, 4, 0.0, 0.0, base + 16384) != 0
    assert resample(base, 8, 8, 8, 0, i, w, 0, 4, 0.0, 0.0, base + 16384) != 0
    assert resample(base, 8, 8, 8, 0, i, i, 2, 4, 0.0, 0.0, i) != 0
    for bad in (8, -1):
        index[1, 2] = bad
        assert resample(base, 8, 8, 8, 0, i, w, 2, 4, 0.0, 0.0, base + 16384) != 0
    index[1, 2] = 7  # the columns are resampled: the axis is the 8 rows
    assert resample(base, 8, 8, 8, 1, i, w, 2, 4, 0.0, 0.0, base + 16384) == 0
    matrix = np.eye(3, dtype=np.float32).ctypes.data
    to_bgr, to_yuv = _api().cn_yuv_to_bgr, _api().cn_bgr_to_yuv
    planes = [base, 64, base + 256, 64, base + 512, 64]
    assert to_bgr(*planes, 4, 4, matrix, 0.0, base + 16384) == 0
    assert to_bgr(*planes, 4, 4, matrix, 0.0, base + 300) != 0
    assert to_bgr(*planes, 4, 65, matrix, 0.0, base + 16384) != 0
    assert to_bgr(*planes, 4, 4, matrix + 2, 0.0, base + 16384) != 0
    outputs = [base + 16384, base + 16384 + 64, base + 16384 + 128]
    assert to_yuv(base, 12, 4, 4, matrix, 0.0, *outputs) == 0
    assert to_yuv(base, 11, 4, 4, matrix, 0.0, *outputs) != 0
    assert (
        to_yuv(base, 12, 4, 4, matrix, 0.0, base + 16384, base + 16384 + 32, outputs[2])
        != 0
    )
    assert to_yuv(base, 12, 4, 4, matrix, 0.0, base + 16, *outputs[1:]) != 0
    assert (
        to_yuv(
            base, 12, 4, 4, matrix, 0.0, *outputs[:2], ct.c_void_p(base + 16384 + 130)
        )
        != 0
    )
