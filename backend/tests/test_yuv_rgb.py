"""YUV planes <-> RGB (nodes/impl/yuv_rgb.py), checked against zimg (VapourSynth's
resize) for every matrix, chroma siting, subsampling and range zimg supports. The zimg
cases need vapoursynth (run this file in the VapourSynth dev venv); elsewhere they are
skipped."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nodes.impl.colour_info import (
    ChromaLocation,
    Matrix,
    chroma_size_for,
    code_to_float,
    float_to_code,
)
from nodes.impl.plane_resample import Kernel
from nodes.impl.yuv_rgb import planes_to_rgb, rgb_to_planes

# zimg's matrix codes (ITU-T H.273) and chroma locations.
MATRIX_CODES = {
    Matrix.BT709: 1,
    Matrix.FCC: 4,
    Matrix.BT470BG: 5,
    Matrix.SMPTE170M: 6,
    Matrix.SMPTE240M: 7,
    Matrix.YCGCO: 8,
    Matrix.BT2020_NCL: 9,
}
CHROMALOC_CODES = {
    ChromaLocation.LEFT: 0,
    ChromaLocation.CENTER: 1,
    ChromaLocation.TOP_LEFT: 2,
    ChromaLocation.TOP: 3,
    ChromaLocation.BOTTOM_LEFT: 4,
    ChromaLocation.BOTTOM: 5,
}
SUBSAMPLINGS = {"444": (0, 0), "422": (1, 0), "420": (1, 1), "411": (2, 0)}
SIZE = (40, 26)


def random_planes(
    size: tuple[int, int], subsampling: tuple[int, int], seed: int = 0
) -> list[np.ndarray]:
    """Y, U, V planes with values outside 0-1: nothing may clip."""
    rng = np.random.default_rng(seed)
    w, h = size
    cw, ch = chroma_size_for(size, subsampling)
    return [
        (rng.standard_normal(shape) * 0.4 + 0.5).astype(np.float32)
        for shape in ((h, w), (ch, cw), (ch, cw))
    ]


def random_rgb(size: tuple[int, int], seed: int = 1) -> np.ndarray:
    w, h = size
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((h, w, 3)) * 0.4 + 0.5).astype(np.float32)


def clip_of(planes: list[np.ndarray], family: str, sample: str, bits: int) -> Any:
    vs = pytest.importorskip("vapoursynth")
    h, w = planes[0].shape
    ssw = (w // planes[1].shape[1]).bit_length() - 1
    ssh = (h // planes[1].shape[0]).bit_length() - 1
    fmt = vs.core.query_video_format(
        getattr(vs, family), getattr(vs, sample), bits, ssw, ssh
    )
    blank = vs.core.std.BlankClip(format=fmt.id, width=w, height=h, length=1)

    def fill(n: int, f: Any) -> Any:
        frame = f.copy()
        for i, plane in enumerate(planes):
            np.asarray(frame[i])[:] = plane
        return frame

    return blank.std.ModifyFrame(blank, fill)


def resize(clip: Any, kernel: Kernel, **args: Any) -> list[np.ndarray]:
    vs = pytest.importorskip("vapoursynth")
    if kernel is Kernel.BICUBIC:
        args.update(filter_param_a_uv=0, filter_param_b_uv=0.5)
    elif kernel is Kernel.LANCZOS:
        args.update(filter_param_a_uv=3)
    out = getattr(vs.core.resize, kernel.name.title())(
        clip, resample_filter_uv=kernel.value, **args
    ).get_frame(0)
    return [np.asarray(out[i]).copy() for i in range(out.format.num_planes)]


def zimg_to_rgb(clip: Any, kernel: Kernel, **args: Any) -> np.ndarray:
    vs = pytest.importorskip("vapoursynth")
    r, g, b = resize(clip, kernel, format=vs.RGBS, **args)
    return np.dstack((b, g, r))


def zimg_to_yuv(
    img: np.ndarray,
    kernel: Kernel,
    subsampling: tuple[int, int],
    sample: str,
    bits: int,
    **args: Any,
) -> list[np.ndarray]:
    vs = pytest.importorskip("vapoursynth")
    rgb = clip_of([img[..., 2], img[..., 1], img[..., 0]], "RGB", "FLOAT", 32)
    fmt = vs.core.query_video_format(vs.YUV, getattr(vs, sample), bits, *subsampling)
    return resize(rgb, kernel, format=fmt.id, dither_type="none", **args)


def max_diff(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    return float(np.abs(a.astype(np.float64) - b).max())


@pytest.mark.parametrize("subsampling", list(SUBSAMPLINGS))
@pytest.mark.parametrize("location", list(CHROMALOC_CODES))
@pytest.mark.parametrize("matrix", list(MATRIX_CODES))
def test_planes_to_rgb_matches_zimg(
    matrix: Matrix, location: ChromaLocation, subsampling: str
):
    y, u, v = random_planes(SIZE, SUBSAMPLINGS[subsampling])
    # zimg's float YUV centres U and V on 0.
    clip = clip_of([y, u - 0.5, v - 0.5], "YUV", "FLOAT", 32)
    expected = zimg_to_rgb(
        clip,
        Kernel.SPLINE36,
        matrix_in=MATRIX_CODES[matrix],
        chromaloc_in=CHROMALOC_CODES[location],
    )
    actual = planes_to_rgb(y, u, v, matrix, location)
    assert actual.dtype == np.float32
    assert max_diff(actual, expected) <= 1e-5


@pytest.mark.parametrize("subsampling", list(SUBSAMPLINGS))
@pytest.mark.parametrize("location", list(CHROMALOC_CODES))
@pytest.mark.parametrize("matrix", list(MATRIX_CODES))
def test_rgb_to_planes_matches_zimg(
    matrix: Matrix, location: ChromaLocation, subsampling: str
):
    img = random_rgb(SIZE)
    expected = zimg_to_yuv(
        img,
        Kernel.SPLINE36,
        SUBSAMPLINGS[subsampling],
        "FLOAT",
        32,
        matrix=MATRIX_CODES[matrix],
        chromaloc=CHROMALOC_CODES[location],
    )
    actual = rgb_to_planes(img, matrix, location, SUBSAMPLINGS[subsampling])
    for plane, reference, centre in zip(actual, expected, (0, 0.5, 0.5)):
        assert plane.dtype == np.float32
        assert max_diff(plane, reference + centre) <= 1e-5


@pytest.mark.parametrize("kernel", list(Kernel))
def test_every_kernel_matches_zimg(kernel: Kernel):
    location = ChromaLocation.TOP_LEFT
    y, u, v = random_planes(SIZE, (1, 1))
    clip = clip_of([y, u - 0.5, v - 0.5], "YUV", "FLOAT", 32)
    expected = zimg_to_rgb(clip, kernel, matrix_in=1, chromaloc_in=2)
    assert (
        max_diff(planes_to_rgb(y, u, v, Matrix.BT709, location, kernel), expected)
        <= 1e-5
    )

    img = random_rgb(SIZE)
    expected_planes = zimg_to_yuv(
        img, kernel, (1, 1), "FLOAT", 32, matrix=1, chromaloc=2
    )
    actual = rgb_to_planes(img, Matrix.BT709, location, (1, 1), kernel)
    for plane, reference, centre in zip(actual, expected_planes, (0, 0.5, 0.5)):
        assert max_diff(plane, reference + centre) <= 1e-5


@pytest.mark.parametrize("bits", [8, 10, 16])
@pytest.mark.parametrize("full_range", [False, True])
@pytest.mark.parametrize("subsampling", ["444", "422", "420"])
def test_codes_to_rgb_match_zimg(bits: int, full_range: bool, subsampling: str):
    # zimg resizes 9-16 bit chroma as integers, which rounds: subsampled cases are 8-bit.
    if bits > 8 and subsampling != "444":
        pytest.skip("zimg resizes subsampled 9-16 bit chroma in integers")
    rng = np.random.default_rng(bits)
    w, h = SIZE
    cw, ch = chroma_size_for(SIZE, SUBSAMPLINGS[subsampling])
    dtype = np.uint8 if bits == 8 else np.uint16
    codes = [
        rng.integers(0, 1 << bits, shape).astype(dtype)
        for shape in ((h, w), (ch, cw), (ch, cw))
    ]
    clip = clip_of(codes, "YUV", "INTEGER", bits)
    expected = zimg_to_rgb(
        clip,
        Kernel.SPLINE36,
        matrix_in=1,
        chromaloc_in=0,
        range_in_s="full" if full_range else "limited",
    )
    y, u, v = (
        code_to_float(c, bits, full_range, chroma=i > 0) for i, c in enumerate(codes)
    )
    actual = planes_to_rgb(y, u, v, Matrix.BT709, ChromaLocation.LEFT)
    assert max_diff(actual, expected) <= 1e-5


@pytest.mark.parametrize("full_range", [False, True])
def test_rgb_to_codes_match_zimg(full_range: bool):
    img = random_rgb(SIZE)
    expected = zimg_to_yuv(
        img,
        Kernel.SPLINE36,
        (1, 1),
        "INTEGER",
        16,
        matrix=9,
        chromaloc=0,
        range_s="full" if full_range else "limited",
    )
    planes = rgb_to_planes(img, Matrix.BT2020_NCL, ChromaLocation.LEFT, (1, 1))
    for i, (plane, reference) in enumerate(zip(planes, expected)):
        codes = float_to_code(plane, 16, full_range, chroma=i > 0)
        assert np.abs(codes.astype(np.int64) - reference).max() <= 1


def test_odd_sizes_are_the_even_frame_cropped():
    # A 37x25 frame's 19x13 chroma covers luma 0-37 x 0-25 like a 38x26 frame's.
    y, u, v = random_planes((38, 26), (1, 1))
    full = planes_to_rgb(y, u, v, Matrix.BT709, ChromaLocation.TOP_LEFT)
    odd = planes_to_rgb(y[:25, :37], u, v, Matrix.BT709, ChromaLocation.TOP_LEFT)
    assert np.array_equal(odd, full[:25, :37])


@pytest.mark.parametrize("subsampling", list(SUBSAMPLINGS.values()))
@pytest.mark.parametrize("size", [(40, 26), (37, 25)])
def test_flat_colours_survive_a_round_trip(
    size: tuple[int, int], subsampling: tuple[int, int]
):
    img = np.empty((size[1], size[0], 3), np.float32)
    img[...] = (0.2, -0.1, 1.3)
    planes = rgb_to_planes(img, Matrix.BT709, ChromaLocation.LEFT, subsampling)
    assert planes[1].shape == chroma_size_for(size, subsampling)[::-1]
    back = planes_to_rgb(*planes, Matrix.BT709, ChromaLocation.LEFT)
    assert np.allclose(back, img, rtol=0, atol=1e-5)


def test_rgb_identity_is_gbr():
    img = random_rgb(SIZE)
    g, b, r = rgb_to_planes(img, Matrix.RGB, ChromaLocation.LEFT, (0, 0))
    assert np.array_equal(g, img[..., 1]) and np.array_equal(b, img[..., 0])
    assert np.array_equal(r, img[..., 2])
    back = planes_to_rgb(g, b, r, Matrix.RGB, ChromaLocation.LEFT)
    assert np.array_equal(back, img)
    with pytest.raises(ValueError, match="cannot be subsampled"):
        rgb_to_planes(img, Matrix.RGB, ChromaLocation.LEFT, (1, 1))


@pytest.mark.parametrize("matrix", [Matrix.BT2020_CL, Matrix.ICTCP])
def test_non_matrix_coding_is_a_clear_error(matrix: Matrix):
    y, u, v = random_planes(SIZE, (1, 1))
    with pytest.raises(ValueError, match="not a plain matrix"):
        planes_to_rgb(y, u, v, matrix, ChromaLocation.LEFT)
    with pytest.raises(ValueError, match="not a plain matrix"):
        rgb_to_planes(random_rgb(SIZE), matrix, ChromaLocation.LEFT, (1, 1))


def test_mismatched_planes_are_a_clear_error():
    y, u, v = random_planes(SIZE, (1, 1))
    with pytest.raises(ValueError, match="one size"):
        planes_to_rgb(y, u, v[:-1], Matrix.BT709, ChromaLocation.LEFT)
    with pytest.raises(ValueError, match="does not fit"):
        planes_to_rgb(y, u[:-2], v[:-2], Matrix.BT709, ChromaLocation.LEFT)
