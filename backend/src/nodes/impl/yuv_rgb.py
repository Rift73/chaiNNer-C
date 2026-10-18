"""YUV planes (colour_info.py's convention) to chaiNNer's float32 BGR images and back,
as zimg (VapourSynth's resize) converts them: chroma is resampled between its stored
size and the luma's with the offsets its siting implies, and the matrix is applied in
float. Nothing is clipped.

A chroma sample covers 2^subsampling luma pixels per axis, also beside an odd luma size
(whose last chroma sample half overhangs it); zimg accepts only sizes it divides.
Matrix.RGB planes are G, B and R as they are (FFmpeg's gbr order), not 0.5-centred.

The matrix step and the resampling run in C (native_planes.py), the chroma's 0.5 centre
inside them; native/tests/reference_yuv_planes keeps the NumPy form they match bit for
bit."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..utils.utils import get_h_w_c
from .colour_info import (
    MATRIX_LABELS,
    ChromaLocation,
    ColourInfo,
    Matrix,
    Primaries,
    Range,
    Transfer,
    chroma_size_for,
    subsampling_from_sizes,
)
from .native_planes import bgr_to_yuv, yuv_to_bgr
from .plane_resample import Kernel, resample_plane_biased

# Kr and Kb of each luma-weighted matrix (zimg's values).
_KR_KB = {
    Matrix.BT709: (0.2126, 0.0722),
    Matrix.BT470BG: (0.299, 0.114),
    Matrix.SMPTE170M: (0.299, 0.114),
    Matrix.BT2020_NCL: (0.2627, 0.0593),
    Matrix.SMPTE240M: (0.212, 0.087),
    Matrix.FCC: (0.3, 0.11),
}

_LEFT = (ChromaLocation.LEFT, ChromaLocation.TOP_LEFT, ChromaLocation.BOTTOM_LEFT)
_TOP = (ChromaLocation.TOP_LEFT, ChromaLocation.TOP)
_BOTTOM = (ChromaLocation.BOTTOM_LEFT, ChromaLocation.BOTTOM)


def _rgb_to_yuv_matrix(matrix: Matrix) -> np.ndarray:
    """Rows Y, U, V (U and V centred on 0) by columns R, G, B, in float64."""
    if matrix is Matrix.YCGCO:
        return np.array([[0.25, 0.5, 0.25], [-0.25, 0.5, -0.25], [0.5, 0, -0.5]])
    if matrix not in _KR_KB:
        raise ValueError(
            f"Planes with the {MATRIX_LABELS[matrix]} matrix cannot be converted to or"
            " from RGB here: it is not a plain matrix. If the video's tag is wrong,"
            " choose its real Matrix."
        )
    kr, kb = _KR_KB[matrix]
    kg = 1.0 - kr - kb
    u_scale = 1.0 / (2.0 - 2.0 * kb)
    v_scale = 1.0 / (2.0 - 2.0 * kr)
    return np.array(
        [
            [kr, kg, kb],
            [-kr * u_scale, -kg * u_scale, (1.0 - kb) * u_scale],
            [(1.0 - kr) * v_scale, -kg * v_scale, -kb * v_scale],
        ]
    )


def _siting(
    location: ChromaLocation, subsampling: tuple[int, int]
) -> tuple[float, float, float, float]:
    """(horizontal scale, vertical scale, horizontal offset, vertical offset) of the
    chroma grid against the luma's, in chroma pixels (zimg's chroma_offset_w/_h)."""
    scale_w = 1.0 / (1 << subsampling[0])
    scale_h = 1.0 / (1 << subsampling[1])
    offset_w = -0.5 + 0.5 * scale_w if location in _LEFT else 0.0
    if location in _TOP:
        offset_h = -0.5 + 0.5 * scale_h
    elif location in _BOTTOM:
        offset_h = 0.5 - 0.5 * scale_h
    else:
        offset_h = 0.0
    return scale_w, scale_h, offset_w, offset_h


def guess_colour_info(
    luma_size: tuple[int, int], chroma_size: tuple[int, int]
) -> ColourInfo:
    """The guesses for untagged YUV planes of these sizes (mpv's, as video detection
    makes them): BT.709 from 1280 wide or above 576 high, else BT.601 (BT.470 BG
    primaries at 576 lines, SMPTE 170M at 480 or 486); limited range, left-sited
    chroma, 8-bit."""
    w, h = luma_size
    sd = w < 1280 and h <= 576
    pal = sd and h == 576
    ntsc = sd and h in (480, 486)
    return ColourInfo(
        matrix=Matrix.BT470BG if pal else Matrix.SMPTE170M if sd else Matrix.BT709,
        transfer=Transfer.BT709,
        primaries=(
            Primaries.BT470BG
            if pal
            else Primaries.SMPTE170M
            if ntsc
            else Primaries.BT709
        ),
        range=Range.LIMITED,
        chroma_location=ChromaLocation.LEFT,
        bit_depth=8,
        luma_size=luma_size,
        chroma_size=chroma_size,
    )


def resolve_colour_info(
    info: ColourInfo | None,
    luma_size: tuple[int, int],
    chroma_size: tuple[int, int],
    matrix: Matrix,
    transfer: Transfer,
    primaries: Primaries,
    range_: Range,
    chroma_location: ChromaLocation,
) -> ColourInfo:
    """`info` (else the guesses for planes of `luma_size`) with every non-Auto choice
    applied, re-issued for planes of these sizes."""
    if info is None:
        info = guess_colour_info(luma_size, chroma_size)
    else:
        info = replace(info, luma_size=luma_size, chroma_size=chroma_size)
    return info.with_choices(matrix, transfer, primaries, range_, chroma_location)


def _check_rgb_subsampling(matrix: Matrix, subsampling: tuple[int, int]) -> None:
    if matrix is Matrix.RGB and subsampling != (0, 0):
        raise ValueError("RGB (identity) planes cannot be subsampled: use 4:4:4.")


def planes_to_rgb(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    matrix: Matrix,
    chroma_location: ChromaLocation,
    kernel: Kernel = Kernel.SPLINE36,
) -> np.ndarray:
    """A float32 BGR image at Y's size from the Y, U and V planes, U and V upsampled to
    Y's size with `kernel` from where `chroma_location` sites them."""
    if u.shape != v.shape:
        raise ValueError(
            f"The U and V planes must have one size, not {u.shape[1]}x{u.shape[0]} and"
            f" {v.shape[1]}x{v.shape[0]}."
        )
    h, w = y.shape
    subsampling = subsampling_from_sizes((w, h), (u.shape[1], u.shape[0]))
    _check_rgb_subsampling(matrix, subsampling)
    if matrix is Matrix.RGB:
        return np.dstack((u, y, v)).astype(np.float32, copy=False)
    to_rgb = np.linalg.inv(_rgb_to_yuv_matrix(matrix)).astype(np.float32)
    y, u, v = (plane.astype(np.float32, copy=False) for plane in (y, u, v))
    if subsampling == (0, 0):
        # zimg skips both axes of 4:4:4 chroma: the matrix step centres it.
        return yuv_to_bgr(y, u, v, to_rgb, 0.5)
    scale_w, scale_h, offset_w, offset_h = _siting(chroma_location, subsampling)
    u, v = (
        resample_plane_biased(
            plane, w, h, kernel, -offset_w, -offset_h, w * scale_w, h * scale_h, 0.5, 0
        )
        for plane in (u, v)
    )
    return yuv_to_bgr(y, u, v, to_rgb, 0)


def rgb_to_planes(
    img: np.ndarray,
    matrix: Matrix,
    chroma_location: ChromaLocation,
    subsampling: tuple[int, int],
    kernel: Kernel = Kernel.SPLINE36,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Y, U and V planes of a BGR image: Y at its size, U and V downsampled by
    `subsampling` (log2 width, height) with `kernel` to where `chroma_location` sites
    them."""
    h, w, c = get_h_w_c(img)
    if c != 3:
        raise ValueError(f"RGB to planes needs a 3-channel image, not {c} channels.")
    _check_rgb_subsampling(matrix, subsampling)
    if matrix is Matrix.RGB:
        b, g, r = (img[..., i].astype(np.float32) for i in range(3))
        return g, b, r
    to_yuv = _rgb_to_yuv_matrix(matrix).astype(np.float32)
    if subsampling == (0, 0):
        # zimg skips both axes of 4:4:4 chroma: the matrix step centres it.
        return bgr_to_yuv(img, to_yuv, 0.5)
    y, u, v = bgr_to_yuv(img, to_yuv, 0)
    chroma_w, chroma_h = chroma_size_for((w, h), subsampling)
    scale_w, scale_h, offset_w, offset_h = _siting(chroma_location, subsampling)
    u, v = (
        resample_plane_biased(
            plane,
            chroma_w,
            chroma_h,
            kernel,
            offset_w / scale_w,
            offset_h / scale_h,
            w * (chroma_w / (w * scale_w)),
            h * (chroma_h / (h * scale_h)),
            0,
            0.5,
        )
        for plane in (u, v)
    )
    return y, u, v
