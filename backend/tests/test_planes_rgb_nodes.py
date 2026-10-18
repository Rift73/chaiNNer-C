"""Planes to RGB and RGB to Planes: Colour Info, overrides and guesses, plane sizes, and
float values outside 0-1 kept through the nodes' outputs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from nodes.impl.colour_info import (
    ChromaLocation,
    ColourInfo,
    HdrInfo,
    Matrix,
    Primaries,
    Range,
    Transfer,
)
from nodes.impl.plane_resample import Kernel
from nodes.impl.yuv_rgb import guess_colour_info, planes_to_rgb
from packages.chaiNNer_standard.image_channel import all_group
from packages.chaiNNer_standard.image_channel.all.planes_to_rgb import (
    planes_to_rgb_node,
)
from packages.chaiNNer_standard.image_channel.all.rgb_to_planes import (
    ChromaSubsampling,
    rgb_to_planes_node,
)

AUTO = (Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO, ChromaLocation.AUTO)

# A 10-bit HDR10 source, 4:2:0, chroma top-left, every field tagged.
SOURCE = ColourInfo(
    matrix=Matrix.BT2020_NCL,
    transfer=Transfer.PQ,
    primaries=Primaries.BT2020,
    range=Range.LIMITED,
    chroma_location=ChromaLocation.TOP_LEFT,
    bit_depth=10,
    luma_size=(64, 36),
    chroma_size=(32, 18),
    hdr=HdrInfo(max_cll=1000, max_fall=400),
    origin=tuple(
        (f, "file")
        for f in ("chroma_location", "matrix", "primaries", "range", "transfer")
    ),
)


def planes(w: int, h: int, cw: int, ch: int) -> list[np.ndarray]:
    rng = np.random.default_rng(w + h)
    return [
        (rng.standard_normal(shape) * 0.4 + 0.5).astype(np.float32)
        for shape in ((h, w), (ch, cw), (ch, cw))
    ]


def flat_rgb(w: int, h: int) -> np.ndarray:
    img = np.empty((h, w, 3), np.float32)
    img[...] = (1.25, -0.125, 0.5)  # BGR, outside 0-1
    return img


@pytest.mark.parametrize(
    ("size", "matrix", "primaries"),
    [
        ((1920, 1080), Matrix.BT709, Primaries.BT709),
        ((1280, 536), Matrix.BT709, Primaries.BT709),
        ((960, 720), Matrix.BT709, Primaries.BT709),
        ((720, 576), Matrix.BT470BG, Primaries.BT470BG),
        ((720, 480), Matrix.SMPTE170M, Primaries.SMPTE170M),
        ((720, 486), Matrix.SMPTE170M, Primaries.SMPTE170M),
        ((640, 360), Matrix.SMPTE170M, Primaries.BT709),
    ],
)
def test_guesses_follow_video_detection(
    size: tuple[int, int], matrix: Matrix, primaries: Primaries
):
    info = guess_colour_info(size, size)
    assert (info.matrix, info.primaries) == (matrix, primaries)
    assert info.transfer is Transfer.BT709 and info.range is Range.LIMITED
    assert info.chroma_location is ChromaLocation.LEFT and info.bit_depth == 8
    assert all(info.origin_of(f) == "guessed" for f in ("matrix", "range"))


def test_planes_to_rgb_without_colour_info_guesses():
    y, u, v = planes(1280, 720, 640, 360)
    img, info, text = planes_to_rgb_node(y, u, v, None, *AUTO, Kernel.SPLINE36)
    assert np.array_equal(
        img, planes_to_rgb(y, u, v, Matrix.BT709, ChromaLocation.LEFT)
    )
    assert info.luma_size == info.chroma_size == (1280, 720)
    assert info.matrix is Matrix.BT709 and "matrix guessed" in text


def test_planes_to_rgb_uses_colour_info_and_overrides():
    y, u, v = planes(64, 36, 32, 18)
    img, info, text = planes_to_rgb_node(
        y,
        u,
        v,
        SOURCE,
        Matrix.BT709,
        Transfer.AUTO,
        Primaries.AUTO,
        Range.FULL,
        ChromaLocation.AUTO,
        Kernel.BICUBIC,
    )
    expected = planes_to_rgb(
        y, u, v, Matrix.BT709, ChromaLocation.TOP_LEFT, Kernel.BICUBIC
    )
    assert np.array_equal(img, expected)
    # Re-issued at the RGB size: Save Video's siting correction sees full-size chroma.
    assert info.luma_size == info.chroma_size == (64, 36)
    assert info.subsampling == (0, 0)
    assert (info.matrix, info.range) == (Matrix.BT709, Range.FULL)
    assert info.origin_of("matrix") == info.origin_of("range") == "set"
    assert info.origin_of("transfer") == "file" and info.transfer is Transfer.PQ
    assert info.bit_depth == 10 and info.hdr == SOURCE.hdr
    assert info.chroma_location is ChromaLocation.TOP_LEFT
    assert text == info.summary()


@pytest.mark.parametrize(
    ("subsampling", "chroma"),
    [
        (ChromaSubsampling.YUV444, (37, 25)),
        (ChromaSubsampling.YUV422, (19, 25)),
        (ChromaSubsampling.YUV420, (19, 13)),
    ],
)
def test_rgb_to_planes_sizes_and_round_trip(
    subsampling: ChromaSubsampling, chroma: tuple[int, int]
):
    img = flat_rgb(37, 25)
    y, u, v, info, _ = rgb_to_planes_node(
        img, SOURCE, *AUTO, subsampling, Kernel.SPLINE36
    )
    assert y.shape == (25, 37) and u.shape == v.shape == chroma[::-1]
    assert info.luma_size == (37, 25) and info.chroma_size == chroma
    assert info.matrix is Matrix.BT2020_NCL and info.origin_of("matrix") == "file"
    back, back_info, _ = planes_to_rgb_node(y, u, v, info, *AUTO, Kernel.SPLINE36)
    assert np.allclose(back, img, rtol=0, atol=1e-5)
    assert back_info == replace(info, chroma_size=(37, 25))


def test_rgb_identity_planes_cannot_be_subsampled():
    with pytest.raises(ValueError, match="cannot be subsampled"):
        rgb_to_planes_node(
            flat_rgb(8, 8),
            None,
            Matrix.RGB,
            *AUTO[1:],
            ChromaSubsampling.YUV420,
            Kernel.SPLINE36,
        )


def test_outputs_keep_values_outside_0_1():
    nodes = {node.schema_id: node for node in all_group.nodes}
    to_planes = nodes["chainner:image:rgb_to_planes"]
    to_rgb = nodes["chainner:image:planes_to_rgb"]
    y, u, v, info, _ = rgb_to_planes_node(
        flat_rgb(16, 8), None, *AUTO, ChromaSubsampling.YUV420, Kernel.SPLINE36
    )
    y, u, v = (o.enforce(p) for o, p in zip(to_planes.outputs, (y, u, v)))
    img, _, _ = planes_to_rgb_node(y, u, v, info, *AUTO, Kernel.SPLINE36)
    enforced = to_rgb.outputs[0].enforce(img)
    assert enforced.min() < -0.1 and enforced.max() > 1.2
