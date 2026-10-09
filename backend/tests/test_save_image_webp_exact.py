"""Save Image's lossless WebP keeps every sample of an RGBA image, the colour under
alpha 0 included, through Pillow's exact=True (OpenCV's libwebp call drops that
colour). RGB lossless and every lossy WebP are still OpenCV's bytes."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

# The node's modules load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import Lazy
from nodes.impl.image_utils import to_uint8
from packages.chaiNNer_standard.image.io import save_image
from packages.chaiNNer_standard.image.io.load_image import load_image_node
from packages.chaiNNer_standard.image.io.save_image import (
    AvifSubsampling,
    BC7Compression,
    DDSErrorMetric,
    ImageFormat,
    JpegSubsampling,
    PngColorDepth,
    TiffColorDepth,
    TiffCompression,
)

RNG = np.random.default_rng(2914)
IMAGE = RNG.random((9, 13, 4), dtype=np.float32)
IMAGE[:3, :, 3] = 0  # transparent rows with colour


def inputs(
    image: np.ndarray, directory: Path, lossless: bool, quality: int
) -> list[object]:
    """Save Image's inputs, input 0 the image (the prepare phase's form)."""
    return [
        image,
        directory,
        None,
        "image",
        ImageFormat.WEBP,
        PngColorDepth.U8,
        lossless,
        quality,
        JpegSubsampling.FACTOR_422,
        False,
        TiffColorDepth.U8,
        TiffCompression.LZW,
        "BC1_UNORM",
        BC7Compression.DEFAULT,
        DDSErrorMetric.PERCEPTUAL,
        False,
        0,
        False,
        AvifSubsampling.FACTOR_420,
        False,
    ]


def save(directory: Path, image: np.ndarray, lossless: bool, quality: int) -> bytes:
    save_image.save_image_node(
        Lazy.ready(image),
        directory,
        None,
        "image",
        ImageFormat.WEBP,
        PngColorDepth.U8,
        lossless,
        quality,
        JpegSubsampling.FACTOR_422,
        False,
        TiffColorDepth.U8,
        TiffCompression.LZW,
        "BC1_UNORM",
        BC7Compression.DEFAULT,
        DDSErrorMetric.PERCEPTUAL,
        False,
        0,
        False,
        AvifSubsampling.FACTOR_420,
        False,
    )
    return (directory / "image.webp").read_bytes()


def test_rgba_lossless_keeps_every_sample(tmp_path: Path):
    saved = save(tmp_path, IMAGE, True, 95)
    written = to_uint8(IMAGE, normalized=True)
    with Image.open(tmp_path / "image.webp") as im:
        assert im.mode == "RGBA"
        assert np.array(im).tobytes() == written[:, :, [2, 1, 0, 3]].tobytes()
    loaded = load_image_node(tmp_path / "image.webp")[0]
    assert loaded.dtype == np.uint8 and loaded.tobytes() == written.tobytes()
    values = inputs(IMAGE, tmp_path, True, 95)
    assert save_image.prepare_save_image(values) == saved


@pytest.mark.parametrize(
    "channels,lossless,quality",
    [(3, True, 95), (4, False, 95), (3, False, 95), (4, False, 30)],
)
def test_other_webp_unchanged(
    tmp_path: Path, channels: int, lossless: bool, quality: int
):
    image = IMAGE[:, :, :channels]
    params = [cv2.IMWRITE_WEBP_QUALITY, 101 if lossless else quality]
    today = cv2.imencode(".webp", to_uint8(image, normalized=True), params)[1]
    assert save(tmp_path, image, lossless, quality) == today.tobytes()
