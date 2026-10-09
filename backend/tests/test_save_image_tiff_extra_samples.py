"""Save Image's RGBA TIFFs declare their alpha with ExtraSamples = 2 (unassociated),
which TIFF 6.0 requires and OpenCV's encoder leaves out (upstream chaiNNer #2950).
The tag goes into a copy of the first IFD at the end of the file; the pixels,
compression and predictor are OpenCV's, and RGB and grey TIFFs are unchanged."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, TiffImagePlugin

from api import Lazy
from nodes.impl.image_utils import to_uint8, to_uint16
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

RNG = np.random.default_rng(2950)
IMAGE = RNG.random((9, 13, 4), dtype=np.float32)
IMAGE[0, :3, 3] = 0  # colour under alpha 0 survives too


def inputs(
    image: np.ndarray,
    directory: Path,
    depth: TiffColorDepth,
    compression: TiffCompression,
) -> list[object]:
    """Save Image's inputs, as the prepare phase takes them (input 0 the image)."""
    return [
        image,
        directory,
        None,
        "image",
        ImageFormat.TIFF,
        PngColorDepth.U8,
        False,
        95,
        JpegSubsampling.FACTOR_422,
        False,
        depth,
        compression,
        "BC1_UNORM",
        BC7Compression.DEFAULT,
        DDSErrorMetric.PERCEPTUAL,
        False,
        0,
        False,
        AvifSubsampling.FACTOR_420,
        False,
    ]


def save(
    directory: Path,
    image: np.ndarray,
    depth: TiffColorDepth,
    compression: TiffCompression,
) -> bytes:
    save_image.save_image_node(
        Lazy.ready(image),
        directory,
        None,
        "image",
        ImageFormat.TIFF,
        PngColorDepth.U8,
        False,
        95,
        JpegSubsampling.FACTOR_422,
        False,
        depth,
        compression,
        "BC1_UNORM",
        BC7Compression.DEFAULT,
        DDSErrorMetric.PERCEPTUAL,
        False,
        0,
        False,
        AvifSubsampling.FACTOR_420,
        False,
    )
    return (directory / "image.tiff").read_bytes()


def opencv_encode(
    image: np.ndarray, depth: TiffColorDepth, compression: TiffCompression
) -> bytes:
    """Today's encoder call: Save Image's conversion, then cv2.imencode."""
    if depth == TiffColorDepth.F32:
        return cv2.imencode(".tiff", image)[1].tobytes()
    pixels = (to_uint8 if depth == TiffColorDepth.U8 else to_uint16)(
        image, normalized=True
    )
    params = [cv2.IMWRITE_TIFF_COMPRESSION, compression.value]
    return cv2.imencode(".tiff", pixels, params)[1].tobytes()


def ifd(data: bytes) -> list[bytes]:
    (offset,) = struct.unpack("<I", data[4:8])
    (count,) = struct.unpack("<H", data[offset : offset + 2])
    return [data[offset + 2 + 12 * i : offset + 14 + 12 * i] for i in range(count + 1)]


def tags(data: bytes) -> dict[int, object]:
    """The first IFD as Pillow's TIFF reader parses it (Pillow opens no 32-bit RGBA)."""
    directory = TiffImagePlugin.ImageFileDirectory_v2(data[:8])
    stream = io.BytesIO(data)
    stream.seek(struct.unpack("<I", data[4:8])[0])
    directory.load(stream)
    return dict(directory)


CASES = [
    (depth, compression)
    for depth in TiffColorDepth
    for compression in TiffCompression
    # F32 ignores the compression option.
    if depth != TiffColorDepth.F32 or compression == TiffCompression.LZW
]


@pytest.mark.parametrize("depth,compression", CASES)
def test_rgba_tiff_declares_unassociated_alpha(
    tmp_path: Path, depth: TiffColorDepth, compression: TiffCompression
):
    saved = save(tmp_path, IMAGE, depth, compression)
    today = opencv_encode(IMAGE, depth, compression)
    assert saved[:2] == b"II" and today[:2] == b"II"
    # Every byte of today's file stays put; the new IFD follows on a word boundary.
    assert saved[:4] == today[:4] and saved[8 : len(today)] == today[8:]
    (offset,) = struct.unpack("<I", saved[4:8])
    assert offset == len(today) + len(today) % 2
    old_entries, new_entries = ifd(today), ifd(saved)
    extra = struct.pack("<HHIHH", 338, 3, 1, 2, 0)
    assert sorted([*old_entries[:-1], extra]) == new_entries[:-1]
    assert old_entries[-1][:4] == new_entries[-1][:4]  # the next-IFD offset
    assert len(saved) == offset + 2 + 12 * len(new_entries[:-1]) + 4
    # Pillow reads every tag as before, plus ExtraSamples.
    assert tags(saved) == {**tags(today), 338: (2,)}
    # Load Image gives back exactly what was written.
    loaded = load_image_node(tmp_path / "image.tiff")[0]
    pixels = cv2.imdecode(np.frombuffer(today, np.uint8), cv2.IMREAD_UNCHANGED)
    assert pixels is not None
    assert loaded.dtype == pixels.dtype and loaded.tobytes() == pixels.tobytes()
    if depth == TiffColorDepth.U8:
        with Image.open(tmp_path / "image.tiff") as im:
            assert im.mode == "RGBA"
            assert np.array(im).tobytes() == pixels[:, :, [2, 1, 0, 3]].tobytes()


@pytest.mark.parametrize("depth,compression", CASES)
@pytest.mark.parametrize("channels", [1, 3])
def test_rgb_and_grey_tiffs_unchanged(
    tmp_path: Path,
    depth: TiffColorDepth,
    compression: TiffCompression,
    channels: int,
):
    image = IMAGE[:, :, :channels] if channels == 3 else IMAGE[:, :, 0]
    assert save(tmp_path, image, depth, compression) == opencv_encode(
        image, depth, compression
    )


def test_prepared_bytes_match_the_saved_file(tmp_path: Path):
    values = inputs(IMAGE, tmp_path, TiffColorDepth.U16, TiffCompression.ZIP)
    assert save_image.prepare_save_image(values) == save(
        tmp_path, IMAGE, TiffColorDepth.U16, TiffCompression.ZIP
    )
