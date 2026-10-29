"""Save Image's AVIF Chroma Subsampling 4:0:0 keeps the luma alone, so a colour image
is saved as grayscale; its label said "Best Compression", which hid that."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# The node's modules load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import Lazy
from packages.chaiNNer_standard.image.io import save_image
from packages.chaiNNer_standard.image.io.save_image import (
    AvifSubsampling,
    BC7Compression,
    DDSColourConversion,
    DDSErrorMetric,
    DDSMipFilter,
    ImageFormat,
    JpegSubsampling,
    PngColorDepth,
    TiffColorDepth,
    TiffCompression,
)


def test_4_0_0_is_labelled_grayscale():
    node = next(
        n for n in save_image.io_group.nodes if n.schema_id == "chainner:image:save"
    )
    # AVIF's Chroma Subsampling
    schema = next(i.to_dict() for i in node.inputs if i.id == 17)
    labels = {option["value"]: option["option"] for option in schema["options"]}
    assert labels["4:0:0"] == "4:0:0 (Grayscale)"


def test_4_0_0_saves_grayscale(tmp_path: Path):
    image = np.zeros((16, 16, 3), np.float32)
    image[..., 2] = 1  # red (BGR)
    save_image.save_image_node(
        Lazy.ready(image),
        tmp_path,
        None,
        "red",
        ImageFormat.AVIF,
        PngColorDepth.U8,
        False,
        95,
        JpegSubsampling.FACTOR_422,
        False,
        TiffColorDepth.U8,
        TiffCompression.LZW,
        "BC1_UNORM",
        BC7Compression.DEFAULT,
        DDSErrorMetric.PERCEPTUAL,
        False,
        0,
        0,
        DDSMipFilter.DEFAULT,
        False,
        DDSColourConversion.KEEP,
        False,
        True,
        AvifSubsampling.FACTOR_400,
        False,
    )
    with Image.open(tmp_path / "red.avif") as saved:
        assert saved.mode == "L"
