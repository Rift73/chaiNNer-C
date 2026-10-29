"""Load Image and Load Images load CMYK JPEGs as colour, as Pillow converts them to RGB.
Pillow decodes them as four CMYK samples, which Load Image took for RGBA: the colours
came out inverted and K became the alpha, so mostly-white images turned transparent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

# The node's modules load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from packages.chaiNNer_standard.image.batch_processing.load_images import (
    load_images_node,
)
from packages.chaiNNer_standard.image.io.load_image import _read_pil, load_image_node

RGB8 = np.random.default_rng(4).integers(0, 256, (5, 7, 3), dtype=np.uint8)


def write(path: Path, orientation: int | None = None) -> None:
    options: dict[str, object] = {"quality": 95}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        options["exif"] = exif.tobytes()
    Image.fromarray(RGB8).convert("CMYK").save(path, "JPEG", **options)


def expected(path: Path) -> np.ndarray:
    """Pillow's RGB of the file, turned upright, in OpenCV's channel order."""
    with Image.open(path) as im:
        assert im.mode == "CMYK"
        pixels = np.array(ImageOps.exif_transpose(im).convert("RGB"))
    return np.ascontiguousarray(pixels[:, :, ::-1])


@pytest.mark.parametrize("orientation", [None, 6])
def test_cmyk_jpeg_loads_as_bgr(tmp_path: Path, orientation: int | None):
    path = tmp_path / "cmyk.jpg"
    write(path, orientation)
    want = expected(path)
    for image in (load_image_node(path)[0], _read_pil(path)):
        assert image is not None
        assert image.dtype == np.uint8 and image.shape == want.shape
        assert image.tobytes() == want.tobytes()
    # The colours are the source's, within JPEG's error.
    source = RGB8[:, :, ::-1] if orientation is None else np.rot90(RGB8[:, :, ::-1], -1)
    assert np.abs(want.astype(int) - source.astype(int)).max() <= 8


def test_load_images_cmyk_jpeg(tmp_path: Path):
    write(tmp_path / "cmyk.jpg")
    generator, _ = load_images_node(tmp_path, False, False, "", False, 0, True)
    (item,) = list(generator.supplier())
    assert not isinstance(item, Exception)
    assert item[0].tobytes() == expected(tmp_path / "cmyk.jpg").tobytes()
