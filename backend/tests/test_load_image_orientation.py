"""Load Image and Load Images turn JPEG, PNG, WebP and AVIF files upright as their EXIF
Orientation tag asks, as ImageOps.exif_transpose does (upstream chaiNNer #1312). TIFFs
were already oriented by their decoders and are not turned twice; files without the tag
or with Orientation 1 load exactly as decoded."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

from packages.chaiNNer_standard.image.batch_processing.load_images import (
    load_images_node,
)
from packages.chaiNNer_standard.image.io.load_image import (
    _read_cv,
    _read_pil,
    load_image_node,
)

RNG = np.random.default_rng(1312)
# Not square, so a wrong turn cannot pass by symmetry.
RGBA8 = RNG.integers(0, 256, (5, 7, 4), dtype=np.uint8)
ORIENTATIONS = [None, *range(1, 9)]
# extension: (Pillow format, save options, modes); lossless so both decoders agree.
FORMATS = {
    ".jpg": ("JPEG", {"quality": 95}, ("RGB", "L")),
    ".png": ("PNG", {}, ("RGB", "RGBA", "L")),
    ".webp": ("WEBP", {"lossless": True, "exact": True}, ("RGB", "RGBA")),
    ".avif": ("AVIF", {"quality": 100}, ("RGB",)),
}


def write(path: Path, mode: str, orientation: int | None) -> None:
    fmt, options, _ = FORMATS[path.suffix]
    im = Image.fromarray(RGBA8).convert(mode)
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        options = {**options, "exif": exif.tobytes()}
    im.save(path, fmt, **options)


def upright(path: Path) -> np.ndarray:
    """Pillow's decode, turned by ImageOps.exif_transpose, in OpenCV's channel order."""
    with Image.open(path) as im:
        pixels = np.array(ImageOps.exif_transpose(im))
    if pixels.ndim == 3:
        pixels = pixels[:, :, [2, 1, 0, 3][: pixels.shape[2]]]
    return np.ascontiguousarray(pixels)


def decoded(path: Path) -> np.ndarray:
    """OpenCV's decode, as Load Image returned it before."""
    pixels = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    assert pixels is not None
    return pixels


def assert_identical(actual: np.ndarray | None, expected: np.ndarray) -> None:
    assert actual is not None
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.flags.c_contiguous
    assert actual.tobytes() == expected.tobytes()


CASES = [
    (ext, mode, orientation)
    for ext, (_, _, modes) in FORMATS.items()
    for mode in modes
    for orientation in ORIENTATIONS
]


@pytest.mark.parametrize("ext,mode,orientation", CASES)
def test_load_image_turns_upright(
    tmp_path: Path, ext: str, mode: str, orientation: int | None
):
    path = tmp_path / f"image{ext}"
    write(path, mode, orientation)
    expected = upright(path)
    assert_identical(load_image_node(path)[0], expected)
    if orientation in (None, 1) and ext in (".png", ".webp"):
        # Exactly OpenCV's decode, as before.
        assert_identical(load_image_node(path)[0], decoded(path))
    # Both read paths orient, whichever the decoder order picks for the extension.
    for read in (_read_cv, _read_pil):
        result = read(path)
        if result is not None:
            assert_identical(result, expected)


def _chunks(data: bytes) -> list[bytes]:
    chunks, at = [], 8
    while at < len(data):
        (length,) = struct.unpack(">I", data[at : at + 4])
        chunks.append(data[at : at + 12 + length])
        at += 12 + length
    return chunks


def test_png_exif_after_idat(tmp_path: Path):
    before = tmp_path / "before.png"
    write(before, "RGB", 6)
    data = before.read_bytes()
    chunks = _chunks(data)
    exif = next(c for c in chunks if c[4:8] == b"eXIf")
    rest = [c for c in chunks if c is not exif]
    path = tmp_path / "after.png"
    path.write_bytes(data[:8] + b"".join(rest[:-1]) + exif + rest[-1])
    assert [c[4:8] for c in _chunks(path.read_bytes())][-2:] == [b"eXIf", b"IEND"]
    expected = upright(path)
    assert expected.shape[:2] == (7, 5)
    for result in (load_image_node(path)[0], _read_pil(path)):
        assert_identical(result, expected)


@pytest.mark.parametrize(
    "exif",
    [
        b"",
        b"Exif\0\0",
        b"Exif\0\0II*\0\xff\xff\xff\xff",  # IFD0 past the end
        b"II*\0\x08\0\0\0\x05\0",  # five entries announced, none present
        # Orientation 9 (out of range) and Orientation as two SHORTs
        b"II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x09\0\0\0\0\0\0\0",
        b"MM\0*\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x02\0\x06\0\x06\0\0\0\0",
    ],
)
def test_malformed_exif_loads_as_decoded(tmp_path: Path, exif: bytes):
    path = tmp_path / "malformed.png"
    Image.fromarray(RGBA8).save(path, exif=exif)
    assert_identical(load_image_node(path)[0], decoded(path))


def test_big_endian_exif(tmp_path: Path):
    path = tmp_path / "motorola.png"
    tiff = b"MM\0*\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01\0\x08\0\0\0\0\0\0"
    Image.fromarray(RGBA8).save(path)
    data = path.read_bytes()
    # The eXIf chunk right after IHDR (8 + 25 bytes).
    chunk = struct.pack(">I", len(tiff)) + b"eXIf" + tiff
    chunk += struct.pack(">I", zlib.crc32(b"eXIf" + tiff))
    path.write_bytes(data[:33] + chunk + data[33:])
    expected = upright(path)
    assert expected.shape == (7, 5, 4)
    assert_identical(load_image_node(path)[0], expected)


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_tiff_turned_once(tmp_path: Path, orientation: int):
    path = tmp_path / "turned.tif"
    exif = Image.Exif()
    exif[0x0112] = orientation
    Image.fromarray(RGBA8[:, :, :3]).save(path, exif=exif.tobytes())
    assert_identical(load_image_node(path)[0], upright(path))


def test_load_images_turns_upright(tmp_path: Path):
    write(tmp_path / "turned.jpg", "RGB", 6)
    generator, _ = load_images_node(tmp_path, False, False, "", False, 0, True)
    (item,) = list(generator.supplier())
    assert not isinstance(item, Exception)
    assert_identical(item[0], upright(tmp_path / "turned.jpg"))
