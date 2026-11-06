"""Load Image and Save Image on files hand-made here, with the samples they hold as the oracle: a file of a kind OpenCV
reads wrongly or a damaged one must load with its true colours and transparency, or be refused, never load wrong."""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from api import Lazy
from packages.chaiNNer_standard.image.io.load_image import load_image_node
from packages.chaiNNer_standard.image.io.save_image import (
    AvifSubsampling,
    BC7Compression,
    DDSColourConversion,
    DDSErrorMetric,
    DDSMipFilter,
    ExrColorDepth,
    ExrCompression,
    ImageFormat,
    JpegSubsampling,
    PngColorDepth,
    TiffColorDepth,
    TiffCompression,
    save_image_node,
)

H, W = 7, 13
RNG = np.random.default_rng(1106)


def load(path: Path) -> np.ndarray:
    return load_image_node(path)[0]


def written(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data)
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def packed_row(row: np.ndarray, bits: int) -> bytes:
    if bits >= 8:
        return row.astype(f">u{bits // 8}").tobytes()
    per_byte = 8 // bits
    padded = np.concatenate([row, np.zeros(-len(row) % per_byte, row.dtype)])
    weights = 1 << (8 - bits - bits * np.arange(per_byte))
    return (padded.reshape(-1, per_byte) * weights).sum(1).astype(np.uint8).tobytes()


def png(samples: np.ndarray, bits: int, colour: int, *chunks: bytes) -> bytes:
    raw = b"".join(b"\0" + packed_row(row, bits) for row in samples)
    header = struct.pack(">IIBBBBB", W, H, bits, colour, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + b"".join(chunks)
        + png_chunk(b"IDAT", zlib.compress(raw))
        + png_chunk(b"IEND", b"")
    )


TIFF_TYPES = {"H": 3, "I": 4}


def tiff(
    width: int, height: int, tags: dict[int, tuple[str, list[int]]], parts: list[bytes]
) -> bytes:
    """A little-endian TIFF: pixel data first, then the IFD; a value that does not fit its entry follows the IFD."""
    data = bytearray(b"II*\0\0\0\0\0")
    offsets = []
    for part in parts:
        offsets.append(len(data))
        data += part + b"\0" * (len(part) % 2)
    all_tags = dict(tags)
    tiled = 322 in tags
    all_tags[324 if tiled else 273] = ("I", offsets)
    all_tags[325 if tiled else 279] = ("I", [len(part) for part in parts])
    all_tags[256] = ("I", [width])
    all_tags[257] = ("I", [height])
    ifd = len(data)
    data[4:8] = struct.pack("<I", ifd)
    extra_at = ifd + 2 + 12 * len(all_tags) + 4
    entries, extra = b"", b""
    for tag in sorted(all_tags):
        kind, values = all_tags[tag]
        packed = struct.pack(f"<{len(values)}{kind}", *values)
        if len(packed) > 4:
            value = struct.pack("<I", extra_at + len(extra))
            extra += packed
        else:
            value = packed.ljust(4, b"\0")
        entries += struct.pack("<HHI", tag, TIFF_TYPES[kind], len(values)) + value
    return bytes(
        data + struct.pack("<H", len(all_tags)) + entries + b"\0\0\0\0" + extra
    )


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16])
def test_a_grey_png_keeps_its_transparent_value(tmp_path: Path, bits: int):
    top = 2**bits - 1
    samples = RNG.integers(0, top + 1, (H, W), dtype=np.uint32)
    samples[0, :2] = [0, top]
    key = top
    trns = png_chunk(b"tRNS", struct.pack(">H", key))
    image = load(written(tmp_path / "x.png", png(samples, bits, 0, trns)))
    assert image.shape == (H, W, 4)
    dtype_top = 65535 if bits == 16 else 255
    grey = samples * dtype_top // top
    for channel in range(3):
        assert np.array_equal(image[..., channel], grey)
    assert np.array_equal(image[..., 3], (samples != key) * dtype_top)


def test_a_grey_png_without_a_transparent_value_stays_grey(tmp_path: Path):
    samples = RNG.integers(0, 256, (H, W), dtype=np.uint32)
    image = load(written(tmp_path / "x.png", png(samples, 8, 0)))
    assert np.array_equal(image, samples)


def bmp(
    bits: int,
    pixels: bytes,
    compression: int = 0,
    after_header: bytes = b"",
    height: int = H,
    width: int = W,
    os2: bool = False,
) -> bytes:
    info = (
        struct.pack("<IHHHH", 12, width, height, 1, bits)
        if os2
        else struct.pack(
            "<IiiHHIIiiII", 40, width, height, 1, bits, compression, 0, 0, 0, 0, 0
        )
    )
    offset = 14 + len(info) + len(after_header)
    return (
        b"BM"
        + struct.pack("<IHHI", offset + len(pixels), 0, 0, offset)
        + info
        + after_header
        + pixels
    )


def bottom_up(rows: np.ndarray, bytes_per_row: int) -> bytes:
    pad = b"\0" * (-bytes_per_row % 4)
    return b"".join(row.tobytes() + pad for row in rows[::-1])


@pytest.mark.parametrize(
    "masks,order",
    [
        ((0xFF0000, 0xFF00, 0xFF), (0, 1, 2)),
        ((0xFF, 0xFF00, 0xFF0000), (2, 1, 0)),
        ((0xFF000000, 0xFF0000, 0xFF00), (1, 2, 3)),
    ],
    ids=["x8r8g8b8", "x8b8g8r8", "r8g8b8x8"],
)
def test_a_32_bit_bmp_with_masks_after_its_header_has_no_alpha(
    tmp_path: Path, masks: tuple[int, int, int], order: tuple[int, int, int]
):
    stored = RNG.integers(0, 256, (H, W, 4), dtype=np.uint8)
    other = ({0, 1, 2, 3} - set(order)).pop()
    stored[..., other] = 0
    data = bmp(
        32, bottom_up(stored, W * 4), 3, struct.pack("<3I", *masks), height=H
    )
    image = load(written(tmp_path / "x.bmp", data))
    assert image.shape == (H, W, 3)
    assert np.array_equal(image, stored[..., order])


def test_a_24_bit_os2_bmp_keeps_its_colours(tmp_path: Path):
    stored = RNG.integers(0, 256, (H, W, 3), dtype=np.uint8)
    data = bmp(24, bottom_up(stored, W * 3), os2=True)
    assert np.array_equal(load(written(tmp_path / "x.bmp", data)), stored)


@pytest.mark.parametrize("bits", [1, 4, 8])
def test_a_palette_os2_bmp_keeps_its_colours(tmp_path: Path, bits: int):
    colours = RNG.integers(0, 256, (2**bits, 3), dtype=np.uint8)
    indices = RNG.integers(0, 2**bits, (H, W), dtype=np.uint32)
    rows = b"".join(
        packed_row(row, bits).ljust((W * bits + 31) // 32 * 4, b"\0")
        for row in indices[::-1]
    )
    data = bmp(bits, rows, after_header=colours.tobytes(), os2=True)
    assert np.array_equal(load(written(tmp_path / "x.bmp", data)), colours[indices])


def test_a_grey_palette_os2_bmp_stays_grey(tmp_path: Path):
    colours = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, 1)
    indices = RNG.integers(0, 256, (H, W), dtype=np.uint8)
    data = bmp(8, bottom_up(indices, W), after_header=colours.tobytes(), os2=True)
    assert np.array_equal(load(written(tmp_path / "x.bmp", data)), indices)


RLE4_PALETTE = RNG.integers(1, 256, (16, 4), dtype=np.uint8)


def rle4_bmp(stream: bytes) -> bytes:
    return bmp(4, stream, 2, RLE4_PALETTE.tobytes(), height=3, width=7)


def test_an_rle4_end_of_bitmap_ends_the_image(tmp_path: Path):
    image = load(written(tmp_path / "x.bmp", rle4_bmp(bytes([4, 0x12, 0, 0, 0, 1]))))
    colours = RLE4_PALETTE[..., :3]
    expected = np.zeros((3, 7), np.uint8)
    expected[2, :4] = [1, 2, 1, 2]
    assert np.array_equal(image, colours[expected])


def test_an_rle4_delta_moves_down_too(tmp_path: Path):
    stream = bytes([2, 0x33, 0, 2, 1, 1, 2, 0x55, 0, 1])
    image = load(written(tmp_path / "x.bmp", rle4_bmp(stream)))
    colours = RLE4_PALETTE[..., :3]
    expected = np.zeros((3, 7), np.uint8)
    expected[2, :2] = 3
    expected[1, 3:5] = 5
    assert np.array_equal(image, colours[expected])


@pytest.mark.parametrize(
    "masks,value",
    [
        (None, 0x7FFF),
        ((0x7C00, 0x3E0, 0x1F), 0x7FFF),
        ((0xF800, 0x7E0, 0x1F), 0xFFFF),
    ],
    ids=["555", "555_masks", "565"],
)
def test_a_16_bit_bmp_reaches_full_white(tmp_path: Path, masks, value: int):
    rows = np.full((H, W), value, np.uint16).astype("<u2")
    after = b"" if masks is None else struct.pack("<3I", *masks)
    data = bmp(16, bottom_up(rows, W * 2), 0 if masks is None else 3, after)
    assert np.all(load(written(tmp_path / "x.bmp", data)) == 255)


def tiff_tags(
    samples: np.ndarray, photometric: int, sample_format: int, planar: int
) -> dict:
    channels = 1 if samples.ndim == 2 else samples.shape[2]
    bits = samples.dtype.itemsize * 8
    tags = {
        258: ("H", [bits] * channels),
        259: ("H", [1]),
        262: ("H", [photometric]),
        277: ("H", [channels]),
        284: ("H", [planar]),
        339: ("H", [sample_format] * channels),
    }
    if channels == 4:
        tags[338] = ("H", [2])
    return tags


DEPTHS = [
    (np.uint16, 1),
    (np.uint32, 1),
    (np.float32, 3),
    (np.float64, 3),
]


def tiff_samples(dtype, shape: tuple[int, ...]) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return RNG.random(shape).astype(dtype)
    return RNG.integers(0, np.iinfo(dtype).max, shape, dtype=dtype, endpoint=True)


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("dtype,sample_format", DEPTHS)
def test_a_planar_tiff_of_strips_loads_its_samples(
    tmp_path: Path, dtype, sample_format: int, channels: int
):
    samples = tiff_samples(dtype, (H, W, channels))
    tags = tiff_tags(samples, 2, sample_format, 2)
    tags[278] = ("I", [3])  # three rows a strip: three strips a plane
    parts = [
        samples[y : y + 3, :, plane].tobytes()
        for plane in range(channels)
        for y in range(0, H, 3)
    ]
    image = load(written(tmp_path / "x.tif", tiff(W, H, tags, parts)))
    expected = samples[..., [2, 1, 0, 3][:channels]]
    assert image.dtype == dtype
    assert np.array_equal(image, expected)


@pytest.mark.parametrize("dtype,sample_format", DEPTHS)
def test_a_planar_tiff_of_tiles_loads_its_samples(
    tmp_path: Path, dtype, sample_format: int
):
    samples = tiff_samples(dtype, (23, 37, 3))
    tags = tiff_tags(samples, 2, sample_format, 2)
    tags[322], tags[323] = ("I", [16]), ("I", [16])
    parts = []
    for plane in range(3):
        for y in range(0, 23, 16):
            for x in range(0, 37, 16):
                tile = np.zeros((16, 16), dtype)
                part = samples[y : y + 16, x : x + 16, plane]
                tile[: part.shape[0], : part.shape[1]] = part
                parts.append(tile.tobytes())
    image = load(written(tmp_path / "x.tif", tiff(37, 23, tags, parts)))
    assert np.array_equal(image, samples[..., ::-1])


@pytest.mark.parametrize(
    "dtype,sample_format,invert",
    [
        (np.uint16, 1, lambda s: 65535 - s),
        (np.uint32, 1, lambda s: 4294967295 - s),
        (np.float32, 3, lambda s: 1 - s),
        (np.float64, 3, lambda s: 1 - s),
    ],
    ids=["u16", "u32", "f32", "f64"],
)
def test_a_deep_min_is_white_tiff_is_inverted(
    tmp_path: Path, dtype, sample_format: int, invert
):
    samples = tiff_samples(dtype, (H, W))
    tags = tiff_tags(samples, 0, sample_format, 1)
    tags[278] = ("I", [H])
    image = load(written(tmp_path / "x.tif", tiff(W, H, tags, [samples.tobytes()])))
    assert image.dtype == dtype
    assert np.array_equal(image, invert(samples))


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("suffix", [".tif", ".tiff"])
def test_a_tif_loads_as_a_tiff_does(tmp_path: Path, bits: int, suffix: str):
    top = 2**bits - 1
    samples = RNG.integers(0, top + 1, (H, W), dtype=np.uint32)
    rows = b"".join(packed_row(row, bits) for row in samples)
    tags = {
        258: ("H", [bits]),
        259: ("H", [1]),
        262: ("H", [1]),
        277: ("H", [1]),
        278: ("I", [H]),
    }
    image = load(written(tmp_path / f"x{suffix}", tiff(W, H, tags, [rows])))
    assert np.array_equal(image, samples * 255 // top)


@pytest.mark.parametrize("suffix", [".tga", ".jp2"])
def test_grey_alpha_through_pillow_loads_as_bgra(tmp_path: Path, suffix: str):
    samples = RNG.integers(0, 256, (16, 16, 2), dtype=np.uint8)
    path = tmp_path / f"x{suffix}"
    options = {"quality_mode": "rates", "irreversible": False} if suffix == ".jp2" else {}
    Image.fromarray(samples, "LA").save(path, **options)
    image = load(path)
    assert image.shape == (16, 16, 4)
    for channel in range(3):
        assert np.array_equal(image[..., channel], samples[..., 0])
    assert np.array_equal(image[..., 3], samples[..., 1])


def test_a_png_that_needs_a_palette_and_has_none_says_so(tmp_path: Path):
    samples = RNG.integers(0, 256, (H, W), dtype=np.uint32)
    path = written(tmp_path / "x.png", png(samples, 8, 3))
    with pytest.raises(ValueError, match="no palette") as caught:
        load(path)
    assert str(path) in str(caught.value)


IMAGE = np.full((4, 6, 3), 0.5, np.float32)


def save(directory: Path, relative: str | None, name: str) -> None:
    save_image_node(
        Lazy.ready(IMAGE),
        directory,
        relative,
        name,
        ImageFormat.PNG,
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
        AvifSubsampling.FACTOR_420,
        ExrColorDepth.F32,
        ExrCompression.ZIP,
        False,
        None,
    )


@pytest.mark.parametrize("name", ["", "sub/", "sub\\"])
def test_an_empty_image_name_is_refused(tmp_path: Path, name: str):
    with pytest.raises(ValueError, match="Image Name .* has no file name"):
        save(tmp_path, None, name)
    assert list(tmp_path.iterdir()) == []


windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows' file names")


@windows
@pytest.mark.parametrize("value", ["ab:cd", "ab?cd", "ab*cd", "ab<cd", "ab>cd", "ab|cd", 'ab"cd', "a\tb"])
def test_a_character_windows_refuses_is_named_and_leaves_no_file(
    tmp_path: Path, value: str
):
    with pytest.raises(ValueError, match="Image Name .* not allowed in a Windows"):
        save(tmp_path, None, value)
    with pytest.raises(ValueError, match="Subdirectory Path .* not allowed in a Windows"):
        save(tmp_path, f"sub/{value}", "x")
    assert list(tmp_path.iterdir()) == []


def test_a_name_ending_in_a_space_stays_allowed(tmp_path: Path):
    save(tmp_path, None, "ab ")
    assert [path.name for path in tmp_path.iterdir()] == ["ab .png"]
