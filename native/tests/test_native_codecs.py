"""The native codecs against today's path on every row of explore-image-io.md table (a).

Each case is a small file (decode) or a Save Image call (encode) made here, run twice in one process:
with CHAINNER_C_NATIVE_CODECS=0 (today's Pillow/OpenCV path, the oracle) and with it unset (native
codecs where they take the file). Decode compares Load Image's raw array, the normalised output and its
previews, or the exception's type and message, plus the warnings Load Image logs. Encode compares the
saved file's bytes and prepare_save_image's bytes, or the exception.

NATIVE_DECODE and NATIVE_ENCODE name the rows the native path must take: there, today's codec calls are
made to raise (Probe) and the file must still decode, or save, to the same result. A row is added as
its codec lands; KEPT_ON_TODAYS_PATH lists the cases of those rows that decline by design.
The full proof is the codec corpus (native/local/codec-corpus, gen_corpus.py and run_corpus.py).
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import io
import json
import logging
import os
import struct
import subprocess
import sys
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cv2
import numpy as np

# Imported for its side effect (x as x keeps it): registers Pillow's AVIF plugin.
import pillow_avif as pillow_avif
import pytest
from PIL import Image

from api import Lazy
from nodes.impl import native, native_image_io
from nodes.impl.native import lib
from nodes.properties.outputs import LargeImageOutput
from packages.chaiNNer_standard.image.io import load_image, save_image

# Decode rows (table (a)) the native path takes, with the Load Image reader that reaches it.
NATIVE_DECODE = {
    "jpeg-pillow": "_read_pil",
    "jpeg-opencv": "_read_cv",
    "png": "_read_cv",
    "webp": "_read_cv",
    "tiff": "_read_cv",
    "bmp": "_read_cv",
    "pxm": "_read_cv",
}
# Encode rows the native path takes (chaiNNer-C's 8-bit PNG encoder takes them with or without the other native codecs).
NATIVE_ENCODE = {
    "jpg",
    "png-u8",
    "png-u16",
    "webp-lossy",
    "webp-lossless-1-3ch",
    "webp-lossless-4ch",
    "bmp",
    "tiff-u8-u16",
    "tiff-f32",
}
# Cases of native rows that today's path decodes without a logged warning, which the native path declines by
# design: libjpeg warns (extraneous bytes, restart markers), an MPO, an APNG (DESIGN A8), a file shorter than
# imdecode's 32-byte signature, PFM content (not ported).
KEPT_ON_TODAYS_PATH = {
    "jpg_junk_between_markers",
    "jpg_rst_out_of_order",
    "jpe_rst_out_of_order",
    "jpg_mpo",
    "jpg_mpf_segment",
    "png_apng",
    "pxm_p4",
    "pxm_pfm_pxm",
}

RNG = np.random.default_rng(20261019)
RGBA8 = RNG.integers(
    0, 256, (9, 13, 4), dtype=np.uint8
)  # RGB order, odd size: partial MCUs
RGBA16 = RNG.integers(0, 65536, (9, 13, 4), dtype=np.uint16)
BGR8, BGRA8, GREY8 = (
    RGBA8[..., 2::-1].copy(),
    RGBA8[..., [2, 1, 0, 3]].copy(),
    RGBA8[..., 0].copy(),
)
BGR16, GREY16 = RGBA16[..., 2::-1].copy(), RGBA16[..., 0].copy()


def sha(data: bytes | np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(data).tobytes() if isinstance(data, np.ndarray) else data
    ).hexdigest()


class Probe:
    """A module whose `blocked` names raise; every other attribute is the module's."""

    def __init__(self, module: object, blocked: set[str]):
        self._module, self._blocked = module, blocked

    def __getattr__(self, name: str):
        if name in self._blocked:
            raise AssertionError(f"today's codec called: {name}")
        return getattr(self._module, name)


@contextmanager
def logged() -> Iterator[list[str]]:
    """The warnings the backend logger (sanic.root) logs inside the block."""
    records: list[str] = []
    handler = logging.Handler(logging.WARNING)
    handler.emit = lambda record: records.append(record.getMessage())
    logger = logging.getLogger("sanic.root")
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


# ---------------------------------------------------------------------------------------------
# Files


def segment(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload


def exif(orientation: int, endian: str = "<", header: bool = True) -> bytes:
    """An EXIF block whose IFD0 holds Orientation (SHORT)."""
    tiff = (b"II" if endian == "<" else b"MM") + struct.pack(endian + "HI", 42, 8)
    ifd = (
        struct.pack(endian + "HHHI", 1, 274, 3, 1)
        + struct.pack(endian + "HH", orientation, 0)
        + bytes(4)
    )
    return (b"Exif\0\0" if header else b"") + tiff + ifd


def exif_resolution(kind: int) -> bytes:
    """An EXIF block whose IFD0 holds XResolution (282) as one value of TIFF type `kind`
    ("5" for ASCII) and ResolutionUnit (296) 2, which Pillow reads when JFIF has no dpi."""
    ifd = (
        struct.pack("<HHHI", 2, 282, kind, 1)
        + b"5\0\0\0"
        + struct.pack("<HHIHH", 296, 3, 1, 2, 0)
        + bytes(4)
    )
    return b"Exif\0\0II" + struct.pack("<HI", 42, 8) + ifd


def with_segments(jpeg: bytes, *segments: bytes) -> bytes:
    """The JPEG with marker segments inserted after SOI."""
    return jpeg[:2] + b"".join(segments) + jpeg[2:]


def cv_bytes(ext: str, image: np.ndarray, params: list[int] | None = None) -> bytes:
    ok, data = cv2.imencode(ext, image, params or [])
    assert ok
    return data.tobytes()


def pil_bytes(image: Image.Image, fmt: str, **options) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)
    return buffer.getvalue()


def set_sof(
    jpeg: bytes, precision: int | None = None, height: int | None = None
) -> bytes:
    at = next(
        i
        for i in range(2, len(jpeg) - 1)
        if jpeg[i] == 0xFF and jpeg[i + 1] in (0xC0, 0xC1, 0xC2)
    )
    data = bytearray(jpeg)
    if precision is not None:
        data[at + 4] = precision
    if height is not None:
        data[at + 5 : at + 7] = struct.pack(">H", height)
    return bytes(data)


def without_marker(jpeg: bytes, marker: int) -> bytes:
    """The JPEG without its segments of one marker (before the scan)."""
    out, at = bytearray(jpeg[:2]), 2
    while jpeg[at + 1] != 0xDA:
        length = struct.unpack(">H", jpeg[at + 2 : at + 4])[0]
        if jpeg[at + 1] != marker:
            out += jpeg[at : at + 2 + length]
        at += 2 + length
    return bytes(out + jpeg[at:])


def associated(tiff: bytes) -> bytes:
    """Pillow's little-endian RGBA TIFF with ExtraSamples 1 (associated alpha) in place of 2."""
    entry = struct.pack("<HHIH", 338, 3, 1, 2)
    assert tiff.count(entry) == 1
    return tiff.replace(entry, struct.pack("<HHIH", 338, 3, 1, 1))


def png_chunk(kind: bytes, data: bytes, crc_ok: bool = True) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", crc if crc_ok else crc ^ 1)
    )


def png_adam7(samples: np.ndarray, color_type: int, depth: int) -> bytes:
    """An interlaced PNG (filter 0) of (h, w, c) samples at depth 2, 8 or 16."""
    h, w = samples.shape[:2]
    raw = b""
    for x0, y0, dx, dy in (
        (0, 0, 8, 8),
        (4, 0, 8, 8),
        (0, 4, 4, 8),
        (2, 0, 4, 4),
        (0, 2, 2, 4),
        (1, 0, 2, 2),
        (0, 1, 1, 2),
    ):
        sub = samples[y0::dy, x0::dx]
        for row in sub.reshape(sub.shape[0], -1):
            if depth == 2:
                padded = np.zeros(-(-row.size // 4) * 4, np.uint8)
                padded[: row.size] = row
                packed = (
                    padded[0::4] << 6
                    | padded[1::4] << 4
                    | padded[2::4] << 2
                    | padded[3::4]
                )
                raw += b"\0" + packed.astype(np.uint8).tobytes()
            else:
                raw += b"\0" + row.astype(">u2" if depth == 16 else np.uint8).tobytes()
    ihdr = struct.pack(">IIBBBBB", w, h, depth, color_type, 0, 0, 1)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(raw))
        + png_chunk(b"IEND", b"")
    )


def png_chunks(data: bytes) -> list[bytes]:
    chunks, at = [], 8
    while at < len(data):
        length = struct.unpack(">I", data[at : at + 4])[0]
        chunks.append(data[at : at + 12 + length])
        at += 12 + length
    return chunks


def png_exif_after_idat(data: bytes) -> bytes:
    """The PNG with its eXIf chunk moved after IDAT."""
    chunks = png_chunks(data)
    exif_chunk = next(c for c in chunks if c[4:8] == b"eXIf")
    rest = [c for c in chunks if c is not exif_chunk]
    return data[:8] + b"".join(rest[:-1]) + exif_chunk + rest[-1]


def png_bad_idat_crc(data: bytes) -> bytes:
    return data[:8] + b"".join(
        png_chunk(c[4:8], c[8:-4], c[4:8] != b"IDAT") for c in png_chunks(data)
    )


def png_filtered(
    color_type: int,
    channels: int,
    filters: list[int],
    pieces: int = 1,
    wbits: int = 15,
    stream_end: Callable[[bytes], bytes] = lambda stream: stream,
    before_idat: bytes = b"",
    after_idat: bytes = b"",
    iend_crc_ok: bool = True,
    depth: int = 8,
    width: int = 13,
    values: list[int] | None = None,
) -> bytes:
    """A width-wide PNG of depth-bit filtered rows, random (random picks of values if given), one per filter type in
    filters, deflated with a 2^wbits window; stream_end rewrites the zlib stream, which is split over pieces IDAT
    chunks between before_idat and after_idat."""
    rng = np.random.default_rng(len(filters) * 7 + color_type)
    shape = (len(filters), (width * channels * depth + 7) // 8)
    samples = (
        rng.integers(0, 256, shape, dtype=np.uint8)
        if values is None
        else rng.choice(np.array(values, dtype=np.uint8), shape)
    )
    raw = b"".join(
        bytes([kind]) + row.tobytes()
        for kind, row in zip(filters, samples, strict=True)
    )
    compressor = zlib.compressobj(9, zlib.DEFLATED, wbits)
    stream = stream_end(compressor.compress(raw) + compressor.flush())
    cuts = [len(stream) * k // pieces for k in range(pieces + 1)]
    ihdr = struct.pack(">IIBBBBB", width, len(filters), depth, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + before_idat
        + b"".join(
            png_chunk(b"IDAT", stream[cuts[k] : cuts[k + 1]]) for k in range(pieces)
        )
        + after_idat
        + png_chunk(b"IEND", b"", iend_crc_ok)
    )


def png_flushed(
    color_type: int,
    channels: int,
    height: int = 40,
    cuts: list[int] | None = None,
    flush: int = zlib.Z_FULL_FLUSH,
    edit: Callable[[list[bytes]], list[bytes]] = lambda pieces: pieces,
    before_idat: bytes = b"",
    depth: int = 8,
) -> bytes:
    """A 13-wide PNG of three random rows in turn with every filter type, so that matches reach back over rows. Its
    zlib stream is flushed at each cut of the rows and split there into IDAT chunks: by default the cuts of
    png_encode.c's rule for five bands. The last chunk ends with an empty final block (03 00) and the Adler-32; edit
    rewrites the chunks' bytes."""
    width = 13
    patterns = np.random.default_rng(height + color_type).integers(
        0, 256, (3, (width * channels * depth + 7) // 8), dtype=np.uint8
    )
    rows = [bytes([y % 5]) + patterns[y % 3].tobytes() for y in range(height)]
    cuts = cuts or [height * k // 5 for k in range(6)]
    compressor = zlib.compressobj(9, zlib.DEFLATED, 15)
    pieces = [
        compressor.compress(b"".join(rows[cuts[k] : cuts[k + 1]]))
        + compressor.flush(flush)
        for k in range(len(cuts) - 1)
    ]
    pieces[-1] += compressor.flush()
    ihdr = struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + before_idat
        + b"".join(png_chunk(b"IDAT", piece) for piece in edit(pieces))
        + png_chunk(b"IEND", b"")
    )


def reaching_back(pieces: list[bytes]) -> list[bytes]:
    """The pieces, once shown not to be independent: the second does not inflate alone."""
    with pytest.raises(zlib.error, match="too far back"):
        zlib.decompressobj(-15).decompress(pieces[1])
    return pieces


def png_stored_tail() -> bytes:
    """Two rows in stored blocks, one per IDAT chunk; the first chunk's non-empty block ends in the bytes of an empty
    stored block."""
    row = bytes([0]) + bytes(35) + b"\0\0\xff\xff"
    stored = struct.pack("<HH", len(row), len(row) ^ 0xFFFF) + row
    ihdr = struct.pack(">IIBBBBB", 13, 2, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", b"\x78\x01\x00" + stored)
        + png_chunk(
            b"IDAT", b"\x01" + stored + struct.pack(">I", zlib.adler32(row + row))
        )
        + png_chunk(b"IEND", b"")
    )


def png_indices(past_palette: bool) -> bytes:
    """An image of 16-colour indices large enough for several store chunks; past_palette puts an index outside the
    palette in a late row."""
    side = 1100
    rows = np.zeros((side, side + 1), np.uint8)
    rows[:, 1:] = np.random.default_rng(9).integers(0, 16, (side, side), dtype=np.uint8)
    if past_palette:
        rows[1000, 500] = 200
    ihdr = struct.pack(">IIBBBBB", side, side, 8, 3, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_palette(16, True)
        + png_chunk(b"IDAT", zlib.compress(rows.tobytes(), 1))
        + png_chunk(b"IEND", b"")
    )


# The parallel inflate (png_inflate.c) takes streams from 1 MiB of compressed data at 5:1 or less. Every byte of these
# rows is a valid filter type, so that a crafted match anywhere keeps the rows valid; they compress about 3:1.
INFLATE_WIDTH = 1024
INFLATE_HEIGHT = 2400
INFLATE_SIZE = (3 * INFLATE_WIDTH + 1) * INFLATE_HEIGHT
INFLATE_DYNAMIC = (6, zlib.Z_DEFAULT_STRATEGY)
INFLATE_FIXED = (6, zlib.Z_FIXED)
INFLATE_STORED = (0, zlib.Z_DEFAULT_STRATEGY)
INFLATE_HUFFMAN = (6, zlib.Z_HUFFMAN_ONLY)
INFLATE_RLE = (6, zlib.Z_RLE)


def inflate_noise(size: int) -> bytes:
    return np.random.default_rng(size).integers(0, 5, size, dtype=np.uint8).tobytes()


@cache
def inflate_rows() -> bytes:
    """Noise with a 512-byte copy every 8 KiB from one row, three rows or nearly the whole window back."""
    rows = bytearray(inflate_noise(INFLATE_SIZE))
    for index, at in enumerate(range(32768, INFLATE_SIZE - 512, 8192)):
        distance = (3073, 9219, 32000)[index % 3]
        rows[at : at + 512] = rows[at - distance : at - distance + 512]
    return bytes(rows)


@cache
def inflate_level6() -> bytes:
    return zlib.compress(inflate_rows(), 6)


def raw_deflate(
    data: bytes,
    kind: tuple[int, int] = INFLATE_DYNAMIC,
    history: bytes = b"",
    final: bool = False,
) -> bytes:
    """Raw DEFLATE of data whose matches may reach history, ending on a byte boundary or, if final, the stream."""
    level, strategy = kind
    arguments = (level, zlib.DEFLATED, -15, 8, strategy)
    compressor = (
        zlib.compressobj(*arguments, history[-32768:])
        if history
        else zlib.compressobj(*arguments)
    )
    ending = zlib.Z_FINISH if final else zlib.Z_SYNC_FLUSH
    return compressor.compress(data) + compressor.flush(ending)


def raw_deflate_pieces(
    rows: bytes, sizes: list[int], kinds: list[tuple[int, int]]
) -> bytes:
    """rows deflated in pieces of the sizes and kinds in turn, each with the output before it as history."""
    pieces, at, index = [], 0, 0
    while at < len(rows):
        end = at + sizes[index % len(sizes)]
        pieces.append(
            raw_deflate(
                rows[at:end],
                kinds[index % len(kinds)],
                rows[max(at - 32768, 0) : at],
                end >= len(rows),
            )
        )
        at, index = end, index + 1
    return b"".join(pieces)


def far_match_block() -> bytes:
    """A fixed Huffman block holding one match of length 258 at distance 32768, then an empty stored block."""
    bits = 0
    count = 0

    def put(value: int, width: int) -> None:
        nonlocal bits, count
        bits |= value << count
        count += width

    def code(value: int, width: int) -> None:
        for bit in reversed(range(width)):
            put((value >> bit) & 1, 1)

    put(0, 1)  # not final
    put(1, 2)  # fixed
    code(0b11000101, 8)  # length symbol 285: 258
    code(29, 5)  # distance symbol 29: 24577 and 13 extra bits
    put(32768 - 24577, 13)
    code(0, 7)  # end of block
    put(0, 3)  # stored, not final
    return bits.to_bytes((count + 7) // 8, "little") + b"\x00\x00\xff\xff"


def zlib_stream(deflate: bytes, rows: bytes) -> bytes:
    return b"\x78\x9c" + deflate + zlib.adler32(rows).to_bytes(4, "big")


def png_rows(stream: bytes, height: int = INFLATE_HEIGHT) -> bytes:
    """An INFLATE_WIDTH-wide RGB PNG whose rows are the zlib stream, in three IDAT chunks."""
    ihdr = struct.pack(">IIBBBBB", INFLATE_WIDTH, height, 8, 2, 0, 0, 0)
    cuts = [len(stream) * k // 3 for k in range(4)]
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + b"".join(png_chunk(b"IDAT", stream[cuts[k] : cuts[k + 1]]) for k in range(3))
        + png_chunk(b"IEND", b"")
    )


def inflate_far_matches() -> bytes:
    """A match at the longest distance after every small dynamic block: some land right after a stretch's start."""
    noise = inflate_noise(INFLATE_SIZE)
    rows = bytearray(noise[:40000])
    deflate = bytearray(raw_deflate(bytes(rows)))
    while len(rows) + 20000 + 258 < INFLATE_SIZE:
        piece = noise[len(rows) : len(rows) + 20000]
        deflate += raw_deflate(piece, history=bytes(rows[-32768:]))
        rows += piece
        deflate += far_match_block()
        rows += bytes(rows[-32768 + k] for k in range(258))
    rest = noise[len(rows) :]
    deflate += raw_deflate(rest, history=bytes(rows[-32768:]), final=True)
    return png_rows(zlib_stream(bytes(deflate), bytes(rows) + rest))


def inflate_lasting_markers() -> bytes:
    """A tag every 16 KiB, each a match of the one before: output from before a stretch to the end of every stretch."""
    rows = bytearray(inflate_noise(INFLATE_SIZE))
    tag = inflate_noise(24)
    for at in range(0, INFLATE_SIZE - 24, 16384):
        rows[at : at + 24] = tag
    return png_rows(zlib.compress(bytes(rows), 6))


def inflate_false_starts() -> bytes:
    """Stored data that holds a real DEFLATE stream: dynamic block headers the true parse never sees, between true blocks."""
    rows = inflate_rows()
    half = len(rows) // 2
    payload = raw_deflate(inflate_noise(len(rows)))[: len(rows) - half]
    first = rows[:half]
    deflate = (
        raw_deflate(first)
        + raw_deflate(payload, INFLATE_STORED, first)
        + raw_deflate(b"", final=True)
    )
    return png_rows(zlib_stream(deflate, first + payload))


def inflate_past_start() -> bytes:
    """A match past the stream's real start, early on: the serial decoder's error, and the first stretch's."""
    early = inflate_noise(1000)
    rows = inflate_rows()[: INFLATE_SIZE - 1258]
    deflate = (
        raw_deflate(early)
        + far_match_block()
        + raw_deflate(rows, history=early, final=True)
    )
    return png_rows(zlib_stream(deflate, early + bytes(258) + rows))


def inflate_cases() -> dict[str, Callable[[], bytes]]:
    rows_cut = INFLATE_SIZE // 3

    def flip(stream: bytes, at: int) -> bytes:
        return stream[:at] + bytes([stream[at] ^ 16]) + stream[at + 1 :]

    return {
        **{
            f"inflate_level{level}": (
                lambda level=level: png_rows(zlib.compress(inflate_rows(), level))
            )
            for level in (1, 6, 9)
        },
        "inflate_block_types": lambda: png_rows(
            zlib_stream(
                raw_deflate_pieces(
                    inflate_rows(),
                    [400000, 120000, 280000, 200000, 80000],
                    [
                        INFLATE_DYNAMIC,
                        INFLATE_FIXED,
                        INFLATE_STORED,
                        INFLATE_HUFFMAN,
                        INFLATE_RLE,
                    ],
                ),
                inflate_rows(),
            )
        ),
        "inflate_stored_runs": lambda: png_rows(
            zlib_stream(
                raw_deflate_pieces(
                    inflate_rows(), [700000], [INFLATE_DYNAMIC, INFLATE_STORED]
                ),
                inflate_rows(),
            )
        ),
        "inflate_only_fixed_after_first": lambda: png_rows(
            zlib_stream(
                raw_deflate_pieces(
                    inflate_rows(),
                    [60000, INFLATE_SIZE],
                    [INFLATE_DYNAMIC, INFLATE_FIXED],
                ),
                inflate_rows(),
            )
        ),
        "inflate_final_stored": lambda: png_rows(
            zlib_stream(
                raw_deflate(inflate_rows()) + b"\x01\x00\x00\xff\xff", inflate_rows()
            )
        ),
        "inflate_far_matches": inflate_far_matches,
        "inflate_lasting_markers": inflate_lasting_markers,
        "inflate_false_starts": inflate_false_starts,
        "inflate_past_start": inflate_past_start,
        # Output far from its share of the compressed bytes, at either end.
        "inflate_zeros_at_end": lambda: png_rows(
            zlib.compress(inflate_rows()[:-rows_cut] + bytes(rows_cut), 6)
        ),
        "inflate_run_at_start": lambda: png_rows(
            zlib.compress(b"\x03" * rows_cut + inflate_rows()[rows_cut:], 6)
        ),
        # What the stream's checks must still catch.
        "inflate_bad_adler": lambda: png_rows(
            flip(inflate_level6(), len(inflate_level6()) - 1)
        ),
        "inflate_bit_flip_early": lambda: png_rows(flip(inflate_level6(), 300000)),
        "inflate_bit_flip_late": lambda: png_rows(
            flip(inflate_level6(), len(inflate_level6()) - 300000)
        ),
        "inflate_truncated": lambda: png_rows(inflate_level6()[:-5000]),
        "inflate_trailing_byte": lambda: png_rows(inflate_level6() + b"\0"),
        "inflate_too_few_rows": lambda: png_rows(inflate_level6(), INFLATE_HEIGHT - 1),
        "inflate_too_many_rows": lambda: png_rows(inflate_level6(), INFLATE_HEIGHT + 1),
    }


def png_palette(colors: int, trns: bool) -> bytes:
    """PLTE of colors random colours, and with trns a tRNS for the first half of them."""
    rng = np.random.default_rng(colors)
    chunks = png_chunk(
        b"PLTE", rng.integers(0, 256, colors * 3, dtype=np.uint8).tobytes()
    )
    if trns:
        chunks += png_chunk(
            b"tRNS", rng.integers(0, 256, max(colors // 2, 1), dtype=np.uint8).tobytes()
        )
    return chunks


def bmp_rle8(indices: np.ndarray, palette: np.ndarray) -> bytes:
    """A BI_RLE8 bitmap of one-index runs (bottom-up rows)."""
    h, w = indices.shape
    data = bytearray()
    for row in indices[::-1]:
        for value in row.tolist():
            data += bytes([1, value])
        data += b"\0\0"
    data = data[:-2] + b"\0\1"
    info = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 8, 1, len(data), 2835, 2835, 256, 0)
    table = b"".join(bytes([b, g, r, 0]) for r, g, b in palette)
    offset = 14 + len(info) + len(table)
    return (
        b"BM"
        + struct.pack("<IHHI", offset + len(data), 0, 0, offset)
        + info
        + table
        + bytes(data)
    )


def pnm(kind: str, samples: np.ndarray, maxval: int) -> bytes:
    h, w = samples.shape[:2]
    head = f"{kind}\n{w} {h}\n" + ("" if kind in ("P1", "P4") else f"{maxval}\n")
    if kind == "P1":
        return (
            head
            + "\n".join(
                " ".join(str(int(v)) for v in row) for row in samples.reshape(h, w)
            )
        ).encode()
    if kind == "P4":
        return (
            head.encode()
            + np.packbits(samples.reshape(h, w).astype(np.uint8), axis=1).tobytes()
        )
    if kind in ("P2", "P3"):
        return (
            head
            + "\n".join(
                " ".join(str(int(v)) for v in row) for row in samples.reshape(h, -1)
            )
        ).encode()
    return head.encode() + samples.astype(">u2" if maxval > 255 else np.uint8).tobytes()


@dataclass(frozen=True)
class DecodeCase:
    name: str
    row: str
    ext: str
    data: Callable[[], bytes]
    foreign: bool = False  # content of another format than the extension's row


def decode_cases() -> list[DecodeCase]:
    rgb, rgba, grey = (
        Image.fromarray(RGBA8[..., :3]),
        Image.fromarray(RGBA8),
        Image.fromarray(GREY8),
    )
    base = cv_bytes(".jpg", BGR8, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cmyk = pil_bytes(rgb.convert("CMYK"), "JPEG", quality=90)
    adobe_at = cmyk.index(b"Adobe")
    ycck = cmyk[: adobe_at + 11] + b"\x02" + cmyk[adobe_at + 12 :]
    # 36x52: several MCUs, so restart markers are written
    rst = cv_bytes(
        ".jpg",
        np.tile(BGR8, (4, 4, 1)),
        [cv2.IMWRITE_JPEG_QUALITY, 90, cv2.IMWRITE_JPEG_RST_INTERVAL, 1],
    )
    rst_at = rst.index(b"\xff\xd1", rst.index(b"\xff\xda"))
    dqt = base.index(b"\xff\xdb")
    cases = []

    def add(
        name: str, row: str, ext: str, make: Callable[[], bytes], foreign: bool = False
    ) -> None:
        cases.append(DecodeCase(name, row, ext, make, foreign))

    jpegs: dict[str, Callable[[], bytes]] = {
        "bgr_420": lambda: base,
        "444": lambda: cv_bytes(
            ".jpg",
            BGR8,
            [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444],
        ),
        "422": lambda: cv_bytes(
            ".jpg",
            BGR8,
            [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422],
        ),
        "440": lambda: cv_bytes(
            ".jpg",
            BGR8,
            [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_440],
        ),
        "progressive": lambda: cv_bytes(
            ".jpg", BGR8, [cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
        ),
        "grey": lambda: cv_bytes(".jpg", GREY8),
        "restart": lambda: rst,
        "cmyk": lambda: cmyk,
        "ycck": lambda: ycck,
        "cmyk_no_adobe": lambda: without_marker(cmyk, 0xEE),
        "cmyk_exif6": lambda: with_segments(cmyk, segment(0xE1, exif(6))),
        "exif6_be": lambda: with_segments(base, segment(0xE1, exif(6, ">"))),
        "no_dht": lambda: without_marker(base, 0xC4),
        "truncated": lambda: base[: len(base) // 2],
        "rst_out_of_order": lambda: rst[:rst_at] + b"\xff\xd5" + rst[rst_at + 2 :],
    }
    for o in range(1, 9):
        jpegs[f"exif{o}"] = lambda o=o: with_segments(base, segment(0xE1, exif(o)))
    for name, make in jpegs.items():
        add(f"jpe_{name}", "jpeg-opencv", ".jpe", make)
    jpegs.update(
        {
            "exif3_be": lambda: with_segments(base, segment(0xE1, exif(3, ">"))),
            "two_exif_6_3": lambda: with_segments(
                base, segment(0xE1, exif(6)), segment(0xE1, exif(3))
            ),
            "pillow_q75": lambda: pil_bytes(rgb, "JPEG", quality=75),
            "pillow_444_prog_opt": lambda: pil_bytes(
                rgb, "JPEG", quality=95, subsampling=0, progressive=True, optimize=True
            ),
            "pillow_grey": lambda: pil_bytes(grey, "JPEG"),
            "icc": lambda: pil_bytes(rgb, "JPEG", icc_profile=bytes(range(256)) * 4),
            "icc_short_fragment": lambda: with_segments(
                base, segment(0xE2, b"ICC_PROFILE\0\x01")
            ),
            "mpo": lambda: pil_bytes(
                rgb, "MPO", save_all=True, append_images=[rgb.rotate(180)]
            ),
            "mpf_segment": lambda: with_segments(
                base, segment(0xE2, b"MPF\0II*\0\x08\0\0\0\0\0")
            ),
            "photoshop": lambda: with_segments(
                base,
                segment(
                    0xED, b"Photoshop 3.0\x008BIM\x03\xed\0\0\0\0\0\x10" + bytes(16)
                ),
            ),
            "jfif_short": lambda: (
                without_marker(base, 0xE0)[:2]
                + segment(0xE0, b"JFIF\0")
                + without_marker(base, 0xE0)[2:]
            ),
            "adobe_short": lambda: with_segments(base, segment(0xEE, b"Adobe\0")),
            "no_eoi": lambda: base[:-2],
            "junk_between_markers": lambda: base[:dqt] + b"\x00\x12\x34" + base[dqt:],
            "trailing_garbage": lambda: base + bytes(range(200)),
            "precision12": lambda: set_sof(base, precision=12),
            "dnl": lambda: (
                set_sof(base, height=0)[:-2] + b"\xff\xdc\x00\x04\x00\x09\xff\xd9"
            ),
            "1x1": lambda: cv_bytes(".jpg", BGR8[:1, :1].copy()),
        }
    )
    # XResolution of one BYTE, ASCII or UNDEFINED value: Pillow's _read_dpi_from_exif
    # indexes it past its end (IndexError), Image.open fails and OpenCV decodes.
    for kind in (1, 2, 7):
        for name, jpeg in (("", base), ("cmyk_", cmyk)):
            jpegs[f"{name}exif_xres_type{kind}"] = lambda jpeg=jpeg, kind=kind: (
                with_segments(
                    without_marker(jpeg, 0xE0), segment(0xE1, exif_resolution(kind))
                )
            )
    for name, make in jpegs.items():
        add(f"jpg_{name}", "jpeg-pillow", ".jpg", make)
    add("jpeg_ext", "jpeg-pillow", ".jpeg", lambda: base)
    add("jfif_ext", "jpeg-pillow", ".jfif", lambda: cmyk)
    add(
        "jpg_upper_ext",
        "jpeg-pillow",
        ".JPG",
        lambda: with_segments(base, segment(0xE1, exif(8))),
    )

    png_exif = pil_bytes(rgb, "PNG", exif=exif(6))
    pngs: dict[str, Callable[[], bytes]] = {
        "grey8": lambda: cv_bytes(".png", GREY8),
        "bgr8": lambda: cv_bytes(".png", BGR8),
        "bgra8": lambda: cv_bytes(".png", BGRA8),
        "grey16": lambda: cv_bytes(".png", GREY16),
        "bgr16": lambda: cv_bytes(".png", BGR16),
        "bgra16": lambda: cv_bytes(".png", RGBA16[..., [2, 1, 0, 3]].copy()),
        "palette_trns": lambda: pil_bytes(
            rgb.quantize(64), "PNG", transparency=bytes(range(0, 255, 7))
        ),
        "palette4": lambda: pil_bytes(rgb.quantize(16), "PNG", bits=4),
        "la": lambda: pil_bytes(rgba.convert("LA"), "PNG"),
        "bilevel": lambda: pil_bytes(grey.convert("1"), "PNG"),
        "rgb_trns": lambda: pil_bytes(
            rgb, "PNG", transparency=tuple(int(v) for v in RGBA8[1, 1, :3])
        ),
        "adam7_rgb8": lambda: png_adam7(RGBA8[..., :3], 2, 8),
        "adam7_grey2": lambda: png_adam7(GREY8[..., None] >> 6, 0, 2),
        "adam7_rgba16": lambda: png_adam7(RGBA16, 6, 16),
        "exif6": lambda: png_exif,
        "exif6_after_idat": lambda: png_exif_after_idat(png_exif),
        "bad_idat_crc": lambda: png_bad_idat_crc(cv_bytes(".png", BGR8)),
        "truncated": lambda: cv_bytes(".png", BGR8)[:60],
        "apng": lambda: pil_bytes(
            rgba, "PNG", save_all=True, append_images=[rgba.rotate(180)]
        ),
        # The direct path (png_codec.c png_decode_direct): every filter on every first row and channel count, and
        # the files it leaves to libpng.
        **{
            f"filters_{first}_{channels}": (
                lambda first=first, channels=channels: png_filtered(
                    {1: 0, 3: 2, 4: 6}[channels],
                    channels,
                    [(first + y) % 5 for y in range(11)],
                    pieces=1 + first % 3,
                )
            )
            for first in range(5)
            for channels in (1, 3, 4)
        },
        **{
            f"filters16_{first}_{channels}": (
                lambda first=first, channels=channels: png_filtered(
                    {1: 0, 3: 2, 4: 6}[channels],
                    channels,
                    [(first + y) % 5 for y in range(11)],
                    pieces=1 + first % 2,
                    depth=16,
                )
            )
            for first in (2, 4)
            for channels in (1, 3, 4)
        },
        # Indexed: a full palette at each depth (every index inside it), with and without tRNS, and indices past a
        # 16-colour palette (libpng's warning and black, through the fallback).
        **{
            f"palette{depth}{'_trns' if trns else ''}": (
                lambda depth=depth, trns=trns: png_filtered(
                    3,
                    1,
                    [4, 1, 2, 3, 0, 4],
                    depth=depth,
                    before_idat=png_palette(2**depth, trns),
                )
            )
            for depth in (1, 2, 4, 8)
            for trns in (False, True)
        },
        "palette8_past_16": lambda: png_filtered(
            3, 1, [4, 1, 2, 3, 0, 4], before_idat=png_palette(16, True)
        ),
        "window_512": lambda: png_filtered(2, 3, [4, 1, 2, 3, 0], wbits=9),
        **inflate_cases(),
        "bad_filter": lambda: png_filtered(2, 3, [4, 1, 5, 3, 0]),
        # An invalid filter type in an indexed image: libpng's error, not decoded pixels.
        **{
            f"palette{depth}_bad_filter": (
                lambda depth=depth: png_filtered(
                    3,
                    1,
                    [4, 1, 7, 3, 0],
                    depth=depth,
                    before_idat=png_palette(2**depth, False),
                )
            )
            for depth in (2, 8)
        },
        "extra_zlib_data": lambda: png_filtered(
            2, 3, [4, 1, 2], stream_end=lambda stream: stream + b"\0\0"
        ),
        "bad_adler": lambda: png_filtered(
            6, 4, [4, 1, 2], stream_end=lambda s: s[:-1] + bytes([s[-1] ^ 1])
        ),
        "short_zlib_stream": lambda: png_filtered(
            0, 1, [4, 1, 2], stream_end=lambda stream: stream[:-6]
        ),
        "text_after_idat": lambda: png_filtered(
            2, 3, [4, 1, 2], after_idat=png_chunk(b"tEXt", b"k\0v")
        ),
        "iend_bad_crc": lambda: png_filtered(2, 3, [4, 1, 2], iend_crc_ok=False),
        "grey_gamma_srgb": lambda: png_filtered(
            0,
            1,
            [4, 1, 2, 3],
            before_idat=png_chunk(b"gAMA", struct.pack(">I", 45455))
            + png_chunk(b"sRGB", b"\0")
            + png_chunk(b"pHYs", struct.pack(">IIB", 2835, 2835, 1)),
        ),
        # Several store chunks (png_codec.c png_store_range): 16-bit samples and indices; test_png_encode.py has 8-bit.
        "store_chunks_grey16": lambda: cv_bytes(
            ".png",
            np.random.default_rng(1).integers(0, 65536, (1100, 600), dtype=np.uint16),
        ),
        "store_chunks_rgba16": lambda: cv_bytes(
            ".png",
            np.random.default_rng(2).integers(0, 65536, (600, 300, 4), dtype=np.uint16),
        ),
        "store_chunks_palette": lambda: png_indices(False),
        "store_chunks_past_palette": lambda: png_indices(True),
        # Independent pieces, inflated in parallel (png_codec.c png_inflate_pieces), in every format.
        "flushed_grey": lambda: png_flushed(0, 1),
        "flushed_rgb": lambda: png_flushed(2, 3),
        "flushed_rgba": lambda: png_flushed(6, 4),
        "flushed_rgb16": lambda: png_flushed(2, 3, depth=16),
        "flushed_palette8": lambda: png_flushed(
            3, 1, before_idat=png_palette(256, True)
        ),
        # An index past the palette among many rows: libpng's result, whichever store chunk meets it.
        "flushed_palette8_past_16": lambda: png_flushed(
            3, 1, before_idat=png_palette(16, True)
        ),
        # Pieces that fail the parallel try, for the stream as a whole to decide: valid files first.
        "flushed_sync_reach_back": lambda: png_flushed(
            2, 3, flush=zlib.Z_SYNC_FLUSH, edit=reaching_back
        ),
        "flushed_other_split": lambda: png_flushed(2, 3, cuts=[0, 1, 20, 40]),
        "flushed_more_chunks_than_rows": lambda: png_flushed(
            2,
            3,
            height=3,
            cuts=[0, 1, 2, 3],
            edit=lambda p: p[:2] + [b"\0\0\0\xff\xff"] * 2 + p[2:],
        ),
        "flushed_empty_chunk": lambda: png_flushed(
            2, 3, edit=lambda p: p[:1] + [b""] + p[1:]
        ),
        "flushed_empty_first_chunk": lambda: png_flushed(
            2, 3, edit=lambda p: [b""] + p
        ),
        "flushed_split_adler": lambda: png_flushed(
            2, 3, edit=lambda p: p[:-1] + [p[-1][:-2], p[-1][-2:]]
        ),
        "flushed_final_then_adler_chunk": lambda: png_flushed(
            2,
            3,
            edit=lambda p: p[:-1] + [p[-1][:-6] + b"\1\0\0\xff\xff", p[-1][-4:]],
        ),
        "flushed_stored_tail": png_stored_tail,
        # And damaged files.
        "flushed_final_stored_in_middle": lambda: png_flushed(
            2, 3, edit=lambda p: [p[0], p[1] + b"\1\0\0\xff\xff", *p[2:]]
        ),
        "flushed_final_huffman_in_middle": lambda: png_flushed(
            2, 3, edit=lambda p: [p[0], p[1] + b"\3\0\0\0\xff\xff", *p[2:]]
        ),
        "flushed_bad_adler": lambda: png_flushed(
            6, 4, edit=lambda p: p[:-1] + [p[-1][:-1] + bytes([p[-1][-1] ^ 1])]
        ),
        "flushed_data_after_end": lambda: png_flushed(
            2, 3, edit=lambda p: p[:-1] + [p[-1] + b"\0\0"]
        ),
        "flushed_short_last_chunk": lambda: png_flushed(
            2, 3, edit=lambda p: p[:-1] + [b"\3"]
        ),
        "flushed_tiny_first_chunk": lambda: png_flushed(
            2, 3, edit=lambda p: [b"\0\0\xff\xff"] + p
        ),
        "flushed_header_only_first_chunk": lambda: png_flushed(
            2, 3, edit=lambda p: [p[0][:2] + b"\0\0\xff\xff", p[0][2:], *p[1:]]
        ),
        "flushed_cut_piece": lambda: png_flushed(
            2, 3, edit=lambda p: [p[0][:-12] + b"\0\0\xff\xff", *p[1:]]
        ),
    }
    for name, make in pngs.items():
        add(f"png_{name}", "png", ".png", make)
    webps: dict[str, Callable[[], bytes]] = {
        "bgr_q80": lambda: cv_bytes(".webp", BGR8, [cv2.IMWRITE_WEBP_QUALITY, 80]),
        "bgra_q80": lambda: cv_bytes(".webp", BGRA8, [cv2.IMWRITE_WEBP_QUALITY, 80]),
        "bgr_lossless": lambda: cv_bytes(
            ".webp", BGR8, [cv2.IMWRITE_WEBP_QUALITY, 101]
        ),
        "bgra_lossless": lambda: cv_bytes(
            ".webp", BGRA8, [cv2.IMWRITE_WEBP_QUALITY, 101]
        ),
        "pillow_exif6": lambda: pil_bytes(
            rgb, "WEBP", quality=90, exif=exif(6, header=False)
        ),
        "pillow_rgba_alph": lambda: pil_bytes(rgba, "WEBP", quality=70),
        "pillow_grey": lambda: pil_bytes(grey, "WEBP", lossless=True),
        "animated": lambda: pil_bytes(
            rgba, "WEBP", save_all=True, append_images=[rgba.rotate(180)], lossless=True
        ),
        "animated_lossy": lambda: pil_bytes(
            rgb, "WEBP", save_all=True, append_images=[rgb.rotate(180)], quality=60
        ),
        "1x1": lambda: cv_bytes(
            ".webp", BGR8[:1, :1].copy(), [cv2.IMWRITE_WEBP_QUALITY, 101]
        ),
        "truncated": lambda: cv_bytes(".webp", BGR8)[:40],
    }
    for name, make in webps.items():
        add(f"webp_{name}", "webp", ".webp", make)
    for name, make in {
        "rgba": lambda: pil_bytes(rgba, "TIFF"),
        "rgba_lzw": lambda: pil_bytes(rgba, "TIFF", compression="tiff_lzw"),
        "la": lambda: pil_bytes(rgba.convert("LA"), "TIFF"),
        "rgba_o6": lambda: pil_bytes(rgba, "TIFF", tiffinfo={274: 6}),
        "rgba_bigtiff": lambda: pil_bytes(rgba, "TIFF", big_tiff=True),
    }.items():
        add(f"tif_{name}", "tiff-straight-alpha", ".tif", make)
    for name, make in {
        "bgr8": lambda: cv_bytes(".tif", BGR8),
        "bgra8": lambda: cv_bytes(".tif", BGRA8),
        "grey16": lambda: cv_bytes(".tif", GREY16),
        "bgr16": lambda: cv_bytes(".tif", BGR16),
        "f32": lambda: cv_bytes(".tif", BGR8.astype(np.float32) / 200 - 0.1),
        "deflate": lambda: pil_bytes(rgb, "TIFF", compression="tiff_adobe_deflate"),
        "packbits": lambda: pil_bytes(rgb, "TIFF", compression="packbits"),
        "jpeg": lambda: pil_bytes(rgb, "TIFF", compression="jpeg"),
        "grey": lambda: pil_bytes(grey, "TIFF"),
        "group4": lambda: pil_bytes(grey.convert("1"), "TIFF", compression="group4"),
        "palette": lambda: pil_bytes(rgb.quantize(32), "TIFF"),
        "cmyk": lambda: pil_bytes(rgb.convert("CMYK"), "TIFF"),
        "i16": lambda: pil_bytes(Image.fromarray(GREY16), "TIFF"),
        "float": lambda: pil_bytes(grey.convert("F"), "TIFF"),
        "rgba_assoc": lambda: associated(pil_bytes(rgba, "TIFF")),
        "rgb_o3": lambda: pil_bytes(rgb, "TIFF", tiffinfo={274: 3}),
        "truncated": lambda: pil_bytes(rgb, "TIFF")[:100],
    }.items():
        add(f"tif_{name}", "tiff", ".tiff" if name == "deflate" else ".tif", make)
    for name, make in {
        "grey": lambda: cv_bytes(".bmp", GREY8),
        "bgr": lambda: cv_bytes(".bmp", BGR8),
        "bgra": lambda: cv_bytes(".bmp", BGRA8),
        "bilevel": lambda: pil_bytes(grey.convert("1"), "BMP"),
        "palette": lambda: pil_bytes(rgb.quantize(16), "BMP"),
        "rle8": lambda: bmp_rle8(GREY8 >> 4, RNG.integers(0, 256, (256, 3))),
        "truncated": lambda: cv_bytes(".bmp", BGR8)[:-40],
    }.items():
        add(f"bmp_{name}", "bmp", ".bmp", make)
    add("dib_bgr", "bmp", ".dib", lambda: cv_bytes(".bmp", BGR8))
    bits = (GREY8 > 127).astype(np.uint8)
    for name, ext, make in (
        ("p1", ".pbm", lambda: pnm("P1", bits, 1)),
        ("p4", ".pbm", lambda: pnm("P4", bits, 1)),
        ("p2_max15", ".pgm", lambda: pnm("P2", GREY8 >> 4, 15)),
        ("p5_max1", ".pgm", lambda: pnm("P5", bits, 1)),
        ("p5_max255", ".pgm", lambda: pnm("P5", GREY8, 255)),
        ("p5_max1023", ".pgm", lambda: pnm("P5", GREY16 >> 6, 1023)),
        ("p5_max65535", ".pnm", lambda: pnm("P5", GREY16, 65535)),
        ("p3_max255", ".ppm", lambda: pnm("P3", RGBA8[..., :3], 255)),
        ("p6_max15", ".ppm", lambda: pnm("P6", RGBA8[..., :3] >> 4, 15)),
        ("p6_max65535", ".ppm", lambda: pnm("P6", RGBA16[..., :3], 65535)),
        ("p6_pxm", ".pxm", lambda: pnm("P6", RGBA8[..., :3], 255)),
        ("pfm_pxm", ".pxm", lambda: cv_bytes(".pfm", BGR8.astype(np.float32) / 99)),
        ("pam_pnm", ".pnm", lambda: cv_bytes(".pam", BGRA8)),
        ("p6_truncated", ".ppm", lambda: pnm("P6", RGBA8[..., :3], 255)[:-30]),
    ):
        add(f"pxm_{name}", "pxm", ext, make)
    hdr = BGR8.astype(np.float32) / 64
    for name, ext, make in (
        ("sr_grey", ".sr", lambda: cv_bytes(".sr", GREY8)),
        ("ras_bgr", ".ras", lambda: cv_bytes(".ras", BGR8)),
        ("hdr", ".hdr", lambda: cv_bytes(".hdr", hdr)),
        ("pic", ".pic", lambda: cv_bytes(".hdr", hdr)),
    ):
        add(f"sun_hdr_{name}", "sunras-hdr", ext, make)
    for name, make in {
        "rgb": lambda: pil_bytes(rgb, "JPEG2000"),
        "rgba": lambda: pil_bytes(rgba, "JPEG2000"),
        "i16": lambda: pil_bytes(Image.fromarray(GREY16), "JPEG2000"),
    }.items():
        add(f"jp2_{name}", "jp2", ".jp2", make)
    add("exr_junk", "exr", ".exr", lambda: b"\x76\x2f\x31\x01" + bytes(60))
    add("dds_rgba", "dds", ".dds", lambda: pil_bytes(rgba, "DDS"))
    for name, ext, make in (
        ("tga_rgb", ".tga", lambda: pil_bytes(rgb, "TGA")),
        ("tga_rgba_rle", ".tga", lambda: pil_bytes(rgba, "TGA", compression="tga_rle")),
        ("ico", ".ico", lambda: pil_bytes(rgba.resize((16, 16)), "ICO")),
        ("pcx", ".pcx", lambda: pil_bytes(rgb, "PCX")),
        ("sgi", ".sgi", lambda: pil_bytes(rgb, "SGI")),
        ("msp", ".msp", lambda: pil_bytes(grey.convert("1"), "MSP")),
        ("xbm", ".xbm", lambda: pil_bytes(grey.convert("1"), "XBM")),
        ("icns", ".icns", lambda: pil_bytes(rgba.resize((32, 32)), "ICNS")),
        ("avif", ".avif", lambda: pil_bytes(rgb, "AVIF", quality=90)),
        ("jpx", ".jpx", lambda: pil_bytes(rgb, "JPEG2000")),
    ):
        add(f"pillow_{name}", "pillow-only", ext, make)
    for name, row, ext, make in (
        ("jpeg_as_png", "png", ".png", lambda: base),
        ("png_as_jpg", "jpeg-pillow", ".jpg", lambda: cv_bytes(".png", BGRA8)),
        ("webp_as_png", "png", ".png", lambda: cv_bytes(".webp", BGR8)),
        ("tiff_as_jpg", "jpeg-pillow", ".jpg", lambda: cv_bytes(".tif", BGR8)),
        ("bmp_as_jpe", "jpeg-opencv", ".jpe", lambda: cv_bytes(".bmp", BGR8)),
        ("gif_as_png", "png", ".png", lambda: pil_bytes(rgb, "GIF")),
        ("gif", "unsupported-ext", ".gif", lambda: pil_bytes(rgb, "GIF")),
        ("empty_jpg", "jpeg-pillow", ".jpg", lambda: b""),
        ("empty_png", "png", ".png", lambda: b""),
        ("empty_webp", "webp", ".webp", lambda: b""),
        ("p5_12_bytes_as_png", "png", ".png", lambda: b"P5\n1 1\n255\n\x80"),
        ("jpeg_prefix_jpe", "jpeg-opencv", ".jpe", lambda: base[:20]),
    ):
        add(f"mismatch_{name}", row, ext, make, foreign=True)
    return cases


DECODE_CASES = decode_cases()


@pytest.fixture(scope="module")
def decode_files(tmp_path_factory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("decode")
    files = {}
    for case in DECODE_CASES:
        files[case.name] = folder / f"{case.name}{case.ext}"
        files[case.name].write_bytes(case.data())
    return files


def preview_pixels(data: dict) -> dict:
    """Broadcast data with each PNG preview's URL as its decoded pixels: chaiNNer-C writes today's PNG
    preview pixels with its own encoder (image_preview.cpp), in other bytes. JPEG previews keep their bytes."""
    previews = []
    for preview in data["previews"]:
        url = str(preview["url"])
        if url.startswith("data:image/png;base64,"):
            encoded = np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8)
            pixels = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            assert pixels is not None, "a PNG preview that does not decode"
            url = f"png:{pixels.dtype.str}:{list(pixels.shape)}:{sha(pixels)}"
        previews.append({**preview, "url": url})
    return {**data, "previews": previews}


def decoded(path: Path) -> dict:
    """Load Image's result for path (raw array, normalised output, previews or the exception) and the
    warnings it logged; PNG previews as their pixels (preview_pixels)."""
    with logged() as warnings:
        try:
            raw = load_image.load_image_node(path)[0]
        except Exception as error:
            result = {"error": type(error).__qualname__, "message": str(error)}
        else:
            output = LargeImageOutput()
            norm = output.enforce(raw)
            previews = json.dumps(
                preview_pixels(output.get_broadcast_data(norm)), sort_keys=True
            )
            result = {
                "dtype": raw.dtype.str,
                "shape": raw.shape,
                "raw": sha(raw),
                "norm": sha(norm),
                "previews": sha(previews.encode()),
            }
    return {**result, "log": warnings}


@pytest.mark.parametrize("name", [case.name for case in DECODE_CASES])
def test_decode_matches_todays_path(
    name: str, decode_files: dict[str, Path], monkeypatch: pytest.MonkeyPatch
):
    case = next(c for c in DECODE_CASES if c.name == name)
    path = decode_files[name]
    monkeypatch.setenv("CHAINNER_C_NATIVE_CODECS", "0")
    expected = decoded(path)
    monkeypatch.delenv("CHAINNER_C_NATIVE_CODECS")
    assert decoded(path) == expected
    if (
        case.row in NATIVE_DECODE
        and not case.foreign
        and "raw" in expected
        and not expected["log"]
        and name not in KEPT_ON_TODAYS_PATH
    ):
        monkeypatch.setattr(load_image, "Image", Probe(load_image.Image, {"open"}))
        monkeypatch.setattr(
            load_image, "cv2", Probe(load_image.cv2, {"imdecode", "imread", "cvtColor"})
        )
        native = getattr(load_image, NATIVE_DECODE[case.row])(path)
        assert (
            native is not None
            and sha(native) == expected["raw"]
            and native.shape == expected["shape"]
        )


@pytest.mark.parametrize(
    "compression", (1, 5, 8), ids=("none", "lzw", "deflate")
)
@pytest.mark.parametrize(("width", "height"), ((1101, 1003), (3, 2)))
def test_plain_rgb_tiff_matches_opencv(
    width: int,
    height: int,
    compression: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Strips read straight into the output give what OpenCV's RGBA interface gives."""
    rng = np.random.default_rng(width)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    pixels[height // 2 :] = pixels[height // 2 :] // 16
    path = tmp_path / "plain.tif"
    path.write_bytes(
        cv_bytes(".tif", pixels, [cv2.IMWRITE_TIFF_COMPRESSION, compression])
    )
    monkeypatch.setattr(
        load_image, "cv2", Probe(load_image.cv2, {"imdecode", "imread", "cvtColor"})
    )
    native = load_image._read_cv(path)
    assert native is not None and np.array_equal(native, pixels)


@pytest.mark.parametrize(
    ("mode", "order"),
    (("RGB", [2, 1, 0]), ("RGBA", [2, 1, 0, 3])),
    ids=("rgb", "rgba"),
)
@pytest.mark.parametrize(("width", "height"), ((1101, 1003), (3, 2)))
def test_pillow_route_swaps_red_and_blue(
    width: int, height: int, mode: str, order: list[int], tmp_path: Path
):
    channels = len(order)
    rng = np.random.default_rng(height)
    pixels = rng.integers(0, 256, (height, width, channels), dtype=np.uint8)
    path = tmp_path / "pillow.tga"
    Image.fromarray(pixels, mode).save(path)
    loaded = load_image._read_pil(path)
    assert loaded.dtype == np.uint8 and np.array_equal(loaded, pixels[..., order])


@pytest.mark.parametrize(
    "name",
    [
        "inflate_level6",
        "inflate_level9",
        "inflate_block_types",
        "inflate_final_stored",
        "inflate_far_matches",
        "inflate_lasting_markers",
        "inflate_zeros_at_end",
        "inflate_run_at_start",
    ],
)
def test_png_stream_inflates_on_the_pool(name: str, decode_files: dict[str, Path]):
    """A taken stream dispatches four calls (search, decode, place, free) that a small PNG does not. Not the stored
    runs: whether a search window finds a block among them depends on the thread count."""
    dll = lib()
    dll.cn_parallel_capacity.restype = ctypes.c_uint
    dll.cn_parallel_dispatches.restype = ctypes.c_uint64
    if dll.cn_parallel_capacity() < 4:
        pytest.skip("the parallel inflate needs 4 logical CPUs")

    def dispatches(path: Path) -> int:
        before = dll.cn_parallel_dispatches()
        assert load_image._read_cv(path) is not None
        return dll.cn_parallel_dispatches() - before

    assert (
        dispatches(decode_files[f"png_{name}"]) - dispatches(decode_files["png_bgr8"])
        >= 4
    )


# ---------------------------------------------------------------------------------------------
# The direct path's row filters against libpng's: the same filtered rows deflated with a 512-byte window leave the file
# to libpng (png_codec.c png_decode_direct). Every filter type directly under every other, at every pixel size, on
# widths around the AVX2 unit's loads, then files above the size of the wave across the rows: narrow blocks, wide short
# rows, more rows than bands of 16 hold, and an invalid filter type part way down.
PIXEL_SIZES = {
    1: (0, 1, 8),
    2: (0, 1, 16),
    3: (2, 3, 8),
    4: (6, 4, 8),
    6: (2, 3, 16),
    8: (6, 4, 16),
}
EVERY_PAIR = [
    kind for above in range(5) for below in range(5) for kind in (above, below)
]


def unfilter_cases() -> Iterator[tuple[int, list[int], int, list[int] | None]]:
    """(bytes per pixel, the rows' filter types, width, the rows' values)."""
    for pixel in PIXEL_SIZES:
        for width in range(1, 71):
            yield pixel, EVERY_PAIR, width, None
            yield pixel, EVERY_PAIR, width, [0, 1, 127, 128, 254, 255]
        yield pixel, EVERY_PAIR * 6, 1100 // pixel, None
        yield pixel, EVERY_PAIR * 2, 4096, None
    yield 3, [(y // 16) % 5 for y in range(5000)], 300, None
    yield 3, [7 if y == 150 else y % 5 for y in range(300)], 400, None


@pytest.fixture(scope="module")
def unfilter_files(tmp_path_factory) -> list[tuple[Path, Path]]:
    directory = tmp_path_factory.mktemp("unfilter")
    pairs = []
    for index, (pixel, filters, width, values) in enumerate(unfilter_cases()):
        color_type, channels, depth = PIXEL_SIZES[pixel]
        pair = tuple(directory / f"{index}_{wbits}.png" for wbits in (15, 9))
        for path, wbits in zip(pair, (15, 9), strict=True):
            path.write_bytes(
                png_filtered(
                    color_type,
                    channels,
                    filters,
                    wbits=wbits,
                    depth=depth,
                    width=width,
                    values=values,
                )
            )
        pairs.append(pair)
    return pairs


def read_outcome(path: Path) -> tuple:
    try:
        image = load_image._read_cv(path)
    except Exception as error:
        return (type(error).__qualname__,)
    return image.shape, sha(image)


@pytest.mark.parametrize("level", (0, 1, 2), ids=("scalar", "avx2", "avx512"))
def test_direct_unfilter_matches_libpng(
    level: int, unfilter_files: list[tuple[Path, Path]]
):
    state = (ctypes.c_int * 3)()
    assert native.lib()["cn_isa_get"](state) == 0
    isa_set = native.lib()["cn_isa_set"]
    isa_set.argtypes = [ctypes.c_int]
    if level > state[2]:
        pytest.skip("this CPU lacks the level")
    dispatches = native.lib()["cn_parallel_dispatches"]
    dispatches.restype = ctypes.c_uint64
    before = dispatches()
    isa_set(level)
    try:
        for direct, through_libpng in unfilter_files:
            assert read_outcome(direct) == read_outcome(through_libpng), direct.name
    finally:
        isa_set(state[0])
    assert dispatches() > before or native.lib()["cn_parallel_capacity"]() == 1


# ---------------------------------------------------------------------------------------------
# Encode


@dataclass(frozen=True)
class EncodeCase:
    fmt: str
    channels: int
    size: tuple[int, int] = (13, 9)
    depth: str = "u8"
    quality: int = 95
    subsampling: str = "422"
    progressive: bool = False
    lossless: bool = False
    compression: str = "lzw"
    special: bool = False

    @property
    def row(self) -> str:
        if self.fmt == "png":
            return "png-u16" if self.depth == "u16" else "png-u8"
        if self.fmt == "webp":
            return (
                "webp-lossy"
                if not self.lossless
                else "webp-lossless-4ch"
                if self.channels == 4
                else "webp-lossless-1-3ch"
            )
        if self.fmt == "tiff":
            return "tiff-f32" if self.depth == "f32" else "tiff-u8-u16"
        return "pillow-gif-tga-avif" if self.fmt in ("gif", "tga", "avif") else self.fmt

    @property
    def name(self) -> str:
        w, h = self.size
        options = f"{self.depth}-q{self.quality}-s{self.subsampling}-p{int(self.progressive)}-l{int(self.lossless)}-{self.compression}"
        return f"{self.fmt}-c{self.channels}-{w}x{h}-{options}{'-special' if self.special else ''}"


def encode_cases() -> list[EncodeCase]:
    cases = [
        EncodeCase("jpg", 3, quality=q, subsampling=s, progressive=p)
        for q in (0, 50, 100)
        for s in ("444", "440", "422", "420")
        for p in (False, True)
    ]
    cases += [
        EncodeCase("jpg", c, quality=95, subsampling=s, progressive=p)
        for c in (1, 4)
        for s, p in (("422", False), ("420", True))
    ]
    cases += [
        EncodeCase("webp", c, quality=q) for q in (0, 1, 50, 100) for c in (1, 3, 4)
    ]
    cases += [EncodeCase("webp", c, lossless=True) for c in (1, 3, 4)]
    cases += [EncodeCase("png", c, depth=d) for d in ("u8", "u16") for c in (1, 3, 4)]
    cases += [
        EncodeCase("png", 3, size=(70000, 1)),
        EncodeCase("png", 4, size=(65535, 1)),
    ]
    cases += [
        EncodeCase("tiff", c, depth=d, compression=z)
        for d in ("u8", "u16")
        for z in ("none", "lzw", "zip")
        for c in (1, 3, 4)
    ]
    cases += [EncodeCase("tiff", c, depth="f32") for c in (1, 3, 4)]
    cases += [
        EncodeCase(fmt, c) for fmt in ("bmp", "gif", "tga", "avif") for c in (1, 3, 4)
    ]
    cases += [
        EncodeCase("jpg", 3, special=True),
        EncodeCase("png", 3, special=True),
        EncodeCase("png", 4, depth="u16", special=True),
        EncodeCase("webp", 4, quality=80, special=True),
        EncodeCase("webp", 3, lossless=True, special=True),
        EncodeCase("tiff", 3, depth="f32", special=True),
        EncodeCase("tiff", 1, depth="u16", compression="none", special=True),
        EncodeCase("bmp", 4, special=True),
    ]
    return cases


ENCODE_CASES = encode_cases()
SPECIAL = np.array([np.nan, np.inf, -np.inf, -0.0, 1.5, -0.5, 2.0, 1e-45], np.float32)


def image_for(case: EncodeCase) -> np.ndarray:
    """A float32 image of the case's size: [0, 1] with every k/255 and midpoint (k + 0.5)/255 in turn,
    k/65535 steps, 0, -0 and 1; the special values first for a special case."""
    w, h = case.size
    rng = np.random.default_rng([w, h, case.channels])
    k = np.arange(256, dtype=np.float64)
    values = np.concatenate(
        [
            k / 255,
            (k[:-1] + 0.5) / 255,
            rng.integers(0, 65536, 200) / 65535,
            [0.0, -0.0, 1.0],
        ]
    ).astype(np.float32)
    flat = rng.random(w * h * case.channels, dtype=np.float32)
    flat[: min(values.size, flat.size)] = values[rng.permutation(values.size)][
        : flat.size
    ]
    if case.special:
        flat[: SPECIAL.size] = SPECIAL
    image = flat.reshape(h, w, case.channels)
    return image[:, :, 0] if case.channels == 1 else image


def saved(case: EncodeCase, image: np.ndarray, directory: Path) -> dict:
    """Save Image's file bytes and prepared bytes for the case, or the exception."""
    m = save_image
    fmt = m.ImageFormat(case.fmt)
    png = m.PngColorDepth.U16 if case.depth == "u16" else m.PngColorDepth.U8
    sampling = m.JpegSubsampling[f"FACTOR_{case.subsampling}"]
    tiff = m.TiffColorDepth(case.depth)
    compression = m.TiffCompression[case.compression.upper()]
    result: dict = {}
    try:
        m.save_image_node(
            Lazy.ready(image),
            directory,
            None,
            "image",
            fmt,
            png,
            case.lossless,
            case.quality,
            sampling,
            case.progressive,
            tiff,
            compression,
            "BC1_UNORM",
            m.BC7Compression.DEFAULT,
            m.DDSErrorMetric.PERCEPTUAL,
            False,
            0,
            0,
            m.DDSMipFilter.DEFAULT,
            False,
            m.DDSColourConversion.KEEP,
            False,
            False,
            m.AvifSubsampling.FACTOR_420,
            m.ExrColorDepth.F32,
            m.ExrCompression.ZIP,
            False,
            None,
        )
        result["file"] = sha((directory / f"image.{case.fmt}").read_bytes())
    except Exception as error:
        result["error"] = (type(error).__qualname__, str(error))
    inputs: list[object] = [
        image,
        directory,
        None,
        "image",
        fmt,
        png,
        case.lossless,
        case.quality,
        sampling,
        case.progressive,
        tiff,
        compression,
        "BC1_UNORM",
        m.BC7Compression.DEFAULT,
        m.DDSErrorMetric.PERCEPTUAL,
        False,
        0,
        0,
        m.DDSMipFilter.DEFAULT,
        False,
        m.DDSColourConversion.KEEP,
        False,
        False,
        m.AvifSubsampling.FACTOR_420,
        m.ExrColorDepth.F32,
        m.ExrCompression.ZIP,
        False,
        None,
    ]
    try:
        prepared = m.prepare_save_image(inputs)
        result["prepared"] = None if prepared is None else sha(bytes(prepared))
    except Exception as error:
        result["prepared_error"] = (type(error).__qualname__, str(error))
    return result


@pytest.mark.parametrize("case", ENCODE_CASES, ids=lambda case: case.name)
def test_encode_matches_todays_path(
    case: EncodeCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image = image_for(case)
    monkeypatch.setenv("CHAINNER_C_NATIVE_CODECS", "0")
    expected = saved(case, image, tmp_path / "oracle")
    monkeypatch.delenv("CHAINNER_C_NATIVE_CODECS")
    assert saved(case, image, tmp_path / "native") == expected
    if case.row in NATIVE_ENCODE:
        assert "file" in expected
        monkeypatch.setattr(
            save_image, "cv2", Probe(save_image.cv2, {"imencode", "imwrite"})
        )
        monkeypatch.setattr(
            native_image_io, "cv2", Probe(native_image_io.cv2, {"imencode", "imwrite"})
        )
        monkeypatch.setattr(save_image, "Image", Probe(save_image.Image, {"fromarray"}))
        assert saved(case, image, tmp_path / "probe") == expected


# ---------------------------------------------------------------------------------------------
# Names with a lone surrogate (NTFS takes unpaired UTF-16, Python keeps it in the str; a
# POSIX name's undecodable byte is the surrogate that escapes it)
LONE_SURROGATE = chr(0xDC00) if sys.platform == "win32" else chr(0xDCFF)


@pytest.mark.parametrize("ext", [".png", ".jpe", ".jpg"])
def test_decode_lone_surrogate_name(
    ext: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image = (np.arange(16 * 16 * 3) % 251).astype(np.uint16).reshape(16, 16, 3) * 200
    if ext != ".png":
        image = (image >> 8).astype(np.uint8)
    path = tmp_path / f"a{LONE_SURROGATE}b{ext}"
    path.write_bytes(cv2.imencode(".png" if ext == ".png" else ".jpg", image)[1])
    monkeypatch.setenv("CHAINNER_C_NATIVE_CODECS", "0")
    expected = decoded(path)
    monkeypatch.delenv("CHAINNER_C_NATIVE_CODECS")
    assert "raw" in expected and decoded(path) == expected
    monkeypatch.setattr(load_image, "Image", Probe(load_image.Image, {"open"}))
    monkeypatch.setattr(
        load_image, "cv2", Probe(load_image.cv2, {"imdecode", "imread", "cvtColor"})
    )
    native = (load_image._read_pil if ext == ".jpg" else load_image._read_cv)(path)
    assert native is not None and sha(native) == expected["raw"]


def test_save_lone_surrogate_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case = EncodeCase("png", 3, depth="u16")
    image = image_for(case)
    folder = tmp_path / f"a{LONE_SURROGATE}b"
    monkeypatch.setenv("CHAINNER_C_NATIVE_CODECS", "0")
    expected = saved(case, image, folder / "oracle")
    monkeypatch.delenv("CHAINNER_C_NATIVE_CODECS")
    assert "file" in expected and saved(case, image, folder / "native") == expected


# ---------------------------------------------------------------------------------------------
# What today's OpenCV path does that the native one would not


@pytest.mark.parametrize(
    ("name", "value", "native"),
    [
        ("OPENCV_IO_MAX_IMAGE_WIDTH", "1024", True),
        ("OPENCV_IO_MAX_IMAGE_WIDTH", str(BGR8.shape[1] - 1), False),
        ("OPENCV_IO_MAX_IMAGE_HEIGHT", str(BGR8.shape[0]), True),
        ("OPENCV_IO_MAX_IMAGE_HEIGHT", str(BGR8.shape[0] - 1), False),
        ("OPENCV_IO_MAX_IMAGE_PIXELS", "1600000000", True),
        ("OPENCV_IO_MAX_IMAGE_PIXELS", str(BGR8.shape[0] * BGR8.shape[1] - 1), False),
        ("OPENCV_IO_MAX_IMAGE_PIXELS", "01048576", True),
        ("OPENCV_IO_MAX_IMAGE_PIXELS", "0", False),
        # Anything but a plain integer: declined (OpenCV's reading is not reproduced).
        ("OPENCV_IO_MAX_IMAGE_PIXELS", "1MB", False),
        ("OPENCV_IO_MAX_IMAGE_PIXELS", " 5", False),
        ("OPENCV_IO_MAX_IMAGE_PIXELS", "5GB", False),
        ("OPENCV_IO_MAX_IMAGE_PIXELS", "99999999999999999999", False),
    ],
)
def test_opencv_size_limit_settings(
    name: str, value: str, native: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The native OpenCV route takes OpenCV's size limits from the environment when
    they are plain integers; any other value sends every file to OpenCV."""
    path = tmp_path / "image.png"
    path.write_bytes(cv2.imencode(".png", BGR8)[1])
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        load_image, "cv2", Probe(load_image.cv2, {"imdecode", "imread"})
    )
    if native:
        image = load_image._read_cv(path)
        assert image is not None
        assert image.shape == BGR8.shape
    else:
        with pytest.raises(RuntimeError, match="Error reading image"):
            load_image._read_cv(path)


@pytest.mark.parametrize("ext", [".hdr", ".pic", ".sr", ".ras", ".jp2"])
def test_unported_opencv_extension_is_not_read_natively(
    ext: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Formats without a native codec go straight to OpenCV (imdecode still takes a PNG by
    its signature), so the file is read once."""
    path = tmp_path / f"image{ext}"
    path.write_bytes(cv2.imencode(".png", BGR8)[1])
    monkeypatch.setattr(
        load_image, "cv2", Probe(load_image.cv2, {"imdecode", "imread"})
    )
    with pytest.raises(RuntimeError, match="Error reading image"):
        load_image._read_cv(path)


# ---------------------------------------------------------------------------------------------
# The image-size limit: 40,000 x 40,000 pixels on every route. Headers only:
# the native decoders' requests are refused (image_io_decode_request), so nothing is allocated.


def png_header(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(64)))
        + chunk(b"IEND", b"")
    )


def bmp_header(width: int, height: int) -> bytes:
    """An 8-bit RLE BMP (grey palette) that ends at once: its data is a few bytes."""
    palette = bytes(1024)
    info = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 8, 1, 2, 0, 0, 0, 0)
    offset = 14 + len(info) + len(palette)
    return (
        b"BM"
        + struct.pack("<IHHI", offset + 2, 0, 0, offset)
        + info
        + palette
        + b"\x00\x01"
    )


def jpeg_header(width: int, height: int) -> bytes:
    jpeg = cv_bytes(".jpg", GREY8)
    at = next(i for i in range(2, len(jpeg)) if jpeg[i : i + 2] == b"\xff\xc0")
    return jpeg[: at + 5] + struct.pack(">HH", height, width) + jpeg[at + 9 :]


def alpha_tiff_header(width: int, height: int) -> bytes:
    short, long = 3, 4
    entries = [
        (256, long, width),
        (257, long, height),
        (258, short, 8),
        (259, short, 5),
        (262, short, 2),
        (273, long, 8),
        (277, short, 4),
        (278, long, height),
        (279, long, 1),
        (338, short, 2),
    ]
    # Little-endian: a SHORT value is the low half of the 4-byte value field.
    ifd = struct.pack("<H", len(entries)) + b"".join(
        struct.pack("<HHII", tag, kind, 1, value) for tag, kind, value in entries
    )
    return b"II*\0" + struct.pack("<I", 8) + ifd + bytes(4)


SIZE_GATES = {
    "pxm": (lambda w, h: f"P2 {w} {h} 255\n0 0 0 0\n".encode().ljust(32), "cv"),
    "bmp": (bmp_header, "cv"),
    "png": (png_header, "cv"),
    "jpeg-opencv": (jpeg_header, "cv"),
    "jpeg-pillow": (jpeg_header, "pil-jpeg"),
    "tiff-alpha": (alpha_tiff_header, "pil-tiff-alpha"),
}


@pytest.mark.parametrize("route", SIZE_GATES)
def test_native_size_gates_take_40000_square(
    route: str, monkeypatch: pytest.MonkeyPatch
):
    """Each native decoder asks for the buffer of a 40,000 x 40,000 image and declines
    one pixel row or column more, under the worker's limits (the engine route uses the
    same decoders and image_io_max_pixels)."""
    import image_limits
    from nodes.impl.native_graph import graph

    make, decoder = SIZE_GATES[route]
    limit = graph().image_io_max_pixels
    assert limit == image_limits.MAX_IMAGE_PIXELS == 40_000 * 40_000
    monkeypatch.setenv("OPENCV_IO_MAX_IMAGE_PIXELS", str(limit))
    request = graph().image_io_decode_request
    assert request(make(40_000, 40_000), decoder, limit)[:2] == (40_000, 40_000)
    assert request(make(40_001, 40_000), decoder, limit) is None
    assert request(make(40_000, 40_001), decoder, limit) is None


# The worker's imports in their order: image_limits, then cv2 and Pillow.
WORKER_GATE = r"""
import sys
import warnings

import image_limits

image_limits.apply()
import cv2
import numpy as np

Image = image_limits.pil_image()

width, height = int(sys.argv[1]), int(sys.argv[2])
with warnings.catch_warnings(record=True) as caught:
    try:
        Image._decompression_bomb_check((width, height))
        print("pillow-taken", len(caught))
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        print("pillow-refused", len(caught))
data = np.frombuffer(f"P5 {width} {height} 255\n".encode() + bytes(32), np.uint8)
try:
    cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    print("opencv-taken")
except cv2.error as error:
    print("opencv-refused" if "CV_IO_MAX_IMAGE_PIXELS" in str(error) else "opencv-error")
"""


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        (40_000, ["pillow-taken", "0", "opencv-taken"]),
        (40_001, ["pillow-refused", "0", "opencv-refused"]),
    ],
)
def test_python_route_in_a_worker_takes_40000_square(width: int, expected: list):
    """Pillow takes 40,000 x 40,000 without a warning and refuses one pixel column more;
    OpenCV's size check does the same (the header-only file then fails at its data)."""
    env = {k: v for k, v in os.environ.items() if k != "OPENCV_IO_MAX_IMAGE_PIXELS"}
    done = subprocess.run(
        [sys.executable, "-B", "-c", WORKER_GATE, str(width), "40000"],
        cwd=Path(__file__).resolve().parents[2] / "backend" / "src",
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == expected, done.stdout + done.stderr
