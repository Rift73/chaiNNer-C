"""The tags Load Image's Metadata output carries to Save Image: colour profile, EXIF and
DPI. The output's value is the file's path, whose tags are read when Save Image writes
them, so a chain that does not connect it reads nothing and a chain that overwrites the
file first writes what it holds by then (a node's body cannot tell that its output is
connected); or the ImageTags read at load, when Convert To sRGB changed what the profile
describes. Save Image's encoders write no tags: they go into the encoded file's bytes,
whose pixels stay as encoded. Tags that fail, to read or to write, warn and stay out."""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from pathlib import Path
from typing import BinaryIO, Iterator, Union

import numpy as np
from PIL import UnidentifiedImageError
from PIL.Image import Exif
from PIL.PngImagePlugin import MAX_TEXT_CHUNK
from sanic.log import logger

import image_limits
from nodes.utils.utils import get_h_w_c

from .native_graph import graph

_EXIF_PREFIX = b"Exif\x00\x00"
_TIFF_SIGNATURES = (b"II*\x00", b"MM\x00*", b"II+\x00")
_ORIENTATION = 0x0112
_EXIF_IFD, _GPS_IFD = 0x8769, 0x8825
# The sizes EXIF states, in IFD 0 and in its Exif IFD (PixelXDimension, PixelYDimension).
_WIDTH_TAGS, _HEIGHT_TAGS = (256, 0xA002), (257, 0xA003)
_ICC_TAG = 34675
_X_RESOLUTION, _Y_RESOLUTION, _RESOLUTION_UNIT = 282, 283, 296
# The IFD 0 tags that describe a TIFF's picture's origin and not how its samples are
# stored: description, make, model, software, date, artist, copyright, Exif and GPS IFDs.
_EXIF_FILE_TAGS = frozenset({270, 271, 272, 305, 306, 315, 33432, 34665, 34853})
_INT32_MAX, _UINT32_MAX = (1 << 31) - 1, (1 << 32) - 1
_JPEG_SEGMENT = 65533
_JPEG_ICC_CHUNK = _JPEG_SEGMENT - len(b"ICC_PROFILE\x00") - 2


@dataclass(frozen=True)
class ImageTags:
    icc_profile: bytes | None = None
    exif: bytes | None = None
    """A TIFF structure, without JPEG's Exif prefix."""
    dpi: tuple[float, float] | None = None


ImageMetadata = Union[Path, ImageTags]


def _exif_entries(
    exif: bytes, endian: str, offset: int
) -> Iterator[tuple[int, int, int]]:
    """Each entry of the directory at `offset` that the block holds: its place, tag, type."""
    if offset + 2 > len(exif):
        return
    count = struct.unpack_from(f"{endian}H", exif, offset)[0]
    end = min(offset + 2 + 12 * count, len(exif) - 11)
    for at in range(offset + 2, end, 12):
        tag, kind = struct.unpack_from(f"{endian}HH", exif, at)
        yield at, tag, kind


def _set_exif_values(exif: bytes, values: dict[int, int], exif_ifd: bool) -> bytes:
    """The EXIF block with these SHORT or LONG values of IFD 0, or of its Exif IFD, set in
    place: the rest of it, a maker's notes too, stays byte for byte. A value its tag's
    type cannot hold stays as it was."""
    out = bytearray(exif)
    endian = "<" if exif[:2] == b"II" else ">"

    def write(offset: int) -> None:
        for at, tag, kind in _exif_entries(exif, endian, offset):
            value = values.get(tag)
            if value is not None and (kind == 4 or (kind == 3 and value < 1 << 16)):
                code = "H" if kind == 3 else "I"
                struct.pack_into(f"{endian}{code}", out, at + 8, value)

    first = struct.unpack_from(f"{endian}I", exif, 4)[0]
    write(first)
    if exif_ifd:
        for at, tag, _ in _exif_entries(exif, endian, first):
            if tag == _EXIF_IFD:
                write(struct.unpack_from(f"{endian}I", exif, at + 8)[0])
    return bytes(out)


def _upright_exif(exif: bytes) -> bytes | None:
    """The file's EXIF with Orientation 1, as Load Image turns the pixels upright; None
    for bytes that are no TIFF structure."""
    exif = exif.removeprefix(_EXIF_PREFIX)
    if exif[:4] not in (b"II*\x00", b"MM\x00*") or len(exif) < 8:
        logger.warning("Ignored EXIF data that is not a TIFF structure.")
        return None
    return _set_exif_values(exif, {_ORIENTATION: 1}, False)


def _without_orientation(exif: bytes) -> bytes:
    """The block without its Orientation entry, all else where it was. Pillow's AVIF takes
    that tag for its container and then writes a block that has it anew, which loses the
    thumbnail, turns text into ASCII and fails on tags it cannot write."""
    endian = "<" if exif[:2] == b"II" else ">"
    first = struct.unpack_from(f"{endian}I", exif, 4)[0]
    for at, tag, _ in _exif_entries(exif, endian, first):
        if tag == _ORIENTATION:
            count = struct.unpack_from(f"{endian}H", exif, first)[0]
            end = first + 2 + 12 * count + 4  # the directory and its next pointer
            if end > len(exif):
                return exif
            out = bytearray(exif)
            struct.pack_into(f"{endian}H", out, first, count - 1)
            out[at:end] = exif[at + 12 : end] + bytes(12)
            return bytes(out)
    return exif


def _only_file_tags(exif: Exif) -> Exif:
    for tag in set(exif) - _EXIF_FILE_TAGS:
        del exif[tag]
    return exif


class _ReadText(str):
    """Text as Pillow read it, as latin-1. Pillow writes a str as ASCII, '?' for the rest,
    and bytes under another tag type, so the str stays and its encoding changes."""

    def encode(self, encoding: str = "latin-1", errors: str = "replace") -> bytes:
        return str.encode(self, "latin-1", errors)


def _write_exif(exif: Exif, offset: int = 8) -> bytes | None:
    """Exif.tobytes with text as it was; None, with a warning, for tags it cannot write (a
    date stored as a number)."""
    try:
        pointers = [ifd for ifd in (_EXIF_IFD, _GPS_IFD) if ifd in exif]
        for values in (exif, *(exif.get_ifd(ifd) for ifd in pointers)):
            for tag, value in list(values.items()):
                if isinstance(value, str) and not value.isascii():
                    values[tag] = _ReadText(value)
        return exif.tobytes(offset)
    except Exception:
        logger.warning("Left out EXIF data that cannot be written.", exc_info=True)
        return None


def _dpi(x: object, y: object, scale: float = 1.0) -> tuple[float, float] | None:
    if not (isinstance(x, Real) and isinstance(y, Real)):
        return None
    dpi = float(x) * scale, float(y) * scale
    return dpi if all(math.isfinite(value) and value > 0 for value in dpi) else None


def _per_metre(dpi: tuple[float, float]) -> list[int]:
    """DPI as PNG and BMP hold it: whole pixels per metre, from 1 to a signed 32-bit's top."""
    return [min(max(int(value / 0.0254 + 0.5), 1), _INT32_MAX) for value in dpi]


def _rational(value: float, endian: str) -> bytes:
    """A TIFF RATIONAL above 0, within its two 32-bit fields."""
    ratio = Fraction(value).limit_denominator(10000)
    if ratio.numerator > _UINT32_MAX:
        return struct.pack(f"{endian}II", _UINT32_MAX, 1)
    return struct.pack(f"{endian}II", max(ratio.numerator, 1), ratio.denominator)


def _read_tiff_tags(file: BinaryIO, head: bytes) -> ImageTags:
    """The tags of a TIFF from its directory: Pillow opens no float or planar RGB one."""
    big = head[2] == 43
    endian = "<" if head[:2] == b"II" else ">"
    offset = struct.unpack_from(
        f"{endian}Q" if big else f"{endian}I", head, 8 if big else 4
    )
    exif = Exif()
    exif.bigtiff, exif.endian = big, endian
    exif.load_from_fp(file, offset[0])
    icc = exif.get(_ICC_TAG)
    unit = exif.get(_RESOLUTION_UNIT, 2)
    # 2: per inch (the default), 3: per centimetre; others say no absolute size.
    dpi = (
        _dpi(
            exif.get(_X_RESOLUTION),
            exif.get(_Y_RESOLUTION),
            2.54 if unit == 3 else 1.0,
        )
        if unit in (2, 3)
        else None
    )
    exif = _only_file_tags(exif)
    exif.bigtiff = False
    written = _write_exif(exif) if len(exif) else None
    return ImageTags(
        icc if isinstance(icc, bytes) and icc else None,
        _upright_exif(written) if written else None,
        dpi,
    )


def read_image_tags(path: Path) -> ImageTags:
    """A file's tags. A format Pillow does not know (OpenEXR, Radiance HDR, ...) has none,
    and tags that cannot be read are none with a warning: they stop no load and no save."""
    try:
        with Path(path).open("rb") as file:
            head = file.read(16)
            if head[:4] == b"MM\x00+":
                # Pillow's directory reader takes this byte order's BigTIFF for a classic one.
                logger.warning(f"The tags of a big-endian BigTIFF are not read: {path}")
                return ImageTags()
            if head[:4] in _TIFF_SIGNATURES:
                return _read_tiff_tags(file, head)
        with image_limits.pil_image().open(path) as im:
            info = im.info
            exif = info.get("exif")
            if exif is None and im.format == "PNG":
                # Pillow reads the chunks after IDAT only as it decodes the pixels.
                exif = graph().image_io_exif_block(Path(path).read_bytes())
            return ImageTags(
                info.get("icc_profile") or None,
                _upright_exif(exif) if exif else None,
                _dpi(*info.get("dpi", (None, None))),
            )
    except UnidentifiedImageError:
        return ImageTags()
    except Exception:
        # The file is gone, or a decoder took what Pillow's plugins refuse in their own ways:
        # OSError, struct.error, NotImplementedError, ValueError past a PNG chunk's limit.
        logger.warning(f"The tags of this file cannot be read: {path}", exc_info=True)
        return ImageTags()


def _tags_for(metadata: ImageMetadata, image: np.ndarray, grey_file: bool) -> ImageTags:
    """The tags to write with this image: EXIF states its size, and a profile of another
    colour space than the file's (a colour image made grey) would be wrong, and leaves."""
    tags = metadata if isinstance(metadata, ImageTags) else read_image_tags(metadata)
    height, width, _ = get_h_w_c(image)
    exif = tags.exif
    if exif is not None:
        sizes = dict.fromkeys(_WIDTH_TAGS, width) | dict.fromkeys(_HEIGHT_TAGS, height)
        exif = _set_exif_values(exif, sizes, True)
    icc = tags.icc_profile
    if icc is not None:
        if icc[16:20] != (b"GRAY" if grey_file else b"RGB "):
            logger.warning(
                f"Save Image: a colour profile of space {icc[16:20]!r} does not fit this"
                " file and is not written."
            )
            icc = None
    return ImageTags(icc, exif, tags.dpi)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind + payload)
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _png(data: bytes, tags: ImageTags) -> bytes:
    chunks = b""
    if tags.icc_profile and len(tags.icc_profile) > MAX_TEXT_CHUNK:
        logger.warning(
            "Save Image: this colour profile is too large for a PNG Pillow can open."
        )
    elif tags.icc_profile:
        name = b"ICC Profile\x00\x00"
        chunks += _png_chunk(b"iCCP", name + zlib.compress(tags.icc_profile))
    if tags.dpi:
        chunks += _png_chunk(b"pHYs", struct.pack(">IIB", *_per_metre(tags.dpi), 1))
    if tags.exif:
        chunks += _png_chunk(b"eXIf", tags.exif)
    after_ihdr = 8 + 12 + 13
    return data[:after_ihdr] + chunks + data[after_ihdr:]


def _jpeg_segment(marker: int, payload: bytes) -> bytes:
    return struct.pack(">BBH", 0xFF, marker, len(payload) + 2) + payload


def _jpeg(data: bytes, tags: ImageTags) -> bytes:
    segments = b""
    if tags.exif:
        if len(_EXIF_PREFIX) + len(tags.exif) > _JPEG_SEGMENT:
            logger.warning("Save Image: this EXIF is too large for a JPEG segment.")
        else:
            segments += _jpeg_segment(0xE1, _EXIF_PREFIX + tags.exif)
    if tags.icc_profile:
        starts = range(0, len(tags.icc_profile), _JPEG_ICC_CHUNK)
        if len(starts) > 255:
            logger.warning("Save Image: this colour profile is too large for a JPEG.")
        else:
            for number, start in enumerate(starts, 1):
                chunk = tags.icc_profile[start : start + _JPEG_ICC_CHUNK]
                head = b"ICC_PROFILE\x00" + bytes((number, len(starts)))
                segments += _jpeg_segment(0xE2, head + chunk)
    head = bytearray(data[:2])
    rest = 2
    if data[2:4] == b"\xff\xe0" and data[6:11] == b"JFIF\x00":
        rest = 4 + struct.unpack_from(">H", data, 4)[0]
        head = bytearray(data[:rest])
        if tags.dpi:
            density = (min(max(round(value), 1), 65535) for value in tags.dpi)
            struct.pack_into(">BHH", head, 13, 1, *density)
    return bytes(head) + segments + data[rest:]


def _webp(data: bytes, tags: ImageTags, image: np.ndarray) -> bytes:
    """The file in WebP's extended form, whose ICCP chunk comes first and EXIF chunk
    after the image data. WebP has no DPI field."""
    height, width, _ = get_h_w_c(image)
    chunks: list[tuple[bytes, bytes]] = []
    at = 12
    while at + 8 <= len(data):
        kind, size = struct.unpack_from("<4sI", data, at)
        chunks.append((kind, data[at + 8 : at + 8 + size]))
        at += 8 + size + (size & 1)
    flags = 0
    if chunks[0][0] == b"VP8X":
        flags = chunks.pop(0)[1][0]
    elif chunks[0][0] == b"VP8L" and chunks[0][1][4] & 0x10:
        flags = 0x10  # alpha_is_used in its header; a plain VP8 has no alpha
    if tags.icc_profile:
        flags |= 0x20
        chunks.insert(0, (b"ICCP", tags.icc_profile))
    if tags.exif:
        flags |= 0x08
        chunks.append((b"EXIF", tags.exif))
    canvas = (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    chunks.insert(0, (b"VP8X", bytes((flags, 0, 0, 0)) + canvas))
    body = b"WEBP"
    for kind, payload in chunks:
        padding = b"\x00" * (len(payload) & 1)
        body += kind + struct.pack("<I", len(payload)) + payload + padding
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _tiff(data: bytes, tags: ImageTags) -> bytes:
    """The file with its directory rewritten at the end with the tags: the samples and
    values already written stay where they are."""
    endian = "<" if data[:2] == b"II" else ">"
    if struct.unpack_from(f"{endian}H", data, 2)[0] != 42:
        logger.warning("Save Image: tags are not written to a BigTIFF.")
        return data
    first = struct.unpack_from(f"{endian}I", data, 4)[0]
    count = struct.unpack_from(f"{endian}H", data, first)[0]
    entries = {
        struct.unpack_from(f"{endian}H", data, at)[0]: data[at : at + 12]
        for at in range(first + 2, first + 2 + 12 * count, 12)
    }
    following = data[first + 2 + 12 * count : first + 6 + 12 * count]
    out = bytearray(data)

    def append(values: bytes) -> int:
        out.extend(b"\x00" * (len(out) & 1))
        at = len(out)
        out.extend(values)
        return at

    def store(tag: int, kind: int, values: int, value: bytes) -> None:
        field = (
            value.ljust(4, b"\x00")
            if len(value) <= 4
            else struct.pack(f"{endian}I", append(value))
        )
        entries[tag] = struct.pack(f"{endian}HHI", tag, kind, values) + field

    if tags.icc_profile:
        store(_ICC_TAG, 7, len(tags.icc_profile), tags.icc_profile)
    if tags.dpi:
        for tag, value in zip((_X_RESOLUTION, _Y_RESOLUTION), tags.dpi):
            store(tag, 5, 1, _rational(value, endian))
        store(_RESOLUTION_UNIT, 3, 1, struct.pack(f"{endian}H", 2))
    if tags.exif:
        exif = Exif()
        exif.load(tags.exif)
        exif = _only_file_tags(exif)
        exif.endian = endian
        at = append(b"")
        written = _write_exif(exif, at)
        if written is not None:
            # Its prefix and TIFF header, then a directory whose values' offsets count
            # from the offset given.
            directory = written[len(_EXIF_PREFIX) + 8 :]
            out.extend(directory)
            added = struct.unpack_from(f"{endian}H", directory, 0)[0]
            for entry in range(2, 2 + 12 * added, 12):
                tag = struct.unpack_from(f"{endian}H", directory, entry)[0]
                entries[tag] = directory[entry : entry + 12]
    table = struct.pack(f"{endian}H", len(entries))
    table += b"".join(entries[tag] for tag in sorted(entries)) + following
    at = append(table)
    if len(out) >= 1 << 32:
        logger.warning("Save Image: tags are not written to a TIFF of 4 GiB.")
        return data
    struct.pack_into(f"{endian}I", out, 4, at)
    return bytes(out)


def _bmp(data: bytes, tags: ImageTags) -> bytes:
    """BMP keeps DPI alone, as pixels per metre."""
    if not tags.dpi:
        return data
    out = bytearray(data)
    struct.pack_into("<ii", out, 38, *_per_metre(tags.dpi))
    return bytes(out)


def embed_image_tags(
    data: bytes | np.ndarray, metadata: ImageMetadata, image: np.ndarray
) -> bytes | np.ndarray:
    """An encoded file (bytes or an encoder's buffer) with the metadata's tags: PNG, JPEG,
    TIFF and WebP take them, BMP its DPI, AVIF takes them from avif_tag_options; other
    formats have no place for them. A file that gains nothing is returned as it is, and
    so is one whose tags fail to go in, with a warning: tags stop no save."""
    head = bytes(data[:12])
    webp = head.startswith(b"RIFF") and head[8:12] == b"WEBP"
    if not (webp or head.startswith((b"\x89PNG", b"\xff\xd8", b"II", b"MM", b"BM"))):
        return data
    try:
        # WebP stores a grey image as RGB.
        tags = _tags_for(metadata, image, not webp and get_h_w_c(image)[2] == 1)
        if tags == ImageTags():
            return data
        file = bytes(data)
        if webp:
            return _webp(file, tags, image) if tags.icc_profile or tags.exif else data
        if head.startswith(b"\x89PNG"):
            return _png(file, tags)
        if head.startswith(b"\xff\xd8"):
            return _jpeg(file, tags)
        if head.startswith(b"BM"):
            return _bmp(file, tags)
        return _tiff(file, tags)
    except Exception:
        logger.warning("Save Image: the tags are not written.", exc_info=True)
        return data


def avif_tag_options(
    metadata: ImageMetadata, image: np.ndarray, subsampling: str
) -> dict[str, bytes]:
    """The save options of a Pillow AVIF that holds the metadata's tags. An AVIF is grey
    by its subsampling, not by the image."""
    try:
        tags = _tags_for(metadata, image, subsampling == "4:0:0")
    except Exception:
        logger.warning("Save Image: the tags are not written.", exc_info=True)
        return {}
    options: dict[str, bytes] = {}
    if tags.icc_profile:
        options["icc_profile"] = tags.icc_profile
    if tags.exif:
        options["exif"] = _without_orientation(tags.exif)
    return options
