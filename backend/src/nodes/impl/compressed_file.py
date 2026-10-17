"""Model files chaiNNer-C writes compressed: Save Engine's and ONNX Save Model's
Compress, with LZMA or Zstandard. Readers recognise a compressed file by its header and
decompress it, so the file keeps its extension and its contents are the same bytes once
read; other programs read only the plain form."""

from __future__ import annotations

import lzma
from enum import Enum
from pathlib import Path

from compression import zstd

# The saving nodes append it to a compressed file's name.
COMPRESSED_SUFFIX = "_compressed"
XZ_HEADER = b"\xfd7zXZ\x00"
ZSTD_HEADER = b"\x28\xb5\x2f\xfd"
# A 1x DUAL Light engine (47.5 MB): 13.3 MB, about 50 ms to load and 6 s to save; LZMA
# is 11.7 MB, 0.44 s and 6.6 s.
ZSTD_LEVEL = 19


class Compression(Enum):
    LZMA = "lzma"
    ZSTD = "zstd"


COMPRESSION_LABELS = {
    Compression.LZMA: "LZMA (smallest)",
    Compression.ZSTD: "Zstandard (fastest to load)",
}


def compressed_name(name: str, compress: bool) -> str:
    """A file name with COMPRESSED_SUFFIX when compressed, never twice."""
    if compress and not name.endswith(COMPRESSED_SUFFIX):
        return name + COMPRESSED_SUFFIX
    return name


def write_file(path: Path, data: bytes, *, compression: Compression | None) -> None:
    if compression is Compression.LZMA:
        data = lzma.compress(data)
    elif compression is Compression.ZSTD:
        data = zstd.compress(data, ZSTD_LEVEL)
    path.write_bytes(data)


def decompressed(path: Path | str) -> bytes | None:
    """The contents of a compressed file; None for a plain one, read as usual."""
    with open(path, "rb") as file:
        header = file.read(len(XZ_HEADER))
        if not header.startswith((XZ_HEADER, ZSTD_HEADER)):
            return None
        data = header + file.read()
    return (
        lzma.decompress(data) if data.startswith(XZ_HEADER) else zstd.decompress(data)
    )


def read_file(path: Path | str) -> bytes:
    """A file's contents, decompressed if compressed."""
    data = decompressed(path)
    return Path(path).read_bytes() if data is None else data
