"""Engine files as Save Engine writes them: a serialized TensorRT engine as is, or
compressed (its Compress input) with LZMA or Zstandard; a DUAL engine's zero padding
and per-layer plugin PTX make it about 4x smaller. Readers here recognise a compressed
file by its header and decompress it, so TensorRT gets the same bytes either way;
TensorRT's own tools read only the plain form."""

from __future__ import annotations

import lzma
from enum import Enum
from pathlib import Path

from compression import zstd

# Save Engine appends it to a compressed engine's name.
COMPRESSED_SUFFIX = "_compressed"
XZ_HEADER = b"\xfd7zXZ\x00"
ZSTD_HEADER = b"\x28\xb5\x2f\xfd"
# A 1x DUAL Light engine (47.5 MB): 13.3 MB, about 50 ms to load and 6 s to save; LZMA
# is 11.7 MB, 0.44 s and 6.6 s.
ZSTD_LEVEL = 19


class EngineCompression(Enum):
    LZMA = "lzma"
    ZSTD = "zstd"


def write_engine_file(
    path: Path, engine_bytes: bytes, *, compression: EngineCompression | None
) -> None:
    if compression is EngineCompression.LZMA:
        engine_bytes = lzma.compress(engine_bytes)
    elif compression is EngineCompression.ZSTD:
        engine_bytes = zstd.compress(engine_bytes, ZSTD_LEVEL)
    path.write_bytes(engine_bytes)


def read_engine_file(path: Path | str) -> bytes:
    """The serialized engine in a file, compressed by Save Engine or not."""
    data = Path(path).read_bytes()
    if data.startswith(XZ_HEADER):
        return lzma.decompress(data)
    if data.startswith(ZSTD_HEADER):
        return zstd.decompress(data)
    return data
