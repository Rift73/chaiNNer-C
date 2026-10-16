"""Engine files as Save Engine writes them: a serialized TensorRT engine as is, or
compressed with LZMA (its Compress input; a DUAL engine's zero padding and per-layer
plugin PTX make it about 4x smaller). Readers here recognise a compressed file by its
xz header and decompress it, so TensorRT gets the same bytes either way; TensorRT's
own tools read only the plain form."""

from __future__ import annotations

import lzma
from pathlib import Path

# Save Engine appends it to a compressed engine's name.
COMPRESSED_SUFFIX = "_compressed"
XZ_HEADER = b"\xfd7zXZ\x00"


def write_engine_file(path: Path, engine_bytes: bytes, *, compressed: bool) -> None:
    path.write_bytes(lzma.compress(engine_bytes) if compressed else engine_bytes)


def read_engine_file(path: Path | str) -> bytes:
    """The serialized engine in a file, compressed by Save Engine or not."""
    data = Path(path).read_bytes()
    return lzma.decompress(data) if data.startswith(XZ_HEADER) else data
