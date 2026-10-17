"""VOSR's TensorRT plans as its engine caches keep them: Zstandard-compressed (about a
quarter smaller), streamed into TensorRT as they decompress at about 1 GB/s. A build
writes plain plans; the cache compresses them, and a set cached plain (before
2026-10-17) on its next use."""

from __future__ import annotations

import os
import shutil
from io import BufferedReader
from pathlib import Path

from compression import zstd

# Weights leave little to find: level 3 makes plans as small as level 19 does
# (74-82 %), at about 1 GB/s instead of a few MB/s.
LEVEL = 3
SUFFIX = ".plan.zst"


def compressed(directory: Path, name: str) -> Path:
    return directory / f"{name}{SUFFIX}"


def stored(directory: Path, manifest: dict, name: str) -> Path:
    """A cached set's plan of a stage, as its manifest describes the set."""
    if manifest.get("compression") == "zstd":
        return compressed(directory, name)
    return directory / f"{name}.plan"


def compress(source: Path, target: Path) -> None:
    """Compress the plan at source into target, which appears only when complete."""
    part = target.with_name(f"{target.name}.part")
    with source.open("rb") as reader, zstd.open(part, "wb", level=LEVEL) as writer:
        shutil.copyfileobj(reader, writer, 16 << 20)
    os.replace(part, target)


def open_plan(path: Path) -> zstd.ZstdFile | BufferedReader:
    """A plan to read, decompressing it if it is compressed. TensorRT reads a plan in
    order and seeks only to where it is or a little ahead, which a Zstandard stream does
    without decompressing anything twice."""
    return zstd.open(path) if path.name.endswith(".zst") else path.open("rb")
