"""VOSR's TensorRT plans as its engine caches keep them: Zstandard-compressed (about a
quarter smaller) in independent frames, which threads decompress at about 3 GB/s (one
thread: 1 GB/s). A build writes plain plans; the cache compresses them, and a set cached
plain (by older versions) on its next use."""

from __future__ import annotations

import mmap
import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BufferedReader, BytesIO
from itertools import islice
from pathlib import Path

from compression import zstd
from sanic.log import logger

# Weights leave little to find: level 3 makes plans as small as level 19 does
# (74-82 %), at about 1 GB/s instead of a few MB/s.
LEVEL = 3
SUFFIX = ".plan.zst"
# Frames of this much plan keep the ratio of one long frame (0.994 GB of 1.207 either way).
FRAME = 64 << 20
THREADS = 8


def compressed(directory: Path, name: str) -> Path:
    """A stage's plan as a cache keeps it."""
    return directory / f"{name}{SUFFIX}"


def plain(directory: Path, name: str) -> Path:
    """A stage's plan as a build writes it."""
    return directory / f"{name}.plan"


def stored(directory: Path, manifest: dict, name: str) -> Path:
    """A cached set's plan of a stage, as its manifest describes the set."""
    if manifest.get("compression") == "zstd":
        return compressed(directory, name)
    return plain(directory, name)


def set_aside(directory: Path, label: str) -> None:
    """Move a corrupt cached set out of its cache's way, so a fresh set can take its
    place; it stays until one has (vosr2_engine_cache.remove_unused_sets)."""
    quarantine = directory.with_name(f"quarantine-{directory.name}-{time.time_ns()}")
    directory.rename(quarantine)
    logger.warning(
        "The cached VOSR %s engine set %s is corrupt; moved it to %s.",
        label,
        directory,
        quarantine,
    )


def compress(source: Path, target: Path) -> None:
    """Compress the plan at source into target, which appears only when complete."""
    part = target.with_name(f"{target.name}.part")
    with source.open("rb") as reader, part.open("wb") as writer:
        with ThreadPoolExecutor(THREADS) as pool:
            blocks = iter(lambda: reader.read(FRAME), b"")
            while batch := list(islice(blocks, THREADS)):
                writer.writelines(
                    pool.map(lambda block: zstd.compress(block, level=LEVEL), batch)
                )
    os.replace(part, target)


def open_plan(path: Path) -> BytesIO | BufferedReader:
    """A plan to read, whole in memory if it is compressed (a set compressed by older
    versions is one frame, which takes one thread)."""
    if not path.name.endswith(".zst"):
        return path.open("rb")
    with path.open("rb") as file, mmap.mmap(
        file.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        with memoryview(data) as view:
            frames = []
            start = 0
            while start < len(view):
                size = zstd.get_frame_size(view[start:])
                frames.append(view[start : start + size])
                start += size
            with ThreadPoolExecutor(THREADS) as pool:
                plan = b"".join(pool.map(zstd.decompress, frames))
            for frame in frames:
                frame.release()
    return BytesIO(plan)
