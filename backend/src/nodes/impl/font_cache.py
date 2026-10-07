"""Exclusively leased FreeType faces for the two bundled-font image nodes.

An active face is removed from the idle pool. Parallel rendering therefore
never shares FreeType's mutable glyph slot. Keep at most 16 idle faces, at
most 8 MiB of their backing font files, and sizes no larger than 512 pixels.
Larger fonts still render through the same Pillow primitive without retention.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from threading import Lock

from PIL import ImageFont

_lock = Lock()
_idle: OrderedDict[tuple, tuple[ImageFont.FreeTypeFont, int]] = OrderedDict()
_retained_bytes = 0
_MAX_BYTES = 8 * 1024 * 1024
_MAX_FACES = 16


@contextmanager
def font_lease(path: str, size: int) -> Iterator[ImageFont.FreeTypeFont]:
    global _retained_bytes
    # Invalid paths/sizes must still raise Pillow's original diagnostic.
    key = None
    cost = 0
    if type(size) is int and 0 < size <= 512:
        try:
            info = os.stat(path)
        except OSError:
            pass
        else:
            cost = info.st_size
            if 0 < cost <= _MAX_BYTES:
                key = (
                    path,
                    size,
                    info.st_dev,
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                )
    font = None
    if key is not None:
        with _lock:
            found = _idle.pop(key, None)
            if found is not None:
                font, old_cost = found
                _retained_bytes -= old_cost
    if font is None:
        if key is not None:
            try:
                with open(path, "rb") as stream:
                    data = stream.read(_MAX_BYTES + 1)
            except OSError:
                # Preserve Pillow's own path/error handling when the file cannot
                # be read for caching (for example, permissions changed).
                key = None
            else:
                if len(data) > _MAX_BYTES:
                    key = None
                else:
                    cost = len(data)
                    # FreeType's filename API keeps Windows handles open. An
                    # owned byte buffer avoids pinning bundled files in a cache.
                    font = ImageFont.truetype(BytesIO(data), size=size)
        if font is None:
            font = ImageFont.truetype(path, size=size)
    try:
        yield font
    finally:
        if key is not None:
            with _lock:
                previous = _idle.pop(key, None)
                if previous is not None:
                    _retained_bytes -= previous[1]
                while _idle and (
                    len(_idle) >= _MAX_FACES or _retained_bytes + cost > _MAX_BYTES
                ):
                    _, (_, removed) = _idle.popitem(last=False)
                    _retained_bytes -= removed
                _idle[key] = (font, cost)
                _retained_bytes += cost


def clear() -> None:
    global _retained_bytes
    with _lock:
        _idle.clear()
        _retained_bytes = 0


def retained() -> tuple[int, int]:
    with _lock:
        return len(_idle), _retained_bytes
