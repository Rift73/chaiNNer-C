"""Image I/O paths: a NUL in a path raises the ValueError open() raises, not a cast
error (native/src/image_io.cpp file_path)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from nodes.impl import native_image_io


def test_a_nul_in_a_path_raises_what_open_raises(tmp_path: Path):
    path = str(tmp_path / "a") + "\0b.png"
    with pytest.raises(ValueError) as before:
        open(path, "wb")
    with pytest.raises(ValueError) as after:
        native_image_io.cv_save_image(path, np.zeros((4, 4, 3), np.uint8), [])
    # open() encodes a POSIX path to bytes first
    message = (
        "embedded null character" if sys.platform == "win32" else "embedded null byte"
    )
    assert str(after.value) == str(before.value) == message
    assert not list(tmp_path.iterdir())
