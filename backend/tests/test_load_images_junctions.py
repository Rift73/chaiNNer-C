"""Load Images follows junctions inside its folder but enters each directory once, by
its resolved path: a junction back to an ancestor ended only at Windows' 63-reparse
limit, listing every image under it 64 times, and a junction to a folder already in
the tree listed that folder's images twice."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows")

# The node's modules load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from nodes.impl.image_formats import get_available_image_formats
from packages.chaiNNer_standard.image.batch_processing.load_images import (
    list_glob,
    load_images_node,
)

PIXELS = np.zeros((2, 3, 3), np.uint8)


def junction(link: Path, target: Path) -> None:
    import _winapi

    _winapi.CreateJunction(str(target), str(link))


def test_loop_and_shared_junctions_list_every_image_once(tmp_path: Path):
    root = tmp_path / "images"
    (root / "sub" / "deeper").mkdir(parents=True)
    other = tmp_path / "other"
    other.mkdir()
    for path in (
        root / "a.png",
        root / "sub" / "b.png",
        root / "sub" / "deeper" / "c.png",
    ):
        Image.fromarray(PIXELS).save(path)
    Image.fromarray(PIXELS).save(other / "d.png")
    links = [
        root / "loop",  # back to the folder itself
        root / "sub" / "deeper" / "up",  # back to an ancestor
        root / "again",  # a second way into sub
        root / "outside",  # a folder outside the tree, followed once
    ]
    junction(links[0], root)
    junction(links[1], root / "sub")
    junction(links[2], root / "sub")
    junction(links[3], other)
    try:
        paths = list_glob(root, "**/*", get_available_image_formats())
        names = sorted(p.name for p in paths)
        assert names == ["a.png", "b.png", "c.png", "d.png"]
        assert len({os.path.realpath(p) for p in paths}) == 4
        generator, _ = load_images_node(root, False, True, "", False, 0, True)
        items = list(generator.supplier())
        assert len(items) == 4 and not any(isinstance(i, Exception) for i in items)
    finally:
        for link in links:
            os.rmdir(link)  # the junction only, never its target
