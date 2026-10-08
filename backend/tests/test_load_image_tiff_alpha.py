"""Load Image and Load Images keep the straight colour of 8-bit TIFFs with unassociated
alpha, where OpenCV premultiplied it and lost the colour under alpha 0 (upstream
chaiNNer #409), and the alpha of 8-bit grey+alpha TIFFs, which OpenCV dropped. Every
other file still loads exactly as OpenCV decodes it."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest
from PIL import Image

from packages.chaiNNer_standard.image.batch_processing.load_images import (
    load_images_node,
)
from packages.chaiNNer_standard.image.io.load_image import _read_cv, load_image_node

RNG = np.random.default_rng(409)
# Straight RGBA: opaque, half and quarter alpha, and a colour under alpha 0 first.
RGBA8 = RNG.integers(0, 256, (7, 11, 4), dtype=np.uint8)
RGBA8[0, :4] = [[255, 0, 0, 255], [0, 200, 0, 128], [0, 0, 255, 0], [200, 200, 200, 64]]
RGBA16 = RNG.integers(0, 65536, (7, 11, 4), dtype=np.uint16)
FLOAT = RNG.random((7, 11, 4), dtype=np.float32)


def write_tiff(
    path: Path, samples: np.ndarray, extra: tuple[int, ...], orientation: int = 1
) -> None:
    """An uncompressed little-endian grey or RGB TIFF in one strip, with exact
    ExtraSamples."""
    h, w, c = samples.shape
    data = samples.astype(samples.dtype.newbyteorder("<")).tobytes()
    tags = {
        256: [w],
        257: [h],
        258: [samples.dtype.itemsize * 8] * c,
        259: [1],
        262: [1 if c < 3 else 2],
        273: [0],
        274: [orientation],
        277: [c],
        278: [h],
        279: [len(data)],
        284: [1],
        338: list(extra),
        339: [3 if samples.dtype.kind == "f" else 1] * c,
    }
    longs = (273, 279)
    sizes = [len(v) * (4 if t in longs else 2) for t, v in tags.items()]
    start = 8 + 2 + 12 * len(tags) + 4
    tags[273] = [start + sum(s for s in sizes if s > 4)]
    entries, blobs = b"", b""
    for tag, values in sorted(tags.items()):
        kind = "I" if tag in longs else "H"
        raw = struct.pack(f"<{len(values)}{kind}", *values)
        if len(raw) > 4:
            raw, blobs = struct.pack("<I", start + len(blobs)), blobs + raw
        entries += struct.pack("<HHI", tag, 4 if tag in longs else 3, len(values))
        entries += raw.ljust(4, b"\0")
    ifd = struct.pack("<H", len(tags)) + entries + b"\0\0\0\0"
    path.write_bytes(b"II*\0" + struct.pack("<I", 8) + ifd + blobs + data)


def encode(ext: str, image: np.ndarray) -> Callable[[Path], object]:
    return lambda path: path.write_bytes(cv2.imencode(ext, image)[1].tobytes())


def pillow(
    image: np.ndarray, compression: str | None = None
) -> Callable[[Path], object]:
    return lambda path: Image.fromarray(image).save(path, compression=compression)


def bgra(rgba: np.ndarray) -> np.ndarray:
    return rgba[:, :, [2, 1, 0, 3]]


def assert_identical(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.flags.c_contiguous == expected.flags.c_contiguous
    assert actual.tobytes() == expected.tobytes()


# name: (extension, writer, the straight RGBA it loads as)
STRAIGHT = {
    **{
        f"pillow-{compression}{ext}": (
            ext,
            pillow(RGBA8, compression=compression),
            RGBA8,
        )
        for compression in (None, "tiff_lzw", "tiff_adobe_deflate", "packbits")
        for ext in (".tif", ".tiff")
    },
    # Mirrored or turned as the Orientation tag says, as OpenCV does too.
    **{
        f"orientation-{o}": (
            ".tif",
            lambda p, o=o: write_tiff(p, RGBA8, (2,), o),
            turned,
        )
        for o, turned in ((2, RGBA8[:, ::-1]), (3, RGBA8[::-1, ::-1]), (4, RGBA8[::-1]))
    },
}


@pytest.mark.parametrize("name", STRAIGHT)
def test_straight_alpha_tiff_keeps_its_colour(tmp_path: Path, name: str):
    ext, write, expected = STRAIGHT[name]
    path = tmp_path / f"straight{ext}"
    write(path)
    assert_identical(load_image_node(path)[0], np.ascontiguousarray(bgra(expected)))


def test_load_images_keeps_straight_alpha(tmp_path: Path):
    pillow(RGBA8, compression="tiff_lzw")(tmp_path / "straight.tif")
    generator, _ = load_images_node(tmp_path, False, False, "", False, 0, True)
    (item,) = list(generator.supplier())
    assert not isinstance(item, Exception)
    assert_identical(item[0], np.ascontiguousarray(bgra(RGBA8)))


# Grey+alpha: opaque, half and quarter alpha, and a grey under alpha 0 first.
LA8 = np.ascontiguousarray(RGBA8[:, :, [0, 3]])
# name: (extension, writer, the grey+alpha it loads as)
GREY_ALPHA = {
    **{
        f"pillow-{compression}{ext}": (
            ext,
            pillow(LA8, compression=compression),
            LA8,
        )
        for compression in (None, "tiff_lzw", "tiff_adobe_deflate", "packbits")
        for ext in (".tif", ".tiff")
    },
    **{
        f"orientation-{o}": (".tif", lambda p, o=o: write_tiff(p, LA8, (2,), o), turned)
        for o, turned in (
            (1, LA8),
            (2, LA8[:, ::-1]),
            (3, LA8[::-1, ::-1]),
            (4, LA8[::-1]),
        )
    },
}


@pytest.mark.parametrize("name", GREY_ALPHA)
def test_grey_alpha_tiff_loads_as_a_grey_alpha_png_does(tmp_path: Path, name: str):
    ext, write, expected = GREY_ALPHA[name]
    path = tmp_path / f"grey-alpha{ext}"
    write(path)
    png = tmp_path / "grey-alpha.png"
    Image.fromarray(np.ascontiguousarray(expected)).save(png)
    grey, alpha = expected[:, :, 0], expected[:, :, 1]
    bgra_png = load_image_node(png)[0]
    assert_identical(bgra_png, np.ascontiguousarray(np.dstack([grey] * 3 + [alpha])))
    assert_identical(load_image_node(path)[0], bgra_png)


def test_16_bit_straight_alpha_tiff_was_already_exact(tmp_path: Path):
    path = tmp_path / "straight16.tif"
    write_tiff(path, RGBA16, (2,))
    assert_identical(load_image_node(path)[0], np.ascontiguousarray(bgra(RGBA16)))


UNCHANGED = {
    "rgb8-opencv.tif": encode(".tif", RGBA8[:, :, :3]),
    "rgb8-pillow-lzw.tif": pillow(RGBA8[:, :, :3], compression="tiff_lzw"),
    "rgba8-associated.tif": lambda p: write_tiff(p, RGBA8, (1,)),
    "rgba8-unspecified.tif": lambda p: write_tiff(p, RGBA8, (0,)),
    "rgba8-no-extrasamples.tif": encode(".tif", RGBA8),
    # Pillow 12.3 does not transpose orientations 5-8; they stay with OpenCV.
    **{
        f"rgba8-straight-orientation-{o}.tif": lambda p, o=o: write_tiff(
            p, RGBA8, (2,), o
        )
        for o in (5, 6, 7, 8)
    },
    "grey8.tif": encode(".tif", RGBA8[:, :, 0]),
    "grey-alpha8-associated.tif": lambda p: write_tiff(p, LA8, (1,)),
    "grey-alpha8-unspecified.tif": lambda p: write_tiff(p, LA8, (0,)),
    **{
        f"grey-alpha8-straight-orientation-{o}.tif": lambda p, o=o: write_tiff(
            p, LA8, (2,), o
        )
        for o in (5, 6, 7, 8)
    },
    "grey-alpha8.png": pillow(LA8),
    "rgb16.tif": encode(".tif", RGBA16[:, :, :3]),
    "rgba16-opencv.tif": encode(".tif", RGBA16),
    "rgba16-straight.tif": lambda p: write_tiff(p, RGBA16, (2,)),
    "rgba16-associated.tif": lambda p: write_tiff(p, RGBA16, (1,)),
    "float1.tiff": encode(".tiff", FLOAT[:, :, 0]),
    "float3.tiff": encode(".tiff", FLOAT[:, :, :3]),
    "float4.tiff": encode(".tiff", FLOAT),
    "float4-straight.tif": lambda p: write_tiff(p, FLOAT, (2,)),
    "rgba8.png": encode(".png", RGBA8),
    "rgb8.png": encode(".png", RGBA8[:, :, :3]),
    "png-named.tif": encode(".png", RGBA8),
}


@pytest.mark.parametrize("name", UNCHANGED)
def test_other_files_load_as_opencv_decodes_them(tmp_path: Path, name: str):
    path = tmp_path / name
    UNCHANGED[name](path)
    expected = _read_cv(path)
    assert expected is not None
    assert_identical(load_image_node(path)[0], expected)
