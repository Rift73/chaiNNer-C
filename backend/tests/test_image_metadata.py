"""Load Image's Convert To sRGB against ImageCms (Little CMS) on profile fixtures, and the
colour profile, EXIF and DPI that Save Image writes from Load Image's Metadata, read back
with Pillow. Untagged and sRGB images stay bit-identical; a file's pixels do not change
when it gains tags."""

from __future__ import annotations

import io
import math
import struct
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms, TiffImagePlugin

# The node's modules load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import Lazy
from nodes.impl import image_tags
from nodes.impl.colour_profile import convert_to_srgb
from nodes.impl.image_tags import ImageTags, _upright_exif, read_image_tags
from nodes.impl.image_utils import normalize
from packages.chaiNNer_standard.image.batch_processing.load_images import (
    load_images_node,
)
from packages.chaiNNer_standard.image.io import save_image
from packages.chaiNNer_standard.image.io.load_image import load_image_node
from packages.chaiNNer_standard.image.io.save_image import (
    AvifSubsampling,
    BC7Compression,
    DDSColourConversion,
    DDSErrorMetric,
    DDSMipFilter,
    ExrColorDepth,
    ExrCompression,
    ImageFormat,
    JpegSubsampling,
    PngColorDepth,
    TiffColorDepth,
    TiffCompression,
)

XY = tuple[float, float]
Primaries = tuple[XY, XY, XY]

D50 = np.array([0.9642, 1.0, 0.8249])
BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]
)
D65_XY = (0.3127, 0.3290)
D50_XY = (0.3457, 0.3585)
SRGB_PARA = (3, (2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045))

MAKE, MODEL, ORIENTATION = 0x010F, 0x0110, 0x0112
EXIF_IFD, DATE_TAKEN, PIXEL_X, PIXEL_Y = 0x8769, 0x9003, 0xA002, 0xA003


def s15(value: float) -> bytes:
    return struct.pack(">i", round(value * 65536))


def xyz_tag(xyz: Iterable[float]) -> bytes:
    return b"XYZ \0\0\0\0" + b"".join(s15(v) for v in xyz)


def gamma_tag(gamma: float) -> bytes:
    return b"curv\0\0\0\0" + struct.pack(">IH", 1, round(gamma * 256)) + b"\0\0"


def table_tag(values: np.ndarray) -> bytes:
    codes = np.round(values * 65535).astype(">u2").tobytes()
    return b"curv\0\0\0\0" + struct.pack(">I", len(values)) + codes


def para_tag(function: int, params: tuple[float, ...]) -> bytes:
    body = struct.pack(">HH", function, 0) + b"".join(s15(p) for p in params)
    return b"para\0\0\0\0" + body


def text_tags() -> dict[bytes, bytes]:
    name = b"fixture\0"
    desc = b"desc\0\0\0\0" + struct.pack(">I", len(name)) + name + b"\0" * 78
    return {b"desc": desc, b"cprt": b"text\0\0\0\0none\0\0\0\0"}


def icc_file(space: bytes, kind: bytes, pcs: bytes, tags: dict[bytes, bytes]) -> bytes:
    table_size = 4 + 12 * len(tags)
    at = 128 + table_size
    table, body = struct.pack(">I", len(tags)), b""
    for signature, value in tags.items():
        table += signature + struct.pack(">II", at + len(body), len(value))
        body += value + b"\0" * (-len(value) % 4)
    header = bytearray(128)
    header[8:12] = struct.pack(">I", 0x02400000)
    header[12:16], header[16:20], header[20:24] = kind, space, pcs
    header[36:40] = b"acsp"
    header[68:80] = b"".join(s15(v) for v in D50)
    struct.pack_into(">I", header, 0, 128 + len(table) + len(body))
    return bytes(header) + table + body


def xy_to_xyz(xy: XY) -> np.ndarray:
    return np.array([xy[0] / xy[1], 1.0, (1 - xy[0] - xy[1]) / xy[1]])


def colorants(primaries: Primaries, white_xy: XY) -> np.ndarray:
    """rXYZ, gXYZ, bXYZ (columns) adapted to D50 by Bradford, as ICC keeps them."""
    xyz = np.stack([xy_to_xyz(p) for p in primaries], axis=1)
    white = xy_to_xyz(white_xy)
    scaled = xyz * np.linalg.solve(xyz, white)
    gain = BRADFORD @ D50 / (BRADFORD @ white)
    return np.linalg.inv(BRADFORD) @ np.diag(gain) @ BRADFORD @ scaled


def rgb_profile(
    primaries: Primaries, white_xy: XY, curve: bytes, c: np.ndarray | None = None
) -> bytes:
    c = colorants(primaries, white_xy) if c is None else c
    return icc_file(
        b"RGB ",
        b"mntr",
        b"XYZ ",
        {
            **text_tags(),
            b"wtpt": xyz_tag(D50),
            b"rXYZ": xyz_tag(c[:, 0]),
            b"gXYZ": xyz_tag(c[:, 1]),
            b"bXYZ": xyz_tag(c[:, 2]),
            b"rTRC": curve,
            b"gTRC": curve,
            b"bTRC": curve,
        },
    )


def grey_profile(curve: bytes) -> bytes:
    tags = {**text_tags(), b"wtpt": xyz_tag(D50), b"kTRC": curve}
    return icc_file(b"GRAY", b"mntr", b"XYZ ", tags)


def lut_profile(primaries: Primaries, white_xy: XY) -> bytes:
    """Linear RGB to XYZ as a lut16 table (linear, so two grid points are exact)."""
    c = colorants(primaries, white_xy)
    identity = struct.pack(">HH", 0, 65535)
    grid = [
        np.round(c @ np.array([r, g, b]) * 32768).clip(0, 65535).astype(">u2").tobytes()
        for r in (0, 1)
        for g in (0, 1)
        for b in (0, 1)
    ]
    head = b"mft2\0\0\0\0" + bytes((3, 3, 2, 0))
    head += b"".join(s15(v) for v in (1, 0, 0, 0, 1, 0, 0, 0, 1))
    head += struct.pack(">HH", 2, 2)
    a2b0 = head + identity * 3 + b"".join(grid) + identity * 3
    tags = {**text_tags(), b"wtpt": xyz_tag(D50), b"A2B0": a2b0}
    return icc_file(b"RGB ", b"scnr", b"XYZ ", tags)


ADOBE_PRIMARIES = ((0.64, 0.33), (0.21, 0.71), (0.15, 0.06))
P3_PRIMARIES = ((0.68, 0.32), (0.265, 0.69), (0.15, 0.06))
PROPHOTO_PRIMARIES = ((0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001))
SRGB_PRIMARIES = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))

ADOBE = rgb_profile(ADOBE_PRIMARIES, D65_XY, gamma_tag(563 / 256))
P3 = rgb_profile(P3_PRIMARIES, D65_XY, para_tag(*SRGB_PARA))
PROPHOTO = rgb_profile(PROPHOTO_PRIMARIES, D50_XY, gamma_tag(1.8))
PROPHOTO_TABLE = rgb_profile(
    PROPHOTO_PRIMARIES, D50_XY, table_tag(np.linspace(0, 1, 1024) ** 1.8)
)
PROFILES = {
    "adobe-rgb": ADOBE,
    "display-p3": P3,
    "prophoto": PROPHOTO,
    "prophoto-table": PROPHOTO_TABLE,
    "bt2020-para0": rgb_profile(
        ((0.708, 0.292), (0.17, 0.797), (0.131, 0.046)),
        D65_XY,
        para_tag(0, (2.4,)),
    ),
    "para1": rgb_profile(ADOBE_PRIMARIES, D65_XY, para_tag(1, (2.2, 0.95, 0.05))),
    "para2": rgb_profile(ADOBE_PRIMARIES, D65_XY, para_tag(2, (2.2, 0.9, 0.05, 0.05))),
    "para4": rgb_profile(
        P3_PRIMARIES,
        D65_XY,
        para_tag(4, (2.4, 0.9, 0.05, 0.08, 0.05, 0.02, 0.01)),
    ),
}
SRGB_CODES = np.linspace(0, 1, 1024)
SRGB_TABLE = table_tag(
    np.where(
        SRGB_CODES <= 0.04045, SRGB_CODES / 12.92, ((SRGB_CODES + 0.055) / 1.055) ** 2.4
    )
)
SRGB_PROFILES = {
    "little-cms": ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes(),
    "parametric": rgb_profile(SRGB_PRIMARIES, D65_XY, para_tag(*SRGB_PARA)),
    "table": rgb_profile(SRGB_PRIMARIES, D65_XY, SRGB_TABLE),
    # The colorants of the sRGB IEC61966-2.1 profile most software embeds.
    "iec61966": rgb_profile(
        SRGB_PRIMARIES,
        D65_XY,
        SRGB_TABLE,
        np.array(
            [
                [0.43607, 0.38515, 0.14307],
                [0.22249, 0.71687, 0.06061],
                [0.01392, 0.09708, 0.71410],
            ]
        ),
    ),
}

RNG = np.random.default_rng(2117)
# RGB in file order: a noise field, then every grey and the pure ramps of each channel.
RAMP = np.arange(256, dtype=np.uint8)
NOISE = RNG.integers(0, 256, (40, 256, 3), dtype=np.uint8)
FIELD = np.concatenate(
    [
        NOISE,
        np.stack([RAMP, RAMP, RAMP], axis=1)[None],
        np.stack([RAMP, 0 * RAMP, 0 * RAMP], axis=1)[None],
        np.stack([0 * RAMP, RAMP, 0 * RAMP], axis=1)[None],
        np.stack([0 * RAMP, 0 * RAMP, RAMP], axis=1)[None],
    ]
)
EXPECTED_DPI = (300.0, 200.0)


def imagecms(rgb: np.ndarray, icc: bytes) -> np.ndarray:
    """Little CMS without the resampling it optimizes some 8-bit pipelines with, which
    is off in the shadows by more than 8 bits' rounding."""
    converted = ImageCms.profileToProfile(
        Image.fromarray(rgb),
        ImageCms.ImageCmsProfile(io.BytesIO(icc)),
        ImageCms.createProfile("sRGB"),
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        outputMode="RGB",
        flags=ImageCms.Flags.NOOPTIMIZE,
    )
    return np.asarray(converted)


def exif_bytes(orientation: int | None = None) -> bytes:
    exif = Image.Exif()
    exif[MAKE] = "Maker"
    exif[MODEL] = "Model X"
    if orientation is not None:
        exif[ORIENTATION] = orientation
    ifd = exif.get_ifd(EXIF_IFD)
    ifd[DATE_TAKEN] = "2026:11:06 03:30:00"
    ifd[PIXEL_X], ifd[PIXEL_Y] = 6000, 4000
    return exif.tobytes()


def tagged_png(
    path: Path, icc: bytes | None = ADOBE, array: np.ndarray = FIELD
) -> Path:
    options: dict[str, Any] = {"exif": exif_bytes(), "dpi": EXPECTED_DPI}
    if icc is not None:
        options["icc_profile"] = icc
    Image.fromarray(array).save(path, **options)
    return path


def inputs(
    image: object, directory: Path, fmt: ImageFormat, metadata: object, **options
) -> list[Any]:
    """Save Image's inputs in order."""
    return [
        image,
        directory,
        None,
        options.get("name", "image"),
        fmt,
        options.get("png_depth", PngColorDepth.U8),
        options.get("lossless", False),
        options.get("quality", 95),
        JpegSubsampling.FACTOR_444,
        False,
        options.get("tiff_depth", TiffColorDepth.U8),
        TiffCompression.LZW,
        "BC1_UNORM",
        BC7Compression.DEFAULT,
        DDSErrorMetric.PERCEPTUAL,
        False,
        0,
        0,
        DDSMipFilter.DEFAULT,
        False,
        DDSColourConversion.KEEP,
        False,
        True,
        options.get("avif", AvifSubsampling.FACTOR_444),
        ExrColorDepth.F32,
        ExrCompression.ZIP,
        False,
        metadata,
    ]


def save(
    image: np.ndarray, directory: Path, fmt: ImageFormat, metadata: object, **options
) -> Path:
    values = inputs(Lazy.ready(image), directory, fmt, metadata, **options)
    save_image.save_image_node(*values)
    return directory / f"{options.get('name', 'image')}.{fmt.value}"


def assert_exif(im: Image.Image) -> None:
    exif = im.getexif()
    assert exif[MAKE] == "Maker" and exif[MODEL] == "Model X"
    assert exif.get(ORIENTATION, 1) == 1
    ifd = exif.get_ifd(EXIF_IFD)
    assert ifd[DATE_TAKEN] == "2026:11:06 03:30:00"
    assert (ifd[PIXEL_X], ifd[PIXEL_Y]) == im.size


@pytest.mark.parametrize("name", PROFILES)
def test_rgb_conversion_matches_imagecms(name: str, tmp_path: Path):
    icc = PROFILES[name]
    path = tagged_png(tmp_path / "image.png", icc)
    converted, _, _, tags = load_image_node(path, True)
    assert converted.dtype == np.float32 and converted.shape == FIELD.shape
    reference = imagecms(FIELD, icc).astype(np.float64)
    difference = np.abs(converted[:, :, ::-1] * 255.0 - reference)
    assert difference.max() <= 1.0, difference.max()
    assert difference.mean() < 0.5
    # The fixture is another colour space indeed, and the pixels no longer need its tag.
    assert np.abs(reference - FIELD).max() > 8
    assert isinstance(tags, ImageTags)
    assert tags.icc_profile is None and tags.dpi is not None and tags.exif is not None


def test_table_based_profile_is_imagecms(tmp_path: Path):
    icc = lut_profile(ADOBE_PRIMARIES, D65_XY)
    path = tagged_png(tmp_path / "image.png", icc)
    converted = load_image_node(path, True)[0]
    reference = imagecms(FIELD, icc).astype(np.float64)
    assert np.abs(converted[:, :, ::-1] * 255.0 - reference).max() <= 1.0
    assert np.abs(reference - FIELD).max() > 8


@pytest.mark.parametrize("name", ["adobe-rgb", "display-p3", "prophoto"])
def test_higher_depths_agree_with_8_bits(name: str, tmp_path: Path):
    icc = PROFILES[name]
    bgr = FIELD[:, :, ::-1]
    eight = load_image_node(tagged_png(tmp_path / "image.png", icc), True)[0]
    metadata = ImageTags(icc_profile=icc)
    # Files of the same values at 16 bits and in float: their tags are Save Image's.
    png16 = save(
        normalize(bgr),
        tmp_path,
        ImageFormat.PNG,
        metadata,
        name="sixteen",
        png_depth=PngColorDepth.U16,
    )
    assert cv2.imread(str(png16), cv2.IMREAD_UNCHANGED).dtype == np.uint16
    np.testing.assert_allclose(load_image_node(png16, True)[0], eight, atol=2e-5)
    for depth, dtype in (
        (TiffColorDepth.U16, np.uint16),
        (TiffColorDepth.F32, np.float32),
    ):
        tiff = save(
            normalize(bgr),
            tmp_path,
            ImageFormat.TIFF,
            metadata,
            name=depth.value,
            tiff_depth=depth,
        )
        assert cv2.imread(str(tiff), cv2.IMREAD_UNCHANGED).dtype == dtype
        np.testing.assert_allclose(load_image_node(tiff, True)[0], eight, atol=2e-5)


def test_alpha_stays(tmp_path: Path):
    alpha = RNG.integers(0, 256, FIELD.shape[:2], dtype=np.uint8)
    rgba = np.dstack([FIELD, alpha])
    path = tagged_png(tmp_path / "image.png", ADOBE, rgba)
    converted = load_image_node(path, True)[0]
    assert converted.shape == rgba.shape
    np.testing.assert_array_equal(converted[:, :, 3], normalize(alpha))
    reference = imagecms(FIELD, ADOBE).astype(np.float64)
    assert np.abs(converted[:, :, 2::-1] * 255.0 - reference).max() <= 1.0


@pytest.mark.parametrize(
    "curve", [gamma_tag(1.8), para_tag(3, (2.2, 0.95, 0.05, 0.08, 0.04))]
)
def test_grey_profile_matches_imagecms(curve: bytes, tmp_path: Path):
    icc = grey_profile(curve)
    grey = np.arange(256, dtype=np.uint8).reshape(8, 32)
    path = tagged_png(tmp_path / "image.png", icc, grey)
    converted = load_image_node(path, True)[0]
    assert converted.ndim == 2 and converted.dtype == np.float32
    reference = imagecms(grey, icc)[:, :, 0].astype(np.float64)
    assert np.abs(converted * 255.0 - reference).max() <= 1.0
    assert np.abs(reference - grey).max() > 4


@pytest.mark.parametrize("name", SRGB_PROFILES)
def test_srgb_tagged_stays_bit_identical(name: str, tmp_path: Path):
    icc = SRGB_PROFILES[name]
    path = tagged_png(tmp_path / "image.png", icc)
    image = load_image_node(path, True)[0]
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image, FIELD[:, :, ::-1])
    tags = load_image_node(path, True)[3]
    assert isinstance(tags, ImageTags) and tags.icc_profile == icc


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".webp", ".tiff", ".bmp"])
def test_untagged_stays_bit_identical(suffix: str, tmp_path: Path):
    path = tmp_path / f"image{suffix}"
    Image.fromarray(FIELD).save(path)
    plain = load_image_node(path)[0]
    converted, _, _, tags = load_image_node(path, True)
    assert converted.dtype == plain.dtype
    np.testing.assert_array_equal(converted, plain)
    assert isinstance(tags, ImageTags) and tags.icc_profile is None


def test_off_leaves_tagged_pixels_and_carries_the_file(tmp_path: Path):
    path = tagged_png(tmp_path / "image.png")
    image, _, _, metadata = load_image_node(path)
    np.testing.assert_array_equal(image, FIELD[:, :, ::-1])
    assert metadata == path


@pytest.mark.parametrize(
    "icc",
    [
        pytest.param(grey_profile(gamma_tag(2.2)), id="grey-profile"),
        pytest.param(b"not a profile" * 40, id="garbage"),
        pytest.param(ADOBE.replace(b"curv", b"nope"), id="unknown-curve"),
        pytest.param(ADOBE[:140], id="truncated"),
        pytest.param(
            rgb_profile(ADOBE_PRIMARIES, D65_XY, para_tag(1, (2.2, 0.0, 0.05))),
            id="curve-dividing-by-zero",
        ),
        pytest.param(
            icc_file(b"RGB ", b"scnr", b"XYZ ", {b"A2B0": b"mft2 and nothing else"}),
            id="unreadable-table",
        ),
    ],
)
def test_unusable_profile_leaves_the_pixels(icc: bytes, tmp_path: Path):
    path = tmp_path / "image.png"
    Image.fromarray(FIELD).save(path, icc_profile=icc)
    image = load_image_node(path, True)[0]
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image, FIELD[:, :, ::-1])


FORMATS = {
    "png": (ImageFormat.PNG, {}, 3, "ied"),
    "png16": (ImageFormat.PNG, {"png_depth": PngColorDepth.U16}, 3, "ied"),
    "png-rgba": (ImageFormat.PNG, {}, 4, "ied"),
    "jpg": (ImageFormat.JPG, {}, 3, "ied"),
    "tiff": (ImageFormat.TIFF, {}, 3, "ied"),
    "tiff16": (ImageFormat.TIFF, {"tiff_depth": TiffColorDepth.U16}, 3, "ied"),
    "tiff-rgba": (ImageFormat.TIFF, {}, 4, "ied"),
    "webp": (ImageFormat.WEBP, {}, 3, "ie"),
    "webp-rgba": (ImageFormat.WEBP, {}, 4, "ie"),
    "webp-lossless": (ImageFormat.WEBP, {"lossless": True}, 3, "ie"),
    "webp-lossless-rgba": (ImageFormat.WEBP, {"lossless": True}, 4, "ie"),
    "avif": (ImageFormat.AVIF, {"quality": 90}, 3, "ie"),
    "bmp": (ImageFormat.BMP, {}, 3, "d"),
}


@pytest.mark.parametrize("name", FORMATS)
def test_saved_tags_read_back(name: str, tmp_path: Path):
    fmt, options, channels, holds = FORMATS[name]
    source = tagged_png(tmp_path / "source.png")
    image = normalize(load_image_node(source)[0])
    if channels == 4:
        ramp = np.linspace(0, 1, image.shape[1], dtype=np.float32)
        alpha = np.broadcast_to(ramp, image.shape[:2])
        image = np.dstack([image, alpha])
    with_tags = save(image, tmp_path, fmt, source, name="tags", **options)
    without = save(image, tmp_path, fmt, None, name="plain", **options)
    with Image.open(without) as im:
        assert "icc_profile" not in im.info
    with Image.open(with_tags) as im:
        assert im.info.get("icc_profile") == (ADOBE if "i" in holds else None)
        if "e" in holds:
            assert_exif(im)
        if "d" in holds:
            assert im.info["dpi"] == pytest.approx(EXPECTED_DPI, abs=0.5)
    np.testing.assert_array_equal(
        load_image_node(with_tags)[0], load_image_node(without)[0]
    )
    values = inputs(image, tmp_path, fmt, source, name="tags", **options)
    prepared = save_image.prepare_save_image(values)
    assert prepared is not None and bytes(prepared) == with_tags.read_bytes()
    assert read_image_tags(with_tags).icc_profile == (ADOBE if "i" in holds else None)


def test_float_tiff_tags_read_back(tmp_path: Path):
    source = tagged_png(tmp_path / "source.png")
    image = normalize(load_image_node(source)[0])
    options = {"tiff_depth": TiffColorDepth.F32}
    tagged = save(image, tmp_path, ImageFormat.TIFF, source, name="tags", **options)
    plain = save(image, tmp_path, ImageFormat.TIFF, None, name="plain", **options)
    # Pillow does not open float RGB, but reads the directory.
    with tagged.open("rb") as file:
        directory: Any = TiffImagePlugin.ImageFileDirectory_v2(file.read(8))
        file.seek(directory.next)
        directory.load(file)
    assert directory[34675] == ADOBE
    dpi = float(directory[282]), float(directory[283])
    assert dpi == pytest.approx(EXPECTED_DPI, abs=0.5) and directory[296] == 2
    assert directory[271] == "Maker" and 34665 in directory
    tags = read_image_tags(tagged)
    assert tags.icc_profile == ADOBE and tags.dpi == dpi
    np.testing.assert_array_equal(load_image_node(tagged)[0], load_image_node(plain)[0])


def test_file_without_tags_saves_the_same_bytes(tmp_path: Path):
    source = tmp_path / "source.png"
    cv2.imwrite(str(source), FIELD[:, :, ::-1])
    image = normalize(load_image_node(source)[0])
    for fmt in (ImageFormat.PNG, ImageFormat.JPG, ImageFormat.TIFF, ImageFormat.WEBP):
        tagged = save(image, tmp_path, fmt, source, name="tags")
        plain = save(image, tmp_path, fmt, None, name="plain")
        assert tagged.read_bytes() == plain.read_bytes(), fmt


def test_tags_stay_when_a_file_is_saved_over_itself(tmp_path: Path):
    source = tagged_png(tmp_path / "image.png")
    image, _, _, metadata = load_image_node(source)
    save(normalize(image), tmp_path, ImageFormat.PNG, metadata)
    with Image.open(source) as im:
        assert im.info["icc_profile"] == ADOBE
        assert_exif(im)


def test_large_profile_spans_jpeg_segments(tmp_path: Path):
    icc = rgb_profile(
        ADOBE_PRIMARIES, D65_XY, table_tag(np.linspace(0, 1, 60000) ** 2.2)
    )
    assert len(icc) > 3 * 65519
    path = save(
        normalize(FIELD[:, :, ::-1]),
        tmp_path,
        ImageFormat.JPG,
        ImageTags(icc_profile=icc),
    )
    with Image.open(path) as im:
        assert im.info["icc_profile"] == icc


def test_jpeg_source_turned_upright_says_so(tmp_path: Path):
    source = tmp_path / "source.jpg"
    Image.fromarray(FIELD).save(
        source, quality=95, exif=exif_bytes(6), icc_profile=ADOBE, dpi=EXPECTED_DPI
    )
    image, _, _, metadata = load_image_node(source)
    assert image.shape[:2] == (FIELD.shape[1], FIELD.shape[0])
    for fmt in (ImageFormat.PNG, ImageFormat.JPG, ImageFormat.TIFF, ImageFormat.WEBP):
        path = save(normalize(image), tmp_path, fmt, metadata)
        with Image.open(path) as im:
            assert_exif(im)
            assert im.info["icc_profile"] == ADOBE
        again = load_image_node(path)[0]
        assert again.shape[:2] == image.shape[:2], fmt


def test_turning_exif_upright_leaves_the_rest_byte_for_byte():
    raw = exif_bytes(6).removeprefix(b"Exif\0\0")
    tags_path = _upright_exif(raw)
    assert tags_path is not None and len(tags_path) == len(raw)
    differing = [i for i, (a, b) in enumerate(zip(raw, tags_path)) if a != b]
    assert (
        len(differing) == 1 and raw[differing[0]] == 6 and tags_path[differing[0]] == 1
    )


@pytest.mark.parametrize(
    "exif", [b"not exif at all", b"II*\0", b"MM\0*\xff\xff\xff\xff"]
)
def test_malformed_exif_does_not_stop_the_save(exif: bytes, tmp_path: Path):
    source = tmp_path / "source.png"
    Image.fromarray(FIELD).save(source, exif=exif)
    path = save(normalize(FIELD[:, :, ::-1]), tmp_path, ImageFormat.PNG, source)
    np.testing.assert_array_equal(load_image_node(path)[0], FIELD[:, :, ::-1])


def test_tiff_source_carries_only_its_origin_tags(tmp_path: Path):
    source = tmp_path / "source.tiff"
    Image.fromarray(FIELD).save(
        source, exif=exif_bytes(), icc_profile=ADOBE, dpi=EXPECTED_DPI
    )
    tags = read_image_tags(source)
    assert tags.icc_profile == ADOBE
    assert tags.dpi == pytest.approx(EXPECTED_DPI)
    exif = Image.Exif()
    exif.load(tags.exif)
    assert exif[MAKE] == "Maker" and 273 not in exif and 256 not in exif
    path = save(normalize(FIELD[:, :, ::-1]), tmp_path, ImageFormat.PNG, source)
    with Image.open(path) as im:
        assert_exif(im)
        assert im.getexif().get(273) is None


def test_bigtiff_source_gives_its_tags(tmp_path: Path):
    source = tmp_path / "source.tiff"
    Image.fromarray(FIELD).save(
        source, big_tiff=True, exif=exif_bytes(), icc_profile=ADOBE, dpi=EXPECTED_DPI
    )
    assert source.read_bytes()[2] == 43
    path = save(normalize(FIELD[:, :, ::-1]), tmp_path, ImageFormat.PNG, source)
    with Image.open(path) as im:
        assert im.info["icc_profile"] == ADOBE
        assert im.info["dpi"] == pytest.approx(EXPECTED_DPI, abs=0.5)
        assert_exif(im)


def test_profile_of_another_colour_space_is_not_written(tmp_path: Path):
    source = tagged_png(tmp_path / "source.png")
    grey = np.full((8, 9), 0.5, np.float32)
    path = save(grey, tmp_path, ImageFormat.PNG, source)
    with Image.open(path) as im:
        assert "icc_profile" not in im.info
        assert_exif(im)


def test_exif_size_stays_when_its_type_cannot_hold_the_image(tmp_path: Path):
    source = tmp_path / "source.png"
    exif = Image.Exif()
    exif.get_ifd(EXIF_IFD)[PIXEL_X] = 6000
    raw = exif.tobytes()
    assert struct.pack(">HH", PIXEL_X, 3) in raw  # a SHORT
    Image.fromarray(FIELD).save(source, exif=raw)
    saved = save(np.zeros((3, 70000, 3), np.float32), tmp_path, ImageFormat.PNG, source)
    with Image.open(saved) as im:
        assert im.getexif().get_ifd(EXIF_IFD)[PIXEL_X] == 6000


def test_png_profile_past_pillows_chunk_limit_gives_no_tags(tmp_path: Path):
    source = tmp_path / "source.png"
    cv2.imwrite(str(source), FIELD[:, :, ::-1])
    data = source.read_bytes()
    payload = b"name\0\0" + zlib.compress(b"\0" * (2 << 20))
    chunk = struct.pack(">I", len(payload)) + b"iCCP" + payload
    chunk += struct.pack(">I", zlib.crc32(b"iCCP" + payload))
    source.write_bytes(data[:33] + chunk + data[33:])
    assert read_image_tags(source) == ImageTags()
    np.testing.assert_array_equal(load_image_node(source, True)[0], FIELD[:, :, ::-1])


def test_png_exif_after_the_pixels_is_read(tmp_path: Path):
    source = tmp_path / "source.png"
    cv2.imwrite(str(source), FIELD[:, :, ::-1])
    data = source.read_bytes()
    exif = exif_bytes(6).removeprefix(b"Exif\0\0")
    chunk = struct.pack(">I", len(exif)) + b"eXIf" + exif
    chunk += struct.pack(">I", zlib.crc32(b"eXIf" + exif))
    source.write_bytes(data[:-12] + chunk + data[-12:])
    upright = Image.Exif()
    tags = read_image_tags(source)
    assert tags.exif is not None
    upright.load(tags.exif)
    assert upright[MAKE] == "Maker" and upright[ORIENTATION] == 1


def test_files_without_a_pillow_format_have_no_tags(tmp_path: Path):
    source = tmp_path / "source.hdr"
    cv2.imwrite(str(source), np.ones((4, 4, 3), np.float32))
    assert read_image_tags(source) == ImageTags()


def test_converted_image_saves_without_the_old_profile(tmp_path: Path):
    source = tagged_png(tmp_path / "source.png")
    image, _, _, metadata = load_image_node(source, True)
    path = save(normalize(image), tmp_path, ImageFormat.PNG, metadata)
    with Image.open(path) as im:
        assert "icc_profile" not in im.info
        assert_exif(im)
        assert im.info["dpi"] == pytest.approx(EXPECTED_DPI, abs=0.5)
        saved = np.array(im)
    assert np.abs(saved.astype(np.int16) - imagecms(FIELD, ADOBE)).max() <= 1


def test_load_images_gives_the_metadata_and_converts(tmp_path: Path):
    folder = tmp_path / "folder"
    folder.mkdir()
    tagged_png(folder / "a.png")
    for convert in (False, True):
        generator, _ = load_images_node(
            folder, False, False, "", False, 0, False, convert
        )
        items = list(generator.supplier())
        assert len(items) == 1 and isinstance(items[0], tuple)
        image, relative, name, index, metadata = items[0]
        assert (relative, name, index) == (".", "a", 0)
        expected = load_image_node(folder / "a.png", convert)
        np.testing.assert_array_equal(image, expected[0])
        assert metadata == expected[3]


def directory(
    entries: list[tuple[int, int, int, bytes]], start: int, endian: str = "<"
) -> bytes:
    """A TIFF directory at `start`, of these entries (tag, type, count, value), with its
    out-of-line values after it."""
    data_at = start + 2 + 12 * len(entries) + 4
    table = struct.pack(f"{endian}H", len(entries))
    tail = b""
    for tag, kind, count, value in sorted(entries):
        field = value.ljust(4, b"\0")
        if len(value) > 4:
            field = struct.pack(f"{endian}I", data_at + len(tail))
            tail += value + b"\0" * (len(value) & 1)
        table += struct.pack(f"{endian}HHI", tag, kind, count) + field
    return table + struct.pack(f"{endian}I", 0) + tail


def raw_exif(entries: list[tuple[int, int, int, bytes]]) -> bytes:
    return b"II*\0" + struct.pack("<I", 8) + directory(entries, 8)


def tiff_bytes(entries: list[tuple[int, int, int, bytes]]) -> bytes:
    """A 4x4 8-bit RGB TIFF with these entries added, as no encoder here writes them."""
    pixels = FIELD[:4, :4].tobytes()
    base = [
        (256, 4, 1, struct.pack("<I", 4)),
        (257, 4, 1, struct.pack("<I", 4)),
        (258, 3, 3, struct.pack("<HHH", 8, 8, 8)),
        (259, 3, 1, struct.pack("<H", 1)),
        (262, 3, 1, struct.pack("<H", 2)),
        (273, 4, 1, struct.pack("<I", 8)),
        (277, 3, 1, struct.pack("<H", 3)),
        (278, 4, 1, struct.pack("<I", 4)),
        (279, 4, 1, struct.pack("<I", len(pixels))),
    ]
    start = 8 + len(pixels)
    head = b"II*\0" + struct.pack("<I", start)
    return head + pixels + directory(base + entries, start)


def test_source_gone_before_the_save_writes_the_image_without_tags(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    source = tagged_png(tmp_path / "source.png")
    image, _, _, metadata = load_image_node(source)
    source.unlink()
    path = save(normalize(image), tmp_path, ImageFormat.PNG, metadata)
    with Image.open(path) as im:
        assert "icc_profile" not in im.info
    assert "cannot be read" in caplog.text


def odd_bitfields_bmp(path: Path) -> Path:
    """A 32-bit BMP whose masks OpenCV decodes and Pillow has no mode for."""
    width, height = 8, 6
    masks = struct.pack("<IIII", 0x0000FF00, 0x00FF0000, 0xFF000000, 0x000000FF)
    header = struct.pack(
        "<IiiHHIIiiII",
        108,
        width,
        height,
        1,
        32,
        3,
        width * height * 4,
        2835,
        2835,
        0,
        0,
    )
    header += masks + b"\0" * (108 - len(header) - len(masks))
    pixels = bytes(width * height * 4)
    offset = 14 + len(header)
    size = offset + len(pixels)
    file = b"BM" + struct.pack("<IHHI", size, 0, 0, offset) + header + pixels
    path.write_bytes(file)
    return path


def test_file_only_pillow_refuses_stops_neither_load_nor_save(tmp_path: Path):
    source = odd_bitfields_bmp(tmp_path / "source.bmp")
    image, _, _, metadata = load_image_node(source)
    assert read_image_tags(source) == ImageTags()
    np.testing.assert_array_equal(load_image_node(source, True)[0], image)
    assert save(normalize(image), tmp_path, ImageFormat.PNG, metadata).exists()


def test_dds_pillow_has_no_format_for_gives_no_tags(tmp_path: Path):
    side = 8
    pixel_format = struct.pack("<II4sIIIII", 32, 4, b"DX10", 0, 0, 0, 0, 0)
    head = struct.pack("<IIIIIII", 124, 0x1007, side, side, side * 8, 0, 1)
    head += b"\0" * 44 + pixel_format + struct.pack("<IIIII", 0x1000, 0, 0, 0, 0)
    dx10 = struct.pack("<IIIII", 11, 3, 0, 1, 0)  # R16G16B16A16_UNORM
    source = tmp_path / "source.dds"
    source.write_bytes(b"DDS " + head + dx10 + bytes(side * side * 8))
    assert read_image_tags(source) == ImageTags()


@pytest.mark.parametrize("size", [4, 5])
def test_truncated_tiff_gives_no_tags(size: int, tmp_path: Path):
    source = tmp_path / "source.tif"
    source.write_bytes(b"II*\0\x08\0\0\0"[:size])
    assert read_image_tags(source) == ImageTags()


UNWRITABLE_TAGS = [
    pytest.param((306, 5, 1, struct.pack("<II", 1, 0)), id="date-as-rational"),
    pytest.param((270, 12, 1, struct.pack("<d", 1.5)), id="description-as-double"),
]


@pytest.mark.parametrize("entry", UNWRITABLE_TAGS)
def test_tiff_source_with_a_tag_pillow_cannot_write_keeps_its_other_tags(
    entry: tuple[int, int, int, bytes], tmp_path: Path
):
    source = tmp_path / "source.tiff"
    resolution = (282, 5, 1, struct.pack("<II", 300, 1))
    other = (283, 5, 1, struct.pack("<II", 300, 1))
    profile = (34675, 7, len(ADOBE), ADOBE)
    source.write_bytes(tiff_bytes([entry, profile, resolution, other]))
    tags = read_image_tags(source)
    assert tags == ImageTags(ADOBE, None, (300.0, 300.0))
    assert load_image_node(source, True)[0].dtype == np.float32


@pytest.mark.parametrize("entry", UNWRITABLE_TAGS)
def test_exif_pillow_cannot_write_leaves_a_tiff_with_its_other_tags(
    entry: tuple[int, int, int, bytes], tmp_path: Path
):
    exif = raw_exif([entry, (271, 2, 6, b"Maker\0")])
    tags = ImageTags(ADOBE, exif, EXPECTED_DPI)
    path = save(normalize(FIELD[:, :, ::-1]), tmp_path, ImageFormat.TIFF, tags)
    with Image.open(path) as im:
        assert im.info["icc_profile"] == ADOBE
        assert im.info["dpi"] == pytest.approx(EXPECTED_DPI, abs=0.5)
        assert MAKE not in im.getexif()


def test_failed_writer_leaves_the_encoded_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def broken(*_: object) -> bytes:
        raise struct.error("a writer's failure")

    plain = save(
        normalize(FIELD[:, :, ::-1]), tmp_path, ImageFormat.PNG, None, name="a"
    )
    monkeypatch.setattr(image_tags, "_png", broken)
    tagged = save(
        normalize(FIELD[:, :, ::-1]),
        tmp_path,
        ImageFormat.PNG,
        ImageTags(ADOBE),
        name="b",
    )
    assert tagged.read_bytes() == plain.read_bytes()


EXTREME_DPI = [
    pytest.param((4294967295.0, 4294967295.0), id="largest-rational"),
    pytest.param((1e8, 1e8), id="1e8"),
    pytest.param((1e-9, 1e-9), id="tiny"),
    pytest.param((math.inf, 300.0), id="inf"),
]


@pytest.mark.parametrize("dpi", EXTREME_DPI)
@pytest.mark.parametrize(
    "fmt", [ImageFormat.PNG, ImageFormat.BMP, ImageFormat.TIFF, ImageFormat.JPG]
)
def test_extreme_dpi_stops_no_save(
    fmt: ImageFormat, dpi: tuple[float, float], tmp_path: Path
):
    image = normalize(FIELD[:, :, ::-1])
    path = save(image, tmp_path, fmt, ImageTags(None, None, dpi))
    with Image.open(path) as im:
        im.load()


@pytest.mark.parametrize("unit", [2, 3])
@pytest.mark.parametrize(
    "fmt", [ImageFormat.PNG, ImageFormat.BMP, ImageFormat.TIFF, ImageFormat.JPG]
)
def test_tiff_with_the_largest_resolution_writes_the_largest_each_format_holds(
    fmt: ImageFormat, unit: int, tmp_path: Path
):
    source = tmp_path / "source.tiff"
    rational = struct.pack("<II", 4294967295, 1)
    source.write_bytes(
        tiff_bytes(
            [
                (282, 5, 1, rational),
                (283, 5, 1, rational),
                (296, 3, 1, struct.pack("<H", unit)),
            ]
        )
    )
    path = save(normalize(FIELD[:, :, ::-1]), tmp_path / "out", fmt, source)
    with Image.open(path) as im:
        im.load()
        assert im.info["dpi"][0] > 1000


def test_infinite_resolution_is_none(tmp_path: Path):
    source = tmp_path / "source.tiff"
    value = struct.pack("<d", math.inf)
    source.write_bytes(tiff_bytes([(282, 12, 1, value), (283, 12, 1, value)]))
    assert read_image_tags(source).dpi is None


def test_profile_past_pillows_limit_stays_out_of_a_png(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    icc = bytearray(ADOBE) + RNG.bytes(2 << 20)
    struct.pack_into(">I", icc, 0, len(icc))
    tags = ImageTags(bytes(icc), None, EXPECTED_DPI)
    path = save(normalize(FIELD[:, :, ::-1]), tmp_path, ImageFormat.PNG, tags)
    with Image.open(path) as im:
        im.load()
        assert "icc_profile" not in im.info
        assert im.info["dpi"] == pytest.approx(EXPECTED_DPI, abs=0.5)
    assert "too large" in caplog.text


GREY_ICC = grey_profile(gamma_tag(2.2))


@pytest.mark.parametrize(
    "fmt,options",
    [
        (ImageFormat.WEBP, {}),
        (ImageFormat.WEBP, {"lossless": True}),
        (ImageFormat.AVIF, {"quality": 90}),
    ],
)
def test_grey_profile_stays_out_of_files_that_are_rgb(
    fmt: ImageFormat, options: dict[str, Any], tmp_path: Path
):
    grey = normalize(FIELD[:, :, 0])
    path = save(grey, tmp_path, fmt, ImageTags(GREY_ICC), **options)
    with Image.open(path) as im:
        assert im.mode == "RGB" and "icc_profile" not in im.info


def test_rgb_profile_stays_out_of_a_grey_avif_and_a_grey_profile_stays_in(
    tmp_path: Path,
):
    image = normalize(FIELD[:, :, ::-1])
    options: dict[str, Any] = {"avif": AvifSubsampling.FACTOR_400, "quality": 90}
    path = save(image, tmp_path, ImageFormat.AVIF, ImageTags(ADOBE), **options)
    with Image.open(path) as im:
        assert im.mode == "L" and "icc_profile" not in im.info
    grey = normalize(FIELD[:, :, 0])
    path = save(grey, tmp_path, ImageFormat.AVIF, ImageTags(GREY_ICC), **options)
    with Image.open(path) as im:
        assert im.mode == "L" and im.info["icc_profile"] == GREY_ICC


@pytest.mark.parametrize("gamma", [1.0, 1.8, 2.4])
def test_grey_with_alpha_converts_and_keeps_its_alpha(gamma: float, tmp_path: Path):
    icc = grey_profile(gamma_tag(gamma))
    grey, alpha = FIELD[:, :, 0], FIELD[:, :, 1]
    path = tmp_path / "image.png"
    Image.fromarray(np.dstack([grey, alpha]), "LA").save(path, icc_profile=icc)
    image = load_image_node(path, True)[0]
    assert image.shape == (*grey.shape, 4) and image.dtype == np.float32
    np.testing.assert_array_equal(image[:, :, 3], normalize(alpha))
    reference = imagecms(grey, icc)[:, :, 0].astype(np.float64)
    for channel in range(3):
        assert np.abs(image[:, :, channel] * 255.0 - reference).max() <= 1.0


def test_grey_profile_on_a_colour_image_with_alpha_stays_unconverted(tmp_path: Path):
    rgba = np.dstack([FIELD, FIELD[:, :, 0]])
    path = tmp_path / "image.png"
    Image.fromarray(rgba).save(path, icc_profile=grey_profile(gamma_tag(1.8)))
    image = load_image_node(path, True)[0]
    assert image.dtype == np.uint8


@pytest.mark.parametrize(
    "grey_curve",
    [gamma_tag(2.2), para_tag(*SRGB_PARA)],
    ids=["other-curve", "srgb-curve"],
)
def test_rgb_profile_that_also_holds_a_grey_curve_uses_its_colours(grey_curve: bytes):
    c = colorants(ADOBE_PRIMARIES, D65_XY)
    curve = gamma_tag(563 / 256)
    icc = icc_file(
        b"RGB ",
        b"mntr",
        b"XYZ ",
        {
            **text_tags(),
            b"wtpt": xyz_tag(D50),
            b"rXYZ": xyz_tag(c[:, 0]),
            b"gXYZ": xyz_tag(c[:, 1]),
            b"bXYZ": xyz_tag(c[:, 2]),
            b"rTRC": curve,
            b"gTRC": curve,
            b"bTRC": curve,
            b"kTRC": grey_curve,
        },
    )
    converted = convert_to_srgb(FIELD[:, :, ::-1], icc, "fixture")
    reference = imagecms(FIELD, icc).astype(np.float64)
    assert np.abs(converted[:, :, ::-1] * 255.0 - reference).max() <= 1.0


def test_profile_little_cms_refuses_leaves_the_pixels(tmp_path: Path):
    tags = {bytes((65, 66, 67, 65 + i)): b"x" * 8 for i in range(101)}
    icc = icc_file(b"RGB ", b"mntr", b"XYZ ", tags | {b"A2B0": b"y" * 8})
    tagged_png(tmp_path / "a.png", icc)
    image = load_image_node(tmp_path / "a.png", True)[0]
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image, FIELD[:, :, ::-1])
    generator, _ = load_images_node(tmp_path, False, False, "", False, 0, False, True)
    assert isinstance(next(iter(generator.supplier())), tuple)


COPYRIGHT = "\xa9 M\xfcller 2026"
COPYRIGHT_BYTES = COPYRIGHT.encode("latin-1") + b"\0"


def test_exif_text_stays_on_the_way_into_a_tiff(tmp_path: Path):
    lens = (0xA434, 2, len(COPYRIGHT_BYTES), COPYRIGHT_BYTES)
    head = directory([(34665, 4, 1, struct.pack("<I", 8 + 18))], 8)
    exif = b"II*\0" + struct.pack("<I", 8) + head + directory([lens], 8 + 18)
    plain = raw_exif([(33432, 2, len(COPYRIGHT_BYTES), COPYRIGHT_BYTES)])
    image = normalize(FIELD[:, :, ::-1])
    path = save(image, tmp_path, ImageFormat.TIFF, ImageTags(None, plain), name="a")
    with Image.open(path) as im:
        assert im.getexif()[33432] == COPYRIGHT
    path = save(image, tmp_path, ImageFormat.TIFF, ImageTags(None, exif), name="b")
    with Image.open(path) as im:
        assert im.getexif().get_ifd(EXIF_IFD)[0xA434] == COPYRIGHT


def test_tiff_source_text_stays(tmp_path: Path):
    source = tmp_path / "source.tiff"
    entry = (33432, 2, len(COPYRIGHT_BYTES), COPYRIGHT_BYTES)
    source.write_bytes(tiff_bytes([entry]))
    tags = read_image_tags(source)
    assert tags.exif is not None and COPYRIGHT_BYTES in tags.exif


def thumbnail_exif(thumbnail: bytes) -> bytes:
    """IFD 0 with Orientation 6, and IFD 1 holding a thumbnail."""
    following = 8 + 2 + 12 * 2 + 4
    first = struct.pack("<H", 2)
    first += struct.pack("<HHI", MAKE, 2, 2) + b"M\0\0\0"
    first += struct.pack("<HHIHH", ORIENTATION, 3, 1, 6, 0)
    first += struct.pack("<I", following)
    data_at = following + 2 + 12 * 2 + 4
    second = struct.pack("<H", 2)
    second += struct.pack("<HHII", 0x0201, 4, 1, data_at)
    second += struct.pack("<HHII", 0x0202, 4, 1, len(thumbnail))
    second += struct.pack("<I", 0)
    return b"II*\0" + struct.pack("<I", 8) + first + second + thumbnail


def test_avif_keeps_the_exif_block_as_it_was_but_for_orientation(tmp_path: Path):
    thumbnail = b"\xff\xd8\xff\xd9thumbnail"
    source = tmp_path / "source.png"
    Image.fromarray(FIELD).save(source, exif=thumbnail_exif(thumbnail))
    image, _, _, metadata = load_image_node(source)
    path = save(normalize(image), tmp_path, ImageFormat.AVIF, metadata, quality=70)
    with Image.open(path) as im:
        assert thumbnail in im.info["exif"]
        assert im.getexif()[MAKE] == "M" and ORIENTATION not in im.getexif()


def test_avif_with_orientation_and_a_tag_pillow_cannot_write_keeps_them(tmp_path: Path):
    entries = [
        (MAKE, 2, 6, b"Maker\0"),
        (ORIENTATION, 3, 1, struct.pack("<H", 6)),
        (306, 5, 1, struct.pack("<II", 1, 0)),
    ]
    source = tmp_path / "source.png"
    Image.fromarray(FIELD).save(source, exif=raw_exif(entries))
    image, _, _, metadata = load_image_node(source)
    path = save(normalize(image), tmp_path, ImageFormat.AVIF, metadata, quality=70)
    with Image.open(path) as im:
        exif = im.getexif()
        assert exif[MAKE] == "Maker" and ORIENTATION not in exif and 306 in exif


@pytest.mark.parametrize("lossless", [False, True])
def test_webp_of_an_opaque_rgba_image_does_not_gain_alpha(
    lossless: bool, tmp_path: Path
):
    rgb = normalize(FIELD[:, :, ::-1])
    image = np.dstack([rgb, np.ones(rgb.shape[:2], np.float32)])
    plain = save(image, tmp_path, ImageFormat.WEBP, None, name="a", lossless=lossless)
    tagged = save(
        image, tmp_path, ImageFormat.WEBP, ImageTags(ADOBE), name="b", lossless=lossless
    )
    before = cv2.imdecode(np.fromfile(plain, np.uint8), cv2.IMREAD_UNCHANGED)
    after = cv2.imdecode(np.fromfile(tagged, np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(after, before)
    with Image.open(plain) as first, Image.open(tagged) as second:
        assert first.mode == second.mode


def test_webp_with_only_dpi_to_write_stays_as_encoded(tmp_path: Path):
    image = normalize(FIELD[:, :, ::-1])
    plain = save(image, tmp_path, ImageFormat.WEBP, None, name="a")
    tags = ImageTags(None, None, EXPECTED_DPI)
    tagged = save(image, tmp_path, ImageFormat.WEBP, tags, name="b")
    assert tagged.read_bytes() == plain.read_bytes()


def test_big_endian_bigtiff_says_its_tags_are_not_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    source = tmp_path / "source.tiff"
    source.write_bytes(
        b"MM\0+" + struct.pack(">HHQ", 8, 0, 16) + struct.pack(">QQ", 0, 0)
    )
    assert read_image_tags(source) == ImageTags()
    assert "BigTIFF" in caplog.text


def test_tags_are_those_the_file_holds_as_save_image_writes(tmp_path: Path):
    source = tagged_png(tmp_path / "source.png")
    image, _, _, metadata = load_image_node(source)
    image = normalize(image)
    assert save(image, tmp_path, ImageFormat.PNG, None, name="source") == source
    path = save(image, tmp_path, ImageFormat.PNG, metadata, name="copy")
    with Image.open(path) as im:
        assert "icc_profile" not in im.info
