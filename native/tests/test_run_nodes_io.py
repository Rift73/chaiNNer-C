"""Native image I/O entry points (native/src/run/nodes/io.cpp) against their Python bodies."""

from __future__ import annotations

import os
import subprocess
import sys
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from native_differential import assert_native_matches

RNG = np.random.default_rng(83)
Y, X = np.mgrid[0:37, 0:53]
RGB = np.clip(np.stack([X * 4.0, Y * 6.0, (X + Y) * 2.0], -1) + RNG.normal(0, 9, (37, 53, 3)), 0, 255).astype(np.uint8)
ALPHA = ((X * 5) % 256).astype(np.uint8)


def jpeg(path: Path, mode: str = "RGB", orientation: int | None = None, **options) -> Path:
    im = Image.fromarray(RGB).convert(mode)
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        options["exif"] = exif.tobytes()
    im.save(path, "JPEG", quality=90, **options)
    return path


def written(path: Path, array: np.ndarray, *params: int) -> Path:
    assert cv2.imwrite(str(path), array, list(params))
    return path


NATIVE = {
    "rgb.jpg": lambda p: jpeg(p),
    "grey.jpeg": lambda p: jpeg(p, "L"),
    "cmyk.jpg": lambda p: jpeg(p, "CMYK"),
    "progressive.jpg": lambda p: jpeg(p, progressive=True, subsampling=0),
    "turned.jpg": lambda p: jpeg(p, orientation=6),
    "flipped.JPG": lambda p: jpeg(p, orientation=2),
    "pillow.jfif": lambda p: jpeg(p),
    "opencv.jpe": lambda p: jpeg(p, "CMYK"),
    "rgb.png": lambda p: written(p, RGB),
    "rgba.png": lambda p: written(p, np.dstack([RGB, ALPHA])),
    "grey16.png": lambda p: written(p, RGB[:, :, 0].astype(np.uint16) * 257),
    "lossy.webp": lambda p: written(p, RGB, cv2.IMWRITE_WEBP_QUALITY, 80),
    "alpha.webp": lambda p: written(p, np.dstack([RGB, ALPHA]), cv2.IMWRITE_WEBP_QUALITY, 101),
    "a.b.c.png": lambda p: written(p, RGB),
    "..dots.png": lambda p: written(p, RGB),
    "雪.png": lambda p: written(p.with_name("snow.png"), RGB).rename(p) or p,
    "jpeg_named.png": lambda p: jpeg(p),
    "rgb.bmp": lambda p: written(p, RGB),
    "rgba.dib": lambda p: written(p.with_suffix(".bmp"), np.dstack([RGB, ALPHA])).rename(p) or p,
    "grey.pgm": lambda p: written(p, RGB[:, :, 1]),
    "rgb.ppm": lambda p: written(p, RGB),
    "deep.pgm": lambda p: written(p, RGB[:, :, 2].astype(np.uint16) * 257),
    # TIFF content under another name reaches the cv route (Pillow's straight-alpha decoder runs only for .tif/.tiff).
    "float_tiff.png": lambda p: written(p.with_suffix(".tif"), RGB.astype(np.float32) / 300 - 0.1).rename(p) or p,
    "int16_tiff.png": lambda p: written(p.with_suffix(".tif"), RGB.astype(np.int16) * 129 - 9000).rename(p) or p,
    "uint16_tiff.png": lambda p: written(p.with_suffix(".tif"), RGB.astype(np.uint16) * 200).rename(p) or p,
    "double_tiff.png": lambda p: written(p.with_suffix(".tif"), RGB[:, :, 0].astype(np.float64) / 255).rename(p) or p,
    "straight.tif": lambda p: Image.fromarray(np.dstack([RGB, ALPHA])).save(p, compression="tiff_lzw") or p,
    "straight_raw.tiff": lambda p: Image.fromarray(np.dstack([RGB, ALPHA])).save(p) or p,
}


@pytest.mark.parametrize("name", sorted(NATIVE))
def test_load_image_matches(tmp_path, name):
    path = NATIVE[name](tmp_path / name)
    outcome = assert_native_matches("chainner:image:load", [path])
    assert outcome[0] == "outputs"


@pytest.mark.parametrize(
    "name,make",
    [
        ("stays.tif", lambda p: written(p, RGB)),
        ("rgb_pillow.tiff", lambda p: Image.fromarray(RGB).save(p) or p),
        ("stays.tga", lambda p: Image.fromarray(RGB).save(p) or p),
        ("empty.png", lambda p: p.write_bytes(b"") or p),
        ("truncated.jpg", lambda p: p.write_bytes(jpeg(p).read_bytes()[:400]) and p),
        ("not_an_image.png", lambda p: p.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(40)) and p),
    ],
)
def test_load_image_declines_to_python(tmp_path, name, make):
    path = make(tmp_path / name)
    assert_native_matches("chainner:image:load", [path], declines=True)


SAVE = "chainner:image:save"


def save_inputs(img: np.ndarray, directory: Path, filename: str = "image", fmt: str = "PNG", relative=None, **options):
    """Save Image's raw inputs in schema order."""
    from inspect import signature

    import packages.chaiNNer_standard.image.io.save_image as module

    values = dict(
        lazy_image=img,
        base_directory=directory,
        relative_path=relative,
        filename=filename,
        image_format=module.ImageFormat[fmt],
        png_color_depth=module.PngColorDepth.U8,
        webp_lossless=False,
        quality=95,
        jpeg_chroma_subsampling=module.JpegSubsampling.FACTOR_422,
        jpeg_progressive=False,
        tiff_color_depth=module.TiffColorDepth.U8,
        tiff_compression=module.TiffCompression.LZW,
        dds_format="BC7_UNORM",
        dds_bc7_compression=module.BC7Compression.DEFAULT,
        dds_error_metric=module.DDSErrorMetric.PERCEPTUAL,
        dds_dithering=False,
        dds_mipmap_levels=0,
        dds_mip_levels=0,
        dds_mip_filter=module.DDSMipFilter.DEFAULT,
        dds_separate_alpha=False,
        dds_colour_conversion=module.DDSColourConversion.KEEP,
        dds_premultiply_alpha=False,
        dds_gpu=False,
        avif_chroma_subsampling=module.AvifSubsampling.FACTOR_420,
        skip_existing_files=False,
    )
    for key, value in options.items():
        values[key] = getattr(module, value[0])[value[1]] if isinstance(value, tuple) else value
    # The UI's raw values: a dropdown sends its option's value.
    return [
        values[key].value if isinstance(values[key], Enum) else values[key]
        for key in signature(module.save_image_node).parameters
    ]


def run_save(inputs: list[object], native: bool) -> None:
    """Save Image's Python body or native entry point on the enforced inputs."""
    import api
    import process
    from native_differential import node_data
    from nodes.impl.native_graph import graph

    node = node_data(SAVE)
    enforced = process.enforce_inputs(list(inputs), node, api.NodeId("save"), [])
    if native:
        graph().run_native(SAVE, enforced)
    else:
        node.run(*enforced)


def float_image(channels: int, seed: int = 4) -> np.ndarray:
    values = np.random.default_rng(seed).random((37, 53, channels), dtype=np.float32)
    values[:2, :2] = 0.5 / 255  # half-way samples round to even
    return values[:, :, 0] if channels == 1 else values


SAVE_CASES = [
    ("PNG", {}),
    ("PNG", {"png_color_depth": ("PngColorDepth", "U16")}),
    ("JPG", {"quality": 0, "jpeg_chroma_subsampling": ("JpegSubsampling", "FACTOR_444")}),
    ("JPG", {"quality": 77, "jpeg_chroma_subsampling": ("JpegSubsampling", "FACTOR_420"), "jpeg_progressive": True}),
    ("JPG", {"jpeg_chroma_subsampling": ("JpegSubsampling", "FACTOR_440")}),
    ("WEBP", {"quality": 0}),
    ("WEBP", {"quality": 63}),
    ("WEBP", {"webp_lossless": True}),
    ("BMP", {}),
    ("TIFF", {}),
    ("TIFF", {"tiff_compression": ("TiffCompression", "NONE")}),
    ("TIFF", {"tiff_compression": ("TiffCompression", "ZIP"), "tiff_color_depth": ("TiffColorDepth", "U16")}),
    ("TIFF", {"tiff_color_depth": ("TiffColorDepth", "F32")}),
]


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("fmt,options", SAVE_CASES)
def test_save_image_writes_todays_bytes(tmp_path, fmt, options, channels):
    img = float_image(channels)
    written = []
    for native in (False, True):
        directory = tmp_path / str(native)
        run_save(save_inputs(img, directory, fmt=fmt, **options), native)
        written.append([(path.name, path.read_bytes()) for path in sorted(directory.iterdir())])
    assert written[0] == written[1] and len(written[0]) == 1


@pytest.mark.parametrize("fmt", ["GIF", "TGA", "AVIF", "DDS"])
def test_save_image_declines_other_formats(tmp_path, fmt):
    from nodes.impl.native_graph import graph

    with pytest.raises(graph().NativeDeclined):
        run_save(save_inputs(float_image(3), tmp_path, fmt=fmt), True)
    assert not any(tmp_path.iterdir())


def test_save_image_skips_existing_without_reading(tmp_path):
    from api import Lazy

    target = tmp_path / "image.png"
    target.write_bytes(b"existing")
    reads = []
    lazy = Lazy(lambda: reads.append(1) or float_image(3))
    run_save(save_inputs(lazy, tmp_path, skip_existing_files=True), True)
    assert target.read_bytes() == b"existing" and not reads


def test_save_image_equal_file_left_untouched(tmp_path):
    run_save(save_inputs(float_image(3), tmp_path), False)
    target = tmp_path / "image.png"
    os.utime(target, (1_000_000_000, 1_000_000_000))
    run_save(save_inputs(float_image(3), tmp_path), True)
    assert target.stat().st_mtime == 1_000_000_000


def junction(link: Path, target: Path) -> Path:
    import _winapi

    _winapi.CreateJunction(str(target), str(link))
    return link


def symlink(link: Path, target: Path, directory: bool) -> Path:
    try:
        os.symlink(target, link, target_is_directory=directory)
    except OSError:
        pytest.skip("creating symbolic links needs the privilege")
    return link


@pytest.mark.parametrize(
    "shape",
    ["plain", "junction", "directory_symlink", "relative", "drive_relative", "dotdot", "missing_tail", "unicode",
     "trailing_dot_space", "case_overwrite"],
)
def test_save_image_targets_the_same_file(tmp_path, monkeypatch, shape):
    # The native entry point's lexical target names the same file as today's Path.resolve(): same bytes, the same
    # file, and the same names in the folder (Fable, lane I Q3).
    real = tmp_path / "real"
    real.mkdir()
    monkeypatch.chdir(tmp_path)
    base, relative, filename = real, None, "image"
    if shape == "junction":
        base = junction(tmp_path / "link", real)
    elif shape == "directory_symlink":
        base = symlink(tmp_path / "link", real, True)
    elif shape == "relative":
        base = Path("real")
    elif shape == "drive_relative":
        base = Path(str(real)[2:])
    elif shape == "dotdot":
        (tmp_path / "other").mkdir()
        base, relative = tmp_path / "other", "../real"
    elif shape == "missing_tail":
        relative = "a/b/c"
    elif shape == "unicode":
        filename = "雪 ünï"
    elif shape == "trailing_dot_space":
        filename = "name. "
    expected = (Path(base) / (relative or "") / f"{filename}.png").resolve()
    outcomes = []
    for native in (False, True):
        if shape == "case_overwrite":
            (real / "IMAGE.png").write_bytes(b"existing")
        run_save(save_inputs(float_image(3), base, filename, relative=relative), native)
        files = sorted(path for path in real.rglob("*") if path.is_file())
        assert len(files) == 1 and os.path.samefile(files[0], expected)
        outcomes.append([(str(path.relative_to(real)), path.read_bytes()) for path in files])
        for path in files:
            path.unlink()
    assert outcomes[0] == outcomes[1]


def test_save_image_declines_a_file_symlink_target(tmp_path):
    from nodes.impl.native_graph import graph

    (tmp_path / "real.png").write_bytes(b"existing")
    symlink(tmp_path / "image.png", tmp_path / "real.png", False)
    with pytest.raises(graph().NativeDeclined):
        run_save(save_inputs(float_image(3), tmp_path), True)


@pytest.mark.parametrize("skip", [False, True])
def test_save_image_declines_a_dangling_link(tmp_path, skip):
    # Path.exists() follows a link: a dangling one does not exist for today's path, so skip-existing writes.
    from nodes.impl.native_graph import graph

    symlink(tmp_path / "image.png", tmp_path / "missing.png", False)
    with pytest.raises(graph().NativeDeclined):
        run_save(save_inputs(float_image(3), tmp_path, skip_existing_files=skip), True)


# server.import_packages()'s packages and node modules, then Pillow's limit; os._exit skips the interpreter shutdown
# that crashes after every package loaded (native/STATUS.md).
MAX_PIXELS = """
import importlib, os
__file__ = os.path.abspath("run.py")
import api

for package in ("standard", "pytorch", "ncnn", "onnx", "external"):
    importlib.import_module("packages.chaiNNer_" + package)
api.registry.load_nodes(os.path.abspath("server.py"))
from PIL import Image
print(Image.MAX_IMAGE_PIXELS, flush=True)
os._exit(0)
"""


def test_pillow_max_image_pixels_is_the_constant():
    # The native Pillow route bounds images by Pillow's default limit (io.cpp pillow_max_pixels); nothing in the
    # worker changes it (Fable, lane I Q4).
    done = subprocess.run(
        [sys.executable, "-B", "-c", MAX_PIXELS],
        cwd=Path(__file__).resolve().parents[2] / "backend" / "src",
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[-1] == "89478485"
