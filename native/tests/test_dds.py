"""DDS through the repository's texconv, CPU codecs only (-nogpu): the uncompressed
intermediates chaiNNer writes and reads in place of PNG."""

from __future__ import annotations

import inspect
import platform
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from nodes.impl.dds import texconv
from nodes.impl.dds.format import DXGI_CODES, DDSFormat
from nodes.impl.image_utils import to_uint8, to_uint16
from nodes.impl.native_image_io import cv_save_image
from packages.chaiNNer_standard.image.batch_processing import (
    load_dds_images,
    save_dds_texture,
)
from packages.chaiNNer_standard.image.io import load_dds, load_image, save_image

pytestmark = pytest.mark.skipif(
    platform.system() != "Windows", reason="texconv runs on Windows only"
)

ROOT = Path(__file__).resolve().parents[2]
TEXCONV = ROOT / "backend/src/texconv/texconv.exe"
# Save Image's formats before full DDS support: 8-bit targets, whose files must not
# change with the intermediate.
PNG_ERA_FORMATS = [
    "BC1_UNORM_SRGB",
    "BC1_UNORM",
    "BC3_UNORM_SRGB",
    "BC3_UNORM",
    "BC4_UNORM",
    "BC5_UNORM",
    "BC5_SNORM",
    "BC7_UNORM_SRGB",
    "BC7_UNORM",
    "DXT1",
    "DXT3",
    "DXT5",
    "R8G8B8A8_UNORM_SRGB",
    "R8G8B8A8_UNORM",
    "B8G8R8A8_UNORM_SRGB",
    "B8G8R8A8_UNORM",
    "B5G5R5A1_UNORM",
    "B5G6R5_UNORM",
    "B8G8R8X8_UNORM_SRGB",
    "B8G8R8X8_UNORM",
    "R8G8_UNORM",
    "R8_UNORM",
]


@pytest.fixture
def runs(monkeypatch):
    """Every texconv run gets -nogpu (once: texconv refuses a repeated option) and
    the repository's texconv; the arguments chaiNNer passed are recorded."""
    recorded: list[list[str]] = []
    original = vars(texconv)["__run_texconv"]

    def run(args, error):
        recorded.append(list(args))
        return original(args if "-nogpu" in args else ["-nogpu", *args], error)

    monkeypatch.setitem(vars(texconv), "__TEXCONV_EXE", str(TEXCONV))
    monkeypatch.setitem(vars(texconv), "__run_texconv", run)
    return recorded


def synthetic(h: int, w: int, channels: int, noise_level: float = 0.03) -> np.ndarray:
    """Float32 BGR(A) or grey in [0, 1]: gradients, a sine texture and noise; alpha
    with a ramp and a transparent hole."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32) / np.float32(max(h, w) - 1)
    noise = np.random.default_rng(h * w + channels).normal(0, noise_level, (h, w, 3))
    b = 0.5 + 0.35 * np.sin(12 * x + 3 * y)
    rgb = np.clip(np.stack([b, 0.15 + 0.7 * x, 0.2 + 0.6 * y], -1) + noise, 0, 1)
    rgb = rgb.astype(np.float32)
    if channels == 1:
        return rgb.mean(axis=-1)
    if channels == 3:
        return rgb
    alpha = np.where((y > 0.5) & (x < 0.4), 0, 0.2 + 0.8 * x).astype(np.float32)
    return np.concatenate([rgb, alpha[..., None]], -1)


class Lazy:
    def __init__(self, value):
        self.value = value


SAVE_DEFAULTS: dict[str, Any] = {
    "relative_path": None,
    "image_format": save_image.ImageFormat.DDS,
    "png_color_depth": save_image.PngColorDepth.U8,
    "webp_lossless": False,
    "quality": 95,
    "jpeg_chroma_subsampling": save_image.JpegSubsampling.FACTOR_422,
    "jpeg_progressive": False,
    "tiff_color_depth": save_image.TiffColorDepth.U8,
    "tiff_compression": save_image.TiffCompression.LZW,
    "dds_bc7_compression": save_image.BC7Compression.DEFAULT,
    "dds_error_metric": save_image.DDSErrorMetric.PERCEPTUAL,
    "dds_dithering": False,
    "dds_mipmap_levels": 0,
    "dds_mip_levels": 0,
    "dds_mip_filter": save_image.DDSMipFilter.DEFAULT,
    "dds_separate_alpha": False,
    "dds_colour_conversion": save_image.DDSColourConversion.KEEP,
    "dds_premultiply_alpha": False,
    "dds_gpu": True,
    "avif_chroma_subsampling": save_image.AvifSubsampling.FACTOR_420,
    "skip_existing_files": False,
}


def save(directory: Path, name: str, image: np.ndarray, fmt: str, **options) -> Path:
    """Save Image's DDS with its defaults but for the given inputs."""
    values: dict[str, Any] = {**SAVE_DEFAULTS, **options}
    values.update(lazy_image=Lazy(image), base_directory=directory, filename=name)
    values["dds_format"] = fmt
    assert set(inspect.signature(save_image.save_image_node).parameters) == set(values)
    save_image.save_image_node(**values)
    return directory / f"{name}.dds"


def read(path: Path) -> np.ndarray:
    image = load_image._read_dds(path)
    assert image is not None
    return image


def png_path_read(path: Path, scratch: Path) -> np.ndarray:
    """The reader before the intermediate: texconv to an RGBA PNG, OpenCV, the
    opaque alpha dropped."""
    command = [str(TEXCONV), "-nologo", "-nogpu", "-f", "rgba", "-srgb", "-ft", "png"]
    subprocess.run([*command, "-o", str(scratch), str(path)], check=True)
    image = cv2.imread(str(scratch / f"{path.stem}.png"), cv2.IMREAD_UNCHANGED)
    assert image is not None
    return image[..., :3] if (image[..., 3] == 255).all() else image


@pytest.mark.parametrize("fmt", PNG_ERA_FORMATS)
def test_8_bit_targets_are_the_png_path_bytes(tmp_path, runs, fmt):
    # Texconv is handed the to_uint8 pixels the PNG held, so it writes the same file;
    # the reader returns the same pixels as texconv's PNG did.
    (tmp_path / "png").mkdir()
    (tmp_path / "old").mkdir()
    (tmp_path / "read").mkdir()
    for channels in (1, 3, 4):
        for mips in (0, 1):
            name = f"c{channels}m{mips}"
            image = synthetic(21, 18, channels)
            written = save(tmp_path / "new", name, image, fmt, dds_mipmap_levels=mips)
            args = runs[-1]
            assert args[args.index("-o") + 1] == str(tmp_path / "new")
            png = tmp_path / "png" / f"{name}.png"
            cv_save_image(str(png), to_uint8(image, normalized=True), [])
            args[args.index("-o") + 1] = str(tmp_path / "old")
            vars(texconv)["__run_texconv"]([*args[:-1], str(png)], "PNG path")
            assert (
                written.read_bytes() == (tmp_path / "old" / f"{name}.dds").read_bytes()
            )
            new = read(written)
            old = png_path_read(written, tmp_path / "read")
            assert new.dtype == old.dtype and np.array_equal(new, old)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_16_bit_and_float_sources_read_at_their_depth(tmp_path, runs, channels):
    # 16-bit UNORM through an R16G16B16A16_UNORM intermediate (normalized as Load
    # Image's output does), float through R32G32B32A32_FLOAT, unclipped.
    image = synthetic(9, 11, channels) * np.float32(2.5) - np.float32(0.5)
    unorm = tmp_path / "unorm.dds"
    texconv.save_as_dds(unorm, image, "R16G16B16A16_UNORM", 1)
    assert runs[-1][-1].endswith("unorm.dds")
    quantized = to_uint16(image, normalized=True).astype(np.float32) / 65535
    expected = quantized if channels != 1 else np.repeat(quantized[..., None], 3, -1)
    back = read(unorm)
    assert back.dtype == np.float32 and np.array_equal(back, expected)
    assert "R16G16B16A16_UNORM" in runs[-1]
    floats = tmp_path / "float.dds"
    texconv.save_as_dds(floats, image, "R32G32B32A32_FLOAT", 1)
    back = read(floats)
    expected = image if channels != 1 else np.repeat(image[..., None], 3, -1)
    assert back.dtype == np.float32 and np.array_equal(back, expected)
    assert "R32G32B32A32_FLOAT" in runs[-1] and back.max() > 1


def test_reader_removes_its_temporary_folder(tmp_path, runs, monkeypatch):
    folders: list[str] = []
    monkeypatch.setitem(
        vars(texconv),
        "mkdtemp",
        lambda **kwargs: folders.append(str(tmp_path / "t")) or folders[-1],
    )
    (tmp_path / "t").mkdir()
    path = save(tmp_path, "a.b", synthetic(8, 8, 3), "BC1_UNORM")
    shutil.rmtree(tmp_path / "t", ignore_errors=True)
    (tmp_path / "t").mkdir()
    assert read(path).shape == (8, 8, 3)  # a dotted stem
    assert not (tmp_path / "t").exists()


# texconv May2026's -f formats (its help), by DXGI name; Save Image offers all but
# the YUV/video ones and XR_BIAS (typeless, depth, palettized, R1 and sampler feedback
# formats are not among them).
TEXCONV_FORMATS = """
R32G32B32A32_FLOAT R32G32B32A32_UINT R32G32B32A32_SINT R32G32B32_FLOAT R32G32B32_UINT
R32G32B32_SINT R16G16B16A16_FLOAT R16G16B16A16_UNORM R16G16B16A16_UINT
R16G16B16A16_SNORM R16G16B16A16_SINT R32G32_FLOAT R32G32_UINT R32G32_SINT
R10G10B10A2_UNORM R10G10B10A2_UINT R11G11B10_FLOAT R8G8B8A8_UNORM R8G8B8A8_UNORM_SRGB
R8G8B8A8_UINT R8G8B8A8_SNORM R8G8B8A8_SINT R16G16_FLOAT R16G16_UNORM R16G16_UINT
R16G16_SNORM R16G16_SINT R32_FLOAT R32_UINT R32_SINT R8G8_UNORM R8G8_UINT R8G8_SNORM
R8G8_SINT R16_FLOAT R16_UNORM R16_UINT R16_SNORM R16_SINT R8_UNORM R8_UINT R8_SNORM
R8_SINT A8_UNORM R9G9B9E5_SHAREDEXP R8G8_B8G8_UNORM G8R8_G8B8_UNORM BC1_UNORM
BC1_UNORM_SRGB BC2_UNORM BC2_UNORM_SRGB BC3_UNORM BC3_UNORM_SRGB BC4_UNORM BC4_SNORM
BC5_UNORM BC5_SNORM B5G6R5_UNORM B5G5R5A1_UNORM B8G8R8A8_UNORM B8G8R8X8_UNORM
R10G10B10_XR_BIAS_A2_UNORM B8G8R8A8_UNORM_SRGB B8G8R8X8_UNORM_SRGB BC6H_UF16 BC6H_SF16
BC7_UNORM BC7_UNORM_SRGB AYUV Y410 Y416 YUY2 Y210 Y216 B4G4R4A4_UNORM A4B4G4R4_UNORM
""".split()
NOT_OFFERED = {
    "AYUV",
    "Y410",
    "Y416",
    "YUY2",
    "Y210",
    "Y216",
    "R10G10B10_XR_BIAS_A2_UNORM",
}
ALL_FORMATS: list[DDSFormat] = [f for f, _ in save_image.SUPPORTED_DDS_FORMATS]
# PSNR floors (dB) of colour and alpha for the smooth 24x16 test image, against its
# 8-bit values (BC6H: its float values, peak = their range), about 2.5 dB under the
# CPU codecs' results (BC1-BC3 colour 29.5, alpha BC2 34.3 and BC3 50.5; BC4 44.3;
# BC5 50.4; BC6H 26.6 unsigned and 24.2 signed; BC7 34.9 and alpha 37.9).
PSNR_FLOORS = {
    "BC1": (27, None),
    "BC2": (27, 31),
    "BC3": (27, 47),
    "BC4": (42, None),
    "BC5": (48, None),
    "BC6H": (22, None),
    "BC7": (32, 35),
    "DXT1": (27, None),
    "DXT3": (27, 31),
    "DXT5": (27, 47),
    "ATI1": (42, None),
    "ATI2": (48, None),
}


def test_every_texconv_format_but_yuv_and_xr_bias_is_offered():
    offered = [save_image.to_dxgi(f) for f in ALL_FORMATS]
    assert sorted(set(offered)) == sorted(set(TEXCONV_FORMATS) - NOT_OFFERED)
    legacy = [f for f in ALL_FORMATS if f in save_image.LEGACY_TO_DXGI]
    assert legacy == ["DXT1", "DXT3", "DXT5", "ATI1", "ATI2"]


def stored(dxgi: str) -> list[int]:
    """The BGRA channels a format keeps; single-channel formats are saved grey."""
    if dxgi == "A8_UNORM":
        return [3]
    if dxgi.startswith(("BC4", "R8_", "R16_", "R32_")):
        return [2]
    if dxgi.startswith(("BC5", "R8G8_U", "R8G8_S", "R16G16_", "R32G32_")):
        return [2, 1]
    if dxgi.startswith(
        ("BC6H", "R32G32B32_", "R11G11B10", "R9G9B9E5", "B5G6R5", "B8G8R8X8")
    ) or dxgi in ("R8G8_B8G8_UNORM", "G8R8_G8B8_UNORM"):
        return [0, 1, 2]
    return [0, 1, 2, 3]


def kind(dxgi: str) -> str:
    if dxgi.startswith("BC"):
        return "bc6h" if dxgi.startswith("BC6H") else "bc"
    if dxgi.endswith(("_UINT", "_SINT")):
        return "int"
    if dxgi in ("R10G10B10A2_UNORM", "R11G11B10_FLOAT", "R9G9B9E5_SHAREDEXP"):
        return dxgi
    if dxgi.startswith("R32") and dxgi.endswith("_FLOAT"):
        return "f32"
    if dxgi.startswith("R16"):
        number = dxgi.rsplit("_", maxsplit=1)[-1]
        return {"FLOAT": "f16", "UNORM": "u16", "SNORM": "s16"}[number]
    return "s8" if dxgi.endswith("_SNORM") else "u8"


def integer_range(dxgi: str) -> tuple[Any, Any]:
    if dxgi == "R10G10B10A2_UINT":
        return 0, np.array([1023, 1023, 1023, 3])
    bits = int(dxgi[1 : dxgi.index("G") if "G" in dxgi else dxgi.index("_")])
    if dxgi.endswith("_UINT"):
        return 0, 2**bits - 1
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def bgra(image: np.ndarray) -> np.ndarray:
    """Float32 BGRA of a read or a source image: uint8 / 255, grey to B=G=R, an
    absent alpha 1."""
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, -1)
    if image.shape[2] == 3:
        image = np.concatenate([image, np.ones_like(image[..., :1])], -1)
    return image


def psnr(a: np.ndarray, b: np.ndarray, peak: float) -> float:
    mse = float(np.mean((a.astype(np.float64) - b) ** 2))
    return np.inf if mse == 0 else float(10 * np.log10(peak * peak / mse))


def ulp(dxgi: str, want: np.ndarray) -> np.ndarray:
    """One unit in the last place of the stored float (rounding errs by half)."""
    if dxgi == "R9G9B9E5_SHAREDEXP":  # 9-bit mantissas, the largest channel's exponent
        top = np.maximum(want[..., :3].max(-1, keepdims=True), 2.0**-16)
        return np.broadcast_to(2.0 ** (np.floor(np.log2(top)) + 1 - 9), want.shape)
    mantissa = np.array([5, 6, 6, 0]) if dxgi == "R11G11B10_FLOAT" else 10
    exponent = np.maximum(np.floor(np.log2(np.maximum(np.abs(want), 2.0**-30))), -14)
    return 2.0 ** (exponent - mantissa)


def expected(dxgi: str, want: np.ndarray) -> tuple[np.ndarray, Any]:
    """The values an uncompressed format stores for float BGRA `want`, and how far a
    read may be from them: half a quantum of the depth, or one float ulp."""
    k = kind(dxgi)
    if k in ("u8", "s8"):
        want = bgra(to_uint8(want, normalized=True))
        if k == "s8":  # 0..1 mapped to -1..1: half of 2/254
            return want, 1 / 254
        bits = {"B5G5R5A1_UNORM": [5, 5, 5, 1], "B5G6R5_UNORM": [5, 6, 5, 8]}.get(
            dxgi, [4] * 4 if "4G4" in dxgi else [8] * 4
        )
        # Below 8 bits texconv keeps the top bits of the 8-bit values (224 -> 28 of
        # 31, as in the PNG era): up to one quantum, then the 8-bit read's half step.
        depth = np.array(bits)
        return want, np.where(depth < 8, 1 / (2.0**depth - 1) + 0.5 / 255, 0)
    if k == "u16":
        return to_uint16(want, normalized=True) / np.float32(65535), 0
    if k == "s16":
        return np.clip(want, -1, 1), 0.5 / 32767
    if k == "R10G10B10A2_UNORM":
        # texconv truncates here too (up to one quantum)
        return np.clip(want, 0, 1), 1 / np.array([1023, 1023, 1023, 3])
    if k == "int":
        low, high = integer_range(dxgi)
        return np.clip(np.round(want), low, high), 0
    if k == "f32":
        return want, 0
    if dxgi in ("R11G11B10_FLOAT", "R9G9B9E5_SHAREDEXP"):  # unsigned floats
        want = np.maximum(want, 0)
    return want, ulp(dxgi, want)


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_round_trip_and_header_of_every_format(tmp_path, runs, fmt):
    # (iii) Values above 1 and below 0 reach the formats that keep them; compressed
    # formats get a smooth image (noise is what block compression cannot keep).
    dxgi = save_image.to_dxgi(fmt)
    channels, k = stored(dxgi), kind(dxgi)
    base = synthetic(16, 24, 4, 0.0 if k.startswith("bc") else 0.03)
    if k.startswith("bc"):  # BC1's 1-bit alpha would black out the colour below 0.5
        ramp = np.linspace(0.1, 1, 24, dtype=np.float32)
        base[..., 3] = 1 if dxgi.startswith("BC1") else ramp
    if k not in ("u8", "s8", "u16", "bc", "int"):
        base = base * np.float32(2.5) - np.float32(0.5)
    if k == "int":
        base = np.round(base * 290 - 40)
    if dxgi in ("R8G8_B8G8_UNORM", "G8R8_G8B8_UNORM"):  # pairs share red and blue
        base[:, 1::2, 0::2] = base[:, ::2, 0::2]
    source = base[..., :3].mean(-1) if channels == [2] else base
    if k == "int":
        # Integer values: texconv truncates fractions for 32-bit integer formats and
        # rounds them half to even for 8- and 16-bit ones.
        source = np.round(source)
    path = save(tmp_path, "image", source, fmt, dds_mipmap_levels=1)
    back, want = bgra(read(path)), bgra(source)
    if k == "bc":
        want = bgra(to_uint8(want, normalized=True))
        floor, alpha_floor = PSNR_FLOORS[fmt[:4].rstrip("_")]
        colour = [c for c in channels if c != 3]
        assert psnr(back[..., colour], want[..., colour], 1) >= floor
        if alpha_floor is not None:
            assert psnr(back[..., 3], want[..., 3], 1) >= alpha_floor
    elif k == "bc6h":
        want = np.maximum(want, 0) if dxgi == "BC6H_UF16" else want
        peak = float(want[..., :3].max() - want[..., :3].min())
        assert psnr(back[..., :3], want[..., :3], peak) >= PSNR_FLOORS["BC6H"][0]
    else:
        want, tolerance = expected(dxgi, want)
        limit = np.broadcast_to(np.asarray(tolerance, np.float64) + 1e-7, want.shape)
        error = np.abs(back - want)[..., channels]
        assert (error <= limit[..., channels]).all(), float(error.max())

    # (iv) The header follows the rule (legacy and PREFER_DX9 formats DX9, every
    # other DX10) and texconv reads the file as the format chosen.
    data = path.read_bytes()
    assert data[:4] == b"DDS " and struct.unpack_from("<2I", data, 12) == (16, 24)
    fourcc = data[84:88]
    if fmt in save_image.LEGACY_TO_DXGI:
        assert fourcc == fmt.encode()
    elif fmt in save_image.PREFER_DX9:
        assert fourcc == b"\0\0\0\0"
    else:
        assert fourcc == b"DX10"
        assert struct.unpack_from("<I", data, 128)[0] == DXGI_CODES[dxgi]
    command = [
        str(TEXCONV),
        "-nologo",
        "-nogpu",
        "-ft",
        "dds",
        "-f",
        "R32G32B32A32_FLOAT",
    ]
    summary = subprocess.run(
        [*command, "-o", str(tmp_path), "-sx", "_summary", str(path)],
        capture_output=True,
        check=True,
    ).stdout.decode("latin-1")
    assert f"(24x16 {dxgi} 2D" in summary, summary  # one mip: no ",1"


@pytest.mark.parametrize(
    "fmt",
    [
        "DXT1",
        "DXT3",
        "DXT5",
        "BC5_UNORM",
        "BC7_UNORM",
        "R8G8B8A8_UNORM",
        "B8G8R8A8_UNORM",
    ],
)
def test_pillow_decodes_what_chainner_reads(tmp_path, runs, fmt):
    # (v) An independent decoder, Pillow's DDS plugin: block interpolation may round
    # one step apart; uncompressed pixels are equal.
    image = synthetic(16, 24, 4, 0.0)
    image[..., 3] = np.linspace(0.1, 1, 24, dtype=np.float32)
    path = save(tmp_path, "image", image, fmt, dds_mipmap_levels=1)
    ours = (bgra(read(path)) * 255).round()
    with Image.open(path) as im:
        theirs = np.array(im.convert("RGBA"), np.float32)[..., [2, 1, 0, 3]]
    channels = [2, 1] if fmt.startswith("BC5") else [0, 1, 2, 3]
    limit = 0 if fmt.startswith(("R8", "B8")) else 1
    assert np.abs(ours - theirs)[..., channels].max() <= limit


def srgb(values: np.ndarray, to_linear: bool) -> np.ndarray:
    a = np.abs(values.astype(np.float64))
    if to_linear:
        curve = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    else:
        curve = np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)
    return np.copysign(curve, values).astype(np.float32)


@pytest.mark.parametrize("conversion", ["SRGB_TO_LINEAR", "LINEAR_TO_SRGB"])
def test_colour_values_change_only_when_asked(tmp_path, runs, conversion):
    image = synthetic(8, 10, 4) * np.float32(1.5) - np.float32(0.25)
    option = getattr(save_image.DDSColourConversion, conversion)
    to_linear = conversion == "SRGB_TO_LINEAR"
    want = image.copy()
    want[..., :3] = srgb(image[..., :3], to_linear)  # alpha stays
    path = save(
        tmp_path, "f", image, "R32G32B32A32_FLOAT", dds_colour_conversion=option
    )
    np.testing.assert_allclose(read(path), want, rtol=1e-6, atol=1e-7)
    # sRGB formats are tags: Keep writes the 8-bit values themselves, a conversion
    # changes them before the 8 bits.
    unit = np.clip(image, 0, 1)
    converted = unit.copy()
    converted[..., :3] = srgb(unit[..., :3], to_linear)
    keep = save_image.DDSColourConversion.KEEP
    for name, value, result in [("keep", keep, unit), ("on", option, converted)]:
        path = save(
            tmp_path, name, unit, "R8G8B8A8_UNORM_SRGB", dds_colour_conversion=value
        )
        assert np.array_equal(read(path), to_uint8(result, normalized=True))


def test_premultiplied_alpha_headers_and_straight_reads(tmp_path, runs):
    image = synthetic(16, 24, 4, 0.0)
    image[..., 3] = np.linspace(0.25, 1, 24, dtype=np.float32)
    for fmt, fourcc in [("DXT3", b"DXT2"), ("DXT5", b"DXT4")]:
        path = save(tmp_path, fmt, image, fmt, dds_premultiply_alpha=True)
        assert path.read_bytes()[84:88] == fourcc
    # A DX9 header cannot say premultiplied, so R8G8B8A8 takes DX10 and its alpha
    # mode (DDS_ALPHA_MODE_PREMULTIPLIED); the reader makes it straight again.
    straight = save(tmp_path, "straight", image, "R8G8B8A8_UNORM")
    assert straight.read_bytes()[84:88] == b"\0\0\0\0"
    path = save(tmp_path, "pm", image, "R8G8B8A8_UNORM", dds_premultiply_alpha=True)
    data = path.read_bytes()
    assert data[84:88] == b"DX10" and struct.unpack_from("<I", data, 144)[0] == 2
    error = np.abs(bgra(read(path)) - bgra(read(straight)))
    assert error.max() <= 3 / 255  # premultiplied 8-bit colour divided by alpha >= 1/4


@pytest.mark.parametrize("gpu", [True, False])
def test_gpu_compression_is_texconv_default(tmp_path, monkeypatch, gpu):
    # Mocked texconv: on, texconv chooses (DirectCompute when present); off, -nogpu.
    recorded: list[list[str]] = []
    monkeypatch.setitem(
        vars(texconv), "__run_texconv", lambda args, error: recorded.append(args)
    )
    save(tmp_path, "g", synthetic(8, 8, 3), "BC7_UNORM", dds_gpu=gpu)
    assert ("-nogpu" in recorded[-1]) is not gpu


def rgba_mips(path: Path, count: int) -> list[np.ndarray]:
    """The mip levels of an R8G8B8A8_UNORM DDS (DX9 header), as written: RGBA bytes."""
    data = path.read_bytes()
    height, width = struct.unpack_from("<2I", data, 12)
    levels, offset = [], 128
    for mip in range(count):
        h, w = max(1, height >> mip), max(1, width >> mip)
        levels.append(np.frombuffer(data, np.uint8, h * w * 4, offset).reshape(h, w, 4))
        offset += h * w * 4
    return levels


def test_mip_levels_and_filter(tmp_path, runs):
    image = synthetic(32, 24, 4)
    path = save(tmp_path, "three", image, "R8G8B8A8_UNORM", dds_mip_levels=3)
    assert struct.unpack_from("<I", path.read_bytes(), 28)[0] == 3
    assert "-if" not in runs[-1]  # Default: texconv's own filter, as before
    path = save(tmp_path, "none", image, "R8G8B8A8_UNORM", dds_mipmap_levels=1)
    assert struct.unpack_from("<I", path.read_bytes(), 28)[0] == 1
    assert runs[-1][runs[-1].index("-m") + 1] == "1"  # No ignores the level count
    # A box filter halves a power-of-two image by 2x2 means (opaque: the filter's
    # handling of alpha is texconv's own).
    path = save(
        tmp_path,
        "box",
        image[..., :3],
        "R8G8B8A8_UNORM",
        dds_mip_filter=save_image.DDSMipFilter.BOX,
    )
    assert runs[-1][runs[-1].index("-if") + 1] == "BOX"
    top, half = rgba_mips(path, 2)
    means = top.reshape(16, 2, 12, 2, 4).astype(np.float64).mean(axis=(1, 3))
    assert np.abs(half - means).max() <= 0.5 + 1e-9


def test_load_dds_reads_every_mip_and_the_header(tmp_path, runs):
    path = save(tmp_path, "mips", synthetic(32, 24, 4), "R8G8B8A8_UNORM")
    levels = rgba_mips(path, 6)
    for mip, expected in enumerate(levels):
        image, *info = load_dds.load_dds_node(path, mip, 0, 0)
        assert np.array_equal(image, expected[..., [2, 1, 0, 3]])
        assert info == ["R8G8B8A8_UNORM", 24, 32, 6, 1, 1, False]
        assert runs[-1][runs[-1].index("-m") + 1] == ("6" if mip else "1")
    with pytest.raises(ValueError, match="Mip level 6 does not exist"):
        load_dds.load_dds_node(path, 6, 0, 0)
    for fmt, premultiply, name in [
        ("DXT1", False, "DXT1"),
        ("DXT5", True, "DXT4"),
        ("ATI2", False, "ATI2"),
        ("BC7_UNORM_SRGB", False, "BC7_UNORM_SRGB"),
    ]:
        written = save(
            tmp_path, fmt, synthetic(8, 8, 4), fmt, dds_premultiply_alpha=premultiply
        )
        assert texconv.read_dds_info(written).format == name


def raw_rgba_dds(
    path: Path, count: int, size: tuple[int, int], mips: int, kind: str = "array"
) -> list[list[np.ndarray]]:
    """Writes an R8G8B8A8_UNORM DDS (DX10 header) of random RGBA: `count` images of
    an array or a cubemap's faces, each with `mips` levels, or one volume of depth
    `count`. Returns the levels written by image and mip, a volume's (d, h, w, 4)."""
    height, width = size
    volume, cubemap = kind == "volume", kind == "cubemap"
    rng = np.random.default_rng(count * 100 + mips)
    header = bytearray(148)
    flags = 0x2100F | 0x800000 * volume  # DDSD_DEPTH for a volume
    struct.pack_into(
        "<4s7I",
        header,
        0,
        b"DDS ",
        124,
        flags,
        height,
        width,
        width * 4,
        count if volume else 1,
        mips,
    )
    struct.pack_into("<2I4s", header, 76, 32, 0x4, b"DX10")
    caps2 = 0xFE00 if cubemap else 0x200000 if volume else 0
    struct.pack_into("<2I", header, 108, 0x401008, caps2)
    array = 1 if volume else count // 6 if cubemap else count
    struct.pack_into(
        "<5I", header, 128, 28, 4 if volume else 3, 0x4 * cubemap, array, 0
    )
    levels = [
        [
            rng.integers(
                0,
                256,
                (max(1, count >> mip),) * volume
                + (max(1, height >> mip), max(1, width >> mip), 4),
                np.uint8,
            )
            for mip in range(mips)
        ]
        for _ in range(1 if volume else count)
    ]
    path.write_bytes(bytes(header) + b"".join(m.tobytes() for i in levels for m in i))
    return levels


def test_load_dds_reads_layers_cubemap_faces_and_volume_slices(tmp_path, runs):
    array = raw_rgba_dds(tmp_path / "array.dds", 3, (8, 12), 2)
    cube = raw_rgba_dds(tmp_path / "cube.dds", 6, (8, 8), 1, "cubemap")
    volume = raw_rgba_dds(tmp_path / "volume.dds", 4, (8, 8), 2, "volume")
    bgra_order = [2, 1, 0, 3]
    for layer in range(3):
        for mip in range(2):
            image, *info = load_dds.load_dds_node(tmp_path / "array.dds", mip, layer, 0)
            assert np.array_equal(image, array[layer][mip][..., bgra_order])
            assert info == ["R8G8B8A8_UNORM", 12, 8, 2, 3, 1, False]
    for face in range(6):
        image, *info = load_dds.load_dds_node(tmp_path / "cube.dds", 0, face, 0)
        assert np.array_equal(image, cube[face][0][..., bgra_order])
        assert info == ["R8G8B8A8_UNORM", 8, 8, 1, 6, 1, True]
    for mip, depth in [(0, 4), (1, 2)]:
        for depth_slice in range(depth):
            image, *info = load_dds.load_dds_node(
                tmp_path / "volume.dds", mip, 0, depth_slice
            )
            assert np.array_equal(image, volume[0][mip][depth_slice][..., bgra_order])
            assert info == ["R8G8B8A8_UNORM", 8, 8, 2, 1, 4, False]
    with pytest.raises(ValueError, match="Layer 6 does not exist"):
        load_dds.load_dds_node(tmp_path / "cube.dds", 0, 6, 0)
    with pytest.raises(ValueError, match="Slice 2 does not exist"):
        load_dds.load_dds_node(tmp_path / "volume.dds", 1, 0, 2)


def texture(
    tmp_path: Path, kind: str, images: list[np.ndarray], fmt: str, **options
) -> Path:
    """Save DDS Texture's collector with Save Image's DDS defaults, but no mips."""
    parameters = inspect.signature(save_dds_texture.save_dds_texture_node).parameters
    values = {k: SAVE_DEFAULTS[k] for k in parameters if k in SAVE_DEFAULTS}
    values.update({"dds_mipmap_levels": 1, "dds_format": fmt, **options})
    del values["relative_path"]
    texture_kind = save_dds_texture.TextureKind[kind.upper()]
    collector = save_dds_texture.save_dds_texture_node(
        None, tmp_path, None, kind, texture_kind, **values
    )
    for image in images:
        collector.on_iterate(image)
    collector.on_complete()
    return tmp_path / f"{kind}.dds"


def generated(path: Path) -> list[tuple[np.ndarray, int]]:
    """Load DDS Images' items for mip level 0."""
    items = []
    for item in load_dds_images.load_dds_images_node(path, 0).supplier():
        assert not isinstance(item, Exception)
        items.append(item)
    return items


def test_save_dds_texture_writes_arrays_cubemaps_and_volumes(tmp_path, runs):
    faces = [synthetic(8, 8, 4, 0.01 * (i + 1)) for i in range(6)]
    cube = texture(tmp_path, "cubemap", faces, "R8G8B8A8_UNORM")
    data = cube.read_bytes()
    assert struct.unpack_from("<I", data, 112)[0] & 0xFE00 == 0xFE00  # all 6 faces
    assert runs[-1][runs[-1].index("-f") + 2] == "-dx9"  # DX9 holds a cubemap
    for face, image in enumerate(faces):
        read_face, *info = load_dds.load_dds_node(cube, 0, face, 0)
        assert np.array_equal(read_face, to_uint8(image, normalized=True))
        assert info == ["R8G8B8A8_UNORM", 8, 8, 1, 6, 1, True]
    items = generated(cube)
    assert [index for _, index in items] == list(range(6))
    assert all(
        np.array_equal(i, to_uint8(f, normalized=True))
        for (i, _), f in zip(items, faces, strict=True)
    )

    # An array of more than one image takes DX10 (a DX9 header has no arrays), even
    # for a legacy format; mips are generated per layer.
    layers = [synthetic(16, 8, 3, 0.02 * (i + 1)) for i in range(3)]
    array = texture(tmp_path, "array", layers, "DXT1", dds_mipmap_levels=0)
    data = array.read_bytes()
    assert data[84:88] == b"DX10" and struct.unpack_from("<I", data, 140)[0] == 3
    for layer in range(3):
        image, *info = load_dds.load_dds_node(array, 2, layer, 0)
        assert image.shape == (4, 2, 3)
        assert info == ["BC1_UNORM", 8, 16, 5, 3, 1, False]

    # A volume's images are its depth slices, written at their depth (float16 here).
    slices = [synthetic(8, 12, 4) * np.float32(3) - np.float32(1) for _ in range(4)]
    volume = texture(tmp_path, "volume", slices, "R16G16B16A16_FLOAT")
    for depth_slice, image in enumerate(slices):
        read_slice, *info = load_dds.load_dds_node(volume, 0, 0, depth_slice)
        assert np.array_equal(read_slice, image.astype(np.float16).astype(np.float32))
        assert info == ["R16G16B16A16_FLOAT", 12, 8, 1, 1, 4, False]
    assert len(generated(volume)) == 4


def test_save_dds_texture_refuses_wrong_faces_and_sizes(tmp_path, runs):
    with pytest.raises(ValueError, match="A cubemap has 6 faces"):
        texture(tmp_path, "cubemap", [synthetic(8, 8, 3)] * 5, "BC1_UNORM")
    with pytest.raises(ValueError, match="must have one size"):
        texture(
            tmp_path, "array", [synthetic(8, 8, 3), synthetic(8, 4, 3)], "BC1_UNORM"
        )
