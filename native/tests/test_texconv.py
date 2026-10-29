"""texconv's results for chaiNNer's command lines, against texconv.exe's: the DDS files
Save Image and Save DDS Texture write and the pixels the DDS readers return, as SHA-256
digests in golden/texconv.json (recorded on Windows from texconv.exe 2026.5.8.1 by
native/tools/texconv_golden.py). Windows runs the repository's texconv.exe, Linux
native/texconv; CPU codecs only (-nogpu). The inputs are integer patterns, the same
bytes on every system. On Linux, texconv also runs without its execute bit (an unzip
that drops file modes) and names itself when it cannot run."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from nodes.impl.dds import texconv
from nodes.impl.dds.format import LEGACY_TO_DXGI, PREFER_DX9, to_dxgi
from packages.chaiNNer_standard.image.io.save_image import SUPPORTED_DDS_FORMATS

ROOT = Path(__file__).resolve().parents[2]
TEXCONV = (
    ROOT
    / "backend/src/texconv"
    / ("texconv.exe" if platform.system() == "Windows" else "texconv")
)
GOLDEN = Path(__file__).with_name("golden") / "texconv.json"

pytestmark = pytest.mark.skipif(
    platform.system() not in ("Windows", "Linux"), reason="texconv: Windows and Linux"
)

FORMATS = [f for f, _ in SUPPORTED_DDS_FORMATS]
# The intermediate depth chaiNNer hands texconv (native/src/image_io.cpp, dds_depth).
UNORM16 = {"R16G16B16A16_UNORM", "R16G16B16A16_SNORM", "R16G16_UNORM", "R16G16_SNORM"}
UNORM16 |= {"R16_UNORM", "R16_SNORM"}
FLOAT = {f for f in FORMATS if f.endswith(("_FLOAT", "_UINT", "_SINT", "UF16", "SF16"))}
FLOAT |= {"R10G10B10A2_UNORM", "R9G9B9E5_SHAREDEXP"}


def depth(fmt: str) -> str:
    return "u16" if fmt in UNORM16 else "f32" if fmt in FLOAT else "u8"


@dataclass(frozen=True)
class Case:
    name: str
    fmt: str
    channels: int = 4
    size: tuple[int, int] = (24, 32)
    mips: int = 0
    kind: str = "array"
    count: int = 1
    options: dict[str, Any] = field(default_factory=dict)
    # A large grey case: a wrapping uint8 gradient, and no read back.
    large: bool = False


def pattern(case: Case, seed: int) -> np.ndarray:
    """Integer-derived pixels of the case's depth; float32 runs -0.5..2 (alpha 0..1)."""
    h, w = case.size
    if case.large:
        rows = (np.arange(h) % 256).astype(np.uint8)
        return rows[:, None] + (np.arange(w) % 256).astype(np.uint8)
    y, x = np.mgrid[0:h, 0:w]
    planes = [
        (x * (37 + 11 * c) + y * (53 + 7 * c) + seed * 101 + (x * y) % (17 + c) * 13)
        % 4096
        for c in range(max(case.channels, 3))
    ]
    if case.channels == 4:  # a ramp with a transparent corner
        alpha = x * 4095 // max(1, w - 1)
        alpha[(y > h // 2) & (x < w // 3)] = 0
        planes[3] = alpha
    v = np.stack(planes[: case.channels], -1)
    kind = depth(case.fmt)
    if kind == "u8":
        image = (v // 16).astype(np.uint8)
    elif kind == "u16":
        image = (v * 16 + v // 256).astype(np.uint16)
    else:
        image = v.astype(np.float32) / np.float32(4095)
        if case.channels != 4:
            image = image * np.float32(2.5) - np.float32(0.5)
        else:
            image[..., :3] = image[..., :3] * np.float32(2.5) - np.float32(0.5)
    return image[..., 0] if case.channels == 1 else image


def cases() -> list[Case]:
    found = [Case(f"{f}-mips", f) for f in FORMATS]
    found += [Case(f"{f}-odd", f, size=(18, 21), mips=1) for f in FORMATS]
    grey = ["BC1_UNORM", "BC4_UNORM", "BC5_UNORM", "BC7_UNORM", "R8_UNORM"]
    grey += ["R8G8B8A8_UNORM", "B5G6R5_UNORM", "R16_UNORM", "R32_FLOAT"]
    found += [Case(f"{f}-grey", f, 1, (18, 21)) for f in grey]
    opaque = ["BC1_UNORM", "BC3_UNORM", "BC7_UNORM_SRGB", "R8G8B8A8_UNORM"]
    found += [Case(f"{f}-rgb", f, 3, (18, 21)) for f in [*opaque, "B8G8R8X8_UNORM"]]
    for fmt, option, value in [
        ("BC7_UNORM", "minimal_compression", True),
        ("BC7_UNORM", "maximum_compression", True),
        ("BC1_UNORM", "dithering", True),
        ("BC1_UNORM", "uniform_weighting", True),
        ("BC3_UNORM", "dithering", True),
        ("BC3_UNORM", "uniform_weighting", True),
        ("R8G8B8A8_UNORM", "separate_alpha", True),
        ("BC7_UNORM_SRGB", "separate_alpha", True),
        ("R8G8B8A8_UNORM", "premultiply_alpha", True),
        ("DXT5", "premultiply_alpha", True),
    ]:
        found.append(
            Case(f"{fmt}-{option}", fmt, size=(18, 21), options={option: value})
        )
    for fmt in ("R8G8B8A8_UNORM", "BC1_UNORM", "R16G16B16A16_FLOAT"):
        for name in ("POINT", "LINEAR", "CUBIC", "FANT", "BOX", "TRIANGLE"):
            found.append(Case(f"{fmt}-{name}", fmt, options={"mip_filter": name}))
    found += [
        Case("cubemap", "R8G8B8A8_UNORM", size=(8, 8), kind="cubemap", count=6),
        Case("array", "DXT1", size=(16, 8), count=3),
        Case(
            "volume", "R16G16B16A16_FLOAT", size=(8, 12), mips=1, kind="volume", count=4
        ),
        # Wider than texconv's 16384 cap: scaled to 16384 x 12000 by WIC's Fant
        Case("capped", "R8_UNORM", 1, (12000, 16385), mips=1, large=True),
    ]
    return found


CASES = {case.name: case for case in cases()}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def pixels(image: np.ndarray) -> str:
    return digest(f"{image.dtype} {image.shape}".encode() + image.tobytes())


def run(case: Case, directory: Path) -> dict[str, str]:
    """Writes the case's DDS as Save Image and Save DDS Texture do, then reads it back:
    the file's digest and the pixels' of its top level (and level 1 with mips). A write
    texconv refuses (texconv.exe's box filter on a size that is not a power of 2) gives
    the error's first words."""
    images = [pattern(case, seed) for seed in range(case.count)]
    many = case.kind == "array" and case.count > 1
    premultiply = case.options.get("premultiply_alpha", False)
    dx9 = not many and (
        case.fmt in LEGACY_TO_DXGI or (case.fmt in PREFER_DX9 and not premultiply)
    )
    path = directory / f"{case.name}.dds"
    try:
        texconv.save_as_dds(
            path,
            images if case.count > 1 else images[0],
            to_dxgi(case.fmt),
            case.mips,
            dx9=dx9,
            gpu=False,
            kind=case.kind,
            **case.options,
        )
    except ValueError as error:
        return {"error": str(error).split(":")[0]}
    result = {"dds": digest(path.read_bytes())}
    if case.large:
        return result
    result["read"] = pixels(texconv.read_dds_texconv(path))
    if texconv.read_dds_info(path).mip_levels > 1:
        result["read_mip1"] = pixels(texconv.read_dds_texconv(path, 1))
    return result


def test_the_golden_file_covers_every_case():
    assert sorted(json.loads(GOLDEN.read_text())["cases"]) == sorted(CASES)


@pytest.mark.parametrize("name", list(CASES))
def test_texconv_gives_texconv_exe_results(tmp_path, monkeypatch, name):
    monkeypatch.setitem(vars(texconv), "__TEXCONV_EXE", str(TEXCONV))
    expected = json.loads(GOLDEN.read_text())["cases"][name]
    assert run(CASES[name], tmp_path) == expected


linux = pytest.mark.skipif(platform.system() != "Linux", reason="Linux's texconv")


@linux
def test_texconv_without_its_execute_bit_runs(tmp_path, monkeypatch):
    copy = tmp_path / "bin" / "texconv"
    copy.parent.mkdir()
    shutil.copyfile(TEXCONV, copy)  # mode 644, as some unzip tools leave it
    monkeypatch.setitem(vars(texconv), "__TEXCONV_EXE", str(copy))
    name = "R8G8B8A8_UNORM-odd"
    expected = json.loads(GOLDEN.read_text())["cases"][name]
    assert run(CASES[name], tmp_path) == expected
    assert os.access(copy, os.X_OK)


@linux
def test_a_texconv_that_cannot_run_is_named(tmp_path, monkeypatch):
    # As on a file system mounted noexec: the mode changes, the program still cannot run.
    copy = tmp_path / "texconv"
    shutil.copyfile(TEXCONV, copy)
    noexec = SimpleNamespace(**{**vars(os), "chmod": lambda path, mode: None})
    monkeypatch.setitem(vars(texconv), "os", noexec)
    monkeypatch.setitem(vars(texconv), "__TEXCONV_EXE", str(copy))
    with pytest.raises(ValueError, match="texconv cannot run") as raised:
        texconv.save_as_dds(
            tmp_path / "t.dds", np.zeros((8, 8), np.uint8), "R8_UNORM", 1
        )
    assert str(copy) in str(raised.value)
    assert isinstance(raised.value.__context__, PermissionError)
