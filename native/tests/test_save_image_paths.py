"""Save Image refuses a Subdirectory Path or Image Name that leaves the output
directory, by its text, before any join (native/include/save_path.hpp): a ".."
component (split on / and \\) or an absolute or rooted value (a drive on Windows).
"." components, ".." inside a name and separators in the name stay allowed. The
installed app joined such a value; Save DDS Texture shares the rule. Both executors
give the same error: the engine's Save Image declines the value to the Python body.
On Windows a component that is a device name (NUL, CON, COM1, ...) is refused too:
older Windows sends the file to the device, as texconv still does on Windows 11."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from engine_differential import KINDS, execute
from verify_runtime import edge, gradient, make_node

from api import Lazy
from chain.json import JsonNode, parse_json
from chain.optimize import optimize
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
    get_full_path,
    save_image_node,
)
from process import NodeExecutionError

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"
TOOLS = Path(__file__).resolve().parents[1] / "tools"
IMAGE = np.full((4, 6, 3), 0.5, np.float32)
LEAVING = [
    "../x",
    "..\\x",
    "a/../../x",
    "..",
    "a/..",
    "/x",
    "\\x",
    *(["C:\\x", "C:x", "D:/x"] if sys.platform == "win32" else []),
]
INSIDE = [
    ("sub/./x", "sub/x"),
    ("..foo", "..foo"),
    (".", "."),
    ("a..b/c", "a..b/c"),
    # Windows drops a path component's trailing dots and spaces.
    *([("a ", "a"), ("a. /b", "a/b"), ("a.", "a")] if sys.platform == "win32" else []),
]


def message(node: str, label: str, value: str) -> str:
    return (
        f'{node}: the {label} "{value}" leaves the output directory: ".." and absolute'
        " paths are not allowed; give a path inside it."
    )


def save(
    directory: Path, relative: str | None, filename: str, skip: bool = False
) -> None:
    save_image_node(
        Lazy.ready(IMAGE),
        directory,
        relative,
        filename,
        ImageFormat.PNG,
        PngColorDepth.U8,
        False,
        95,
        JpegSubsampling.FACTOR_422,
        False,
        TiffColorDepth.U8,
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
        AvifSubsampling.FACTOR_420,
        ExrColorDepth.F32,
        ExrCompression.ZIP,
        skip,
        None,
    )


def files(folder: Path) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.png"))


@pytest.mark.parametrize("value", LEAVING)
def test_a_subdirectory_path_that_leaves_is_refused(tmp_path: Path, value: str):
    out = tmp_path / "a" / "out"
    with pytest.raises(ValueError) as caught:
        save(out, value, "image")
    assert str(caught.value) == message("Save Image", "Subdirectory Path", value)
    assert files(tmp_path) == []


@pytest.mark.parametrize("value", LEAVING)
def test_an_image_name_that_leaves_is_refused(tmp_path: Path, value: str):
    out = tmp_path / "a" / "out"
    with pytest.raises(ValueError) as caught:
        save(out, None, value)
    assert str(caught.value) == message("Save Image", "Image Name", value)
    assert files(tmp_path) == []


@pytest.mark.parametrize(("value", "folder"), INSIDE)
def test_a_path_inside_stays_allowed(tmp_path: Path, value: str, folder: str):
    save(tmp_path, value, "image")
    expected = "image.png" if folder == "." else f"{folder}/image.png"
    assert files(tmp_path) == [expected]


def test_separators_and_dots_in_a_name_stay_allowed(tmp_path: Path):
    save(tmp_path, None, "sub/..image")
    assert files(tmp_path) == ["sub/..image.png"]


def test_save_dds_texture_names_itself_and_its_input(tmp_path: Path):
    with pytest.raises(ValueError) as caught:
        get_full_path(
            tmp_path,
            None,
            "../x",
            ImageFormat.DDS,
            "Save DDS Texture",
            "Texture Name",
        )
    assert str(caught.value) == message("Save DDS Texture", "Texture Name", "../x")


def save_chain(
    schemas, output: Path, relative: str | None, filename: str, skip: bool = False
):
    inputs: dict[int, object] = {0: edge("gradient"), 1: str(output), 2: relative, 3: filename, 4: "png"}
    if skip:
        inputs[1000] = True  # Skip existing files
    nodes = [
        gradient(schemas, "gradient", rgba=False),
        make_node(
            schemas,
            "save",
            "chainner:image:save",
            inputs,
        ),
    ]
    chain = parse_json(cast("list[JsonNode]", nodes))
    optimize(chain)
    return chain


@pytest.mark.parametrize(
    ("relative", "filename", "label", "value"),
    [
        ("../x", "image", "Subdirectory Path", "../x"),
        ("sub", "..\\image", "Image Name", "..\\image"),
    ],
)
def test_both_executors_refuse_alike(
    schemas, tmp_path: Path, relative: str, filename: str, label: str, value: str
):
    out = tmp_path / "out"
    for kind in KINDS:
        error = execute(
            kind, save_chain(schemas, out, relative, filename), storage=tmp_path
        ).error
        assert isinstance(error, NodeExecutionError), f"{kind}: {error!r}"
        assert error.node_id == "save"
        assert str(error) == message("Save Image", label, value), kind
    assert files(tmp_path) == []


def test_both_executors_save_inside(schemas, tmp_path: Path):
    for kind in KINDS:
        out = tmp_path / kind
        result = execute(
            kind, save_chain(schemas, out, "sub/./x", "..image"), storage=tmp_path
        )
        assert result.error is None, kind
        assert files(out) == ["sub/x/..image.png"]


@pytest.mark.skipif(
    sys.platform != "win32", reason="Windows drops trailing dots and spaces"
)
def test_a_folder_with_a_trailing_space_is_one_place_on_both_paths(
    schemas, tmp_path: Path
):
    for kind in KINDS:
        out = tmp_path / kind
        result = execute(kind, save_chain(schemas, out, "a /b. ", "c"), storage=tmp_path)
        assert result.error is None, f"{kind}: {result.error!r}"
        assert files(out) == ["a/b/c.png"]
    save(tmp_path / "body", "a /b. ", "c")
    assert files(tmp_path / "body") == ["a/b/c.png"]


WINDOWS_ONLY = pytest.mark.skipif(
    sys.platform != "win32", reason="device names are reserved on Windows alone"
)
# value: the component that is a device name
DEVICES = {
    "nul": "nul",
    "CON": "CON",
    "Com1": "Com1",
    "lpt9.x": "lpt9.x",
    "aux .tar": "aux .tar",
    "prn": "prn",
    "a/nul": "nul",
    "con\\b": "con",
    "a/b/LPT1.png": "LPT1.png",
    "conin$": "conin$",
    "CONOUT$.txt": "CONOUT$.txt",
    "COM¹": "COM¹",
    "a/lpt³.x": "lpt³.x",
}
NOT_DEVICES = [
    "nul_x",
    "con1",
    "com0",
    "lpt10",
    "connect",
    "a.nul",
    "auxiliary",
    "conin",
    "com¹x",
    "lpt⁴",
]


def device_message(node: str, label: str, value: str, part: str) -> str:
    if part == value:
        return (
            f'{node}: the {label} "{value}" is a reserved Windows device name; choose'
            " another name."
        )
    return (
        f'{node}: the {label} "{value}" contains the reserved Windows device name'
        f' "{part}"; choose another name.'
    )


@WINDOWS_ONLY
@pytest.mark.parametrize("value", DEVICES)
def test_a_device_name_is_refused(tmp_path: Path, value: str):
    for relative, filename, label in (
        (value, "image", "Subdirectory Path"),
        (None, value, "Image Name"),
    ):
        with pytest.raises(ValueError) as caught:
            save(tmp_path, relative, filename)
        assert str(caught.value) == device_message(
            "Save Image", label, value, DEVICES[value]
        )
    assert list(tmp_path.iterdir()) == []


@WINDOWS_ONLY
@pytest.mark.parametrize("value", NOT_DEVICES)
def test_a_name_that_only_looks_like_a_device_stays_allowed(tmp_path: Path, value: str):
    save(tmp_path, None, value)
    assert files(tmp_path) == [f"{value}.png"]


@pytest.mark.skipif(sys.platform == "win32", reason="Windows reserves device names")
def test_device_names_are_ordinary_names_elsewhere(tmp_path: Path):
    save(tmp_path, "con", "nul")
    assert files(tmp_path) == ["con/nul.png"]


@WINDOWS_ONLY
def test_save_dds_texture_refuses_a_device_name(tmp_path: Path):
    with pytest.raises(ValueError) as caught:
        get_full_path(
            tmp_path, "aux", "x", ImageFormat.DDS, "Save DDS Texture", "Texture Name"
        )
    assert str(caught.value) == device_message(
        "Save DDS Texture", "Subdirectory Path", "aux", "aux"
    )


@WINDOWS_ONLY
@pytest.mark.parametrize(
    ("relative", "filename", "label", "value", "part"),
    [
        ("sub", "nul", "Image Name", "nul", "nul"),
        ("a/COM3", "image", "Subdirectory Path", "a/COM3", "COM3"),
    ],
)
def test_both_executors_refuse_a_device_name_alike(
    schemas,
    tmp_path: Path,
    relative: str,
    filename: str,
    label: str,
    value: str,
    part: str,
):
    out = tmp_path / "out"
    for kind in KINDS:
        error = execute(
            kind, save_chain(schemas, out, relative, filename), storage=tmp_path
        ).error
        assert isinstance(error, NodeExecutionError), f"{kind}: {error!r}"
        assert error.node_id == "save"
        assert str(error) == device_message("Save Image", label, value, part), kind
    assert files(tmp_path) == []


def test_skip_existing_files_does_not_hide_a_folder_in_the_way(tmp_path: Path):
    # A folder named as the file: the write raises, with or without Skip existing files.
    (tmp_path / "x.png").mkdir()
    for skip in (False, True):
        with pytest.raises(PermissionError) as raised:
            save(tmp_path, None, "x", skip)
        assert raised.value.filename == str(tmp_path / "x.png"), skip


def test_both_executors_raise_for_a_folder_in_the_way_with_skip(
    schemas, tmp_path: Path
):
    for kind in KINDS:
        out = tmp_path / kind
        (out / "x.png").mkdir(parents=True)
        error = execute(
            kind, save_chain(schemas, out, None, "x", True), storage=tmp_path
        ).error
        assert isinstance(error, NodeExecutionError), f"{kind}: {error!r}"
        assert error.node_id == "save"


# Clears this process's long-path flag (PEB.IsLongPathAwareProcess, which
# RtlAreLongPathsEnabled reads), as on a PC where the setting is off, then saves a
# file whose full path passes MAX_PATH twice, the second time with Skip existing files.
LONG_PATH_CHILD = r"""
import ctypes, ctypes.wintypes as wt, sys
from pathlib import Path


class PBI(ctypes.Structure):
    _fields_ = [("r1", ctypes.c_void_p), ("peb", ctypes.c_void_p), ("r2", ctypes.c_void_p * 2),
                ("pid", ctypes.c_void_p), ("r3", ctypes.c_void_p)]


ntdll, kernel32 = ctypes.WinDLL("ntdll"), ctypes.WinDLL("kernel32")
kernel32.GetCurrentProcess.restype = wt.HANDLE
ntdll.NtQueryInformationProcess.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p]
ntdll.RtlAreLongPathsEnabled.restype = ctypes.c_ubyte
pbi = PBI()
assert ntdll.NtQueryInformationProcess(kernel32.GetCurrentProcess(), 0, ctypes.byref(pbi), ctypes.sizeof(pbi), None) == 0
ctypes.c_ubyte.from_address(pbi.peb + 3).value &= 0x7F
assert not ntdll.RtlAreLongPathsEnabled()
sys.path[:0] = sys.argv[2:]
import numpy as np
import test_save_image_paths as module

folder = Path(sys.argv[1])
name = "n" * (275 - len(str(folder)) - 5)
final = chr(92) * 2 + "?" + chr(92) + str(folder / f"{name}.png")
module.save(folder, None, name)
before = open(final, "rb").read()
module.IMAGE = np.full((4, 6, 3), 1.0, np.float32)
module.save(folder, None, name, True)
print("kept" if open(final, "rb").read() == before else "replaced")
"""


@pytest.mark.skipif(sys.platform != "win32", reason="MAX_PATH is a Windows limit")
def test_skip_existing_files_sees_a_file_past_max_path(tmp_path: Path):
    tests = Path(__file__).resolve().parent
    # A script file: texconv.py reads the main module's __file__ at import.
    child = tmp_path / "child.py"
    child.write_text(LONG_PATH_CHILD, encoding="utf-8")
    out = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, str(child), str(out), str(tests), str(BACKEND), str(TOOLS)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
        cwd=tests,
    )
    assert result.stdout.strip().splitlines()[-1:] == ["kept"], result.stderr[-2000:]
