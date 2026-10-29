"""Save Image refuses a Subdirectory Path or Image Name that leaves the output
directory, by its text, before any join (native/include/save_path.hpp): a ".."
component (split on / and \\) or an absolute or rooted value (a drive on Windows).
"." components, ".." inside a name and separators in the name stay allowed. The
installed app joined such a value; Save DDS Texture shares the rule. Both executors
give the same error: the engine's Save Image declines the value to the Python body.
On Windows a component that is a device name (NUL, CON, COM1, ...) is refused too:
older Windows sends the file to the device, as texconv still does on Windows 11."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from engine_differential import KINDS, execute, load_registry_errors
from verify_runtime import edge, gradient, make_node

import api
from api import Lazy
from chain.json import JsonNode, parse_json
from chain.optimize import optimize
from packages.chaiNNer_standard.image.io.save_image import (
    AvifSubsampling,
    BC7Compression,
    DDSColourConversion,
    DDSErrorMetric,
    DDSMipFilter,
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
INSIDE = [("sub/./x", "sub/x"), ("..foo", "..foo"), (".", "."), ("a..b/c", "a..b/c")]


def message(node: str, label: str, value: str) -> str:
    return (
        f'{node}: the {label} "{value}" leaves the output directory: ".." and absolute'
        " paths are not allowed; give a path inside it."
    )


def save(directory: Path, relative: str | None, filename: str) -> None:
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
        False,
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


@pytest.fixture(scope="module")
def schemas():
    main = sys.modules["__main__"]
    previous = getattr(main, "__file__", None)
    main.__file__ = str(BACKEND / "run.py")
    importlib.import_module("packages.chaiNNer_standard")
    errors = load_registry_errors(BACKEND / "server.py")
    assert not errors, [e.module for e in errors]
    yield {
        schema: {"inputs": [i.to_dict() for i in data.inputs], "kind": data.kind}
        for schema, (data, _) in api.registry.nodes.items()
    }
    if previous is None:
        del main.__file__
    else:
        main.__file__ = previous


def save_chain(schemas, output: Path, relative: str | None, filename: str):
    nodes = [
        gradient(schemas, "gradient", rgba=False),
        make_node(
            schemas,
            "save",
            "chainner:image:save",
            {0: edge("gradient"), 1: str(output), 2: relative, 3: filename, 4: "png"},
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
