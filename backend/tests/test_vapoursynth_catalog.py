"""The VapourSynth catalog: the scan child (vs-jetpack, plugins, user functions), the
cached catalog file, Methods with their families and overlays, and stable input ids."""

from __future__ import annotations

import ast
import functools
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import pytest

from nodes.impl.vapoursynth import catalog
from nodes.impl.vapoursynth.catalog_types import Method, Param
from nodes.impl.vapoursynth.runner import VSScriptError

SMALL = Path(__file__).parent / "data" / "vapoursynth" / "catalog_small.json"

# The family names of the generated nodes (Task 4's FAMILIES).
FAMILY_NAMES = {
    "Load Clip",
    "Clip Properties",
    "Convert Format",
    "Resize",
    "Cut and Splice",
    "Planes",
    "Deinterlace",
    "Denoise",
    "Sharpen and Blur",
    "Anti-alias",
    "Dehalo",
    "Deband and Grain",
    "Masks",
    "Colour",
    "Expression",
    "User Functions",
    "Any Function",
}

GOOD_USER_FILE = '''
import vapoursynth as vs


def brighten(clip: vs.VideoNode, amount: float = 1.0) -> vs.VideoNode:
    """Brightens a clip.

    Args:
        amount: How much brighter.
    """
    return clip.std.Expr(f"x {amount} *")


def _helper(clip: vs.VideoNode) -> vs.VideoNode:
    return clip
'''


def by_id(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {fn["id"]: fn for fn in data["functions"]}


IMPORT = re.compile(r"^from ([\w.]+) import (\w+)$")


def resolve(import_line: str, expr: str) -> Any:
    """The value of an enum member's expression in a script with its entry's import
    line: a dotted name on the imported name, or a literal that needs no import."""
    first, *attributes = expr.split(".")
    if not first.isidentifier():
        return ast.literal_eval(expr)
    match = IMPORT.match(import_line)
    assert match is not None, import_line
    assert match[2] == first, (import_line, expr)
    imported = getattr(importlib.import_module(match[1]), match[2])
    return functools.reduce(getattr, attributes, imported)


def entry_named(enums: dict[str, Any], suffix: str) -> dict[str, Any]:
    (key,) = [key for key in enums if key.endswith(suffix)]
    return enums[key]


@pytest.fixture(scope="module")
def scanned(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One real scan: vs-jetpack, the plugins, a user folder with a good, a broken and
    a raising file, and user_modules.toml listing vs_colorfix (installed in the venv)."""
    pytest.importorskip("vapoursynth")
    # The scan pip-installs a listed module that is missing, into the Python running
    # the tests: only the test venv, which has vs_colorfix, may run it.
    if importlib.util.find_spec("vs_colorfix") is None:
        pytest.skip("vs_colorfix is not installed; the scan would install it here")
    storage = tmp_path_factory.mktemp("vapoursynth")
    user = storage / "user"
    user.mkdir()
    (user / "good.py").write_text(GOOD_USER_FILE, encoding="utf-8")
    (user / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    (user / "raising.py").write_text(
        'raise RuntimeError("this user file fails on import")\n', encoding="utf-8"
    )
    # A name the generated scripts bind (core = vs.core).
    (user / "core.py").write_text(GOOD_USER_FILE, encoding="utf-8")
    (storage / "user_modules.toml").write_text(
        'modules = ["vs_colorfix"]\n', encoding="utf-8"
    )
    path = storage / "catalog.json"
    assert catalog.ensure_catalog(storage) == path
    return json.loads(path.read_text(encoding="utf-8"))


def test_scan_lists_jetpack_bm3d_params(scanned: dict[str, Any]) -> None:
    assert scanned["schema"] == 1
    assert scanned["key"]["vsjetpack"] == "2.2.6"
    bm3d = by_id(scanned)["vsdenoise.bm3d"]
    assert (bm3d["source"], bm3d["module"], bm3d["call"], bm3d["kind"]) == (
        "jetpack",
        "vsdenoise",
        "bm3d",
        "function",
    )
    assert bm3d["import"] == "from vsdenoise import bm3d"
    params = {p["name"]: p for p in bm3d["params"]}
    assert list(params)[:4] == ["clip", "sigma", "tr", "refine"]
    assert params["clip"] == {
        "name": "clip",
        "annotation": "VideoNode",
        "default": None,
        "kind": "keyword",
    }
    assert params["sigma"] == {
        "name": "sigma",
        "annotation": "float | Sequence[float]",
        "default": "0.5",
        "kind": "keyword",
    }
    # An enum annotation names its "enums" key; its default is a member expression.
    assert params["profile"]["annotation"] == "vsdenoise.blockmatch.BM3D.Profile"
    assert params["profile"]["default"] == "bm3d.Profile.FAST"
    assert params["ref"]["annotation"] == "VideoNode | None"
    assert params["planes"]["annotation"] == "Planes"
    assert params["kwargs"]["kind"] == "var_keyword"
    assert bm3d["returns"] == "VideoNode"
    assert bm3d["doc"].startswith("Block-Matching and 3D filtering")
    assert "Args:" not in bm3d["doc"]
    assert bm3d["arg_docs"]["sigma"].startswith("Strength of denoising.")

    profile = scanned["enums"]["vsdenoise.blockmatch.BM3D.Profile"]
    assert profile["import"] == "from vsdenoise import bm3d"
    members = {m["name"]: m["expr"] for m in profile["members"]}
    assert members["HIGH"] == "bm3d.Profile.HIGH"
    assert resolve(profile["import"], members["HIGH"]).name == "HIGH"
    # A preset of the default's own class, without special cases.
    presets = scanned["enums"]["vsdenoise.mvtools.presets.MVToolsPreset"]
    assert presets["import"] == "from vsdenoise import MVToolsPreset"
    assert "MVToolsPreset.HQ_SAD" in {m["expr"] for m in presets["members"]}
    # Every member expression works with its entry's import line.
    for entry in scanned["enums"].values():
        for member in entry["members"]:
            resolve(entry["import"], member["expr"])


def test_scan_lists_bound_methods(scanned: dict[str, Any]) -> None:
    functions = by_id(scanned)
    nnedi3 = functions["vsaa.NNEDI3.deinterlace"]
    assert (nnedi3["kind"], nnedi3["call"], nnedi3["import"]) == (
        "bound",
        "NNEDI3(...).deinterlace",
        "from vsaa import NNEDI3",
    )
    init = {p["name"]: p for p in nnedi3["init_params"]}
    assert init["pscrn"]["default"] == "4"
    assert init["double_rate"]["kind"] == "keyword"
    # A parameter typed Never takes no value (a removed option kept for its error).
    assert "opencl" not in init
    assert nnedi3["params"][0]["name"] == "clip"

    scale = functions["vskernels.Mitchell.scale"]
    assert scale["call"] == "Mitchell(...).scale"
    assert {"width", "height", "linear"} <= {p["name"] for p in scale["params"]}
    # KernelLike takes a kernel's name as a string: no import.
    kernels = entry_named(scanned["enums"], ".KernelLike")
    assert kernels["import"] == ""
    assert {"name": "Mitchell", "expr": "'Mitchell'"} in kernels["members"]
    # An alias of VapourSynth's and vs-jetpack's enums offers vs-jetpack's.
    field_based = entry_named(scanned["enums"], ".FieldBasedLike")
    assert field_based["import"] == "from vstools import FieldBased"
    assert "FieldBased.TFF" in {m["expr"] for m in field_based["members"]}
    sources = functions["vssource.BestSource.source"]
    assert (sources["kind"], sources["params"][0]["name"]) == ("bound", "file")
    # A method behind two wrappers (inject_self around inject_kwargs_params).
    sobel = functions["vsmasktools.Sobel.edgemask"]
    assert sobel["call"] == "Sobel(...).edgemask"
    assert sobel["params"][0]["name"] == "clip"


def test_scan_lists_enum_members(scanned: dict[str, Any]) -> None:
    """An enum whose members filter a clip is one entry with its member names."""
    prefilter = by_id(scanned)["vsdenoise.Prefilter"]
    assert (prefilter["kind"], prefilter["call"], prefilter["import"]) == (
        "members",
        "Prefilter.<member>",
        "from vsdenoise import Prefilter",
    )
    assert {"MINBLUR", "DFTTEST"} <= set(prefilter["members"])
    assert [p["name"] for p in prefilter["params"]][:3] == [
        "clip",
        "planes",
        "full_range",
    ]
    # vstools' property enums are decorators, no filters.
    assert "vstools.Matrix" not in by_id(scanned)


def test_scan_lists_plugin_signature(scanned: dict[str, Any]) -> None:
    functions = by_id(scanned)
    merge = functions["std.Merge"]
    assert (merge["source"], merge["module"], merge["call"], merge["kind"]) == (
        "plugin",
        "std",
        "core.std.Merge",
        "plugin",
    )
    # The script's own `core` reaches plugins.
    assert merge["import"] == ""
    assert merge["params"] == [
        {
            "name": "clipa",
            "annotation": "VideoNode",
            "default": None,
            "kind": "keyword",
        },
        {
            "name": "clipb",
            "annotation": "VideoNode",
            "default": None,
            "kind": "keyword",
        },
        {
            "name": "weight",
            "annotation": "float | Sequence[float] | None",
            "default": "None",
            "kind": "keyword",
        },
    ]
    assert merge["returns"] == "VideoNode"
    # "any" in a signature takes further arguments by name.
    assert functions["std.SetFrameProps"]["params"][-1]["kind"] == "var_keyword"
    assert "cas.CAS" in functions


def test_plugin_signature_parser() -> None:
    from nodes.impl.vapoursynth import scan

    assert scan.plugin_params("clips:vnode[];mismatch:int:opt;func:func:opt;any") == [
        {
            "name": "clips",
            "annotation": "VideoNode | Sequence[VideoNode]",
            "default": None,
            "kind": "keyword",
        },
        {
            "name": "mismatch",
            "annotation": "int | None",
            "default": "None",
            "kind": "keyword",
        },
        {
            "name": "func",
            "annotation": "Callable[..., Any] | None",
            "default": "None",
            "kind": "keyword",
        },
        {"name": "kwargs", "annotation": "Any", "default": None, "kind": "var_keyword"},
    ]
    assert scan.plugin_returns("clip:vnode;") == "VideoNode"
    assert scan.plugin_returns("") == "None"


@pytest.mark.parametrize(
    ("returns", "many"),
    [
        ("VideoNode", False),
        ("list[VideoNode]", True),
        ("Sequence[vs.VideoNode]", True),
        # adg_mask: one clip, or a list when it gets several luma scalings.
        ("VideoNode | list[VideoNode]", False),
        ("vs.VideoNode | list[vs.VideoNode]", False),
    ],
)
def test_returns_many(returns: str, many: bool) -> None:
    assert catalog.returns_many(returns) is many


def test_scan_skips_broken_user_file(scanned: dict[str, Any]) -> None:
    functions = by_id(scanned)
    brighten = functions["good.brighten"]
    assert (brighten["source"], brighten["import"], brighten["call"]) == (
        "user",
        "from good import brighten",
        "brighten",
    )
    assert [p["name"] for p in brighten["params"]] == ["clip", "amount"]
    assert brighten["arg_docs"] == {"amount": "How much brighter."}
    assert "good._helper" not in functions
    errors = "\n".join(scanned["errors"])
    assert "broken.py" in errors
    assert "SyntaxError" in errors
    assert "raising.py" in errors
    assert "this user file fails on import" in errors
    assert "core.py: chaiNNer's scripts use the name core" in errors
    assert not any(
        fn["module"] in ("broken", "raising", "core") for fn in functions.values()
    )


def test_scan_skips_user_file_named_like_a_chainner_module(tmp_path: Path) -> None:
    """The scripts put chaiNNer's backend/src on sys.path before the user folder, so
    `import util` of a user's util.py would import chaiNNer's util. The scan runs as in
    chaiNNer: isolated, without backend/src on its sys.path."""
    from nodes.impl.vapoursynth import scan

    user = tmp_path / "user"
    user.mkdir()
    for name in ("util", "system", "mine"):
        (user / f"{name}.py").write_text(
            "def brighten(clip):\n    return clip\n", encoding="utf-8"
        )
    code = (
        "import importlib.util, json, pathlib\n"
        f"spec = importlib.util.spec_from_file_location('scan', {scan.__file__!r})\n"
        "scan = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(scan)\n"
        "errors = []\n"
        f"modules = scan._import_user_files(pathlib.Path({str(user)!r}), errors)\n"
        "print(json.dumps([sorted(modules), errors]))\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        timeout=60,
        check=True,
    )
    modules, errors = json.loads(done.stdout)
    assert modules == ["mine"]
    assert any("util.py" in error and "rename the file" in error for error in errors)
    assert any("system.py" in error for error in errors)


def test_scan_user_module_vs_colorfix(scanned: dict[str, Any]) -> None:
    average = by_id(scanned)["vs_colorfix.average"]
    assert (average["source"], average["import"], average["call"], average["kind"]) == (
        "user",
        "from vs_colorfix import average",
        "average",
        "function",
    )
    assert {"radius", "planes", "fast"} <= {p["name"] for p in average["params"]}
    assert scanned["key"]["user_modules"] == {"vs_colorfix": "2.3.0"}


def test_user_module_install_uses_chainner_pip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    assert catalog.pip_command()[-2:] == ["-m", "chainner_pip"]
    assert catalog.PIP_TIMEOUT == 900
    calls = tmp_path / "calls.txt"
    fake_pip = tmp_path / "fake_pip.py"
    fake_pip.write_text(
        "import sys, time\n"
        "with open(sys.argv[1], 'a', encoding='utf-8') as log:\n"
        "    log.write(' '.join(sys.argv[2:]) + '\\n')\n"
        "if 'chainner-test-hangs' in sys.argv:\n"
        "    time.sleep(60)\n"
        "sys.exit(1 if 'chainner-test-fails' in sys.argv else 0)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        catalog, "pip_command", lambda: [sys.executable, str(fake_pip), str(calls)]
    )
    monkeypatch.setattr(catalog, "PIP_TIMEOUT", 2)
    # pytest is installed: no pip run. A failed install, or one that takes longer
    # than PIP_TIMEOUT, skips only that module.
    start = time.monotonic()
    with caplog.at_level(logging.WARNING):
        catalog.install_user_modules(
            [
                "pytest",
                "chainner-test-hangs",
                "chainner-test-fails",
                "chainner-test-new",
            ]
        )
    assert 2 <= time.monotonic() - start < 30
    runs = [line.split() for line in calls.read_text(encoding="utf-8").splitlines()]
    # (Without the hanging pip's own line, which a slow start may not reach.)
    assert [run[:2] for run in runs if run[1] != "chainner-test-hangs"] == [
        ["install", "chainner-test-fails"],
        ["install", "chainner-test-new"],
    ]
    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "chainner-test-hangs took longer than 2 s" in messages


def test_family_rules() -> None:
    functions = by_id(json.loads(SMALL.read_text(encoding="utf-8")))
    # Module table.
    assert catalog.family_of(functions["vsdenoise.bm3d"], None) == "Denoise"
    assert catalog.family_of(functions["vsaa.NNEDI3.deinterlace"], None) == "Anti-alias"
    # The overlay's family wins; an overlay without one changes nothing.
    nnedi3 = functions["vsaa.NNEDI3.deinterlace"]
    assert catalog.family_of(nnedi3, {"family": "Deinterlace"}) == "Deinterlace"
    assert (
        catalog.family_of(functions["vsdenoise.bm3d"], {"label": "BM3D"}) == "Denoise"
    )
    # Function table for vstools and std; the rest of them go to Any Function.
    assert catalog.family_of(functions["vstools.depth"], None) == "Convert Format"
    assert catalog.family_of(functions["std.Merge"], None) == "Planes"
    assert catalog.family_of(functions["vstools.padder"], None) == "Any Function"
    # User functions, unless an overlay places them.
    assert catalog.family_of(functions["good.brighten"], None) == "User Functions"
    assert (
        catalog.family_of(functions["good.brighten"], {"family": "Colour"}) == "Colour"
    )
    # Also a user file named like a plugin namespace (resize.py, bs.py).
    for module in ("resize", "bs", "cas"):
        mine = {**functions["good.brighten"], "id": f"{module}.f", "module": module}
        assert catalog.family_of(mine, None) == "User Functions"
    other_plugin = {**functions["std.Merge"], "id": "foo.Bar", "module": "foo"}
    assert catalog.family_of(other_plugin, None) == "Any Function"
    tables = {**catalog.FUNCTION_FAMILIES, **catalog.MODULE_FAMILIES}
    assert set(tables.values()) <= FAMILY_NAMES


def test_input_ids_stable_and_unique() -> None:
    def raw(method: str, param: str) -> int:
        return zlib.crc32(f"Denoise/{method}/{param}".encode()) & 0x7FFFFFFF

    keys = [(f"vsdenoise.f{m}", f"p{p}") for m in range(40) for p in range(30)]
    # A real CRC-32 collision, and a hash below 1000.
    clash_a, clash_b = ("vsdenoise.m26602200", "sigma"), ("vsdenoise.m948942", "sigma")
    low = ("vsdenoise.m150026", "sigma")
    assert raw(*clash_a) == raw(*clash_b)
    assert raw(*low) == 344
    keys += [clash_b, low, clash_a]

    ids = catalog.input_ids("Denoise", keys)
    shuffled = keys[:]
    random.Random(7).shuffle(shuffled)
    assert catalog.input_ids("Denoise", shuffled) == ids
    assert catalog.input_ids("Denoise", keys) == ids
    assert set(ids) == set(keys)
    assert len(set(ids.values())) == len(keys)
    assert all(1000 <= i <= 0x7FFFFFFF for i in ids.values())
    assert ids[("vsdenoise.f0", "p0")] == raw("vsdenoise.f0", "p0")
    assert ids[low] == 1344
    # The sorted-earlier key keeps its hash; the later one rehashes with a suffix.
    assert ids[clash_a] == raw(*clash_a)
    assert ids[clash_b] == zlib.crc32(b"Denoise/vsdenoise.m948942/sigma#1") & 0x7FFFFFFF
    # Adding a method whose keys collide with none moves no existing id.
    more = catalog.input_ids("Denoise", [*keys, ("vsdenoise.zzz", "sigma")])
    assert {key: more[key] for key in keys} == ids


def test_missing_catalog_registers_io_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_scan(*args: object) -> None:
        raise AssertionError("no scan without VapourSynth")

    def not_installed() -> Path:
        raise VSScriptError("VapourSynth is not installed.")

    monkeypatch.setattr(catalog, "run_scan", no_scan)
    monkeypatch.delenv("CHAINNER_VS_DIR", raising=False)
    assert catalog.storage_dir() is None
    assert catalog.current_catalog() == catalog.Catalog([], {})

    monkeypatch.setenv("CHAINNER_VS_DIR", str(tmp_path))
    monkeypatch.setattr(catalog, "vspipe", not_installed)
    assert catalog.storage_dir() == tmp_path
    assert catalog.ensure_catalog(tmp_path) is None
    assert catalog.current_catalog() == catalog.Catalog([], {})
    assert not (tmp_path / "catalog.json").exists()


def test_ensure_catalog_rescans_on_key_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "vapoursynth"
    (storage / "user").mkdir(parents=True)
    user_file = storage / "user" / "mine.py"
    user_file.write_text("def mine(clip):\n    return clip\n", encoding="utf-8")
    plugins = tmp_path / "site" / "vapoursynth" / "plugins"
    plugins.mkdir(parents=True)
    monkeypatch.setattr(catalog, "vspipe", lambda: plugins.parent / "vspipe.exe")
    scans: list[Path] = []

    def fake_scan(out: Path, user_dir: Path, modules_file: Path) -> None:
        assert (user_dir, modules_file) == (
            storage / "user",
            storage / "user_modules.toml",
        )
        scans.append(out)
        out.write_text(SMALL.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(catalog, "run_scan", fake_scan)
    path = storage / "catalog.json"
    assert catalog.ensure_catalog(storage) == path
    assert len(scans) == 1
    key = json.loads(path.read_text(encoding="utf-8"))["key"]
    stat = user_file.stat()
    assert key["user_files"] == {"mine.py": [stat.st_mtime_ns, stat.st_size]}

    assert catalog.ensure_catalog(storage) == path
    assert len(scans) == 1
    os.utime(user_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert catalog.ensure_catalog(storage) == path
    (plugins / "new_plugin.dll").write_bytes(b"")
    assert catalog.ensure_catalog(storage) == path
    (storage / "user_modules.toml").write_text("modules = []\n", encoding="utf-8")
    assert catalog.ensure_catalog(storage) == path
    assert len(scans) == 4

    # A failed scan keeps the last catalog; without one there is none.
    def failed_scan(out: Path, user_dir: Path, modules_file: Path) -> None:
        raise RuntimeError("the scan failed")

    monkeypatch.setattr(catalog, "run_scan", failed_scan)
    user_file.unlink()
    assert catalog.ensure_catalog(storage) == path
    path.unlink()
    assert catalog.ensure_catalog(storage) is None


def test_ensure_catalog_retries_a_missing_user_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listed module whose install failed is installed again at the next start, and
    the catalog is rescanned once that install worked."""
    storage = tmp_path / "vapoursynth"
    storage.mkdir()
    (storage / "user_modules.toml").write_text(
        'modules = ["chainner-test-module"]\n', encoding="utf-8"
    )
    plugins = tmp_path / "site" / "vapoursynth" / "plugins"
    plugins.mkdir(parents=True)
    monkeypatch.setattr(catalog, "vspipe", lambda: plugins.parent / "vspipe.exe")
    scans: list[Path] = []

    def fake_scan(out: Path, user_dir: Path, modules_file: Path) -> None:
        scans.append(out)
        out.write_text(SMALL.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(catalog, "run_scan", fake_scan)
    installs: list[list[str]] = []
    installed: set[str] = set()
    online = False

    def fake_install(names: list[str]) -> None:
        installs.append(list(names))
        if online:
            installed.update(names)

    real_version = importlib.metadata.version
    monkeypatch.setattr(catalog, "install_user_modules", fake_install)
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "1.0" if name in installed else real_version(name),
    )
    path = storage / "catalog.json"
    # Offline: the install fails at each start; nothing changed, so no rescan.
    assert catalog.ensure_catalog(storage) == path
    assert catalog.ensure_catalog(storage) == path
    assert (len(installs), len(scans)) == (2, 1)
    # Online: the install works and the catalog lists the module.
    online = True
    assert catalog.ensure_catalog(storage) == path
    assert (len(installs), len(scans)) == (3, 2)
    key = json.loads(path.read_text(encoding="utf-8"))["key"]
    assert key["user_modules"] == {"chainner-test-module": "1.0"}
    # Installed: no pip run and no scan.
    assert catalog.ensure_catalog(storage) == path
    assert (len(installs), len(scans)) == (3, 2)


def methods_of(path: Path, overlays: list[Path]) -> dict[str, Method]:
    return {m.id: m for m in catalog.load_catalog(path, overlays).methods}


def test_load_methods_overlay(tmp_path: Path) -> None:
    plain = methods_of(SMALL, [])
    assert plain["vsdenoise.bm3d"] == Method(
        id="vsdenoise.bm3d",
        family="Denoise",
        label="Bm3d",
        # Module-qualified: no two imports of one script can shadow each other.
        call="vsdenoise.bm3d",
        kind="function",
        init_params=(),
        params=(
            Param("clip", "VideoNode", None, "keyword"),
            Param("sigma", "float | Sequence[float]", "0.5", "keyword"),
            Param("tr", "int | Sequence[int | None] | None", "None", "keyword"),
            Param("refine", "int", "1", "keyword"),
            Param(
                "profile",
                "vsdenoise.blockmatch.BM3D.Profile",
                "bm3d.Profile.FAST",
                "keyword",
            ),
            Param("ref", "VideoNode | None", "None", "keyword"),
            Param("planes", "Planes", "None", "keyword"),
            Param("kwargs", "Any", None, "var_keyword"),
        ),
        doc=(
            "Block-Matching and 3D filtering (BM3D) is a state-of-the-art algorithm"
            " for image denoising."
        ),
        arg_docs={
            "sigma": "Strength of denoising.",
            "ref": "A clip to be used as the basic estimate.",
        },
        # The first three non-clip parameters, and those without a default.
        simple=frozenset({"clip", "sigma", "tr", "refine"}),
        import_line="import vsdenoise",
    )
    assert (
        catalog.load_catalog(SMALL, []).enums
        == json.loads(SMALL.read_text(encoding="utf-8"))["enums"]
    )
    nnedi3 = plain["vsaa.NNEDI3.deinterlace"]
    assert (nnedi3.label, nnedi3.family, nnedi3.call, nnedi3.import_line) == (
        "NNEDI3 Deinterlace",
        "Anti-alias",
        "vsaa.NNEDI3(...).deinterlace",
        "import vsaa",
    )
    # The method's own tff would repeat the constructor's: one name, one input.
    assert [p.name for p in nnedi3.init_params] == ["pscrn", "tff", "double_rate"]
    assert [p.name for p in nnedi3.params] == ["clip", "kwargs"]
    assert nnedi3.simple == frozenset({"clip", "pscrn", "tff", "double_rate"})
    merge = plain["std.Merge"]
    assert (merge.label, merge.family, merge.kind, merge.call) == (
        "Merge (std)",
        "Planes",
        "plugin",
        "core.std.Merge",
    )
    assert merge.import_line == ""
    # A user function's label names its module (two files may share a name).
    assert plain["good.brighten"].label == "Brighten (good)"

    shipped = tmp_path / "shipped.toml"
    shipped.write_text(
        '[method."vsdenoise.bm3d"]\n'
        'label = "BM3D"\n'
        'simple = ["sigma", "tr", "profile", "ref"]\n'
        'help = "Removes noise."\n'
        'hidden = ["refine", "clip"]\n'
        "\n"
        '[method."vsaa.NNEDI3.deinterlace"]\n'
        'family = "Deinterlace"\n',
        encoding="utf-8",
    )
    broken = tmp_path / "broken.toml"
    broken.write_text("[method.\n", encoding="utf-8")
    user = tmp_path / "user.toml"
    user.write_text('[method."vsdenoise.bm3d"]\nlabel = "My BM3D"\n', encoding="utf-8")
    methods = methods_of(SMALL, [shipped, broken, user])
    bm3d = methods["vsdenoise.bm3d"]
    assert bm3d.label == "My BM3D"
    # A method with an overlay is listed first in its node.
    assert bm3d.common and not plain["vsdenoise.bm3d"].common
    assert bm3d.doc == "Removes noise."
    assert bm3d.simple == frozenset({"sigma", "tr", "profile", "ref"})
    # Hidden parameters keep their defaults; one without a default stays.
    assert [p.name for p in bm3d.params] == [
        "clip",
        "sigma",
        "tr",
        "profile",
        "ref",
        "planes",
        "kwargs",
    ]
    assert methods["vsaa.NNEDI3.deinterlace"].family == "Deinterlace"
    assert methods["std.Merge"] == merge


def test_overlay_annotations_and_labels(tmp_path: Path) -> None:
    """Annotations replace a parameter's type, or add a keyword argument to a
    function that takes **kwargs; labels name the inputs."""
    overlay = tmp_path / "o.toml"
    overlay.write_text(
        '[method."vsdenoise.bm3d"]\n'
        'annotations = { profile = "int", _Matrix = "vstools.enums.color.MatrixLike" }\n'
        'labels = { sigma = "Strength" }\n'
        '[method."std.Merge"]\n'
        'annotations = { extra = "int" }\n',
        encoding="utf-8",
    )
    methods = methods_of(SMALL, [overlay])
    bm3d = methods["vsdenoise.bm3d"]
    params = {p.name: p for p in bm3d.params}
    assert params["profile"].annotation == "int"
    # Added before **kwargs, keeping the default (None: not passed).
    assert [p.name for p in bm3d.params][-2:] == ["_Matrix", "kwargs"]
    assert params["_Matrix"] == Param(
        "_Matrix", "vstools.enums.color.MatrixLike", "None", "keyword"
    )
    assert bm3d.labels == {"sigma": "Strength"}
    # std.Merge takes no **kwargs: a name it does not have is left out.
    assert [p.name for p in methods["std.Merge"].params] == ["clipa", "clipb", "weight"]


def test_overlay_keys_of_the_wrong_type_are_left_out(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A key of the wrong type, or a family that is no node, is logged and left out
    (an earlier overlay's value or the default stays); everything else loads."""
    first = tmp_path / "a.toml"
    first.write_text('[method."vsdenoise.bm3d"]\nlabel = "BM3D"\n', encoding="utf-8")
    second = tmp_path / "b.toml"
    second.write_text(
        '[method."vsdenoise.bm3d"]\n'
        "label = 5\n"
        'annotations = "int"\n'
        'simple = "sigma"\n'
        'labels = { sigma = "Strength", tr = 3 }\n'
        'family = "Denoiser"\n'
        'help = "Denoises."\n'
        '[method."std.Merge"]\n'
        "annotations = { weight = 5 }\n"
        'hidden = "weight"\n'
        "defaults = 1\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        methods = methods_of(SMALL, [first, second])
    plain = methods_of(SMALL, [])
    bm3d = methods["vsdenoise.bm3d"]
    assert (bm3d.label, bm3d.family, bm3d.doc) == ("BM3D", "Denoise", "Denoises.")
    assert (bm3d.params, bm3d.simple) == (
        plain["vsdenoise.bm3d"].params,
        plain["vsdenoise.bm3d"].simple,
    )
    assert bm3d.labels == {}
    merge = methods["std.Merge"]
    assert (merge.params, merge.defaults) == (plain["std.Merge"].params, {})
    assert methods.keys() == plain.keys()
    messages = "\n".join(r.getMessage() for r in caplog.records)
    for key in ("label", "annotations", "simple", "labels", "family"):
        assert f"Skipping {key} of vsdenoise.bm3d in the VapourSynth overlay" in (
            messages
        )
    for key in ("annotations", "hidden", "defaults"):
        assert f"Skipping {key} of std.Merge" in messages
    assert "b.toml" in messages


def test_toml_files_with_a_byte_order_mark(tmp_path: Path) -> None:
    """Notepad's "UTF-8 with BOM" and Windows PowerShell's Set-Content -Encoding UTF8
    start a file with a byte order mark, which tomllib refuses."""
    from nodes.impl.vapoursynth import scan

    overlay = tmp_path / "o.toml"
    overlay.write_text(
        '[method."vsdenoise.bm3d"]\nlabel = "BM3D"\n', encoding="utf-8-sig"
    )
    assert methods_of(SMALL, [overlay])["vsdenoise.bm3d"].label == "BM3D"
    modules = tmp_path / "user_modules.toml"
    modules.write_text('modules = ["vs-colorfix"]\n', encoding="utf-8-sig")
    assert scan.user_module_names(modules) == ["vs-colorfix"]


def test_overlay_defaults(tmp_path: Path) -> None:
    """Overlay defaults: the values the node's inputs start with."""
    overlay = tmp_path / "o.toml"
    overlay.write_text(
        '[method."vstools.depth"]\ndefaults = { bitdepth = 16 }\n', encoding="utf-8"
    )
    methods = methods_of(SMALL, [overlay])
    assert methods["vstools.depth"].defaults == {"bitdepth": 16}
    assert methods["vsdenoise.bm3d"].defaults == {}


def test_user_files_go_on_the_script_path(tmp_path: Path) -> None:
    """A function from the user folder needs that folder on the script's sys.path; a
    listed pip module (installed) does not."""
    storage = tmp_path / "vapoursynth"
    (storage / "user").mkdir(parents=True)
    path = storage / "catalog.json"
    path.write_text(SMALL.read_text(encoding="utf-8"), encoding="utf-8")
    assert methods_of(path, [])["good.brighten"].search_path == ""
    (storage / "user" / "good.py").write_text(GOOD_USER_FILE, encoding="utf-8")
    brighten = methods_of(path, [])["good.brighten"]
    assert brighten.search_path == str(storage / "user")
    assert (brighten.call, brighten.import_line) == ("good.brighten", "import good")
    assert methods_of(path, [])["vsdenoise.bm3d"].search_path == ""


def _bound(cls: str, method: str = "scale") -> dict[str, Any]:
    return {
        "id": f"vskernels.{cls}.{method}",
        "source": "jetpack",
        "module": "vskernels",
        "import": f"from vskernels import {cls}",
        "call": f"{cls}(...).{method}",
        "kind": "bound",
        "init_params": [],
        "params": [
            {
                "name": "clip",
                "annotation": "VideoNode",
                "default": None,
                "kind": "keyword",
            },
            {
                "name": "width",
                "annotation": "int | None",
                "default": "None",
                "kind": "keyword",
            },
        ],
        "returns": "VideoNode",
        "doc": f"{cls} {method}.",
        "arg_docs": {},
    }


def test_groups_collapse_classes_and_members(tmp_path: Path) -> None:
    """The same method of GROUP_MIN classes of one module is one method with a class
    dropdown (fewer classes stay apart); an enum of filters is one method."""
    data = json.loads(SMALL.read_text(encoding="utf-8"))
    kernels = ["Mitchell", "Catrom", "Hermite", "Lanczos", "Point"]
    data["functions"] += [_bound(cls) for cls in kernels]
    data["functions"] += [_bound(cls, "descale") for cls in kernels[:2]]
    data["functions"].append(
        {
            "id": "vsdenoise.Prefilter",
            "source": "jetpack",
            "module": "vsdenoise",
            "import": "from vsdenoise import Prefilter",
            "call": "Prefilter.<member>",
            "kind": "members",
            "members": ["MINBLUR", "DFTTEST"],
            "params": [
                {
                    "name": "clip",
                    "annotation": "VideoNode",
                    "default": None,
                    "kind": "keyword",
                }
            ],
            "returns": "VideoNode",
            "doc": "",
            "arg_docs": {},
        }
    )
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    overlay = tmp_path / "o.toml"
    overlay.write_text(
        '[method."vskernels.*.scale"]\n'
        'label = "Scale"\n'
        'class_default = "Catrom"\n'
        'labels = { class = "Kernel" }\n'
        'annotations = { width = "int" }\n'
        'defaults = { width = "1280" }\n'
        '[method."vsdenoise.Prefilter"]\n'
        'class_default = "NOPE"\n',
        encoding="utf-8",
    )
    methods = methods_of(path, [overlay])
    assert not any(m.startswith("vskernels.Mitchell.scale") for m in methods)
    scale = methods["vskernels.*.scale"]
    assert (scale.kind, scale.label, scale.family, scale.call) == (
        "group",
        "Scale",
        "Resize",
        "",
    )
    assert [m.choice for m in scale.members] == sorted(kernels)
    assert scale.default_choice == "Catrom"
    assert scale.labels == {catalog.CLASS: "Kernel"}
    assert catalog.CLASS in scale.simple
    mitchell = next(m for m in scale.members if m.choice == "Mitchell")
    assert (mitchell.id, mitchell.call, mitchell.import_line) == (
        "vskernels.Mitchell.scale",
        "vskernels.Mitchell(...).scale",
        "import vskernels",
    )
    # The group's overlay defaults reach every member.
    assert mitchell.defaults == {"width": "1280"}
    # The group's overlay annotations reach every member.
    assert {p.name: p.annotation for p in mitchell.params}["width"] == "int"
    # Two classes: two methods.
    assert {"vskernels.Mitchell.descale", "vskernels.Catrom.descale"} <= set(methods)
    prefilter = methods["vsdenoise.Prefilter"]
    assert prefilter.kind == "group"
    assert prefilter.labels[catalog.CLASS] == "Prefilter"
    assert [(m.choice, m.call, m.kind) for m in prefilter.members] == [
        ("MINBLUR", "vsdenoise.Prefilter.MINBLUR", "function"),
        ("DFTTEST", "vsdenoise.Prefilter.DFTTEST", "function"),
    ]
    assert prefilter.members[0].import_line == "import vsdenoise"
    # A preselected member the enum does not have is ignored.
    assert prefilter.default_choice == ""
    assert prefilter.family == "Denoise"
