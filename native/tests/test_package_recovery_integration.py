"""Exercise real package publication, refresh and recovery.

Tiny installed-app, provisioned-runtime and project trees live in tmp_path; git,
the reviewed UI patcher and the runtime's version probe are replaced by
fixtures. The runtime holds dist-info records like pip's, and the project a lock
like provision_runtime.ps1's.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import marshal
import os
import platform
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import independent_ui
import package_files
import package_manifest
import package_port
import pytest
from packaging.utils import canonicalize_name

UI_BUNDLE = "resources/app/main.js"
SITE = "Lib/site-packages/"
# The provisioned runtime's distributions as their RECORDs list them (relative to
# site-packages): the eleven the package leaves out, then the shipped ones.
EXCLUDED_RECORDS = {
    "pytest": [
        "_pytest/__init__.py",
        "pytest/__init__.py",
        "py.py",
        "__pycache__/py.cpython-314.pyc",
        "../../Scripts/pytest.exe",
        "../../Scripts/py.test.exe",
    ],
    "pytest-asyncio": ["pytest_asyncio/__init__.py"],
    "pytest-cov": ["pytest_cov/__init__.py"],
    "coverage": [
        "coverage/__init__.py",
        "a1_coverage.pth",
        "../../Scripts/coverage.exe",
    ],
    "pluggy": ["pluggy/__init__.py"],
    "iniconfig": ["iniconfig/__init__.py"],
    "Pygments": ["pygments/__init__.py", "../../Scripts/pygmentize.exe"],
    "pybind11": ["pybind11/__init__.py", "../../Scripts/pybind11-config.exe"],
    "google-re2": ["re2/__init__.py", "google_re2.libs/re2.dll"],
    "Sanic-Cors": ["sanic_cors/__init__.py", "sanic_cors/core.py"],
    # pynvml 13.0.1: its warning import hook (a .pth) and pynvml_utils, the hook
    # listed twice, once in a nested site-packages directory only it owns. The
    # pynvml module itself is nvidia-ml-py's.
    "pynvml": [
        "_pynvml_redirector.pth",
        "_pynvml_redirector.py",
        "__pycache__/_pynvml_redirector.cpython-314.pyc",
        "pynvml_utils/__init__.py",
        "site-packages/_pynvml_redirector.pth",
        "site-packages/_pynvml_redirector.py",
    ],
}
SHIPPED_RECORDS = {
    "pip": ["pip/__init__.py", "../../Scripts/pip.exe"],
    "chainner-pip": ["chainner_pip/__init__.py", "../../Scripts/chainner_pip.exe"],
    "colorama": ["colorama/__init__.py"],
    "typing-extensions": [
        "typing_extensions.py",
        "__pycache__/typing_extensions.cpython-314.pyc",
    ],
    "nvidia-ml-py": ["pynvml.py", "__pycache__/pynvml.cpython-314.pyc"],
}
VERSIONS = {
    "pip": "26.2.1",
    "chainner-pip": "23.2.0",
    "Sanic-Cors": "2.2.0",
    "pynvml": "13.0.1",
    "nvidia-ml-py": "13.615.71",
}
# Requirements as measured (Consult 6 P2): the dev set requires shared packages;
# a shipped one names pytest only under an extra nobody asks for.
REQUIRES = {
    "pytest": ["colorama>=0.4; sys_platform == 'win32'", "iniconfig", "pluggy"],
    "pytest-cov": ["coverage[toml]>=7", "pytest>=7"],
    "Sanic-Cors": ["sanic (>=21.9.3)", "packaging (>=21.3)"],
    "pynvml": ["nvidia-ml-py>=12.0.0", 'pytest>=3.6; extra == "test"'],
    "typing-extensions": ["pytest; extra == 'test'"],
}
# A stray cache in an excluded import directory, in no RECORD.
STRAY = SITE + "_pytest/__pycache__/stray.cpython-314.pyc"
# A shipped module in no RECORD, with the caches numba's cache=True leaves beside it
# (machine code for the build CPU) and a timestamp .pyc: none is copied (D-9).
NUMBA_MODULE = SITE + "pymatting/util/util.py"
SOURCE_CACHES = (
    SITE + "pymatting/util/__pycache__/util.boxfilter-58.py314.nbi",
    SITE + "pymatting/util/__pycache__/util.boxfilter-58.py314.1.nbc",
    SITE + "pymatting/util/__pycache__/util.cpython-314.pyc",
)
TAG = "cpython-314"
FIRST = "nodes/impl/first.py"
SECOND = "nodes/impl/second.py"
DLL = "nodes/impl/chainner_native.dll"
EXE = "texconv/texconv.exe"
BACKEND_FILES = {
    FIRST: b"def value():\n    return 1\n",
    SECOND: b"def value():\n    return 1\n",
    DLL: b"native library fixture",
    "nodes/impl/_chainner_graph.pyd": b"graph extension fixture",
    "chainner_ext/chainner_ext.pyd": b"chainner_ext extension fixture",
    EXE: b"tracked executable fixture",
}
# This repository, whose native toolchain pins each fixture project copies.
REPOSITORY = Path(__file__).resolve().parents[2]
NATIVE_TOOLCHAIN = {
    "clang_cl": "23.1.2",
    "msvc_toolset": "14.44.35207",
    "windows_sdk": "10.0.26100.0",
}
PROFILE = {
    "settings.json": b'{"hardwareAcceleration":false}',
    "Cache/Cache_Data/data_0": b"profile cache bytes\x00\xff",
    "backend-storage/dependencies.json": b"private package state",
}


def write(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def snapshot(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def packaged(name):
    return package_manifest.BACKEND + name


def cached(relative):
    """A Lib-relative source's .pyc path, as compileall names it."""
    source = PurePosixPath(relative)
    return (source.parent / "__pycache__" / f"{source.stem}.{TAG}.pyc").as_posix()


def fake_pyc(source, flags=1):
    """A .pyc as compileall writes one: the magic, the flags word (1:
    unchecked-hash), the source's hash, and a stand-in for the code."""
    head = importlib.util.MAGIC_NUMBER + flags.to_bytes(4, "little")
    return head + bytes.fromhex(package_files.sha(source)[:16]) + b"code " + source


def compile_lib(destination, flags=1, skip=()):
    """package_port.compile_runtime's effect: one .pyc per Lib .py, each written
    with flags; the sources in skip are left out."""
    lib = destination / package_manifest.RUNTIME / "Lib"
    for relative in package_files.walk(lib):
        if relative.endswith(".py") and relative not in skip:
            write(lib, cached(relative), fake_pyc((lib / relative).read_bytes(), flags))


def in_runtime(entry):
    """A RECORD entry (relative to site-packages) relative to the runtime root."""
    return os.path.normpath(SITE + entry).replace("\\", "/")


def install(runtime, name, files, version=None, requires=()):
    """Install name as pip would: its files, METADATA and a RECORD listing both.

    Returns the runtime-relative paths written.
    """
    version = version or VERSIONS.get(name, "1.0")
    info = f"{name.replace('-', '_')}-{version}.dist-info"
    metadata = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        f"Version: {version}",
        *(f"Requires-Dist: {requirement}" for requirement in requires),
    ]
    entries = [*files, f"{info}/METADATA", f"{info}/RECORD"]
    rows = []
    for entry in files:
        data = f"{name} {entry}".encode()
        write(runtime, in_runtime(entry), data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        rows.append(f"{entry},sha256={digest.rstrip(b'=').decode()},{len(data)}\n")
    write(runtime, f"{SITE}{info}/METADATA", "\n".join(metadata).encode() + b"\n")
    rows += [f"{info}/METADATA,,\n", f"{info}/RECORD,,\n"]
    write(runtime, f"{SITE}{info}/RECORD", "".join(rows).encode())
    return {in_runtime(entry) for entry in entries}


def write_lock(project, runtime, pip="26.2.1", interpreter="3.14.8", drop=()):
    """The lock provision_runtime.ps1 writes for runtime, less the drop fields."""
    header = {
        "interpreter": f"# Interpreter: CPython {interpreter} (standard build, not free-threaded).",
        "build_tag": "# Build tag: python-build-standalone 20261003",
        "archive_sha256": "# Archive SHA-256: " + "ab" * 32,
        "pip": f"# pip: {pip} (installed; pip freeze does not list it)",
    }
    pins = sorted(
        f"{dist.name}=={dist.version}"
        for name, dist in package_files.runtime_distributions(runtime).items()
        if name != "pip"
    )
    lines = [line for key, line in header.items() if key not in drop]
    write(project, package_manifest.LOCK, "\r\n".join([*lines, *pins, ""]).encode())


@pytest.fixture
def package_case(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    app = tmp_path / "installed-app"
    python = tmp_path / "runtime/cpython-3.14.8"
    destination = project / "out/package"
    backend = project / "backend/src"
    for name, data in BACKEND_FILES.items():
        write(backend, name, data)
    write(backend, "nodes/impl/__pycache__/first.cpython-311.pyc", b"project pyc")
    for name, _ in package_manifest.TOOLCHAIN_PINS.values():
        write(project, name, (REPOSITORY / name).read_bytes())
    write(app, "chaiNNer.exe", b"tiny executable fixture")
    write(app, "resources/app/package.json", b'{"version":"fixture-1"}')
    write(app, UI_BUNDLE, b"upstream fixture")
    write(app, "debug.log", b"installer log")
    write(app, "resources/src/" + FIRST, b"installed backend module")
    write(app, "resources/src/installed_only.py", b"installed-only module")
    write(app, "resources/src/nodes/impl/__pycache__/first.cpython-311.pyc", b"pyc")
    write(python, "python.exe", b"tiny Python fixture")
    write(python, "python314.dll", b"tiny Python DLL fixture")
    write(python, "Lib/dependency.py", b"original dependency")
    write(python, "Lib/__pycache__/dependency.cpython-314.pyc", b"runtime pyc")
    write(python, NUMBA_MODULE, b"numba-jitted module")
    for name in SOURCE_CACHES:
        write(python, name, b"cache for the build CPU")
    excluded = {STRAY}
    write(python, STRAY, b"stray cache")
    for name, files in EXCLUDED_RECORDS.items():
        excluded |= install(python, name, files, requires=REQUIRES.get(name, ()))
    for name, files in SHIPPED_RECORDS.items():
        install(python, name, files, requires=REQUIRES.get(name, ()))
    write_lock(project, python)
    case = SimpleNamespace(
        project=project,
        app=app,
        python=python,
        backend=backend,
        destination=destination,
        manifest=destination / package_manifest.MANIFEST,
        tracked={
            name for name in BACKEND_FILES if name not in package_files.NATIVE_BINARIES
        },
        converted=b"independent fixture",
        excluded=excluded,
        version=("3.14.8", 3, 14),
        compiles=[],
    )
    monkeypatch.setattr(package_port, "PROJECT", project)
    monkeypatch.setattr(package_port, "interpreter_version", lambda _: case.version)

    def compile_runtime(target, workers):
        case.compiles.append((target, workers))
        compile_lib(target)

    # The fixture's python.exe cannot run compileall; test_the_compile_step_*
    # run the real one.
    monkeypatch.setattr(package_port, "compile_runtime", compile_runtime)

    def git_snapshot(root):
        assert root == project
        provenance = {
            "git_head": "fixture-head",
            "git_branch": "fixture-branch",
            "backend_dirty": False,
        }
        return set(case.tracked), provenance

    def build_ui_overrides(source):
        assert source == app
        return {UI_BUNDLE: case.converted}, [
            {
                "destination": UI_BUNDLE,
                "installed_sha256": package_files.sha((app / UI_BUNDLE).read_bytes()),
                "converted_sha256": package_files.sha(case.converted),
                "reason": "fixture: no upstream updates",
            }
        ]

    monkeypatch.setattr(package_files, "git_snapshot", git_snapshot)
    monkeypatch.setattr(independent_ui, "build_ui_overrides", build_ui_overrides)
    arguments = [
        "package_port.py",
        "--installed-app",
        str(app),
        "--installed-python",
        str(python),
        "--destination",
        str(destination),
        "--copy-workers",
        "1",
    ]

    def run(*extra):
        """Run main(); success must print exactly one JSON line on stdout."""
        monkeypatch.setattr(sys, "argv", [*arguments, *extra])
        capsys.readouterr()
        assert package_port.main() == 0
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1
        return json.loads(lines[0])

    case.run = run
    return case


@contextmanager
def held_open(path, share):
    """Hold path open with the given share mode, like a running backend."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    generic_read, generic_write, open_existing = 0x80000000, 0x40000000, 3
    handle = kernel32.CreateFileW(
        str(path), generic_read | generic_write, share, None, open_existing, 0, None
    )
    if handle in (None, ctypes.c_void_p(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        kernel32.CloseHandle(handle)


def test_fresh_package_never_copies_installed_backend_or_bytecode(package_case):
    case = package_case
    app_before, python_before = snapshot(case.app), snapshot(case.python)
    result = case.run()
    assert result["result"] == "fresh"
    assert result["backend_files"] == len(BACKEND_FILES)
    manifest = package_manifest.load(case.manifest)
    assert manifest["schema"] == package_manifest.SCHEMA
    assert manifest["state"] == "complete"
    assert manifest["identity"] == {
        "project": str(case.project),
        "destination": str(case.destination),
        "installed_app": str(case.app),
        "installed_python": str(case.python),
    }
    assert manifest["provenance"]["git_head"] == "fixture-head"
    assert set(manifest["provenance"]) == {
        "utc",
        "git_head",
        "git_branch",
        "backend_dirty",
    }
    assert manifest["app_version"] == "fixture-1"
    assert manifest["upstream_updates"] == "disabled"
    # The UI bundle is recorded as a patch, never as a verbatim shell file.
    assert set(manifest["shell"]) == {"chaiNNer.exe", "resources/app/package.json"}
    assert [entry["destination"] for entry in manifest["ui_patches"]] == [UI_BUNDLE]
    assert manifest["backend"] == {
        packaged(name): {"sha256": package_files.sha(data), "bytes": len(data)}
        for name, data in BACKEND_FILES.items()
    }
    # The runtime less every file of the eleven excluded distributions: their
    # import directories (a stray cache in one included), dist-info, .pth and
    # entry points; Scripts keeps the shipped files, and the pynvml module ships
    # as nvidia-ml-py's. No __pycache__ of the source is copied (its .pyc and
    # numba's .nbi/.nbc); the build compiles one .pyc per Lib .py (D-9).
    caches = {name for name in python_before if "/__pycache__/" in "/" + name}
    assert set(SOURCE_CACHES) <= caches
    shipped = python_before.keys() - case.excluded - caches
    compiled = {
        "Lib/" + cached(name.removeprefix("Lib/"))
        for name in shipped
        if name.startswith("Lib/") and name.endswith(".py")
    }
    assert {
        "Lib/__pycache__/dependency.cpython-314.pyc",
        SITE + "__pycache__/typing_extensions.cpython-314.pyc",
        SITE + "pymatting/util/__pycache__/util.cpython-314.pyc",
    } <= compiled
    assert set(manifest["runtime"]) == {
        package_manifest.RUNTIME + name for name in shipped | compiled
    }
    assert {
        SITE + "pynvml.py",
        "Scripts/chainner_pip.exe",
        "Scripts/pip.exe",
    } <= shipped
    assert len(case.excluded) == 50
    runtime = case.destination / package_manifest.RUNTIME
    for name in compiled:
        module = PurePosixPath(name).name.split(".")[0]
        source = (runtime / name).parent.parent / (module + ".py")
        data = fake_pyc(source.read_bytes())
        assert (runtime / name).read_bytes() == data
        assert manifest["runtime"][package_manifest.RUNTIME + name] == {
            "sha256": package_files.sha(data),
            "bytes": len(data),
        }
    assert case.compiles == [(case.destination, 1)]
    assert manifest["python_stack"] == {
        "interpreter_version": "3.14.8",
        "build_tag": "20261003",
        "archive_sha256": "ab" * 32,
        "lock_sha256": package_files.sha(
            (case.project / package_manifest.LOCK).read_bytes().replace(b"\r\n", b"\n")
        ),
        "runtime": str(case.python),
        "excluded": {
            **dict.fromkeys(
                [
                    "pytest",
                    "pytest-asyncio",
                    "pytest-cov",
                    "coverage",
                    "pluggy",
                    "iniconfig",
                    "Pygments",
                    "pybind11",
                ],
                "dev-only",
            ),
            "google-re2": "oracle-only",
            "Sanic-Cors": "oracle-only",
            "pynvml": "oracle-only",
        },
        "chainner_pip_version": "23.2.0",
        "bytecode": "unchecked-hash",
    }
    assert manifest["native_toolchain"] == NATIVE_TOOLCHAIN
    files = snapshot(case.destination)
    assert {name for name in files if name.startswith(package_manifest.RUNTIME)} == {
        package_manifest.RUNTIME + name for name in shipped | compiled
    }
    # Neither the installed resources/src nor any __pycache__ (installed or
    # project) reaches the package; installer logs are skipped.
    assert {name for name in files if name.startswith("resources/src/")} == set(
        manifest["backend"]
    )
    assert files[packaged(FIRST)] == BACKEND_FILES[FIRST]
    assert files[UI_BUNDLE] == case.converted
    assert files["portable"] == b""
    assert "debug.log" not in files
    assert snapshot(case.app) == app_before
    assert snapshot(case.python) == python_before


def test_repeat_run_is_verified_no_op_with_identical_manifest(package_case):
    case = package_case
    case.run()
    # Bytecode the packaged backend writes at runtime is not package content.
    write(case.destination, "resources/src/nodes/impl/__pycache__/x.pyc", b"runtime")
    before = snapshot(case.destination)
    manifest_bytes = case.manifest.read_bytes()
    result = case.run()
    assert result["result"] == "verified_no_op"
    assert result["written_files"] == 0
    assert case.manifest.read_bytes() == manifest_bytes
    assert snapshot(case.destination) == before


def test_validate_only_writes_nothing(package_case):
    case = package_case
    result = case.run("--validate-only")
    assert result["validation_only"] is True
    assert result["package_modified"] is False
    assert (result["shell_files"], result["ui_patches"]) == (2, 1)
    assert result["backend_files"] == len(BACKEND_FILES)
    assert result["git_head"] == "fixture-head"
    assert not case.destination.exists()
    case.run()
    manifest = package_manifest.load(case.manifest)
    before = snapshot(case.destination)
    again = case.run("--validate-only")
    assert snapshot(case.destination) == before
    # The printed digests and records are those a build writes.
    assert again["shell_sha256"] == package_port.digest(manifest["shell"])
    assert again["ui_patches_sha256"] == package_port.digest(manifest["ui_patches"])
    assert again["backend_sha256"] == package_port.digest(manifest["backend"])
    runtime = manifest["runtime"]
    copied = [key for key in runtime if not package_manifest.runtime_bytecode(key)]
    assert again["runtime_files"] == len(copied)
    assert again["python_stack"] == manifest["python_stack"]
    assert again["native_toolchain"] == manifest["native_toolchain"]


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("untracked", "untracked: ['nodes/stray.txt']"),
        ("missing", "missing: ['nodes/impl/missing.py']"),
        ("binary", "missing: ['nodes/impl/_chainner_graph.pyd']"),
    ),
)
def test_backend_must_equal_git_files_plus_native_binaries(
    package_case, change, message
):
    case = package_case
    if change == "untracked":
        write(case.backend, "nodes/stray.txt", b"editor leftover")
    elif change == "missing":
        case.tracked.add("nodes/impl/missing.py")
    else:
        (case.backend / "nodes/impl/_chainner_graph.pyd").unlink()
    with pytest.raises(ValueError, match=re.escape(message)):
        case.run()
    assert not case.destination.exists()


@pytest.mark.parametrize(
    ("relative", "data"),
    (
        ("chaiNNer.exe", b"modified shell"),
        (UI_BUNDLE, b"modified bundle"),
        (packaged(FIRST), b"modified backend"),
        (packaged("nodes/impl/extra.py"), b"unrecorded backend"),
    ),
)
def test_refresh_refuses_a_package_changed_outside_the_tool(
    package_case, relative, data
):
    case = package_case
    case.run()
    write(case.destination, relative, data)
    write(case.backend, FIRST, b"def value():\n    return 4\n")
    before = snapshot(case.destination)
    with pytest.raises(ValueError, match="changed outside the package tool"):
        case.run()
    assert snapshot(case.destination) == before


def test_refresh_interruption_recovers_without_touching_profile_or_runtime(
    package_case, monkeypatch
):
    case = package_case
    app_before, python_before = snapshot(case.app), snapshot(case.python)
    case.run()
    completed = package_manifest.load(case.manifest)
    for name, data in PROFILE.items():
        write(case.destination, name, data)
    # A refresh must not recopy the Python tree over this portable installation's
    # own dependency state. The installed source tree remains independent.
    dependency = write(
        case.destination, "python/python/Lib/dependency.py", b"portable dependency"
    )
    for name in (FIRST, SECOND):
        write(case.backend, name, b"def value():\n    return 2\n")
    first, second = (case.destination / packaged(name) for name in (FIRST, SECOND))
    previous_second = second.read_bytes()
    real_replace = package_files.replace_verified
    calls = []

    def interrupt_second_write(target, data, sha256):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("simulated interruption")
        real_replace(target, data, sha256)

    with monkeypatch.context() as patch:
        patch.setattr(package_files, "replace_verified", interrupt_second_write)
        with pytest.raises(OSError, match="simulated interruption"):
            case.run()
    assert calls == [first, second]
    assert package_manifest.load(case.manifest) == {**completed, "state": "copying"}
    assert first.read_bytes() == b"def value():\n    return 2\n"
    assert second.read_bytes() == previous_second

    def forbid_runtime_recopy(_pair):
        pytest.fail("Interrupted refresh must not recopy the shell or Python")

    monkeypatch.setattr(package_files, "copy_verified", forbid_runtime_recopy)
    assert case.run()["result"] == "recovered"
    assert len(case.compiles) == 1  # A refresh never recompiles the runtime.
    recovered = package_manifest.load(case.manifest)
    assert recovered["state"] == "complete"
    assert recovered["shell"] == completed["shell"]
    assert recovered["runtime"] == completed["runtime"]
    for key, record in recovered["backend"].items():
        target = case.destination / key
        source = case.backend / key.removeprefix(package_manifest.BACKEND)
        assert target.read_bytes() == source.read_bytes()
        assert package_files.hash_file(target) == record["sha256"]
    owned = recovered["shell"].keys() | recovered["backend"].keys()
    for name, data in PROFILE.items():
        assert (case.destination / name).read_bytes() == data
        assert name not in owned | recovered["runtime"].keys()
    assert dependency.read_bytes() == b"portable dependency"
    assert snapshot(case.app) == app_before
    assert snapshot(case.python) == python_before
    before_noop = snapshot(case.destination)
    assert case.run()["result"] == "verified_no_op"
    assert snapshot(case.destination) == before_noop


@pytest.mark.skipif(os.name != "nt", reason="Windows directory-junction guard")
def test_refresh_recovery_rejects_profile_junction_before_any_write(package_case):
    import _winapi

    case = package_case
    case.run()
    manifest = package_manifest.load(case.manifest)
    package_manifest.write_json(case.manifest, {**manifest, "state": "copying"})
    before = snapshot(case.destination)
    outside = case.project.parent / "outside-profile"
    write(outside, "data_0", b"must stay outside the package")
    junction = case.destination / "Cache"
    # Windows junction creation requires no symlink/developer-mode privilege.
    _winapi.CreateJunction(str(outside), str(junction))
    try:
        with pytest.raises(ValueError, match="Refusing reparse point"):
            case.run()
        assert (outside / "data_0").read_bytes() == b"must stay outside the package"
    finally:
        junction.rmdir()
    assert snapshot(case.destination) == before


def test_changed_backend_files_are_replaced_and_verified_binaries_first(
    package_case, monkeypatch
):
    case = package_case
    case.run()
    for name, data in PROFILE.items():
        write(case.destination, name, data)
    changed = {
        FIRST: b"def value():\n    return 3\n",
        DLL: b"rebuilt native library",
        EXE: b"updated executable",
    }
    for name, data in changed.items():
        write(case.backend, name, data)
    real_replace = package_files.replace_verified
    order = []

    def record_order(target, data, sha256):
        order.append(target.relative_to(case.destination).as_posix())
        real_replace(target, data, sha256)

    def forbid_copy(_pair):
        pytest.fail("A refresh must not copy installed files")

    monkeypatch.setattr(package_files, "replace_verified", record_order)
    monkeypatch.setattr(package_files, "copy_verified", forbid_copy)
    result = case.run()
    assert (result["result"], result["written_files"]) == ("refreshed", 3)
    assert order == [packaged(DLL), packaged(EXE), packaged(FIRST)]
    manifest = package_manifest.load(case.manifest)
    assert manifest["state"] == "complete"
    for name, data in changed.items():
        assert (case.destination / packaged(name)).read_bytes() == data
        assert manifest["backend"][packaged(name)]["sha256"] == package_files.sha(data)
    for name, data in PROFILE.items():
        assert (case.destination / name).read_bytes() == data


def test_removed_backend_file_is_deleted(package_case):
    case = package_case
    case.run()
    (case.backend / SECOND).unlink()
    case.tracked.discard(SECOND)
    result = case.run()
    assert (result["result"], result["written_files"]) == ("refreshed", 1)
    assert not (case.destination / packaged(SECOND)).exists()
    assert packaged(SECOND) not in package_manifest.load(case.manifest)["backend"]
    assert case.run()["result"] == "verified_no_op"


def test_removed_backend_file_differing_from_its_record_is_kept(package_case):
    case = package_case
    case.run()
    manifest = package_manifest.load(case.manifest)
    # An interrupted refresh skips the up-front guard; deletion still checks.
    package_manifest.write_json(case.manifest, {**manifest, "state": "copying"})
    (case.backend / SECOND).unlink()
    case.tracked.discard(SECOND)
    target = write(case.destination, packaged(SECOND), b"edited by hand")
    pending = case.manifest.read_bytes()
    with pytest.raises(ValueError, match="differs from its record"):
        case.run()
    assert target.read_bytes() == b"edited by hand"
    assert case.manifest.read_bytes() == pending


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing modes")
@pytest.mark.parametrize(
    ("share", "error"),
    (
        # FILE_SHARE_READ: readable but not writable, like a mapped DLL; the
        # write probe reports it.
        (0x1, RuntimeError),
        # No sharing: even the up-front hash verification cannot read it.
        (0x0, PermissionError),
    ),
)
def test_lock_preflight_fails_before_any_write(package_case, share, error):
    case = package_case
    case.run()
    write(case.backend, FIRST, b"def value():\n    return 5\n")
    write(case.backend, DLL, b"rebuilt native library")
    target = case.destination / packaged(DLL)
    before = snapshot(case.destination)
    manifest_bytes = case.manifest.read_bytes()
    with held_open(target, share), pytest.raises(error) as raised:
        case.run()
    assert "chainner_native.dll" in str(raised.value)
    if error is RuntimeError:
        assert "locked or read-only" in str(raised.value)
        assert str(target) in str(raised.value)
    assert case.manifest.read_bytes() == manifest_bytes
    assert package_manifest.load(case.manifest)["state"] == "complete"
    assert snapshot(case.destination) == before


def test_cli_reports_failures_on_stderr_with_exit_code_1(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(package_port.__file__)),
            "--installed-app",
            str(tmp_path),
            "--installed-python",
            str(tmp_path),
            "--copy-workers",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr == "Package failed: copy-workers must be between 1 and 32\n"


def test_refresh_requires_an_existing_package(package_case):
    with pytest.raises(ValueError, match="empty or absent package"):
        package_case.run("--refresh")
    assert not package_case.destination.exists()


@pytest.mark.parametrize("child", ("out", "elsewhere/package"))
def test_destination_must_be_a_child_of_project_out(package_case, child):
    case = package_case
    with pytest.raises(ValueError, match="child directory"):
        case.run("--destination", str(case.project / child))
    assert not (case.project / "elsewhere").exists()


def test_the_excluded_distributions_and_their_reasons():
    assert package_files.EXCLUDED == {
        **dict.fromkeys(
            [
                "pytest",
                "pytest-asyncio",
                "pytest-cov",
                "coverage",
                "pluggy",
                "iniconfig",
                "Pygments",
                "pybind11",
            ],
            "dev-only",
        ),
        "google-re2": "oracle-only",
        "Sanic-Cors": "oracle-only",
        "pynvml": "oracle-only",
    }


def server_dependencies(project):
    """install_server_deps.py's DependencyInfo declarations, {package_name:
    version}. Importing the module would run its install, so it is parsed."""
    tree = ast.parse(
        (project / "backend/src/dependencies/install_server_deps.py").read_text(
            encoding="utf-8"
        )
    )
    declared = [
        {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DependencyInfo"
    ]
    return {entry["package_name"]: entry["version"] for entry in declared}


def test_the_hosts_server_dependencies_ship_at_the_locks_pins():
    """At start the packaged host pip-installs each declared server dependency
    that pip list does not show at its version or newer (dependencies.store), so
    every one is a shipped distribution at the lock's pin, under pip's name:
    nvidia-ml-py, which provides the pynvml module gpu.py imports, and never
    pynvml, which the runtime keeps for upstream's host (Consult 8 sweep item 4)."""
    project = Path(__file__).resolve().parents[2]
    declared = server_dependencies(project)
    pins = package_manifest.read_lock(project / package_manifest.LOCK)["pins"]
    assert "nvidia-ml-py" in declared
    assert "pynvml" not in declared
    assert {name: pins.get(name) for name in declared} == declared
    excluded = {canonicalize_name(name) for name in package_files.EXCLUDED}
    assert not {canonicalize_name(name) for name in declared} & excluded


def refused(case, message):
    """The run fails with message in both modes, before anything is written."""
    for mode in (("--validate-only",), ()):
        with pytest.raises(ValueError, match=message):
            case.run(*mode)
    assert not case.destination.exists()


@pytest.mark.parametrize(
    "name", ["pybind11", "google-re2", "Pygments", "Sanic-Cors", "pynvml"]
)
def test_every_excluded_distribution_must_be_in_the_runtime(package_case, name):
    case = package_case
    info = next((case.python / SITE).glob(f"{name.replace('-', '_')}-*.dist-info"))
    for path in info.iterdir():
        path.unlink()
    info.rmdir()
    write_lock(case.project, case.python)
    refused(case, re.escape(f"Excluded distributions missing from {case.python}"))


@pytest.mark.parametrize(
    ("requires", "requiring"),
    [
        (["pytest>=8"], "colorama: pytest>=8"),
        (["coverage; sys_platform == 'win32'"], "colorama: coverage"),
        # The extra typing-extensions declares for tests becomes active once a
        # shipped distribution asks for it.
        (["typing-extensions[test]"], "typing-extensions: pytest; extra == 'test'"),
    ],
    ids=["plain", "marker-holds", "extra-asked-for"],
)
def test_a_shipped_distribution_must_not_require_an_excluded_one(
    package_case, requires, requiring
):
    case = package_case
    install(case.python, "colorama", SHIPPED_RECORDS["colorama"], requires=requires)
    refused(
        case, "Shipped distributions require excluded ones: .*" + re.escape(requiring)
    )


@pytest.mark.parametrize(
    "requires",
    [
        ["pytest; python_version < '3.11'"],
        ["pytest; extra == 'dev'"],
        ["typing-extensions", "colorama[test]"],
    ],
    ids=["other-interpreter", "extra-nobody-asks-for", "extras-without-excluded"],
)
def test_inactive_requirements_of_excluded_ones_are_allowed(package_case, requires):
    case = package_case
    install(case.python, "colorama", SHIPPED_RECORDS["colorama"], requires=requires)
    assert case.run()["result"] == "fresh"


def test_a_shared_namespace_directory_loses_only_the_excluded_files(package_case):
    case = package_case
    install(
        case.python,
        "google-re2",
        [*EXCLUDED_RECORDS["google-re2"], "google/re2_shim.py"],
    )
    install(case.python, "protobuf", ["google/protobuf/__init__.py"])
    write_lock(case.project, case.python)
    case.run()
    runtime = set(package_manifest.load(case.manifest)["runtime"])
    assert package_manifest.RUNTIME + SITE + "google/protobuf/__init__.py" in runtime
    assert package_manifest.RUNTIME + SITE + "google/re2_shim.py" not in runtime
    assert package_manifest.RUNTIME + SITE + "re2/__init__.py" not in runtime


def test_a_file_owned_by_an_excluded_and_a_shipped_distribution_fails(package_case):
    case = package_case
    install(case.python, "colorama", [*SHIPPED_RECORDS["colorama"], "py.py"])
    refused(
        case,
        re.escape(
            "Excluded and shipped distributions share ['lib/site-packages/py.py']"
        ),
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("unpinned", r"pins differ \[\('extra-tool', '1.0'\)\]"),
        ("version", r"pins differ \[\('colorama', '1.0'\), \('colorama', '2.0'\)\]"),
        ("pip", r"pip 26.2.1 \(lock 26.3\)"),
        ("interpreter", "The lock records CPython 3.14.9"),
        ("header", re.escape("lacks the header fields ['archive_sha256']")),
    ],
)
def test_the_runtime_must_be_what_the_lock_describes(package_case, change, message):
    case = package_case
    if change == "unpinned":
        install(case.python, "extra-tool", ["extra_tool.py"])
    elif change == "version":
        lock = case.project / package_manifest.LOCK
        lock.write_bytes(lock.read_bytes().replace(b"colorama==1.0", b"colorama==2.0"))
    elif change == "pip":
        write_lock(case.project, case.python, pip="26.3")
    elif change == "interpreter":
        write_lock(case.project, case.python, interpreter="3.14.9")
    else:
        write_lock(case.project, case.python, drop=("archive_sha256",))
    refused(case, message)


@pytest.mark.parametrize(
    ("version", "remove", "message"),
    [
        (("3.13.5", 3, 13), None, "is CPython 3.13.5; chaiNNer-C needs 3.14 or newer"),
        (("3.14.8", 3, 14), "python314.dll", "lacks python314.dll"),
    ],
)
def test_the_runtime_must_be_cpython_3_14_or_newer_with_its_dll(
    package_case, version, remove, message
):
    case = package_case
    case.version = version
    if remove:
        (case.python / remove).unlink()
    refused(case, message)


def test_chainner_pip_must_be_preinstalled(package_case):
    case = package_case
    info = case.python / SITE / "chainner_pip-23.2.0.dist-info"
    for path in info.iterdir():
        path.unlink()
    info.rmdir()
    write_lock(case.project, case.python)
    refused(case, "chainner-pip is not installed")


@pytest.mark.parametrize("overlay", [None, {"source": "retired"}])
def test_refresh_ignores_the_retired_overlay_record(package_case, overlay):
    """A package copied before 2026-10-07 carries python_stack.overlay (null, or a
    record); it is no longer part of the identity, so the package still refreshes."""
    case = package_case
    case.run()
    manifest = package_manifest.load(case.manifest)
    manifest["python_stack"]["overlay"] = overlay
    package_manifest.write_json(case.manifest, manifest)
    assert case.run()["result"] == "verified_no_op"


def test_refresh_refuses_a_package_copied_from_another_stack(package_case):
    case = package_case
    case.run()
    lock = case.project / package_manifest.LOCK
    lock.write_bytes(lock.read_bytes().replace(b"ab" * 32, b"cd" * 32))
    before = snapshot(case.destination)
    with pytest.raises(ValueError, match="changed since this package was copied"):
        case.run()
    assert snapshot(case.destination) == before


def test_the_lock_reads_its_header_and_pins_and_hashes_as_git_stores_it(tmp_path):
    lock = tmp_path / "lock.txt"
    text = (
        "# The tested set, not a ceiling.\n"
        "# Interpreter: CPython 3.14.8 (standard build, not free-threaded).\n"
        "# Build tag: python-build-standalone 20261003\n"
        "# Archive SHA-256: " + "ab" * 32 + "\n"
        "# pip: 26.2.1 (installed; pip freeze does not list it)\n"
        "aiofiles==25.1.0\n"
        "google-re2==1.1.20251105  # oracle/reference only\n"
        "torch==2.14.1+cu132\n"
    )
    lock.write_bytes(text.encode())
    assert package_manifest.read_lock(lock) == {
        "interpreter": "3.14.8",
        "build_tag": "20261003",
        "archive_sha256": "ab" * 32,
        "pip": "26.2.1",
        "pins": {
            "aiofiles": "25.1.0",
            "google-re2": "1.1.20251105",
            "torch": "2.14.1+cu132",
        },
    }
    lf = package_manifest.lock_sha256(lock)
    lock.write_bytes(text.replace("\n", "\r\n").encode())
    assert package_manifest.lock_sha256(lock) == lf == package_files.sha(text.encode())
    lock.write_bytes((text + "chainner-pip @ file:///C:/wheel.whl\n").encode())
    with pytest.raises(ValueError, match="Unreadable pin"):
        package_manifest.read_lock(lock)


def test_provisioning_reads_the_lock_header_as_the_manifest_does():
    # provision_runtime.ps1 runs before any Python exists, so it mirrors LOCK_FIELDS.
    script = REPOSITORY / "native/tools/provision_runtime.ps1"
    text = script.read_text(encoding="utf-8")
    block = re.search(r"\$LockFields = \[ordered\]@\{\n(.*?)\n\}", text, re.S)
    assert block, "provision_runtime.ps1 has no $LockFields block"
    patterns = dict(re.findall(r"^\s+(\w+) = '(.+)'$", block.group(1), re.M))
    assert patterns == {
        key: f"^{pattern.pattern}$"
        for key, pattern in package_manifest.LOCK_FIELDS.items()
    }


def test_the_runtime_is_read_as_it_is_now(package_case):
    # importlib.metadata's listing cache is keyed by the directory's mtime, and an
    # install within one timestamp tick of a read would otherwise go unseen.
    case = package_case
    for index in range(20):
        package_files.runtime_distributions(case.python)
        install(case.python, f"late-{index}", [f"late_{index}.py"])
        assert f"late-{index}" in package_files.runtime_distributions(case.python)


def test_the_version_probe_asks_the_interpreter_itself():
    python = Path(sys.executable).parent
    assert package_port.interpreter_version(python) == (
        platform.python_version(),
        *sys.version_info[:2],
    )


# The runtime's bytecode (Consult 10 D-9).

LIB_SOURCES = {
    "top.py": b"VALUE = 1\n",
    "pkg/__init__.py": b"from .mod import twice\n",
    "pkg/mod.py": b"def twice(x):\n    return 2 * x\n",
}


def lib_tree(destination, sources=LIB_SOURCES):
    """A package destination whose runtime Lib holds sources; returns the
    runtime record of them, as copy_tree keys it."""
    lib = destination / package_manifest.RUNTIME / "Lib"
    for name, data in sources.items():
        write(lib, name, data)
    return {
        package_manifest.RUNTIME + "Lib/" + name: {"sha256": "", "bytes": len(data)}
        for name, data in sources.items()
    }


@pytest.fixture
def test_interpreter_compiles(monkeypatch):
    """compile_runtime with this test's interpreter standing in for the packaged
    python.exe, every other argument as built; returns the commands it ran."""
    real_run = subprocess.run
    commands = []

    def run(command, **options):
        commands.append(command)
        return real_run([sys.executable, *command[1:]], **options)

    monkeypatch.setattr(package_port.subprocess, "run", run)
    return commands


def test_the_compile_command_is_the_ruled_one(tmp_path):
    destination = tmp_path / "package"
    runtime = destination / package_manifest.RUNTIME
    assert package_port.compile_command(destination, 4) == [
        str(runtime / "python.exe"),
        "-I",
        "-S",
        "-B",
        "-m",
        "compileall",
        "-q",
        "-f",
        "--invalidation-mode",
        "unchecked-hash",
        "-s",
        str(destination),
        "-j",
        "4",
        str(runtime / "Lib"),
    ]


def test_the_compile_step_writes_identical_unchecked_hash_bytecode(
    tmp_path, test_interpreter_compiles
):
    tag = package_port.cache_tag(platform.python_version())
    assert tag == sys.implementation.cache_tag
    trees = {}
    for name in ("a", "b"):
        destination = tmp_path / name / "package"
        runtime = lib_tree(destination)
        package_port.compile_runtime(destination, 2)
        records = package_port.compiled_bytecode(destination, runtime, tag, 2)
        # Only the .pyc: the compiler's own imports wrote nothing (-B).
        lib = destination / package_manifest.RUNTIME / "Lib"
        assert set(package_files.walk(lib)) == set(LIB_SOURCES) | {
            name.removeprefix(package_manifest.RUNTIME + "Lib/") for name in records
        }
        trees[name] = {key: (destination / key).read_bytes() for key in sorted(records)}
        for key, data in trees[name].items():
            assert int.from_bytes(data[4:8], "little") == 1, key  # unchecked-hash
            # co_filename is relative to the package (-s), whatever the directory.
            code = marshal.loads(data[16:])
            source = PurePosixPath(key).parent.parent / (
                key.split("/")[-1].split(".")[0] + ".py"
            )
            assert code.co_filename == str(Path(source))
    assert test_interpreter_compiles == [
        package_port.compile_command(tmp_path / name / "package", 2) for name in "ab"
    ]
    assert trees["a"] == trees["b"]
    assert len(trees["a"]) == 3


def test_a_file_that_does_not_compile_fails_the_compile_step(
    tmp_path, test_interpreter_compiles
):
    destination = tmp_path / "package"
    lib_tree(destination, {**LIB_SOURCES, "pkg/broken.py": b"def broken(:\n"})
    with pytest.raises(RuntimeError, match=r"failed \(exit 1\)") as raised:
        package_port.compile_runtime(destination, 1)
    assert "broken.py" in str(raised.value)
    assert "SyntaxError" in str(raised.value)


def test_a_compile_error_fails_the_build_and_a_rerun_recovers(
    package_case, monkeypatch
):
    case = package_case

    def interrupted(target, workers):
        compile_lib(target, skip={"dependency.py"})
        raise RuntimeError("Compiling failed (exit 1): *** Error compiling 'x.py'")

    with monkeypatch.context() as patch:
        patch.setattr(package_port, "compile_runtime", interrupted)
        with pytest.raises(RuntimeError, match=re.escape("Error compiling 'x.py'")):
            case.run()
    assert package_manifest.load(case.manifest)["state"] == "copying"
    assert not (case.destination / "portable").exists()
    # The rerun takes the partial bytecode as a derivative and compiles again.
    assert case.run()["result"] == "recovered"
    manifest = package_manifest.load(case.manifest)
    assert manifest["state"] == "complete"
    pyc = package_manifest.RUNTIME + "Lib/" + cached("dependency.py")
    assert pyc in manifest["runtime"]


@pytest.mark.parametrize(
    ("compile_runtime", "message"),
    [
        (lambda target, _: compile_lib(target, skip={"dependency.py"}), "missing"),
        (
            lambda target, _: (
                compile_lib(target),
                write(target / package_manifest.RUNTIME, "Lib/__pycache__/x.nbi", b"x"),
            ),
            "unexpected",
        ),
        (lambda target, _: compile_lib(target, flags=0), "not unchecked-hash"),
        (lambda target, _: compile_lib(target, flags=3), "not unchecked-hash"),
    ],
    ids=["missing-pyc", "unexpected-file", "timestamp-pyc", "checked-hash-pyc"],
)
def test_the_compiled_bytecode_must_match_the_runtime_sources(
    package_case, monkeypatch, compile_runtime, message
):
    case = package_case
    monkeypatch.setattr(package_port, "compile_runtime", compile_runtime)
    with pytest.raises(RuntimeError, match=message):
        case.run()
    assert package_manifest.load(case.manifest)["state"] == "copying"


def test_the_build_verifies_its_bytecode_and_a_refresh_carries_it_forward(
    package_case,
):
    case = package_case
    case.run()
    manifest = package_manifest.load(case.manifest)
    pyc = package_manifest.RUNTIME + "Lib/" + cached("dependency.py")
    package_manifest.verify_recorded(case.destination, manifest, bytecode=True)
    extra = package_manifest.RUNTIME + "Lib/__pycache__/other.cpython-314.pyc"
    write(case.destination, extra, b"unrecorded")
    with pytest.raises(ValueError, match=re.escape(f"unrecorded: ['{extra}']")):
        package_manifest.verify_recorded(case.destination, manifest, bytecode=True)
    (case.destination / extra).unlink()
    write(case.destination, pyc, b"rewritten by a dependency install")
    with pytest.raises(ValueError, match=re.escape(f"differing: ['{pyc}']")):
        package_manifest.verify_recorded(case.destination, manifest, bytecode=True)
    # A refresh carries the runtime forward unverified, its bytecode included.
    package_manifest.verify_recorded(case.destination, manifest)
    write(case.backend, FIRST, b"def value():\n    return 3\n")
    assert case.run()["result"] == "refreshed"
    assert package_manifest.load(case.manifest)["runtime"] == manifest["runtime"]
    assert len(case.compiles) == 1


def test_a_package_copied_without_compiled_bytecode_refuses_refresh(package_case):
    case = package_case
    case.run()
    manifest = package_manifest.load(case.manifest)
    del manifest["python_stack"]["bytecode"]
    package_manifest.write_json(case.manifest, manifest)
    before = snapshot(case.destination)
    with pytest.raises(ValueError, match="bytecode changed since this package"):
        case.run()
    assert snapshot(case.destination) == before


def test_a_package_without_the_toolchain_record_gains_it_at_its_next_refresh(
    package_case,
):
    """A package written before native_toolchain was recorded (Consult 14 D-30):
    its next refresh writes the record and nothing else, the one after is a no-op."""
    case = package_case
    case.run()
    manifest = package_manifest.load(case.manifest)
    del manifest["native_toolchain"]
    package_manifest.write_json(case.manifest, manifest)
    result = case.run()
    assert (result["result"], result["written_files"]) == ("refreshed", 0)
    assert package_manifest.load(case.manifest)["native_toolchain"] == NATIVE_TOOLCHAIN
    assert case.run()["result"] == "verified_no_op"


@pytest.mark.parametrize("pins", [0, 2])
@pytest.mark.parametrize("key", list(package_manifest.TOOLCHAIN_PINS))
def test_each_toolchain_version_needs_exactly_one_pin(package_case, key, pins):
    """A pin missing from its file, or two different ones, fails the run."""
    case = package_case
    name, pattern = package_manifest.TOOLCHAIN_PINS[key]
    path = case.project / name
    text = path.read_text(encoding="utf-8")
    match = pattern.search(text)
    assert match is not None
    other = match.group(0).replace(match.group(1), "0.0.1")
    replacement = "" if pins == 0 else f"{match.group(0)}\n{other}"
    path.write_text(text.replace(match.group(0), replacement), encoding="utf-8")
    with pytest.raises(ValueError, match=f"does not pin exactly one {key} version"):
        case.run()
    assert not case.destination.exists()
