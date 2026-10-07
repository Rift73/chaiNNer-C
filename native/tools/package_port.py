"""Build or refresh the independent portable chaiNNer-C package.

The installed nightly supplies the shell (without its resources/src and logs);
the provisioned runtime native/runtime/cpython-<version> supplies Python, less
the distributions package_files.EXCLUDED names and every __pycache__; this
project's backend/src is the whole backend. A fresh build compiles the copied
runtime's Lib (BYTECODE). Reviewed UI patches remove upstream updates and add
the TensorRT types and the TensorRT nodes' Clear item (independent_ui.py).
Installation and profile files are never modified, and files are copied, never
linked. Run with --help for the deliberately explicit inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import independent_ui
import package_files
import package_manifest
from package_manifest import BACKEND, MANIFEST, RUNTIME, SCHEMA
from packaging.utils import canonicalize_name

PROJECT = Path(__file__).resolve().parents[2]
# The runtime reports its own version; -I -S -B: no environment, no site and no
# bytecode, so the probe writes nothing.
VERSION_PROBE = (
    "import json, platform, sys; "
    "print(json.dumps([platform.python_version(), *sys.version_info[:2]]))"
)
# The runtime's bytecode (Consult 10 D-9). The source runtime's __pycache__ is never
# copied: its .pyc are timestamp-based and incidental, and numba's .nbi/.nbc caches
# there are machine code for the build CPU. A fresh build compiles the copy's whole
# Lib, stdlib and site-packages, into .pyc that depend on the source bytes alone, so
# the first launch writes no bytecode there and two builds are byte-identical.
# Nothing edits that tree in place (pip rewrites its own .pyc), so the .pyc are not
# checked against their sources at import. The backend is not compiled: it is the
# tree that changes (--refresh, owner edits).
BYTECODE = "unchecked-hash"
# A .pyc's flags word (PEP 552): bit 0 hash-based, bit 1 checked against the source.
UNCHECKED_HASH_FLAGS = 1


def under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--installed-app",
        required=True,
        type=Path,
        help="Versioned directory containing chaiNNer.exe and resources",
    )
    parser.add_argument(
        "--installed-python",
        required=True,
        type=Path,
        help="The provisioned runtime native\\runtime\\cpython-<the lock's version> "
        "(python.exe, Lib and python3XX.dll; CPython 3.14 or newer), copied less the "
        "excluded distributions",
    )
    parser.add_argument("--destination", type=Path, default=PROJECT / "out/chaiNNer-C")
    parser.add_argument(
        "--copy-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Bounded parallel independent-file copy/hash jobs (default up to 8)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check the inputs (installed app, UI pins, backend inventory, git state) and print digests; write nothing; the destination is not inspected",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Require an existing package (refreshing one is the default anyway)",
    )
    return parser.parse_args()


def digest(value: object) -> str:
    return package_files.sha(json.dumps(value, sort_keys=True).encode("utf-8"))


def interpreter_version(python: Path) -> tuple[str, int, int]:
    """The runtime's version string, major and minor, as its interpreter reports them."""
    result = subprocess.run(
        [str(python / "python.exe"), "-I", "-S", "-B", "-c", VERSION_PROBE],
        capture_output=True,
        check=False,
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(
            f"{python / 'python.exe'} did not report its version: {result.stderr.strip()}"
        )
    version, major, minor = json.loads(result.stdout)
    return version, major, minor


def python_stack(python: Path) -> tuple[dict[str, Any], Callable[[str, bool], bool]]:
    """The manifest's python_stack record and the runtime walk's exclusions.

    python must be the provisioned runtime, which is what the lock describes:
    CPython 3.14 or newer with its python3XX.dll, the lock's interpreter, every
    distribution at the lock's pin and nothing else, pip at its version, and
    chainner-pip installed (else the packaged host installs it at first
    launch). The exclusions' own checks follow (package_files.runtime_exclusion).
    Any failed check fails the run. The walk also skips every __pycache__ (BYTECODE).
    """
    lock_path = PROJECT / package_manifest.LOCK
    lock = package_manifest.read_lock(lock_path)
    version, major, minor = interpreter_version(python)
    if (major, minor) < (3, 14):
        raise ValueError(
            f"{python} is CPython {version}; chaiNNer-C needs 3.14 or newer"
        )
    if not (python / f"python{major}{minor}.dll").is_file():
        raise ValueError(f"{python} lacks python{major}{minor}.dll")
    if version != lock["interpreter"]:
        raise ValueError(
            f"The lock records CPython {lock['interpreter']}; {python} is {version}"
        )
    dists = package_files.runtime_distributions(python)
    pinned = {canonicalize_name(name): pin for name, pin in lock["pins"].items()}
    installed = {name: dist.version for name, dist in dists.items() if name != "pip"}
    differing = sorted(set(pinned.items()) ^ set(installed.items()))
    pip = dists["pip"].version if "pip" in dists else None
    if differing or pip != lock["pip"]:
        raise ValueError(
            f"The lock does not describe {python}: pins differ {differing[:10]}, pip "
            f"{pip} (lock {lock['pip']}); provision it from the lock with "
            "provision_runtime.ps1"
        )
    if "chainner-pip" not in dists:
        raise ValueError(
            f"chainner-pip is not installed in {python}, so the packaged host would "
            "install it at first launch; run provision_runtime.ps1"
        )
    excluded = package_files.runtime_exclusion(python, dists, version)

    def skip(relative: str, is_dir: bool) -> bool:
        return package_files.cache_skip(relative, is_dir) or excluded(relative, is_dir)

    record = {
        "interpreter_version": version,
        "build_tag": lock["build_tag"],
        "archive_sha256": lock["archive_sha256"],
        "lock_sha256": package_manifest.lock_sha256(lock_path),
        "runtime": str(python),
        "excluded": dict(package_files.EXCLUDED),
        "chainner_pip_version": dists["chainner-pip"].version,
        "bytecode": BYTECODE,
    }
    return record, skip


def copy_tree(
    source: Path,
    paths: list[str],
    target: Path,
    destination: Path,
    workers: int,
    reuse: bool,
) -> dict[str, dict[str, Any]]:
    """copy_verified every path; return records keyed relative to destination."""
    pairs = [(source / path, target / path, reuse) for path in paths]
    copied = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(package_files.copy_verified, pairs)
        for count, (written, sha256, length) in enumerate(results, 1):
            key = written.relative_to(destination).as_posix()
            copied[key] = {"sha256": sha256, "bytes": length}
            if count % 2000 == 0:
                print(f"Verified {count} copied files", file=sys.stderr, flush=True)
    return copied


def compile_command(destination: Path, workers: int) -> list[str]:
    """The packaged interpreter compiling its own Lib to BYTECODE: -I -S -B (no
    environment, no site, and no timestamp .pyc for the compiler's own imports),
    every file (-f), co_filename relative to the package (-s, so the bytes do not
    depend on the build directory), on workers processes."""
    runtime = destination / RUNTIME
    return [
        str(runtime / "python.exe"),
        "-I",
        "-S",
        "-B",
        "-m",
        "compileall",
        "-q",
        "-f",
        "--invalidation-mode",
        BYTECODE,
        "-s",
        str(destination),
        "-j",
        str(workers),
        str(runtime / "Lib"),
    ]


def compile_runtime(destination: Path, workers: int) -> None:
    """Run compile_command; a file that does not compile fails the build, and the
    error compileall prints (on stdout under -q) is reported."""
    result = subprocess.run(
        compile_command(destination, workers),
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        lines = (result.stdout.strip() or result.stderr.strip()).splitlines()
        raise RuntimeError(
            f"Compiling {destination / RUNTIME / 'Lib'} failed (exit "
            f"{result.returncode}): " + "\n".join(lines[-40:])
        )


def cache_tag(version: str) -> str:
    """CPython's cache tag for an interpreter version (standard build)."""
    major, minor = version.split(".")[:2]
    return f"cpython-{major}{minor}"


def compiled_bytecode(
    destination: Path, runtime: dict[str, Any], tag: str, workers: int
) -> dict[str, dict[str, Any]]:
    """The compiled .pyc records, keyed like runtime's.

    Exactly one .pyc per Lib .py runtime records, and nothing else in any
    __pycache__ below Lib; each one unchecked-hash (its flags word). Any
    difference fails the build.
    """
    lib = RUNTIME + "Lib/"
    expected = set()
    for key in runtime:
        if key.startswith(lib) and key.endswith(".py"):
            source = PurePosixPath(key)
            expected.add(
                (source.parent / "__pycache__" / f"{source.stem}.{tag}.pyc").as_posix()
            )
    found = {
        lib + relative
        for relative in package_files.walk(destination / lib)
        if "__pycache__" in PurePosixPath(relative).parts
    }
    missing, unexpected = sorted(expected - found), sorted(found - expected)
    if missing or unexpected:
        raise RuntimeError(
            "The compiled bytecode does not match the runtime's sources; "
            f"missing: {missing[:10]}, unexpected: {unexpected[:10]}"
        )

    def record(key: str) -> tuple[str, dict[str, Any], bool]:
        data = (destination / key).read_bytes()
        flags = int.from_bytes(data[4:8], "little")
        entry = {"sha256": package_files.sha(data), "bytes": len(data)}
        return key, entry, flags == UNCHECKED_HASH_FLAGS

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(record, sorted(found)))
    wrong = [key for key, _, unchecked in records if not unchecked]
    if wrong:
        raise RuntimeError(f"Bytecode that is not {BYTECODE}: {wrong[:10]}")
    return {key: entry for key, entry, _ in records}


def pending_changes(
    destination: Path,
    wanted: dict[str, tuple[bytes, str]],
    recorded_backend: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Owned files to write (binaries first) and recorded backend files to delete."""
    pending = sorted(
        (
            key
            for key, (_, sha256) in wanted.items()
            if not (destination / key).is_file()
            or package_files.hash_file(destination / key) != sha256
        ),
        key=lambda key: (not key.lower().endswith(package_files.BINARY_SUFFIXES), key),
    )
    removed = []
    for key in sorted(set(recorded_backend) - set(wanted)):
        target = destination / key
        if not target.exists():
            continue  # Already deleted before an interruption.
        if package_files.hash_file(target) != recorded_backend[key]["sha256"]:
            raise ValueError(f"Removed backend file differs from its record: {target}")
        removed.append(key)
    return pending, removed


def finish(result: str, destination: Path, backend: int, written: int = 0) -> int:
    summary = {
        "result": result,
        "destination": str(destination),
        "manifest": str(destination / MANIFEST),
        "backend_files": backend,
        "written_files": written,
        "upstream_updates": "disabled",
    }
    print(json.dumps(summary), flush=True)
    return 0


def main() -> int:
    args = parse_arguments()
    if not 1 <= args.copy_workers <= 32:
        raise ValueError("copy-workers must be between 1 and 32")
    app = args.installed_app.resolve(strict=True)
    python = args.installed_python.resolve(strict=True)
    out = (PROJECT / "out").resolve()
    destination = args.destination.resolve()
    if not under(destination, out) or destination == out:
        raise ValueError(
            "Destination must be a child directory of this project's out directory"
        )
    if any(
        under(destination, source) or under(source, destination)
        for source in (app, python)
    ):
        raise ValueError("Source and destination trees must not overlap")
    for required in (
        app / "chaiNNer.exe",
        app / "resources/app/package.json",
        python / "python.exe",
    ):
        if not required.is_file():
            raise ValueError(f"Required runtime file missing: {required}")
    version = json.loads(
        (app / "resources/app/package.json").read_text(encoding="utf-8")
    )["version"]
    stack, runtime_skip = python_stack(python)
    ui_bytes, ui_patches = independent_ui.build_ui_overrides(app)
    shell_files = [
        path
        for path in package_files.walk(app, package_files.shell_skip)
        if path not in ui_bytes
    ]
    tracked, git = package_files.git_snapshot(PROJECT)
    source = package_files.backend_inventory(PROJECT / "backend/src", tracked)
    wanted = {
        entry["destination"]: (
            ui_bytes[entry["destination"]],
            entry["converted_sha256"],
        )
        for entry in ui_patches
    }
    wanted.update(
        (BACKEND + path, (data, package_files.sha(data)))
        for path, data in source.items()
    )
    backend = {
        key: {"sha256": wanted[key][1], "bytes": len(wanted[key][0])}
        for key in sorted(wanted)
        if key.startswith(BACKEND)
    }
    desired = {
        "app_version": version,
        "ui_patches": ui_patches,
        "backend": backend,
        "python_stack": stack,
        "native_toolchain": package_manifest.read_toolchain(PROJECT),
    }
    if args.validate_only:
        shell = {
            path: {
                "sha256": package_files.hash_file(app / path),
                "bytes": (app / path).stat().st_size,
            }
            for path in shell_files
        }
        summary = {
            "validation_only": True,
            "app_version": version,
            "shell_files": len(shell),
            "shell_sha256": digest(shell),
            "ui_patches": len(ui_patches),
            "ui_patches_sha256": digest(ui_patches),
            "backend_files": len(backend),
            "backend_sha256": digest(backend),
            "runtime_files": len(package_files.walk(python, runtime_skip)),
            "python_stack": stack,
            "native_toolchain": desired["native_toolchain"],
            **git,
            "package_modified": False,
        }
        print(json.dumps(summary), flush=True)
        return 0

    identity = {
        "project": str(PROJECT),
        "destination": str(destination),
        "installed_app": str(app),
        "installed_python": str(python),
    }
    manifest_path = destination / MANIFEST
    base = {
        "schema": SCHEMA,
        "identity": identity,
        "provenance": {"utc": datetime.now(UTC).isoformat(), **git},
        "upstream_updates": "disabled",
        **desired,
    }
    old = None
    if destination.exists() and any(destination.iterdir()):
        if not manifest_path.is_file():
            raise ValueError(
                "Destination is nonempty and has no generated package manifest"
            )
        old = package_manifest.load(manifest_path)
    elif args.refresh:
        raise ValueError("Cannot refresh an empty or absent package")

    # Fresh: copy everything. Initial copy interrupted: copy again, reusing
    # hash-matched files. Refresh, or refresh interrupted: never touch the
    # shell, runtime or profile; only UI patches and backend files change.
    result, copy_installed = "fresh", old is None
    shell: dict[str, Any] = {}
    runtime: dict[str, Any] = {}
    recorded_backend: dict[str, Any] = {}
    if old is not None:
        package_manifest.check_existing(old, identity)
        shell, runtime, recorded_backend = old["shell"], old["runtime"], old["backend"]
        if old["state"] == "complete":
            result = "refreshed"
            package_manifest.verify_recorded(destination, old)
            package_manifest.check_refresh(old, desired)
            if package_manifest.is_no_op(old, desired):
                # Same inputs write nothing, not even a new manifest timestamp.
                return finish("verified_no_op", destination, len(backend))
        else:
            result = "recovered"
            baseline = set(shell_files) | set(ui_bytes)
            baseline |= {
                RUNTIME + path for path in package_files.walk(python, runtime_skip)
            }
            refreshing = package_manifest.validate_recovery(
                destination,
                baseline,
                package_manifest.recorded_inventory(old),
                set(wanted) | {key + ".tmp" for key in wanted},
            )
            if refreshing:
                package_manifest.check_refresh(old, desired)
            copy_installed = not refreshing
            print("Recovering interrupted package", file=sys.stderr, flush=True)

    if copy_installed:
        if old is None:
            destination.mkdir(parents=True, exist_ok=True)
            empty = {"shell": {}, "ui_patches": [], "backend": {}, "runtime": {}}
            package_manifest.write_json(
                manifest_path, {**base, "state": "copying", **empty}
            )
        print(f"Copying {app} and {python}", file=sys.stderr, flush=True)
        workers, reuse = args.copy_workers, old is not None
        shell = copy_tree(app, shell_files, destination, destination, workers, reuse)
        runtime_files = package_files.walk(python, runtime_skip)
        runtime = copy_tree(
            python, runtime_files, destination / RUNTIME, destination, workers, reuse
        )
        print(f"Compiling the runtime's Lib ({BYTECODE})", file=sys.stderr, flush=True)
        compile_runtime(destination, workers)
        tag = cache_tag(stack["interpreter_version"])
        runtime |= compiled_bytecode(destination, runtime, tag, workers)
        (destination / "portable").write_bytes(b"")

    pending, removed = pending_changes(destination, wanted, recorded_backend)
    package_files.probe_unlocked(destination / key for key in pending + removed)
    if old is not None and old["state"] == "complete":
        package_manifest.write_json(manifest_path, {**old, "state": "copying"})
    for key in pending:
        package_files.replace_verified(destination / key, *wanted[key])
    for key in removed:
        (destination / key).unlink()
    manifest = {**base, "state": "complete", "shell": shell, "runtime": runtime}
    package_manifest.verify_recorded(destination, manifest, bytecode=copy_installed)
    package_manifest.write_json(manifest_path, manifest)
    return finish(result, destination, len(backend), len(pending) + len(removed))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Package failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
