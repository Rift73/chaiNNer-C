"""The v2 package manifest: schema and guarded state checks.

Layout (keys are written sorted, so equal content is byte-identical):
  schema, state ("copying" | "complete"),
  identity {project, destination, installed_app, installed_python},
  provenance {utc, git_head, git_branch, backend_dirty}, app_version,
  upstream_updates, shell {rel: {sha256, bytes}},
  ui_patches [{destination, installed_sha256, converted_sha256, reason}],
  backend {"resources/src/<rel>": {sha256, bytes}},
  runtime {"python/python/<rel>": {sha256, bytes}},
  python_stack {interpreter_version, build_tag, archive_sha256, lock_sha256,
                runtime, excluded {distribution: reason}, chainner_pip_version,
                bytecode},
  native_toolchain {clang_cl, msvc_toolset, windows_sdk}.
The runtime is recorded at the initial copy, with the .pyc that build compiles
into its Lib (package_port.BYTECODE; that build verifies them), and is carried
forward unverified: portable dependency installs legitimately change it.
python_stack describes the provisioned runtime it was copied from (the lock's
interpreter, archive and hash), the distributions the copy leaves out and the
runtime's bytecode mode. A package copied before 2026-10-07 also carries
python_stack.overlay, a retired record that is not part of its identity.
native_toolchain names the versions the native build accepts (read_toolchain),
which a refresh rewrites with the backend's binaries.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import package_files

SCHEMA = "chaiNNer-C/package/v2"
MANIFEST = "chainner-c-package.json"
BACKEND = "resources/src/"
RUNTIME = "python/python/"
CONTROL_FILES = frozenset({MANIFEST, MANIFEST + ".tmp", "portable"})
# The tracked lock and the provisioned runtimes, relative to the project
# (native/tools/provision_runtime.ps1 provisions a runtime from the lock; its -Relock
# rewrites the lock).
LOCK = "native/python-stack.lock.txt"
RUNTIMES = "native/runtime"
# The lock header's fields, as provision_runtime.ps1 reads and (-Relock) writes them.
LOCK_FIELDS = {
    "interpreter": re.compile(r"# Interpreter: CPython (\d+\.\d+\.\d+) .*"),
    "build_tag": re.compile(r"# Build tag: python-build-standalone (\d+)"),
    "archive_sha256": re.compile(r"# Archive SHA-256: ([0-9a-f]{64})"),
    "pip": re.compile(r"# pip: (\S+) .*"),
}
# A pin, with an optional note after it ("google-re2==1.1  # oracle/reference only").
PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)(?:  # .*)?")
# A wheel installed from a URL, as pip freeze writes it ("spandrel @ https://.../
# spandrel-0.4.2%2Bc1-py3-none-any.whl#sha256=..."): its version is the wheel's.
URL_PIN = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9._-]*) @ \S+/[^/\s-]+-([^/\s-]+)-[^/\s]+\.whl(?:#\S*)?"
    r"(?:  # .*)?"
)
# The native toolchain's pins, relative to the project: CMakeLists.txt refuses any
# other clang-cl, toolchain.cmake any other MSVC toolset or Windows SDK (Consult 14
# D-30), so they name the versions every native binary of the tree is built with.
TOOLCHAIN_PINS = {
    "clang_cl": (
        "native/CMakeLists.txt",
        re.compile(r"_COMPILER_VERSION VERSION_EQUAL (\d+\.\d+\.\d+)\)"),
    ),
    "msvc_toolset": (
        "native/toolchain.cmake",
        re.compile(r'set\(chainner_c_msvc_toolset_version "([^"]+)"\)'),
    ),
    "windows_sdk": (
        "native/toolchain.cmake",
        re.compile(r'set\(chainner_c_windows_sdk_version "([^"]+)"\)'),
    ),
}


def lock_sha256(path: Path) -> str:
    """The lock's SHA-256 with LF line ends, as git stores it, so checkouts agree."""
    return package_files.sha(path.read_bytes().replace(b"\r\n", b"\n"))


def read_lock(path: Path) -> dict[str, Any]:
    """The lock's header fields and its pins, {name: version} as pip froze them (a
    wheel installed from a URL: the wheel's version)."""
    fields: dict[str, Any] = {}
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            for key, pattern in LOCK_FIELDS.items():
                if match := pattern.fullmatch(line):
                    fields[key] = match.group(1)
        elif line:
            if match := PIN.fullmatch(line):
                pins[match.group(1)] = match.group(2)
            elif match := URL_PIN.fullmatch(line):
                pins[match.group(1)] = unquote(match.group(2))
            else:
                raise ValueError(f"Unreadable pin in {path}: {line!r}")
    missing = sorted(set(LOCK_FIELDS) - set(fields))
    if missing:
        raise ValueError(
            f"The lock {path} lacks the header fields {missing}; regenerate it with "
            "provision_runtime.ps1 -Relock"
        )
    return {**fields, "pins": pins}


def read_toolchain(project: Path) -> dict[str, str]:
    """The manifest's native_toolchain record: each TOOLCHAIN_PINS version, which
    its file must pin exactly once."""
    versions = {}
    for key, (name, pattern) in TOOLCHAIN_PINS.items():
        found = sorted(
            set(pattern.findall((project / name).read_text(encoding="utf-8")))
        )
        if len(found) != 1:
            raise ValueError(
                f"{project / name} does not pin exactly one {key} version: {found}"
            )
        versions[key] = found[0]
    return versions


def provisioned_runtime(project: Path) -> Path:
    """native/runtime/cpython-<the lock's interpreter>: the runtime chaiNNer-C is
    provisioned, tested and packaged with, and the one the oracle runs on."""
    return project / RUNTIMES / f"cpython-{read_lock(project / LOCK)['interpreter']}"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def check_existing(old: dict[str, Any], identity: dict[str, str]) -> None:
    if (
        old.get("schema") != SCHEMA
        or old.get("identity") != identity
        or old.get("state") not in ("complete", "copying")
    ):
        raise ValueError(
            "Destination manifest identity/state does not exactly match this package"
        )


def owned_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    """Every file the package tool writes and re-verifies (not the runtime)."""
    hashes = {key: entry["sha256"] for key, entry in manifest["shell"].items()}
    for entry in manifest["ui_patches"]:
        hashes[entry["destination"]] = entry["converted_sha256"]
    for key, entry in manifest["backend"].items():
        hashes[key] = entry["sha256"]
    return hashes


def runtime_bytecode(name: str) -> bool:
    """A runtime .pyc in a __pycache__ (the build compiles them; Python may too)."""
    return (
        name.startswith(RUNTIME) and "/__pycache__/" in name and name.endswith(".pyc")
    )


def verify_recorded(
    destination: Path, manifest: dict[str, Any], *, bytecode: bool = False
) -> None:
    """Every owned file matches its record and resources/src holds nothing else.

    bytecode (a build that compiled the runtime's Lib): the runtime's .pyc match
    their records too, and its Lib holds no other. A refresh carries the runtime
    forward unverified, its bytecode included.
    """
    hashes = owned_hashes(manifest)
    if bytecode:
        hashes |= {
            key: entry["sha256"]
            for key, entry in manifest["runtime"].items()
            if runtime_bytecode(key)
        }
    differing = sorted(
        key
        for key, digest in hashes.items()
        if not (destination / key).is_file()
        or package_files.hash_file(destination / key) != digest
    )
    present = package_files.walk(destination / BACKEND, package_files.cache_skip)
    unrecorded = sorted({BACKEND + key for key in present} - set(manifest["backend"]))
    if bytecode:
        compiled = {
            RUNTIME + "Lib/" + key
            for key in package_files.walk(destination / RUNTIME / "Lib")
            if runtime_bytecode(RUNTIME + "Lib/" + key)
        }
        unrecorded += sorted(compiled - set(manifest["runtime"]))
    if differing or unrecorded:
        raise ValueError(
            "Package changed outside the package tool; "
            f"differing: {differing[:10]}, unrecorded: {unrecorded[:10]}"
        )
    if not (destination / "portable").is_file():
        raise ValueError("Portable isolation marker is missing")


def check_refresh(old: dict[str, Any], desired: dict[str, Any]) -> None:
    """A refresh rewrites UI patches and backend files only, never the shell or
    the runtime, so the runtime it was copied from must be unchanged too."""
    if old["app_version"] != desired["app_version"]:
        raise ValueError("Installed app version changed; build a new package")
    stack = old.get("python_stack")
    if stack is not None:
        stack = {key: value for key, value in stack.items() if key != "overlay"}
    if stack != desired["python_stack"]:
        raise ValueError(
            "The provisioned runtime, its lock, the exclusions or the runtime's "
            "bytecode changed since this package was copied; build a new package"
        )
    if {entry["destination"] for entry in old["ui_patches"]} != {
        entry["destination"] for entry in desired["ui_patches"]
    }:
        raise ValueError("Changing the set of patched UI bundles requires review")


def is_no_op(old: dict[str, Any], desired: dict[str, Any]) -> bool:
    """A package written before native_toolchain was recorded lacks it, so its
    first refresh writes the record."""
    return all(
        old.get(key) == desired[key]
        for key in ("app_version", "ui_patches", "backend", "native_toolchain")
    )


def recorded_inventory(manifest: dict[str, Any]) -> set[str]:
    """Installed-derived paths; empty until an initial copy completes."""
    ui = {entry["destination"] for entry in manifest["ui_patches"]}
    return set(manifest["shell"]) | ui | set(manifest["runtime"])


def validate_recovery(
    destination: Path,
    baseline: set[str],
    recorded: set[str],
    generated: set[str],
) -> bool:
    """Return whether an interrupted operation refreshes an already copied app.

    Initial-copy manifests have no file inventory until the copy completes.
    Refresh manifests retain the complete verified inventory of the old package.
    Only that second case may contain its subsequently created portable profile;
    those unowned files are never copied, overwritten or removed here. Runtime
    bytecode is a disposable derivative in both: an interrupted build's compile
    left it (the next build compiles every file again), or Python wrote it after
    the original copy; it is not imported during a refresh.
    """
    existing = set(package_files.walk(destination))

    def without_runtime_caches(files: set[str]) -> set[str]:
        return {name for name in files if not runtime_bytecode(name)}

    if recorded:
        baseline_files = without_runtime_caches(baseline)
        if without_runtime_caches(recorded) != baseline_files:
            raise ValueError("Interrupted refresh has an incomplete baseline inventory")
        if not baseline_files <= existing or "portable" not in existing:
            raise ValueError(
                "Interrupted refresh is missing a copied runtime or portable marker"
            )
        return True
    unexpected = sorted(
        without_runtime_caches(existing) - baseline - generated - CONTROL_FILES
    )
    if unexpected:
        raise ValueError(
            f"Incomplete destination contains non-generated paths: {unexpected[:20]}"
        )
    return False
