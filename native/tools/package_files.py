"""File primitives for package_port: hashing, guarded walks, verified writes and
the runtime distributions the package leaves out."""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterable
from importlib import metadata
from pathlib import Path, PurePosixPath

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# Git-ignored build outputs that ship inside backend/src. chainner_ext's tracked
# __init__.py and __init__.pyi ship as git lists them.
NATIVE_BINARIES = frozenset(
    {
        "nodes/impl/chainner_native.dll",
        "nodes/impl/_chainner_graph.pyd",
        "chainner_ext/chainner_ext.pyd",
    }
)
# A running backend maps or executes these, so they are written first.
BINARY_SUFFIXES = (".dll", ".pyd", ".exe")
REPARSE_POINT = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT: symlinks and junctions
SITE_PACKAGES = "Lib/site-packages"
# Distributions of the provisioned runtime that the package copy leaves out
# (Consult 6 P2, Consult 7 and its precedent, Consult 8 sweep item 4), with the
# reason the manifest records: the test and build tools, and the ones only
# upstream's code needs (the oracle and the frozen references): google-re2 (the
# ONNX loader), Sanic-Cors (the server; chaiNNer-C has cors.py) and pynvml (the
# server's declared dependency; chaiNNer-C declares nvidia-ml-py, which provides
# the pynvml module). The runtime itself keeps them.
EXCLUDED = {
    "pytest": "dev-only",
    "pytest-asyncio": "dev-only",
    "pytest-cov": "dev-only",
    "coverage": "dev-only",
    "pluggy": "dev-only",
    "iniconfig": "dev-only",
    "Pygments": "dev-only",
    "pybind11": "dev-only",
    "google-re2": "oracle-only",
    "Sanic-Cors": "oracle-only",
    "pynvml": "oracle-only",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shell_skip(relative: str, is_dir: bool) -> bool:
    """Installed app walk: the backend comes from the project, logs are not assets.

    The root squirrel.exe is the installer's update helper: inert in a portable
    package that never updates, so it is not shipped.
    """
    if is_dir:
        return relative == "resources/src"
    return relative.lower().endswith(".log") or relative.lower() == "squirrel.exe"


def cache_skip(relative: str, is_dir: bool) -> bool:
    """Bytecode caches are runtime derivatives, never package content."""
    return is_dir and PurePosixPath(relative).name == "__pycache__"


def walk(
    root: Path,
    skip: Callable[[str, bool], bool] | None = None,
    *,
    directories: bool = False,
) -> list[str]:
    """Sorted POSIX paths of the files below root, relative to it.

    Every entry of every visited directory is checked first, so a symlink or
    junction is refused instead of followed, copied or deleted through. A
    directory that cannot be listed is an error, never a silent omission. With
    directories, each visited directory is listed too, with a trailing "/", so
    an empty one is seen.
    """

    def fail(error: OSError) -> None:
        raise error

    found = []
    for directory, dirs, files in os.walk(root, onerror=fail, followlinks=False):
        base = Path(directory)
        for name in dirs + files:
            candidate = base / name
            if candidate.is_symlink() or (
                candidate.lstat().st_file_attributes & REPARSE_POINT
            ):
                raise ValueError(f"Refusing reparse point: {candidate}")
        kept = []
        for name in dirs:
            relative = (base / name).relative_to(root).as_posix()
            if skip is None or not skip(relative, True):
                kept.append(name)
                if directories:
                    found.append(relative + "/")
        dirs[:] = kept
        for name in files:
            relative = (base / name).relative_to(root).as_posix()
            if skip is None or not skip(relative, False):
                found.append(relative)
    return sorted(found)


def runtime_distributions(runtime: Path) -> dict[str, metadata.Distribution]:
    """The runtime's installed distributions by canonical name, as they are now.

    importlib.metadata caches a directory's listing by its mtime, whose tick can
    hide an install made just after an earlier read; the caches are dropped first.
    """
    importlib.invalidate_caches()
    found: dict[str, metadata.Distribution] = {}
    for dist in metadata.distributions(path=[str(runtime / SITE_PACKAGES)]):
        name = canonicalize_name(dist.name)
        if name in found:
            raise ValueError(f"{runtime} holds two installations of {dist.name}")
        found[name] = dist
    return found


def excluded_requirements(
    dists: dict[str, metadata.Distribution], version: str
) -> list[str]:
    """Every shipped distribution's active requirement that names an excluded one.

    Markers are evaluated for the runtime's interpreter (version). A requirement
    is active when its marker holds with no extra, or with an extra that an
    active requirement asks of its distribution, followed to a fixed point.
    """
    excluded = {canonicalize_name(name) for name in EXCLUDED}
    shipped = {name: dist for name, dist in dists.items() if name not in excluded}
    extras = {name: {""} for name in shipped}
    environment = {
        "python_version": ".".join(version.split(".")[:2]),
        "python_full_version": version,
        "implementation_version": version,
    }
    found = set()
    changed = True
    while changed:
        changed = False
        for name, dist in shipped.items():
            for line in dist.requires or []:
                requirement = Requirement(line)
                marker = requirement.marker
                if marker is not None and not any(
                    marker.evaluate({**environment, "extra": extra})
                    for extra in extras[name]
                ):
                    continue
                target = canonicalize_name(requirement.name)
                wanted = {canonicalize_name(extra) for extra in requirement.extras}
                if target in excluded:
                    found.add(f"{dist.name}: {line}")
                elif target in extras and not wanted <= extras[target]:
                    extras[target] |= wanted
                    changed = True
    return sorted(found)


def runtime_exclusion(
    runtime: Path, dists: dict[str, metadata.Distribution], version: str
) -> Callable[[str, bool], bool]:
    """The runtime walk's skip for EXCLUDED, once both of its checks hold.

    (i) Every excluded name is installed, so the list cannot rot silently. (ii)
    No shipped distribution requires an excluded one (excluded_requirements).
    Every file an excluded distribution's RECORD lists is skipped, its .pth and
    Scripts entry points included, and so is each top-level site-packages
    directory that only excluded distributions own files in, stray caches
    included. A shared directory (Scripts, __pycache__, a namespace package)
    loses only the excluded files; a file two sides both own fails.
    """
    excluded = {canonicalize_name(name) for name in EXCLUDED}
    missing = sorted(name for name in EXCLUDED if canonicalize_name(name) not in dists)
    if missing:
        raise ValueError(
            f"Excluded distributions missing from {runtime}'s metadata: {missing}"
        )
    requiring = excluded_requirements(dists, version)
    if requiring:
        raise ValueError(f"Shipped distributions require excluded ones: {requiring}")
    site = runtime / SITE_PACKAGES
    directories: defaultdict[str, set[str]] = defaultdict(set)
    files: defaultdict[str, set[str]] = defaultdict(set)
    for name, dist in dists.items():
        if dist.files is None and name in excluded:
            raise ValueError(f"Excluded {dist.name} has no RECORD in {runtime}")
        for entry in dist.files or []:
            path = Path(os.path.normpath(site / entry))
            if not path.is_relative_to(runtime):
                if name in excluded:
                    raise ValueError(f"Excluded {dist.name} installed {path}")
                continue
            files[path.relative_to(runtime).as_posix().casefold()].add(name)
            if len(entry.parts) > 1 and entry.parts[0] != "..":
                directories[f"{SITE_PACKAGES}/{entry.parts[0]}".casefold()].add(name)
    shared = sorted(
        path for path, names in files.items() if names & excluded and names - excluded
    )
    if shared:
        raise ValueError(f"Excluded and shipped distributions share {shared[:10]}")
    skipped_files = {path for path, names in files.items() if names <= excluded}
    skipped_directories = {
        path for path, names in directories.items() if names <= excluded
    }

    def skip(relative: str, is_dir: bool) -> bool:
        return relative.casefold() in (skipped_directories if is_dir else skipped_files)

    return skip


def copy_verified(pair: tuple[Path, Path, bool]) -> tuple[Path, str, int]:
    """copy2 one installed file (mtimes kept, so runtime .pyc stay valid)."""
    source, target, reuse = pair
    target.parent.mkdir(parents=True, exist_ok=True)
    source_hash = hash_file(source)
    if target.exists():
        if os.path.samefile(source, target):
            raise ValueError(
                f"Destination shares a source file instead of owning a copy: {target}"
            )
        if reuse and hash_file(target) == source_hash:
            return target, source_hash, source.stat().st_size
    shutil.copy2(source, target)
    if hash_file(target) != source_hash:
        raise OSError(f"Copied bytes differ: {source}")
    return target, source_hash, source.stat().st_size


def replace_verified(target: Path, data: bytes, sha256: str) -> None:
    """Write target.tmp, swap it in with os.replace, then verify the target.

    os.replace swaps the directory entry, so a hard-linked alias of the old
    target is never written through.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    if hash_file(target) != sha256:
        raise OSError(f"Written bytes differ from the planned record: {target}")


def probe_unlocked(targets: Iterable[Path]) -> None:
    """Fail before any write if an existing target cannot be opened for writing.

    A DLL or PYD mapped by a running backend refuses write access, as does a
    read-only file. Every such path is reported; no process is ever stopped.
    """
    locked = []
    for target in targets:
        if not target.exists():
            continue
        try:
            os.close(os.open(target, os.O_RDWR | os.O_BINARY))
        except PermissionError:
            locked.append(str(target))
    if locked:
        raise RuntimeError(
            "Package files are locked or read-only; close chaiNNer-C and retry "
            "(nothing was written): " + ", ".join(locked)
        )


def git_snapshot(project: Path) -> tuple[set[str], dict[str, object]]:
    """Tracked backend/src paths and provenance: the packager's only git calls.

    --no-optional-locks keeps `git status` from refreshing the index, so even
    this read leaves the repository untouched.
    """

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(project), *arguments],
            capture_output=True,
            check=False,
            encoding="utf-8",
        )
        if result.returncode:
            raise RuntimeError(
                f"git {' '.join(arguments)} failed in {project}: {result.stderr.strip()}"
            )
        return result.stdout

    tracked = {
        name.removeprefix("backend/src/")
        for name in git("ls-files", "-z", "--", "backend/src").split("\0")
        if name
    }
    provenance: dict[str, object] = {
        "git_head": git("rev-parse", "HEAD").strip(),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "backend_dirty": bool(git("status", "--porcelain", "--", "backend/src")),
    }
    return tracked, provenance


def backend_inventory(root: Path, tracked: set[str]) -> dict[str, bytes]:
    """Read backend/src exactly as git tracks it, plus the native binaries.

    __pycache__ directories are excluded. Any other untracked file, or a
    tracked file missing from disk, is refused so nothing stray is packaged.
    """
    found = set(walk(root, cache_skip))
    expected = tracked | NATIVE_BINARIES
    untracked, missing = sorted(found - expected), sorted(expected - found)
    if untracked or missing:
        raise ValueError(
            "backend/src must equal git ls-files plus the native binaries; "
            f"untracked: {untracked[:10]}, missing: {missing[:10]}"
        )
    return {relative: (root / relative).read_bytes() for relative in sorted(found)}
