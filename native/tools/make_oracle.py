"""Copy the installed chaiNNer's backend into the oracle tree and patch it.

The oracle (spec 3) is the installed app's resources/src, run on this project's
runtime and wheels. Each run copies it afresh, without __pycache__, to
native/build/oracle/src, applies the one tracked native/tests/oracle/compat.diff
there with git apply, and adds chaiNNer-C's C chainner_ext from backend/src
(verify_runtime.CHAINNER_EXT: the package, with chainner_native.dll beside its
pyd). It records the source hashes, the app version, the diff hash and the four
chainner_ext hashes in native/build/oracle/oracle.json. The installed app and
backend/src are only read, and nothing is written outside native/build/oracle.

chainner_ext is neither a compat.diff hunk nor a site-packages install, and
backend/src is never on the oracle's sys.path. The runtime verifiers refuse an
oracle whose chainner_ext hashes differ from the package manifest's, so run this
again after every native build that a package ships.

compat.diff paths are relative to the source root (a/nodes/..., b/nodes/...). An
empty compat.diff applies nothing; any other content must apply in full.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import package_files
from verify_runtime import CHAINNER_EXT, ORACLE, ORACLE_RECORD, digest, write_json

PROJECT = Path(__file__).resolve().parents[2]
INSTALLED_APP = Path(
    os.environ["LOCALAPPDATA"], "chaiNNer", "app-0.25.1-nightly2025-10-21"
)
BACKEND = PROJECT / "backend/src"
COMPAT = PROJECT / "native/tests/oracle/compat.diff"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the sources exist and the diff applies; write nothing",
    )
    return parser.parse_args()


def git_apply(compat: Path, root: Path, check: bool) -> None:
    """Apply compat to root/src, or only check that it applies; all or nothing.

    root is git's whole working area: the repository above it is hidden, so git
    refuses a patch path that leaves root instead of resolving it against, or
    silently skipping it in, this project. core.autocrlf is off, so the patched
    files keep their line endings byte for byte.
    """
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "apply",
            "--directory=src",
            *(["--check"] if check else []),
            str(compat),
        ],
        cwd=root,
        env={**os.environ, "GIT_CEILING_DIRECTORIES": str(root.parent)},
        capture_output=True,
        check=False,
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(
            f"{compat.name} does not apply to {root / 'src'}: {result.stderr.strip()}"
        )


def make_oracle(
    app: Path, backend: Path, compat: Path, oracle: Path, check: bool
) -> dict[str, Any]:
    """Copy app's resources/src to oracle/src, patch it, add chainner_ext, record it.

    backend is chaiNNer-C's backend/src, where the four CHAINNER_EXT files are
    built. Every write stays inside oracle, which must be a real absolute path.
    With check, nothing is written: the sources are checked and the diff is
    checked against the installed source.
    """
    source = app / "resources/src"
    if not source.is_dir():
        raise ValueError(f"Installed backend source missing: {source}")
    missing = [
        str(backend / name) for name in CHAINNER_EXT if not (backend / name).is_file()
    ]
    if missing:
        raise ValueError(
            f"chaiNNer-C's chainner_ext missing (build it first): {', '.join(missing)}"
        )
    package = json.loads(
        (app / "resources/app/package.json").read_text(encoding="utf-8")
    )
    patch = compat.read_bytes()
    if oracle.resolve() != oracle:
        raise ValueError(f"Oracle directory is not a real absolute path: {oracle}")
    names = package_files.walk(source, package_files.cache_skip)
    if not names:
        raise ValueError(f"Installed backend source is empty: {source}")
    target = oracle / "src"
    if check:
        files = {name: package_files.hash_file(source / name) for name in names}
        extension = {
            name: package_files.hash_file(backend / name) for name in CHAINNER_EXT
        }
    else:
        if target.exists():
            package_files.walk(target)  # Refuse a link before deleting through it.
        # A failed run leaves no record, never one that describes another tree.
        (oracle / ORACLE_RECORD).unlink(missing_ok=True)
        if target.exists():
            shutil.rmtree(target)
        files = {
            name: package_files.copy_verified((source / name, target / name, False))[1]
            for name in names
        }
        extension = {
            name: package_files.copy_verified((backend / name, target / dest, False))[1]
            for name, dest in CHAINNER_EXT.items()
        }
    if patch:
        git_apply(compat, source.parent if check else oracle, check)
    record = {
        "app_version": package["version"],
        "app_directory": app.name,
        "source": str(source),
        "source_files": files,
        "source_sha256": digest(files),
        "compat_diff_sha256": package_files.sha(patch),
        "chainner_ext": extension,
        "utc": datetime.now(UTC).isoformat(),
    }
    if not check:
        write_json(oracle / ORACLE_RECORD, record)
    return record


def main() -> int:
    args = parse_arguments()
    record = make_oracle(INSTALLED_APP, BACKEND, COMPAT, ORACLE, args.check)
    summary = {
        "result": "checked" if args.check else "written",
        "record": None if args.check else str(ORACLE / ORACLE_RECORD),
        "app_version": record["app_version"],
        "source_files": len(record["source_files"]),
        "source_sha256": record["source_sha256"],
        "compat_diff_sha256": record["compat_diff_sha256"],
        "chainner_ext": record["chainner_ext"],
    }
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Oracle failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
