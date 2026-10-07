"""Freeze and verify immutable backend trees used as benchmark baselines.

Usage: python native/tools/bench_baseline.py SOURCE NAME
Copies SOURCE (a backend src tree) to native/reports/baselines/NAME/src, records
SHA-256 hashes in NAME/manifest.json and marks every copied file read-only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from verify_framework_runtime import source_hashes

PROJECT = Path(__file__).resolve().parents[2]
BASELINES = PROJECT / "native/reports/baselines"


def freeze(source: Path, name: str, root: Path = BASELINES) -> Path:
    """Copy a backend tree to root/name/src and return the copy's path."""
    target = root / name
    if target.exists():
        raise FileExistsError(f"Baseline already exists: {target}")
    expected = source_hashes(source)
    if not expected:
        raise ValueError(f"Empty backend tree: {source}")
    copy = target / "src"
    shutil.copytree(source, copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if source_hashes(copy) != expected:
        raise RuntimeError(f"Baseline copy differs from {source}")
    for path in copy.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD)
    manifest = {
        "name": name,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": str(source),
        "git_head": git_head(),
        "files": expected,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return copy


def changed_paths(a: dict[str, str], b: dict[str, str]) -> list[str]:
    """Sorted keys whose values differ or that only one side has."""
    return sorted(key for key in a.keys() | b.keys() if a.get(key) != b.get(key))


def verify(baseline: Path) -> None:
    """Raise if any file under baseline/src differs from baseline/manifest.json."""
    recorded = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    changed = changed_paths(source_hashes(baseline / "src"), recorded["files"])
    if changed:
        raise RuntimeError(
            f"Baseline {baseline.name} changed ({len(changed)} files): {changed[:10]}"
        )


def git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("name")
    args = parser.parse_args()
    print(freeze(args.source.resolve(), args.name))


if __name__ == "__main__":
    main()
