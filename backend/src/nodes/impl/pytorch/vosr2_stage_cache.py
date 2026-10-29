"""Separate stage-plan identity from runtime scheduling and migrate verified caches."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from api import NodeContext

from .vosr2_plan_file import compressed, plain, set_aside, stored
from .vosr2_trt_models import STAGES

# Bump when stage inputs/outputs or preprocessing semantics change.
RUNTIME_CONTRACT = 1

_RUNTIME_SOURCES = {
    "vosr2_trt_pipeline.py",
    "vosr2_trt_execution.py",
    "vosr2_trt_restore.py",
    "vosr2_scene.py",
    "vosr2_context.py",
}


# migrate_stage_cache's temporary folder: the prefix and tempfile's eight characters
MIGRATING = "migrating-"
_MIGRATING_NAME = re.compile(re.escape(MIGRATING) + "[a-z0-9_]{8}")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def intact(context: NodeContext, folder: Path, manifest: dict, name: str) -> bool:
    """Whether folder holds the stage's plan that manifest records."""
    context.suspend()
    source = stored(folder, manifest, name)
    return source.is_file() and _digest(source) == manifest["engines"][name]


def stage_identity(identity: dict, *, legacy: bool = False) -> dict:
    result = {**identity}
    # Unversioned legacy manifests always mean contract 1, even after a bump.
    result.setdefault("runtime_contract", 1 if legacy else RUNTIME_CONTRACT)
    result["pipeline_sources"] = {
        name: digest
        for name, digest in identity["pipeline_sources"].items()
        if name not in _RUNTIME_SOURCES
    }
    return result


def migrate_stage_cache(
    context: NodeContext, root: Path, destination: Path, identity: dict
) -> bool:
    """Caller owns the cache lock. Preserve old plans; publish only complete sets."""
    if destination.exists():
        return False
    candidates: list[tuple[Path, dict]] = []
    for path in sorted(root.glob("*/manifest.json")):
        context.suspend()
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(manifest, dict):
            continue
        old = manifest.get("identity")
        if not isinstance(old, dict) or not isinstance(
            old.get("pipeline_sources"), dict
        ):
            continue
        old_key = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
        if path.parent.name != old_key or stage_identity(old, legacy=True) != identity:
            continue
        engines = manifest.get("engines")
        if not isinstance(engines, dict) or set(engines) != set(STAGES):
            continue
        if candidates and candidates[0][1]["engines"] != engines:
            raise RuntimeError(
                "Two cached VOSR engine sets fit this version but hold different "
                f"plans: {candidates[0][0].parent} and {path.parent}. Delete one of "
                "them and run again."
            )
        candidates.append((path, manifest))
    # A corrupt set goes aside; with none intact, the caller builds a fresh one.
    for path, manifest in candidates:
        if all(intact(context, path.parent, manifest, name) for name in STAGES):
            break
        set_aside(path.parent, "BF16")
    else:
        return False
    engines = manifest["engines"]
    with tempfile.TemporaryDirectory(prefix=MIGRATING, dir=root) as work:
        staged = Path(work) / "cache"
        staged.mkdir()
        for name in STAGES:
            source = stored(path.parent, manifest, name)
            target = staged / source.name
            os.link(source, target)
            if _digest(target) != engines[name]:
                raise RuntimeError(f"VOSR {name} engine changed during migration.")
        (staged / "manifest.json").write_text(
            json.dumps(
                {
                    **manifest,
                    "identity": identity,
                    "migrated_from": {
                        "directory": str(path.parent),
                        "manifest_sha256": _digest(path),
                    },
                }
            ),
            encoding="utf-8",
        )
        os.replace(staged, destination)
    return True


def migration_leftover(path: Path) -> bool:
    """Whether path is what a migration killed midway left: migrate_stage_cache's
    temporary folder, empty or holding only its `cache` folder of stage plans and their
    manifest."""
    if not _MIGRATING_NAME.fullmatch(path.name) or not path.is_dir():
        return False
    entries = list(path.iterdir())
    if not entries:
        return True
    if len(entries) != 1:
        return False
    staged = entries[0]
    if (
        staged.name != "cache"
        or staged.is_symlink()
        or staged.is_junction()
        or not staged.is_dir()
    ):
        return False
    names = {"manifest.json"}
    names.update(plain(staged, name).name for name in STAGES)
    names.update(compressed(staged, name).name for name in STAGES)
    return all(
        file.name in names and file.is_file() and not file.is_symlink()
        for file in staged.iterdir()
    )
