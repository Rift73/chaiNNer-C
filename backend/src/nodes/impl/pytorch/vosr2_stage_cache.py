"""Separate stage-plan identity from runtime scheduling and migrate verified caches."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from api import NodeContext

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


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
                "Ambiguous compatible VOSR caches contain different engine plans."
            )
        candidates.append((path, manifest))
    if not candidates:
        return False
    path, manifest = candidates[0]
    engines = manifest["engines"]
    for name in STAGES:
        context.suspend()
        source = path.parent / f"{name}.plan"
        if not source.is_file() or _digest(source) != engines[name]:
            raise RuntimeError(f"Cannot migrate corrupt VOSR {name} engine: {source}")
    with tempfile.TemporaryDirectory(prefix="migrating-", dir=root) as work:
        staged = Path(work) / "cache"
        staged.mkdir()
        for name in STAGES:
            source = path.parent / f"{name}.plan"
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
