"""Runtime scheduling changes reuse only complete, compatible calibrated plans."""

from __future__ import annotations

import hashlib
import json
import shutil
from types import SimpleNamespace
from typing import cast

import pytest

from api import NodeContext
from nodes.impl.pytorch.vosr2_stage_cache import migrate_stage_cache, stage_identity
from nodes.impl.pytorch.vosr2_trt_models import STAGES


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def legacy(tmp_path):
    identity: dict = {
        "recipe": "accepted",
        "checkpoint": "weights",
        "runtime": "11.2",
        "pipeline_sources": {
            "vosr2_trt_models.py": "model",
            "vosr2_trt_restore.py": "old",
        },
    }
    source = tmp_path / key(identity)
    source.mkdir()
    engines = {}
    for name in STAGES:
        data = name.encode()
        (source / f"{name}.plan").write_bytes(data)
        engines[name] = hashlib.sha256(data).hexdigest()
    (source / "manifest.json").write_text(
        json.dumps({"identity": identity, "engines": engines})
    )
    return identity, source


def test_runtime_changes_do_not_invalidate_plan_but_model_and_contract_changes_do(
    tmp_path,
):
    old, _ = legacy(tmp_path)
    identity = stage_identity(old)
    old["pipeline_sources"]["vosr2_trt_restore.py"] = "new"
    assert stage_identity(old) == identity
    old["pipeline_sources"]["vosr2_trt_models.py"] = "different-model"
    assert stage_identity(old) != identity
    assert stage_identity({**identity, "runtime_contract": 2}) != identity


def test_migration_preserves_old_manifest_and_plans_and_is_idempotent(tmp_path):
    old, source = legacy(tmp_path)
    identity = stage_identity(old)
    destination = tmp_path / key(identity)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    context = cast(NodeContext, SimpleNamespace(suspend=lambda: None))
    assert migrate_stage_cache(context, tmp_path, destination, identity)
    assert not migrate_stage_cache(context, tmp_path, destination, identity)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert all(
        (destination / f"{name}.plan").read_bytes() == name.encode() for name in STAGES
    )
    assert (
        json.loads((destination / "manifest.json").read_text())["identity"] == identity
    )


@pytest.mark.parametrize(
    "field", ["recipe", "checkpoint", "runtime", "runtime_contract"]
)
def test_migration_rejects_other_models_recipes_and_runtimes(tmp_path, field):
    old, _ = legacy(tmp_path)
    identity = {**stage_identity(old), field: "different"}
    destination = tmp_path / key(identity)
    assert not migrate_stage_cache(
        cast(NodeContext, SimpleNamespace(suspend=lambda: None)),
        tmp_path,
        destination,
        identity,
    )
    assert not destination.exists()


def test_migration_sets_a_corrupt_set_aside_so_a_fresh_build_follows(tmp_path):
    old, source = legacy(tmp_path)
    (source / "dino.plan").write_bytes(b"corrupt")
    identity = stage_identity(old)
    destination = tmp_path / key(identity)
    context = cast(NodeContext, SimpleNamespace(suspend=lambda: None))
    assert not migrate_stage_cache(context, tmp_path, destination, identity)
    assert not destination.exists() and not source.exists()
    (quarantine,) = tmp_path.glob(f"quarantine-{source.name}-*")
    assert (quarantine / "dino.plan").read_bytes() == b"corrupt"
    # Nothing is left to migrate: the next run builds instead of failing again.
    assert not migrate_stage_cache(context, tmp_path, destination, identity)


def test_migration_takes_an_intact_copy_after_setting_a_corrupt_one_aside(tmp_path):
    old, source = legacy(tmp_path)
    # Another runtime's copy of the same plans
    sources = {**old["pipeline_sources"], "vosr2_scene.py": "other"}
    copy = {**old, "pipeline_sources": sources}
    other = tmp_path / key(copy)
    shutil.copytree(source, other)
    manifest = json.loads((other / "manifest.json").read_text())
    (other / "manifest.json").write_text(json.dumps({**manifest, "identity": copy}))
    first, second = sorted([source, other], key=lambda path: path.name)
    (first / "dino.plan").write_bytes(b"corrupt")
    identity = stage_identity(old)
    destination = tmp_path / key(identity)
    context = cast(NodeContext, SimpleNamespace(suspend=lambda: None))
    assert migrate_stage_cache(context, tmp_path, destination, identity)
    assert not first.exists() and second.exists()
    assert len(list(tmp_path.glob("quarantine-*"))) == 1
    assert (destination / "dino.plan").read_bytes() == b"dino"
    migrated = json.loads((destination / "manifest.json").read_text())
    assert migrated["migrated_from"]["directory"] == str(second)


def test_contract_bump_cannot_relabel_legacy_plans(tmp_path, monkeypatch):
    from nodes.impl.pytorch import vosr2_stage_cache as cache

    old, _ = legacy(tmp_path)
    monkeypatch.setattr(cache, "RUNTIME_CONTRACT", 2)
    identity = stage_identity(old)
    assert identity["runtime_contract"] == 2
    assert stage_identity(old, legacy=True)["runtime_contract"] == 1
    assert not migrate_stage_cache(
        cast(NodeContext, SimpleNamespace(suspend=lambda: None)),
        tmp_path,
        tmp_path / key(identity),
        identity,
    )


def test_migration_rejects_ambiguous_compatible_recipes(tmp_path):
    old, source = legacy(tmp_path)
    manifest = json.loads((source / "manifest.json").read_text())
    old["pipeline_sources"]["vosr2_trt_restore.py"] = "different-runtime"
    other = tmp_path / key(old)
    other.mkdir()
    manifest["identity"] = old
    manifest["engines"]["dino"] = "another-engine"
    (other / "manifest.json").write_text(json.dumps(manifest))
    identity = stage_identity(old)
    with pytest.raises(RuntimeError, match="Delete one of them and run again") as error:
        migrate_stage_cache(
            cast(NodeContext, SimpleNamespace(suspend=lambda: None)),
            tmp_path,
            tmp_path / key(identity),
            identity,
        )
    assert str(source) in str(error.value) and str(other) in str(error.value)
