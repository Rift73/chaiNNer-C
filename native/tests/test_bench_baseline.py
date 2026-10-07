from __future__ import annotations

import os

import bench_baseline
import pytest


def make_tree(root):
    (root / "pkg" / "__pycache__").mkdir(parents=True)
    (root / "run.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "pkg" / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pkg" / "__pycache__" / "a.cpython-311.pyc").write_bytes(b"cache")
    return root


def test_freeze_copies_sources_without_caches_and_protects_them(tmp_path):
    source = make_tree(tmp_path / "src")
    copy = bench_baseline.freeze(source, "B-test", root=tmp_path / "baselines")
    files = sorted(
        p.relative_to(copy).as_posix() for p in copy.rglob("*") if p.is_file()
    )
    assert files == ["pkg/a.py", "run.py"]
    assert not os.access(copy / "run.py", os.W_OK)
    bench_baseline.verify(tmp_path / "baselines" / "B-test")


def test_freeze_refuses_to_overwrite(tmp_path):
    source = make_tree(tmp_path / "src")
    bench_baseline.freeze(source, "B-test", root=tmp_path / "baselines")
    with pytest.raises(FileExistsError):
        bench_baseline.freeze(source, "B-test", root=tmp_path / "baselines")


def test_verify_detects_modified_baseline(tmp_path):
    source = make_tree(tmp_path / "src")
    copy = bench_baseline.freeze(source, "B-test", root=tmp_path / "baselines")
    target = copy / "pkg" / "a.py"
    target.chmod(0o666)
    target.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"pkg/a\.py"):
        bench_baseline.verify(tmp_path / "baselines" / "B-test")


def test_verify_reports_total_count_of_changed_files(tmp_path):
    source = make_tree(tmp_path / "src")
    copy = bench_baseline.freeze(source, "B-test", root=tmp_path / "baselines")
    (copy / "run.py").chmod(0o666)
    (copy / "run.py").unlink()
    (copy / "extra.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"\(2 files\): \['extra\.py', 'run\.py'\]"):
        bench_baseline.verify(tmp_path / "baselines" / "B-test")


def test_changed_paths_lists_differing_and_one_sided_keys_sorted():
    a = {"same": "1", "changed": "1", "only-a": "1"}
    b = {"same": "1", "changed": "2", "b-only": "1"}
    assert bench_baseline.changed_paths(a, b) == ["b-only", "changed", "only-a"]
    assert bench_baseline.changed_paths(a, dict(a)) == []
