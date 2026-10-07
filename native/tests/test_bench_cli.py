from __future__ import annotations

import copy
import hashlib
import http.client
import itertools
import json
import math
import os
import subprocess
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, cast

import bench
import bench_backend
import bench_baseline
import bench_oracle
import cv2
import numpy as np
import pytest
from bench_fixtures import Fixtures


def test_require_identical_rejects_different_trees():
    with pytest.raises(ValueError, match=r"nodes/x\.py"):
        bench.require_identical(
            {"nodes/x.py": "1", "run.py": "2"}, {"nodes/x.py": "9", "run.py": "2"}
        )


def test_require_identical_reports_total_count():
    a = {f"f{i:02}.py": "1" for i in range(12)}
    with pytest.raises(ValueError, match=r"\(12 files differ\)") as error:
        bench.require_identical(a, {})
    assert "f09.py" in str(error.value)
    assert "f10.py" not in str(error.value)


def test_require_identical_accepts_same_tree():
    bench.require_identical({"run.py": "2"}, {"run.py": "2"})


def test_verify_if_baseline_rejects_edited_tree(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "run.py").write_text("x = 1\n", encoding="utf-8")
    copy = bench_baseline.freeze(source, "B-test", root=tmp_path / "baselines")
    bench.verify_if_baseline(copy)
    (copy / "run.py").chmod(0o666)
    (copy / "run.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"run\.py"):
        bench.verify_if_baseline(copy)


def test_verify_if_baseline_ignores_ordinary_trees(tmp_path):
    bench.verify_if_baseline(tmp_path)


def fake_python(directory: Path) -> Path:
    """An existing file named python.exe, which is all compare() checks of one."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "python.exe"
    path.write_bytes(b"MZ")
    return path


def refused_start(tmp_path, monkeypatch):
    """Redirect compare()'s run directories into tmp_path; return a lister of them.

    A refused start must create no bench-* directory and start no work. The
    harness variables are cleared and the default interpreter exists, so only
    the refusal under test can fire.
    """
    monkeypatch.delenv("CHAINNER_C_ITEM_WINDOW", raising=False)
    monkeypatch.delenv("CHAINNER_C_PROFILE", raising=False)
    monkeypatch.delenv("CHAINNER_C_ISA", raising=False)
    reports = tmp_path / "project" / "native" / "reports"
    diagnostics = tmp_path / "diagnostics"
    reports.mkdir(parents=True)
    diagnostics.mkdir()
    monkeypatch.setattr(bench, "PROJECT", tmp_path / "project")
    monkeypatch.setattr(bench, "DIAGNOSTICS", diagnostics)
    monkeypatch.setattr(bench, "PYTHON", fake_python(tmp_path / "packaged"))
    monkeypatch.setattr(
        bench, "prepare", lambda media: pytest.fail("a refused run must start no work")
    )

    def created():
        return {p.name for root in (reports, diagnostics) for p in root.glob("bench-*")}

    return created


def test_record_noise_refuses_short_runs_before_creating_anything(
    tmp_path, monkeypatch
):
    created = refused_start(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="at least 6 pairs"):
        bench.compare(tmp_path / "a", tmp_path / "b", ["gaussian"], 1, True)
    assert created() == set()


def test_odd_repeats_are_refused_before_creating_anything(tmp_path, monkeypatch):
    created = refused_start(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="even number of pairs"):
        bench.compare(tmp_path / "a", tmp_path / "b", ["gaussian"], 3, False)
    assert created() == set()


@pytest.mark.parametrize("value", ["1", "0", ""])
def test_profile_variable_without_the_flag_is_refused_before_creating_anything(
    tmp_path, monkeypatch, value
):
    created = refused_start(tmp_path, monkeypatch)
    # Any value: the timer reads only "1", but a set variable means a mistaken shell.
    monkeypatch.setenv("CHAINNER_C_PROFILE", value)
    with pytest.raises(ValueError, match="CHAINNER_C_PROFILE is set"):
        bench.compare(tmp_path / "a", tmp_path / "b", ["gaussian"], 2, False)
    assert created() == set()


@pytest.mark.parametrize("value", ["avx2", "scalar", ""])
@pytest.mark.parametrize(
    ("profile", "record_noise"),
    [(False, False), (True, False), (False, True)],
    ids=["gate", "profile", "record-noise"],
)
def test_isa_variable_in_the_harness_is_refused_before_creating_anything(
    tmp_path, monkeypatch, value, profile, record_noise
):
    created = refused_start(tmp_path, monkeypatch)
    # Any value and any run kind: it would reach both backends and silently cap them.
    monkeypatch.setenv("CHAINNER_C_ISA", value)
    # Six repeats keep the --record-noise row clear of the MIN_PAIRS refusal.
    with pytest.raises(ValueError, match="CHAINNER_C_ISA is set"):
        bench.compare(
            tmp_path / "a",
            tmp_path / "b",
            ["gaussian"],
            6,
            record_noise,
            profile=profile,
        )
    assert created() == set()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        # Windows reads the name without case, so this would turn B's timer on.
        ({"env_b": {"chainner_c_profile": "1"}}, "may not set CHAINNER_C_PROFILE"),
        (
            {"env_a": {"chainner_c_isa": "avx2"}},
            "side environments come from parse_env",
        ),
        ({"env_b": {"PYTHONPATH": "x"}}, "may not set PYTHONPATH"),
        # It would silently override the timer of a profile run.
        (
            {"env_b": {"CHAINNER_C_PROFILE": "0"}, "profile": True},
            "may not set CHAINNER_C_PROFILE",
        ),
    ],
)
def test_compare_refuses_side_environments_parse_env_would_not_return(
    tmp_path, monkeypatch, options, message
):
    created = refused_start(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=message):
        bench.compare(tmp_path / "a", tmp_path / "b", ["gaussian"], 2, False, **options)
    assert created() == set()


def add_timer(tree: Path) -> Path:
    """Give a tree the timer module that has_timer() looks for."""
    (tree / "nodes" / "impl").mkdir(parents=True)
    (tree / "nodes" / "impl" / "native_profile.py").write_text("", encoding="utf-8")
    return tree


def test_profile_refuses_record_noise_and_trees_without_the_timer(
    tmp_path, monkeypatch
):
    created = refused_start(tmp_path, monkeypatch)
    a, b = identical_trees(tmp_path)
    assert not bench.has_timer(a)
    assert not bench.has_timer(b)
    with pytest.raises(ValueError, match="--profile needs a tree with the timer"):
        bench.compare(a, b, ["gaussian"], 6, False, profile=True)
    assert bench.has_timer(add_timer(b))
    with pytest.raises(ValueError, match="--profile refuses --record-noise"):
        bench.compare(a, b, ["gaussian"], 6, True, profile=True)
    assert created() == set()


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["CHAINNER_C_ISA=avx2"], {"CHAINNER_C_ISA": "avx2"}),
        (["X"], ValueError),
        (["=v"], ValueError),
        (["CHAINNER_C_ITEM_WINDOW=1"], ValueError),
        (["CHAINNER_C_PROFILE=1"], ValueError),
        (["CUDA_VISIBLE_DEVICES=0"], ValueError),
        (["TEMP=x"], ValueError),
        (None, {}),
        (["CHAINNER_C_ISA=avx2", "X=a=b"], {"CHAINNER_C_ISA": "avx2", "X": "a=b"}),
        # Windows reads names without case, so they are kept as os.environ keeps them.
        (["chainner_c_isa=avx2"], {"CHAINNER_C_ISA": "avx2"}),
        (["chainner_c_profile=1"], ValueError),
        (["PYTHONPATH=x"], ValueError),
        (["PYTHONHOME=x"], ValueError),
        (["X=1", "x=2"], ValueError),  # One name twice.
    ],
)
def test_env_options_parse_and_refuse_reserved_names(values, expected):
    if expected is ValueError:
        with pytest.raises(ValueError, match="--env-a/--env-b"):
            bench.parse_env(values)
    else:
        assert bench.parse_env(values) == expected


@pytest.mark.parametrize(
    ("env_a", "env_b"),
    [({"CHAINNER_C_ISA": "avx2"}, None), (None, {"CHAINNER_C_ISA": "avx2"})],
    ids=["env_a", "env_b"],
)
def test_record_noise_refuses_a_per_side_environment(
    tmp_path, monkeypatch, env_a, env_b
):
    created = refused_start(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="--record-noise refuses --env-a and --env-b"):
        bench.compare(
            *identical_trees(tmp_path),
            ["gaussian"],
            6,
            True,
            env_a=env_a,
            env_b=env_b,
        )
    assert created() == set()


class FakeBenchBackend:
    """Stands in for bench.Backend in compare(); starts no process.

    envs and pythons hold each backend's side environment and interpreter in
    creation order (A, then B), and png_modes the PNG mode of each trial that
    stub_compare's run_trial made; stub_compare clears them.
    """

    envs: ClassVar[list[dict[str, str]]] = []
    pythons: ClassVar[list[Path]] = []
    png_modes: ClassVar[list[str]] = []

    def __init__(self, label, source, root, ffmpeg, ffprobe, env, *, python):
        self.label = label
        self.schemas = {}
        FakeBenchBackend.envs.append(env)
        FakeBenchBackend.pythons.append(python)
        self.info = {
            "schema_sha256": "same",
            "owned_process_exited": True,
            "owned_job_closed": True,
            "backend_unchanged": True,
            "interpreter_unchanged": True,
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


FILES = {"a.png": "1", "b.png": "2", "c.png": "3"}
STATE = {
    "controls": [{"event": "chain-start", "data": {"nodes": ["load", "save"]}}],
    "final_state": [
        {"event": "node-finish", "data": {"nodeId": "load"}},
        {"event": "node-finish", "data": {"nodeId": "save"}},
    ],
}


def trial_record(label, case, pair, files=FILES, final_state=STATE):
    return {
        "backend": label,
        "case": case,
        "trial": "warmup" if pair == 0 else f"trial-{pair}",
        "pair": pair,
        "seconds": 1.0,
        "count": 32,
        "unit": "images/s",
        "throughput": 32.0,
        "job_cpu_seconds": 4.0,
        "background_busy_percent": 2.0,
        "files": files,
        "final_state": final_state,
    }


# A timer table as a profiled trial records it (P6).
TABLE = {
    "cn_box_separable": {
        "calls": 2,
        "inclusive_ns": 70,
        "exclusive_ns": 50,
        "max_ns": 30,
    }
}


def stub_compare(tmp_path, monkeypatch, make_record=trial_record):
    """Redirect compare() into tmp_path with fakes; return its run_trial call log.

    As the real run_trial, a trial records a native_profile table only when its
    side is profiled. The default interpreter is a fake python.exe in tmp_path.
    """
    monkeypatch.delenv("CHAINNER_C_ITEM_WINDOW", raising=False)
    monkeypatch.delenv("CHAINNER_C_PROFILE", raising=False)
    monkeypatch.delenv("CHAINNER_C_ISA", raising=False)
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    FakeBenchBackend.envs.clear()
    FakeBenchBackend.pythons.clear()
    FakeBenchBackend.png_modes.clear()
    calls = []

    def run_trial(backend, case, pair, fixtures, media, profiled, png_compare):
        calls.append(backend.label)
        FakeBenchBackend.png_modes.append(png_compare)
        record = make_record(backend.label, case, pair)
        return {**record, "native_profile": TABLE if profiled else None}

    monkeypatch.setattr(bench, "Backend", FakeBenchBackend)
    monkeypatch.setattr(
        bench, "prepare", lambda media: Fixtures(tmp_path / "fixtures", {"assets": {}})
    )
    monkeypatch.setattr(bench, "run_trial", run_trial)
    monkeypatch.setattr(bench, "verify_unchanged", lambda fixtures: None)
    monkeypatch.setattr(bench, "environment", lambda: {"busy_percent_before": 1.0})
    monkeypatch.setattr(bench, "PROJECT", tmp_path / "project")
    monkeypatch.setattr(bench, "DIAGNOSTICS", diagnostics)
    monkeypatch.setattr(bench, "NOISE", tmp_path / "noise-floor.json")
    monkeypatch.setattr(bench, "PYTHON", fake_python(tmp_path / "packaged"))
    return calls


def only_result(tmp_path):
    (path,) = (tmp_path / "project" / "native" / "reports").glob("bench-*/result.json")
    return json.loads(path.read_text(encoding="utf-8"))


def write_noise(tmp_path: Path, entries: dict, cpu_floors: dict | None = None) -> None:
    """Seed the redirected noise-floor.json that compare() reads.

    entries are the shared throughput floors; cpu_floors, when given, the CPU
    floors per K setting.
    """
    document = {"floors": entries}
    if cpu_floors is not None:
        document["cpu_floors"] = cpu_floors
    (tmp_path / "noise-floor.json").write_text(json.dumps(document), encoding="utf-8")


LEGEND = (
    "Paired change = median of per-pair A/B time ratios \N{MINUS SIGN} 1; "
    "positive = B faster. Verdicts need \N{GREATER-THAN OR EQUAL TO}6 pairs; "
    "win = beyond both floors with a majority of pairs, regression = beyond the "
    "noise floor (within current A/A unless also beyond the current floor). "
    "Noise floor = largest |ln| of position-balanced pair means "
    "(B-first \N{MULTIPLICATION SIGN} A-first) across "
    "\N{GREATER-THAN OR EQUAL TO}3 A/A launches; current floor = the same over "
    "this K setting's A/A launches; both shown as upward/downward bounds."
)
CPU_LEGEND = (
    "CPU/item change = median of per-pair A/B CPU-per-item ratios "
    "\N{MINUS SIGN} 1; positive = B uses less CPU; CPU verdicts are information "
    "only and never confirmed."
)


def test_compare_alternates_backends_in_balanced_pairs(tmp_path, monkeypatch):
    calls = stub_compare(tmp_path, monkeypatch)
    # B has the timer, as every tree from Task 4 on; it is not profiled without --profile.
    run = bench.compare(
        tmp_path / "a", add_timer(tmp_path / "b"), ["gaussian"], 2, False
    )
    # Warm-up AB, pair 1 BA, pair 2 AB.
    assert calls == ["A", "B", "B", "A", "A", "B"]
    result = only_result(tmp_path)
    assert result["success"] is True
    assert [(t["backend"], t["pair"]) for t in result["trials"]] == [
        ("A", 0),
        ("B", 0),
        ("B", 1),
        ("A", 1),
        ("A", 2),
        ("B", 2),
    ]
    assert result["verdicts"] == {"gaussian": "unknown"}  # Fewer than MIN_PAIRS.
    assert result["cpu_verdicts"] == {"gaussian": "unknown"}
    assert result["current_floors"] == {}  # No floor is recorded.
    assert result["k_setting"] == "auto"
    # An ordinary run: no side environment, nothing profiled, both sides under the
    # packaged interpreter, PNGs compared by their bytes.
    assert result["profile"] is False
    assert FakeBenchBackend.envs == [{}, {}]
    assert FakeBenchBackend.pythons == [bench.PYTHON, bench.PYTHON]
    assert result["pythons"] == {"A": str(bench.PYTHON), "B": str(bench.PYTHON)}
    assert result["png_compare"] == "sha256"
    assert FakeBenchBackend.png_modes == ["sha256"] * 6
    assert [t["native_profile"] for t in result["trials"]] == [None] * 6
    # Not a cross-stack run: A's warm-up is both sides' reference, and no label.
    assert result["cross_stack"] is None
    summary = (run / "summary.md").read_text(encoding="utf-8")
    assert "- Label:" not in summary
    assert "## Cross-stack" not in summary
    assert not (tmp_path / "noise-floor.json").exists()


@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("kind", ["missing", "other-name", "directory"])
def test_interpreters_are_refused_unless_an_existing_python_exe(
    tmp_path, monkeypatch, side, kind
):
    created = refused_start(tmp_path, monkeypatch)
    if kind == "missing":
        python = tmp_path / "gone" / "python.exe"
    elif kind == "other-name":
        python = tmp_path / "pythonw.exe"
        python.write_bytes(b"MZ")
    else:
        python = tmp_path / "folder" / "python.exe"
        python.mkdir(parents=True)
    with pytest.raises(
        ValueError,
        match=rf"side {side.upper()}'s interpreter \(--python-{side}\) must be an "
        "existing python.exe",
    ):
        bench.compare(
            tmp_path / "a",
            tmp_path / "b",
            ["gaussian"],
            2,
            False,
            python_a=python if side == "a" else None,
            python_b=python if side == "b" else None,
        )
    assert created() == set()


def test_the_default_interpreter_is_checked_too(tmp_path, monkeypatch):
    created = refused_start(tmp_path, monkeypatch)
    monkeypatch.setattr(bench, "PYTHON", tmp_path / "out" / "python.exe")
    with pytest.raises(ValueError, match=r"side A's interpreter \(--python-a\)"):
        bench.compare(tmp_path / "a", tmp_path / "b", ["gaussian"], 2, False)
    assert created() == set()


@pytest.mark.parametrize("side", ["a", "b"])
def test_record_noise_refuses_another_interpreter(tmp_path, monkeypatch, side):
    created = refused_start(tmp_path, monkeypatch)
    installed = fake_python(tmp_path / "installed")
    with pytest.raises(
        ValueError, match="--record-noise refuses --python-a and --python-b"
    ):
        bench.compare(
            *identical_trees(tmp_path),
            ["gaussian"],
            6,
            True,
            python_a=installed if side == "a" else None,
            python_b=installed if side == "b" else None,
        )
    assert created() == set()


# chaiNNer-C's chainner_ext as backend/src names it (verify_runtime.CHAINNER_EXT).
EXT = {
    "chainner_ext/__init__.py": b"from . import chainner_ext\n",
    "chainner_ext/__init__.pyi": b"def f() -> None: ...\n",
    "chainner_ext/chainner_ext.pyd": b"MZ pyd",
    "nodes/impl/chainner_native.dll": b"MZ dll",
}
LOCK = (
    "# Interpreter: CPython 3.14.8 (standard build, not free-threaded).\n"
    "# Build tag: python-build-standalone 20261003\n"
    "# Archive SHA-256: " + "ab" * 32 + "\n"
    "# pip: 26.2.1 (installed; pip freeze does not list it)\n"
    "numpy==2.5.3\n"
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parity_trees(tmp_path, monkeypatch, package, ext=EXT, stack=True):
    """make_oracle.py's tree, a candidate B and the provisioned runtime.

    The oracle holds EXT; B holds ext, as a package's resources/src whose manifest
    records it and the runtime (package), or as a plain backend tree. Returns the
    oracle's src, B and the runtime's python.exe. Call after refused_start or
    stub_compare, which redirect bench.PROJECT into tmp_path.
    """
    project = tmp_path / "project"
    (project / "native").mkdir(parents=True, exist_ok=True)
    (project / "native/python-stack.lock.txt").write_text(LOCK, encoding="utf-8")
    python = fake_python(project / "native/runtime/cpython-3.14.8")
    oracle = project / "native/build/oracle"
    for name, dest in bench.CHAINNER_EXT.items():
        (oracle / "src" / dest).parent.mkdir(parents=True, exist_ok=True)
        (oracle / "src" / dest).write_bytes(EXT[name])
    record = {"chainner_ext": {name: sha(data) for name, data in EXT.items()}}
    (oracle / "oracle.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(bench, "ORACLE", oracle)
    b = tmp_path / ("out/package/resources/src" if package else "backend/src")
    for name, data in ext.items():
        (b / name).parent.mkdir(parents=True, exist_ok=True)
        (b / name).write_bytes(data)
    if package:
        manifest = {
            "identity": {"project": str(project)},
            "backend": {
                "resources/src/" + name: {"sha256": sha(data), "bytes": len(data)}
                for name, data in ext.items()
            },
        }
        if stack:
            manifest["python_stack"] = {
                "runtime": str(python.parent),
                "lock_sha256": sha(LOCK.encode()),
            }
        (b.parents[1] / "chainner-c-package.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    return oracle / "src", b, python


@pytest.mark.parametrize("package", [True, False], ids=["package", "backend-tree"])
def test_the_oracle_side_runs_on_the_provisioned_runtime_and_is_recorded(
    tmp_path, monkeypatch, package
):
    stub_compare(tmp_path, monkeypatch)
    oracle, b, python = parity_trees(tmp_path, monkeypatch, package)
    bench.compare(oracle, b, ["gaussian"], 2, False)
    assert FakeBenchBackend.pythons == [python, bench.PYTHON]
    result = only_result(tmp_path)
    assert result["success"] is True
    assert result["pythons"] == {"A": str(python), "B": str(bench.PYTHON)}
    assert result["oracle"] == {
        "chainner_ext": {name: sha(data) for name, data in EXT.items()},
        "python": str(python),
    }


def test_another_tree_a_records_no_oracle(tmp_path, monkeypatch):
    stub_compare(tmp_path, monkeypatch)
    bench.compare(tmp_path / "a", tmp_path / "b", ["gaussian"], 2, False)
    assert only_result(tmp_path)["oracle"] is None


@pytest.mark.parametrize("package", [True, False], ids=["package", "backend-tree"])
def test_the_oracle_side_refuses_another_interpreter(tmp_path, monkeypatch, package):
    created = refused_start(tmp_path, monkeypatch)
    oracle, b, _ = parity_trees(tmp_path, monkeypatch, package)
    with pytest.raises(ValueError, match="The oracle runs only on the provisioned"):
        bench.compare(
            oracle, b, ["gaussian"], 2, False, python_a=fake_python(tmp_path / "other")
        )
    assert created() == set()


@pytest.mark.parametrize("package", [True, False], ids=["package", "backend-tree"])
def test_a_stale_oracle_is_refused(tmp_path, monkeypatch, package):
    created = refused_start(tmp_path, monkeypatch)
    rebuilt = {**EXT, "nodes/impl/chainner_native.dll": b"MZ rebuilt dll"}
    oracle, b, _ = parity_trees(tmp_path, monkeypatch, package, rebuilt)
    with pytest.raises(ValueError, match="Stale oracle refused") as error:
        bench.compare(oracle, b, ["gaussian"], 2, False)
    assert "['nodes/impl/chainner_native.dll']" in str(error.value)
    assert created() == set()


def test_a_package_without_its_provisioned_runtime_record_is_refused(
    tmp_path, monkeypatch
):
    created = refused_start(tmp_path, monkeypatch)
    oracle, b, _ = parity_trees(tmp_path, monkeypatch, True, stack=False)
    with pytest.raises(ValueError, match="Oracle interpreter refused"):
        bench.compare(oracle, b, ["gaussian"], 2, False)
    assert created() == set()


def test_a_tree_b_without_chainner_ext_is_refused(tmp_path, monkeypatch):
    created = refused_start(tmp_path, monkeypatch)
    partial = {k: v for k, v in EXT.items() if not k.endswith(".pyd")}
    oracle, b, _ = parity_trees(tmp_path, monkeypatch, False, partial)
    with pytest.raises(ValueError, match="Tree B holds no chaiNNer-C chainner_ext"):
        bench.compare(oracle, b, ["gaussian"], 2, False)
    assert created() == set()


def test_the_oracle_is_never_side_b(tmp_path, monkeypatch):
    created = refused_start(tmp_path, monkeypatch)
    oracle, b, _ = parity_trees(tmp_path, monkeypatch, False)
    with pytest.raises(ValueError, match="pass it as --a, never --b"):
        bench.compare(b, oracle, ["gaussian"], 2, False)
    assert created() == set()


def test_an_unknown_png_mode_is_refused_before_creating_anything(tmp_path, monkeypatch):
    created = refused_start(tmp_path, monkeypatch)
    with pytest.raises(
        ValueError, match="--png-compare takes sha256 or decoded, got 'pixels'"
    ):
        bench.compare(
            tmp_path / "a", tmp_path / "b", ["gaussian"], 2, False, png_compare="pixels"
        )
    assert created() == set()


def test_compare_runs_each_side_under_its_interpreter_and_records_both(
    tmp_path, monkeypatch
):
    stub_compare(tmp_path, monkeypatch)
    installed = fake_python(tmp_path / "installed")
    run = bench.compare(
        tmp_path / "a",
        tmp_path / "b",
        ["gaussian"],
        2,
        False,
        python_a=installed,
        png_compare="decoded",
    )
    assert FakeBenchBackend.pythons == [installed, bench.PYTHON]
    assert FakeBenchBackend.png_modes == ["decoded"] * 6
    result = only_result(tmp_path)
    assert result["success"] is True
    assert result["pythons"] == {"A": str(installed), "B": str(bench.PYTHON)}
    assert result["png_compare"] == "decoded"
    summary = (run / "summary.md").read_text(encoding="utf-8")
    for line in (
        f"- Interpreters: A `{installed}`, B `{bench.PYTHON}`",
        (
            "- PNG outputs compared by decoded pixels (shape, dtype and every byte) plus "
            "colour/orientation chunks"
        ),
    ):
        assert line + "\n" in summary


@pytest.mark.parametrize(
    "files",
    [
        {"a.png": "1", "b.png": "changed", "c.png": "also changed"},
        {"a.png": "1", "c.png": "3"},
    ],
)
def test_compare_fails_on_different_outputs_and_keeps_the_record(
    tmp_path, monkeypatch, files
):
    def make_record(label, case, pair):
        if label == "B" and pair == 1:
            return trial_record(label, case, pair, files=files)
        return trial_record(label, case, pair)

    stub_compare(tmp_path, monkeypatch, make_record)
    with pytest.raises(
        AssertionError, match=r"gaussian trial-1 B: outputs differ, first at b\.png"
    ):
        bench.compare(tmp_path / "a", tmp_path / "b", ["gaussian"], 2, False)
    result = only_result(tmp_path)
    assert result["success"] is False
    assert "outputs differ" in result["error"]
    assert result["trials"][-1]["backend"] == "B"
    assert result["trials"][-1]["files"] == files


def test_compare_fails_on_different_final_ui_state(tmp_path, monkeypatch):
    changed = {
        **STATE,
        "final_state": [
            STATE["final_state"][0],
            {"event": "node-finish", "data": {"nodeId": "save", "extra": 1}},
        ],
    }

    def make_record(label, case, pair):
        if label == "A" and pair == 1:
            return trial_record(label, case, pair, final_state=changed)
        return trial_record(label, case, pair)

    stub_compare(tmp_path, monkeypatch, make_record)
    with pytest.raises(
        AssertionError,
        match=r"gaussian trial-1 A: final UI state differs at "
        r"\('node-finish', 'save'\)",
    ):
        bench.compare(tmp_path / "a", tmp_path / "b", ["gaussian"], 2, False)
    result = only_result(tmp_path)
    assert result["success"] is False
    assert result["trials"][-1]["final_state"] == changed


# The save node's final state with one field more, as a different stack might show.
CHANGED_STATE = {
    **STATE,
    "final_state": [
        STATE["final_state"][0],
        {"event": "node-finish", "data": {"nodeId": "save", "extra": 1}},
    ],
}
CROSS_LABEL = "cross-stack 3.11 \N{RIGHTWARDS ARROW} 3.14, recorded, not gating"
# A 4x3 BGR image, and the same with three channel values changed in two pixels.
IMAGE = np.arange(4 * 3 * 3, dtype=np.uint8).reshape(4, 3, 3)
CHANGED_IMAGE = IMAGE.copy()
CHANGED_IMAGE[0, 0, 0] += 5
CHANGED_IMAGE[0, 0, 1] += 2
CHANGED_IMAGE[2, 1, 2] += 1


def versions_by_python(
    monkeypatch, versions: dict[Path, tuple[int, int]]
) -> list[Path]:
    """Make bench.python_version report versions[python]; return the probed paths."""
    probed: list[Path] = []

    def python_version(python):
        probed.append(python)
        return versions[python]

    monkeypatch.setattr(bench, "python_version", python_version)
    return probed


def cross_stack_sides(tmp_path, monkeypatch) -> Path:
    """Side A on a fake 3.11 interpreter, B on the default one as 3.14; returns A's.

    Call after stub_compare, which makes the default interpreter a fake python.exe.
    """
    py311 = fake_python(tmp_path / "py311")
    versions_by_python(monkeypatch, {py311: (3, 11), bench.PYTHON: (3, 14)})
    return py311


def png_trials(tmp_path, sides: dict[str, dict], states: dict[str, dict] | None = None):
    """A stub_compare make_record whose trials write real PNGs, as run_trial would.

    sides maps a label to {file name: (pixels, cv2.imwrite params)}; each trial
    writes them into its own output directory and records them by the default PNG
    mode, beside that directory and its side's final state (STATE unless states
    gives one).
    """

    def make_record(label, case, pair):
        output = tmp_path / "media" / case / str(pair) / label
        output.mkdir(parents=True)
        for name, (pixels, params) in sides[label].items():
            assert cv2.imwrite(str(output / name), pixels, params)
        state = (states or {}).get(label, STATE)
        files = bench_oracle.png_outputs(output)
        record = trial_record(label, case, pair, files=files, final_state=state)
        return {**record, "output": str(output)}

    return make_record


def test_python_version_reports_the_interpreters_major_and_minor():
    assert bench.python_version(Path(sys.executable)) == sys.version_info[:2]


def test_cross_stack_refuses_record_noise_before_creating_anything(
    tmp_path, monkeypatch
):
    created = refused_start(tmp_path, monkeypatch)
    probed = versions_by_python(monkeypatch, {})
    with pytest.raises(ValueError, match="--cross-stack refuses --record-noise"):
        bench.compare(
            *identical_trees(tmp_path), ["gaussian"], 6, True, cross_stack=True
        )
    assert probed == []
    assert created() == set()


@pytest.mark.parametrize("python_a", ["default", "other"])
def test_cross_stack_refuses_interpreters_of_one_python_version(
    tmp_path, monkeypatch, python_a
):
    created = refused_start(tmp_path, monkeypatch)
    other = fake_python(tmp_path / "other")
    given = other if python_a == "other" else None
    # One major.minor: a micro release is the same stack for this purpose.
    probed = versions_by_python(monkeypatch, {bench.PYTHON: (3, 14), other: (3, 14)})
    with pytest.raises(
        ValueError,
        match=r"--cross-stack needs interpreters of different Python versions; A "
        r"and B both report 3\.14",
    ):
        bench.compare(
            tmp_path / "a",
            tmp_path / "b",
            ["gaussian"],
            2,
            False,
            python_a=given,
            cross_stack=True,
        )
    # Each side's interpreter was probed, A's first.
    assert probed == [given or bench.PYTHON, bench.PYTHON]
    assert created() == set()


def test_cross_stack_refuses_a_python_exe_it_cannot_probe(tmp_path, monkeypatch):
    created = refused_start(tmp_path, monkeypatch)
    # The default interpreter is a file named python.exe that Windows cannot run;
    # the probe runs it for real, before anything is created.
    with pytest.raises(OSError):
        bench.compare(
            tmp_path / "a", tmp_path / "b", ["gaussian"], 2, False, cross_stack=True
        )
    assert created() == set()


def test_cross_stack_keeps_the_schema_gate(tmp_path, monkeypatch):
    stub_compare(tmp_path, monkeypatch)
    python_a = cross_stack_sides(tmp_path, monkeypatch)

    class OwnSchemas(FakeBenchBackend):
        def __enter__(self):
            self.info["schema_sha256"] = self.label
            return self

    monkeypatch.setattr(bench, "Backend", OwnSchemas)
    with pytest.raises(RuntimeError, match="Node schemas differ"):
        bench.compare(
            tmp_path / "a",
            tmp_path / "b",
            ["gaussian"],
            2,
            False,
            python_a=python_a,
            cross_stack=True,
        )
    result = only_result(tmp_path)
    assert result["success"] is False
    assert result["trials"] == []


@pytest.mark.parametrize(
    ("difference", "message"),
    [
        ("outputs", r"outputs differ, first at c\.png"),
        ("final_state", r"final UI state differs at \('node-finish', 'save'\)"),
    ],
)
def test_cross_stack_fails_a_trial_unlike_its_own_sides_first(
    tmp_path, monkeypatch, difference, message
):
    # A and B differ from each other in every trial, which a cross-stack run only
    # records; B's trial-1 also differs from B's warm-up, which fails the run.
    def make_record(label, case, pair):
        files = {**FILES, "b.png": "B"} if label == "B" else FILES
        state = STATE
        if label == "B" and pair == 1:
            if difference == "outputs":
                files = {**files, "c.png": "changed"}
            else:
                state = CHANGED_STATE
        return trial_record(label, case, pair, files=files, final_state=state)

    stub_compare(tmp_path, monkeypatch, make_record)
    python_a = cross_stack_sides(tmp_path, monkeypatch)
    with pytest.raises(AssertionError, match=f"gaussian trial-1 B: {message}"):
        bench.compare(
            tmp_path / "a",
            tmp_path / "b",
            ["gaussian"],
            2,
            False,
            python_a=python_a,
            cross_stack=True,
        )
    result = only_result(tmp_path)
    assert result["success"] is False
    # The warm-up pair passed: A and B differ, and each side matched itself.
    assert [(t["backend"], t["pair"]) for t in result["trials"]] == [
        ("A", 0),
        ("B", 0),
        ("B", 1),
    ]
    assert result["cross_stack"]["cases"] == {}


def test_cross_stack_records_sides_equal_when_decoded_and_labels_the_run(
    tmp_path, monkeypatch
):
    # B's encoder compresses differently: other file bytes, the same pixels.
    fast = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    small = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    sides = {
        "A": {"a.png": (IMAGE, fast), "b.png": (IMAGE[::-1].copy(), fast)},
        "B": {"a.png": (IMAGE, small), "b.png": (IMAGE[::-1].copy(), small)},
    }
    stub_compare(tmp_path, monkeypatch, png_trials(tmp_path, sides))
    python_a = cross_stack_sides(tmp_path, monkeypatch)
    run = bench.compare(
        tmp_path / "a",
        tmp_path / "b",
        ["gaussian"],
        2,
        False,
        python_a=python_a,
        cross_stack=True,
    )
    result = only_result(tmp_path)
    assert result["success"] is True
    trials = {t["backend"]: t["files"] for t in result["trials"] if t["pair"] == 0}
    assert trials["A"] != trials["B"]  # By their bytes, the sides differ.
    assert result["cross_stack"] == {
        "label": CROSS_LABEL,
        "versions": {"A": "3.11", "B": "3.14"},
        "cases": {
            "gaussian": {
                "outputs_equal": True,
                "files": [
                    {"name": name, "equal": True, "max_abs": 0, "differing_pixels": 0}
                    for name in ("a.png", "b.png")
                ],
                "final_state_equal": True,
                "first_difference": None,
            }
        },
    }
    assert result["warnings"] == []
    # Verdicts as usual: fewer than MIN_PAIRS pairs.
    assert result["verdicts"] == {"gaussian": "unknown"}
    summary = (run / "summary.md").read_text(encoding="utf-8")
    assert f"- Label: {CROSS_LABEL}\n" in summary
    assert summary.endswith(
        "\n## Cross-stack\n\n"
        "| Case | Outputs | Final UI state | First difference |\n"
        "| --- | --- | --- | --- |\n"
        "| gaussian | equal (2 files) | equal | - |\n"
    )


def test_cross_stack_records_differing_sides_and_flags_the_final_state(
    tmp_path, monkeypatch
):
    plain: list[int] = []
    sides = {
        "A": {
            "a.png": (IMAGE, plain),
            "b.png": (IMAGE, plain),
            "c.png": (IMAGE, plain),
        },
        "B": {
            "a.png": (IMAGE, plain),
            "b.png": (CHANGED_IMAGE, plain),
            "d.png": (IMAGE, plain),
        },
    }
    make_record = png_trials(tmp_path, sides, {"B": CHANGED_STATE})
    stub_compare(tmp_path, monkeypatch, make_record)
    python_a = cross_stack_sides(tmp_path, monkeypatch)
    run = bench.compare(
        tmp_path / "a",
        tmp_path / "b",
        ["gaussian"],
        2,
        False,
        python_a=python_a,
        cross_stack=True,
    )
    result = only_result(tmp_path)
    assert result["success"] is True
    assert result["cross_stack"]["cases"] == {
        "gaussian": {
            "outputs_equal": False,
            "files": [
                {"name": "a.png", "equal": True, "max_abs": 0, "differing_pixels": 0},
                # Three channel values in two pixels.
                {"name": "b.png", "equal": False, "max_abs": 5, "differing_pixels": 2},
                # Written by one side only: nothing to measure.
                {
                    "name": "c.png",
                    "equal": False,
                    "max_abs": None,
                    "differing_pixels": None,
                },
                {
                    "name": "d.png",
                    "equal": False,
                    "max_abs": None,
                    "differing_pixels": None,
                },
            ],
            "final_state_equal": False,
            "first_difference": "outputs at b.png",
        }
    }
    warning = (
        "gaussian: final UI state differs between A and B at ('node-finish', "
        f"'save'); recorded, not failed ({CROSS_LABEL})"
    )
    assert result["warnings"] == [warning]
    assert result["verdicts"] == {"gaussian": "unknown"}
    summary = (run / "summary.md").read_text(encoding="utf-8")
    assert f"- WARNING {warning}\n" in summary
    assert summary.endswith(
        "| gaussian | differ: 3 of 4 files, max abs 5 | **differs** (flagged) "
        "| outputs at b.png |\n"
    )


def test_cross_stack_names_a_final_state_difference_when_outputs_are_equal(
    tmp_path, monkeypatch
):
    sides = {label: {"a.png": (IMAGE, [])} for label in ("A", "B")}
    stub_compare(
        tmp_path, monkeypatch, png_trials(tmp_path, sides, {"B": CHANGED_STATE})
    )
    python_a = cross_stack_sides(tmp_path, monkeypatch)
    bench.compare(
        tmp_path / "a",
        tmp_path / "b",
        ["gaussian"],
        2,
        False,
        python_a=python_a,
        cross_stack=True,
    )
    record = only_result(tmp_path)["cross_stack"]["cases"]["gaussian"]
    assert record["outputs_equal"] is True
    assert record["final_state_equal"] is False
    assert record["first_difference"] == "final UI state at ('node-finish', 'save')"


def test_cross_stack_compares_a_video_by_its_decoded_frame_snapshot(
    tmp_path, monkeypatch
):
    snapshot = {"decoded_frame_count": 600, "decoded_frames_sha256": "1"}

    def make_record(label, case, pair):
        frames = (
            {**snapshot, "decoded_frames_sha256": "2"} if label == "B" else snapshot
        )
        return trial_record(label, case, pair, files={"video.mkv": frames})

    stub_compare(tmp_path, monkeypatch, make_record)
    python_a = cross_stack_sides(tmp_path, monkeypatch)
    monkeypatch.setattr(
        bench, "compare_images", lambda python, pairs: pytest.fail("not an image")
    )
    bench.compare(
        tmp_path / "a",
        tmp_path / "b",
        ["video-ffv1"],
        2,
        False,
        python_a=python_a,
        cross_stack=True,
    )
    result = only_result(tmp_path)
    assert result["success"] is True
    assert result["cross_stack"]["cases"] == {
        "video-ffv1": {
            "outputs_equal": False,
            "files": [
                {
                    "name": "video.mkv",
                    "equal": False,
                    "max_abs": None,
                    "differing_pixels": None,
                }
            ],
            "final_state_equal": True,
            "first_difference": "outputs at video.mkv",
        }
    }


def launch(
    run: str, floor: float, background: float | None = 2.0, repeats: int = 6
) -> dict:
    """One recorded --record-noise launch of a case."""
    return {
        "run": run,
        "repeats": repeats,
        "floor": floor,
        "background_busy_percent": background,
    }


def entry(*launches: dict) -> dict:
    """A case's stored noise-floor entry: its launches and the largest floor."""
    return {
        "floor": max(item["floor"] for item in launches),
        "launches": list(launches),
    }


def three_launches() -> dict:
    return entry(
        launch("bench-1", 0.03), launch("bench-2", 0.05), launch("bench-3", 0.02)
    )


def cpu_launch(
    run: str,
    floor: float,
    throughput_floor: float = 0.0,
    background: float | None = 2.0,
    repeats: int = 6,
) -> dict:
    """One recorded --record-noise launch of a case's CPU floor."""
    return {
        "run": run,
        "repeats": repeats,
        "floor": floor,
        "throughput_floor": throughput_floor,
        "background_busy_percent": background,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "auto"),
        ("1", "k1"),
        ("3", "k3"),
        ("0", "auto"),
        ("x", "auto"),
        ("", "auto"),
        (" 1", "auto"),
    ],
)
def test_k_setting_mirrors_the_item_window_override(value, expected):
    environ = {} if value is None else {"CHAINNER_C_ITEM_WINDOW": value}
    assert bench.k_setting(environ) == expected


def test_usable_floors_require_six_pairs_and_three_launches():
    summary = {
        "ok": {"pairs": 6},
        "short": {"pairs": 4},
        "few": {"pairs": 6},
        "old": {"pairs": 6},
        "unrecorded": {"pairs": 6},
    }
    entries = {
        "ok": three_launches(),
        "short": three_launches(),  # Enough launches, too few pairs.
        "few": entry(launch("bench-1", 0.03), launch("bench-2", 0.05)),
        # Pre-quad format: a single max-pair floor with no launches.
        "old": {"floor": 0.04, "run": "bench-0", "repeats": 6},
        "unmeasured": three_launches(),  # Recorded, but not in this run.
    }
    assert bench.usable_floors(summary, entries) == {"ok": 0.05}


def test_current_floor_is_the_max_launch_throughput_floor_of_the_setting():
    noise = {
        "floors": {"gaussian": three_launches()},
        "cpu_floors": {
            "k1": {
                # The CPU floors (0.3 the largest) are not the current floor.
                "gaussian": entry(
                    cpu_launch("bench-1", 0.3, 0.02),
                    cpu_launch("bench-2", 0.1, 0.05),
                    cpu_launch("bench-3", 0.2, 0.03),
                ),
                "few": entry(
                    cpu_launch("bench-1", 0.1, 0.09), cpu_launch("bench-2", 0.1, 0.09)
                ),
            }
        },
    }
    assert bench.current_floors(noise, "k1") == {"gaussian": 0.05}
    assert bench.current_floors(noise, "auto") == {}
    assert bench.current_floors({"floors": {}}, "k1") == {}  # Nothing recorded yet.


class FakeProcess:
    """Stands in for subprocess.Popen; it exits only when its job closes."""

    pid = 4242

    def __init__(self, *_args, stuck=False, **_kwargs):
        self.returncode = None
        self.stuck = stuck

    def poll(self):
        return self.returncode

    def wait(self, timeout: float = 0.0):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("backend", timeout)
        return self.returncode


class FakeJob:
    """Stands in for OwnedJob; closing it ends every member that is not stuck."""

    def __init__(self):
        self.handle = 1
        self.members = []

    def assign(self, process):
        self.members.append(process)

    def close(self):
        self.handle = None
        for process in self.members:
            if not process.stuck:
                process.returncode = 1


def fake_backend(
    tmp_path, monkeypatch, request, job: type = FakeJob, env=None, python=None
):
    """A Backend whose FFmpeg copy, job, HTTP calls and source hashes are fakes.

    env is its side environment, {} unless given; python its interpreter, the
    Backend's default unless given.
    """
    monkeypatch.setattr(bench_backend, "provision_ffmpeg", lambda *args: {})
    monkeypatch.setattr(bench_backend, "OwnedJob", job)
    monkeypatch.setattr(bench_backend, "source_hashes", lambda source: {})
    monkeypatch.setattr(bench_backend, "request", request)
    return bench_backend.Backend(
        "A",
        tmp_path / "src",
        tmp_path,
        Path("ffmpeg.exe"),
        Path("ffprobe.exe"),
        {} if env is None else env,
        **({} if python is None else {"python": python}),
    )


def never_ready(port, path, data=None, timeout=15):
    if path == "/status":
        raise urllib.error.URLError("connection refused")
    raise json.JSONDecodeError("Expecting value", "", 0)


def test_startup_timeout_names_last_poll_error_and_log(tmp_path, monkeypatch):
    backend = fake_backend(tmp_path, monkeypatch, never_ready)
    monkeypatch.setattr(bench_backend.subprocess, "Popen", FakeProcess)
    clock = itertools.chain([0.0, 0.0], itertools.repeat(1000.0))
    monkeypatch.setattr(
        bench_backend,
        "time",
        SimpleNamespace(monotonic=clock.__next__, sleep=lambda seconds: None),
    )
    with pytest.raises(
        RuntimeError,
        match=r"A startup timeout; last poll error: URLError\(.*connection refused.*"
        r"see .*backend\.log",
    ):
        backend.__enter__()
    # The failed start still ran the whole cleanup and wrote its record.
    info = json.loads((tmp_path / "A" / "backend.json").read_text(encoding="utf-8"))
    assert info["shutdown_error"].startswith("JSONDecodeError(")
    assert info["owned_process_exited"] is True
    assert info["owned_job_closed"] is True


def test_exit_records_a_stuck_process_and_still_cleans_up(tmp_path, monkeypatch):
    def broken_shutdown(port, path, data=None, timeout=15):
        raise http.client.HTTPException("connection closed")

    backend = fake_backend(tmp_path, monkeypatch, broken_shutdown)
    # The fake stands in for the Popen the start would have made.
    backend.process = cast(subprocess.Popen, FakeProcess(stuck=True))
    backend.job.assign(backend.process)
    backend.__exit__(None, None, None)
    info = json.loads((tmp_path / "A" / "backend.json").read_text(encoding="utf-8"))
    assert info["shutdown_error"] == "HTTPException('connection closed')"
    assert info["exit_wait_error"].startswith("TimeoutExpired(")
    assert info["owned_process_exited"] is False
    assert info["owned_job_closed"] is True
    assert info["native_isa"] is None  # This backend never opened its log.


def test_exit_removes_its_ffmpeg_copy(tmp_path, monkeypatch):
    backend = fake_backend(tmp_path, monkeypatch, never_ready)
    # provision_ffmpeg is faked, so lay out the private copy it would have made.
    copy = tmp_path / "A" / "storage" / "ffmpeg"
    copy.mkdir()
    (copy / "ffmpeg.exe").write_bytes(b"MZ")
    backend.__exit__(None, None, None)
    assert not copy.exists()
    assert (tmp_path / "A" / "storage").is_dir()
    info = json.loads((tmp_path / "A" / "backend.json").read_text(encoding="utf-8"))
    assert "ffmpeg_cleanup_error" not in info


def test_cpu_seconds_reads_owned_job_accounting(tmp_path, monkeypatch):
    # A real job object with no member process: accounting starts at zero.
    backend = fake_backend(
        tmp_path, monkeypatch, never_ready, job=bench_backend.OwnedJob
    )
    try:
        assert backend.cpu_seconds() == 0.0
    finally:
        backend.job.close()
    # A NULL handle would silently report the test process's own job instead.
    with pytest.raises(RuntimeError, match="already closed"):
        backend.cpu_seconds()


def test_page_faults_reads_owned_job_accounting(tmp_path, monkeypatch):
    backend = fake_backend(
        tmp_path, monkeypatch, never_ready, job=bench_backend.OwnedJob
    )
    try:
        # A real job object with no member process: nothing has faulted yet.
        assert backend.page_faults() == 0
        # The base interpreter, not a venv's python.exe: that one starts the real
        # interpreter as a child before the job could take it. The child waits for
        # its stdin, so the job holds it before it first-touches 10000 pages.
        child = subprocess.Popen(
            [
                Path(sys.base_prefix) / "python.exe",
                "-B",
                "-c",
                (
                    "import sys; sys.stdin.read(); "
                    "b = bytearray(10_000 * 4096); b[::4096] = bytes(10_000)"
                ),
            ],
            stdin=subprocess.PIPE,
        )
        backend.job.assign(child)
        child.communicate(b"")
        # An exited member still counts; a touched page faults at least once.
        assert backend.page_faults() >= 10_000
    finally:
        backend.job.close()
    with pytest.raises(RuntimeError, match="already closed"):
        backend.page_faults()


def test_run_trial_records_job_page_faults_over_the_window_of_job_cpu(
    tmp_path, monkeypatch
):
    order = []
    readings = {"cpu_seconds": iter([10.0, 12.5]), "page_faults": iter([1000, 1250])}

    def reader(name):
        def read():
            order.append(name)
            return next(readings[name])

        return read

    def run(port, path, payload, timeout):
        order.append("run")
        return {"success": True}

    backend = SimpleNamespace(
        label="A",
        directory=tmp_path,
        port=0,
        job=None,
        schemas={"n": {"outputs": [], "kind": "regular"}},
        events=SimpleNamespace(
            events=[], error=None, wait_for=lambda *args, **kwargs: []
        ),
        cpu_seconds=reader("cpu_seconds"),
        page_faults=reader("page_faults"),
    )
    (tmp_path / "backend.log").write_bytes(b"")
    monkeypatch.setattr(
        bench.bench_cases,
        "graph",
        lambda schemas, case, assets, output: ([{"id": "n", "schemaId": "n"}], 2, "x"),
    )
    monkeypatch.setattr(bench, "request", run)
    monkeypatch.setattr(bench, "ffmpeg_children", lambda job: {})
    monkeypatch.setattr(bench.bench_oracle, "validate_record", lambda *args: {})
    monkeypatch.setattr(
        bench.bench_oracle,
        "png_outputs",
        lambda output, mode: {"a.png": "1", "b.png": "2"},
    )
    monkeypatch.setattr(bench, "sse_contract", lambda events, output: STATE)
    trial = bench.run_trial(
        cast(bench.Backend, backend),
        "gaussian",
        1,
        Fixtures(tmp_path / "fixtures", {"assets": {}}),
        tmp_path / "media",
        False,
        "sha256",
    )
    # Both counters are read just before the request and just after its response.
    assert order == ["cpu_seconds", "page_faults", "run", "cpu_seconds", "page_faults"]
    assert trial["job_cpu_seconds"] == 2.5
    assert trial["job_page_faults"] == 250


STARTUP_LINE = b"[x] [1] [INFO] [Worker] Starting worker...\r\n"
ISA_LINE = b"[x] [1] [INFO] [Worker] native isa=avx2 (requested avx2, cpu avx512)\r\n"
AUTO_ISA_LINE = (
    b"[x] [1] [INFO] [Worker] native isa=avx512 (requested auto, cpu avx512)\r\n"
)


class FakeClock:
    """Stands in for bench_backend's time: sleep() advances monotonic() and is logged."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_log_matches_reads_after_the_offset_and_polls_until_a_match(
    tmp_path, monkeypatch
):
    log = tmp_path / "backend.log"
    log.write_bytes(ISA_LINE)  # Before the offset, so never matched.
    offset = log.stat().st_size
    clock = FakeClock()

    def sleep(seconds):
        clock.sleep(seconds)
        if len(clock.sleeps) == 3:  # The line reaches the log after three polls.
            with log.open("ab") as stream:
                stream.write(STARTUP_LINE + ISA_LINE)

    monkeypatch.setattr(
        bench_backend,
        "time",
        SimpleNamespace(monotonic=clock.monotonic, sleep=sleep),
    )
    pattern = bench_backend.NATIVE_ISA
    assert bench_backend.log_matches(log, offset, pattern, 0) == []
    assert clock.sleeps == []  # No wait: one read.
    (found,) = bench_backend.log_matches(log, offset, pattern, 10)
    assert found.groups() == ("avx2", "avx2", "avx512")
    assert clock.sleeps == [0.1, 0.1, 0.1]


def test_backend_applies_its_side_environment_last_and_records_it(
    tmp_path, monkeypatch
):
    # Inherited from the harness, then replaced by the side's own value.
    monkeypatch.setenv("CHAINNER_C_ISA", "scalar")
    backend = fake_backend(
        tmp_path, monkeypatch, never_ready, env={"CHAINNER_C_ISA": "avx2"}
    )
    seen: dict[str, str] = {}

    def stop(command, **kwargs):
        seen.update(kwargs["env"])
        raise RuntimeError("stop")

    monkeypatch.setattr(bench_backend.subprocess, "Popen", stop)
    with pytest.raises(RuntimeError, match="stop"):
        backend.__enter__()
    assert seen["CHAINNER_C_ISA"] == "avx2"
    assert seen["CUDA_VISIBLE_DEVICES"] == "-1"  # Isolation still applies.
    # A host's pip or chainner_pip install exits instead of writing into the
    # interpreter.
    assert seen["PIP_REQUIRE_VIRTUALENV"] == "1"
    assert seen["TEMP"] == str(tmp_path / "A" / "temp")
    info = json.loads((tmp_path / "A" / "backend.json").read_text(encoding="utf-8"))
    assert info["env"] == {"CHAINNER_C_ISA": "avx2"}


@pytest.mark.parametrize("given", [False, True], ids=["default", "given"])
def test_backend_starts_its_tree_under_its_interpreter_and_records_it(
    tmp_path, monkeypatch, given
):
    # Set in the harness, so their removal is proven.
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "elsewhere"))
    python = tmp_path / "installed" / "python.exe"
    backend = fake_backend(
        tmp_path, monkeypatch, never_ready, python=python if given else None
    )
    expected = python if given else bench_backend.PYTHON
    seen: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def stop(command, **kwargs):
        seen.append(command)
        environments.append(kwargs["env"])
        raise RuntimeError("stop")

    monkeypatch.setattr(bench_backend.subprocess, "Popen", stop)
    with pytest.raises(RuntimeError, match="stop"):
        backend.__enter__()
    # The isolation does not depend on the interpreter.
    (env,) = environments
    assert bench_backend.ISOLATION.items() <= env.items()
    for name, directory in (
        ("APPDATA", "appdata"),
        ("LOCALAPPDATA", "localappdata"),
        ("TEMP", "temp"),
    ):
        assert env[name] == str(tmp_path / "A" / directory)
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    # Without bytecode, so a read-only tree and interpreter stay as they were.
    assert seen == [
        [
            str(expected),
            "-B",
            str(tmp_path / "src" / "run.py"),
            str(backend.port),
            "--storage-dir",
            str(tmp_path / "A" / "storage"),
        ]
    ]
    info = json.loads((tmp_path / "A" / "backend.json").read_text(encoding="utf-8"))
    assert info["python"] == str(expected)
    assert info["command"] == seen[0]


@pytest.mark.parametrize("written", [False, True], ids=["unchanged", "installed"])
def test_backend_records_whether_its_interpreter_was_written(
    tmp_path, monkeypatch, written
):
    python = fake_python(tmp_path / "installed")
    packages = tmp_path / "installed" / "Lib" / "site-packages"
    (packages / "numpy").mkdir(parents=True)
    backend = fake_backend(tmp_path, monkeypatch, never_ready, python=python)

    def install(command, **kwargs):
        if written:  # What a host's pip install would leave behind.
            (packages / "newpackage-1.0.dist-info").mkdir()
        raise RuntimeError("stop")

    monkeypatch.setattr(bench_backend.subprocess, "Popen", install)
    with pytest.raises(RuntimeError, match="stop"):
        backend.__enter__()
    info = json.loads((tmp_path / "A" / "backend.json").read_text(encoding="utf-8"))
    assert info["interpreter_unchanged"] is not written


def test_site_packages_fingerprints_top_level_entries(tmp_path):
    python = fake_python(tmp_path / "python")
    assert bench_backend.site_packages(python) == {}  # No Lib\site-packages.
    package = tmp_path / "python" / "Lib" / "site-packages" / "numpy"
    package.mkdir(parents=True)
    before = bench_backend.site_packages(python)
    assert set(before) == {"numpy"}
    assert bench_backend.site_packages(python) == before
    # A replaced package: the same name with a new mtime.
    os.utime(package, ns=(0, before["numpy"] + 100))
    assert bench_backend.site_packages(python) == {"numpy": before["numpy"] + 100}
    # A new package: a new name.
    (package.parent / "scipy").mkdir()
    assert set(bench_backend.site_packages(python)) == {"numpy", "scipy"}


def test_compare_fails_when_a_backend_wrote_into_its_interpreter(tmp_path, monkeypatch):
    stub_compare(tmp_path, monkeypatch)

    class WrittenInterpreter(FakeBenchBackend):
        def __exit__(self, *exc):
            self.info["interpreter_unchanged"] = self.label != "A"

    monkeypatch.setattr(bench, "Backend", WrittenInterpreter)
    with pytest.raises(RuntimeError, match="Backend A cleanup/identity check failed"):
        bench.compare(tmp_path / "a", tmp_path / "b", ["gaussian"], 2, False)
    assert only_result(tmp_path)["success"] is False


def ready_then_stop(port, path, data=None, timeout=15):
    """The backend is ready at once; its /nodes call ends the start."""
    if path == "/status":
        return {"ready": True}
    if path == "/nodes":
        raise RuntimeError("stop")
    return {}  # /shutdown


def logging_popen(lines: bytes):
    """A fake Popen whose process writes lines to its stdout, backend.log, at start."""

    def popen(command, *, stdout, **kwargs):
        stdout.write(lines)
        stdout.flush()
        return FakeProcess()

    return popen


@pytest.mark.parametrize(
    ("env", "logged", "error", "expected"),
    [
        pytest.param(
            {"CHAINNER_C_ISA": "avx2"},
            STARTUP_LINE + ISA_LINE,
            "stop",
            {"level": "avx2", "requested": "avx2", "cpu": "avx512"},
            id="logged",
        ),
        pytest.param({}, STARTUP_LINE, "stop", None, id="b3-logs-none"),
        pytest.param(
            {"CHAINNER_C_ISA": "avx2"},
            STARTUP_LINE,
            "Backend A set CHAINNER_C_ISA but logged no native isa line",
            None,
            id="isa-set-but-not-logged",
        ),
        # The DLL reads any value but the exact lowercase levels as auto.
        pytest.param(
            {"CHAINNER_C_ISA": "AVX2"},
            STARTUP_LINE + AUTO_ISA_LINE,
            r"Backend A set CHAINNER_C_ISA='AVX2' but logged native isa=avx512 "
            r"\(requested auto, cpu avx512\)",
            {"level": "avx512", "requested": "auto", "cpu": "avx512"},
            id="isa-request-not-taken",
        ),
    ],
)
def test_backend_records_native_isa_or_none(
    tmp_path, monkeypatch, env, logged, error, expected
):
    clock = FakeClock()
    monkeypatch.setattr(bench_backend, "time", clock)
    backend = fake_backend(tmp_path, monkeypatch, ready_then_stop, env=env)
    monkeypatch.setattr(bench_backend.subprocess, "Popen", logging_popen(logged))
    with pytest.raises(RuntimeError, match=error):
        backend.__enter__()
    # The failed start still ran the cleanup, which reads the whole log.
    info = json.loads((tmp_path / "A" / "backend.json").read_text(encoding="utf-8"))
    assert info["native_isa"] == expected
    assert info["env"] == env
    if expected is not None or error == "stop":
        # Found at once, or never waited for without the variable.
        assert clock.sleeps == []
    else:
        assert set(clock.sleeps) == {0.1}
        assert sum(clock.sleeps) == pytest.approx(10, abs=0.2)


def test_merge_floors_keeps_other_cases():
    existing = {
        "old": entry(cpu_launch("bench-1", 0.03)),
        "pre-quad": {"floor": 0.03, "run": "bench-0", "repeats": 6},
    }
    before = copy.deepcopy(existing)
    merged = bench.merge_floors(
        existing,
        {"new": 0.05},
        "bench-2",
        6,
        {"new": 4.0, "other": 9.0},
        {"new": 0.02, "other": 0.01},
    )
    assert set(merged) == {"old", "pre-quad", "new"}
    assert merged["old"] == before["old"]
    assert merged["pre-quad"] == before["pre-quad"]
    assert merged["new"] == {
        "floor": 0.05,
        "launches": [
            {
                "run": "bench-2",
                "repeats": 6,
                "floor": 0.05,
                "throughput_floor": 0.02,
                "background_busy_percent": 4.0,
            }
        ],
    }
    assert existing == before  # Merging builds a new section.


def test_merge_floors_accumulates_launches_and_keeps_the_largest_floor():
    merged: dict = {}
    for run, floor, throughput_floor, background in (
        ("bench-1", 0.03, 0.01, 2.0),
        ("bench-2", 0.05, 0.04, 4.0),
        ("bench-3", 0.02, 0.06, None),
    ):
        merged = bench.merge_floors(
            merged, {"g": floor}, run, 6, {"g": background}, {"g": throughput_floor}
        )
    assert merged == {
        "g": {
            # The largest of the launches' own floors; throughput floors are only kept.
            "floor": 0.05,
            "launches": [
                cpu_launch("bench-1", 0.03, 0.01, 2.0),
                cpu_launch("bench-2", 0.05, 0.04, 4.0),
                cpu_launch("bench-3", 0.02, 0.06, None),
            ],
        }
    }


def test_merge_floors_discards_pre_quad_entries():
    pre_quad = {
        "floor": 0.30,  # A max-pair floor includes the position effect.
        "run": "bench-0",
        "repeats": 6,
        "background_busy_percent": 2.0,
    }
    merged = bench.merge_floors(
        {"g": pre_quad}, {"g": 0.02}, "bench-1", 6, {"g": 3.0}, {"g": 0.01}
    )
    assert merged["g"] == {
        "floor": 0.02,
        "launches": [cpu_launch("bench-1", 0.02, 0.01, 3.0)],
    }


def test_merge_floors_replaces_a_relaunch_of_the_same_run():
    first = {
        "g": entry(
            cpu_launch("bench-1", 0.03),
            cpu_launch("bench-2", 0.05),
            cpu_launch("bench-3", 0.02),
        )
    }
    again = bench.merge_floors(first, {"g": 0.05}, "bench-2", 6, {"g": 2.0}, {"g": 0.0})
    assert again == first  # Idempotent.
    changed = bench.merge_floors(
        first, {"g": 0.01}, "bench-2", 8, {"g": 9.0}, {"g": 0.04}
    )
    assert changed["g"] == {
        "floor": 0.03,  # The replaced launch held the largest floor.
        "launches": [
            cpu_launch("bench-1", 0.03),
            cpu_launch("bench-2", 0.01, 0.04, 9.0, repeats=8),
            cpu_launch("bench-3", 0.02),
        ],
    }


@pytest.mark.parametrize(
    ("busy", "total", "job", "expected"),
    [
        (50.0, 100.0, 20.0, 30.0),
        (10.0, 100.0, 12.5, 0.0),  # Job accounting can lag the machine sample.
        (5.0, 0.0, 1.0, None),
    ],
)
def test_background_percent(busy, total, job, expected):
    assert bench.background_percent(busy, total, job) == expected


def test_background_by_case_takes_median_of_measured_trials():
    def load(case, pair, value):
        return {"case": case, "pair": pair, "background_busy_percent": value}

    trials = [
        load("g", 0, 90.0),  # Warm-ups are not measured.
        load("g", 1, 1.0),
        load("g", 1, 3.0),
        load("g", 2, None),
        load("g", 2, 8.0),
        load("v", 0, 5.0),
        load("v", 1, None),
    ]
    assert bench.background_by_case(trials) == {"g": 3.0, "v": None}


def test_background_warnings_need_a_recorded_load_beyond_the_margin():
    entries = {
        # Pre-quad entries have no launches, so no recorded load either.
        "old": {"floor": 0.03, "run": "bench-0", "repeats": 6},
        "unmeasured": entry(launch("bench-1", 0.03, None)),
        "calm": entry(launch("bench-1", 0.03, 2.0)),
        "edge": entry(launch("bench-1", 0.03, 2.0)),
        "busy": entry(launch("bench-1", 0.03, 2.0)),
    }
    background = {
        "old": 50.0,
        "unmeasured": 50.0,
        "calm": None,
        "edge": 2.0 + bench.BACKGROUND_MARGIN,
        "busy": 7.5,
        "unfloored": 50.0,
    }
    warnings = bench.background_warnings(background, entries)
    assert len(warnings) == 1
    assert warnings[0].startswith("busy: median background load 7.5% exceeds the 2.0%")


def test_background_warnings_compare_with_the_busiest_launch():
    entries = {
        "g": entry(
            launch("bench-1", 0.03, 1.0),
            launch("bench-2", 0.03, None),  # Skipped, not read as zero.
            launch("bench-3", 0.03, 6.0),
        )
    }
    margin = bench.BACKGROUND_MARGIN
    # Within the margin of the busiest launch (6.0), though 9+ points above the first.
    assert bench.background_warnings({"g": 6.0 + margin}, entries) == []
    (message,) = bench.background_warnings({"g": 6.0 + margin + 0.5}, entries)
    assert message.startswith(
        f"g: median background load {6.0 + margin + 0.5:.1f}% exceeds the 6.0%"
    )


def test_summary_header_and_background_warnings(tmp_path, monkeypatch):
    def make_record(label, case, pair):
        return {**trial_record(label, case, pair), "background_busy_percent": 10.0}

    stub_compare(tmp_path, monkeypatch, make_record)
    write_noise(
        tmp_path,
        {
            "gaussian": entry(
                launch("bench-1", 0.05, 1.0),
                launch("bench-2", 0.05, 1.0),
                launch("bench-3", 0.05, 1.0),
            )
        },
    )
    run = bench.compare(tmp_path / "a", tmp_path / "b", ["gaussian"], 2, False)
    result = only_result(tmp_path)
    assert result["background_busy_percent"] == {"gaussian": 10.0}
    assert len(result["warnings"]) == 1
    assert result["warnings"][0].startswith("gaussian: median background load 10.0%")
    summary = (run / "summary.md").read_text(encoding="utf-8")
    for line in (
        "- Repeats: 2 AB/BA pairs per case, after one warm-up",
        "- Machine busy before the run: 1.0%",
        "- Median background load (CPU outside the owned backends): gaussian 10.0%",
        "- Noise floors from: none usable",  # 2 pairs < MIN_PAIRS.
        "- K setting: auto; CPU floors from: none usable",
        f"- WARNING {result['warnings'][0]}",
        LEGEND,
        CPU_LEGEND,
    ):
        assert line + "\n" in summary
    assert summary.index(LEGEND) < summary.index(CPU_LEGEND)


PAGE_FAULTS_LINE = "- Page faults per item (median of measured trials), A / B: "


def test_summary_shows_median_page_faults_per_item_of_measured_trials(
    tmp_path, monkeypatch
):
    # Page faults per item by pair; the warm-up (pair 0) must not count.
    per_item = {
        ("gaussian", "A"): [900, 110, 120],
        ("gaussian", "B"): [900, 55, 60],
        ("caption", "A"): [900, 14, 14],
        ("caption", "B"): [900, 9, 9],
    }

    def make_record(label, case, pair):
        faults = 32 * per_item[case, label][pair]  # Each trial has 32 items.
        return {**trial_record(label, case, pair), "job_page_faults": faults}

    stub_compare(tmp_path, monkeypatch, make_record)
    run = bench.compare(
        tmp_path / "a", tmp_path / "b", ["gaussian", "caption"], 2, False
    )
    line = f"{PAGE_FAULTS_LINE}gaussian 115.0 / 57.5, caption 14.0 / 9.0\n"
    summary = (run / "summary.md").read_text(encoding="utf-8")
    assert line in summary
    # Information only: it is in the header, before the legends, and judges nothing.
    assert summary.index(line) < summary.index(LEGEND)
    result = only_result(tmp_path)
    assert result["verdicts"] == {"gaussian": "unknown", "caption": "unknown"}
    assert result["cpu_verdicts"] == {"gaussian": "unknown", "caption": "unknown"}
    assert result["warnings"] == []


@pytest.mark.parametrize(
    "lacks",
    [
        pytest.param(lambda label, pair: True, id="no-trial-has-it"),
        pytest.param(lambda label, pair: pair > 0, id="only-the-warm-up-has-it"),
        pytest.param(lambda label, pair: label == "B", id="one-side-lacks-it"),
        pytest.param(lambda label, pair: pair == 2, id="one-trial-lacks-it"),
    ],
)
def test_page_faults_are_unknown_unless_every_measured_trial_has_them(lacks):
    trials = [
        trial_record(label, "gaussian", pair)
        if lacks(label, pair)
        else {**trial_record(label, "gaussian", pair), "job_page_faults": 64}
        for pair in (0, 1, 2)
        for label in ("A", "B")
    ]
    assert bench.page_faults_by_case(trials) == {"gaussian": None}


def test_trials_without_page_faults_leave_summary_and_confirmation_working(
    tmp_path, monkeypatch, capsys
):
    def make_record(label, case, pair):
        # A trial recorded before job_page_faults existed; B takes 1.25 times as long.
        seconds = 1.25 if label == "B" else 1.0
        return {**trial_record(label, case, pair), "seconds": seconds}

    stub_compare(tmp_path, monkeypatch, make_record)
    write_noise(tmp_path, {"gaussian": three_launches()})
    a, b = tmp_path / "a", tmp_path / "b"
    first = bench.compare(a, b, ["gaussian"], 6, False)
    result = only_result(tmp_path)
    assert all("job_page_faults" not in t for t in result["trials"])
    assert result["verdicts"] == {"gaussian": "regression"}
    assert f"{PAGE_FAULTS_LINE}gaussian n/a\n" in (first / "summary.md").read_text(
        encoding="utf-8"
    )
    fake = FakeCompare(
        tmp_path, {"gaussian": "regression"}, ratios=({"gaussian": 0.8},)
    )
    monkeypatch.setattr(bench, "compare", fake)
    assert bench.confirm(first, a, b, 6) == {"gaussian": "regression"}
    assert "FINAL gaussian: regression" in capsys.readouterr().out.splitlines()
    summary = (first / "summary.md").read_text(encoding="utf-8")
    assert f"{PAGE_FAULTS_LINE}gaussian n/a\n" in summary
    assert "\n## Confirmed verdicts\n" in summary


@pytest.mark.parametrize("window", [None, "1"])
def test_background_warnings_also_cover_the_cpu_floors_of_the_k_setting(
    tmp_path, monkeypatch, window
):
    def make_record(label, case, pair):
        return {**trial_record(label, case, pair), "background_busy_percent": 10.0}

    stub_compare(tmp_path, monkeypatch, make_record)
    if window is not None:
        monkeypatch.setenv("CHAINNER_C_ITEM_WINDOW", window)
    # The shared floor was recorded under 8% (within the margin of this run's 10%),
    # the auto CPU floor under 1% (beyond it) and the K = 1 CPU floor under 9%.
    write_noise(
        tmp_path,
        {
            "gaussian": entry(
                launch("bench-1", 0.05, 8.0),
                launch("bench-2", 0.05, 8.0),
                launch("bench-3", 0.05, 8.0),
            )
        },
        {
            "auto": {"gaussian": entry(cpu_launch("bench-a", 0.01, background=1.0))},
            "k1": {"gaussian": entry(cpu_launch("bench-k", 0.01, background=9.0))},
        },
    )
    run = bench.compare(tmp_path / "a", tmp_path / "b", ["gaussian"], 6, False)
    result = only_result(tmp_path)
    if window is None:
        message = (
            "gaussian: median background load 10.0% exceeds the 1.0% its CPU floor "
            "was recorded under by more than 5 points; its CPU verdict and current "
            "floor are less reliable"
        )
        assert result["warnings"] == [message]
        summary = (run / "summary.md").read_text(encoding="utf-8")
        assert f"- WARNING {message}\n" in summary
    else:
        assert result["warnings"] == []  # Only the run's own K setting counts.


def test_summary_lists_the_launches_behind_the_usable_floors(tmp_path, monkeypatch):
    stub_compare(tmp_path, monkeypatch)
    write_noise(
        tmp_path,
        {
            "gaussian": entry(
                launch("bench-1", 0.03),
                launch("bench-2", 0.05),
                launch("bench-3", 0.02),
            ),
            "morphology": entry(
                launch("bench-2", 0.04),
                launch("bench-3", 0.01),
                launch("bench-4", 0.02),
            ),
            # Two launches are not enough, so these runs are not listed.
            "caption": entry(launch("bench-8", 0.03), launch("bench-9", 0.03)),
        },
    )
    cases = ["gaussian", "morphology", "caption"]
    run = bench.compare(tmp_path / "a", tmp_path / "b", cases, 6, False)
    result = only_result(tmp_path)
    assert result["verdicts"] == {
        "gaussian": "neutral",
        "morphology": "neutral",
        "caption": "unknown",
    }
    assert result["warnings"] == []
    summary = (run / "summary.md").read_text(encoding="utf-8")
    runs = "bench-1, bench-2, bench-3, bench-4 (4 launches)"
    assert f"- Noise floors from: {runs}\n" in summary
    assert LEGEND + "\n" in summary
    # Both trees used the same CPU, and no CPU or current floor is recorded.
    assert "| 0/6 | +5.1/-4.9% | n/a | neutral | +0.0% | n/a | unknown |" in summary
    assert "| 0/6 | n/a | n/a | unknown | +0.0% | n/a | unknown |" in summary  # caption


def test_pre_quad_floors_stay_unusable_without_failing_a_run(tmp_path, monkeypatch):
    stub_compare(tmp_path, monkeypatch)
    write_noise(
        tmp_path,
        {
            "gaussian": {
                "floor": 0.08,
                "run": "bench-old",
                "repeats": 6,
                "busy_percent_before": 9.5,
            }
        },
    )
    run = bench.compare(tmp_path / "a", tmp_path / "b", ["gaussian"], 6, False)
    result = only_result(tmp_path)
    assert result["success"] is True
    assert result["verdicts"] == {"gaussian": "unknown"}
    assert result["warnings"] == []
    summary = (run / "summary.md").read_text(encoding="utf-8")
    assert "- Noise floors from: none usable\n" in summary


def identical_trees(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "run.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path / "a", tmp_path / "b"


def test_record_noise_stores_per_case_background_and_names_its_run(
    tmp_path, monkeypatch
):
    stub_compare(tmp_path, monkeypatch)
    run = bench.compare(*identical_trees(tmp_path), ["gaussian"], 6, True)
    stored = json.loads((tmp_path / "noise-floor.json").read_text(encoding="utf-8"))
    assert stored["cpu_floors"] == {
        "auto": {
            "gaussian": {
                "floor": 0.0,
                "launches": [cpu_launch(run.name, 0.0, 0.0, 2.0)],
            }
        }
    }
    # One launch is not a usable floor yet, and no shared floor is recorded.
    result = only_result(tmp_path)
    assert result["cpu_verdicts"] == {"gaussian": "unknown"}
    assert result["verdicts"] == {"gaussian": "unknown"}
    summary = (run / "summary.md").read_text(encoding="utf-8")
    assert "- Noise floors from: none usable\n" in summary
    assert "- K setting: auto; CPU floors from: none usable\n" in summary


def test_record_noise_judges_itself_by_the_merged_launches(tmp_path, monkeypatch):
    stub_compare(tmp_path, monkeypatch)
    write_noise(
        tmp_path,
        {},
        {
            "auto": {
                "gaussian": entry(
                    cpu_launch("bench-a", 0.02), cpu_launch("bench-b", 0.01)
                )
            },
            "k1": {"gaussian": entry(cpu_launch("bench-k", 0.04))},
        },
    )
    run = bench.compare(*identical_trees(tmp_path), ["gaussian"], 6, True)
    stored = json.loads((tmp_path / "noise-floor.json").read_text(encoding="utf-8"))
    # Another K setting's CPU floors are kept as they were.
    assert stored["cpu_floors"]["k1"] == {
        "gaussian": entry(cpu_launch("bench-k", 0.04))
    }
    assert stored["cpu_floors"]["auto"]["gaussian"] == {
        "floor": 0.02,
        "launches": [
            cpu_launch("bench-a", 0.02),
            cpu_launch("bench-b", 0.01),
            cpu_launch(run.name, 0.0, 0.0, 2.0),
        ],
    }
    # This run is the third launch, so the merged CPU floor already gates its CPU
    # verdicts.
    assert only_result(tmp_path)["cpu_verdicts"] == {"gaussian": "neutral"}
    summary = (run / "summary.md").read_text(encoding="utf-8")
    runs = ", ".join(sorted(["bench-a", "bench-b", run.name]))
    assert f"- K setting: auto; CPU floors from: {runs} (3 launches)\n" in summary


def test_record_noise_writes_floors_only_after_the_summary(tmp_path, monkeypatch):
    stub_compare(tmp_path, monkeypatch)
    monkeypatch.setenv("CHAINNER_C_ITEM_WINDOW", "1")
    write_noise(
        tmp_path,
        {"gaussian": three_launches()},
        {"k1": {"gaussian": entry(cpu_launch("bench-a", 0.02))}},
    )
    seeded = (tmp_path / "noise-floor.json").read_text(encoding="utf-8")

    def broken_table(*args):
        raise RuntimeError("summary failed")

    monkeypatch.setattr(bench.bench_stats, "table", broken_table)
    with pytest.raises(RuntimeError, match="summary failed"):
        bench.compare(*identical_trees(tmp_path), ["gaussian"], 6, True)
    assert (tmp_path / "noise-floor.json").read_text(encoding="utf-8") == seeded
    assert only_result(tmp_path)["success"] is False


def writing_trials(monkeypatch):
    """Make each stubbed trial leave an output image and its events.json in media."""
    inner = bench.run_trial

    def run_trial(backend, case, pair, fixtures, media, profiled, png_compare):
        output = media / case / f"trial-{pair}" / backend.label
        output.mkdir(parents=True, exist_ok=True)
        (output / "000.png").write_bytes(b"png")
        (output / "events.json").write_text("[]", encoding="utf-8")
        return inner(backend, case, pair, fixtures, media, profiled, png_compare)

    monkeypatch.setattr(bench, "run_trial", run_trial)


def test_successful_run_keeps_only_trial_json_in_media(tmp_path, monkeypatch):
    stub_compare(tmp_path, monkeypatch)
    writing_trials(monkeypatch)
    bench.compare(tmp_path / "a", add_timer(tmp_path / "b"), ["gaussian"], 2, False)
    media = tmp_path / "diagnostics"
    assert not list(media.rglob("*.png"))
    assert len(list(media.rglob("events.json"))) == 6  # warm-up and 2 pairs, 2 sides
    result = only_result(tmp_path)
    assert result["success"] is True
    assert result["media_files_removed"] == 6
    assert "media_cleanup_error" not in result


def test_failed_run_keeps_its_media(tmp_path, monkeypatch):
    stub_compare(tmp_path, monkeypatch)
    writing_trials(monkeypatch)

    def broken_table(*args):
        raise RuntimeError("summary failed")

    monkeypatch.setattr(bench.bench_stats, "table", broken_table)
    with pytest.raises(RuntimeError, match="summary failed"):
        bench.compare(tmp_path / "a", add_timer(tmp_path / "b"), ["gaussian"], 2, False)
    assert len(list((tmp_path / "diagnostics").rglob("000.png"))) == 6
    assert "media_files_removed" not in only_result(tmp_path)


def test_record_noise_never_touches_the_shared_floors(tmp_path, monkeypatch):
    stub_compare(tmp_path, monkeypatch)
    monkeypatch.setenv("CHAINNER_C_ITEM_WINDOW", "1")
    write_noise(tmp_path, {"gaussian": three_launches()})
    seeded = json.loads((tmp_path / "noise-floor.json").read_text(encoding="utf-8"))
    run = bench.compare(*identical_trees(tmp_path), ["gaussian"], 6, True)
    stored = json.loads((tmp_path / "noise-floor.json").read_text(encoding="utf-8"))
    assert stored["floors"] == seeded["floors"]
    (recorded,) = stored["cpu_floors"]["k1"]["gaussian"]["launches"]
    assert set(recorded) == {
        "run",
        "repeats",
        "floor",
        "throughput_floor",
        "background_busy_percent",
    }
    assert recorded["run"] == run.name
    assert "auto" not in stored["cpu_floors"]
    result = only_result(tmp_path)
    assert result["k_setting"] == "k1"
    # Throughput verdicts use the shared floors as they were; this launch's
    # throughput floor (0) is within them, so nothing is reported.
    assert result["verdicts"] == {"gaussian": "neutral"}
    assert result["warnings"] == []


def test_record_noise_warns_when_its_throughput_floor_exceeds_the_shared_floor(
    tmp_path, monkeypatch
):
    def make_record(label, case, pair):
        # B takes 1.25 s in pairs 1 and 2: the first quad is 0.8, a floor of ln 1.25.
        seconds = 1.25 if label == "B" and pair in (1, 2) else 1.0
        return {**trial_record(label, case, pair), "seconds": seconds}

    stub_compare(tmp_path, monkeypatch, make_record)
    shared = entry(
        launch("bench-1", 0.001), launch("bench-2", 0.001), launch("bench-3", 0.001)
    )
    write_noise(tmp_path, {"gaussian": shared})
    seeded = json.loads((tmp_path / "noise-floor.json").read_text(encoding="utf-8"))
    run = bench.compare(*identical_trees(tmp_path), ["gaussian"], 6, True)
    message = (
        "gaussian: this launch's throughput floor +25.0% exceeds the shared floor "
        "+0.1% (median background load 2.0%, machine busy before 1.0%); re-basing "
        "the shared floors is the stand-in's decision"
    )
    assert only_result(tmp_path)["warnings"] == [message]
    summary = (run / "summary.md").read_text(encoding="utf-8")
    assert f"- WARNING {message}\n" in summary
    stored = json.loads((tmp_path / "noise-floor.json").read_text(encoding="utf-8"))
    assert stored["floors"] == seeded["floors"]
    (recorded,) = stored["cpu_floors"]["auto"]["gaussian"]["launches"]
    assert recorded["throughput_floor"] == pytest.approx(math.log(1.25))


def test_record_noise_checks_cpu_background_before_merging_its_launch(
    tmp_path, monkeypatch
):
    def make_record(label, case, pair):
        return {**trial_record(label, case, pair), "background_busy_percent": 10.0}

    stub_compare(tmp_path, monkeypatch, make_record)
    write_noise(
        tmp_path,
        {},
        {"auto": {"gaussian": entry(cpu_launch("bench-a", 0.01, background=1.0))}},
    )
    bench.compare(*identical_trees(tmp_path), ["gaussian"], 6, True)
    # Compared with the earlier launch's 1.0%. Once merged, this launch's own 10.0%
    # is the highest recorded load, which would hide the noisier machine.
    assert only_result(tmp_path)["warnings"] == [
        (
            "gaussian: median background load 10.0% exceeds the 1.0% its CPU floor "
            "was recorded under by more than 5 points; its CPU verdict and current "
            "floor are less reliable"
        )
    ]
    stored = json.loads((tmp_path / "noise-floor.json").read_text(encoding="utf-8"))
    launches = stored["cpu_floors"]["auto"]["gaussian"]["launches"]
    assert [launch["background_busy_percent"] for launch in launches] == [1.0, 10.0]


def test_record_noise_reads_the_current_floors_before_merging_its_launch(
    tmp_path, monkeypatch
):
    stub_compare(tmp_path, monkeypatch)
    two = entry(cpu_launch("bench-a", 0.1, 0.06), cpu_launch("bench-b", 0.1, 0.06))
    write_noise(tmp_path, {"gaussian": three_launches()}, {"auto": {"gaussian": two}})
    bench.compare(*identical_trees(tmp_path), ["gaussian"], 6, True)
    stored = json.loads((tmp_path / "noise-floor.json").read_text(encoding="utf-8"))
    assert len(stored["cpu_floors"]["auto"]["gaussian"]["launches"]) == 3
    # Merged, this launch makes three, but its throughput verdicts use the current
    # floors as recorded before it, as they use the shared floors.
    result = only_result(tmp_path)
    assert result["current_floors"] == {}
    assert result["verdicts"] == {"gaussian": "neutral"}


@pytest.mark.parametrize(
    ("window", "expected", "sources"),
    [
        (None, "win", "bench-1, bench-2, bench-3 (3 launches)"),
        ("1", "unknown", "none usable"),  # No CPU floor is recorded at K = 1.
    ],
)
def test_cpu_verdicts_use_the_floors_of_the_runs_k_setting(
    tmp_path, monkeypatch, window, expected, sources
):
    def make_record(label, case, pair):
        # B uses half A's CPU in the same time.
        cpu = 4.0 if label == "A" else 2.0
        return {**trial_record(label, case, pair), "job_cpu_seconds": cpu}

    stub_compare(tmp_path, monkeypatch, make_record)
    if window is not None:
        monkeypatch.setenv("CHAINNER_C_ITEM_WINDOW", window)
    floors = entry(
        cpu_launch("bench-1", 0.01),
        cpu_launch("bench-2", 0.01),
        cpu_launch("bench-3", 0.01),
    )
    write_noise(tmp_path, {}, {"auto": {"gaussian": floors}})
    seeded = (tmp_path / "noise-floor.json").read_text(encoding="utf-8")
    run = bench.compare(tmp_path / "a", tmp_path / "b", ["gaussian"], 6, False)
    result = only_result(tmp_path)
    setting = "auto" if window is None else f"k{window}"
    assert result["k_setting"] == setting
    assert result["cpu_verdicts"] == {"gaussian": expected}
    assert result["verdicts"] == {"gaussian": "unknown"}  # No shared floor.
    summary = (run / "summary.md").read_text(encoding="utf-8")
    assert f"- K setting: {setting}; CPU floors from: {sources}\n" in summary
    shown = "+1.0/-1.0%" if window is None else "n/a"
    assert f"| +100.0% | {shown} | {expected} |\n" in summary
    # An ordinary run writes no floors.
    assert (tmp_path / "noise-floor.json").read_text(encoding="utf-8") == seeded


IN_SAMPLE_NOTE = (
    "- Recording run: CPU verdicts are in-sample (this launch is part of its own "
    "CPU floors) and do not confirm them; throughput verdicts use the shared "
    "floors, which this run leaves unchanged, and the current floors as recorded "
    "before it.\n"
)


def test_record_noise_summary_says_its_verdicts_are_in_sample(tmp_path, monkeypatch):
    stub_compare(tmp_path, monkeypatch)
    run = bench.compare(*identical_trees(tmp_path), ["gaussian"], 6, True)
    summary = (run / "summary.md").read_text(encoding="utf-8")
    assert IN_SAMPLE_NOTE in summary
    assert summary.index(IN_SAMPLE_NOTE) < summary.index(LEGEND)  # In the header.
    # A recording run is never confirmed, so it does not point at a confirmation.md.
    assert "provisional" not in summary
    assert only_result(tmp_path)["confirms"] is None


PROVISIONAL_NOTE = (
    "- win/regression in this table are provisional; a claim needs the Final "
    "column of the first run's confirmation.md.\n"
)


def test_ordinary_summary_marks_win_and_regression_as_provisional(
    tmp_path, monkeypatch
):
    stub_compare(tmp_path, monkeypatch)
    run = bench.compare(tmp_path / "a", tmp_path / "b", ["gaussian"], 6, False)
    summary = (run / "summary.md").read_text(encoding="utf-8")
    assert PROVISIONAL_NOTE in summary
    assert summary.index(PROVISIONAL_NOTE) < summary.index(LEGEND)  # In the header.
    assert "in-sample" not in summary
    assert "Confirmation run for" not in summary
    assert "Tie-break run for" not in summary
    assert "Profile run" not in summary
    assert " environment: " not in summary  # Neither side has its own variables.
    result = only_result(tmp_path)
    assert result["confirms"] is None
    assert result["tiebreak"] is False


def test_confirmation_run_summary_names_the_run_it_confirms(tmp_path, monkeypatch):
    stub_compare(tmp_path, monkeypatch)
    run = bench.compare(
        tmp_path / "a", tmp_path / "b", ["gaussian"], 6, False, "bench-first"
    )
    summary = (run / "summary.md").read_text(encoding="utf-8")
    assert "- Confirmation run for bench-first.\n" in summary
    assert "Tie-break run for" not in summary
    assert PROVISIONAL_NOTE in summary  # Still a table of provisional verdicts.
    result = only_result(tmp_path)
    assert result["confirms"] == "bench-first"
    assert result["tiebreak"] is False


def test_tiebreak_run_summary_says_it_is_a_tiebreak(tmp_path, monkeypatch):
    stub_compare(tmp_path, monkeypatch)
    run = bench.compare(
        tmp_path / "a",
        tmp_path / "b",
        ["gaussian"],
        6,
        False,
        confirms="bench-first",
        tiebreak=True,
    )
    summary = (run / "summary.md").read_text(encoding="utf-8")
    assert "- Tie-break run for bench-first.\n" in summary
    assert "Confirmation run for" not in summary
    assert PROVISIONAL_NOTE in summary
    result = only_result(tmp_path)
    assert result["confirms"] == "bench-first"
    assert result["tiebreak"] is True


PROFILE_NOTE = (
    "- Profile run: CHAINNER_C_PROFILE=1 on both sides puts the timer's overhead in "
    "every timing, so no floors apply, every verdict is unknown, and the run is "
    "never confirmed or used as a gate.\n"
)


def test_profile_run_times_both_sides_applies_no_floors_and_never_confirms(
    tmp_path, monkeypatch
):
    def make_record(label, case, pair):
        # B uses half A's CPU in the same time, on a busier machine than the floors'.
        cpu = 4.0 if label == "A" else 2.0
        return {
            **trial_record(label, case, pair),
            "job_cpu_seconds": cpu,
            "background_busy_percent": 10.0,
        }

    stub_compare(tmp_path, monkeypatch, make_record)
    cpu_floors = entry(
        cpu_launch("bench-1", 0.01),
        cpu_launch("bench-2", 0.01),
        cpu_launch("bench-3", 0.01),
    )
    write_noise(
        tmp_path, {"gaussian": three_launches()}, {"auto": {"gaussian": cpu_floors}}
    )
    a, b = identical_trees(tmp_path)
    add_timer(b)  # A has no timer, as B3.
    run = bench.compare(a, b, ["gaussian"], 6, False, profile=True)
    assert FakeBenchBackend.envs == [{"CHAINNER_C_PROFILE": "1"}] * 2
    result = only_result(tmp_path)
    assert result["profile"] is True
    # The seeded floors are usable and would judge this run, but none applies, and
    # no floor's recorded load is compared either.
    item = result["summary"]["gaussian"]
    assert bench.bench_stats.verdict(item, 0.05) == "neutral"
    assert bench.bench_stats.verdict(item["cpu"], 0.01) == "win"
    assert result["verdicts"] == {"gaussian": "unknown"}
    assert result["cpu_verdicts"] == {"gaussian": "unknown"}
    assert result["current_floors"] == {}  # The seeded CPU launches would give one.
    assert result["warnings"] == []
    # Only the side whose tree has the timer must log a table.
    profiled = {(t["backend"], t["native_profile"] == TABLE) for t in result["trials"]}
    assert profiled == {("A", False), ("B", True)}
    summary = (run / "summary.md").read_text(encoding="utf-8")
    for line in (
        "- Noise floors from: none usable",
        "- K setting: auto; CPU floors from: none usable",
        "- A environment: CHAINNER_C_PROFILE=1",
        "- B environment: CHAINNER_C_PROFILE=1",
    ):
        assert line + "\n" in summary
    assert PROFILE_NOTE in summary
    assert PROVISIONAL_NOTE not in summary
    # main() never confirms a profile run, even with a flagged case on hand.
    fake = FakeCompare(tmp_path, {"gaussian": "win"}, {"gaussian": "win"})
    monkeypatch.setattr(bench, "compare", fake)
    a, b = run_main(monkeypatch, tmp_path, "--cases", "gaussian", "--profile")
    assert fake.calls == [(a, b, ["gaussian"], 6, False)]
    assert fake.profiles == [True]
    assert fake.environments == [({}, {})]
    assert sorted(p.name for p in (tmp_path / "bench-1").iterdir()) == RUN_FILES


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        pytest.param("win", "win", "win", id="confirmed-win"),
        pytest.param("regression", "regression", "regression", id="confirmed-slowdown"),
        pytest.param("win", "neutral", "unconfirmed", id="win-then-neutral"),
        pytest.param("regression", "win", "unconfirmed", id="regression-then-win"),
        pytest.param("win", "regression", "unconfirmed", id="win-then-regression"),
        pytest.param("win", "unknown", "unconfirmed", id="win-then-unknown"),
        pytest.param("win", None, "unconfirmed", id="win-missing-from-second"),
        pytest.param("regression", None, "unconfirmed", id="regression-missing"),
        pytest.param("neutral", "win", "neutral", id="neutral-passes-through"),
        pytest.param("neutral", None, "neutral", id="neutral-missing-passes-through"),
        pytest.param("unknown", "regression", "unknown", id="unknown-passes-through"),
        pytest.param("unknown", None, "unknown", id="unknown-missing-passes-through"),
    ],
)
def test_final_verdicts_for_one_case(first, second, expected):
    later = {} if second is None else {"g": second}
    assert bench.final_verdicts({"g": first}, later) == {"g": expected}


def test_final_verdicts_cover_exactly_the_first_run_in_its_order():
    first = {"z": "win", "a": "neutral", "m": "regression", "k": "unknown"}
    second = {"m": "regression", "extra": "win", "z": "neutral"}
    before = (copy.deepcopy(first), copy.deepcopy(second))
    final = bench.final_verdicts(first, second)
    assert list(final) == ["z", "a", "m", "k"]  # No case from second alone.
    assert final == {
        "z": "unconfirmed",
        "a": "neutral",
        "m": "regression",
        "k": "unknown",
    }
    assert (first, second) == before


@pytest.mark.parametrize(
    ("first", "second", "third", "expected"),
    [
        pytest.param("regression", "regression", None, "regression", id="confirmed"),
        pytest.param(
            "regression",
            "regression",
            "neutral",
            "regression",
            id="two-of-three-second",
        ),
        pytest.param(
            "regression", "neutral", "regression", "regression", id="two-of-three-third"
        ),
        pytest.param(
            "regression",
            "win",
            "regression",
            "regression",
            id="opposite-then-regression",
        ),
        pytest.param(
            "regression", None, "regression", "regression", id="missing-then-regression"
        ),
        pytest.param(
            "regression", "neutral", "neutral", "unconfirmed", id="one-of-three"
        ),
        pytest.param(
            "regression", "neutral", "win", "unconfirmed", id="one-of-three-opposite"
        ),
        pytest.param(
            "regression", "unknown", "unknown", "unconfirmed", id="one-of-three-unknown"
        ),
        pytest.param(
            "regression", "neutral", None, "unconfirmed", id="no-tiebreak-run"
        ),
        pytest.param(
            "regression", None, None, "unconfirmed", id="missing-without-tiebreak"
        ),
        pytest.param("win", "win", "neutral", "win", id="confirmed-win"),
        pytest.param(
            "win", "win", "regression", "win", id="confirmed-win-ignores-third"
        ),
        pytest.param("win", "neutral", "win", "unconfirmed", id="win-is-not-rescued"),
        pytest.param(
            "win", "neutral", "regression", "unconfirmed", id="win-then-neutral-no-flip"
        ),
        pytest.param(
            "win", "regression", "win", "unconfirmed", id="win-is-not-rescued-2"
        ),
        # R13: a win that the confirmation run flips to regression needs the third.
        pytest.param(
            "win", "regression", "regression", "regression", id="flip-two-of-three"
        ),
        pytest.param(
            "win", "regression", "neutral", "unconfirmed", id="flip-one-of-three"
        ),
        pytest.param(
            "win", "regression", "unknown", "unconfirmed", id="flip-third-unknown"
        ),
        pytest.param(
            "win", "regression", None, "unconfirmed", id="flip-without-tiebreak-run"
        ),
        pytest.param(
            "neutral",
            "regression",
            "regression",
            "neutral",
            id="neutral-passes-through",
        ),
        pytest.param("unknown", "win", "win", "unknown", id="unknown-passes-through"),
    ],
)
def test_final_verdicts_with_a_tiebreak_run(first, second, third, expected):
    later = {} if second is None else {"g": second}
    tiebreak = None if third is None else {"g": third}
    assert bench.final_verdicts({"g": first}, later, tiebreak) == {"g": expected}


@pytest.mark.parametrize("first", ["regression", "win"])
def test_final_verdicts_treat_a_tiebreak_run_without_the_case_as_no_support(first):
    # For a win, the confirmation run flipped it to regression.
    second = {"g": "regression"} if first == "win" else {}
    assert bench.final_verdicts({"g": first}, second, {}) == {"g": "unconfirmed"}


def test_tiebreak_cases_are_disputed_regressions_and_flipped_wins():
    first = {
        "w1": "win",
        "r1": "regression",
        "w2": "win",
        "r2": "regression",
        "w3": "win",
        "r3": "regression",
        "w4": "win",
        "r4": "regression",
        "n": "neutral",
        "u": "unknown",
    }
    second = {
        "w1": "neutral",  # A win that went neutral: never tie-broken.
        "r1": "regression",  # A repeated regression: decided.
        "w2": "regression",  # A win that flipped to regression: tie-broken.
        "r2": "win",  # A regression that flipped to win: tie-broken.
        "w3": "win",  # A repeated win: decided.
        "w4": "unknown",  # A win that went unknown: never tie-broken.
        "r4": "neutral",  # A regression that went neutral: tie-broken.
        "n": "regression",  # Neutral and unknown cases are not flagged.
        "u": "regression",
    }
    # r3 is missing from the confirmation run; w4 is not tie-broken. First-run order.
    assert bench.tiebreak_cases(first, second) == ["w2", "r2", "r3", "r4"]
    assert bench.tiebreak_cases({"w": "win", "n": "neutral"}, {}) == []


WITHIN = "regression (within current A/A)"


@pytest.mark.parametrize(
    ("first", "second", "third", "expected"),
    [
        pytest.param(WITHIN, WITHIN, None, WITHIN, id="within-confirmed"),
        pytest.param(WITHIN, "neutral", WITHIN, WITHIN, id="within-two-of-three"),
        pytest.param(WITHIN, "neutral", "neutral", "unconfirmed", id="within-once"),
        # The block threshold needs 2 of the 3 runs beyond the current floor too.
        pytest.param("regression", WITHIN, "regression", "regression", id="blocks"),
        pytest.param("regression", WITHIN, WITHIN, WITHIN, id="beyond-once"),
        pytest.param("regression", WITHIN, "neutral", WITHIN, id="beyond-once-2"),
        pytest.param(WITHIN, "regression", "regression", "regression", id="blocks-2"),
        pytest.param(WITHIN, "neutral", "regression", WITHIN, id="beyond-in-third"),
        pytest.param("win", "regression", WITHIN, WITHIN, id="flip-within"),
        pytest.param("win", WITHIN, "regression", WITHIN, id="flip-within-2"),
        pytest.param("win", WITHIN, "neutral", "unconfirmed", id="flip-once"),
        pytest.param("win", WITHIN, WITHIN, WITHIN, id="flip-within-twice"),
    ],
)
def test_a_final_regression_blocks_only_beyond_the_current_floor_in_two_runs(
    first, second, third, expected
):
    tiebreak = None if third is None else {"g": third}
    assert bench.final_verdicts({"g": first}, {"g": second}, tiebreak) == {
        "g": expected
    }


def test_tiebreak_cases_include_a_regression_given_the_other_label():
    first = {"a": "regression", "b": WITHIN, "c": WITHIN, "d": "win", "e": WITHIN}
    second = {"a": WITHIN, "b": "regression", "c": WITHIN, "d": WITHIN, "e": "win"}
    # c repeated its label; the others need a third run to settle either label.
    assert bench.tiebreak_cases(first, second) == ["a", "b", "d", "e"]


def result_json(
    verdicts: dict[str, str],
    ratios: dict[str, float] | None,
    cpu_verdicts: dict[str, str] | None = None,
    cpu_ratios: dict[str, float] | None = None,
) -> dict:
    """A finished run's result.json: verdicts and each case's median paired ratio.

    A case has a CPU summary, holding its median CPU-per-item ratio, only when
    cpu_ratios gives one.
    """
    ratios = ratios or {}
    cpu_ratios = cpu_ratios or {}
    return {
        "success": True,
        "verdicts": verdicts,
        "cpu_verdicts": cpu_verdicts or {},
        "summary": {
            case: {
                "median_ratio": ratios.get(case, 1.0),
                "cpu": (
                    {"median_ratio": cpu_ratios[case]} if case in cpu_ratios else None
                ),
            }
            for case in verdicts
        },
    }


def write_run(
    run: Path,
    verdicts: dict[str, str],
    ratios: dict[str, float] | None = None,
    cpu_verdicts: dict[str, str] | None = None,
    cpu_ratios: dict[str, float] | None = None,
) -> Path:
    """A finished run directory: result.json and a summary.md ending in its table."""
    run.mkdir()
    (run / "result.json").write_text(
        json.dumps(result_json(verdicts, ratios, cpu_verdicts, cpu_ratios)),
        encoding="utf-8",
    )
    (run / "summary.md").write_text(f"# Benchmark {run.name}\n", encoding="utf-8")
    return run


RUN_FILES = ["result.json", "summary.md"]


class FakeCompare:
    """Stands in for bench.compare in confirm() and main(); starts no backend.

    Call n writes bench-n holding the n-th verdicts (and ratios, CPU verdicts and
    CPU ratios) it was given. Each call's per-side environments and interpreters,
    profile flag, PNG mode and cross-stack mode are recorded beside its positional
    arguments.
    """

    def __init__(
        self,
        tmp_path: Path,
        *verdicts: dict[str, str],
        ratios=(),
        cpu_verdicts=(),
        cpu_ratios=(),
    ):
        self.tmp_path = tmp_path
        self.verdicts = verdicts
        self.ratios = ratios
        self.cpu_verdicts = cpu_verdicts
        self.cpu_ratios = cpu_ratios
        self.calls: list[tuple] = []
        self.confirms: list[str | None] = []
        self.tiebreaks: list[bool] = []
        self.environments: list[tuple] = []
        self.profiles: list[bool] = []
        self.interpreters: list[tuple] = []
        self.png_modes: list[str] = []
        self.cross_stacks: list[bool] = []

    def __call__(
        self,
        a,
        b,
        cases,
        repeats,
        record_noise,
        confirms=None,
        tiebreak=False,
        env_a=None,
        env_b=None,
        profile=False,
        python_a=None,
        python_b=None,
        png_compare="sha256",
        cross_stack=False,
    ):
        n = len(self.calls)
        self.calls.append((a, b, list(cases), repeats, record_noise))
        self.confirms.append(confirms)
        self.tiebreaks.append(tiebreak)
        self.environments.append((env_a, env_b))
        self.profiles.append(profile)
        self.interpreters.append((python_a, python_b))
        self.png_modes.append(png_compare)
        self.cross_stacks.append(cross_stack)

        def nth(values):
            return values[n] if n < len(values) else None

        return write_run(
            self.tmp_path / f"bench-{n + 1}",
            self.verdicts[n],
            nth(self.ratios),
            nth(self.cpu_verdicts),
            nth(self.cpu_ratios),
        )


def read_record(run: Path) -> dict:
    return json.loads((run / "confirmation.json").read_text(encoding="utf-8"))


def test_confirm_reruns_only_flagged_cases_on_fresh_backends(
    tmp_path, monkeypatch, capsys
):
    first = write_run(
        tmp_path / "bench-first",
        {"x": "win", "y": "neutral", "z": "regression"},
        {"x": 1.5, "y": 1.0, "z": 0.5},
    )
    fake = FakeCompare(
        tmp_path,
        {"x": "win", "z": "regression"},
        ratios=({"x": 1.125, "z": 0.75},),
    )
    monkeypatch.setattr(bench, "compare", fake)
    a, b = tmp_path / "a", tmp_path / "b"
    final = bench.confirm(first, a, b, 8)
    # One confirmation run: only the flagged cases, in order, never --record-noise,
    # and it knows which run it confirms.
    assert fake.calls == [(a, b, ["x", "z"], 8, False)]
    assert fake.confirms == ["bench-first"]
    assert fake.tiebreaks == [False]
    assert final == {"x": "win", "y": "neutral", "z": "regression"}
    assert read_record(first) == {
        "first_run": "bench-first",
        "confirmation_run": "bench-1",
        "tiebreak_run": None,
        "flagged": ["x", "z"],
        "first": {"x": "win", "y": "neutral", "z": "regression"},
        "confirmation": {"x": "win", "z": "regression"},
        "tiebreak": {},
        "final": {"x": "win", "y": "neutral", "z": "regression"},
        "paired_change_percent": {
            "first": {"x": 50.0, "y": 0.0, "z": -50.0},
            "confirmation": {"x": 12.5, "z": -25.0},
            "tiebreak": {},
        },
    }
    rows = [
        "| x | win (+50.0%) | win (+12.5%) | - | win |\n",
        "| z | regression (-50.0%) | regression (-25.0%) | - | regression |\n",
    ]
    table = (first / "confirmation.md").read_text(encoding="utf-8")
    assert "| Case | First | Confirmation | Tie-break | Final |\n" in table
    for row in rows:
        assert row in table
    assert "| y |" not in table  # Only flagged cases are confirmed.
    for text in (
        "Runs: first `bench-first`, confirmation `bench-1`, tie-break none.",
        (
            "Claims require the Final column: a regression counts if 2 of the 3 runs "
            "flag it, and a win counts only if the confirmation run repeated it"
        ),
        "wins it flipped to regression",
        (
            "A regression blocks only when 2 of the 3 runs put it beyond the current "
            "floor (the A/A launches of the run's K setting); otherwise it is final as "
            '"regression (within current A/A)", which does not block'
        ),
        "Unconfirmed means inconclusive, not neutral",
        "an unconfirmed regression is not cleared",
        "Always quote the confirmation run's paired change",
        "first-run magnitudes are inflated by selection",
        "raises the false-claim rate",
        "thermal state, persistent background load",
    ):
        assert text in table
    # The magnitude advice has one source, never the tie-break run.
    assert "tie-break run for a regression" not in table
    # The first run's summary.md gets the same table under its own heading.
    summary = (first / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith("# Benchmark bench-first\n")
    heading, section = summary.split("\n## Confirmed verdicts\n", 1)
    assert heading == "# Benchmark bench-first\n"
    assert "| Case | First | Confirmation | Tie-break | Final |\n" in section
    for row in rows:
        assert row in section
    assert "confirmation.md" in section
    # Everything lands in the first run's directory, none in the confirmation run's.
    assert sorted(p.name for p in (tmp_path / "bench-1").iterdir()) == RUN_FILES
    assert (tmp_path / "bench-1" / "summary.md").read_text(
        encoding="utf-8"
    ) == "# Benchmark bench-1\n"
    printed = capsys.readouterr().out.splitlines()
    assert [line for line in printed if "FINAL" in line] == [
        "FINAL x: win",
        "FINAL z: regression",
    ]


def test_confirm_tiebreaks_disputed_regressions_and_flipped_wins_in_first_run_order(
    tmp_path, monkeypatch, capsys
):
    first = write_run(
        tmp_path / "bench-first",
        {
            "a": "win",  # Goes neutral: unconfirmed, never tie-broken.
            "b": "regression",  # Repeated: decided.
            "c": "regression",  # Goes neutral: tie-broken.
            "w": "win",  # Flips to regression: tie-broken.
            "d": "regression",  # Missing from the confirmation run: tie-broken.
            "v": "win",  # Flips to regression: tie-broken.
            "e": "neutral",
        },
        {"a": 1.5, "b": 0.5, "c": 0.75, "w": 1.25, "d": 0.5, "v": 1.125, "e": 1.0},
    )
    fake = FakeCompare(
        tmp_path,
        {
            "a": "neutral",
            "b": "regression",
            "c": "neutral",
            "w": "regression",
            "v": "regression",
        },
        {"c": "regression", "w": "regression", "d": "neutral", "v": "neutral"},
        ratios=(
            {"a": 1.0, "b": 0.75, "c": 1.0, "w": 0.5, "v": 0.75},
            {"c": 0.5, "w": 0.75, "d": 1.0, "v": 1.0},
        ),
    )
    monkeypatch.setattr(bench, "compare", fake)
    a, b = tmp_path / "a", tmp_path / "b"
    final = bench.confirm(first, a, b, 6)
    # One tie-break run gets exactly the flip cases together with the disputed
    # regressions, in the first run's order: not a (a plain failed win) or b.
    assert fake.calls == [
        (a, b, ["a", "b", "c", "w", "d", "v"], 6, False),
        (a, b, ["c", "w", "d", "v"], 6, False),
    ]
    assert fake.confirms == ["bench-first", "bench-first"]
    assert fake.tiebreaks == [False, True]
    assert final == {
        "a": "unconfirmed",
        "b": "regression",
        "c": "regression",  # First and tie-break flagged it: 2 of 3.
        "w": "regression",  # Confirmation and tie-break flagged it: 2 of 3.
        "d": "unconfirmed",  # Only the first run flagged it.
        "v": "unconfirmed",  # Only the confirmation run flagged it.
        "e": "neutral",
    }
    record = read_record(first)
    assert record["confirmation_run"] == "bench-1"
    assert record["tiebreak_run"] == "bench-2"
    assert record["flagged"] == ["a", "b", "c", "w", "d", "v"]
    assert record["tiebreak"] == {
        "c": "regression",
        "w": "regression",
        "d": "neutral",
        "v": "neutral",
    }
    assert record["final"] == final
    assert record["paired_change_percent"]["tiebreak"] == {
        "c": -50.0,
        "w": -25.0,
        "d": 0.0,
        "v": 0.0,
    }
    table = (first / "confirmation.md").read_text(encoding="utf-8")
    for row in (
        "| a | win (+50.0%) | neutral (+0.0%) | - | unconfirmed |\n",
        "| b | regression (-50.0%) | regression (-25.0%) | - | regression |\n",
        "| c | regression (-25.0%) | neutral (+0.0%) | regression (-50.0%) | regression |\n",
        "| w | win (+25.0%) | regression (-50.0%) | regression (-25.0%) | regression |\n",
        # d is absent from the confirmation run, which shows as n/a.
        "| d | regression (-50.0%) | n/a | neutral (+0.0%) | unconfirmed |\n",
        "| v | win (+12.5%) | regression (-25.0%) | neutral (+0.0%) | unconfirmed |\n",
    ):
        assert row in table
    assert "| e |" not in table
    assert "tie-break `bench-2`" in table
    printed = capsys.readouterr().out.splitlines()
    assert [line for line in printed if "FINAL" in line] == [
        "FINAL a: unconfirmed",
        "FINAL b: regression",
        "FINAL c: regression",
        "FINAL w: regression",
        "FINAL d: unconfirmed",
        "FINAL v: unconfirmed",
    ]
    for name in ("bench-1", "bench-2"):
        assert sorted(p.name for p in (tmp_path / name).iterdir()) == RUN_FILES


def test_cpu_flags_never_join_confirmation(tmp_path, monkeypatch, capsys):
    # CPU per item is information only: a CPU win or regression sends no case to a
    # confirmation run, though a result is on hand for one.
    neutral = {"x": "neutral", "y": "neutral"}
    first = write_run(
        tmp_path / "bench-first",
        neutral,
        cpu_verdicts={"x": "win", "y": "regression"},
        cpu_ratios={"x": 1.5, "y": 0.5},
    )
    fake = FakeCompare(tmp_path, neutral, cpu_verdicts=({"x": "win"},))
    monkeypatch.setattr(bench, "compare", fake)
    a, b = tmp_path / "a", tmp_path / "b"
    assert bench.confirm(first, a, b, 6) == neutral
    assert fake.calls == []
    assert sorted(p.name for p in first.iterdir()) == RUN_FILES
    # Beside a throughput flag only that case is confirmed, and nothing about CPU
    # is settled, recorded or printed.
    first = write_run(
        tmp_path / "bench-other",
        {"x": "win", "y": "neutral"},
        {"x": 1.5},
        cpu_verdicts={"x": "regression", "y": "win"},
        cpu_ratios={"x": 0.5, "y": 1.5},
    )
    fake = FakeCompare(
        tmp_path,
        {"x": "win"},
        ratios=({"x": 1.25},),
        cpu_verdicts=({"x": "regression"},),
        cpu_ratios=({"x": 0.5},),
    )
    monkeypatch.setattr(bench, "compare", fake)
    assert bench.confirm(first, a, b, 6) == {"x": "win", "y": "neutral"}
    assert fake.calls == [(a, b, ["x"], 6, False)]
    record = read_record(first)
    assert record["flagged"] == ["x"]
    assert "cpu" not in record
    text = (first / "confirmation.md").read_text(encoding="utf-8")
    assert "| x | win (+50.0%) | win (+25.0%) | - | win |\n" in text
    assert "| y |" not in text
    assert "CPU per item" not in text
    summary = (first / "summary.md").read_text(encoding="utf-8")
    assert "CPU per item" not in summary.split("\n## Confirmed verdicts\n", 1)[1]
    printed = capsys.readouterr().out.splitlines()
    assert [line for line in printed if "FINAL" in line] == ["FINAL x: win"]


@pytest.mark.parametrize(("ratio", "expected"), [(0.96, WITHIN), (0.93, "regression")])
def test_regression_between_floors_is_labelled_and_does_not_block(
    tmp_path, monkeypatch, capsys, ratio, expected
):
    def make_record(label, case, pair):
        # B takes 1 / ratio seconds, so every pair's ratio is ratio.
        seconds = 1 / ratio if label == "B" else 1.0
        return {**trial_record(label, case, pair), "seconds": seconds}

    stub_compare(tmp_path, monkeypatch, make_record)
    monkeypatch.setenv("CHAINNER_C_ITEM_WINDOW", "1")

    def launches(throughput_floor):
        return entry(
            *(cpu_launch(f"bench-{n}", 0.5, throughput_floor) for n in (1, 2, 3))
        )

    # The shared floor is 2% and K = 1's current floor 6%; auto's current floor (1%)
    # would make either change a plain regression.
    write_noise(
        tmp_path,
        {"gaussian": entry(*(launch(f"bench-{n}", 0.02) for n in (1, 2, 3)))},
        {"k1": {"gaussian": launches(0.06)}, "auto": {"gaussian": launches(0.01)}},
    )
    a, b = tmp_path / "a", tmp_path / "b"
    first = bench.compare(a, b, ["gaussian"], 6, False)
    result = only_result(tmp_path)
    assert result["success"] is True
    assert result["current_floors"] == {"gaussian": 0.06}
    assert result["verdicts"] == {"gaussian": expected}
    change = f"{100 * (ratio - 1):+.1f}%"
    warning = (
        f"gaussian: regression {change} is beyond the shared floor +2.0/-2.0% but "
        "within the current floor +6.2/-5.8% of the k1 A/A launches; it does not "
        "block and goes to the stand-in"
    )
    assert result["warnings"] == ([warning] if expected == WITHIN else [])
    summary = (first / "summary.md").read_text(encoding="utf-8")
    assert f"| {change} | 0/6 | +2.0/-2.0% | +6.2/-5.8% | {expected} |" in summary
    # Flagged like a regression; the confirmation run repeats the label, which the
    # Final column keeps.
    fake = FakeCompare(tmp_path, {"gaussian": expected}, ratios=({"gaussian": ratio},))
    monkeypatch.setattr(bench, "compare", fake)
    assert bench.confirm(first, a, b, 6) == {"gaussian": expected}
    assert fake.calls == [(a, b, ["gaussian"], 6, False)]
    row = (
        f"| gaussian | {expected} ({change}) | {expected} ({change}) | - | {expected} |"
    )
    assert row + "\n" in (first / "confirmation.md").read_text(encoding="utf-8")
    assert f"FINAL gaussian: {expected}" in capsys.readouterr().out.splitlines()
    assert only_result(tmp_path)["success"] is True  # Confirming changes no record.


@pytest.mark.parametrize(
    ("third", "expected"),
    [("regression", "regression"), ("neutral", "unconfirmed")],
)
def test_confirm_tiebreak_decides_a_disputed_regression(
    tmp_path, monkeypatch, third, expected
):
    first = write_run(tmp_path / "bench-first", {"x": "regression"})
    fake = FakeCompare(tmp_path, {"x": "neutral"}, {"x": third})
    monkeypatch.setattr(bench, "compare", fake)
    final = bench.confirm(first, tmp_path / "a", tmp_path / "b", 6)
    assert [call[2] for call in fake.calls] == [["x"], ["x"]]
    assert final == {"x": expected}
    assert read_record(first)["final"] == {"x": expected}


@pytest.mark.parametrize(
    ("third", "expected"),
    [
        ("regression", "regression"),  # Win, regression, regression.
        ("neutral", "unconfirmed"),  # Win, regression, neutral.
        ("win", "unconfirmed"),
    ],
)
def test_confirm_tiebreak_decides_a_win_flipped_to_regression(
    tmp_path, monkeypatch, third, expected
):
    first = write_run(tmp_path / "bench-first", {"x": "win"})
    fake = FakeCompare(tmp_path, {"x": "regression"}, {"x": third})
    monkeypatch.setattr(bench, "compare", fake)
    final = bench.confirm(first, tmp_path / "a", tmp_path / "b", 6)
    assert [call[2] for call in fake.calls] == [["x"], ["x"]]
    assert fake.tiebreaks == [False, True]
    assert final == {"x": expected}
    record = read_record(first)
    assert record["final"] == {"x": expected}
    assert record["tiebreak_run"] == "bench-2"


@pytest.mark.parametrize(
    ("confirmation", "expected"),
    [
        ({"x": "neutral"}, "unconfirmed"),
        ({"x": "unknown"}, "unconfirmed"),
        ({}, "unconfirmed"),  # Missing from the confirmation run.
        ({"x": "win"}, "win"),
    ],
)
def test_confirm_never_tiebreaks_a_win_that_did_not_flip(
    tmp_path, monkeypatch, confirmation, expected
):
    first = write_run(tmp_path / "bench-first", {"x": "win"})
    # A second result is on hand, but a third run must not be made.
    fake = FakeCompare(tmp_path, confirmation, {"x": "regression"})
    monkeypatch.setattr(bench, "compare", fake)
    final = bench.confirm(first, tmp_path / "a", tmp_path / "b", 6)
    assert len(fake.calls) == 1
    assert final == {"x": expected}
    record = read_record(first)
    assert record["tiebreak_run"] is None
    assert record["tiebreak"] == {}


def test_confirm_without_flagged_cases_runs_and_writes_nothing(
    tmp_path, monkeypatch, capsys
):
    verdicts = {"x": "neutral", "y": "unknown"}
    first = write_run(tmp_path / "bench-first", verdicts)
    fake = FakeCompare(tmp_path)
    monkeypatch.setattr(bench, "compare", fake)
    assert bench.confirm(first, tmp_path / "a", tmp_path / "b", 6) == verdicts
    assert fake.calls == []
    assert sorted(p.name for p in first.iterdir()) == RUN_FILES
    assert (first / "summary.md").read_text(encoding="utf-8") == (
        "# Benchmark bench-first\n"
    )
    assert "FINAL" not in capsys.readouterr().out


def test_confirm_lets_a_failed_confirmation_run_propagate(tmp_path, monkeypatch):
    first = write_run(tmp_path / "bench-first", {"x": "win"})

    def failing_compare(*args, **options):
        raise RuntimeError("A startup timeout")

    monkeypatch.setattr(bench, "compare", failing_compare)
    with pytest.raises(RuntimeError, match="A startup timeout"):
        bench.confirm(first, tmp_path / "a", tmp_path / "b", 6)
    # The first run's own records stay as they were; no partial confirmation.
    assert sorted(p.name for p in first.iterdir()) == RUN_FILES
    assert (first / "summary.md").read_text(encoding="utf-8") == (
        "# Benchmark bench-first\n"
    )


def test_confirm_lets_a_failed_tiebreak_run_propagate(tmp_path, monkeypatch):
    first = write_run(tmp_path / "bench-first", {"x": "regression"})
    fake = FakeCompare(tmp_path, {"x": "neutral"})

    def compare_then_fail(*args, **options):
        if fake.calls:
            raise RuntimeError("B startup timeout")
        return fake(*args, **options)

    monkeypatch.setattr(bench, "compare", compare_then_fail)
    with pytest.raises(RuntimeError, match="B startup timeout"):
        bench.confirm(first, tmp_path / "a", tmp_path / "b", 6)
    assert sorted(p.name for p in first.iterdir()) == RUN_FILES


def run_main(monkeypatch, tmp_path: Path, *options: str) -> tuple[Path, Path]:
    """Run bench.main() with these options; return the resolved --a and --b trees."""
    monkeypatch.delenv("CHAINNER_C_ITEM_WINDOW", raising=False)
    monkeypatch.delenv("CHAINNER_C_PROFILE", raising=False)
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setattr(
        sys, "argv", ["bench.py", "--a", str(a), "--b", str(b), *options]
    )
    bench.main()
    return a.resolve(), b.resolve()


def test_main_confirms_flagged_cases_with_further_runs(tmp_path, monkeypatch):
    fake = FakeCompare(
        tmp_path,
        {"gaussian": "win", "caption": "neutral", "morphology": "regression"},
        {"gaussian": "win", "morphology": "neutral"},
        {"morphology": "neutral"},
    )
    monkeypatch.setattr(bench, "compare", fake)
    cases = ["gaussian", "caption", "morphology"]
    a, b = run_main(monkeypatch, tmp_path, "--cases", *cases, "--repeats", "8")
    assert fake.calls == [
        (a, b, cases, 8, False),
        (a, b, ["gaussian", "morphology"], 8, False),
        (a, b, ["morphology"], 8, False),  # The disputed regression only.
    ]
    # The first run is an ordinary run; the other two know which run they confirm,
    # and only the last one is the tie-break.
    assert fake.confirms == [None, "bench-1", "bench-1"]
    assert fake.tiebreaks == [False, False, True]
    record = read_record(tmp_path / "bench-1")
    assert record["final"] == {
        "gaussian": "win",
        "caption": "neutral",
        "morphology": "unconfirmed",
    }
    assert record["confirmation_run"] == "bench-2"
    assert record["tiebreak_run"] == "bench-3"
    summary = (tmp_path / "bench-1" / "summary.md").read_text(encoding="utf-8")
    assert "\n## Confirmed verdicts\n" in summary
    for name in ("bench-2", "bench-3"):
        assert sorted(p.name for p in (tmp_path / name).iterdir()) == RUN_FILES


def test_main_record_noise_never_runs_a_confirmation(tmp_path, monkeypatch, capsys):
    fake = FakeCompare(tmp_path, {"gaussian": "win"}, {"gaussian": "win"})
    monkeypatch.setattr(bench, "compare", fake)
    a, b = run_main(
        monkeypatch, tmp_path, "--cases", "gaussian", "--record-noise", "--repeats", "6"
    )
    assert fake.calls == [(a, b, ["gaussian"], 6, True)]
    assert sorted(p.name for p in (tmp_path / "bench-1").iterdir()) == RUN_FILES
    assert "## Confirmed verdicts" not in (
        (tmp_path / "bench-1" / "summary.md").read_text(encoding="utf-8")
    )
    assert "FINAL" not in capsys.readouterr().out


def test_main_without_flagged_cases_runs_no_confirmation(tmp_path, monkeypatch):
    fake = FakeCompare(
        tmp_path, {"gaussian": "neutral", "caption": "unknown"}, {"gaussian": "win"}
    )
    monkeypatch.setattr(bench, "compare", fake)
    a, b = run_main(monkeypatch, tmp_path, "--cases", "gaussian", "caption")
    assert fake.calls == [(a, b, ["gaussian", "caption"], 6, False)]
    # Without the options, compare() picks the packaged interpreter for both sides.
    assert fake.interpreters == [(None, None)]
    assert fake.png_modes == ["sha256"]
    assert fake.cross_stacks == [False]
    assert sorted(p.name for p in (tmp_path / "bench-1").iterdir()) == RUN_FILES
    assert not (tmp_path / "bench-2").exists()


def test_confirmation_runs_keep_each_sides_interpreter_and_the_png_mode(
    tmp_path, monkeypatch
):
    fake = FakeCompare(
        tmp_path,
        {"gaussian": "win", "morphology": "regression"},
        {"gaussian": "win", "morphology": "neutral"},
        {"morphology": "neutral"},
    )
    monkeypatch.setattr(bench, "compare", fake)
    installed = tmp_path / "installed" / "python.exe"
    other = tmp_path / "other" / "python.exe"
    run_main(
        monkeypatch,
        tmp_path,
        "--cases",
        "gaussian",
        "morphology",
        "--python-a",
        str(installed),
        "--python-b",
        str(other),
        "--png-compare",
        "decoded",
    )
    # The first run, its confirmation run and the tie-break run.
    assert len(fake.calls) == 3
    assert fake.interpreters == [(installed.resolve(), other.resolve())] * 3
    assert fake.png_modes == ["decoded"] * 3


def test_confirmation_runs_keep_the_cross_stack_mode(tmp_path, monkeypatch):
    fake = FakeCompare(
        tmp_path,
        {"gaussian": "win", "morphology": "regression"},
        {"gaussian": "win", "morphology": "neutral"},
        {"morphology": "neutral"},
    )
    monkeypatch.setattr(bench, "compare", fake)
    run_main(
        monkeypatch, tmp_path, "--cases", "gaussian", "morphology", "--cross-stack"
    )
    # The first run, its confirmation run and the tie-break run.
    assert len(fake.calls) == 3
    assert fake.cross_stacks == [True] * 3


def test_confirmation_runs_keep_each_sides_environment(tmp_path, monkeypatch):
    fake = FakeCompare(
        tmp_path,
        {"gaussian": "win", "morphology": "regression"},
        {"gaussian": "win", "morphology": "neutral"},
        {"morphology": "neutral"},
    )
    monkeypatch.setattr(bench, "compare", fake)
    a, b = run_main(
        monkeypatch,
        tmp_path,
        "--cases",
        "gaussian",
        "morphology",
        "--env-a",
        "CHAINNER_C_ISA=avx2",
        "--env-b",
        "X=1",
        "--env-b",
        "Y=2",
    )
    # The first run, its confirmation run and the tie-break run.
    assert fake.calls == [
        (a, b, ["gaussian", "morphology"], 6, False),
        (a, b, ["gaussian", "morphology"], 6, False),
        (a, b, ["morphology"], 6, False),
    ]
    sides = ({"CHAINNER_C_ISA": "avx2"}, {"X": "1", "Y": "2"})
    assert fake.environments == [sides] * 3
    assert fake.profiles == [False] * 3


def profile_line(table: dict) -> bytes:
    """The timer's backend.log line (P6): sorted keys, no spaces, CRLF."""
    text = json.dumps(table, sort_keys=True, separators=(",", ":"))
    return f"[x] [1] [INFO] [Worker] native profile: {text}\r\n".encode()


def test_native_profile_reads_exactly_one_table_after_the_offset(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(bench_backend, "time", clock)
    earlier = {"cn_gaussian_f32": {"calls": 9, "inclusive_ns": 9, "exclusive_ns": 9}}
    log = tmp_path / "backend.log"
    log.write_bytes(profile_line(earlier))  # The previous trial's table.
    offset = log.stat().st_size
    with log.open("ab") as stream:
        stream.write(b"[x] [1] [INFO] [Worker] item window K=6 (auto)\r\n")
        stream.write(profile_line(TABLE))
    assert bench.native_profile(log, offset, True) == TABLE
    # A table in a trial that is not profiled means the timer was on in a gate.
    with pytest.raises(RuntimeError, match="not profiled: the timer was on"):
        bench.native_profile(log, offset, False)
    end = log.stat().st_size
    assert bench.native_profile(log, end, False) is None
    assert clock.sleeps == []  # Found at once, or not required.
    with pytest.raises(RuntimeError, match="no native profile"):
        bench.native_profile(log, end, True)
    assert set(clock.sleeps) == {0.1}
    assert sum(clock.sleeps) == pytest.approx(10, abs=0.2)
    with log.open("ab") as stream:
        stream.write(profile_line(TABLE))
    for required in (True, False):
        with pytest.raises(RuntimeError, match="2 native profile lines"):
            bench.native_profile(log, offset, required)


def test_item_windows_reads_only_new_decisions(tmp_path):
    log = tmp_path / "backend.log"
    log.write_bytes(b"[x] [1] [INFO] [Worker] item window K=1 (forced)\r\n")
    offset = log.stat().st_size
    with log.open("ab") as stream:
        stream.write(b"[x] [1] [INFO] [Worker] Running new executor...\r\n")
        stream.write(b"[x] [1] [INFO] [Worker] item window K=6 (auto)\r\n")
        stream.write(
            b"[x] [1] [INFO] [Worker] item window K=6: 32 items, 30 ahead, 31 jobs, 30 replayed, 1 awaited, 0 discarded\r\n"
        )
        stream.write(
            b"[x] [1] [INFO] [Worker] item window K=1 (source under output directory: save)\r\n"
        )
    windows, end = bench.item_windows(log, offset)
    assert windows == [
        {
            "k": 6,
            "reason": "auto",
            "items": 32,
            "ahead": 30,
            "jobs": 31,
            "replayed": 30,
            "awaited": 1,
            "discarded": 0,
        },
        {
            "k": 1,
            "reason": "source under output directory: save",
        },  # K = 1 logs no summary
    ]
    assert end == log.stat().st_size
    assert bench.item_windows(log, end) == ([], end)


def test_item_windows_keeps_a_summary_whose_decision_is_missing(tmp_path):
    log = tmp_path / "backend.log"
    log.write_bytes(
        b"[x] [1] [INFO] [Worker] item window K=6: 8 items, 5 ahead\r\n"
        b"[x] [1] [INFO] [Worker] item window K=1 (forced)\r\n"
        b"[x] [1] [INFO] [Worker] item window K=3: 4 items, 2 ahead\r\n"
        b"[x] [1] [INFO] [Worker] item window K=3 (auto)\r\n"
        b"[x] [1] [INFO] [Worker] item window K=3: 6 items, 4 ahead\r\n"
        b"[x] [1] [INFO] [Worker] item window K=3: 2 items, 1 ahead\r\n"
    )
    # A summary without its decision gets an entry of its own: first in the log,
    # after a decision of another K, after a decision that already has its summary.
    assert bench.item_windows(log, 0)[0] == [
        {"k": 6, "reason": None, "items": 8, "ahead": 5},
        {"k": 1, "reason": "forced"},
        {"k": 3, "reason": None, "items": 4, "ahead": 2},
        {"k": 3, "reason": "auto", "items": 6, "ahead": 4},
        {"k": 3, "reason": None, "items": 2, "ahead": 1},
    ]


def test_environment_records_the_affinity_cpu_count(monkeypatch):
    monkeypatch.setattr(bench.psutil, "cpu_percent", lambda interval: 3.0)
    assert bench.environment()["affinity_cpus"] == len(
        bench.psutil.Process().cpu_affinity()
    )


@pytest.mark.parametrize(("count", "expected"), [(5, 5), (None, 1)])
def test_environment_without_an_affinity_api(monkeypatch, count, expected):
    monkeypatch.setattr(bench.psutil, "cpu_percent", lambda interval: 3.0)
    monkeypatch.delattr(bench.psutil.Process, "cpu_affinity")  # as on macOS
    monkeypatch.setattr(bench.os, "cpu_count", lambda: count)
    assert bench.environment()["affinity_cpus"] == expected
