"""The runtime verifiers' baseline (the oracle) and SSE comparison policy.

Every verifier's baseline is make_oracle.py's tree on the provisioned runtime the
package was copied from, and verify_runtime.oracle_source refuses one whose
chainner_ext differs from the package's, or any other oracle interpreter, before
anything runs.

verify_runtime.sse_event_mismatches is the one rule every runtime verifier uses.
Upstream re-runs a generator node once per item, and each re-run sends one more
node-start and one more node-broadcast; the port runs a generator node once.
Upstream sends each broadcast as its own task, and the port keeps one latest
pending preview per node, so it may send fewer previews, never another.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import verify_framework_runtime
import verify_io_execution_runtime
import verify_runtime
import verify_utility_runtime
import verify_video_runtime

RUN = Path("C:/run")
# chaiNNer-C's chainner_ext as backend/src (and the package manifest) names it.
EXT = {
    "chainner_ext/__init__.py": b"from . import chainner_ext\n",
    "chainner_ext/__init__.pyi": b"def f() -> None: ...\n",
    "chainner_ext/chainner_ext.pyd": b"MZ pyd",
    "nodes/impl/chainner_native.dll": b"MZ dll",
}
VERIFIERS = [
    verify_runtime,
    verify_io_execution_runtime,
    verify_utility_runtime,
    verify_video_runtime,
    verify_framework_runtime,
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data=b""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def make_oracle_tree(oracle, ext=EXT):
    """A make_oracle.py tree: upstream files, the NCNN selectors as compat.diff
    leaves them, and chainner_ext with the DLL beside the pyd."""
    write(oracle / "src/run.py", b"import nodes\n")
    for relative in verify_framework_runtime.CPU_SELECTORS:
        write(oracle / "src" / relative, b"use_gpu = ncnn.get_gpu_count() > 0\n")
    for name, dest in verify_runtime.CHAINNER_EXT.items():
        write(oracle / "src" / dest, ext[name])
    record = {
        "app_version": "0.25.1-nightly.2025-10-21",
        "source_files": {"run.py": sha(b"import nodes\n")},
        "compat_diff_sha256": sha(b""),
        "chainner_ext": {name: sha(data) for name, data in ext.items()},
    }
    verify_runtime.write_json(oracle / "oracle.json", record)
    return oracle


def package_manifest(ext=EXT, **records):
    return {
        "backend": {
            "resources/src/" + name: {"sha256": sha(data), "bytes": len(data)}
            for name, data in ext.items()
        },
        **records,
    }


# The tested versions of the dependencies declared_metadata() declares.
PINS = {
    "einops": "0.8.2",
    "google-re2": "1.1.20251105",
    "ncnn": "1.0.20260526",
    "onnxruntime-gpu": "1.30.0",
    "PyMatting": "1.1.16",
    "torch": "2.14.1+cu132",
}
# The project's lock as provision_runtime.ps1 writes it and as checked out (CRLF);
# the package records it as git stores it.
LOCK = "\r\n".join(
    [
        "# Interpreter: CPython 3.14.8 (standard build, not free-threaded).",
        "# Build tag: python-build-standalone 20261003",
        "# Archive SHA-256: " + "0" * 64,
        "# pip: 26.2.1 (installed; pip freeze does not list it)",
        *(f"{name}=={version}" for name, version in PINS.items()),
        "",
    ]
).encode()


@pytest.fixture
def provisioned(tmp_path):
    """A project's lock and the runtime a package recorded with it."""
    project = tmp_path / "project"
    write(project / "native/python-stack.lock.txt", LOCK)
    runtime = tmp_path / "runtime/cpython-3.14.8"
    write(runtime / "python.exe", b"MZ python")
    return {
        "identity": {"project": str(project)},
        "python_stack": {
            "runtime": str(runtime),
            "lock_sha256": sha(LOCK.replace(b"\r\n", b"\n")),
        },
    }


def test_an_oracle_with_the_packages_chainner_ext_is_the_baseline(
    tmp_path, provisioned
):
    oracle = make_oracle_tree(tmp_path / "oracle")
    source, python, record = verify_runtime.oracle_source(
        package_manifest(**provisioned), oracle
    )
    assert source == oracle / "src"
    assert python == Path(provisioned["python_stack"]["runtime"]) / "python.exe"
    assert record == {
        "app_version": "0.25.1-nightly.2025-10-21",
        "compat_diff_sha256": sha(b""),
        "chainner_ext": {name: sha(data) for name, data in EXT.items()},
        "python": str(python),
    }


@pytest.mark.parametrize(
    "change", ["no-python-stack", "lock-changed", "no-lock", "no-runtime"]
)
def test_any_interpreter_but_the_provisioned_runtime_is_refused(
    tmp_path, provisioned, change
):
    oracle = make_oracle_tree(tmp_path / "oracle")
    lock = Path(provisioned["identity"]["project"]) / "native/python-stack.lock.txt"
    runtime = Path(provisioned["python_stack"]["runtime"])
    if change == "no-python-stack":
        del provisioned["python_stack"]
    elif change == "lock-changed":
        write(lock, LOCK + b"numpy==2.5.4\r\n")
    elif change == "no-lock":
        lock.unlink()
    else:
        (runtime / "python.exe").unlink()
    with pytest.raises(ValueError, match="Oracle interpreter refused"):
        verify_runtime.oracle_source(package_manifest(**provisioned), oracle)


@pytest.mark.parametrize("name", list(EXT))
def test_an_oracle_built_from_other_binaries_is_refused(tmp_path, name):
    oracle = make_oracle_tree(tmp_path / "oracle")
    rebuilt = {**EXT, name: EXT[name] + b" rebuilt"}
    with pytest.raises(ValueError, match="Stale oracle refused") as error:
        verify_runtime.oracle_source(package_manifest(rebuilt), oracle)
    assert f"for ['{name}']" in str(error.value)


@pytest.mark.parametrize(
    "recorded",
    [
        None,
        {name: sha(data) for name, data in EXT.items() if name.endswith(".dll")},
        {**{name: sha(data) for name, data in EXT.items()}, "chainner_ext/x.py": "0"},
    ],
    ids=["record-before-chainner-ext", "missing-files", "extra-file"],
)
def test_an_oracle_record_without_exactly_the_four_files_is_refused(tmp_path, recorded):
    oracle = make_oracle_tree(tmp_path / "oracle")
    record = json.loads((oracle / "oracle.json").read_text(encoding="utf-8"))
    if recorded is None:
        del record["chainner_ext"]
    else:
        record["chainner_ext"] = recorded
    verify_runtime.write_json(oracle / "oracle.json", record)
    with pytest.raises(ValueError, match="Stale oracle refused"):
        verify_runtime.oracle_source(package_manifest(), oracle)


@pytest.mark.parametrize("dest", list(verify_runtime.CHAINNER_EXT.values()))
@pytest.mark.parametrize("change", ["edited", "removed"])
def test_an_oracle_tree_that_differs_from_its_record_is_refused(tmp_path, dest, change):
    oracle = make_oracle_tree(tmp_path / "oracle")
    if change == "edited":
        write(oracle / "src" / dest, b"other bytes")
    else:
        (oracle / "src" / dest).unlink()
    with pytest.raises(ValueError, match="tree's chainner_ext differs") as error:
        verify_runtime.oracle_source(package_manifest(), oracle)
    assert dest in str(error.value)


def test_a_missing_oracle_record_or_package_entry_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Oracle record missing"):
        verify_runtime.oracle_source(package_manifest(), tmp_path / "oracle")
    oracle = make_oracle_tree(tmp_path / "oracle")
    manifest = package_manifest()
    del manifest["backend"]["resources/src/chainner_ext/chainner_ext.pyd"]
    with pytest.raises(ValueError, match="does not record chaiNNer-C's chainner_ext"):
        verify_runtime.oracle_source(manifest, oracle)


@pytest.fixture
def verifier_setup(tmp_path, monkeypatch, provisioned):
    """A completed fake package and an oracle, with each verifier's PROJECT here."""
    project = Path(provisioned["identity"]["project"])
    (project / "native/reports").mkdir(parents=True)
    package = (tmp_path / "package").resolve()
    write(package / "python/python/python.exe")
    write(package / "portable")
    tool = tmp_path / "ffmpeg.exe"
    write(tool)

    def setup(module, ext=EXT, stack=provisioned["python_stack"]):
        manifest = {
            "state": "complete",
            "identity": {"destination": str(package), "project": str(project)},
            **package_manifest(ext),
            **({"python_stack": stack} if stack else {}),
        }
        verify_runtime.write_json(package / "chainner-c-package.json", manifest)
        oracle = make_oracle_tree(tmp_path / "oracle")
        monkeypatch.setattr(module, "PROJECT", project)
        monkeypatch.setattr(module, "ORACLE", oracle)
        tools = ["--ffmpeg", str(tool), "--ffprobe", str(tool)]
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify",
                "--package",
                str(package),
                *(tools if module is verify_video_runtime else []),
            ],
        )
        return project / "native/reports", oracle

    return setup


@pytest.mark.parametrize("module", VERIFIERS, ids=lambda module: module.__name__)
def test_each_verifier_refuses_a_stale_oracle_before_writing(verifier_setup, module):
    reports, _ = verifier_setup(
        module, {**EXT, "nodes/impl/chainner_native.dll": b"new"}
    )
    with pytest.raises(ValueError, match="Stale oracle refused"):
        module.main()
    assert not any(reports.iterdir())


@pytest.mark.parametrize("module", VERIFIERS, ids=lambda module: module.__name__)
def test_each_verifier_refuses_another_oracle_interpreter_before_writing(
    verifier_setup, module
):
    reports, _ = verifier_setup(module, stack=None)
    with pytest.raises(ValueError, match="Oracle interpreter refused"):
        module.main()
    assert not any(reports.iterdir())


def start_backends(module, monkeypatch, stop):
    """Replace module's run_backend; return its calls, (label, backend, python).

    The labels in stop raise, ending main(); any other returns a result.
    """
    started = []

    def run_backend(label, backend, python, *_arguments, **_options):
        started.append((label, backend, python))
        if label in stop:
            raise RuntimeError("stopped before starting a backend")
        return {"label": label}

    monkeypatch.setattr(module, "run_backend", run_backend)
    if module is verify_video_runtime:
        monkeypatch.setattr(module, "prepare_fixtures", lambda *_arguments: None)
    if module is verify_framework_runtime:
        prepared = subprocess.CompletedProcess([], 0, "", "")
        monkeypatch.setattr(module.subprocess, "run", lambda *_a, **_k: prepared)
    return started


@pytest.mark.parametrize("module", VERIFIERS, ids=lambda module: module.__name__)
def test_each_verifier_runs_its_baseline_on_a_copy_of_the_oracle_on_the_provisioned_runtime(
    verifier_setup, module, monkeypatch, provisioned
):
    reports, oracle = verifier_setup(module)
    started = start_backends(module, monkeypatch, {"baseline"})
    assert module.main() == 1
    runtime = Path(provisioned["python_stack"]["runtime"])
    [(label, backend, python)] = started
    assert (label, python) == ("baseline", runtime / "python.exe")
    assert backend.parent.parent == reports
    assert backend != oracle / "src"
    for name, dest in verify_runtime.CHAINNER_EXT.items():
        assert (backend / dest).read_bytes() == EXT[name]
    [written] = reports.glob("*/report.json")
    report = json.loads(written.read_text(encoding="utf-8"))
    assert report["error"] == "stopped before starting a backend"
    assert report["oracle"]["chainner_ext"] == {
        name: sha(data) for name, data in EXT.items()
    }
    assert report["oracle"]["python"] == str(runtime / "python.exe")


@pytest.mark.parametrize(
    ("module", "options"),
    [(verify_runtime, []), (verify_utility_runtime, ["--include-port"])],
    ids=["verify_runtime", "verify_utility_runtime"],
)
def test_the_port_side_runs_on_the_packages_runtime(
    verifier_setup, module, monkeypatch, provisioned, options
):
    verifier_setup(module)
    package = Path(sys.argv[sys.argv.index("--package") + 1])
    (package / "resources/src").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", [*sys.argv, *options])
    started = start_backends(module, monkeypatch, {"converted"})
    assert module.main() == 1
    runtime = Path(provisioned["python_stack"]["runtime"])
    assert [(label, python) for label, _, python in started] == [
        ("baseline", runtime / "python.exe"),
        ("converted", package / "python/python/python.exe"),
    ]


@pytest.mark.parametrize(
    "module",
    [verify_runtime, verify_framework_runtime, verify_video_runtime],
    ids=lambda module: module.__name__,
)
def test_backend_hosts_hide_the_gpu_and_may_not_install(tmp_path, monkeypatch, module):
    """Upstream's host pip-installs a missing server dependency at start; with the
    oracle on the shared provisioned runtime, pip must refuse instead. Each host
    hides CUDA and every Vulkan driver itself, not by inheriting conftest's."""
    monkeypatch.delenv("VK_LOADER_DRIVERS_DISABLE")
    environments = []

    def popen(command, **options):
        environments.append(options["env"])
        raise RuntimeError("stopped before starting a backend")

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    tool = tmp_path / "tools/ffmpeg.exe"
    write(tool)
    tools = [tool, tool] if module is verify_video_runtime else []
    (tmp_path / "root").mkdir()
    with pytest.raises(RuntimeError, match="stopped before starting a backend"):
        module.run_backend(
            "baseline",
            tmp_path / "src",
            tmp_path / "python.exe",
            tmp_path / "root",
            *tools,
            1,
        )
    [environment] = environments
    assert environment["PIP_REQUIRE_VIRTUALENV"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "-1"
    assert environment["VK_LOADER_DRIVERS_DISABLE"] == "*"


def start(node):
    return {"event": "node-start", "data": {"nodeId": node}}


def finish(node):
    return {"event": "node-finish", "data": {"nodeId": node, "executionTime": 0.5}}


def broadcast(node, value=None, *, length=None, static=None):
    types = {}
    if value is not None:
        types["0"] = {"value": value}
    if static is not None:
        types["1"] = {"value": static}
    return {
        "event": "node-broadcast",
        "data": {
            "nodeId": node,
            "data": dict.fromkeys(types),
            "types": types,
            "sequenceTypes": {} if length is None else {"0": {"length": length}},
        },
    }


# Range ("range", a generator) yields 2, 3 and 4; Double ("double") runs per item;
# Accumulate ("accumulate") sums the doubles. The restoration here is the only
# broadcast of Range's output 1, so losing it changes the final state. A real
# restoration re-sends static outputs already in the merged state, so its loss
# leaves the final state unchanged and this rule cannot see it; that loss is pinned
# by bench_oracle.validate_record and the execution-scheduler tests instead.
CHAIN_START = {
    "event": "chain-start",
    "data": {"nodes": ["accumulate", "double", "range"]},
}
RESTORATION = broadcast("range", static="restored")


def item(value):
    return [
        broadcast("range", value),
        start("double"),
        broadcast("double", 2 * value),
        finish("double"),
    ]


# The port, in the order it sends: Range starts once.
PORT_EVENTS = [
    CHAIN_START,
    start("range"),
    broadcast("range", length=3),
    *item(2),
    *item(3),
    *item(4),
    finish("range"),
    RESTORATION,
    start("accumulate"),
    broadcast("accumulate", 18),
    finish("accumulate"),
]
# Upstream re-runs Range before each later item: one more start and sequence
# broadcast each.
RERUN = [start("range"), broadcast("range", length=3)]
BASELINE_EVENTS = [
    CHAIN_START,
    start("range"),
    broadcast("range", length=3),
    *item(2),
    *RERUN,
    *item(3),
    *RERUN,
    *item(4),
    *PORT_EVENTS[-5:],
]


def record(events, generators=("range",)) -> dict:
    return {
        "name": "range-double",
        "generator_node_ids": list(generators),
        **verify_runtime.sse_record(events, RUN),
    }


def mismatches(baseline, converted, generators=("range",)):
    return verify_runtime.sse_event_mismatches(
        record(baseline, generators), record(converted, generators)
    )


def without(events, removed):
    position = events.index(removed)
    return events[:position] + events[position + 1 :]


def replaced(events, old, new):
    position = events.index(old)
    return [*events[:position], new, *events[position + 1 :]]


def test_identical_streams_match():
    assert mismatches(PORT_EVENTS, PORT_EVENTS) == []


def test_generator_reruns_on_the_baseline_side_match():
    assert mismatches(BASELINE_EVENTS, PORT_EVENTS) == []


def test_reruns_of_a_node_that_is_no_generator_fail():
    found = mismatches(BASELINE_EVENTS, PORT_EVENTS, generators=())
    assert any('"node-start"}: 3 baseline, 1 converted' in m for m in found), found


@pytest.mark.parametrize(
    "converted",
    [
        without(PORT_EVENTS, broadcast("range", 3)),
        without(PORT_EVENTS, broadcast("double", 4)),
        # Consult D-35: an iterated item's last value is timing-dependent upstream
        # on 3.14, so the port may end on any item the oracle broadcast.
        without(PORT_EVENTS, broadcast("range", 4)),
    ],
    ids=[
        "dropped-generator-item-broadcast",
        "non-generator-iterated-node-drops-an-item-preview",
        "dropped-last-item-ends-on-an-earlier-one",
    ],
)
def test_coalesced_previews_match(converted):
    assert mismatches(BASELINE_EVENTS, converted) == []


@pytest.mark.parametrize(
    ("converted", "expected"),
    [
        (
            replaced(PORT_EVENTS, broadcast("range", 3), broadcast("range", 5)),
            '"value": 5',
        ),
        (without(PORT_EVENTS, RESTORATION), "range: final state differs"),
        (
            [*PORT_EVENTS, start("range")],
            "range: converted generator started 2 times, expected once",
        ),
        (
            without(PORT_EVENTS, start("range")),
            "range: converted generator started 0 times, expected once",
        ),
        (
            without(PORT_EVENTS, finish("accumulate")),
            '"node-finish"}: 1 baseline, 0 converted',
        ),
        (
            [*PORT_EVENTS, start("accumulate")],
            '"node-start"}: 1 baseline, 2 converted',
        ),
        (
            replaced(
                PORT_EVENTS,
                CHAIN_START,
                {
                    "event": "chain-start",
                    "data": {"nodes": ["range", "double", "accumulate"]},
                },
            ),
            '"nodes": ["range", "double", "accumulate"]',
        ),
        (
            [*PORT_EVENTS, {"event": "execution-error", "data": {"message": "x"}}],
            "event_kinds differ",
        ),
        (
            replaced(
                PORT_EVENTS, broadcast("accumulate", 18), broadcast("accumulate", 19)
            ),
            '"value": 19',
        ),
    ],
    ids=[
        "different-generator-item-payload",
        "dropped-restoration",
        "generator-started-twice",
        "generator-never-started",
        "missing-finish",
        "extra-non-generator-start",
        "reordered-chain-start-nodes",
        "extra-execution-error",
        "different-broadcast-payload",
    ],
)
def test_differences_fail(converted, expected):
    found = mismatches(BASELINE_EVENTS, converted)
    assert any(expected in mismatch for mismatch in found), found


def test_final_state_is_merged_newest_wins_per_output_id():
    # Not the last payload: the restoration carries only output 1.
    assert record(PORT_EVENTS)["final_state"]["range"] == {
        "nodeId": "range",
        "data": {"0": None, "1": None},
        "types": {"0": {"value": 4}, "1": {"value": "restored"}},
        "sequenceTypes": {"0": {"length": 3}},
    }


@pytest.mark.parametrize("key", ["name", "event_kinds", "generator_node_ids"])
def test_other_record_keys_must_be_equal(key):
    port = record(PORT_EVENTS)
    baseline = {**port, key: ["other"] if key != "name" else "other"}
    assert verify_runtime.sse_event_mismatches(baseline, port) == [f"{key} differ"]


def dependency(display_name, pypi_name, version, find_link=None):
    """One /packages dependency as Dependency.to_dict gives it."""
    return {
        "autoUpdate": True,
        "displayName": display_name,
        "findLink": find_link,
        "pypiName": pypi_name,
        "sizeEstimate": 1024,
        "version": version,
    }


def package(package_id, *dependencies):
    """One /packages entry as Package.to_dict gives it."""
    return {
        "id": package_id,
        "name": package_id.removeprefix("chaiNNer_"),
        "description": "",
        "features": [],
        "dependencies": list(dependencies),
    }


# Upstream's findLinks: PyTorch's CUDA index (cu128; ours is cu132) and the ONNX
# Runtime CUDA 12 feed (ours is PyPI's wheel).
CU = "https://download.pytorch.org/whl/cu"
AIINFRA = (
    "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/"
    "onnxruntime-cuda-12/pypi/simple/"
)


def declared_metadata():
    """The oracle's and the port's metadata_record as their hosts give them:
    upstream's Dependency declarations against the tested stack (PINS), on two
    runtimes holding PINS (the oracle's also google-re2, which only it declares)."""
    listed = {
        name: PINS[name] for name in ("einops", "onnxruntime-gpu", "PyMatting", "torch")
    }
    ort = "ONNX Runtime (GPU)"
    oracle = {
        "packages": [
            package(
                "chaiNNer_pytorch",
                dependency("PyTorch", "torch", "2.7.0+cu128", CU + "128"),
                dependency("Einops", "einops", "0.6.1"),
            ),
            package("chaiNNer_ncnn", dependency("NCNN", "ncnn-vulkan", "2023.6.18")),
            package(
                "chaiNNer_onnx", dependency(ort, "onnxruntime-gpu", "1.17.1", AIINFRA)
            ),
        ],
        "installed-dependencies": {**listed, "google-re2": PINS["google-re2"]},
    }
    port = {
        "packages": [
            package(
                "chaiNNer_pytorch",
                dependency("PyTorch", "torch", "2.14.1+cu132", CU + "132"),
                dependency("Einops", "einops", "0.8.2"),
            ),
            package("chaiNNer_ncnn", dependency("NCNN", "ncnn", "1.0.20260526")),
            package("chaiNNer_onnx", dependency(ort, "onnxruntime-gpu", "1.30.0")),
        ],
        "installed-dependencies": {**listed, "ncnn": PINS["ncnn"]},
    }
    return tuple(
        verify_runtime.metadata_record({"nodes": {"nodes": []}, **side, "features": []})
        for side in (oracle, port)
    )


@pytest.fixture
def declared(provisioned, monkeypatch):
    """declared_metadata(), compared against the provisioned project's LOCK."""
    monkeypatch.setattr(
        verify_runtime, "PROJECT", Path(provisioned["identity"]["project"])
    )
    return declared_metadata()


def declared_dependency(record, package_id, name):
    packages = record["dependency_metadata"]["packages"]
    [entry] = [x for x in packages if x["id"] == package_id]
    [found] = [x for x in entry["dependencies"] if x["pypiName"] == name]
    return found


def test_only_the_tracked_declaration_changes_may_differ(declared):
    compared = verify_runtime.compare_metadata(*declared)
    assert compared["metadata_equal"] == dict.fromkeys(
        verify_runtime.METADATA_ENDPOINTS, True
    )
    assert compared["metadata_mismatches"] == {}
    rows = verify_runtime.DECLARATION_CHANGES
    assert compared["declaration_changes_applied"] == [
        ("findLink", "torch", rows["findLink"]["torch"][2]),
        ("pypiName", "ncnn-vulkan", rows["pypiName"]["ncnn-vulkan"][1]),
        ("findLink", "onnxruntime-gpu", rows["findLink"]["onnxruntime-gpu"][2]),
        ("oracle_only", "google-re2", rows["oracle_only"]["google-re2"]),
        ("port_only", "ncnn", rows["port_only"]["ncnn"]),
    ]


def describe_otherwise(records):
    records[1]["dependency_metadata"]["packages"][0]["description"] = "other"


def dependency_field(package_id, name, field, value, side=1):
    def change(records):
        declared_dependency(records[side], package_id, name)[field] = value

    return change


def renamed(package_id, name, new_name):
    """The port declares name as new_name, a locked name its runtime holds."""

    def change(records):
        found = declared_dependency(records[1], package_id, name)
        found.update(pypiName=new_name, version=PINS[new_name])

    return change


def listed(name, version, side):
    def change(records):
        records[side]["dependency_metadata"]["installed-dependencies"][name] = version

    return change


def hashed(key):
    def change(records):
        records[1]["metadata_sha256"][key] = "other"

    return change


def reorder(records):
    records[1]["dependency_metadata"]["packages"].reverse()


def add_dependency(records):
    records[1]["dependency_metadata"]["packages"][0]["dependencies"].append(
        dependency("safetensors", "safetensors", "0.8.0")
    )


@pytest.mark.parametrize(
    ("change", "endpoint"),
    [
        (describe_otherwise, "packages"),
        (dependency_field("chaiNNer_pytorch", "torch", "sizeEstimate", 1), "packages"),
        (dependency_field("chaiNNer_pytorch", "einops", "autoUpdate", 0), "packages"),
        (dependency_field("chaiNNer_pytorch", "einops", "pypiName", "e"), "packages"),
        (
            dependency_field("chaiNNer_pytorch", "einops", "findLink", CU + "132"),
            "packages",
        ),
        (
            dependency_field("chaiNNer_pytorch", "torch", "findLink", CU + "130"),
            "packages",
        ),
        (
            dependency_field("chaiNNer_pytorch", "torch", "findLink", CU + "126", 0),
            "packages",
        ),
        (
            dependency_field("chaiNNer_onnx", "onnxruntime-gpu", "findLink", AIINFRA),
            "packages",
        ),
        (dependency_field("chaiNNer_ncnn", "ncnn", "pypiName", "ncnn-gpu"), "packages"),
        (renamed("chaiNNer_pytorch", "einops", "PyMatting"), "packages"),
        (renamed("chaiNNer_ncnn", "ncnn", "PyMatting"), "packages"),
        (reorder, "packages"),
        (add_dependency, "packages"),
        (listed("ncnn-vulkan", "2023.6.18", 0), "installed-dependencies"),
        (listed("google-re2", PINS["google-re2"], 1), "installed-dependencies"),
        (listed("PyMatting", "1.1.15", 0), "installed-dependencies"),
        (hashed("nodes"), "nodes"),
        (hashed("features"), "features"),
    ],
    ids=[
        "package-field",
        "dependency-size",
        "dependency-auto-update",
        "unlisted-rename",
        "unlisted-find-link",
        "tracked-find-link-other-target",
        "tracked-find-link-other-source",
        "tracked-find-link-kept",
        "tracked-rename-other-target",
        "unlisted-rename-to-a-locked-name",
        "tracked-rename-to-another-locked-name",
        "package-order",
        "dependency-count",
        "unlisted-oracle-only-name",
        "tracked-oracle-only-name-listed-by-both",
        "shared-name-other-version",
        "nodes",
        "features",
    ],
)
def test_any_other_metadata_difference_fails(declared, change, endpoint):
    records = copy.deepcopy(list(declared))
    change(records)
    compared = verify_runtime.compare_metadata(*records)
    assert compared["metadata_equal"] == {
        key: key != endpoint for key in verify_runtime.METADATA_ENDPOINTS
    }
    assert list(compared["metadata_mismatches"]) == [endpoint]


@pytest.mark.parametrize(
    ("pin", "installed", "failing"),
    [
        ("0.8.3", PINS["einops"], {"packages"}),
        ("0.8.3", "0.8.3", {"packages", "installed-dependencies"}),
        (PINS["einops"], "0.8.3", {"packages", "installed-dependencies"}),
    ],
    ids=["pin-off-the-lock", "pin-and-runtime-off-the-lock", "runtime-off-the-pin"],
)
def test_a_port_pin_or_installed_version_off_the_lock_fails(
    declared, pin, installed, failing
):
    oracle, port = copy.deepcopy(list(declared))
    declared_dependency(port, "chaiNNer_pytorch", "einops")["version"] = pin
    for side in (oracle, port):  # Both runtimes agree, so only the lock tells.
        side["dependency_metadata"]["installed-dependencies"]["einops"] = installed
    compared = verify_runtime.compare_metadata(oracle, port)
    assert {key for key, equal in compared["metadata_equal"].items() if not equal} == (
        failing
    )


@pytest.mark.parametrize("version", ["9.9.9", "", "0.8.2", "2.7.0+cu999"])
def test_the_oracles_declared_versions_are_recorded_not_gated(declared, version):
    # Consult 8 R-g: the oracle's versions are upstream's declarations for another
    # stack; each run records its declarations whole, and only the port's are gated.
    oracle, port = copy.deepcopy(list(declared))
    declared_dependency(oracle, "chaiNNer_pytorch", "einops")["version"] = version
    declared_dependency(oracle, "chaiNNer_pytorch", "torch")["version"] = version
    compared = verify_runtime.compare_metadata(oracle, port)
    assert all(compared["metadata_equal"].values())
    endpoints = {
        "nodes": {"nodes": []},
        "features": [],
        **oracle["dependency_metadata"],
    }
    recorded = {
        "dependency_metadata": verify_runtime.metadata_record(endpoints)[
            "dependency_metadata"
        ]
    }
    einops = declared_dependency(recorded, "chaiNNer_pytorch", "einops")
    assert einops["version"] == version


def test_the_declaration_table_is_frozen_at_its_ruled_rows():
    # Consult 8 R-g: pypiName and findLink differ only by these rows, each citing its
    # ruling (ncnn: Consult 2 Q4; torch and torchvision: the design ruling, section
    # 1; onnxruntime-gpu: PyPI's 1.30.0). A new row is a consult.
    rows = verify_runtime.DECLARATION_CHANGES
    assert {old: row[0] for old, row in rows["pypiName"].items()} == {
        "ncnn-vulkan": "ncnn"
    }
    assert {name: row[:2] for name, row in rows["findLink"].items()} == {
        "torch": (CU + "128", CU + "132"),
        "torchvision": (CU + "128", CU + "132"),
        "onnxruntime-gpu": (AIINFRA, None),
    }
    assert "Consult 2 Q4" in rows["pypiName"]["ncnn-vulkan"][1]
    for name in ("torch", "torchvision"):
        assert "design ruling section 1" in rows["findLink"][name][2]
    assert "PyPI onnxruntime-gpu 1.30.0" in rows["findLink"]["onnxruntime-gpu"][2]
    # /installed-dependencies: exactly {google-re2: oracle-only, ncnn: port-only}.
    assert set(rows["oracle_only"]) == {"google-re2"}
    assert set(rows["port_only"]) == {"ncnn"}


def test_the_lock_is_the_projects(declared, provisioned):
    lock = Path(provisioned["identity"]["project"]) / "native/python-stack.lock.txt"
    write(lock, LOCK.replace(b"einops==0.8.2", b"einops==0.8.1"))
    compared = verify_runtime.compare_metadata(*declared)
    assert not compared["metadata_equal"]["packages"]
    assert not compared["metadata_equal"]["installed-dependencies"]


@pytest.mark.parametrize("unlisted", [False, True])
def test_verify_runtime_reports_metadata_by_the_declaration_rule(
    verifier_setup, monkeypatch, unlisted
):
    reports, _ = verifier_setup(verify_runtime)
    records = dict(zip(("baseline", "converted"), declared_metadata(), strict=True))
    if unlisted:
        declared_dependency(records["converted"], "chaiNNer_pytorch", "einops")[
            "findLink"
        ] = CU + "132"

    def run_backend(label, *_arguments):
        return {**records[label], "fixtures": [], "fixture_schema_ids": []}

    monkeypatch.setattr(verify_runtime, "run_backend", run_backend)
    monkeypatch.setattr(verify_runtime, "compare_images", lambda python, pairs: [])
    assert verify_runtime.main() == int(unlisted)
    [written] = reports.glob("*/report.json")
    report = json.loads(written.read_text(encoding="utf-8"))
    assert report["metadata_equal"]["packages"] is not unlisted
    assert len(report["declaration_changes_applied"]) == 5
    assert verify_runtime.METADATA_COMPARISON in report["fixture_limitations"]
    # Consult 11 D-17.3: the matting fixtures compare the whole RGBA result.
    [matting] = [x for x in report["fixture_limitations"] if "Alpha Matting" in x]
    assert "whole RGBA result" in matting
    assert "nondeterministic" not in matting


class AnySchema(dict):
    """A /nodes schemas stand-in for fixture_graphs: every schema has inputs 0-63,
    each a dropdown whose one option is Blend's Screen (the one option it reads)."""

    def __missing__(self, schema_id):
        option = {"option": "Screen", "value": 12}
        inputs = [{"id": i, "options": [option]} for i in range(64)]
        return {"inputs": inputs, "kind": "regularNode"}


def test_matting_fixtures_save_the_whole_rgba_result(tmp_path):
    # Consult 11 D-17.3: PyMatting 1.1.16's foreground is deterministic, so the
    # trimap-matting fixtures save the node's own RGBA output, not its alpha alone.
    graphs = dict(verify_runtime.fixture_graphs(AnySchema(), tmp_path))
    for name, schema in (
        ("chroma-matting", "chainner:image:chroma_key"),
        ("alpha-matting", "chainner:image:alpha_matting"),
    ):
        nodes = {node["id"]: node for node in graphs[name]}
        source = nodes[name + "-save"]["inputs"][0]
        assert source["type"] == "edge"
        assert source["index"] == 0
        assert nodes[source["id"]]["schemaId"] == schema
        assert "chainner:image:split_transparency" not in {
            node["schemaId"] for node in nodes.values()
        }


def attempt(events, response=None):
    """One attempt of a request: its status, response, sse_record and files."""
    return {
        "http_status": 200,
        "response": response or {"type": "success"},
        **verify_runtime.sse_record(events, RUN),
        "files": {},
    }


def compared_run(events, response, metadata, repeat=None):
    """A framework run of one fixture; its second attempt sends repeat's events
    (events' by default)."""
    return {
        **metadata,
        "fixtures": [
            {
                "name": "range-double",
                "generator_node_ids": ["range"],
                "contract": attempt(events, response),
                "repeat_contract": attempt(
                    events if repeat is None else repeat, response
                ),
                "output_directory": str(RUN),
            }
        ],
    }


# Range's preview of a value it never yields, sent before its last item's preview so
# the final state is unchanged: only the sub-multiset rule can see it.
LAST_ITEM = PORT_EVENTS.index(broadcast("range", 4))
FOREIGN = [*PORT_EVENTS[:LAST_ITEM], broadcast("range", 7), *PORT_EVENTS[LAST_ITEM:]]


def test_framework_compare_runs_applies_the_rule_per_fixture(monkeypatch, declared):
    # No output files: the image comparison has nothing to decode.
    monkeypatch.setattr(
        verify_framework_runtime, "compare_images", lambda python, pairs: []
    )
    success = {"type": "success"}

    def compare(converted, response=success):
        return verify_framework_runtime.compare_runs(
            compared_run(BASELINE_EVENTS, success, declared[0]),
            compared_run(converted, response, declared[1]),
            RUN,
        )

    for matching in (PORT_EVENTS, without(PORT_EVENTS, broadcast("range", 3))):
        matched = compare(matching)
        assert matched["success"]
        assert matched["sse_event_mismatches"] == {}
    # Consult D-35: ending on an earlier item the oracle broadcast passes.
    assert compare(without(PORT_EVENTS, broadcast("range", 4)))["success"]
    lost = compare(without(PORT_EVENTS, RESTORATION))
    assert not lost["success"]
    assert lost["contracts_equal"] == {"range-double": False}
    assert any(
        mismatch.startswith("range: final state differs")
        for mismatch in lost["sse_event_mismatches"]["range-double"]
    )
    failed = compare(PORT_EVENTS, {"type": "error"})
    assert not failed["success"]
    assert failed["sse_event_mismatches"] == {}


def test_framework_compare_runs_compares_metadata_by_the_declaration_rule(
    monkeypatch, declared
):
    monkeypatch.setattr(
        verify_framework_runtime, "compare_images", lambda python, pairs: []
    )
    success = {"type": "success"}
    oracle, port = copy.deepcopy(list(declared))

    def compare():
        return verify_framework_runtime.compare_runs(
            compared_run(PORT_EVENTS, success, oracle),
            compared_run(PORT_EVENTS, success, port),
            RUN,
        )

    matched = compare()
    assert matched["success"]
    assert len(matched["declaration_changes_applied"]) == 5
    declared_dependency(port, "chaiNNer_pytorch", "einops")["findLink"] = CU + "132"
    unlisted = compare()
    assert not unlisted["success"]
    assert list(unlisted["metadata_mismatches"]) == ["packages"]


@pytest.mark.parametrize(
    ("second", "port", "differing"),
    [
        (PORT_EVENTS, False, []),
        (without(PORT_EVENTS, broadcast("range", 3)), True, []),
        (without(PORT_EVENTS, broadcast("range", 3)), False, ["events differ"]),
        (without(PORT_EVENTS, finish("double")), True, ["events differ"]),
        ([*PORT_EVENTS, start("double")], True, ["events differ"]),
        (
            without(PORT_EVENTS, broadcast("range", 4)),
            True,
            [
                "range: the first attempt's final types/0 "
                + json.dumps({"value": 4})
                + " is not among the other attempt's broadcasts"
            ],
        ),
    ],
    ids=[
        "identical",
        "port-coalesced-preview",
        "oracle-coalesced-preview",
        "port-missing-finish",
        "port-extra-start",
        "port-lost-final-preview",
    ],
)
def test_a_repeat_is_exact_but_for_the_ports_coalesced_previews(
    second, port, differing
):
    assert (
        verify_runtime.repeat_mismatches(
            attempt(PORT_EVENTS), attempt(second), port=port
        )
        == differing
    )


@pytest.mark.parametrize("key", ["http_status", "response", "files"])
def test_a_repeat_with_other_results_differs_on_either_side(key):
    second = {**attempt(PORT_EVENTS), key: "other"}
    for port in (False, True):
        assert verify_runtime.repeat_mismatches(
            attempt(PORT_EVENTS), second, port=port
        ) == [f"{key} differ"]


@pytest.mark.parametrize(
    ("first", "second", "equal"),
    [
        (PORT_EVENTS, without(PORT_EVENTS, broadcast("range", 3)), True),
        (without(PORT_EVENTS, broadcast("range", 3)), PORT_EVENTS, True),
        (FOREIGN, PORT_EVENTS, False),
        (PORT_EVENTS, FOREIGN, False),
    ],
    ids=[
        "repeat-coalesced",
        "first-coalesced",
        "foreign-preview-first-attempt",
        "foreign-preview-repeat",
    ],
)
def test_framework_compare_runs_checks_both_port_attempts(
    monkeypatch, declared, first, second, equal
):
    monkeypatch.setattr(
        verify_framework_runtime, "compare_images", lambda python, pairs: []
    )
    success = {"type": "success"}
    compared = verify_framework_runtime.compare_runs(
        compared_run(BASELINE_EVENTS, success, declared[0]),
        compared_run(first, success, declared[1], repeat=second),
        RUN,
    )
    assert compared["success"] is equal
    if not equal:
        [mismatch] = compared["sse_event_mismatches"]["range-double"]
        assert mismatch.startswith(
            "repeat: range: broadcast" if second is FOREIGN else "range: broadcast"
        )


def range_previews(multiset):
    """Range's item previews in one attempt's broadcast multiset: value, times sent."""
    return {
        json.loads(key)["types"]["0"]["value"]: count
        for key, count in multiset["range"].items()
        if "0" in json.loads(key)["types"]
    }


def test_each_attempts_broadcast_multiset_is_reported(monkeypatch, declared):
    # Consult 8 R-h (4): the report shows what the port's repeat rule left out, as
    # T131909Z's evidence was: Range's previews {2, 4}, then {2, 3, 4}.
    monkeypatch.setattr(
        verify_framework_runtime, "compare_images", lambda python, pairs: []
    )
    success = {"type": "success"}
    coalesced = without(PORT_EVENTS, broadcast("range", 3))
    compared = verify_framework_runtime.compare_runs(
        compared_run(BASELINE_EVENTS, success, declared[0]),
        compared_run(coalesced, success, declared[1], repeat=PORT_EVENTS),
        RUN,
    )
    assert compared["success"]
    found = compared["broadcast_multisets"]["range-double"]
    assert [range_previews(attempt) for attempt in found["port"]] == [
        {2: 1, 4: 1},
        {2: 1, 3: 1, 4: 1},
    ]
    assert [range_previews(attempt) for attempt in found["oracle"]] == [
        {2: 1, 3: 1, 4: 1}
    ] * 2
    # The sequence broadcast (broadcast("range", length=3) but its nodeId): once per
    # Range start, three times on the oracle.
    sequence = json.dumps(
        {"data": {}, "types": {}, "sequenceTypes": {"0": {"length": 3}}},
        sort_keys=True,
    )
    assert [attempt["range"][sequence] for attempt in found["oracle"]] == [3, 3]
    assert [attempt["range"][sequence] for attempt in found["port"]] == [1, 1]
    assert found["port"][0]["accumulate"] == found["oracle"][0]["accumulate"]


def test_a_foreign_payload_in_the_second_attempt_alone_fails(monkeypatch, declared):
    # Consult 8 R-h (5): the within-side rule leaves the repeat's previews out, so
    # only the cross-side rule over BOTH attempts can see a foreign one there.
    within = verify_runtime.repeat_mismatches(
        attempt(PORT_EVENTS), attempt(FOREIGN), port=True
    )
    assert within == []
    monkeypatch.setattr(
        verify_framework_runtime, "compare_images", lambda python, pairs: []
    )
    success = {"type": "success"}
    compared = verify_framework_runtime.compare_runs(
        compared_run(BASELINE_EVENTS, success, declared[0]),
        compared_run(PORT_EVENTS, success, declared[1], repeat=FOREIGN),
        RUN,
    )
    assert not compared["success"]
    assert compared["contracts_equal"] == {"range-double": False}
    repeat = compared["broadcast_multisets"]["range-double"]["port"][1]
    assert range_previews(repeat) == {2: 1, 3: 1, 4: 1, 7: 1}


def test_the_repeat_rules_name_what_each_side_compares():
    assert verify_runtime.REPEAT_RULES == {
        False: "exact but iterated items' final values (broadcast multisets)",
        True: "exact but broadcast multiplicities and iterated items' final values",
    }


def test_compare_images_counts_differing_components_and_pixels(tmp_path):
    colour = np.arange(4 * 3 * 3, dtype=np.uint8).reshape(4, 3, 3)
    changed = colour.copy()
    changed[0, 0, 0] += 5
    changed[0, 0, 1] += 2
    changed[2, 1, 2] += 1
    grey = np.arange(12, dtype=np.uint8).reshape(4, 3)
    grey_changed = grey.copy()
    grey_changed[1, 1] += 3
    images = {
        "colour": (colour, changed),
        "grey": (grey, grey_changed),
        "same": (colour, colour),
        "shape": (colour, colour[:2]),
    }
    pairs = []
    for name, (baseline, converted) in images.items():
        for side, pixels in (("baseline", baseline), ("converted", converted)):
            assert cv2.imwrite(str(tmp_path / f"{name}-{side}.png"), pixels)
        pairs.append(
            {
                "name": name,
                "baseline": str(tmp_path / f"{name}-baseline.png"),
                "converted": str(tmp_path / f"{name}-converted.png"),
            }
        )
    rows = verify_runtime.compare_images(Path(sys.executable), pairs)
    counts = {
        row["name"]: (
            row["pass"],
            row["max_channel_difference"],
            row["differing_components"],
            row["differing_pixels"],
        )
        for row in rows
    }
    assert counts == {
        # Three channel values in two pixels.
        "colour": (False, 5, 3, 2),
        # One channel: pixels and components are the same count.
        "grey": (False, 3, 1, 1),
        "same": (True, 0, 0, 0),
        "shape": (False, None, None, None),
    }


# Consult D-35: iterated items' broadcasts arrive in timing-dependent order upstream
# on 3.14 (CPython's _chain_future), so their last value is not defined.


def item_type(value):
    """Range's broadcast of one item's type, as upstream sends it."""
    return {
        "event": "node-broadcast",
        "data": {
            "nodeId": "range",
            "data": {"0": None},
            "types": {"0": {"type": "numeric-literal", "value": value}},
            "sequenceTypes": {},
        },
    }


SEQUENCE_OF_3 = {
    "event": "node-broadcast",
    "data": {
        "nodeId": "range",
        "data": {},
        "types": {},
        "sequenceTypes": {
            "0": {"type": "named", "name": "Sequence", "fields": {"length": 3}}
        },
    },
}
CLEARED = {
    "event": "node-broadcast",
    "data": {"nodeId": "range", "data": {}, "types": {}, "sequenceTypes": {}},
}
PRODUCT = {
    "event": "node-broadcast",
    "data": {
        "nodeId": "accumulate",
        "data": {"0": None},
        "types": {"0": {"type": "numeric-literal", "value": 60}},
        "sequenceTypes": {},
    },
}


def recorded(*order):
    """io-execution T204546Z's oracle range-prod-0-1, its range broadcasts in the
    given item order: the attempts sent 3, 5, 4 and 3, 4, 5."""
    return [
        {"event": "chain-start", "data": {"nodes": ["accumulate", "range"]}},
        start("range"),
        start("accumulate"),
        SEQUENCE_OF_3,
        *(item_type(value) for value in order),
        finish("range"),
        finish("accumulate"),
        SEQUENCE_OF_3,
        CLEARED,
        PRODUCT,
    ]


def typed(value):
    return json.dumps({"type": "numeric-literal", "value": value}, sort_keys=True)


def fixture_record(events) -> dict[str, Any]:
    """One attempt as sse_event_mismatches takes a side: Range's fixture."""
    return {**attempt(events), "name": "p", "generator_node_ids": ["range"]}


def test_the_recorded_oracle_pair_passes_and_is_flagged():
    first, second = attempt(recorded(3, 5, 4)), attempt(recorded(3, 4, 5))
    # Last-wins, the attempts end on 4 and 5; their broadcast multisets are equal.
    assert first["final_state"] != second["final_state"]
    assert verify_runtime.repeat_mismatches(first, second, port=False) == []
    assert verify_runtime.repeat_final_value_differences(first, second) == {
        "range": {"types/0": [typed(4), typed(5)]}
    }
    assert verify_runtime.repeat_final_value_differences(first, first) == {}
    # Either order is a valid oracle for the port, which ends on the last item.
    port = fixture_record(recorded(3, 4, 5))
    for order in ((3, 5, 4), (3, 4, 5)):
        oracle = fixture_record(recorded(*order))
        assert verify_runtime.sse_event_mismatches(oracle, port) == []


def test_the_oracles_self_disagreement_reaches_the_report(monkeypatch, declared):
    monkeypatch.setattr(
        verify_framework_runtime, "compare_images", lambda python, pairs: []
    )
    first, second = attempt(recorded(3, 5, 4)), attempt(recorded(3, 4, 5))
    flagged = verify_runtime.repeat_final_value_differences(first, second)
    success = {"type": "success"}
    oracle = compared_run(BASELINE_EVENTS, success, declared[0])
    oracle["fixtures"][0]["repeat_final_values_differ"] = flagged
    compared = verify_framework_runtime.compare_runs(
        oracle, compared_run(PORT_EVENTS, success, declared[1]), RUN
    )
    assert compared["success"]  # Flagged, not failed.
    assert compared["oracle_repeat_final_values_differ"] == {"range-double": flagged}
    assert verify_runtime.oracle_repeat_flags(oracle) == {"range-double": flagged}
    assert verify_runtime.oracle_repeat_flags({"fixtures": []}) == {}


def test_an_iterated_items_multiset_difference_fails():
    # Within the oracle: another multiset is another event set.
    first, second = attempt(recorded(3, 4, 5)), attempt(recorded(3, 4, 4))
    assert "events differ" in verify_runtime.repeat_mismatches(
        first, second, port=False
    )
    # Across: a value the oracle never broadcast.
    base = fixture_record(recorded(3, 4, 5))
    port = fixture_record(recorded(3, 4, 6))
    found = verify_runtime.sse_event_mismatches(base, port)
    assert any("sent 1 times by converted, 0 by baseline" in m for m in found), found


def test_a_port_final_value_outside_the_oracles_multiset_fails():
    oracle = fixture_record(recorded(3, 5, 4))
    port = copy.deepcopy(oracle)
    # Every broadcast the oracle's, but the port ends on a value none of them carried.
    port["final_state"]["range"]["types"]["0"] = {"type": "numeric-literal", "value": 7}
    sent = sorted([typed(3), typed(4), typed(5)])
    across = (
        f"range: final types/0 {typed(7)} converted is not among the baseline's "
        f"broadcasts {sent}"
    )
    assert verify_runtime.sse_event_mismatches(oracle, port) == [across]
    # Within one side, likewise against the other attempt's broadcasts.
    within = (
        f"range: the second attempt's final types/0 {typed(7)} is not among the "
        "other attempt's broadcasts"
    )
    assert verify_runtime.repeat_mismatches(oracle, port, port=True) == [within]


@pytest.mark.parametrize(
    ("node", "field", "output", "value"),
    [
        ("accumulate", "types", "0", {"type": "numeric-literal", "value": 61}),
        ("range", "sequenceTypes", "0", {"type": "named", "name": "Sequence"}),
    ],
    ids=["a-node-broadcast-once", "a-field-of-the-iterated-node-sent-once"],
)
def test_other_final_state_drift_fails(node, field, output, value):
    oracle = fixture_record(recorded(3, 5, 4))
    drifted = copy.deepcopy(oracle)
    drifted["final_state"][node][field][output] = value
    found = verify_runtime.sse_event_mismatches(oracle, drifted)
    assert [m.split(" (")[0] for m in found] == [f"{node}: final state differs"]
    for port in (False, True):
        assert verify_runtime.repeat_mismatches(oracle, drifted, port=port) == [
            "final_state differ"
        ]


def test_iterated_fields_are_those_sent_with_more_than_one_value():
    fields = verify_runtime.iterated_fields(attempt(recorded(3, 5, 4)))
    assert fields == {"range": {"types/0"}}
    assert verify_runtime.iterated_fields(attempt(recorded(3, 3, 3))) == {}
