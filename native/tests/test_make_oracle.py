"""make_oracle on a fake installed app and backend/src.

Fresh copy, patch, chaiNNer-C's chainner_ext, record and confinement.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import make_oracle
import pytest
from verify_runtime import digest, oracle_source

VERSION = "0.25.1-nightly.2025-10-21"
LF = b"a = 1\nb = 2\nc = 3\n"
CRLF = b"a = 1\r\nb = 2\r\nc = 3\r\n"
SOURCE = {"run.py": b"import nodes\n", "nodes/lf.py": LF, "nodes/crlf.py": CRLF}
FILES = {name: hashlib.sha256(data).hexdigest() for name, data in SOURCE.items()}
# chaiNNer-C's chainner_ext in backend/src, and where the oracle holds it: the
# package beside upstream's, chainner_native.dll beside the pyd.
EXT = {
    "chainner_ext/__init__.py": b"from . import chainner_ext\n",
    "chainner_ext/__init__.pyi": b"def f() -> None: ...\n",
    "chainner_ext/chainner_ext.pyd": b"MZ pyd",
    "nodes/impl/chainner_native.dll": b"MZ dll",
}
EXT_HASHES = {name: hashlib.sha256(data).hexdigest() for name, data in EXT.items()}
ORACLE_EXT = {
    "chainner_ext/__init__.py": EXT["chainner_ext/__init__.py"],
    "chainner_ext/__init__.pyi": EXT["chainner_ext/__init__.pyi"],
    "chainner_ext/chainner_ext.pyd": EXT["chainner_ext/chainner_ext.pyd"],
    "chainner_ext/chainner_native.dll": EXT["nodes/impl/chainner_native.dll"],
}
PATCH = (
    b"--- a/nodes/lf.py\n+++ b/nodes/lf.py\n@@ -1,3 +1,3 @@\n"
    b" a = 1\n-b = 2\n+b = 20\n c = 3\n"
    b"--- a/nodes/crlf.py\n+++ b/nodes/crlf.py\n@@ -1,3 +1,3 @@\n"
    b" a = 1\r\n-b = 2\r\n+b = 20\r\n c = 3\r\n"
)


def write(root, relative, data=b"x"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def tree(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def app(tmp_path):
    app = tmp_path / "app-0.25.1-nightly2025-10-21"
    write(app, "resources/app/package.json", json.dumps({"version": VERSION}).encode())
    for name, data in SOURCE.items():
        write(app / "resources/src", name, data)
    write(app, "resources/src/__pycache__/run.cpython-311.pyc")
    write(app, "resources/src/nodes/__pycache__/lf.cpython-311.pyc")
    return app


@pytest.fixture
def backend(tmp_path):
    """backend/src with the four chainner_ext files among others never copied."""
    backend = tmp_path / "backend/src"
    for name, data in EXT.items():
        write(backend, name, data)
    write(backend, "run.py", b"import chaiNNer_C\n")
    write(backend, "nodes/impl/_chainner_graph.pyd", b"MZ graph")
    return backend


def diff(tmp_path, data):
    return write(tmp_path, "compat.diff", data)


def test_empty_diff_gives_a_fresh_copy_without_bytecode(app, backend, tmp_path):
    oracle = tmp_path / "build/oracle"
    write(oracle, "src/stale.py")
    installed, built = tree(app), tree(backend)
    record = make_oracle.make_oracle(
        app, backend, diff(tmp_path, b""), oracle, check=False
    )
    assert tree(oracle / "src") == {**SOURCE, **ORACLE_EXT}
    assert tree(app) == installed
    assert tree(backend) == built
    assert record["compat_diff_sha256"] == hashlib.sha256(b"").hexdigest()


def test_diff_is_applied_byte_for_byte_and_recorded(app, backend, tmp_path):
    oracle = tmp_path / "build/oracle"
    installed = tree(app)
    record = make_oracle.make_oracle(
        app, backend, diff(tmp_path, PATCH), oracle, check=False
    )
    # Each patched file keeps its own line endings.
    assert tree(oracle / "src") == {
        **SOURCE,
        "nodes/lf.py": b"a = 1\nb = 20\nc = 3\n",
        "nodes/crlf.py": b"a = 1\r\nb = 20\r\nc = 3\r\n",
        **ORACLE_EXT,
    }
    assert tree(app) == installed
    assert set(tree(oracle)) == {
        "oracle.json",
        *(f"src/{name}" for name in (*SOURCE, *ORACLE_EXT)),
    }
    assert json.loads((oracle / "oracle.json").read_text(encoding="utf-8")) == record
    assert record == {
        "app_version": VERSION,
        "app_directory": app.name,
        "source": str(app / "resources/src"),
        "source_files": FILES,
        "source_sha256": digest(FILES),
        "compat_diff_sha256": hashlib.sha256(PATCH).hexdigest(),
        "chainner_ext": EXT_HASHES,
        "utc": record["utc"],
    }
    assert datetime.fromisoformat(record["utc"]).utcoffset() == timedelta(0)


def test_the_record_passes_the_verifiers_guard_for_a_package_of_the_same_build(
    app, backend, tmp_path
):
    oracle = tmp_path / "build/oracle"
    record = make_oracle.make_oracle(
        app, backend, diff(tmp_path, PATCH), oracle, check=False
    )
    # The provisioned runtime the package was copied from, and its project's lock.
    lock = write(tmp_path, "project/native/python-stack.lock.txt", b"# lock\n")
    python = write(tmp_path, "runtime/cpython-3.14.8/python.exe", b"MZ python")
    manifest = {
        "identity": {"project": str(lock.parents[1])},
        "backend": {
            "resources/src/" + name: {"sha256": sha256, "bytes": len(EXT[name])}
            for name, sha256 in EXT_HASHES.items()
        },
        "python_stack": {
            "runtime": str(python.parent),
            "lock_sha256": hashlib.sha256(b"# lock\n").hexdigest(),
        },
    }
    source, interpreter, returned = oracle_source(manifest, oracle)
    assert (source, interpreter) == (oracle / "src", python)
    del record["source_files"]
    assert returned == {**record, "python": str(python)}


@pytest.mark.parametrize(
    "patch", [PATCH.replace(b"-b = 2\n", b"-b = 3\n"), b"not a patch\n"]
)
def test_diff_that_does_not_apply_fails_and_leaves_no_record(
    app, backend, tmp_path, patch
):
    oracle = tmp_path / "build/oracle"
    make_oracle.make_oracle(app, backend, diff(tmp_path, b""), oracle, check=False)
    assert (oracle / "oracle.json").is_file()
    with pytest.raises(RuntimeError, match="does not apply"):
        make_oracle.make_oracle(
            app, backend, diff(tmp_path, patch), oracle, check=False
        )
    assert not (oracle / "oracle.json").exists()
    # git apply is all or nothing: the hunk that matches is not applied either.
    assert tree(oracle / "src") == {**SOURCE, **ORACLE_EXT}


def test_check_verifies_the_sources_and_diff_and_writes_nothing(app, backend, tmp_path):
    oracle = tmp_path / "build/oracle"
    installed = tree(app)
    record = make_oracle.make_oracle(
        app, backend, diff(tmp_path, PATCH), oracle, check=True
    )
    assert record["source_files"] == FILES
    assert record["chainner_ext"] == EXT_HASHES
    with pytest.raises(RuntimeError, match="does not apply"):
        make_oracle.make_oracle(app, backend, diff(tmp_path, b"x\n"), oracle, True)
    with pytest.raises(ValueError, match="source missing"):
        make_oracle.make_oracle(
            tmp_path / "absent", backend, diff(tmp_path, b""), oracle, True
        )
    assert not oracle.exists()
    assert tree(app) == installed


@pytest.mark.parametrize("check", [True, False])
@pytest.mark.parametrize("name", list(EXT))
def test_a_missing_chainner_ext_source_fails_and_writes_nothing(
    app, backend, tmp_path, name, check
):
    oracle = tmp_path / "build/oracle"
    make_oracle.make_oracle(app, backend, diff(tmp_path, b""), oracle, check=False)
    before = tree(oracle)
    (backend / name).unlink()
    with pytest.raises(ValueError, match="chainner_ext missing") as error:
        make_oracle.make_oracle(app, backend, diff(tmp_path, b""), oracle, check)
    assert str(backend / name) in str(error.value)
    # The previous oracle and the record describing it are left whole.
    assert tree(oracle) == before


def test_refuses_a_diff_that_writes_outside_the_oracle(app, backend, tmp_path):
    oracle = tmp_path / "build/oracle"
    escape = b"--- /dev/null\n+++ b/../../../escape.py\n@@ -0,0 +1 @@\n+x = 1\n"
    with pytest.raises(RuntimeError, match="invalid path"):
        make_oracle.make_oracle(
            app, backend, diff(tmp_path, escape), oracle, check=False
        )
    assert not list(tmp_path.rglob("escape.py"))
    assert not (oracle / "oracle.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows directory junctions")
def test_refuses_an_oracle_that_is_not_a_real_absolute_path(app, backend, tmp_path):
    import _winapi

    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "build/oracle"
    linked.parent.mkdir()
    _winapi.CreateJunction(str(outside), str(linked))
    try:
        for oracle in (linked, Path("build/oracle")):
            with pytest.raises(ValueError, match="not a real absolute path"):
                make_oracle.make_oracle(
                    app, backend, diff(tmp_path, b""), oracle, check=False
                )
    finally:
        linked.rmdir()
    assert not any(outside.iterdir())


@pytest.mark.skipif(os.name != "nt", reason="Windows directory junctions")
def test_refuses_to_delete_through_a_junction_in_the_old_tree(app, backend, tmp_path):
    import _winapi

    outside = tmp_path / "outside"
    write(outside, "keep.py")
    oracle = tmp_path / "build/oracle"
    junction = oracle / "src/nodes"
    junction.parent.mkdir(parents=True)
    _winapi.CreateJunction(str(outside), str(junction))
    try:
        with pytest.raises(ValueError, match="Refusing reparse point"):
            make_oracle.make_oracle(
                app, backend, diff(tmp_path, b""), oracle, check=False
            )
    finally:
        junction.rmdir()
    assert tree(outside) == {"keep.py": b"x"}
