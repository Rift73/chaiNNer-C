"""Interrupted refresh preserves an existing profile without accepting partial copies."""

import pytest
from package_manifest import MANIFEST, validate_recovery


def fixture(root, names):
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())


def test_interrupted_refresh_keeps_existing_profile(tmp_path):
    baseline = {"chaiNNer.exe", "python/python/python.exe"}
    backend = {"resources/src/run.py"}
    profile = {"settings.json", "Cache/Cache_Data/data_0"}
    fixture(tmp_path, baseline | backend | profile | {"portable", MANIFEST})
    before = {p: (tmp_path / p).read_bytes() for p in profile}
    assert validate_recovery(tmp_path, baseline, baseline, backend)
    assert before == {p: (tmp_path / p).read_bytes() for p in profile}


@pytest.mark.parametrize("missing", ("portable", "chaiNNer.exe"))
def test_inventory_alone_cannot_authorize_an_incomplete_refresh(tmp_path, missing):
    baseline = {"chaiNNer.exe", "python/python/python.exe"}
    fixture(tmp_path, (baseline | {"portable", "settings.json"}) - {missing})
    with pytest.raises(ValueError, match="missing a copied runtime or portable marker"):
        validate_recovery(tmp_path, baseline, baseline, set())


def test_partial_recorded_inventory_is_rejected(tmp_path):
    baseline = {"chaiNNer.exe", "python/python/python.exe"}
    fixture(tmp_path, baseline | {"portable", "settings.json"})
    with pytest.raises(ValueError, match="incomplete baseline inventory"):
        validate_recovery(tmp_path, baseline, {"chaiNNer.exe"}, set())


def test_first_copy_cannot_adopt_unrelated_profile_files(tmp_path):
    fixture(tmp_path, {"chaiNNer.exe", "settings.json"})
    with pytest.raises(ValueError, match="non-generated paths"):
        validate_recovery(tmp_path, {"chaiNNer.exe"}, set(), set())


def test_first_copy_resumes_only_generated_paths(tmp_path):
    backend = "resources/src/run.py"
    fixture(tmp_path, {"chaiNNer.exe", MANIFEST, backend, backend + ".tmp"})
    generated = {backend, backend + ".tmp"}
    assert not validate_recovery(
        tmp_path, {"chaiNNer.exe", "python/python/python.exe"}, set(), generated
    )


def test_python_bytecode_cache_churn_does_not_recopy_runtime(tmp_path):
    baseline = {
        "chaiNNer.exe",
        "python/python/python.exe",
        "python/python/Lib/new_module.py",
    }
    old_cache = "python/python/Lib/__pycache__/old_module.cpython-311.pyc"
    new_cache = "python/python/Lib/__pycache__/new_module.cpython-311.pyc"
    fixture(tmp_path, baseline | {"portable", "settings.json"})
    assert validate_recovery(
        tmp_path, baseline | {new_cache}, baseline | {old_cache}, set()
    )
    assert not (tmp_path / new_cache).exists()
    with pytest.raises(ValueError, match="incomplete baseline inventory"):
        validate_recovery(
            tmp_path,
            baseline | {"python/python/Lib/changed_source.py"},
            baseline,
            set(),
        )


def test_first_copy_resumes_past_an_interrupted_compile(tmp_path):
    # Consult 10 D-9: the build compiles the runtime's Lib after the copy; an
    # interruption leaves some .pyc, which the rerun compiles again.
    source = "python/python/Lib/mod.py"
    pyc = "python/python/Lib/__pycache__/mod.cpython-314.pyc"
    fixture(tmp_path, {"chaiNNer.exe", source, pyc, MANIFEST})
    assert not validate_recovery(tmp_path, {"chaiNNer.exe", source}, set(), set())


def test_first_copy_still_refuses_bytecode_outside_the_runtime(tmp_path):
    pyc = "resources/app/__pycache__/x.cpython-314.pyc"
    fixture(tmp_path, {"chaiNNer.exe", pyc})
    with pytest.raises(ValueError, match="non-generated paths"):
        validate_recovery(tmp_path, {"chaiNNer.exe"}, set(), set())
