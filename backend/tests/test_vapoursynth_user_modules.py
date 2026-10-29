"""The VapourSynth user modules of user_modules.toml: which ones pip has to install at a
start (a version spec per module, as pip reads it), their install with chaiNNer's pip,
and their names in the catalog's key and the scan."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pytest

from dependencies import store
from nodes.impl.vapoursynth import catalog, scan, user_modules


def test_missing_user_modules_follow_the_version_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        store,
        "installed_packages",
        {"vapoursynth": "81", "vs-colorfix": "2.3.0", "vsjetpack": "2.2.6"},
    )
    monkeypatch.setattr(user_modules, "installed_packages", store.installed_packages)
    (tmp_path / "vapoursynth").mkdir()
    (tmp_path / "vapoursynth" / "user_modules.toml").write_text(
        "modules = [\n"
        '  "vs_colorfix", "VS-Colorfix==2.3.0", "vs-colorfix>=2,<3", "vs-colorfix>=3",\n'
        '  "vsjetpack[deinterlace]~=2.2", "chainner-test-new",\n'
        '  "chainner-test-pinned[extra]>=1", "not a requirement",\n'
        # For another system: pip installs nothing, so it is never missing.
        "  \"chainner-test-elsewhere; sys_platform == 'no-such-system'\",\n"
        "]\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        missing = user_modules.missing_user_modules(str(tmp_path))
    assert missing == [
        "vs-colorfix>=3",
        "chainner-test-new",
        "chainner-test-pinned[extra]>=1",
    ]
    assert "'not a requirement'" in caplog.text

    # Without a storage folder (a start without --storage-dir) there is no file; without
    # VapourSynth nothing is installed: its scan could not list them.
    assert user_modules.missing_user_modules(None) == []
    del store.installed_packages["vapoursynth"]
    assert user_modules.missing_user_modules(str(tmp_path)) == []


def test_user_module_install_uses_chainner_pip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    assert user_modules.pip_command()[-2:] == ["-m", "chainner_pip"]
    assert user_modules.PIP_TIMEOUT == 900
    calls = tmp_path / "calls.txt"
    fake_pip = tmp_path / "fake_pip.py"
    fake_pip.write_text(
        "import sys, time\n"
        "with open(sys.argv[1], 'a', encoding='utf-8') as log:\n"
        "    log.write(' '.join(sys.argv[2:]) + '\\n')\n"
        "if 'chainner-test-hangs' in sys.argv:\n"
        "    time.sleep(60)\n"
        "sys.exit(1 if 'chainner-test-fails' in sys.argv else 0)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        user_modules,
        "pip_command",
        lambda: [sys.executable, str(fake_pip), str(calls)],
    )
    monkeypatch.setattr(user_modules, "PIP_TIMEOUT", 2)
    # A failed install, or one that takes longer than PIP_TIMEOUT, skips only that
    # module. A version spec goes to pip as written.
    start = time.monotonic()
    with caplog.at_level(logging.INFO):
        for name in ("chainner-test-hangs", "chainner-test-fails", "chainner-new>=2"):
            user_modules.install_user_module(name)
    assert 2 <= time.monotonic() - start < 30
    runs = [line.split() for line in calls.read_text(encoding="utf-8").splitlines()]
    # (Without the hanging pip's own line, which a slow start may not reach.)
    assert [run[:2] for run in runs if run[1] != "chainner-test-hangs"] == [
        ["install", "chainner-test-fails"],
        ["install", "chainner-new>=2"],
    ]
    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "chainner-test-hangs took longer than 2 s" in messages
    assert "Could not install the VapourSynth user module chainner-test-fails" in (
        messages
    )
    assert "Installed the VapourSynth user module chainner-new>=2." in messages


def test_a_version_spec_is_not_part_of_the_module_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pinned modules used to install but never list: their import name and version
    # were looked up with the spec in the name.
    assert scan.project_name("vs-colorfix>=2,<3") == "vs-colorfix"
    assert scan.project_name("vs_colorfix[extra] == 2.3") == "vs_colorfix"
    assert scan.project_name("vs-colorfix @ https://example.org/x.whl") == (
        "vs-colorfix"
    )
    distributions = {"vs_colorfix": ["vs-colorfix"], "pytest": ["pytest"]}
    assert scan.import_names("vs-colorfix>=2", distributions) == ["vs_colorfix"]
    assert scan.import_names("other-module<1", distributions) == ["other_module"]

    plugins = tmp_path / "site" / "vapoursynth" / "plugins"
    plugins.mkdir(parents=True)
    monkeypatch.setattr(catalog, "vspipe", lambda: plugins.parent / "vspipe.exe")
    key = catalog.catalog_key(tmp_path, ["pytest>=1"])
    assert key["user_modules"] == {"pytest>=1": pytest.__version__}
