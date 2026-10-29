"""The AMD kit (native/tools/amd, native/AMD.md) on what a bare rented image lacks: no
system Python for the scripts among ROCm's tools, no C compiler for the launchers Triton
builds, and a test run that may outlast its time limit."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1] / "tools" / "amd"


def gpu_proof():
    spec = importlib.util.spec_from_file_location("kit_gpu_proof", KIT / "gpu_proof.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_python_tool_runs_with_this_python(tmp_path: Path):
    # amd-smi in ROCm's wheel starts with "#!/usr/bin/env python3", which a bare
    # ubuntu image cannot run; an interpreter no system has stands for it here.
    script = tmp_path / "amd-smi"
    script.write_text(
        "#!/usr/bin/env python3-not-installed\nimport sys\n"
        "print(sys.executable, *sys.argv[1:])\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    assert gpu_proof().tool(script, "list", "--json") == f"{sys.executable} list --json"


def kit_packages(manager: str) -> list[str]:
    """The kit's own system packages for apt or dnf, as setup.sh lists them."""
    text = (KIT / "setup.sh").read_text(encoding="utf-8")
    match = re.search(rf"\n    {manager}\)\n.*?kit_packages=\(([^)]*)\)", text, re.S)
    assert match is not None
    return match[1].split()


def test_the_kit_installs_what_triton_builds_its_launchers_with():
    # The DUAL ops test launches Triton's own kernels, whose launchers Triton compiles
    # with the system's C compiler against the C library's headers.
    assert {"gcc", "libc6-dev"} <= set(kit_packages("apt"))
    assert {"gcc", "glibc-devel"} <= set(kit_packages("dnf"))


def test_a_test_run_at_its_time_limit_is_interrupted_and_keeps_its_report():
    # pytest writes its JUnit report when interrupted (SIGINT), not when terminated.
    text = (KIT / "test.sh").read_text(encoding="utf-8")
    limit = re.search(r"\nlimit=\(([^)]*)\)", text)
    assert limit is not None and limit[1].split()[:3] == ["timeout", "-s", "INT"]
    runs = [line for line in text.splitlines() if '"${pytest_args[@]}"' in line]
    assert len(runs) == 2
    assert all('"${limit[@]}"' in line and "timeout" not in line for line in runs)


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="runs the script with POSIX bash",
)
def test_the_fake_gpu_refuses_to_run_outside_the_self_tests_container(tmp_path: Path):
    # On a machine it would replace /dev/kfd and the render nodes and mount over /sys.
    # The commands it would run are stubs here that only record their calls.
    calls = tmp_path / "calls"
    for command in ("rm", "mkdir", "mknod", "mount", "nproc", "head"):
        stub = tmp_path / command
        stub.write_text(f'#!/bin/sh\necho "{command} $*" >>"{calls}"\n')
        stub.chmod(0o755)
    result = subprocess.run(
        ["bash", str(KIT / "fake_gpu.sh"), "gfx1100"],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
        check=False,
    )
    assert result.returncode != 0
    assert "only inside a container" in result.stderr
    assert not calls.exists()


def test_the_self_test_runs_the_fake_gpu_as_its_container_run():
    text = (KIT / "selftest.sh").read_text(encoding="utf-8")
    runs = [line.split() for line in text.splitlines() if "fake_gpu.sh" in line]
    runs = [words for words in runs if words[0] == "bash"]
    assert runs and all(words[2] == "--container" for words in runs)
