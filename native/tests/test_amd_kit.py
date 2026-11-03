"""The AMD kit (native/tools/amd, native/AMD.md) on what a bare Linux image lacks: no
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
BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"


def kit_module(name: str):
    spec = importlib.util.spec_from_file_location(f"kit_{name}", KIT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up by name
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gpu_proof():
    return kit_module("gpu_proof")


def backend_environment() -> dict[str, str]:
    """This environment with backend/src on the path, as the kit runs its scripts."""
    path = os.environ.get("PYTHONPATH")
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(BACKEND), *([path] if path else [])]),
    }


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


def test_the_spandrel_checks_run_between_the_chains_and_the_speed_table():
    steps = [
        line.split()[1].strip('"')
        for line in (KIT / "kit.sh").read_text(encoding="utf-8").splitlines()
        if line.startswith("step ")
    ]
    chains = steps.index("chains")
    assert steps[chains + 1 : chains + 3] == ["every", "HIP"]
    assert steps[chains + 3] == "speed"


def test_every_architecture_has_a_network_or_a_reason():
    # A spandrel that adds an architecture whose loader is generic in no single
    # network class needs NETWORKS or SKIPPED to say what the check does with it.
    check = kit_module("arch_check")
    for name, architecture in check.registry().items():
        if name not in check.SKIPPED:
            assert check.networks(architecture), name


def test_the_architecture_check_runs_small_architectures_on_the_cpu():
    # Compact from its defaults, SPAN from the check's own settings; each in a process
    # of its own, FP32 and BF16 (the CPU leaves FP16 out) against FP32 on the CPU.
    result = subprocess.run(
        [
            *(sys.executable, "-B", str(KIT / "arch_check.py")),
            *("--device", "cpu", "--only", "^(Compact|SPAN)$"),
        ],
        capture_output=True,
        text=True,
        env=backend_environment(),
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "## Every architecture"
    # The rows come as each architecture finishes, the Result line last
    assert lines[-1].startswith(
        "Result: 2 architectures on the CPU, 4 rows: 4 ok, 0 FAILED,"
    )
    assert sum(line.startswith("Result: ") for line in lines) == 1
    # Architecture, network, input, precision, result, max abs diff, ms
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in lines
        if line.startswith(("| Compact", "| SPAN"))
    ]
    assert [(row[0], row[2], row[3], row[4]) for row in rows] == [
        ("Compact", "3x64x64", "FP32", "ok"),
        ("Compact", "3x64x64", "BF16", "ok"),
        ("SPAN", "3x64x64", "FP32", "ok"),
        ("SPAN", "3x64x64", "BF16", "ok"),
    ]
    assert rows[0][1] == "defaults; 0.62 M parameters"
    assert rows[2][1].startswith("num_in_ch=3, num_out_ch=3; ")
    assert rows[1][1] == rows[3][1] == ""
    # The same model on the same device in FP32 gives the same output; BF16 does not
    assert rows[0][5] == rows[2][5] == "0"
    assert float(rows[1][5].split()[0]) > 0
    assert all(float(row[6]) > 0 for row in rows)


def test_a_worker_that_ends_early_leaves_failed_rows():
    check = kit_module("arch_check")
    network = {
        "kind": "network",
        "arch": "DAT",
        "variant": "",
        "settings": "defaults",
        "parameters": 1_000_000,
        "input": "3x64x64",
        "precisions": ["FP32", "FP16", "BF16"],
    }
    fp32 = {
        "kind": "row",
        "arch": "DAT",
        "variant": "",
        "precision": "FP32",
        "result": "ok",
        "max_diff": 1e-6,
        "ms": 2.0,
    }
    ending = "ended by signal 11; last error line: Segmentation fault"
    rows, skips = check.rows_of("DAT", check.Outcome([network, fp32], ending))
    assert skips == []
    assert [(row.precision, row.result, row.failed) for row in rows] == [
        ("FP32", "ok", False),
        ("FP16", "error", True),
        ("BF16", "error", True),
    ]
    assert rows[1].markdown(False) == (
        f"| DAT |  | 3x64x64 | FP16 | FAILED: {ending} |  |  |"
    )
    # A process that ended before naming a network
    rows, _ = check.rows_of("DAT", check.Outcome([], "no result after 180 s (stopped)"))
    assert len(rows) == 1 and rows[0].failed


def test_a_failure_the_cpu_repeats_is_listed_but_not_counted():
    check = kit_module("arch_check")
    network = {
        "kind": "network",
        "arch": "ATD",
        "variant": "",
        "settings": "defaults",
        "parameters": 1_000_000,
        "input": "3x64x64",
        "precisions": ["BF16"],
    }
    bf16 = {
        "kind": "row",
        "arch": "ATD",
        "variant": "",
        "precision": "BF16",
        "result": "error",
        "error": "RuntimeError: expected m1 and m2 to have the same dtype",
        "cpu_too": True,
    }
    rows, _ = check.rows_of("ATD", check.Outcome([network, bf16], None))
    assert len(rows) == 1 and not rows[0].failed
    assert "| FAILED on the CPU too: RuntimeError: expected m1" in rows[0].markdown(
        True
    )


def test_sst_failures_are_expected_only_on_gpus_without_flash_attention():
    # SST and SSTReal share SST's blocks, whose attention allows only flash or cuDNN
    # attention
    check = kit_module("arch_check")

    def failed_rows():
        return [
            check.Row("SST", "", "scale=4", 1, "3x64x64", "FP32", "error", "boom"),
            check.Row("SSTReal", "", "scale=4", 1, "3x64x64", "FP16", "error", "boom"),
            check.Row("DAT", "", "defaults", 1, "3x64x64", "FP32", "error", "boom"),
            check.Row("SST", "", "scale=4", 1, "3x64x64", "BF16", "nonfinite"),
        ]

    for gfx, accepted in (("gfx1030", True), ("gfx908", True), ("gfx1100", False)):
        rows = failed_rows()
        check.expected(rows, gfx)
        assert [row.failed for row in rows] == [
            not accepted,
            not accepted,
            True,
            True,
        ], gfx


def dat_network(precisions: list[str]) -> dict[str, object]:
    return {
        "kind": "network",
        "arch": "DAT",
        "variant": "",
        "settings": "defaults",
        "parameters": 1_000_000,
        "input": "3x64x64",
        "precisions": precisions,
    }


def test_an_architecture_the_time_cap_stops_is_not_run_not_failed():
    check = kit_module("arch_check")
    log: list[str] = []
    # A worker stopped at its time limit: the flag the check tells the cap by
    outcome = check.run_worker(
        [sys.executable, "-c", "import time; time.sleep(30)"], 1, log.append
    )
    assert outcome.timed_out
    assert outcome.ending.startswith("no result after 1 s (stopped)")
    # Stopped by the check's time cap: its unfinished precisions are not run
    stopped = check.Outcome([dat_network(["FP32", "FP16"])], outcome.ending, [], True)
    rows, skips = check.rows_of("DAT", stopped, capped=True)
    assert [(row.precision, row.result, row.failed) for row in rows] == [
        ("FP32", "not run", False),
        ("FP16", "not run", False),
    ]
    assert "| not run: the check's time cap ran out" in rows[0].markdown(True)
    assert skips == []
    rows, skips = check.rows_of("DAT", check.Outcome([], outcome.ending), capped=True)
    assert rows == []
    assert skips == ["- DAT: not run, the check's time cap ran out before it finished"]
    # Stopped by its own time limit: a failure
    rows, _ = check.rows_of("DAT", stopped)
    assert all(row.failed for row in rows)
    assert check.arguments([]).timeout == 180


def test_unreadable_worker_output_is_that_architectures_failure():
    check = kit_module("arch_check")
    log: list[str] = []
    # Bytes that are not UTF-8 and a line that is not JSON, then a valid record
    script = (
        "import sys; out = sys.stdout.buffer;"
        " out.write(b'{\\xff not json\\n'); out.write(b'[1]\\n');"
        ' out.write(b\'{"kind": "row"}\\n\'); out.flush()'
    )
    outcome = check.run_worker([sys.executable, "-c", script], 60, log.append)
    assert outcome.ending is None
    assert outcome.records == [{"kind": "row"}]
    assert outcome.unreadable == ["{\ufffd not json"]
    rows, _ = check.rows_of("DAT", outcome)
    # The line that is not JSON, and the record that is not a whole row
    assert len(rows) == 2 and all(row.failed for row in rows)
    assert rows[0].error == "unreadable worker output: {\ufffd not json"
    assert rows[1].error == 'unreadable worker output: {"kind": "row"}'


def tiny_esrgan(folder: Path) -> Path:
    import torch
    from spandrel.architectures.ESRGAN import ESRGAN

    torch.manual_seed(0)
    path = folder / "esrgan.pth"
    torch.save(ESRGAN(num_filters=8, num_blocks=1, scale=2).state_dict(), path)
    return path


def hip_kernels_section(monkeypatch, models: Path, rocm: bool):
    """hip_kernels.py's section on the CPU, as if spandrel had spandrel.rocm or not
    (the processes it starts see the installed spandrel)."""
    check = kit_module("hip_kernels")
    find_spec = importlib.util.find_spec

    def with_or_without_rocm(name: str, *args):
        if name == "spandrel.rocm":
            return find_spec("spandrel") if rocm else None
        return find_spec(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", with_or_without_rocm)
    monkeypatch.setenv("PYTHONPATH", backend_environment()["PYTHONPATH"])
    lines: list[str] = []
    status = check.section(hip_kernels_arguments(check, models), lines.append)
    return lines, status


def hip_kernels_arguments(check, models: Path):
    return check.arguments(
        [
            *("--models", str(models)),
            *("--image", str(KIT / "chains" / "input.png")),
            *("--device", "cpu", "--size", "32", "--iterations", "1"),
        ]
    )


def test_the_hip_kernel_check_passes_with_a_spandrel_without_them(
    monkeypatch, tmp_path: Path
):
    import spandrel

    models = tmp_path / "models.txt"
    models.write_text(f"esrgan {tiny_esrgan(tmp_path)}\n", encoding="utf-8")
    lines, status = hip_kernels_section(monkeypatch, models, rocm=False)
    assert status == 0
    assert lines == [
        f"Result: spandrel without HIP kernels (version {spandrel.__version__})."
    ]


def test_the_hip_kernel_check_compares_both_runs_and_skips_missing_models(
    monkeypatch, tmp_path: Path
):
    # ESRGAN on the CPU with the kernels off and on (spandrel runs no HIP kernel on
    # a CPU, so the outputs are the same); no DAT2 in models.txt
    models = tmp_path / "models.txt"
    models.write_text(f"esrgan {tiny_esrgan(tmp_path)}\n", encoding="utf-8")
    lines, status = hip_kernels_section(monkeypatch, models, rocm=True)
    assert status == 0, lines
    assert lines[-1].startswith("Result: spandrel ")
    assert lines[-1].endswith(" on CPU: 1 rows, 0 FAILED, 1 skipped.")
    rows = [line for line in lines if line.startswith("| esrgan |")]
    assert len(rows) == 1
    cells = rows[0].split(" | ")
    assert cells[1] == "FP32" and cells[5] == "0"
    assert cells[6].startswith(("no spandrel.rocm", "no: spandrel.rocm.is_available()"))
    assert "- dat2: skipped, no dat2 model (copy it to --models as dat2.pth)" in lines


def test_a_kernel_that_fell_back_fails_its_row():
    check = kit_module("hip_kernels")
    debug = "DEBUG: HIP kernel for 3 convs"
    used = {"available": True, "logs": [debug], "warnings": []}
    assert check.kernels_used(used) == ("yes (1 log lines)", False)
    warning = "WARNING: HIP depthwise kernel failed, using PyTorch's: no code object"
    fell_back = {"available": True, "logs": [debug, warning], "warnings": [warning]}
    assert check.kernels_used(fell_back) == (f"FAILED: fell back: {warning}", True)
    unavailable = {"available": False, "logs": [], "warnings": []}
    assert check.kernels_used(unavailable)[1] is False


def test_the_hip_kernel_check_writes_each_row_when_its_runs_finish(
    monkeypatch, tmp_path: Path
):
    # Two models; the runs are stand-ins that write their outputs and records. The
    # ESRGAN row is written before DAT2's runs start, and DAT2's default run fell back.
    import numpy as np

    check = kit_module("hip_kernels")
    for role in ("esrgan", "dat2"):
        (tmp_path / f"{role}.pth").write_bytes(b"")
    models = tmp_path / "models.txt"
    models.write_text(
        f"esrgan {tmp_path / 'esrgan.pth'}\ndat2 {tmp_path / 'dat2.pth'}\n",
        encoding="utf-8",
    )
    lines: list[str] = []
    written_at_start: dict[str, int] = {}
    warning = "WARNING: HIP depthwise kernel failed, using PyTorch's"

    def run(model: Path, precision: str, kernels: bool, args, folder: Path):
        written_at_start.setdefault(model.stem, len(lines))
        output = folder / f"{model.stem}-{precision}-{kernels}.npy"
        np.save(output, np.zeros((1, 3, 4, 4), np.float32))
        fell_back = kernels and model.stem == "dat2"
        record = {
            "gpu": "Radeon",
            "torch": "2",
            "architecture": model.stem,
            "supports_half": False,
            "available": True,
            "finite": True,
            "ms": 2.0,
            "logs": [warning] if fell_back else [],
            "warnings": [warning] if fell_back else [],
        }
        return check.Run(record, "", output)

    monkeypatch.setattr(check.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(check, "run", run)
    status = check.section(hip_kernels_arguments(check, models), lines.append)
    assert status == 1
    esrgan_row = next(i for i, line in enumerate(lines) if line.startswith("| esrgan"))
    assert esrgan_row < written_at_start["dat2"]
    dat2_row = next(line for line in lines if line.startswith("| dat2"))
    assert dat2_row.endswith(f"| FAILED: fell back: {warning} |")
    assert lines[-1].endswith(": 2 rows, 1 FAILED, 0 skipped.")


def test_the_kit_waits_for_the_checks_and_keeps_the_rows_of_one_it_stops():
    arch_check = kit_module("arch_check")
    hip_kernels = kit_module("hip_kernels")
    text = (KIT / "kit.sh").read_text(encoding="utf-8")

    def outer(script: str) -> int:
        match = re.search(rf"timeout -k 60 (\d+) .*{script}", text)
        assert match is not None, script
        return int(match[1])

    defaults = arch_check.arguments([])
    # The check ends by its time cap, the last architecture's stop included
    assert outer("arch_check.py") >= defaults.time_cap + defaults.timeout
    # Every run of the HIP check at its time limit: models, FP32 and FP16, off and on
    runs = len(hip_kernels.MODELS) * 2 * 2
    timeout = hip_kernels.arguments(["--models", "m", "--image", "i"]).timeout
    assert outer("hip_kernels.py") > runs * timeout
    for fragment in ("44-archs", "46-hip-kernels"):
        assert f'finished "$(fragment {fragment})" "$?"' in text


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="runs the script with POSIX bash",
)
def test_a_check_the_kit_stopped_gets_a_result_line(tmp_path: Path):
    text = (KIT / "kit.sh").read_text(encoding="utf-8")
    function = re.search(r"\nfinished\(\) \{\n.*?\n\}\n", text, re.S)
    assert function is not None
    fragment = tmp_path / "46-hip-kernels.md"
    fragment.write_text("## spandrel's HIP kernels on and off\n\n| esrgan | row |\n")
    script = f'{function[0]}\nfinished "$1" 124\necho "status $?"\n'
    result = subprocess.run(
        ["bash", "-c", script, "bash", str(fragment)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout == "status 124\n"
    lines = fragment.read_text().splitlines()
    assert lines[2] == "| esrgan | row |"
    assert lines[-1].startswith("Result: STOPPED (exit 124)")
    # A check that finished keeps its own Result line
    subprocess.run(["bash", "-c", script, "bash", str(fragment)], check=True)
    assert fragment.read_text().count("Result: ") == 1


def test_an_unreadable_hip_kernels_run_is_that_runs_failure():
    check = kit_module("hip_kernels")
    script = r"import sys; sys.stdout.buffer.write(b'{\xff not json\n')"
    record, error = check.launch([sys.executable, "-c", script], dict(os.environ), 60)
    assert record is None
    assert error == "unreadable worker output: {\ufffd not json"
