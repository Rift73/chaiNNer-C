"""The test suite (README Test) in parallel.

The test files of native/tests and backend/tests are split into shards balanced by their
last measured durations (by file size before any are measured), and each shard runs in
its own pytest process with the environment of README Test. Every process loads the
suite's conftest.py as the one-process run does, so a shard runs exactly like the suite.
Each shard reports every test phase as a JSON line (the shard_report.py plugin), which
gives one summary, the failures, and each file's duration for the next run's balance,
kept in the git-ignored native/local/test-durations.json.

usage: python -B native/tools/run_tests.py [--workers N] [PYTEST_ARG ...]

Arguments it does not know go to every shard's pytest. The exit status is 1 when a test
failed or errored, or a shard could not run its tests (collection or internal error).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
SUITES = ("native/tests", "backend/tests")
DURATIONS = ROOT / "native" / "local" / "test-durations.json"
# Seconds per byte of test file, for files without a measured duration
SECONDS_PER_BYTE = 1e-4
# They test the native pool against the machine's CPUs (and skip below four), so they
# run in a shard of their own on every CPU, without thread caps
WHOLE_MACHINE = (
    "native/tests/test_parallel.py",
    "native/tests/test_parallel_affinity.py",
)


@dataclass
class Totals:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    failures: list[str] = field(default_factory=list)
    durations: dict[str, float] = field(default_factory=dict)


def test_files() -> list[str]:
    return sorted(
        path.relative_to(ROOT).as_posix()
        for suite in SUITES
        for path in (ROOT / suite).glob("test_*.py")
    )


def shards(
    files: list[str], durations: dict[str, float], workers: int
) -> list[list[str]]:
    """Longest first, each file to the least loaded shard."""

    def cost(name: str) -> float:
        measured = durations.get(name)
        if measured is not None:
            return measured
        return (ROOT / name).stat().st_size * SECONDS_PER_BYTE

    loads = [0.0] * workers
    groups: list[list[str]] = [[] for _ in range(workers)]
    for name in sorted(files, key=cost, reverse=True):
        index = loads.index(min(loads))
        loads[index] += cost(name)
        groups[index].append(name)
    return [group for group in groups if group]


def environment(threads: int | None, report: Path) -> dict[str, str]:
    env = dict(os.environ)
    # pytest imports -p plugins before it applies its pythonpath setting
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "native" / "tools"), *filter(None, [env.get("PYTHONPATH")])]
    )
    env["CHAINNER_C_SHARD_REPORT"] = str(report)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["NUMBA_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / "chainner-c-numba")
    # Thread pools sized to the shard's CPUs, not the machine's: the native pool reads
    # the process affinity (parallel.c), these libraries their variables
    if threads is not None:
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ):
            env[name] = str(threads)
    return env


def read_report(report: Path, totals: Totals) -> None:
    phases: dict[str, dict[str, str]] = {}
    for line in report.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        phases.setdefault(record["nodeid"], {})[record["when"]] = record["outcome"]
        name = record["nodeid"].split("::", 1)[0]
        totals.durations[name] = totals.durations.get(name, 0.0) + record["duration"]
    for nodeid, outcome in phases.items():
        if outcome.get("call") == "failed":
            totals.failed += 1
            totals.failures.append(nodeid)
        elif "failed" in outcome.values():
            totals.errors += 1
            totals.failures.append(nodeid)
        elif "skipped" in outcome.values():
            totals.skipped += 1
        else:
            totals.passed += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="The test suite in parallel.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="pytest processes (default: half the logical CPUs)",
    )
    args, pytest_args = parser.parse_known_args()
    durations: dict[str, float] = (
        json.loads(DURATIONS.read_text(encoding="utf-8")) if DURATIONS.exists() else {}
    )
    files = [name for name in test_files() if name not in WHOLE_MACHINE]
    groups = shards(files, durations, args.workers)
    # Each shard on its own logical CPUs (consecutive, so a core's two hyperthreads)
    cpus = list(range(os.cpu_count() or 1))
    per_shard = max(1, len(cpus) // len(groups))
    placed: list[tuple[list[str], list[int] | None]] = [
        (group, cpus[i * per_shard % len(cpus) :][:per_shard])
        for i, group in enumerate(groups)
    ]
    placed.append((list(WHOLE_MACHINE), None))
    start = time.perf_counter()
    totals = Totals()
    broken: list[str] = []
    with tempfile.TemporaryDirectory(prefix="chainner-c-tests-") as temporary:
        running: list[tuple[subprocess.Popen[bytes], Path, Path]] = []
        for index, (group, shard_cpus) in enumerate(placed):
            report = Path(temporary) / f"shard-{index}.jsonl"
            log = Path(temporary) / f"shard-{index}.log"
            command = [
                sys.executable,
                "-B",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "-p",
                "shard_report",
                *pytest_args,
                *group,
            ]
            with log.open("wb") as output:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment(
                        None if shard_cpus is None else len(shard_cpus), report
                    ),
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
            if shard_cpus is not None:
                # Right after the process starts, long before it loads the native pool
                psutil.Process(process.pid).cpu_affinity(shard_cpus)
            running.append((process, report, log))
        for process, report, log in running:
            # 0: passed, 1: a test failed, 5: no tests; otherwise tests did not run
            if process.wait() in (0, 1, 5) and report.exists():
                read_report(report, totals)
            else:
                tail = log.read_text(encoding="utf-8", errors="replace")[-2000:]
                broken.append(f"{log.name} (exit {process.returncode}):\n{tail}")
    seconds = time.perf_counter() - start
    if not broken:
        DURATIONS.parent.mkdir(parents=True, exist_ok=True)
        DURATIONS.write_text(
            json.dumps(
                dict(sorted({**durations, **totals.durations}.items())), indent=1
            ),
            encoding="utf-8",
        )
    for name in totals.failures:
        print(f"FAILED {name}")
    for shard in broken:
        print(f"SHARD DID NOT RUN ITS TESTS {shard}")
    print(
        f"{totals.passed} passed, {totals.failed} failed, {totals.errors} errors,"
        f" {totals.skipped} skipped in {seconds:.0f} s ({len(placed)} shards)"
    )
    return 1 if totals.failed or totals.errors or broken else 0


if __name__ == "__main__":
    sys.exit(main())
