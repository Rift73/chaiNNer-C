"""gpu_guard, the VRAM guard of the U4-light GPU runs, tested without a GPU."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator

import gpu_guard
import psutil
import pytest

SLEEPER = [sys.executable, "-B", "-c", "import time; time.sleep(60)"]
# Starts a grandchild that sleeps, prints its PID and sleeps.
PARENT = [
    sys.executable,
    "-B",
    "-c",
    (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(60)'])\n"
        "print(p.pid, flush=True)\n"
        "time.sleep(60)\n"
    ),
]
GB = 1_000_000_000


def instance(pid: int, adapter: int = 0x42EFC, segment: int = 0) -> str:
    return f"pid_{pid}_luid_0x00000000_0x{adapter:08X}_phys_{segment}"


@pytest.fixture
def owned() -> Iterator[list[psutil.Process]]:
    """The processes a test starts; any still running at the end is ended by PID."""
    started: list[psutil.Process] = []
    yield started
    for process in started:
        if process.is_running():
            process.kill()
    psutil.wait_procs(started, timeout=10)


def start(command: list[str], owned: list[psutil.Process]) -> subprocess.Popen[str]:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    owned.append(psutil.Process(process.pid))
    return process


def test_instance_pid_reads_the_counter_instance_names():
    assert (
        gpu_guard.instance_pid("pid_13644_luid_0x00000000_0x0004499C_phys_0") == 13644
    )
    assert gpu_guard.instance_pid(instance(12)) == 12
    assert gpu_guard.instance_pid("_Total") is None
    assert gpu_guard.instance_pid("xpid_12_luid_0x0_0x1_phys_0") is None


def test_vram_by_pid_sums_every_adapter_and_segment_per_pid():
    sample = {
        instance(12): 5,
        instance(12, adapter=0x4499C): 7,
        instance(12, segment=1): 11,
        instance(123): 1000,
        "_Total": 99999,
    }
    assert gpu_guard.vram_by_pid(sample) == {12: 23, 123: 1000}


def test_guard_warns_once_then_terminates_only_the_child_tree(owned):
    child = start(PARENT, owned)
    bystander = start(SLEEPER, owned)
    assert child.stdout is not None
    grandchild = psutil.Process(int(child.stdout.readline()))
    owned.append(grandchild)
    readings = iter(
        [
            (GB, 0),
            (1_400_000_000, 1_200_000_000),
            (1_300_000_000, 1_300_000_000),
            (1_600_000_000, 1_500_000_000),
        ]
    )

    def sample() -> dict[str, int]:
        own, below = next(readings)
        return {
            instance(child.pid): own,
            instance(grandchild.pid, segment=1): below,
            # Never counted, never terminated: outside the child's tree.
            instance(bystander.pid): 10 * GB,
            instance(os.getpid()): 10 * GB,
        }

    logged: list[str] = []
    guard = gpu_guard.VramGuard(
        psutil.Process(child.pid), sample, device_used=lambda: 0, log=logged.append
    )
    assert guard.poll() == GB
    assert not guard.report.soft_warned
    assert guard.poll() == 2_600_000_000
    assert guard.report.soft_warned
    assert not guard.report.aborted
    assert guard.poll() == 2_600_000_000
    assert sum("soft limit" in line for line in logged) == 1
    assert child.poll() is None

    assert guard.poll() == 3_100_000_000
    report = guard.report
    assert report.aborted
    assert report.abort_reason == "3100000000 bytes of VRAM reached the hard limit"
    assert child.wait(10) is not None
    assert not grandchild.is_running()
    assert grandchild.pid in report.terminated
    assert report.terminated[-1] == child.pid
    assert bystander.pid not in report.terminated
    assert os.getpid() not in report.terminated
    assert bystander.poll() is None
    assert report.survivors == []
    assert report.peak == 3_100_000_000
    assert report.peak_by_pid == {
        child.pid: 1_600_000_000,
        grandchild.pid: 1_500_000_000,
    }
    assert report.polls == 4


def test_guard_fails_closed_when_the_counter_cannot_be_read():
    def sample() -> dict[str, int]:
        raise OSError("PdhCollectQueryData failed with PDH status 0xc0000bc6")

    logged: list[str] = []
    began = time.monotonic()
    report = gpu_guard.run_guarded(
        SLEEPER, sample=sample, device_used=lambda: 0, log=logged.append
    )
    assert time.monotonic() - began < 30
    assert report.aborted
    assert report.abort_reason is not None
    assert "could not be read" in report.abort_reason
    assert report.terminated[-1] == report.pid
    assert report.returncode not in (None, 0)
    assert any("ABORT" in line for line in logged)


def test_guard_lets_a_child_under_the_soft_limit_finish():
    command = [
        sys.executable,
        "-B",
        "-c",
        "import time; time.sleep(1.5); raise SystemExit(7)",
    ]

    def sample() -> dict[str, int]:
        # The guarded child is this process's only child.
        return {instance(p.pid): 500_000_000 for p in psutil.Process().children()}

    device = itertools.count(1000, 10)
    report = gpu_guard.run_guarded(
        command, sample=sample, device_used=lambda: next(device), log=print
    )
    assert report.returncode == 7
    assert not report.aborted
    assert not report.soft_warned
    assert report.terminated == []
    assert report.peak == 500_000_000
    assert report.peak_by_pid == {report.pid: 500_000_000}
    assert report.polls >= 3
    assert report.device_baseline == 1000
    assert report.device_delta_peak == 10 * report.polls
    times = [moment for moment, _ in report.timeline]
    assert times == sorted(times)


def test_counter_array_reads_gpu_process_memory():
    with gpu_guard.CounterArray() as counter:
        sample = counter.sample()
    assert all(gpu_guard.instance_pid(name) is not None for name in sample)
    assert all(isinstance(value, int) and value >= 0 for value in sample.values())


def test_counter_array_reads_an_instance_that_appears_after_it_opened(owned):
    with gpu_guard.CounterArray(r"\Process(*)\ID Process") as counter:
        assert os.getpid() in counter.sample().values()
        child = start(SLEEPER, owned)
        seen = False
        for _ in range(100):
            if child.pid in counter.sample().values():
                seen = True
                break
            time.sleep(0.05)
        assert seen
    with pytest.raises(OSError, match="is closed"):
        counter.sample()


def test_counter_array_refuses_an_unknown_counter():
    with pytest.raises(OSError, match="PdhAddEnglishCounterW"):
        gpu_guard.CounterArray(r"\No Such Object(*)\No Such Counter")


def test_engine_caps_sit_below_the_guard():
    total = 32607 * 1024**2  # an RTX 5090
    assert gpu_guard.torch_memory_fraction(total) * total == pytest.approx(2 * 1024**3)
    assert gpu_guard.torch_memory_fraction(1024**3) == 1.0
    assert gpu_guard.ALLOCATOR_CAP < gpu_guard.SOFT_LIMIT < gpu_guard.HARD_LIMIT
    assert gpu_guard.HARD_LIMIT == 3 * GB
    assert gpu_guard.ORT_CUDA_OPTIONS == {
        "device_id": 0,
        "gpu_mem_limit": 2 * 1024**3,
        "arena_extend_strategy": "kSameAsRequested",
    }


def test_child_environment_unhides_the_gpu_only_on_request(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("VK_LOADER_DRIVERS_DISABLE", "*")
    hidden = gpu_guard.child_environment(unhide_gpu=False)
    assert hidden["CUDA_VISIBLE_DEVICES"] == "-1"
    assert hidden["VK_LOADER_DRIVERS_DISABLE"] == "*"
    unhidden = gpu_guard.child_environment(unhide_gpu=True)
    assert not set(gpu_guard.HIDDEN) & unhidden.keys()
    assert unhidden == {k: v for k, v in hidden.items() if k not in gpu_guard.HIDDEN}


class FakeDevices:
    """DeviceMemory without NVML."""

    def __enter__(self) -> FakeDevices:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def used(self) -> int:
        return 0


def test_cli_returns_the_childs_status_and_writes_the_report(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_guard, "DeviceMemory", FakeDevices)
    path = tmp_path / "report.json"
    command = [sys.executable, "-B", "-c", "raise SystemExit(5)"]
    assert gpu_guard.main(["--json", str(path), "--", *command]) == 5
    report = json.loads(path.read_text("utf-8"))
    assert report["command"] == command
    assert report["returncode"] == 5
    assert report["aborted"] is False
    assert report["hard_limit"] == 3 * GB
    assert report["device_delta_peak"] in (None, 0)
