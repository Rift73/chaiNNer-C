"""The VRAM guard of the functional GPU runs (U4-light, Consult 13 D-24).

The owner's rule (2026-10-07): no GPU benchmark, and never more than 3 GB of VRAM.

Usage:
  python native/tools/gpu_guard.py [--json PATH] [--unhide-gpu] -- COMMAND [ARG ...]

COMMAND runs as a child. Every POLL_SECONDS the guard sums the dedicated VRAM of the
child's process tree from the Windows perf counter
\\GPU Process Memory(pid_<pid>_*)\\Dedicated Usage: one wildcard query (COUNTER), whose
instances are named pid_<pid>_luid_<adapter>_phys_<n>. The tree is the child and every
process it starts (psutil: parent PID, and a creation time not before the parent's),
kept once seen. At SOFT_LIMIT the guard warns once. At HARD_LIMIT, or when the counter
cannot be read, it terminates the tree by PID, the child last: only processes the
child started, never any other. NVML's device-level used memory is recorded as a delta
from the start, as a cross-check only: on WDDM NVML has no per-process figure, and the
device total moves with every other process. "GB" is 10**9 bytes, the stricter reading.

The guard is the backstop; each engine is capped below it:
- torch: torch.cuda.set_per_process_memory_fraction(torch_memory_fraction(total)) stops
  the caching allocator at about ALLOCATOR_CAP. The CUDA context and cuDNN's workspace
  sit outside that cap.
- ONNX Runtime: ORT_CUDA_OPTIONS on the CUDA execution provider.
- ncnn has no cap: a small model on an input of at most 256 px.

--unhide-gpu removes HIDDEN (CUDA_VISIBLE_DEVICES, VK_LOADER_DRIVERS_DISABLE) from the
child's environment only; the guard itself stays a CPU process. The exit status is the
child's, or ABORTED after an abort. --json writes the GuardReport.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from ctypes import wintypes
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil
import pynvml

COUNTER = r"\GPU Process Memory(*)\Dedicated Usage"
SOFT_LIMIT = 2_500_000_000
HARD_LIMIT = 3_000_000_000
POLL_SECONDS = 0.25
ALLOCATOR_CAP = 2 * 1024**3
ORT_CUDA_OPTIONS: dict[str, int | str] = {
    "device_id": 0,
    "gpu_mem_limit": ALLOCATOR_CAP,
    "arena_extend_strategy": "kSameAsRequested",
}
HIDDEN = ("CUDA_VISIBLE_DEVICES", "VK_LOADER_DRIVERS_DISABLE")
ABORTED = 3
TERMINATE_WAIT = 10.0

_INSTANCE = re.compile(r"pid_(\d+)_")
_PDH_FMT_LARGE = 0x00000400
_PDH_MORE_DATA = 0x800007D2
_PDH_NO_DATA = 0x800007D5
_PDH_VALID = (0x0, 0x1)  # PDH_CSTATUS_VALID_DATA, PDH_CSTATUS_NEW_DATA


def torch_memory_fraction(total: int) -> float:
    """The set_per_process_memory_fraction that caps a device of total bytes at
    ALLOCATOR_CAP."""
    return min(1.0, ALLOCATOR_CAP / total)


def instance_pid(name: str) -> int | None:
    """The PID in a GPU Process Memory instance name, or None."""
    match = _INSTANCE.match(name)
    return None if match is None else int(match.group(1))


def vram_by_pid(sample: Mapping[str, int]) -> dict[int, int]:
    """A counter sample summed per PID, over every adapter and segment."""
    totals: dict[int, int] = {}
    for name, value in sample.items():
        pid = instance_pid(name)
        if pid is not None:
            totals[pid] = totals.get(pid, 0) + value
    return totals


class _Value(ctypes.Structure):
    """PDH_FMT_COUNTERVALUE, read through the LONGLONG member of its union."""

    _fields_ = (("status", wintypes.DWORD), ("value", ctypes.c_longlong))


class _Item(ctypes.Structure):
    """PDH_FMT_COUNTERVALUE_ITEM_W."""

    _fields_ = (("name", wintypes.LPWSTR), ("value", _Value))


class CounterArray:
    """One wildcard counter path as a PDH query, read as {instance: value}.

    PDH expands the wildcard again at every collection, so an instance that appears
    after the query opened is read too. Raises OSError on any PDH failure.
    """

    def __init__(self, path: str = COUNTER) -> None:
        self._pdh = ctypes.WinDLL("pdh")
        signatures = {
            "PdhOpenQueryW": (wintypes.LPCWSTR, ctypes.c_size_t, ctypes.c_void_p),
            "PdhAddEnglishCounterW": (
                ctypes.c_void_p,
                wintypes.LPCWSTR,
                ctypes.c_size_t,
                ctypes.c_void_p,
            ),
            "PdhCollectQueryData": (ctypes.c_void_p,),
            "PdhGetFormattedCounterArrayW": (
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
                ctypes.POINTER(wintypes.DWORD),
                ctypes.c_void_p,
            ),
            "PdhCloseQuery": (ctypes.c_void_p,),
        }
        for name, arguments in signatures.items():
            function = getattr(self._pdh, name)
            function.argtypes = arguments
            function.restype = ctypes.c_uint32
        self.path = path
        self._query = ctypes.c_void_p()
        self._check(
            self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)),
            "PdhOpenQueryW",
        )
        self._counter = ctypes.c_void_p()
        status = self._pdh.PdhAddEnglishCounterW(
            self._query, path, 0, ctypes.byref(self._counter)
        )
        if status:
            self.close()
            self._check(status, f"PdhAddEnglishCounterW({path!r})")

    @staticmethod
    def _check(status: int, call: str) -> None:
        if status:
            raise OSError(f"{call} failed with PDH status {status:#010x}")

    def sample(self) -> dict[str, int]:
        """Every instance with valid data, collected now."""
        if not self._query:
            raise OSError(f"The query of {self.path!r} is closed")
        status = self._pdh.PdhCollectQueryData(self._query)
        if status == _PDH_NO_DATA:
            return {}
        self._check(status, "PdhCollectQueryData")
        size, count = wintypes.DWORD(0), wintypes.DWORD(0)
        status = self._pdh.PdhGetFormattedCounterArrayW(
            self._counter, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None
        )
        if status in (0, _PDH_NO_DATA):
            return {}
        if status != _PDH_MORE_DATA:
            self._check(status, "PdhGetFormattedCounterArrayW")
        buffer = (ctypes.c_byte * size.value)()
        self._check(
            self._pdh.PdhGetFormattedCounterArrayW(
                self._counter,
                _PDH_FMT_LARGE,
                ctypes.byref(size),
                ctypes.byref(count),
                buffer,
            ),
            "PdhGetFormattedCounterArrayW",
        )
        items = ctypes.cast(buffer, ctypes.POINTER(_Item))
        return {
            items[index].name: items[index].value.value
            for index in range(count.value)
            if items[index].value.status in _PDH_VALID
        }

    def close(self) -> None:
        if self._query:
            self._pdh.PdhCloseQuery(self._query)
            self._query = ctypes.c_void_p()

    def __enter__(self) -> CounterArray:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class DeviceMemory:
    """NVML's used memory, summed over every device (nvidia-ml-py)."""

    def __init__(self) -> None:
        pynvml.nvmlInit()
        self._open = True
        self._handles = [
            pynvml.nvmlDeviceGetHandleByIndex(index)
            for index in range(pynvml.nvmlDeviceGetCount())
        ]

    def used(self) -> int:
        return sum(
            int(pynvml.nvmlDeviceGetMemoryInfo(handle).used) for handle in self._handles
        )

    def close(self) -> None:
        if self._open:
            self._open = False
            pynvml.nvmlShutdown()

    def __enter__(self) -> DeviceMemory:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass
class GuardReport:
    """What one guarded run saw. VRAM is in bytes, times in seconds from the start."""

    command: list[str]
    pid: int
    soft_limit: int
    hard_limit: int
    interval: float
    returncode: int | None = None
    polls: int = 0
    peak: int = 0
    peak_by_pid: dict[int, int] = field(default_factory=dict)
    soft_warned: bool = False
    aborted: bool = False
    abort_reason: str | None = None
    terminated: list[int] = field(default_factory=list)
    survivors: list[int] = field(default_factory=list)
    device_baseline: int | None = None
    device_delta_peak: int | None = None
    device_error: str | None = None
    timeline: list[tuple[float, int]] = field(default_factory=list)
    seconds: float = 0.0


def _log(message: str) -> None:
    print(f"[gpu_guard] {message}", file=sys.stderr, flush=True)


class VramGuard:
    """Polls a process tree's dedicated VRAM on a thread (a context manager).

    sample returns a counter sample ({instance: bytes}, CounterArray.sample);
    device_used, when given, NVML's device total (DeviceMemory.used).
    """

    def __init__(
        self,
        root: psutil.Process,
        sample: Callable[[], Mapping[str, int]],
        *,
        device_used: Callable[[], int] | None = None,
        soft_limit: int = SOFT_LIMIT,
        hard_limit: int = HARD_LIMIT,
        interval: float = POLL_SECONDS,
        command: Sequence[str] = (),
        log: Callable[[str], None] = _log,
    ) -> None:
        self.root = root
        self.report = GuardReport(
            list(command), root.pid, soft_limit, hard_limit, interval
        )
        self._sample = sample
        self._device_used = device_used
        self._log = log
        self._known: dict[int, psutil.Process] = {root.pid: root}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gpu_guard", daemon=True)
        self._start = time.monotonic()
        if device_used is not None:
            self.report.device_baseline = self._device_reading()

    def processes(self) -> list[psutil.Process]:
        """The tree's living processes: each one seen so far, and their children."""
        for process in list(self._known.values()):
            try:
                children = process.children(recursive=True)
            except psutil.NoSuchProcess:
                continue
            self._known.update((child.pid, child) for child in children)
        return [process for process in self._known.values() if process.is_running()]

    def poll(self) -> int:
        """One reading of the tree's VRAM; warns or aborts on it. Returns the sum."""
        pids = {process.pid for process in self.processes()}
        usage = {
            pid: value
            for pid, value in vram_by_pid(self._sample()).items()
            if pid in pids
        }
        total = sum(usage.values())
        report = self.report
        report.polls += 1
        report.timeline.append((round(time.monotonic() - self._start, 3), total))
        report.peak = max(report.peak, total)
        for pid, value in usage.items():
            report.peak_by_pid[pid] = max(report.peak_by_pid.get(pid, 0), value)
        if self._device_used is not None and report.device_baseline is not None:
            used = self._device_reading()
            if used is not None:
                delta = used - report.device_baseline
                if report.device_delta_peak is None or delta > report.device_delta_peak:
                    report.device_delta_peak = delta
        if total >= report.hard_limit:
            self.abort(f"{total} bytes of VRAM reached the hard limit")
        elif total >= report.soft_limit and not report.soft_warned:
            report.soft_warned = True
            self._log(f"WARNING: {total} bytes of VRAM reached the soft limit")
        return total

    def _device_reading(self) -> int | None:
        if self._device_used is None:
            return None
        try:
            return self._device_used()
        except pynvml.NVMLError as error:
            self.report.device_error = repr(error)
            return None

    def abort(self, reason: str) -> None:
        """Terminates the tree by PID, the child last; once."""
        if self.report.aborted:
            return
        self.report.aborted = True
        self.report.abort_reason = reason
        self._log(f"ABORT: {reason}; terminating the child's process tree")
        living = self.processes()
        ordered = [p for p in reversed(living) if p.pid != self.root.pid]
        ordered += [p for p in living if p.pid == self.root.pid]
        for process in ordered:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                continue
            self.report.terminated.append(process.pid)
        _, alive = psutil.wait_procs(ordered, timeout=TERMINATE_WAIT)
        self.report.survivors = [process.pid for process in alive]

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except (OSError, psutil.Error) as error:
                # Fail closed: a guard that cannot read the counter stops the run.
                self.abort(f"the VRAM counter could not be read: {error!r}")
            if self.report.aborted:
                return
            self._stop.wait(self.report.interval)

    def __enter__(self) -> VramGuard:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self.report.seconds = round(time.monotonic() - self._start, 3)
        if not self.report.aborted:
            self.report.survivors = [
                process.pid
                for process in self.processes()
                if process.pid != self.root.pid
            ]


def run_guarded(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    sample: Callable[[], Mapping[str, int]] | None = None,
    device_used: Callable[[], int] | None = None,
    soft_limit: int = SOFT_LIMIT,
    hard_limit: int = HARD_LIMIT,
    interval: float = POLL_SECONDS,
    log: Callable[[str], None] = _log,
) -> GuardReport:
    """Runs command as a guarded child and returns the report once it exits.

    Without sample, the perf counter (CounterArray). Without device_used, NVML
    (DeviceMemory); if NVML cannot start, the run goes on without the cross-check
    and the report's device_error says why.
    """
    device_error = None
    with contextlib.ExitStack() as stack:
        if sample is None:
            sample = stack.enter_context(CounterArray()).sample
        if device_used is None:
            try:
                device_used = stack.enter_context(DeviceMemory()).used
            except pynvml.NVMLError as error:
                device_error = repr(error)
                log(f"NVML unavailable, no device cross-check: {device_error}")
        child = subprocess.Popen(list(command), env=None if env is None else dict(env))
        guard = VramGuard(
            psutil.Process(child.pid),
            sample,
            device_used=device_used,
            soft_limit=soft_limit,
            hard_limit=hard_limit,
            interval=interval,
            command=command,
            log=log,
        )
        guard.report.device_error = guard.report.device_error or device_error
        with guard:
            guard.report.returncode = child.wait()
        return guard.report


def child_environment(unhide_gpu: bool) -> dict[str, str]:
    """This process's environment, without HIDDEN when unhide_gpu is set."""
    return {
        key: value
        for key, value in os.environ.items()
        if not (unhide_gpu and key in HIDDEN)
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a command under the VRAM guard (soft 2.5 GB, hard 3 GB)."
    )
    parser.add_argument("--json", type=Path, help="write the GuardReport here")
    parser.add_argument(
        "--unhide-gpu",
        action="store_true",
        help="remove CUDA_VISIBLE_DEVICES and VK_LOADER_DRIVERS_DISABLE for the child",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    command = arguments.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("no command given")
    report = run_guarded(command, env=child_environment(arguments.unhide_gpu))
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(asdict(report), indent=2), "utf-8")
    _log(
        f"exit {report.returncode}; peak {report.peak} bytes over {report.polls} "
        f"polls; NVML delta peak {report.device_delta_peak}; "
        f"soft warned {report.soft_warned}; aborted {report.aborted}"
    )
    return ABORTED if report.aborted else int(report.returncode or 0)


if __name__ == "__main__":
    sys.exit(main())
