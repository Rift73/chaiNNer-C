"""An owned, isolated chaiNNer backend server for benchmark trials."""

from __future__ import annotations

import ctypes
import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
from ctypes import wintypes
from pathlib import Path

from verify_framework_runtime import source_hashes
from verify_runtime import Events, OwnedJob, digest, request, write_json
from verify_video_runtime import provision_ffmpeg

PROJECT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT / "out/chaiNNer-C/python/python/python.exe"
ISOLATION = {
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONIOENCODING": "utf-8",
    "CUDA_VISIBLE_DEVICES": "-1",
    "NVIDIA_VISIBLE_DEVICES": "none",
    "OPENCV_OPENCL_RUNTIME": "disabled",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    # The hosts' dependency installers must never write into an interpreter; pip
    # list ignores it.
    "PIP_REQUIRE_VIRTUALENV": "1",
}
PRIVATE_DIRS = {
    "APPDATA": "appdata",
    "LOCALAPPDATA": "localappdata",
    "USERPROFILE": "profile",
    "HOME": "profile",
    "TMP": "temp",
    "TEMP": "temp",
}
# The DLL's ISA override (spec 4.3). A side that sets it must log its ISA line, or
# it may be a tree that ignores the variable and runs the same code as the other.
ISA_VARIABLE = "CHAINNER_C_ISA"
# The worker's startup line: effective level, requested level and the CPU's maximum.
NATIVE_ISA = re.compile(r"native isa=(\w+) \(requested (\w+), cpu (\w+)\)$")


def site_packages(python: Path) -> dict[str, int]:
    """The top-level entries of python's Lib\\site-packages, by name, with mtime_ns.

    An install, upgrade or removal adds, replaces or removes such an entry, so
    the fingerprint changes when something writes a package into the interpreter.
    It is empty for an interpreter without that directory.
    """
    root = python.parent / "Lib" / "site-packages"
    if not root.is_dir():
        return {}
    with os.scandir(root) as entries:
        return {entry.name: entry.stat().st_mtime_ns for entry in entries}


def log_matches(
    log: Path, offset: int, pattern: re.Pattern[str], wait: float
) -> list[re.Match[str]]:
    """The matches of pattern in the lines of log after offset, in log order.

    With wait > 0, polls every 0.1 s until a line matches or wait seconds have
    passed: the host re-logs each worker line from a pipe, so a line can reach
    the log after the event that caused it.
    """
    limit = time.monotonic() + wait
    while True:
        with log.open("rb") as stream:
            stream.seek(offset)
            text = stream.read().decode("utf-8", errors="replace")
        matches = [
            found for line in text.splitlines() if (found := pattern.search(line))
        ]
        if matches or time.monotonic() >= limit:
            return matches
        time.sleep(0.1)


class BasicAccounting(ctypes.Structure):
    """JOBOBJECT_BASIC_ACCOUNTING_INFORMATION; times are in 100 ns units."""

    _fields_ = [
        ("TotalUserTime", wintypes.LARGE_INTEGER),
        ("TotalKernelTime", wintypes.LARGE_INTEGER),
        ("ThisPeriodTotalUserTime", wintypes.LARGE_INTEGER),
        ("ThisPeriodTotalKernelTime", wintypes.LARGE_INTEGER),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


def job_accounting(job: OwnedJob) -> BasicAccounting:
    """The owned job's JOBOBJECT_BASIC_ACCOUNTING_INFORMATION."""
    if job.handle is None:
        # A NULL handle would query the calling process's own job instead.
        raise RuntimeError("The owned job is already closed")
    kernel = job.kernel
    kernel.QueryInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    kernel.QueryInformationJobObject.restype = ctypes.c_int
    info = BasicAccounting()
    if not kernel.QueryInformationJobObject(
        job.handle,
        1,  # JobObjectBasicAccountingInformation
        ctypes.byref(info),
        ctypes.sizeof(info),
        None,
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return info


class BenchEvents(Events):
    def collect(self) -> None:
        # The server can be idle for minutes during long encodes and validation.
        self.connection.timeout = 600
        super().collect()


class Backend:
    """One backend tree served from private dirs inside a kill-on-close job.

    env holds this side's own variables, applied last and recorded as
    info["env"]; info["native_isa"] is the tree's ISA line, or None. python is
    the interpreter that runs the tree's run.py with -B, PYTHON unless given,
    recorded as info["python"]; info["interpreter_unchanged"] compares its
    site_packages() fingerprint before the start and after the close.
    """

    def __init__(
        self,
        label: str,
        source: Path,
        root: Path,
        ffmpeg: Path,
        ffprobe: Path,
        env: dict[str, str],
        python: Path = PYTHON,
    ) -> None:
        self.label, self.source, self.env = label, source, env
        self.python = python
        self.directory = root / label
        self.directory.mkdir()
        for name in set(PRIVATE_DIRS.values()) | {"storage"}:
            (self.directory / name).mkdir()
        provision_ffmpeg(self.directory / "storage", ffmpeg, ffprobe)
        self.job = OwnedJob()
        self.process: subprocess.Popen | None = None
        self.events: BenchEvents | None = None
        self.log = None
        self.port = 0
        self.schemas: dict[str, dict] = {}
        self.protected = source_hashes(source)
        self.interpreter = site_packages(python)
        self.info: dict = {
            "source": str(source),
            "source_digest": digest(self.protected),
            "env": env,
            "python": str(python),
        }

    def __enter__(self) -> Backend:
        env = {
            k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}
        }
        env.update(ISOLATION)
        env.update(
            {key: str(self.directory / name) for key, name in PRIVATE_DIRS.items()}
        )
        # Last, so the side's own values win (bench.parse_env refuses the names above).
        env.update(self.env)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        command = [
            str(self.python),
            "-B",
            str(self.source / "run.py"),
            str(self.port),
            "--storage-dir",
            str(self.directory / "storage"),
        ]
        self.info["command"] = command
        try:
            self.log = (self.directory / "backend.log").open("wb")
            self.process = subprocess.Popen(
                command,
                cwd=self.directory,
                env=env,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW
                | subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            self.job.assign(self.process)
            self.info["pid"] = self.process.pid
            self._wait_ready()
            if ISA_VARIABLE in self.env:
                found = log_matches(self.directory / "backend.log", 0, NATIVE_ISA, 10)
                if not found:
                    raise RuntimeError(
                        f"Backend {self.label} set {ISA_VARIABLE} but logged no native "
                        "isa line"
                    )
                # The DLL reads any value but scalar, avx2 or avx512 as auto, so a
                # request that did not take effect would run the same code as the
                # other side.
                value = self.env[ISA_VARIABLE]
                if found[0][2] != value:
                    raise RuntimeError(
                        f"Backend {self.label} set {ISA_VARIABLE}={value!r} but logged "
                        f"{found[0][0]}"
                    )
            metadata = request(self.port, "/nodes", timeout=30)
            write_json(self.directory / "nodes.json", metadata)
            self.schemas = {node["schemaId"]: node for node in metadata["nodes"]}
            self.info["schema_sha256"] = digest(metadata)
            self.events = BenchEvents(self.port)
            if not self.events.ready.wait(10) or self.events.error:
                raise RuntimeError(f"SSE unavailable: {self.events.error}")
            print(f"READY {self.label} {self.source}", flush=True)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _wait_ready(self) -> None:
        process = self.process
        if process is None:
            raise RuntimeError(f"{self.label} has no backend process to wait for")
        log = self.directory / "backend.log"
        last_error: Exception | None = None
        limit = time.monotonic() + 180
        while time.monotonic() < limit:
            if process.poll() is not None:
                raise RuntimeError(
                    f"{self.label} exited with {process.returncode}; see {log}"
                )
            try:
                if request(self.port, "/status", timeout=3).get("ready"):
                    return
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
                # Not listening yet; keep polling until the deadline.
                last_error = error
            time.sleep(0.25)
        raise RuntimeError(
            f"{self.label} startup timeout; last poll error: {last_error!r}; see {log}"
        )

    def cpu_seconds(self) -> float:
        """User plus kernel CPU seconds of every process ever in the owned job.

        The job holds the backend and its FFmpeg children; exited members count.
        """
        info = self._accounting()
        return (info.TotalUserTime + info.TotalKernelTime) / 1e7

    def page_faults(self) -> int:
        """Page faults, soft and hard, of every process ever in the owned job.

        As cpu_seconds, the count covers the backend and its FFmpeg children,
        exited members included. It only grows, so a span's faults are the
        difference of two reads.
        """
        return self._accounting().TotalPageFaultCount

    def _accounting(self) -> BasicAccounting:
        """The owned job's JOBOBJECT_BASIC_ACCOUNTING_INFORMATION."""
        if self.job.handle is None:
            raise RuntimeError(f"{self.label} owned job is already closed")
        return job_accounting(self.job)

    def __exit__(self, *_: object) -> None:
        try:
            if self.process is not None and self.process.poll() is None:
                try:
                    request(self.port, "/shutdown", {}, timeout=15)
                    self.process.wait(timeout=15)
                except (
                    OSError,
                    urllib.error.URLError,
                    http.client.HTTPException,
                    json.JSONDecodeError,
                    subprocess.TimeoutExpired,
                ) as error:
                    # Shutdown may drop its connection; closing the job still ends it.
                    self.info["shutdown_error"] = repr(error)
        finally:
            if self.events is not None:
                self.events.close()
            self.job.close()  # Kill-on-close ends anything shutdown left running.
            if self.process is not None:
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired as error:
                    self.info["exit_wait_error"] = repr(error)
            if self.log is not None:
                self.log.close()
            # The private FFmpeg copy (about 285 MB per side) is not evidence. The
            # closed job has ended every FFmpeg child, so none still maps its files.
            ffmpeg = self.directory / "storage" / "ffmpeg"
            if ffmpeg.is_dir():
                try:
                    shutil.rmtree(ffmpeg)
                except OSError as error:
                    self.info["ffmpeg_cleanup_error"] = repr(error)
                    print(
                        f"WARNING {self.label}: private FFmpeg copy not removed: {error!r}",
                        flush=True,
                    )
            # The first ISA line of the whole log; None for a tree that logs none
            # (B3), and when the backend never opened its log.
            log = self.directory / "backend.log"
            found = log_matches(log, 0, NATIVE_ISA, 0) if log.is_file() else []
            self.info["native_isa"] = (
                dict(zip(("level", "requested", "cpu"), found[0].groups(), strict=True))
                if found
                else None
            )
            self.info["owned_process_exited"] = (
                self.process is None or self.process.poll() is not None
            )
            self.info["owned_job_closed"] = self.job.handle is None
            self.info["backend_unchanged"] = (
                source_hashes(self.source) == self.protected
            )
            self.info["interpreter_unchanged"] = (
                site_packages(self.python) == self.interpreter
            )
            write_json(self.directory / "backend.json", self.info)
