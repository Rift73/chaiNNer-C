"""Child processes that end with their owner: OwnedJob, the kill-on-close job of one
child and its descendants (a compiler), using the existing runtime verifier job
pattern; and worker_child, a child of the worker that ends when the worker does, also
when the worker is killed or crashes (vspipe, the VapourSynth scan)."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes
from typing import Any

import psutil
from sanic.log import logger

CREATE_SUSPENDED = 0x00000004

# Linux: the first program of a worker_child, run by this Python. It has the kernel
# kill the process once the thread that started it ends (prctl's PR_SET_PDEATHSIG,
# which exec keeps), exits at once if the worker named in argv[1] has already ended
# (the kernel signals only for an end after the request), and then becomes the child.
_LINUX_LAUNCHER = (
    "import ctypes, os, signal, sys\n"
    "libc = ctypes.CDLL(None, use_errno=True)\n"
    "if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:\n"
    "    sys.exit(os.strerror(ctypes.get_errno()))\n"
    "if os.getppid() != int(sys.argv[1]):\n"
    "    sys.exit(f'The worker (pid {sys.argv[1]}) that started it has ended.')\n"
    "os.execv(sys.argv[2], sys.argv[2:])\n"
)


class OwnedJob:
    """Kill-on-close job for one child process (a VOSR compiler) and its descendants:
    a Windows job object; elsewhere the process group the child leads (Popen's
    start_new_session), killed on close."""

    def __init__(self):
        if os.name != "nt":
            self.group = None
            return

        class IoCounters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class BASIC(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class EXTENDED(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self.kernel.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.IsProcessInJob.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        ]
        self.kernel.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        self.kernel.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = (
            0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not self.kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process: subprocess.Popen, soft: bool = False):
        """Puts the process in the job. When Windows refuses, the process is ended and
        the error raised; with `soft`, the refusal is logged and the process runs on
        outside the job."""
        if os.name != "nt":
            self.group = process.pid
            return
        # Popen still holds its own handle, so Windows cannot reuse the pid: the
        # handle opened here names the same process.
        handle = self.kernel.OpenProcess(
            0x0101,  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
            False,
            process.pid,
        )
        assigned = bool(handle) and self.kernel.AssignProcessToJobObject(
            self.handle, handle
        )
        error = ctypes.get_last_error()
        if handle:
            self.kernel.CloseHandle(handle)
        if not assigned and soft:
            logger.warning(
                "Process %d runs outside its job, so it can outlive its owner: %s",
                process.pid,
                ctypes.WinError(error),
            )
        elif not assigned:
            process.terminate()
            process.wait(timeout=10)
            raise ctypes.WinError(error)

    def close(self):
        if os.name != "nt":
            group, self.group = self.group, None
            if group is not None:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # every process of the group had already exited
            return
        if self.handle:

            class ProcessList(ctypes.Structure):
                _fields_ = [
                    ("assigned", wintypes.DWORD),
                    ("count", wintypes.DWORD),
                    ("pids", ctypes.c_size_t * 128),
                ]

            handles = []
            try:
                info = ProcessList()
                if not self.kernel.QueryInformationJobObject(
                    self.handle, 3, ctypes.byref(info), ctypes.sizeof(info), None
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                if info.count > 128:
                    raise RuntimeError("Unexpected count of the job's processes.")
                for pid in info.pids[: info.count]:
                    handle = self.kernel.OpenProcess(0x100400, False, pid)
                    if not handle:
                        error = ctypes.get_last_error()
                        if error == 87:  # Process exited before opening its handle.
                            continue
                        raise ctypes.WinError(error)
                    member = wintypes.BOOL()
                    if not self.kernel.IsProcessInJob(
                        handle, self.handle, ctypes.byref(member)
                    ):
                        error = ctypes.get_last_error()
                        self.kernel.CloseHandle(handle)
                        raise ctypes.WinError(error)
                    if member.value:
                        handles.append(handle)
                    else:
                        self.kernel.CloseHandle(handle)
                if not self.kernel.TerminateJobObject(self.handle, 1):
                    raise ctypes.WinError(ctypes.get_last_error())
                # A zero job-process count can precede handle teardown. Wait on
                # process handles captured before termination, immune to PID reuse.
                deadline = time.monotonic() + 20
                for handle in handles:
                    remaining = max(0, int((deadline - time.monotonic()) * 1000))
                    result = self.kernel.WaitForSingleObject(handle, remaining)
                    if result == 0xFFFFFFFF:
                        raise ctypes.WinError(ctypes.get_last_error())
                    if result != 0:
                        raise TimeoutError("The job's processes did not exit.")
            finally:
                for handle in handles:
                    self.kernel.CloseHandle(handle)
                self.kernel.CloseHandle(self.handle)
                self.handle = None


# Windows: the job of every worker_child, one per worker and never closed: the system
# closes its handle when the worker ends, which ends the job's processes.
_worker_job: OwnedJob | None = None
_worker_job_lock = threading.Lock()
# Linux: the thread that starts every worker_child. The kernel kills a child when the
# thread that started it ends, and this one lives as long as the worker; a run's thread
# can end while another thread still reads its child (Clip Frames' frames).
_linux_starter = ThreadPoolExecutor(max_workers=1, thread_name_prefix="worker children")


def _job() -> OwnedJob:
    global _worker_job
    with _worker_job_lock:
        if _worker_job is None:
            _worker_job = OwnedJob()
        return _worker_job


def worker_child(command: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
    """subprocess.Popen(command, **kwargs) for a child of the worker that ends when the
    worker ends, also when the worker is killed or crashes: on Windows it runs in the
    worker's kill-on-close job, which it joins before it runs (a child Windows keeps out
    runs without it, logged); on Linux the kernel kills it when the worker ends."""
    if sys.platform == "linux":
        launcher = [sys.executable, "-I", "-S", "-c", _LINUX_LAUNCHER, str(os.getpid())]
        starting = _linux_starter.submit(
            subprocess.Popen, [*launcher, *command], **kwargs
        )
        return starting.result()
    if os.name != "nt":
        return subprocess.Popen(command, **kwargs)
    flags = kwargs.pop("creationflags", 0) | CREATE_SUSPENDED
    process = subprocess.Popen(command, creationflags=flags, **kwargs)
    try:
        _job().assign(process, soft=True)
        psutil.Process(process.pid).resume()
    except BaseException:
        process.kill()
        process.wait()
        raise
    return process
