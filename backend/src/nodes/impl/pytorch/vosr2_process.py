"""Owned compiler process lifetime, using the existing runtime verifier job pattern."""

import ctypes
import os
import signal
import subprocess
import sys
import time
from ctypes import wintypes

# prctl's option that signals a process when the thread that started it ends
PR_SET_PDEATHSIG = 1

# The environment variable in which a process that starts a compiler on Linux gives it
# its pid, for end_with_parent
PARENT_PID_VARIABLE = "CHAINNER_PARENT_PID"


def end_with_parent():
    """On Linux, have the kernel kill this compiler process once the thread that
    started it ends (that thread waits for it), as Windows' job does when its worker
    dies, and exit at once if that process has already ended: the kernel signals only
    for a parent that ends after this call, and the parent names itself in
    PARENT_PID_VARIABLE for the check. A compiler calls it before the imports that take
    seconds; a no-op elsewhere."""
    if sys.platform != "linux":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    parent = os.environ.pop(PARENT_PID_VARIABLE, None)
    if parent is not None and os.getppid() != int(parent):
        sys.exit(f"The process that started this compiler (pid {parent}) has ended.")


class OwnedJob:
    """Kill-on-close job for one VOSR compiler and its descendants: a Windows job
    object; elsewhere the process group the compiler leads (Popen's
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

    def assign(self, process: subprocess.Popen):
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
        if not assigned:
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
                    raise RuntimeError("Unexpected VOSR compiler process count.")
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
                        raise TimeoutError("VOSR compiler process tree did not exit.")
            finally:
                for handle in handles:
                    self.kernel.CloseHandle(handle)
                self.kernel.CloseHandle(self.handle)
                self.handle = None
