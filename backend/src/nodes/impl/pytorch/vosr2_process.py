"""Owned compiler process lifetime on Linux: end_with_parent, which a compiler calls
first. Its job is owned_job.OwnedJob, which this module does not import: a compiler
asks for the signal before its slow imports."""

import ctypes
import os
import signal
import sys

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
