from __future__ import annotations

import os
import platform
import re
import sys

is_mac = sys.platform == "darwin"
is_arm_mac = is_mac and platform.machine() == "arm64"
is_windows = sys.platform == "win32"
is_linux = sys.platform == "linux"


def glibc_version() -> tuple[int, int] | None:
    """The (major, minor) version of the glibc this process runs on; None where there is
    none (Windows, macOS, a musl system)."""
    if sys.platform != "linux" or "CS_GNU_LIBC_VERSION" not in os.confstr_names:
        return None
    version = os.confstr("CS_GNU_LIBC_VERSION") or ""
    match = re.fullmatch(r"glibc (\d+)\.(\d+)(?:\..*)?", version)
    return (int(match[1]), int(match[2])) if match else None


def load_system_cpp_runtime() -> None:
    """Loads Windows' own msvcp140.dll by its full path; elsewhere, nothing.

    The first msvcp140.dll a process loads serves every later module. Pillow bundles
    version 14.29, and Torch, ONNX Runtime and other builds from newer compilers fail to
    initialize against it (WinError 1114). The worker imports the node modules that use
    Pillow before the frameworks, which load in the background, so it loads the system
    copy first. Raises OSError if Windows has none (no Visual C++ runtime installed).
    """
    if not is_windows:
        return

    import ctypes
    import os

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetSystemDirectoryW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint]
    buffer = ctypes.create_unicode_buffer(32768)
    if not kernel32.GetSystemDirectoryW(buffer, len(buffer)):
        raise ctypes.WinError(ctypes.get_last_error())
    ctypes.WinDLL(os.path.join(buffer.value, "msvcp140.dll"))
