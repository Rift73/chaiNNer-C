"""The process C++ runtime: Pillow's bundled msvcp140.dll must never load first.

The first msvcp140.dll a process loads serves every later module. Pillow 9.2
bundles version 14.29; Torch, ONNX Runtime and other builds from newer compilers
fail to initialize against it (WinError 1114), and Windows records each failure
as a crash and keeps a clone of the process that locks its DLLs.
"""

import ctypes as ct
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="msvcp140.dll is the Windows C++ runtime"
)

# The first steps of server.import_packages. The torch import runs only when
# nothing loaded Pillow or a C++ runtime before it: after Pillow's copy it would
# fail, and Windows would log a crash and keep a clone that locks DLLs.
WORKER_IMPORTS = """
import ctypes, importlib, importlib.util, sys
import server


def runtime():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    handle = kernel32.GetModuleHandleW("msvcp140.dll")
    if not handle:
        return ""
    buffer = ctypes.create_unicode_buffer(32768)
    if not kernel32.GetModuleFileNameW(handle, buffer, len(buffer)):
        raise ctypes.WinError(ctypes.get_last_error())
    return buffer.value


importlib.import_module("packages.chaiNNer_standard")
early = "PIL._imaging" in sys.modules or bool(runtime())
print("early", early)
if not early and importlib.util.find_spec("torch") is not None:
    # A Pillow import ahead of torch inside the package fails here as an ImportError.
    sys.modules["PIL"] = None
    importlib.import_module("packages.chaiNNer_pytorch")
    print("runtime", runtime())
"""


def loaded_runtime():
    kernel32 = ct.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.restype = ct.c_void_p
    kernel32.GetModuleHandleW.argtypes = [ct.c_wchar_p]
    kernel32.GetModuleFileNameW.argtypes = [ct.c_void_p, ct.c_wchar_p, ct.c_uint32]
    handle = kernel32.GetModuleHandleW("msvcp140.dll")
    if not handle:
        return None
    buffer = ct.create_unicode_buffer(32768)
    if not kernel32.GetModuleFileNameW(handle, buffer, len(buffer)):
        raise ct.WinError(ct.get_last_error())
    return Path(buffer.value)


def test_pillow_runs_on_the_system_cpp_runtime(system_runtime):
    importlib.import_module("PIL._imaging")
    runtime = loaded_runtime()
    assert runtime is not None
    assert runtime.samefile(system_runtime)


def test_worker_loads_the_system_cpp_runtime_before_any_node_module(system_runtime):
    # The worker has no preload. It is safe because nothing up to and including
    # packages.chaiNNer_standard loads Pillow or a C++ runtime, and the package-level
    # `import torch` of packages.chaiNNer_pytorch then loads msvcp140.dll by name:
    # the system copy. Node modules, Pillow's users among them, load afterwards.
    has_torch = importlib.util.find_spec("torch") is not None
    done = subprocess.run(
        [sys.executable, "-B", "-c", WORKER_IMPORTS],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert "early False" in lines
    if has_torch:
        loaded = [
            line.removeprefix("runtime ")
            for line in lines
            if line.startswith("runtime ")
        ]
        assert len(loaded) == 1
        assert loaded[0], "packages.chaiNNer_pytorch loaded no msvcp140.dll"
        assert Path(loaded[0]).samefile(system_runtime)
