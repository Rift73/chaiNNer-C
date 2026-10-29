"""The process C++ runtime: Pillow's bundled msvcp140.dll must never load first.

The first msvcp140.dll a process loads serves every later module. Pillow 9.2
bundles version 14.29; Torch, ONNX Runtime and other builds from newer compilers
fail to initialize against it (WinError 1114), and Windows records each failure
as a crash and keeps a clone of the process that locks its DLLs.
"""

import ctypes as ct
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="msvcp140.dll is the Windows C++ runtime"
)

# The worker's node loading (server.import_packages):
# it loads the system copy by its path first, then the light node modules (Pillow's
# users among them), then the frameworks.
WORKER_IMPORTS = """
import asyncio, ctypes, os, pathlib, sys, types
import server
from server_config import ServerConfig


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


print("early", "PIL._imaging" in sys.modules or bool(runtime()))
config = ServerConfig(
    port=0,
    close_after_start=False,
    install_builtin_packages=False,
    error_on_failed_node=False,
    storage_dir=None,
    trace=False,
)
context = types.SimpleNamespace(config=config, storage_dir=pathlib.Path(os.environ["STORAGE_DIR"]))
asyncio.run(server.import_packages(context))
print("pillow", "PIL._imaging" in sys.modules)
print("torch", "torch._C" in sys.modules)
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


def test_worker_loads_the_system_cpp_runtime_before_any_node_module(
    system_runtime, tmp_path
):
    # Nothing up to `import server` loads Pillow or a C++ runtime, and the node loading
    # loads the system copy by its path before any node module: the light node modules
    # import Pillow before the frameworks, which then load against the system copy.
    has_torch = importlib.util.find_spec("torch") is not None
    done = subprocess.run(
        [sys.executable, "-B", "-c", WORKER_IMPORTS],
        env={**os.environ, "STORAGE_DIR": str(tmp_path)},
        cwd=BACKEND,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert "early False" in lines
    assert "pillow True" in lines
    if has_torch:
        assert "torch True" in lines, done.stderr
    loaded = [
        line.removeprefix("runtime ") for line in lines if line.startswith("runtime ")
    ]
    assert len(loaded) == 1
    assert loaded[0], "the worker loaded no msvcp140.dll"
    assert Path(loaded[0]).samefile(system_runtime)
