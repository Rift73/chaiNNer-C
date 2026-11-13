"""Process setup shared by every test in this directory."""

import ctypes as ct
import importlib
import os
import sys
from pathlib import Path

import pytest

# Hides every Vulkan driver from ncnn, as CUDA_VISIBLE_DEVICES=-1 hides CUDA:
# the port's ncnn modules call ncnn.get_gpu_count() on
# import, which would otherwise create a Vulkan instance that crashes the test
# process at exit and can leave a clone holding DLLs. Set before any test module
# loads; the backend hosts tests start inherit it. Only the test environment and
# the runtime verifiers' hosts set it, never the package, the bench or a launch.
os.environ["VK_LOADER_DRIVERS_DISABLE"] = "*"

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"


def system_cpp_runtime():
    """Path of msvcp140.dll in the system directory Windows reports."""
    kernel32 = ct.WinDLL("kernel32", use_last_error=True)
    kernel32.GetSystemDirectoryW.argtypes = [ct.c_wchar_p, ct.c_uint]
    buffer = ct.create_unicode_buffer(32768)
    if not kernel32.GetSystemDirectoryW(buffer, len(buffer)):
        raise ct.WinError(ct.get_last_error())
    return Path(buffer.value, "msvcp140.dll")


def load_system_cpp_runtime():
    """Make the system msvcp140.dll the C++ runtime of the test process.

    The first msvcp140.dll a process loads serves every later module. Pillow 9.2
    bundles version 14.29, and Torch, ONNX Runtime and other builds from newer
    compilers fail to initialize against it (WinError 1114). The backend worker
    loads the system copy first through its import order; test collection has no
    such order, so it is loaded here. A missing system copy is an error, never a
    silent return to the broken order.
    """
    ct.WinDLL(str(system_cpp_runtime()))


if sys.platform == "win32":
    load_system_cpp_runtime()


@pytest.fixture(scope="session")
def system_runtime():
    return system_cpp_runtime()


def pytest_collection_modifyitems(items):
    # The comparisons whose reference library is built otherwise on Linux
    if sys.platform == "linux":
        import linux_differences

        linux_differences.skip_listed(items)


@pytest.fixture(scope="module")
def schemas():
    """The standard package's nodes, registered as the worker registers them; their schemas as /nodes gives them."""
    from engine_differential import load_registry_errors

    import api

    main = sys.modules["__main__"]
    previous = getattr(main, "__file__", None)
    # Caption loads its font from beside __main__'s file, the app's run.py.
    main.__file__ = str(BACKEND / "run.py")
    importlib.import_module("packages.chaiNNer_standard")
    errors = load_registry_errors(BACKEND / "server.py")
    assert not errors, [e.module for e in errors]
    yield {
        schema: {"inputs": [i.to_dict() for i in data.inputs], "kind": data.kind}
        for schema, (data, _) in api.registry.nodes.items()
    }
    if previous is None:
        del main.__file__
    else:
        main.__file__ = previous


@pytest.fixture
def levels():
    """Every level this CPU has, scalar first; the original level afterwards."""
    from test_isa_dispatch import isa_get, isa_set

    effective, _, cpu = isa_get()
    try:
        yield range(cpu + 1)
    finally:
        assert isa_set(effective) == effective
