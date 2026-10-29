"""chaiNNer-C's native MIGraphX runner (native/migraphx_runner): loading its module,
_chainner_migraphx, which compiles, saves and loads programs, registers custom-op libraries
and runs programs (session.py). The DUAL, DRFT and VOSR routes and the compile worker load it
here.

MIGraphX itself is not among AMD's ROCm pip packages: its libraries ship inside chaiNNer-C's
ONNX Runtime MIGraphX build (onnxruntime-migraphx, in onnxruntime/capi/, linked to the ROCm pip
packages beside it), and the module, which links them by name, loads after them (load_migraphx).
On Windows, whose DLLs have no run path, the ROCm pip packages' DLL folders join the DLL search
path first. The package needs the module: a build without it has no MIGraphX nodes (the package
says so). Importing this module loads nothing; the package loads MIGraphX at start once its
dependencies are installed."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import inspect
import os
import sys
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType

MODULE = "nodes.impl.migraphx._chainner_migraphx"
# The distribution whose onnxruntime/capi/ holds MIGraphX's libraries
MIGRAPHX_DISTRIBUTION = "onnxruntime-migraphx"
MODULE_FILE = (
    "_chainner_migraphx.pyd" if sys.platform == "win32" else "_chainner_migraphx.so"
)
# Windows: the ROCm pip packages whose bin folders hold the DLLs MIGraphX's link to (HIP,
# hipRTC, MIOpen, rocBLAS, hipBLASLt and theirs); a GPU kernel package unpacks into the
# libraries' folder
ROCM_DLL_PACKAGES = ("_rocm_sdk_core", "_rocm_sdk_libraries")


def runner_built() -> bool:
    """Whether this build has the runner's module (without loading it)."""
    return find_spec(MODULE) is not None


def migraphx_libraries() -> list[Path]:
    """MIGraphX's libraries in the installed ONNX Runtime MIGraphX build, or an ImportError
    that says what is missing."""
    spec = find_spec("onnxruntime")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "MIGraphX's libraries come with ONNX Runtime's MIGraphX build"
            f" ({MIGRAPHX_DISTRIBUTION}), which is not installed: install the MIGraphX"
            " package's dependencies."
        )
    capi = Path(spec.submodule_search_locations[0]) / "capi"
    # Windows: migraphx.dll, migraphx_gpu.dll and the others
    pattern = "migraphx*.dll" if sys.platform == "win32" else "libmigraphx*.so*"
    libraries = sorted(capi.glob(pattern))
    if not libraries:
        raise ImportError(
            f"The installed ONNX Runtime ({capi.parent}) has no MIGraphX libraries: install"
            f" the MIGraphX package's dependencies ({MIGRAPHX_DISTRIBUTION})."
        )
    return libraries


def rocm_dll_directories() -> list[Path]:
    """Windows: the bin folders of the installed ROCm pip packages (ROCM_DLL_PACKAGES),
    or an ImportError that says which is missing."""
    folders: list[Path] = []
    for package in ROCM_DLL_PACKAGES:
        spec = find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            raise ImportError(
                f"MIGraphX needs ROCm's {package} (rocm-sdk-core and rocm-sdk-libraries),"
                " which is not installed: install the MIGraphX package's dependencies."
            )
        folders.append(Path(spec.submodule_search_locations[0]) / "bin")
    return folders


def load_migraphx_libraries() -> None:
    """Load MIGraphX's libraries, so that references to them by name resolve (the
    runner's module's, and on Windows ONNX Runtime's MIGraphX provider's). Each finds its
    own (HIP and the ROCm libraries) through its run path. Windows has none: the ROCm
    packages' folders join the DLL search path, which the module and the custom-op
    libraries use too, and lead PATH after MIGraphX's own folder (each folder once), for
    what searches the standard way: MIGraphX loading a target by its bare name (one of
    the libraries loaded here) and its kernel compiles in a child process
    (migraphx-hiprtc-driver.exe beside the libraries), which inherits PATH and not this
    process's DLLs. They load locally: their bundled msgpack, protobuf and SQLite symbols,
    made global, would take the place of Triton's and PyTorch's in this process (a crash
    at Triton's import). A missing library raises (OSError or ImportError) with its
    name."""
    libraries = migraphx_libraries()
    if sys.platform == "win32":
        folders = [str(libraries[0].parent), *map(str, rocm_dll_directories())]
        for folder in folders[1:]:
            os.add_dll_directory(folder)
        path = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
        os.environ["PATH"] = os.pathsep.join(
            [*folders, *(p for p in path if p not in folders)]
        )
    for library in libraries:
        ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)


@lru_cache(maxsize=1)
def load_migraphx() -> ModuleType:
    """The runner's binding, _chainner_migraphx, without any custom-op library, after
    MIGraphX's libraries (load_migraphx_libraries): what the program worker and the DUAL
    and DRFT routes' load_ops use."""
    load_migraphx_libraries()
    return import_module(MODULE)


@lru_cache(maxsize=1)
def runner_identity() -> str:
    """The MIGraphX version the runner was built against, the SHA-256 of its module and
    the version of the build MIGraphX's libraries come from, for Build Program's cache:
    programs of another runner or MIGraphX are not reused."""
    module = load_migraphx()
    digest = hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
    libraries = importlib.metadata.version(MIGRAPHX_DISTRIBUTION)
    return f"{module.MIGRAPHX_VERSION} {digest} {libraries}"
