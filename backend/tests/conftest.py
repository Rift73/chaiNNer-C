"""Process setup shared by every test in this directory."""

from __future__ import annotations

import importlib.util
import os
import sys
from importlib import metadata

import pytest

# Hides every Vulkan driver from ncnn, as native/tests/conftest.py does: the
# backend's ncnn modules call ncnn.get_gpu_count() on import, which would otherwise
# create a Vulkan instance that crashes the test process at exit. Set before any
# test module loads; the processes tests start inherit it.
os.environ["VK_LOADER_DRIVERS_DISABLE"] = "*"


def _triton_rocm_installed() -> bool:
    try:
        metadata.version("triton-rocm")
    except metadata.PackageNotFoundError:
        return False
    return True


# ROCm's Triton, which spandrel imports, and VapourSynth's akarin plugin each carry
# LLVM, which registers its options once per process: a VapourSynth core created after
# Triton has loaded aborts the process. With ROCm's Triton, the core is created here,
# before any test module loads, as a script creates it before AI Upscale loads a model.
# (VapourSynth by name: the Windows runtime the type checker runs on has none.)
if importlib.util.find_spec("vapoursynth") is not None and _triton_rocm_installed():
    importlib.import_module("vapoursynth").core.plugins()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # The comparisons whose reference library is built otherwise on Linux
    # (native/tools/linux_differences.py)
    if sys.platform == "linux":
        import linux_differences

        linux_differences.skip_listed(items)
