"""Process setup shared by every test in this directory."""

import os
import sys

# Hides every Vulkan driver from ncnn, as native/tests/conftest.py does: the
# backend's ncnn modules call ncnn.get_gpu_count() on import, which would otherwise
# create a Vulkan instance that crashes the test process at exit. Set before any
# test module loads; the processes tests start inherit it.
os.environ["VK_LOADER_DRIVERS_DISABLE"] = "*"


def pytest_collection_modifyitems(items):
    # The comparisons whose reference library is built otherwise on Linux
    # (native/tools/linux_differences.py)
    if sys.platform == "linux":
        import linux_differences

        linux_differences.skip_listed(items)
