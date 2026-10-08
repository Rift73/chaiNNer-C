"""Process setup shared by every test in this directory."""

import os

# Hides every Vulkan driver from ncnn, as native/tests/conftest.py does: the
# backend's ncnn modules call ncnn.get_gpu_count() on import, which would otherwise
# create a Vulkan instance that crashes the test process at exit. Set before any
# test module loads; the processes tests start inherit it.
os.environ["VK_LOADER_DRIVERS_DISABLE"] = "*"
