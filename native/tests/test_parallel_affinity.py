"""Kernel bits must not depend on the CPU count or on overlapping callers.

Each child process restricts itself to a CPU set before the native pool starts,
runs a sample of kernels alone and under overlapping callers, and prints one
digest. The sample is elementwise work (adjust, blend, clip), separable row
filters (gaussian, separable box, box blur), a per-row scan combined serially
(bounding box) and kernels whose vector or scalar path depends on where a chunk
ends: convolve always, the separable filters when a chunk ends mid-vector, which
the five-CPU mask forces on a pool that sizes chunks by CPU count. None of them
consults an engine thread count.
"""

import ctypes as ct
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"
MASKS = (0x1, 0x3, 0xF, 0x1F, 0xFF, 0)  # 1, 2, 4, 5 and 8 CPUs, then all of them
CHILD = """
import ctypes, hashlib, sys
from concurrent.futures import ThreadPoolExecutor

mask = int(sys.argv[1], 0)
kernel32 = ctypes.WinDLL("kernel32")
kernel32.GetCurrentProcess.restype = ctypes.c_void_p
kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
if mask and not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
    raise OSError(f"cannot set affinity mask {mask:#x}")

import numpy as np
from nodes.impl.blend import BlendMode, blend_images
from nodes.impl.native import lib
from nodes.impl.native_adjustments import adjust
from nodes.impl.native_box import separable_box
from nodes.impl.native_channels import bounding_box
from nodes.impl.native_convolution import box_blur, convolve
from nodes.impl.native_framework_shared import clip_image
from nodes.impl.native_gaussian import gaussian

rng = np.random.default_rng(20261002)
image = rng.random((384, 512, 3), dtype=np.float32)
other = rng.random(image.shape, dtype=np.float32)
kernel = rng.random((5, 5), dtype=np.float32)
content = np.zeros((384, 512), np.float32)
content[37:301, 59:444] = 1


def sample():
    digest = hashlib.sha256()
    for result in (
        adjust(image, 1, 0.75),
        blend_images(image, other, BlendMode(0)),
        gaussian(image, 3.0, 3.0),
        gaussian(image, 6.0, 6.0),
        separable_box(image, 2.5, 3.5),
        box_blur(image, 3, 2),
        convolve(image, kernel, 2),
        clip_image(image * np.float32(2) - np.float32(0.5)),
        np.asarray(bounding_box(content, 0.5), np.int64),
    ):
        digest.update(np.ascontiguousarray(result).tobytes())
    return digest.hexdigest()


dll = lib()
dll.cn_parallel_capacity.restype = ctypes.c_uint
dll.cn_parallel_dispatches.restype = ctypes.c_uint64
before = dll.cn_parallel_dispatches()
alone = sample()
dispatched = dll.cn_parallel_dispatches() - before
with ThreadPoolExecutor(max_workers=8) as executor:
    overlapped = set(executor.map(lambda _: sample(), range(16)))
print(dll.cn_parallel_capacity(), alone, len(overlapped | {alone}), dispatched)
"""

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="CPU sets are set through the Windows affinity mask"
)


def process_mask():
    kernel32 = ct.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = ct.c_void_p
    kernel32.GetProcessAffinityMask.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_size_t),
        ct.POINTER(ct.c_size_t),
    ]
    process, system = ct.c_size_t(), ct.c_size_t()
    if not kernel32.GetProcessAffinityMask(
        kernel32.GetCurrentProcess(), ct.byref(process), ct.byref(system)
    ):
        raise ct.WinError()
    return process.value


def run(mask):
    done = subprocess.run(
        [sys.executable, "-B", "-c", CHILD, hex(mask)],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    capacity, digest, distinct, dispatched = done.stdout.split()[-4:]
    return int(capacity), digest, int(distinct), int(dispatched)


def test_kernel_bits_do_not_depend_on_cpu_count_or_overlap():
    allowed = process_mask()
    results = {mask: run(mask) for mask in MASKS if mask & allowed == mask}
    if len({capacity for capacity, _digest, _distinct, _calls in results.values()}) < 2:
        pytest.skip("needs at least two different CPU counts to compare")
    for mask, (capacity, _digest, distinct, dispatched) in results.items():
        assert capacity == (mask or allowed).bit_count()
        # The sample reached the pool, and overlapping callers computed exactly
        # the digest computed alone.
        assert dispatched > 0
        assert distinct == 1
    assert (
        len({digest for _capacity, digest, _distinct, _calls in results.values()}) == 1
    )
    print("kernel sample digest:", results[0][1])
