"""The shared scratch pool (SP4b D10): bounded reuse across morphology and resample.

cn_scratch_info reports (allocations, temporary leases, idle retained bytes, leased
slots): every lease that allocated (a slot grown, or a temporary), the leases served by
a temporary because every slot was busy, the idle slots' capacities, and the slots
leased now. No test reads a timing.
"""

from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from nodes.impl import native, native_filters, native_resample

RETAIN_LIMIT = 64 << 20  # 4 slots of 16 MiB
SLOT_RETAIN_BYTES = 16 << 20


def scratch_info():
    function = native.lib()["cn_scratch_info"]
    function.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    function.restype = ctypes.c_int
    values = (ctypes.c_uint64 * 4)()
    assert function(values, 4) == 0
    return dict(
        zip(("allocations", "temporary", "idle_bytes", "leased"), values, strict=True)
    )


def image(shape, seed):
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


def test_repeated_morphology_reuses_pooled_scratch():
    """A second identical call leases the slot the first one grew: no allocation."""
    source = image((384, 512, 3), 41)
    for shape, maximum in ((cv2.MORPH_ELLIPSE, True), (cv2.MORPH_CROSS, False)):
        first = native_filters.morphology(source, shape, 3, 2, maximum=maximum)
        before = scratch_info()
        second = native_filters.morphology(source, shape, 3, 2, maximum=maximum)
        after = scratch_info()
        assert second.tobytes() == first.tobytes()
        assert after["allocations"] == before["allocations"], shape
        assert after["temporary"] == before["temporary"], shape
        assert after["leased"] == 0
        assert after["idle_bytes"] <= RETAIN_LIMIT


def test_oversized_lease_is_not_retained():
    """A resample whose intermediate (2111 rows x 2048 columns) exceeds a slot's 16 MiB
    retention is allocated again on every call and freed on release."""
    source = image((2111, 1, 1), 43)
    assert 2111 * 2048 * 4 > SLOT_RETAIN_BYTES
    expected = native_resample.filtered(source, (2048, 3), 2, False)
    for _ in range(2):
        before = scratch_info()
        actual = native_resample.filtered(source, (2048, 3), 2, False)
        after = scratch_info()
        assert actual.tobytes() == expected.tobytes()
        assert after["allocations"] == before["allocations"] + 1
        assert after["leased"] == 0
        assert after["idle_bytes"] <= RETAIN_LIMIT


def test_concurrent_pool_users_complete_with_bounded_retention():
    """8 threads run Dilate's ellipse, Erode's cross and a Hermite resize 5 times each,
    more leases at once than the pool's 4 slots (the rest are temporaries): every output
    equals one serial run, and afterwards no slot is leased and at most 64 MiB is
    retained."""
    source = image((192, 256, 3), 47)

    def work():
        return (
            native_filters.morphology(source, cv2.MORPH_ELLIPSE, 3, 2, maximum=True),
            native_filters.morphology(source, cv2.MORPH_CROSS, 2, 2, maximum=False),
            native_resample.filtered(source, (128, 96), 5, False),
        )

    expected = [array.tobytes() for array in work()]

    def repeat(_):
        return [[array.tobytes() for array in work()] for _ in range(5)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(repeat, range(8)))
    assert all(run == expected for runs in results for run in runs)
    info = scratch_info()
    assert info["leased"] == 0
    assert info["idle_bytes"] <= RETAIN_LIMIT


def test_lease_size_overflow_fails_like_b3():
    """A lease whose 64-byte-rounded sum overflows size_t returns CN_ALLOCATION_FAILED
    (3), the status B3's first malloc (2**62 bytes) returned for the same call, before
    any read of the 1-float buffers; no slot stays leased."""
    function = native.lib()["cn_filter_morphology"]
    function.argtypes = (
        [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 5 + [ctypes.c_int] * 2
    )
    function.restype = ctypes.c_int
    src = np.zeros(1, np.float32)
    out = np.full(1, 0x7FC0DEAD, np.uint32)
    assert function(src.ctypes.data, out.ctypes.data, 2**30, 2**30, 1, 2, 2, 1, 1) == 3
    assert out[0] == 0x7FC0DEAD
    assert scratch_info()["leased"] == 0


def test_scratch_info_checks_its_arguments():
    """count must be 4 and values non-null (CN_INVALID_ARGUMENT, 1); a refused call
    writes nothing."""
    function = native.lib()["cn_scratch_info"]
    function.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    function.restype = ctypes.c_int
    values = (ctypes.c_uint64 * 5)(*([917] * 5))
    assert function(None, 4) == 1
    assert function(values, 3) == 1
    assert function(values, 5) == 1
    assert tuple(values) == (917,) * 5
