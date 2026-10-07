"""Exact resize parity and bounded scratch/plan reuse, without timing tests."""

from __future__ import annotations

import ctypes as ct
import hashlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from test_resample_complete import FILTERS, compare, exact
from test_resample_ops import image, reference

from nodes.impl import native_resample as native
from nodes.impl import resize as current


def cache_info():
    function = native._api().cn_resample_cache_info  # C boundary
    function.argtypes = [ct.POINTER(ct.c_uint64), ct.c_size_t]
    function.restype = ct.c_int
    values = (ct.c_uint64 * 6)()
    assert function(values, 6) == 0
    return tuple(values)


def counted(source, dimensions, filter_id, gamma=False, separate=True):
    """compare(), returning cache_info() before and after the node path's resize.

    The frozen reference's chainner_ext is chaiNNer-C's C module, whose resize runs
    the same kernel and plan cache, so its call stays outside the counted window.
    """
    before = source.copy()
    with np.errstate(all="ignore"):
        expected = reference.resize(
            source, dimensions, reference.ResizeFilter(filter_id), separate, gamma
        )
        first = cache_info()
        actual = current.resize(
            source, dimensions, current.ResizeFilter(filter_id), separate, gamma
        )
        last = cache_info()
    exact(actual, expected)
    exact(source, before)
    assert not np.shares_memory(actual, source)
    assert actual.flags.writeable
    return first, last


@pytest.mark.parametrize("filter_id", FILTERS)
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("gamma", [False, True])
@pytest.mark.parametrize("separate", [False, True])
def test_repeated_images_reuse_plan_and_scratch(filter_id, channels, gamma, separate):
    compare(image(channels, shape=(29, 31)), (43, 37), filter_id, gamma, separate)
    # A new image must overwrite all reused scratch; changing pixel content is
    # deliberately independent of geometry and coefficient identity.
    before, after = counted(
        image(channels, shape=(29, 31), seed=filter_id + 707),
        (43, 37),
        filter_id,
        gamma,
        separate,
    )
    assert after[0] == before[0]
    assert after[1] == before[1] + 1
    assert after[2] == before[2]
    assert after[3] == before[3]
    assert after[4] <= 64 * 1024 * 1024
    assert after[5] == 0


@pytest.mark.parametrize("filter_id", FILTERS)
def test_channels_gamma_and_alpha_share_coefficients(filter_id):
    compare(image(4, shape=(23, 31)), (41, 37), filter_id, True, False)
    changes = [0, 0, 0]
    for index, channels in enumerate((0, 1, 2, 3, 4)):
        for gamma in (False, True):
            for separate in (False, True):
                before, after = counted(
                    image(
                        channels,
                        ("readonly", "strided", "unaligned")[index % 3],
                        shape=(23, 31),
                        seed=index + 509,
                    ),
                    (41, 37),
                    filter_id,
                    gamma,
                    separate,
                )
                for counter in range(3):
                    changes[counter] += after[counter] - before[counter]
                assert after[5] == 0
    assert changes == [0, 20, 0]


@pytest.mark.parametrize("filter_id", FILTERS)
def test_eviction_and_dirty_scratch_preserve_all_filters(filter_id):
    source = image(4, shape=(11, 13), seed=filter_id)
    expected = compare(source, (71, 67), filter_id, True, False)
    for index in range(7):
        data = image(index % 5, shape=(3 + index, 5 + index), seed=31 + index)
        if index % 2:
            data[::2] = np.nan
        else:
            data[::2] = np.inf
            data[1::2] = -0.0
        compare(
            data,
            (17 + index, 31 - index),
            1 + (filter_id + index) % 11,
            bool(index % 2),
            bool(index % 3),
        )
    actual = compare(source, (71, 67), filter_id, True, False)
    exact(actual, expected)
    assert cache_info()[4] <= 64 * 1024 * 1024


def test_large_scratch_is_released_but_small_plan_is_reused():
    # The separable intermediate is >16 MiB although input and output are tiny.
    # Repeat to prove it is allocated again, not silently retained above the cap.
    source = image(1, shape=(2111, 1))
    compare(source, (2048, 3), 2, True)
    before, after = counted(source, (2048, 3), 2, True)
    assert after[0] == before[0]
    assert after[1] == before[1] + 1
    assert after[2] == before[2] + 1
    assert after[4] <= 64 * 1024 * 1024
    assert after[5] == 0


def test_large_coefficient_plan_is_not_retained():
    # At this destination width, the hash table + line array exceed 16 MiB.
    # Image buffers are small; another identical call must rebuild the plan.
    source = image(1, shape=(1, 1))
    compare(source, (280000, 1), 2)
    before, after = counted(source, (280000, 1), 2)
    assert after[0] == before[0] + 1
    assert after[1] == before[1]
    assert after[4] <= 64 * 1024 * 1024
    assert after[5] == 0


def test_concurrent_identical_and_changing_keys_have_private_scratch():
    barrier = threading.Barrier(8)

    def run(index):
        barrier.wait(timeout=30)
        for step in range(11):
            shape = (257, 263) if step % 2 == 0 else (97 + index, 131 - index)
            dimensions = (269, 271) if step % 2 == 0 else (149 - index, 101 + index)
            compare(
                image(
                    (1, 3, 4)[index % 3],
                    ("readonly", "strided")[index % 2],
                    shape=shape,
                    seed=101 * index + step,
                ),
                dimensions,
                step + 1,
                bool(index % 2),
                bool(step % 2),
            )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(run, range(8)))
    # The bounded cache can be busy without changing output or keeping a lease.
    # Scheduling is deliberately not used to require a particular miss count.
    assert cache_info()[4] <= 64 * 1024 * 1024
    assert cache_info()[5] == 0


# Output hashes under the nearest/down/up/toward-zero FP controls: the real
# chainner_ext 0.3.10's resize (upstream, checked once on 2026-10-06 in each mode).
# They were frozen from the MSVC-built pre-cache C DLL (SHA256 0cce2b50...12f8,
# native/reports/resize-pipeline-20260921/before-src/nodes/impl/), which equals
# upstream except Mitchell (6) and Gauss (11) under the three directed modes: that
# build computed at run time the constants upstream's compiler folds to nearest
# (resize-cache-rounding-evidence.json), so those six entries are upstream's now.
ROUNDING_HASHES = {
    1: (
        "e7afce5677acd788f0dca5dba32b251a85ecbf39e230883f4f0b39c464d8bdb0",
        "909c8baeabf6dbbf223e6c7f025f4eaa62aa96128adbba5da91aff8583c5eae5",
        "716afeb1410f2f057efee7f1f1b698c092d4ef98e2d2b5c5cd77e74ced312da0",
        "28ef896b30f2a8c3a0e9b9e3c28a6502780e92526aecfac99db803b71e6fdb0b",
    ),
    2: (
        "9eee5314acfabde1ebfa37addb2c5c2af2b8218d478a153f75927903f4437398",
        "3838bedfc28055142f4ebbde1f1c0b989e1ecbf52db43319d1ceb1b8dd0d40ed",
        "bd5eea033568398b253d5fc0fc5c6012d1aeba5c62a148c543f84b2c0af44ce3",
        "fb9e00bbc79baecf43a201b13165428c6c74c7b3236d2c3775ac7d63a29aa6ff",
    ),
    3: (
        "37ed3d1fa0f8437fb4e94e9099ada6cff165469b898855fbe2683084e132e8c4",
        "32f91562f15b99e0ac1b74eadb242a10aac69a0ba464177c6c4b1e5315633d80",
        "af44e0c35c0c9b6155dcbe21cece4602fd3de924cea460a93eabe318f7f73796",
        "1e11168493f0002cd7589ec4aa967b53c6f2ab0d8c83a5bb0d861334b143c012",
    ),
    4: (
        "a5f91a79c8fe931802c2e79018b2fad8215d9a398ca609df3e7eabaab5a4fdfc",
        "a5f91a79c8fe931802c2e79018b2fad8215d9a398ca609df3e7eabaab5a4fdfc",
        "a5f91a79c8fe931802c2e79018b2fad8215d9a398ca609df3e7eabaab5a4fdfc",
        "a5f91a79c8fe931802c2e79018b2fad8215d9a398ca609df3e7eabaab5a4fdfc",
    ),
    5: (
        "8812f80aa481b75c4a75954e2bc11de652e085352b0b27084c1c8df90a8c14a9",
        "f44ffb5cf121582f04c90ae0e046074523740f4da6e2db48d9688e201563ae75",
        "63d92e7c7e48bb5bb82ccae51d43f2250544e85e76e4debd3dc91a2ed05ca378",
        "a06002ee4b8028dd8db4beaca77e1bd7cd09b5dd198a14977c9703cbb341bc45",
    ),
    6: (
        "a1507ea2d4851671f57c5d817aaf64468835d2701a7c606c4f2db77dee128ddc",
        "60b10a8ad3b910275a85dd32bf6cb3df2927290c610007d766c6a837070ce69a",
        "f888832d8c8aa0d97f0144d94516e2922e70724e70c0901c41baf8ce30aabe0f",
        "994d3622b30095f8d2373a789d34c8e38583a6be0ae235988b07ea0fec330d42",
    ),
    7: (
        "ea8ce81f07865c54c5b75f19f61d9419c1971e339a827be3ca502aa4c12c2c95",
        "fd00fd646f6c4cbdc1f0114cdc04b255df848ff49fd6fcce2d48ef222b74d1cb",
        "3dbc0a9867a0396143ab7625b59da8c72114dce18070356fbab0a383ad5ca5c2",
        "e733f239bb5d8ec8fa9d8c943f183a6dec10f616a435d310a6a02208fa44adc0",
    ),
    8: (
        "909644edc4756789a6e53a9e208e8d0584c21d13022baddf22f03df0e4c94350",
        "57382e05d546b6f14616b7057d64acc4580dd2e0efa5981f64f84d0c06ec8ad1",
        "7337a8e5c5ef62036face7d08411ceaa98e1007c5ddb9faaa2b9ed5a01894cad",
        "58d2acba3d8f9daf6e126a6de651e56af0292d9456b5fefac7994c1d705e16ac",
    ),
    9: (
        "2ed49369028c07fb73ae0cce613170da9aa6f71d684a04113f9d5d91ff3e8b29",
        "1a074999cbdd786a5159e65adfa3f7a38df4f855cc4ee8cf73ae4eb25c089da2",
        "e3865c57b66521ec26175fcf7cfd0b241ff733020c15d4aee02796f922f059bf",
        "d74d6d4d99e464a71a05b8e0796bae1d983cd83825aad49efada33428a57ee06",
    ),
    10: (
        "e64fbad7fe0c2d14351d08150a9e072c7cc23cba314830f6e591d65bd49c7e30",
        "35a57ebe787a5b92fdd18bad80f306fed3cf9289feb6068507a28a45bbcb7d2c",
        "443563d5f14d31d4d994676ee6a6ee43fb67fe5e7da664d933945924f6e3f878",
        "f8f7e218a5f66df7c90a28159a8831b558129107be8e975da331bf3c63d11ca1",
    ),
    11: (
        "6d1f43719719d4414d734a292730bcd9d34e2762fdedf789fe54eb8abbdd503d",
        "6dbced9bbc990f0eca750c0ee226ab58d5655b10e1d24a6bef2822785b8550c4",
        "4371b40babf2305db9346e68a12a7da2fbca0d45a75efcbd6f0da3e820586a21",
        "9dd4140e742067ced0dc45eac3d6cb39e47662104bb5709e5f84d5333d3a94c5",
    ),
}


@pytest.mark.parametrize("filter_id", FILTERS)
@pytest.mark.skipif(sys.platform != "win32", reason="Public Windows CRT controls")
def test_rounding_mode_is_part_of_coefficient_identity(filter_id):
    runtime = ct.CDLL("ucrtbase.dll")
    control = runtime._controlfp_s  # public CRT API
    control.argtypes = [ct.POINTER(ct.c_uint), ct.c_uint, ct.c_uint]
    control.restype = ct.c_int
    saved = ct.c_uint()
    assert control(ct.byref(saved), 0, 0) == 0
    # _MCW_RC and the four documented rounding-control values. Preserve all
    # unrelated controls and restore the current thread even if parity fails.
    rounding_mask = 0x300
    try:
        for rounding in (0, 0x100, 0x200, 0x300, 0):
            observed = ct.c_uint()
            assert control(ct.byref(observed), rounding, rounding_mask) == 0
            source = np.arange(35, dtype=np.float32).reshape(5, 7, 1)
            source *= np.float32(0.03125)
            expected_hash = ROUNDING_HASHES[filter_id][rounding >> 8]
            first = native.filtered(source, (13, 11), filter_id, False)
            assert hashlib.sha256(first.tobytes()).hexdigest() == expected_hash
            before = cache_info()
            repeated = native.filtered(source, (13, 11), filter_id, False)
            assert hashlib.sha256(repeated.tobytes()).hexdigest() == expected_hash
            after = cache_info()
            assert after[1] == before[1] + 1
    finally:
        restored = ct.c_uint()
        assert control(ct.byref(restored), saved.value, rounding_mask) == 0


def test_cache_diagnostics_reject_invalid_buffers():
    cache_info()
    function = native._api().cn_resample_cache_info  # C boundary
    values = (ct.c_uint64 * 6)(*([917] * 6))
    assert function(None, 6) == 1
    assert function(values, 5) == 1
    assert function(values, 7) == 1
    assert tuple(values) == (917,) * 6
