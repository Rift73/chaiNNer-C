"""SP4c NumPy data-memory handler (numpy_pool.cpp in _chainner_graph).

Requests of 512 KiB or more are served by a pool that keeps freed blocks warm; smaller
ones go to NumPy's default handler. Each handler test runs its body in a fresh
interpreter, because the pyd reads CHAINNER_C_NUMPY_POOL and CHAINNER_C_PROFILE once
per process, at the first numpy_pool_install(). The counters are kept with
CHAINNER_C_PROFILE=1 only, so the functional tests run with and without it: a body
checks counters only when PROFILE is set.

The wiring tests at the end cover nodes/impl/numpy_pool.py, its native_profile entry
and its install points: in a fresh interpreter, with the pyd's entries patched in this
process (never installing the handler here), or from the sources' syntax trees.
"""

import ast
import json
import logging
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nodes.impl import numpy_pool
from nodes.impl.native_graph import graph

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"
POOL_VARIABLE = "CHAINNER_C_NUMPY_POOL"
PROFILE_VARIABLE = "CHAINNER_C_PROFILE"
HANDLER = "chainner_c_numpy_pool"
KIB = 1024
MIB = 1024 * KIB
# P5: the counters and the size-class table.
KEYS = {
    "requests",
    "hits",
    "hit_bytes",
    "misses_cold",
    "misses_fit",
    "fresh",
    "fresh_bytes",
    "realloc_inplace",
    "realloc_passthrough",
    "frees",
    "evictions",
    "evicted_bytes",
    "dropped",
    "dropped_bytes",
    "releases",
    "released_bytes",
    "release_failures",
    "sample_failures",
    "held_bytes",
    "held_blocks",
    "held_peak",
    "live_bytes",
    "live_blocks",
    "live_peak",
    "threads",
    "cap_bytes",
    "threshold_bytes",
    "classes",
}
# Kept whatever the profile: the idle bytes the cap needs, the VirtualFree failures,
# and the two settings.
ALWAYS = {"held_bytes", "release_failures", "cap_bytes", "threshold_bytes"}
PRELUDE = """
import json

import numpy as np
from numpy._core.multiarray import get_handler_name

from nodes.impl.native_graph import graph

g = graph()
HANDLER = "chainner_c_numpy_pool"
MIB = 1 << 20
THRESHOLD = MIB // 2
"""
BOTH = pytest.mark.parametrize("profile", [False, True], ids=["plain", "profile"])


def launch(source, pool=None, profile=False):
    """source run from backend/src in a fresh interpreter, with CHAINNER_C_NUMPY_POOL
    set to pool (None: unset) and CHAINNER_C_PROFILE=1 when profile is set.

    A hung child fails the test at the timeout instead of blocking the suite.
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in {POOL_VARIABLE, PROFILE_VARIABLE}
    }
    if pool is not None:
        environment[POOL_VARIABLE] = pool
    if profile:
        environment[PROFILE_VARIABLE] = "1"
    try:
        return subprocess.run(
            [sys.executable, "-B", "-c", source],
            cwd=BACKEND,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the child interpreter hung")


def child(body, pool=None, profile=False):
    """The `result` that body sets, as JSON; body's own asserts run in the child."""
    source = (
        PRELUDE
        + f"PROFILE = {profile}\n"
        + textwrap.dedent(body)
        + "\nprint(json.dumps(result))\n"
    )
    done = launch(source, pool, profile)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


@BOTH
def test_alloc_and_free_route_across_the_threshold(profile):
    result = child(
        """
        assert get_handler_name() == "default_allocator"
        assert g.numpy_pool_install() == ("installed", "")
        assert get_handler_name() == HANDLER
        g.numpy_pool_reset()
        small = np.empty(THRESHOLD - 1, np.uint8)     # below: NumPy's default handler
        big = np.empty(THRESHOLD, np.uint8)           # at the threshold: a fresh pool block
        address = big.ctypes.data
        assert address % 65536 == 0
        s = g.numpy_pool_stats()
        if PROFILE:
            assert (s["requests"], s["misses_cold"], s["fresh"], s["fresh_bytes"]) == (
                1, 1, 1, THRESHOLD), s
            assert (s["live_bytes"], s["live_blocks"]) == (THRESHOLD, 1), s
            assert s["classes"]["256K"]["malloc"] == (1, THRESHOLD - 1), s
            assert s["classes"]["512K"]["malloc"] == (1, THRESHOLD), s
        del small                                     # a default free: the pool is untouched
        assert g.numpy_pool_idle_blocks() == []
        if PROFILE:
            assert g.numpy_pool_stats()["frees"] == 0
        del big                                       # a pool free: kept idle
        assert g.numpy_pool_idle_blocks() == [(address, THRESHOLD)]
        s = g.numpy_pool_stats()
        assert s["held_bytes"] == THRESHOLD, s
        if PROFILE:
            assert (s["frees"], s["held_blocks"], s["live_bytes"]) == (1, 1, 0), s
        again = np.empty(THRESHOLD, np.uint8)         # reused, warm
        assert again.ctypes.data == address
        s = g.numpy_pool_stats()
        assert s["held_bytes"] == 0, s
        if PROFILE:
            assert (s["hits"], s["hit_bytes"]) == (1, THRESHOLD), s
            assert s["classes"]["512K"]["hits"] == (1, THRESHOLD), s
        del again
        g.numpy_pool_release()
        # Equal idle blocks: the newest serves.
        x = np.empty(2 * MIB, np.uint8)
        y = np.empty(2 * MIB, np.uint8)
        y_address = y.ctypes.data
        del x
        del y
        z = np.empty(2 * MIB, np.uint8)
        assert z.ctypes.data == y_address
        del z
        g.numpy_pool_release()
        # Best fit within [size, 2 size]: an idle 4 MiB block does not serve 1.5 MiB,
        # does serve 3 MiB.
        four = np.empty(4 * MIB, np.uint8)
        four_address = four.ctypes.data
        del four
        middle = np.empty(3 * MIB // 2, np.uint8)
        assert middle.ctypes.data != four_address
        three = np.empty(3 * MIB, np.uint8)
        assert three.ctypes.data == four_address
        del middle, three
        g.numpy_pool_release()
        # The smallest fitting block wins: idle 2 MiB and 3 MiB, 1.6 MiB takes the 2 MiB one.
        two = np.empty(2 * MIB, np.uint8)
        three = np.empty(3 * MIB, np.uint8)
        two_address = two.ctypes.data
        del two, three
        pick = np.empty(1600 * 1024, np.uint8)
        assert pick.ctypes.data == two_address
        s = g.numpy_pool_stats()
        result = {"misses_fit": s["misses_fit"], "hits": s["hits"]}
        """,
        profile=profile,
    )
    if profile:
        assert result["misses_fit"] >= 1
        assert result["hits"] >= 4
    else:
        assert result == {"misses_fit": 0, "hits": 0}


@BOTH
def test_realloc_grows_from_below_to_above_and_shrinks_back(profile):
    result = child(
        """
        g.numpy_pool_install()
        g.numpy_pool_reset()
        # np.fromiter grows a default block past the threshold through realloc: it stays
        # with the default handler, and its free goes there too.
        n = 200_000
        a = np.fromiter((float(i) for i in range(n)), dtype=np.float64, count=-1)
        assert a.shape == (n,) and a.sum() == n * (n - 1) / 2
        assert g.numpy_pool_idle_blocks() == []
        s = g.numpy_pool_stats()
        if PROFILE:
            reallocs = sum(entry.get("realloc", (0, 0))[0] for entry in s["classes"].values())
            assert reallocs > 0 and s["realloc_passthrough"] >= 1, s
            assert s["live_blocks"] == 0, s
        del a
        assert g.numpy_pool_idle_blocks() == []

        # A pool block shrunk below the threshold moves to the default handler, contents kept.
        b = np.arange(300_000, dtype=np.float64)        # 2.4 MB, a pool block
        address = b.ctypes.data
        assert address % 65536 == 0
        b.resize(1000, refcheck=False)
        assert b.ctypes.data != address and np.array_equal(b, np.arange(1000.0))
        assert [block[0] for block in g.numpy_pool_idle_blocks()] == [address]
        if PROFILE:
            assert g.numpy_pool_stats()["live_blocks"] == 0
        del b                                           # a default free now
        assert [block[0] for block in g.numpy_pool_idle_blocks()] == [address]

        # A pool block grown past its capacity moves to another pool block, contents kept.
        c = np.arange(300_000, dtype=np.float64)
        assert c.ctypes.data == address                 # the idle 2.4 MB block, reused
        c.resize(600_000, refcheck=False)
        assert np.array_equal(c[:300_000], np.arange(300_000.0)) and not c[300_000:].any()
        moved = c.ctypes.data
        assert moved != address and moved % 65536 == 0
        # Shrunk within [capacity / 2, capacity] and above the threshold: in place.
        c.resize(400_000, refcheck=False)
        assert c.ctypes.data == moved
        assert np.array_equal(c[:300_000], np.arange(300_000.0)) and not c[300_000:].any()
        if PROFILE:
            assert g.numpy_pool_stats()["realloc_inplace"] == 1
        del c
        idle = [block[0] for block in g.numpy_pool_idle_blocks()]
        assert address in idle and moved in idle, idle
        s = g.numpy_pool_stats()
        result = {"live_blocks": s["live_blocks"], "held_blocks": s["held_blocks"]}
        """,
        profile=profile,
    )
    if profile:
        # Idle: the 2.4 MB and 4.8 MB blocks, and the comparisons' own temporaries.
        assert result["live_blocks"] == 0
        assert result["held_blocks"] >= 2
    else:
        assert result == {"live_blocks": 0, "held_blocks": 0}


@BOTH
def test_calloc_zeroes_a_dirty_reused_block(profile):
    result = child(
        """
        g.numpy_pool_install()
        g.numpy_pool_reset()
        n = MIB                                         # 4 MiB of float32
        dirty = np.empty(n, np.float32)
        dirty.fill(7.0)
        address = dirty.ctypes.data
        del dirty
        zeros = np.zeros(n, np.float32)
        assert zeros.ctypes.data == address, "np.zeros did not reuse the dirty block"
        assert not zeros.any()
        s = g.numpy_pool_stats()
        result = {"calloc": s["classes"].get("4M", {}).get("calloc"), "hits": s["hits"]}
        """,
        profile=profile,
    )
    if profile:
        assert result == {"calloc": [1, 4 * MIB], "hits": 1}
    else:
        assert result == {"calloc": None, "hits": 0}


@BOTH
def test_cap_evicts_oldest_first_and_idle_bytes_never_exceed_it(profile):
    result = child(
        """
        g.numpy_pool_install()
        g.numpy_pool_reset()
        cap = g.numpy_pool_stats()["cap_bytes"]
        assert cap == 8 * MIB
        size = 3 * MIB
        a, b, c = (np.empty(size, np.uint8) for _ in range(3))
        addresses = [x.ctypes.data for x in (a, b, c)]
        del a
        del b
        assert g.numpy_pool_idle_blocks() == [(addresses[0], size), (addresses[1], size)]
        del c                                           # 9 MiB > 8 MiB: a, the oldest, goes
        assert g.numpy_pool_idle_blocks() == [(addresses[1], size), (addresses[2], size)]
        s = g.numpy_pool_stats()
        assert s["held_bytes"] == 2 * size, s
        if PROFILE:
            assert (s["evictions"], s["evicted_bytes"]) == (1, size), s
        big = np.empty(10 * MIB, np.uint8)              # larger than the cap: never kept
        del big
        assert g.numpy_pool_idle_blocks() == [(addresses[1], size), (addresses[2], size)]
        if PROFILE:
            s = g.numpy_pool_stats()
            assert (s["dropped"], s["dropped_bytes"]) == (1, 10 * MIB), s
        # A seeded random walk of allocations and frees: idle bytes stay within the cap.
        rng = np.random.default_rng(5)
        live = []
        for step in range(3000):
            if live and (len(live) > 6 or rng.random() < 0.5):
                live.pop(int(rng.integers(len(live))))
            else:
                live.append(np.empty(int(rng.integers(THRESHOLD, 3 * MIB)), np.uint8))
            held = g.numpy_pool_stats()["held_bytes"]
            idle = sum(capacity for _, capacity in g.numpy_pool_idle_blocks())
            assert held == idle <= cap, (step, held, idle)
        live.clear()
        s = g.numpy_pool_stats()
        if PROFILE:
            assert s["held_peak"] <= cap and s["live_bytes"] == 0, s
        released = g.numpy_pool_release()
        assert released == (s["held_bytes"], 0), released
        assert g.numpy_pool_stats()["held_bytes"] == 0
        assert g.numpy_pool_idle_blocks() == []
        result = {"evictions": s["evictions"], "held_peak": s["held_peak"], "cap": cap}
        """,
        pool="8",
        profile=profile,
    )
    if profile:
        assert result["evictions"] > 1
        assert 0 < result["held_peak"] <= result["cap"]
    else:
        assert result == {"evictions": 0, "held_peak": 0, "cap": 8 * MIB}


@BOTH
def test_per_thread_install_through_the_initializer_only(profile):
    result = child(
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        assert g.numpy_pool_install() == ("installed", "")  # the main thread, as the worker does
        assert g.numpy_pool_install() == ("installed", "")  # idempotent per thread
        assert get_handler_name() == HANDLER
        g.numpy_pool_reset()

        def work():
            array = np.empty(2 * MIB, np.uint8)
            return get_handler_name(), get_handler_name(array), array.ctypes.data % 65536 == 0

        with ThreadPoolExecutor(1, initializer=g.numpy_pool_install) as pool:
            with_initializer = pool.submit(work).result()
        s1 = g.numpy_pool_stats()
        idle = g.numpy_pool_idle_blocks()
        with ThreadPoolExecutor(1) as pool:
            without = pool.submit(work).result()
        s2 = g.numpy_pool_stats()
        assert g.numpy_pool_idle_blocks() == idle

        # The event loop's tasks copy the installing thread's context; run_in_executor
        # copies none, so a pool thread without the initializer allocates by default.
        async def loop_side():
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(1) as pool:
                in_pool = await loop.run_in_executor(pool, get_handler_name)
            return get_handler_name(), in_pool
        in_task, in_executor = asyncio.run(loop_side())

        # An array keeps its handler: allocated on a handler thread, freed on one without.
        g.numpy_pool_release()
        with ThreadPoolExecutor(1, initializer=g.numpy_pool_install) as pool:
            held = [pool.submit(np.empty, 3 * MIB, np.uint8).result()]
        address = held[0].ctypes.data
        assert get_handler_name(held[0]) == HANDLER
        assert g.numpy_pool_idle_blocks() == []
        frees = g.numpy_pool_stats()["frees"]
        with ThreadPoolExecutor(1) as pool:
            pool.submit(held.clear).result()
        assert g.numpy_pool_idle_blocks() == [(address, 3 * MIB)]
        s3 = g.numpy_pool_stats()
        result = {
            "with": with_initializer, "without": without,
            "requests": [s1["requests"], s2["requests"]],
            "threads": [s1["threads"], s2["threads"]],
            "in_task": in_task, "in_executor": in_executor, "frees": [frees, s3["frees"]],
        }
        """,
        profile=profile,
    )
    assert result["with"] == [HANDLER, HANDLER, True]
    assert result["without"] == ["default_allocator", "default_allocator", False]
    assert result["in_task"] == HANDLER
    assert result["in_executor"] == "default_allocator"
    if profile:
        assert result["requests"] == [1, 1]
        assert result["threads"] == [1, 1]
        assert result["frees"][1] == result["frees"][0] + 1
    else:
        assert result["requests"] == result["threads"] == result["frees"] == [0, 0]


def test_variable_is_read_as_specified():
    physical = child(
        """
        import ctypes
        import os

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "page_total", "page_available", "virtual_total",
                 "virtual_available", "extended")]
        status = Status()
        status.length = ctypes.sizeof(Status)
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        first = g.numpy_pool_install()
        os.environ["CHAINNER_C_NUMPY_POOL"] = "0"       # read once per process
        again = g.numpy_pool_install()
        s = g.numpy_pool_stats()
        result = {"total": status.total, "statuses": [first, again],
                  "cap": s["cap_bytes"], "threshold": s["threshold_bytes"]}
        """
    )
    assert physical["statuses"] == [["installed", ""], ["installed", ""]]
    assert physical["cap"] == min(512 * MIB, physical["total"] // 10)
    assert physical["threshold"] == MIB // 2
    body = 'result = [g.numpy_pool_install(), g.numpy_pool_stats()["cap_bytes"]]'
    assert child(body, pool="") == [["installed", ""], physical["cap"]]
    assert child(body, pool="64") == [["installed", ""], 64 * MIB]


def test_profile_samples_residency_fresh_cold_and_reused_warm():
    result = child(
        """
        g.numpy_pool_install()
        g.numpy_pool_reset()
        fresh = np.empty(4 * MIB, np.uint8)             # VirtualAlloc: no page resident yet
        fresh.fill(1)
        del fresh
        reused = np.empty(4 * MIB, np.uint8)            # the same block, every page resident
        del reused
        s = g.numpy_pool_stats()
        result = {"sampled": s["classes"]["4M"]["sampled"], "failures": s["sample_failures"]}
        """,
        profile=True,
    )
    assert result == {"sampled": [16, 8], "failures": 0}
    unsampled = child(
        """
        g.numpy_pool_install()
        array = np.empty(4 * MIB, np.uint8)
        s = g.numpy_pool_stats()
        result = {"classes": s["classes"], "failures": s["sample_failures"]}
        """
    )
    assert unsampled == {"classes": {}, "failures": 0}


@pytest.mark.parametrize(
    ("pool", "profile"),
    [(None, True), ("8", True), (None, False), ("8", False)],
    ids=["default-profile", "cap8-profile", "default-plain", "cap8-plain"],
)
def test_threads_never_share_a_block(pool, profile):
    result = child(
        """
        from concurrent.futures import ThreadPoolExecutor
        g.numpy_pool_install()
        g.numpy_pool_reset()
        cap = g.numpy_pool_stats()["cap_bytes"]

        def worker(seed):
            rng = np.random.default_rng(seed)
            live = []
            for step in range(400):
                if live and (len(live) > 4 or rng.random() < 0.5):
                    array, value = live.pop(int(rng.integers(len(live))))
                    assert (array == value).all(), (seed, step)
                else:
                    size = int(rng.integers(64 * 1024, 3 * MIB))
                    array = np.empty(size, np.uint8)
                    value = (seed * 31 + step) % 251
                    array.fill(value)
                    live.append((array, value))
            for array, value in live:
                assert (array == value).all(), seed
            return True

        with ThreadPoolExecutor(8, initializer=g.numpy_pool_install) as pool:
            assert all(pool.map(worker, range(8)))
        s = g.numpy_pool_stats()
        assert s["held_bytes"] <= cap and s["release_failures"] == 0, s
        assert s["held_bytes"] == sum(capacity for _, capacity in g.numpy_pool_idle_blocks())
        if PROFILE:
            assert s["live_bytes"] == 0 and s["held_peak"] <= cap, s
        result = {key: s[key] for key in ("threads", "hits", "evictions", "held_bytes")}
        """,
        pool=pool,
        profile=profile,
    )
    assert result["held_bytes"] > 0
    if profile:
        assert result["threads"] == 8
        assert result["hits"] > 0
        if pool == "8":
            assert result["evictions"] > 0
    else:
        assert (result["threads"], result["hits"], result["evictions"]) == (0, 0, 0)


def test_off_means_not_installed():
    # Profiling on too: "0" is not "installed and counting".
    result = child(
        """
        statuses = [g.numpy_pool_install(), g.numpy_pool_install()]
        arrays = [np.empty(2 * MIB, np.uint8), np.zeros(2 * MIB, np.uint8),
                  np.empty(512 * 1024, np.uint8)]
        names = [get_handler_name(), *(get_handler_name(array) for array in arrays)]
        del arrays
        result = {"statuses": statuses, "names": names, "idle": g.numpy_pool_idle_blocks(),
                  "release": g.numpy_pool_release(), "stats": g.numpy_pool_stats()}
        """,
        pool="0",
        profile=True,
    )
    assert result["statuses"] == [["off", ""], ["off", ""]]
    assert result["names"] == ["default_allocator"] * 4
    assert result["idle"] == []
    assert result["release"] == [0, 0]
    stats = result["stats"]
    assert set(stats) == KEYS
    assert stats["classes"] == {}
    assert stats["threshold_bytes"] == MIB // 2
    assert {
        key: stats[key] for key in KEYS - {"classes", "threshold_bytes"}
    } == dict.fromkeys(KEYS - {"classes", "threshold_bytes"}, 0)


@pytest.mark.parametrize(
    "value", ["64M", "off", "-1", " 8", "8 ", chr(0xFF18), "99999999999999999999"]
)
def test_invalid_value_is_not_installed_and_explained(value):
    result = child(
        """
        statuses = [g.numpy_pool_install(), g.numpy_pool_install()]
        array = np.empty(2 * MIB, np.uint8)
        result = {"statuses": statuses, "names": [get_handler_name(), get_handler_name(array)],
                  "idle": g.numpy_pool_idle_blocks(), "cap": g.numpy_pool_stats()["cap_bytes"]}
        """,
        pool=value,
    )
    message = (
        f"CHAINNER_C_NUMPY_POOL='{value}' is not empty, 0 or a cap in MiB "
        "(ASCII digits); the NumPy pool is not installed"
    )
    assert result["statuses"] == [["invalid", message]] * 2
    status, text = result["statuses"][0]
    assert status == "invalid"
    for part in ("CHAINNER_C_NUMPY_POOL", f"'{value}'", "0 or a cap in MiB"):
        assert part in text
    assert result["names"] == ["default_allocator"] * 2
    assert result["idle"] == []
    assert result["cap"] == 0


def test_unsupported_api_is_not_installed_and_explained():
    # The C API import fails as with another NumPy: the first install finds a
    # _multiarray_umath without _ARRAY_API, then the real module is restored.
    result = child(
        """
        import sys
        import types
        name = "numpy._core._multiarray_umath"
        real = sys.modules[name]
        sys.modules[name] = types.ModuleType(name)
        try:
            statuses = [g.numpy_pool_install(), g.numpy_pool_install()]
        finally:
            sys.modules[name] = real
        statuses.append(g.numpy_pool_install())
        array = np.empty(2 * MIB, np.uint8)
        result = {"statuses": statuses, "names": [get_handler_name(), get_handler_name(array)],
                  "idle": g.numpy_pool_idle_blocks(), "cap": g.numpy_pool_stats()["cap_bytes"]}
        """
    )
    message = (
        "NumPy's data-memory handler API is unavailable (AttributeError: module "
        "'numpy._core._multiarray_umath' has no attribute '_ARRAY_API'); "
        "the NumPy pool is not installed"
    )
    assert result["statuses"] == [["unsupported", message]] * 3
    assert result["names"] == ["default_allocator"] * 2
    assert result["idle"] == []
    assert result["cap"] == 0


def test_counters_stay_zero_without_profile():
    result = child(
        """
        assert g.numpy_pool_install() == ("installed", "")
        g.numpy_pool_reset()
        for size in (256 * 1024, 2 * MIB):
            for _ in range(3):
                array = np.empty(size, np.uint8)
                del array
            zeros = np.zeros(size, np.uint8)
            del zeros
        grown = np.arange(300_000, dtype=np.float64)
        grown.resize(200_000, refcheck=False)           # in place
        grown.resize(10, refcheck=False)                # moved to the default handler
        grown.resize(300_000, refcheck=False)           # a default block, passed through
        del grown
        s = g.numpy_pool_stats()
        idle = g.numpy_pool_idle_blocks()
        released = g.numpy_pool_release()
        result = {"stats": s, "idle": idle, "released": released,
                  "after": g.numpy_pool_stats()}
        """
    )
    stats = result["stats"]
    assert set(stats) == KEYS
    assert stats["classes"] == {}
    zero = dict.fromkeys(KEYS - ALWAYS - {"classes"}, 0)
    assert {key: stats[key] for key in zero} == zero
    assert {key: result["after"][key] for key in zero} == zero
    # The pool still worked: its idle bytes are the ones the cap needs.
    assert result["idle"]
    assert stats["held_bytes"] == sum(capacity for _, capacity in result["idle"]) > 0
    assert result["released"] == [stats["held_bytes"], 0]


def test_profile_counts_both_sides_of_the_threshold_from_64k():
    result = child(
        """
        assert g.numpy_pool_install() == ("installed", "")
        g.numpy_pool_reset()
        below = np.empty(256 * 1024, np.uint8)
        s1 = g.numpy_pool_stats()
        above = np.empty(2 * MIB, np.uint8)
        s2 = g.numpy_pool_stats()
        tiny = np.empty(32 * 1024, np.uint8)
        s3 = g.numpy_pool_stats()
        result = [s1, s2, s3]
        """,
        profile=True,
    )
    s1, s2, s3 = result
    assert set(s1) == KEYS
    assert set(s1["classes"]) == {"256K"}
    assert s1["classes"]["256K"]["malloc"] == [1, 256 * KIB]
    assert s1["classes"]["256K"]["sampled"][0] == 8
    assert "hits" not in s1["classes"]["256K"]
    assert s1["requests"] == 0
    assert set(s2["classes"]) == {"256K", "2M"}
    assert s2["classes"]["2M"]["malloc"] == [1, 2 * MIB]
    assert s2["classes"]["2M"]["sampled"] == [8, 8]  # fresh: no page resident
    assert (s2["requests"], s2["misses_cold"], s2["fresh"]) == (1, 1, 1)
    assert s2["threads"] == 1
    assert s3["classes"] == s2["classes"]
    assert s3["requests"] == 1


@BOTH
def test_release_returns_idle_bytes_and_empties_the_pool(profile):
    result = child(
        """
        g.numpy_pool_install()
        g.numpy_pool_reset()
        assert g.numpy_pool_release() == (0, 0)
        a = np.empty(MIB, np.uint8)
        b = np.empty(3 * MIB, np.uint8)
        c = np.empty(5 * MIB // 2, np.uint8)
        addresses = [x.ctypes.data for x in (a, b, c)]
        del a
        del b
        del c
        idle = g.numpy_pool_idle_blocks()
        assert idle == list(zip(addresses, (MIB, 3 * MIB, 5 * MIB // 2))), idle
        held = g.numpy_pool_stats()["held_bytes"]
        first = g.numpy_pool_release()
        second = g.numpy_pool_release()
        after = np.empty(MIB, np.uint8)                 # an idle backend holds nothing
        s = g.numpy_pool_stats()
        result = {"held": held, "first": first, "second": second,
                  "idle": g.numpy_pool_idle_blocks(), "held_after": s["held_bytes"],
                  "releases": [s["releases"], s["released_bytes"]],
                  "fresh": s["fresh"]}
        """,
        profile=profile,
    )
    held = MIB + 3 * MIB + 5 * MIB // 2
    assert result["held"] == held
    assert result["first"] == [held, 0]
    assert result["second"] == [0, 0]
    assert result["idle"] == []
    assert result["held_after"] == 0
    if profile:
        assert result["releases"] == [3, held]
        assert result["fresh"] == 4
    else:
        assert result["releases"] == [0, 0]
        assert result["fresh"] == 0


@BOTH
def test_video_frames_are_served_by_the_pool(profile):
    # D8 (spec 4.7): Load Video reads each frame into a fresh NumPy buffer, so a
    # 1280x720 frame (2.76 MB) is a pool block: never another live frame's, and reused
    # warm by the next frame once freed.
    result = child(
        """
        import io
        import types

        WIDTH, HEIGHT = 1280, 720
        SIZE = WIDTH * HEIGHT * 3
        payload = b"".join(bytes([i]) * SIZE for i in (1, 2, 3))  # three distinct frames

        class Process:                                  # what Load Video uses of Popen
            def __init__(self):
                self.stdout = io.BufferedReader(io.BytesIO(payload))

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.stdout.close()
                return False

        stream = types.SimpleNamespace()
        stream.output = lambda *args, **kwargs: stream
        stream.run_async = lambda **kwargs: Process()
        names = {
            "ffmpeg": types.SimpleNamespace(input=lambda path: stream),
            "subprocess": types.SimpleNamespace(Popen=Process),
            "BufferedIOBase": io.BufferedIOBase,
            "np": np,
            "logger": types.SimpleNamespace(debug=lambda *args: None),
        }
        loader = types.SimpleNamespace(
            path="x",
            ffmpeg_env=types.SimpleNamespace(ffmpeg="f"),
            metadata=types.SimpleNamespace(width=WIDTH, height=HEIGHT),
        )
        assert g.numpy_pool_install() == ("installed", "")
        frames = g.video_stream_frames(names, loader)
        g.numpy_pool_reset()
        first = next(frames)
        address = first.base.ctypes.data
        assert get_handler_name(first.base) == HANDLER
        second = next(frames)                           # first is alive: its own block
        assert get_handler_name(second.base) == HANDLER
        assert second.base.ctypes.data != address
        assert first.tobytes() == payload[:SIZE]        # unchanged by the later read
        assert second.tobytes() == payload[SIZE : 2 * SIZE]
        del first                                       # its block goes back idle
        assert [a for a, _ in g.numpy_pool_idle_blocks()] == [address]
        third = next(frames)                            # and serves the next frame
        assert third.base.ctypes.data == address
        assert third.tobytes() == payload[2 * SIZE :]
        del second, third
        s = g.numpy_pool_stats()
        assert list(frames) == []
        result = [s["requests"], s["hits"]]
        """,
        profile=profile,
    )
    assert result == ([3, 1] if profile else [0, 0])


DOUBLE_FREE = """
import ctypes

SEM_FAILCRITICALERRORS = 0x0001
SEM_NOGPFAULTERRORBOX = 0x0002
ctypes.windll.kernel32.SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX)
assert g.numpy_pool_install() == ("installed", "")
array = np.empty(2 * MIB, np.uint8)
del array
[(address, capacity)] = g.numpy_pool_idle_blocks()

# The current handler through NumPy's own C API: _ARRAY_API[305] is PyDataMem_GetHandler.
api = ctypes.pythonapi
api.PyCapsule_GetPointer.restype = ctypes.c_void_p
api.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
table = api.PyCapsule_GetPointer(np._core.multiarray._ARRAY_API, None)
pointer = ctypes.sizeof(ctypes.c_void_p)
get_handler = ctypes.PYFUNCTYPE(ctypes.py_object)(
    ctypes.c_void_p.from_address(table + 305 * pointer).value
)
handler = api.PyCapsule_GetPointer(get_handler(), b"mem_handler")
# PyDataMem_Handler: name[127] at 0, version at 127, then the allocator: ctx at 128,
# malloc, calloc, realloc, and free at 160.
assert ctypes.string_at(handler) == b"chainner_c_numpy_pool"
assert ctypes.c_uint8.from_address(handler + 127).value == 1
ctx = ctypes.c_void_p.from_address(handler + 128).value
free = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)(
    ctypes.c_void_p.from_address(handler + 160).value
)
print(f"freeing {address}", flush=True)
free(ctx, address, capacity)
print("survived", flush=True)
"""


def test_double_free_writes_one_line_then_aborts():
    done = launch(PRELUDE + DOUBLE_FREE)
    assert done.returncode in {3, 0xC0000409}, (done.returncode, done.stderr)
    lines = done.stdout.splitlines()
    assert lines and lines[-1].startswith("freeing "), done.stdout
    address = int(lines[-1].removeprefix("freeing "))
    reported = [
        line
        for line in done.stderr.splitlines()
        if line.startswith("chaiNNer-C numpy pool: ")
    ]
    assert len(reported) == 1, done.stderr
    prefix = "chaiNNer-C numpy pool: double free of pool block "
    assert reported[0].startswith(prefix), reported
    assert reported[0].endswith("; aborting"), reported
    assert (
        int(reported[0].removeprefix(prefix).removesuffix("; aborting"), 16) == address
    )


# The wiring (nodes/impl/numpy_pool.py, native_profile, the install points).


@pytest.fixture
def first_install(monkeypatch):
    """numpy_pool's process state as before any install, restored afterwards."""
    for name in ("_installed", "_warned", "_failed", "_release_warned"):
        monkeypatch.setattr(numpy_pool, name, False)


def sanic_records(caplog):
    return [
        (r.levelno, r.getMessage()) for r in caplog.records if r.name == "sanic.root"
    ]


@pytest.mark.usefixtures("first_install")
@pytest.mark.parametrize(
    ("status", "message"),
    [
        (
            "invalid",
            (
                "CHAINNER_C_NUMPY_POOL='512M' is not empty, 0 or a cap in MiB "
                "(ASCII digits); the NumPy pool is not installed"
            ),
        ),
        (
            "unsupported",
            (
                "NumPy's data-memory handler API is unavailable "
                "(AttributeError: _ARRAY_API not found); the NumPy pool is not installed"
            ),
        ),
    ],
    ids=["invalid", "unsupported"],
)
def test_install_warns_once_for_invalid_or_unsupported(
    monkeypatch, caplog, status, message
):
    # Review focus 1 and 4: a typo in the variable, or a NumPy without the handler
    # API, gives one warning and NumPy's default handler, never a raise.
    assert numpy_pool.VARIABLE == POOL_VARIABLE
    calls = []
    monkeypatch.setattr(
        graph(), "numpy_pool_install", lambda: calls.append(status) or (status, message)
    )
    with caplog.at_level(logging.INFO, logger="sanic.root"):
        for _ in range(3):
            numpy_pool.install()
    assert calls == [status] * 3
    assert sanic_records(caplog) == [(logging.WARNING, message)]
    assert numpy_pool.installed() is False


@pytest.mark.usefixtures("first_install")
def test_install_logs_an_unexpected_error_once_and_never_raises(monkeypatch, caplog):
    # An executor's initializer that raises breaks that executor for every later
    # submit, and main() must start the worker: an exception the pyd's install raises
    # (as configure() does if GlobalMemoryStatusEx fails) is logged once with its
    # traceback, and the thread keeps NumPy's default handler.
    def fail():
        raise OSError(1450, "Insufficient system resources exist")

    monkeypatch.setattr(graph(), "numpy_pool_install", fail)
    with caplog.at_level(logging.INFO, logger="sanic.root"):
        numpy_pool.install()
        with ThreadPoolExecutor(2, initializer=numpy_pool.install) as pool:
            assert pool.submit(sum, [1, 2]).result() == 3
        numpy_pool.install()
    records = [r for r in caplog.records if r.name == "sanic.root"]
    assert [r.levelno for r in records] == [logging.ERROR]
    exc_info = records[0].exc_info
    assert exc_info is not None
    assert exc_info[0] is OSError
    assert numpy_pool.installed() is False


@pytest.mark.usefixtures("first_install")
def test_release_is_a_no_op_when_not_installed(monkeypatch):
    calls = []
    monkeypatch.setattr(
        graph(), "numpy_pool_release", lambda: calls.append(1) or (MIB, 0)
    )
    assert numpy_pool.release() == 0
    assert calls == []


@pytest.mark.usefixtures("first_install")
def test_release_returns_the_bytes_and_warns_once_on_virtualfree_failures(
    monkeypatch, caplog
):
    # The pyd returns the process's VirtualFree failures so far, never reset.
    monkeypatch.setattr(numpy_pool, "_installed", True)
    results = iter([(3 * MIB, 0), (MIB, 2), (0, 3)])
    monkeypatch.setattr(graph(), "numpy_pool_release", results.__next__)
    with caplog.at_level(logging.INFO, logger="sanic.root"):
        released = [numpy_pool.release() for _ in range(3)]
    assert released == [3 * MIB, MIB, 0]
    [(level, text)] = sanic_records(caplog)
    assert level == logging.WARNING
    assert "VirtualFree failed 2 time(s)" in text


# /run's order: the release, then native_profile's line.
PROFILE_LINE = """
from nodes.impl import native_profile, numpy_pool

numpy_pool.install()
native_profile.reset()
array = np.empty(2 * MIB, np.uint8)
del array
released = numpy_pool.release()
line = json.loads(native_profile.log_line().removeprefix("native profile: "))
result = {"installed": numpy_pool.installed(), "released": released, "line": line,
          "snapshot": native_profile.snapshot()}
"""


def test_profile_line_has_numpy_pool_only_while_installed():
    on = child(PROFILE_LINE, profile=True)
    assert on["installed"] is True
    assert on["released"] == 2 * MIB
    entry = on["line"]["numpy_pool"]
    assert set(entry) == KEYS
    assert (entry["requests"], entry["fresh"], entry["held_bytes"]) == (1, 1, 0)
    assert (entry["releases"], entry["released_bytes"]) == (1, 2 * MIB)
    # Untimed, like "pool": the counters are read past the timer.
    assert not any(name.startswith("numpy_pool") for name in on["snapshot"])
    off = child(PROFILE_LINE, pool="0", profile=True)
    assert off["installed"] is False
    assert off["released"] == 0
    assert "numpy_pool" not in off["line"]


def test_reset_rebases_numpy_pool_counters():
    result = child(
        """
        from nodes.impl import native_profile, numpy_pool

        numpy_pool.install()
        arrays = [np.empty(2 * MIB, np.uint8) for _ in range(3)]

        def entry():
            line = native_profile.log_line().removeprefix("native profile: ")
            return json.loads(line)["numpy_pool"]

        before = entry()
        native_profile.reset()
        result = {"before": before, "after": entry()}
        """,
        profile=True,
    )
    before, after = result["before"], result["after"]
    assert (before["requests"], before["fresh"]) == (3, 3)
    assert (after["requests"], after["fresh"], after["fresh_bytes"]) == (0, 0, 0)
    # The counts restart; the three arrays are still live.
    assert after["live_bytes"] == before["live_bytes"] == 6 * MIB


def imported(tree, relative, name):
    """The absolute module a module-level `from ... import name` in the file at
    relative (to backend/src) binds; None without one."""
    package = list(relative.parent.parts)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == name and alias.asname is None for alias in node.names
        ):
            parts = package[: len(package) - node.level + 1] if node.level else []
            if node.module:
                parts += node.module.split(".")
            return ".".join([*parts, name])
    return None


def test_every_worker_executor_installs_the_handler():
    # P9: no unit test starts a worker, so the install points are read from the
    # sources. server_host.py is the host process, which never loads the pyd.
    found = {}
    for path in sorted(BACKEND.rglob("*.py")):
        relative = path.relative_to(BACKEND)
        if relative.parts[0] == "tests" or relative.as_posix() == "server_host.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func).rsplit(".", 1)[-1] == "ThreadPoolExecutor"
        ]
        if calls:
            initializers = [
                [ast.unparse(k.value) for k in call.keywords if k.arg == "initializer"]
                for call in calls
            ]
            found[relative.as_posix()] = (
                initializers,
                imported(tree, relative, "numpy_pool"),
            )
    expected = ([["numpy_pool.install"]], "nodes.impl.numpy_pool")
    assert found == {
        "nodes/impl/ffmpeg.py": expected,
        "packages/chaiNNer_external/web_ui.py": expected,
        "server.py": expected,
    }
    # The worker's main thread, before the event loop whose tasks copy its context.
    server = ast.parse((BACKEND / "server.py").read_text(encoding="utf-8"))
    (main,) = (
        node
        for node in server.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    body = [ast.unparse(statement) for statement in main.body]
    run = next(i for i, line in enumerate(body) if line.startswith("app.run("))
    assert "numpy_pool.install()" in body[:run]
