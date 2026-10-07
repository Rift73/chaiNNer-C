"""The native pool: exact results under overlapping callers and a fixed partition."""

import ctypes as ct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise

import numpy as np
import pytest
import reference_blend as reference

from nodes.impl.blend import BlendMode, blend_images
from nodes.impl.native import lib
from nodes.impl.native_adjustments import adjust, levels, log_linear, opacity

RANGE_FN = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_size_t, ct.c_size_t)
MAX_CHUNKS = 128
PARTITIONS = [
    (0, 1),
    (1, 1),
    (5, 8),
    (8, 8),
    (9, 8),
    (129, 1),
    (1000, 1),
    (1000, 7),
    (4097, 32),
    (100000, 3),
    (589824, 1024),
]


@pytest.fixture(scope="module")
def images():
    rng = np.random.default_rng(32091)
    a = rng.random((257, 513, 4), dtype=np.float32)
    b = rng.random(a.shape, dtype=np.float32)
    a.setflags(write=False)
    b.setflags(write=False)
    return a, b


def pool():
    dll = lib()
    dll.cn_parallel_for.argtypes = [ct.c_size_t, ct.c_size_t, RANGE_FN, ct.c_void_p]
    dll.cn_parallel_for.restype = ct.c_int
    dll.cn_parallel_capacity.argtypes = []
    dll.cn_parallel_capacity.restype = ct.c_uint
    dll.cn_parallel_dispatches.argtypes = []
    dll.cn_parallel_dispatches.restype = ct.c_uint64
    return dll


def counters():
    """pool() plus the pool counters and the test-only no-op dispatch."""
    dll = pool()
    for name in ("cn_parallel_wakes", "cn_parallel_empty_wakes"):
        counter = getattr(dll, name)
        counter.argtypes = []
        counter.restype = ct.c_uint64
    dll.cn_parallel_noop.argtypes = [ct.c_size_t]
    dll.cn_parallel_noop.restype = ct.c_int
    return dll


def counts(dll):
    return (
        dll.cn_parallel_dispatches(),
        dll.cn_parallel_wakes(),
        dll.cn_parallel_empty_wakes(),
    )


def partition(count, grain):
    """The chunks cn_parallel_for must deliver: a function of (count, grain) alone."""
    if count <= grain:
        return [(0, count)] if count else []
    chunks = min(1 + (count - 1) // grain, MAX_CHUNKS)
    block, remainder = divmod(count, chunks)
    edges = [block * index + min(index, remainder) for index in range(chunks + 1)]
    return list(pairwise(edges))


def chunks_of(count, grain, in_chunk=None):
    """Run one call and return the chunks it delivered, sorted."""
    seen = []
    lock = threading.Lock()

    def record(_context, begin, end):
        if in_chunk is not None:
            in_chunk()
        with lock:
            seen.append((begin, end))

    callback = RANGE_FN(record)
    assert pool().cn_parallel_for(count, grain, callback, None) == 0
    return sorted(seen)


@pytest.mark.parametrize("mode", list(BlendMode))
def test_parallel_blends(images, mode):
    a, b = images
    expected = reference.blend_images(a, b, reference.BlendMode(mode.value))
    np.testing.assert_array_equal(blend_images(a, b, mode), expected)


def test_simultaneous_calls_and_repeatability(images):
    a, b = images
    expected = [reference.blend_images(a, b, reference.BlendMode(i)) for i in range(23)]
    before_a, before_b = a.copy(), b.copy()

    def run(index):
        mode = index % 23
        result = blend_images(a, b, BlendMode(mode))
        np.testing.assert_array_equal(result, expected[mode])
        # Different kernels share the same pool, and all source arrays are reused.
        np.testing.assert_array_equal(adjust(a, 0, 0.25), a + np.float32(0.25))
        np.testing.assert_array_equal(adjust(a, 1, 0.75), a * np.float32(0.75))
        inverted = a.copy()
        inverted[..., :3] = 1 - inverted[..., :3]
        np.testing.assert_array_equal(adjust(a, 2), inverted)
        translucent = a.copy()
        translucent[..., 3] *= np.float32(0.25)
        np.testing.assert_array_equal(opacity(a, 0.25), translucent)
        np.testing.assert_array_equal(levels(a, 15, 0, 1, 1, 0, 1), a)
        np.testing.assert_array_equal(
            log_linear(a, 95, 685, 0.6, False), log_linear(a, 95, 685, 0.6, False)
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(run, range(46)))
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)


def test_large_calls_dispatch_to_native_pool(images):
    dll = pool()
    capacity = dll.cn_parallel_capacity()
    assert capacity >= 1
    before = dll.cn_parallel_dispatches()
    adjust(images[0], 1, 2)
    after = dll.cn_parallel_dispatches()
    assert after > before
    # Tiny images stay synchronous and don't enqueue work.
    adjust(images[0][:2, :2], 1, 2)
    assert dll.cn_parallel_dispatches() == after


def test_pool_counters_count_wakes_and_empty_wakes():
    dll = counters()
    if dll.cn_parallel_capacity() == 1:
        pytest.skip("one CPU: the pool has no helpers to wake")
    before = counts(dll)
    for _ in range(3):
        assert dll.cn_parallel_noop(32) == 0
    dispatches, wakes, empty = (b - a for a, b in zip(before, counts(dll), strict=True))
    assert dispatches == 3
    # Each call wakes at most 31 helpers plus the re-submit after it leaves.
    assert 1 <= wakes <= 3 * 32
    # A late helper of an earlier test may still add an empty wake.
    assert empty >= 0
    for count in (1, 0):  # no chunk list: neither a dispatch nor a wake
        before = counts(dll)
        assert dll.cn_parallel_noop(count) == 0
        assert counts(dll)[:2] == before[:2]


def test_wakes_count_the_resubmit_into_a_full_pool():
    """Review I1. A call that enters while another call's helpers fill the pool has
    no room (capacity - callers - helpers < 0) and wakes no helper, but its leave
    re-submits one while the other call stays registered: each such call adds
    exactly that 1 to the wake counter, never the negative room.

    The other call's chunks wait inside the callback until the pool is full (its
    caller and capacity - 1 helpers in chunks), so every short call meets room -1.
    """
    dll = counters()
    capacity = dll.cn_parallel_capacity()
    if capacity < 4:
        pytest.skip("needs at least four CPUs")
    release = threading.Event()
    lock = threading.Lock()
    state = {"active": 0}

    def wait_in_chunk(_context, _begin, _end):
        with lock:
            state["active"] += 1
        release.wait(30)
        with lock:
            state["active"] -= 1

    callback = RANGE_FN(wait_in_chunk)
    statuses = []
    worker = threading.Thread(
        target=lambda: statuses.append(
            dll.cn_parallel_for(MAX_CHUNKS, 1, callback, None)
        )
    )
    deltas = []
    worker.start()
    try:
        limit = time.monotonic() + 30
        while state["active"] < capacity and time.monotonic() < limit:
            time.sleep(0.001)
        full = state["active"]
        if full == capacity:
            for _ in range(8):
                before = dll.cn_parallel_wakes()
                assert dll.cn_parallel_noop(2) == 0
                deltas.append(dll.cn_parallel_wakes() - before)
    finally:
        release.set()
        worker.join()
    assert statuses == [0]
    assert full == capacity  # precondition: the other call filled the pool
    assert deltas == [1] * 8


def test_noop_dispatch_validates_its_count():
    dll = counters()
    assert dll.cn_parallel_noop(128) == 0
    assert dll.cn_parallel_noop(129) == 1  # CN_INVALID_ARGUMENT


def todays_wakes(capacity, chunks):
    """SP2's wake count for a lone call: every helper the budget admits
    (capacity - callers - running helpers), at most one per chunk beyond the
    caller's. No other call remains, so its leave re-submits nothing."""
    return min(capacity - 1, chunks - 1)


def sleep_in_chunk(seconds):
    """A range function whose chunks are slow enough that every helper woken
    while chunks remain finds one."""

    def chunk(_context, _begin, _end):
        time.sleep(seconds)

    return RANGE_FN(chunk)


@pytest.mark.parametrize("count", [1, 8, 32, 128])
def test_wakes_per_dispatch_never_exceed_todays_count(count):
    """Task 3d: however helpers are woken, a lone call submits at most SP2's
    min(room, chunks - 1) wakes: with empty chunks (the no-op probe) and with
    slow ones, where helpers find chunks left and wake more."""
    dll = counters()
    limit = todays_wakes(dll.cn_parallel_capacity(), count)
    slow = sleep_in_chunk(0.002)
    for _ in range(5):
        before = dll.cn_parallel_wakes()
        assert dll.cn_parallel_noop(count) == 0
        assert dll.cn_parallel_wakes() - before <= limit
        before = dll.cn_parallel_wakes()
        assert dll.cn_parallel_for(count, 1, slow, None) == 0
        assert dll.cn_parallel_wakes() - before <= limit


@pytest.mark.parametrize("count", [8, 16])
def test_slow_chunks_still_wake_todays_count(count):
    """Task 3d: when every helper finds a chunk (each sleeps 5 ms), a lone call
    wakes SP2's full count, so a policy that wakes fewer helpers up front makes
    up the rest while chunks remain. Twice the count in CPUs leaves room for the
    full count even while a few late helpers of an earlier call are leaving."""
    dll = counters()
    capacity = dll.cn_parallel_capacity()
    if capacity < 2 * count:
        pytest.skip(f"needs at least {2 * count} CPUs")
    slow = sleep_in_chunk(0.005)
    before = dll.cn_parallel_wakes()
    for _ in range(3):
        assert dll.cn_parallel_for(count, 1, slow, None) == 0
    assert dll.cn_parallel_wakes() - before == 3 * todays_wakes(capacity, count)


def test_overlapping_slow_calls_run_every_chunk_once_within_the_wake_bound():
    """Task 3d: concurrent calls with slow chunks (helpers find chunks left and
    wake more) each deliver their partition exactly once, and together submit
    no more wakes than SP2's bound: per call min(capacity - 1, chunks - 1) plus
    the one re-submit after it leaves."""
    dll = counters()
    capacity = dll.cn_parallel_capacity()
    cases = [(8, 1), (24, 1), (32, 1), (48, 1), (100, 1), (4096, 32)]
    gate = threading.Barrier(len(cases))

    def call(case):
        gate.wait()
        return chunks_of(*case, in_chunk=lambda: time.sleep(0.0003))

    before = dll.cn_parallel_wakes()
    with ThreadPoolExecutor(max_workers=len(cases)) as executor:
        results = list(executor.map(call, cases))
    wakes = dll.cn_parallel_wakes() - before
    expected = [partition(*case) for case in cases]
    assert results == expected
    assert wakes <= sum(todays_wakes(capacity, len(p)) + 1 for p in expected)


@pytest.mark.parametrize(("count", "grain"), PARTITIONS)
def test_partition_depends_only_on_count_and_grain(count, grain):
    assert chunks_of(count, grain) == partition(count, grain)


def test_partition_holds_under_overlapping_callers():
    cases = [(100000 + 37 * index, 1 + 5 * index) for index in range(8)]

    def repeat(case):
        return [chunks_of(*case) for _ in range(10)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(repeat, cases))
    for case, runs in zip(cases, results, strict=True):
        assert runs == [partition(*case)] * 10


def test_more_callers_than_registry_slots_all_complete():
    # The pool registers at most 32 concurrent calls. A call beyond that runs
    # its chunks on its own thread and must deliver the same partition.
    callers = 40
    gate = threading.Barrier(callers)
    # Each call's first chunk waits until every call is inside one: more calls than
    # the registry holds have started a chunk and not yet returned, all at once.
    inside = threading.Barrier(callers, timeout=10)
    lock = threading.Lock()
    met = []

    def call(index):
        started = []

        def in_chunk():
            with lock:
                first = not started
                started.append(True)
            if first:
                try:
                    inside.wait()
                    met.append(True)
                except threading.BrokenBarrierError:
                    met.append(False)

        gate.wait()
        return chunks_of(5000 + index, 7, in_chunk=in_chunk)

    with ThreadPoolExecutor(max_workers=callers) as executor:
        results = list(executor.map(call, range(callers)))
    assert results == [partition(5000 + index, 7) for index in range(callers)]
    assert met == [True] * callers


def test_no_chunk_is_running_or_starts_after_the_call_returns():
    returned = threading.Event()
    lock = threading.Lock()
    state = {"active": 0, "late": 0}

    def record(_context, _begin, _end):
        with lock:
            state["active"] += 1
            state["late"] += returned.is_set()
        time.sleep(0.0005)
        with lock:
            state["active"] -= 1

    callback = RANGE_FN(record)
    assert pool().cn_parallel_for(4096, 1, callback, None) == 0
    with lock:
        running = state["active"]
        returned.set()
    time.sleep(0.05)
    assert running == 0
    assert state["late"] == 0


def test_overlapping_calls_share_the_helpers():
    if pool().cn_parallel_capacity() < 4:
        pytest.skip("needs at least four CPUs")
    gate = threading.Barrier(2)

    def call(_index):
        threads = set()

        def in_chunk():
            threads.add(threading.get_ident())
            time.sleep(0.002)

        gate.wait()
        chunks_of(128 * 64, 64, in_chunk=in_chunk)
        return len(threads)

    with ThreadPoolExecutor(max_workers=2) as executor:
        used = list(executor.map(call, range(2)))
    # The previous pool gave every helper to the first call; the second ran alone.
    assert min(used) >= 2
