"""The cached decorator must hash inputs once without changing cache behavior.

These checks use local arrays and fake node functions. They never call an
Automatic1111 server, perform inference, or measure execution speed.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from threading import Barrier, Event

import numpy as np
import pytest

from nodes import node_cache


@pytest.fixture(autouse=True)
def isolated_cache_registry(monkeypatch):
    # Keep temporary entries and limit/eviction checks local to each test.
    monkeypatch.setattr(node_cache, "CACHE_REGISTRY", [])
    monkeypatch.setattr(node_cache, "CACHE_MAX_BYTES", 1024 * 1024)
    yield
    for cache in node_cache.CACHE_REGISTRY:
        for output in cache._data.values():  # inspect cached resources for test cleanup
            for item in output:
                if isinstance(item, node_cache.CachedNumpyArray):
                    item.file.close()


@pytest.fixture
def hash_calls(monkeypatch):
    calls = []
    real_sha256 = node_cache.hashlib.sha256

    def counted_sha256(data):
        calls.append(data)
        return real_sha256(data)

    monkeypatch.setattr(node_cache.hashlib, "sha256", counted_sha256)
    return calls


def test_one_hash_per_image_on_miss_and_hit(hash_calls):
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.flip(a, axis=1)
    a.setflags(write=False)
    calls = []

    @node_cache.cached
    def combine(left, right):
        calls.append((left, right))
        return left + right

    expected = a + b
    first = combine(a, b)
    assert len(hash_calls) == 2
    np.testing.assert_array_equal(first, expected)
    second = combine(a.copy(), b.copy())
    assert len(hash_calls) == 4
    assert len(calls) == 1
    np.testing.assert_array_equal(second, expected)
    assert not np.shares_memory(first, second)
    assert isinstance(second, np.ndarray)
    assert not second.flags.writeable


def test_value_shape_dtype_and_parameter_changes_miss(hash_calls):
    calls = []

    @node_cache.cached
    def identify(image, text):
        calls.append((image.copy(), text))
        return len(calls)

    a = np.arange(8, dtype=np.float32).reshape(2, 4)
    assert identify(a, "same") == 1
    assert identify(a.copy(), "same") == 1
    changed = a.copy()
    changed[0, 0] = 99
    assert identify(changed, "same") == 2
    assert identify(a.reshape(4, 2), "same") == 3
    assert identify(a.astype(np.float64), "same") == 4
    assert identify(a, "changed") == 5
    assert len(hash_calls) == 6
    assert len(calls) == 5


def test_independent_decorated_nodes_do_not_share_results(hash_calls):
    calls = []

    @node_cache.cached
    def first(image):
        calls.append("first")
        return "A"

    @node_cache.cached
    def second(image):
        calls.append("second")
        return "B"

    image = np.zeros((2, 2), np.float32)
    assert first(image) == "A"
    assert second(image) == "B"
    assert first(image) == "A"
    assert second(image) == "B"
    assert calls == ["first", "second"]
    assert len(hash_calls) == 4


def test_exceptions_are_not_cached_and_original_error_propagates(hash_calls):
    failure = ValueError("node failed")
    calls = []

    @node_cache.cached
    def node(image):
        calls.append(image)
        raise failure

    image = np.zeros((2, 2), np.float32)
    for _ in range(2):
        with pytest.raises(ValueError) as caught:
            node(image)
        assert caught.value is failure
    assert len(calls) == len(hash_calls) == 2
    assert all(cache.empty() for cache in node_cache.CACHE_REGISTRY)


def test_none_output_keeps_existing_miss_semantics(hash_calls):
    calls = []

    @node_cache.cached
    def node(image):
        calls.append(image)

    image = np.zeros((2, 2), np.float32)
    assert node(image) is None
    assert node(image) is None
    assert len(calls) == len(hash_calls) == 2


def test_invalid_key_fails_before_node_execution():
    calls = []

    @node_cache.cached
    def node(value):
        calls.append(value)
        return "never"

    with pytest.raises(RuntimeError, match="Unexpected argument type object"):
        node(object())
    assert calls == []


def test_public_get_put_remain_compatible(hash_calls):
    cache = node_cache.NodeOutputCache()
    image = np.ones((2, 3), np.float32)
    assert cache.get((image,)) is None
    cache.put((image,), ("value", 17))
    assert cache.get((image.copy(),)) == ["value", 17]
    assert len(hash_calls) == 3


def test_result_identity_and_tuple_conversion_unchanged():
    original = ("value", np.arange(4, dtype=np.float32))

    @node_cache.cached
    def node():
        return original

    assert node() is original
    cached = node()
    assert isinstance(cached, list)
    assert cached[0] == original[0]
    np.testing.assert_array_equal(cached[1], original[1])
    assert not np.shares_memory(cached[1], original[1])


def test_enum_and_stable_custom_key_equality():
    class Mode(Enum):
        FIRST = "first"
        SECOND = "second"

    class Key:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def cache_key_func(self):
            self.calls += 1
            return self.value

    calls = []

    @node_cache.cached
    def node(mode, key):
        calls.append((mode, key.value))
        return len(calls)

    key = Key(12)
    assert node(Mode.FIRST, key) == 1
    assert key.calls == 1
    assert node(Mode.FIRST, Key(12)) == 1
    assert node(Mode.SECOND, Key(12)) == 2
    assert node(Mode.FIRST, Key(13)) == 3
    assert len(calls) == 3


def test_cache_budget_still_evicts_old_entries(monkeypatch):
    monkeypatch.setattr(node_cache, "CACHE_MAX_BYTES", 16)
    calls = []

    @node_cache.cached
    def node(value):
        calls.append(value)
        return np.full(4, value, dtype=np.float32)

    node(1)
    node(2)
    node(2)
    node(1)
    assert calls == [1, 2, 1]
    assert sum(cache.size() for cache in node_cache.CACHE_REGISTRY) <= 16


def test_decorator_metadata_is_preserved():
    def original(value):
        """Node documentation."""
        return value

    wrapped = node_cache.cached(original)
    assert wrapped.__name__ == original.__name__
    assert wrapped.__doc__ == original.__doc__
    assert wrapped.__wrapped__ is original


@pytest.fixture
def waiting_follower(monkeypatch):
    """Observe a caller waiting without sleeping or relying on thread timing."""
    waiting = Event()

    class ObservedFuture(Future):
        def result(self, timeout=None):
            waiting.set()
            return super().result(timeout)

    monkeypatch.setattr(node_cache, "Future", ObservedFuture)
    return waiting


def test_concurrent_identical_inputs_execute_once(waiting_follower, hash_calls):
    started, release = Event(), Event()
    calls = []

    @node_cache.cached
    def node(image):
        calls.append(image)
        started.set()
        assert release.wait(5)
        return image + 1

    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(node, image)
        assert started.wait(5)
        second = pool.submit(node, image.copy())
        try:
            assert waiting_follower.wait(5)
            assert len(calls) == 1
        finally:
            release.set()
        a, b = first.result(5), second.result(5)
    np.testing.assert_array_equal(a, image + 1)
    np.testing.assert_array_equal(b, a)
    assert not np.shares_memory(a, b)
    assert isinstance(b, np.ndarray)
    assert not b.flags.writeable
    assert len(calls) == 1
    assert len(hash_calls) == 2


def test_distinct_keys_execute_concurrently():
    all_running = Barrier(4)
    calls = []

    @node_cache.cached
    def node(value):
        calls.append(value)
        # All four computations must enter before any is allowed to finish.
        all_running.wait(timeout=5)
        return value * 2

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(node, range(4)))
    assert outputs == [0, 2, 4, 6]
    assert sorted(calls) == list(range(4))


def test_shared_failure_reaches_waiter_and_later_call_retries(waiting_follower):
    started, release = Event(), Event()
    failure = ValueError("remote request failed")
    calls = []

    @node_cache.cached
    def node(value):
        calls.append(value)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
            raise failure
        return "recovered"

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(node, 1)
        assert started.wait(5)
        follower = pool.submit(node, 1)
        try:
            assert waiting_follower.wait(5)
        finally:
            release.set()
        for result in (owner, follower):
            with pytest.raises(ValueError) as caught:
                result.result(5)
            assert caught.value is failure
    assert calls == [1]
    assert node(1) == "recovered"
    assert node(1) == "recovered"
    assert calls == [1, 1]


def test_waiter_snapshot_survives_immediate_eviction(monkeypatch, waiting_follower):
    monkeypatch.setattr(node_cache, "CACHE_MAX_BYTES", 1)
    started, release = Event(), Event()
    calls = []
    original = np.arange(24, dtype=np.float32).reshape(4, 6)

    @node_cache.cached
    def node():
        calls.append(None)
        started.set()
        assert release.wait(5)
        return original

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(node)
        assert started.wait(5)
        follower = pool.submit(node)
        try:
            assert waiting_follower.wait(5)
        finally:
            release.set()
        assert owner.result(5) is original
        snapshot = follower.result(5)
    assert all(cache.empty() for cache in node_cache.CACHE_REGISTRY)
    np.testing.assert_array_equal(snapshot, original)
    assert not np.shares_memory(snapshot, original)
    assert isinstance(snapshot, np.ndarray)
    assert not snapshot.flags.writeable
    assert len(calls) == 1


def test_cache_write_failure_propagates_to_waiter(monkeypatch, waiting_follower):
    started, release = Event(), Event()
    failure = OSError("cache disk unavailable")
    # Inject a disk-writer failure.
    real_writer = node_cache.NodeOutputCache._write_arrays_to_disk
    attempts = []

    def write(output):
        attempts.append(None)
        if len(attempts) == 1:
            raise failure
        return real_writer(output)

    monkeypatch.setattr(
        node_cache.NodeOutputCache, "_write_arrays_to_disk", staticmethod(write)
    )

    @node_cache.cached
    def node():
        started.set()
        assert release.wait(5)
        return np.arange(4, dtype=np.float32)

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(node)
        assert started.wait(5)
        follower = pool.submit(node)
        try:
            assert waiting_follower.wait(5)
        finally:
            release.set()
        for result in (owner, follower):
            with pytest.raises(OSError) as caught:
                result.result(5)
            assert caught.value is failure
    np.testing.assert_array_equal(node(), np.arange(4, dtype=np.float32))
    assert len(attempts) == 2


def test_different_caches_share_budget_safely():
    barrier = Barrier(4)

    def make_node(offset):
        @node_cache.cached
        def node(value):
            barrier.wait(timeout=5)
            return np.full(4, offset + value, dtype=np.float32)

        return node

    first, second = make_node(0), make_node(100)
    node_cache.CACHE_MAX_BYTES = 32
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(first if i % 2 == 0 else second, i) for i in range(24)]
        outputs = [future.result(5) for future in futures]
    for i, output in enumerate(outputs):
        np.testing.assert_array_equal(
            output, np.full(4, i if i % 2 == 0 else i + 100, np.float32)
        )
    assert sum(cache.size() for cache in node_cache.CACHE_REGISTRY) <= 32


def test_concurrent_cache_hits_read_complete_snapshots():
    @node_cache.cached
    def node(value):
        return np.arange(1024, dtype=np.float32) + value

    expected = node(12)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(node, [12] * 64))
    for result in results:
        np.testing.assert_array_equal(result, expected)
        assert isinstance(result, np.ndarray)
        assert not result.flags.writeable


def test_recursive_same_key_fails_instead_of_waiting_for_itself():
    @node_cache.cached
    def node(value):
        return node(value)

    with pytest.raises(RuntimeError, match="Recursive evaluation"):
        node(12)
