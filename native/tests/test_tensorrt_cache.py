"""TensorRT engines and sessions kept across runs (nodes/impl/tensorrt/cache.py).

Fakes only: the cache never imports TensorRT, so no GPU is needed.
"""

import os
import weakref
from threading import Event, Thread

import pytest

from api import NodeId
from nodes.impl.pytorch.resource_cache import file_identity
from nodes.impl.tensorrt import cache
from nodes.impl.tensorrt.cache import SharedCache
from nodes.impl.tensorrt.model import (
    DeserializedEngine,
    TensorRTEngine,
    TensorRTEngineInfo,
)

A, B = NodeId("a"), NodeId("b")
C, D = NodeId("c"), NodeId("d")


class Fake:
    def __init__(self, name):
        self.name = name
        self.closed = False


class Recorder:
    """A SharedCache whose created and closed values are recorded."""

    def __init__(self):
        self.created: list[Fake] = []
        self.closed: list[Fake] = []
        self.cache: SharedCache[Fake] = SharedCache(self.close)

    def close(self, value):
        value.closed = True
        self.closed.append(value)

    def create(self, name):
        def make():
            value = Fake(name)
            self.created.append(value)
            return value

        return make

    def use(self, key, node_id):
        with self.cache.use(key, node_id, self.create(key)) as value:
            return value


def test_entries_are_shared_and_freed_by_their_last_node():
    r = Recorder()
    first = r.use("engine", A)
    assert r.use("engine", A) is first  # a second run reuses it
    assert r.use("engine", B) is first  # another node shares it
    assert len(r.created) == 1
    r.cache.release_node(A)
    assert r.closed == [] and len(r.cache) == 1
    r.cache.release_node(B)
    assert r.closed == [first] and len(r.cache) == 0


def test_unknown_ids_are_a_no_op():
    r = Recorder()
    value = r.use("engine", A)
    r.cache.release_node(B)
    r.cache.release_node(A)
    r.cache.release_node(A)
    assert r.closed == [value]


def test_a_node_holds_one_entry_per_cache():
    r = Recorder()
    old = r.use("old", A)
    shared = r.use("shared", B)
    new = r.use("new", A)  # A switches engines: its old one has no node left
    assert r.closed == [old]
    r.use("shared", A)
    assert r.closed == [old, new] and not shared.closed


def test_freeing_an_entry_in_use_waits_for_its_run():
    r = Recorder()
    entered, leave = Event(), Event()
    seen = []

    def run():
        with r.cache.use("engine", A, r.create("engine")) as value:
            seen.append(value)
            entered.set()
            assert leave.wait(10)

    worker = Thread(target=run)
    worker.start()
    assert entered.wait(10)
    r.cache.release_node(A)  # /clear-cache/individual mid-run
    assert r.closed == [] and len(r.cache) == 1
    leave.set()
    worker.join(10)
    assert r.closed == seen and len(r.cache) == 0


def test_a_node_that_references_the_entry_before_the_run_ends_keeps_it():
    # Use is exclusive (one stream, one context): B waits for A, then reuses it.
    r = Recorder()
    seen = []
    with r.cache.use("engine", A, r.create("engine")) as value:
        r.cache.release_node(A)
        waiter = Thread(target=lambda: seen.append(r.use("engine", B)))
        waiter.start()
        while B not in r.cache._entries["engine"].refs:
            waiter.join(0.01)
        assert not seen
    waiter.join(10)
    assert seen == [value] and r.closed == [] and len(r.cache) == 1


def test_an_error_closes_the_value_and_the_next_use_recreates_it():
    r = Recorder()
    with pytest.raises(RuntimeError), r.cache.use("engine", A, r.create("engine")):
        raise RuntimeError("aborted mid-tiling")
    assert len(r.closed) == 1 and len(r.cache) == 1  # the node still references it
    again = r.use("engine", A)
    assert len(r.created) == 2 and not again.closed


def test_a_changed_engine_file_reloads(tmp_path):
    # Load Engine's key: the file identity and the GPU.
    r = Recorder()
    path = tmp_path / "x.engine"
    path.write_bytes(b"one")
    first = r.use((file_identity(path), 0), A)
    assert r.use((file_identity(path), 0), A) is first
    path.write_bytes(b"three")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = r.use((file_identity(path), 0), A)
    assert second is not first and r.closed == [first] and len(r.cache) == 1
    assert r.use((file_identity(path), 1), A) is not second  # another GPU


def test_release_node_clears_every_cache():
    # The /clear-cache/individual hook: Load Engine and Upscale ids alike.
    engines, sessions = Recorder(), Recorder()
    engine = engines.use("file", A)
    session = sessions.use(("engine", 0), B)
    cache.release_node(A)
    assert engines.closed == [engine] and sessions.closed == []
    cache.release_node(B)
    assert sessions.closed == [session]
    assert len(engines.cache) == len(sessions.cache) == 0


class Handle:
    """A fake TensorRT object (runtime, engine, context)."""


class FakeSession:
    """TensorRTSession's engine handling: a context on the shared engine."""

    def __init__(self, trt, engine, gpu_index):
        self._loaded = engine.acquire(gpu_index, trt.deserialize)
        self.context = trt.handle("context")

    def close(self):
        del self.context
        self._loaded.release()


class FakeTrt:
    """Load Engine and Upscale Image over fakes; deserializations and frees are recorded."""

    def __init__(self):
        self.deserialized = 0
        self.freed: list[str] = []
        self.engines: SharedCache[TensorRTEngine] = SharedCache(TensorRTEngine.close)
        self.sessions: SharedCache[FakeSession] = SharedCache(FakeSession.close)

    def handle(self, name):
        value = Handle()
        weakref.finalize(value, self.freed.append, name)
        return value

    def deserialize(self, data, gpu_index):
        self.deserialized += 1
        label = data.decode()
        runtime = self.handle(f"runtime {label}")
        return DeserializedEngine(runtime, self.handle(f"engine {label}"), gpu_index)

    def load(self, path, node_id, gpu_index=0):
        # load_engine.py: keyed by the file identity and the GPU
        def create():
            data = path.read_bytes()
            info = TensorRTEngineInfo(
                "fp16", 3, 3, 4, "sm_89", "11", True, None, None, None
            )
            return TensorRTEngine(data, info, self.deserialize(data, gpu_index))

        key = (file_identity(path), gpu_index)
        with self.engines.use(key, node_id, create) as engine:
            return engine

    def upscale(self, engine, node_id, gpu_index=0):
        # inference.py: one session per engine and GPU
        def create():
            return FakeSession(self, engine, gpu_index)

        with self.sessions.use((engine, gpu_index), node_id, create) as session:
            return session


@pytest.fixture
def engine_file(tmp_path):
    path = tmp_path / "x.engine"
    path.write_bytes(b"one")
    return path


def test_two_loads_and_two_sessions_deserialize_once(engine_file):
    trt = FakeTrt()
    engine = trt.load(engine_file, A)
    assert trt.load(engine_file, B) is engine  # a second Load Engine node
    first = trt.upscale(engine, C)
    assert trt.upscale(engine, D) is first  # sessions are shared too
    cache.release_node(C)
    cache.release_node(D)
    assert trt.freed == ["context"]
    second = trt.upscale(engine, C)
    assert second is not first and trt.deserialized == 1


def test_a_session_recreated_after_a_clear_reuses_the_engine(engine_file):
    trt = FakeTrt()
    engine = trt.load(engine_file, A)
    first = trt.upscale(engine, C)
    trt.sessions.release_node(C)  # Clear on the Upscale node only
    assert trt.freed == ["context"]
    again = trt.upscale(trt.load(engine_file, A), C)  # the next run
    assert again is not first and again._loaded is engine.deserialized
    assert trt.deserialized == 1


@pytest.mark.parametrize("first_cleared", [A, C])
def test_contexts_are_freed_before_the_engine_and_the_engine_before_its_runtime(
    engine_file, first_cleared
):
    trt = FakeTrt()
    trt.upscale(trt.load(engine_file, A), C)
    cache.release_node(first_cleared)
    assert trt.freed == (["context"] if first_cleared == C else [])
    cache.release_node(C if first_cleared == A else A)
    assert trt.freed == ["context", "engine one", "runtime one"]
    assert len(trt.engines) == len(trt.sessions) == 0


def test_a_changed_engine_file_deserializes_the_new_one(engine_file):
    trt = FakeTrt()
    old = trt.load(engine_file, A)
    trt.upscale(old, C)
    engine_file.write_bytes(b"two")
    stat = engine_file.stat()
    os.utime(engine_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    new = trt.load(engine_file, A)
    assert new is not old and trt.deserialized == 2
    assert trt.freed == []  # the old session still uses the old engine
    trt.upscale(new, C)
    assert trt.freed == ["context", "engine one", "runtime one"]
    assert len(trt.engines) == len(trt.sessions) == 1


def test_a_session_whose_load_engine_node_was_cleared_deserializes(engine_file):
    trt = FakeTrt()
    engine = trt.load(engine_file, A)
    cache.release_node(A)  # cleared before the Upscale node ran
    assert trt.freed == ["engine one", "runtime one"]
    trt.upscale(engine, C)
    assert trt.deserialized == 2
