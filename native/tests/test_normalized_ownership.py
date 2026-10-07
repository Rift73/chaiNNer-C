"""Unknown read-only arrays remain isolated; only enforced copies may be reused."""

from __future__ import annotations

import gc
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Thread

import numpy as np
import pytest

from nodes.impl import native_buffers
from nodes.impl.image_utils import normalize
from nodes.properties.outputs.numpy_outputs import ImageOutput


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_preexisting_writable_alias_of_readonly_owner_cannot_change_output(channels):
    source = np.full((2, 3, channels), 0.5, np.float32)
    alias = source.view()
    source.flags.writeable = False
    assert alias.flags.writeable
    result = ImageOutput().enforce(source)
    before = result.tobytes()
    assert not np.shares_memory(result, source)
    alias[0, 0, 0] = 0.25
    assert result.tobytes() == before


@pytest.mark.parametrize("layers", [1, 2, 4])
def test_readonly_memoryview_does_not_hide_mutable_backing(layers):
    storage = bytearray(np.full((2, 3, 3), 0.5, np.float32).tobytes())
    view = memoryview(storage).toreadonly()
    for _ in range(layers):
        view = memoryview(view).toreadonly()
    source = np.frombuffer(view, np.float32).reshape(2, 3, 3)
    result = ImageOutput().enforce(source)
    before = result.tobytes()
    assert not np.shares_memory(source, result)
    storage[:4] = np.float32(0.25).tobytes()
    assert result.tobytes() == before


def test_owned_readonly_array_without_provenance_keeps_copy():
    source = np.full((2, 3, 3), 0.5, np.float32)
    source.flags.writeable = False
    result = ImageOutput().enforce(source)
    assert not np.shares_memory(source, result)
    source.flags.writeable = True
    source[:] = 0.25
    assert np.all(result == 0.5)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_repeated_default_enforcement_borrows_known_isolated_copy(channels):
    source = np.linspace(0, 1, 6 * channels, dtype=np.float32).reshape(2, 3, channels)
    first = ImageOutput().enforce(source)
    assert not np.shares_memory(first, source)
    expected = first.tobytes()
    for _ in range(5):
        result = ImageOutput().enforce(first)
        assert np.shares_memory(result, first)
        assert result.tobytes() == expected
        assert not result.flags.writeable
    source[:] = 0.17
    assert first.tobytes() == expected


def test_contiguous_view_of_registered_copy_can_borrow():
    first = ImageOutput().enforce(np.full((4, 5, 3), 0.5, np.float32))
    result = ImageOutput().enforce(first[1:3])
    assert np.shares_memory(first, result)
    assert not result.flags.writeable


@pytest.mark.parametrize("kind", ["bytes", "memoryview"])
def test_intrinsically_immutable_bytes_need_no_registry(kind):
    storage = np.full((2, 3, 3), 0.5, np.float32).tobytes()
    if kind == "memoryview":
        storage = memoryview(storage)
    source = np.frombuffer(storage, np.float32).reshape(2, 3, 3)
    result = ImageOutput().enforce(source)
    assert np.shares_memory(source, result)
    assert result.tobytes() == normalize(source).tobytes()


def test_assume_normalized_does_not_register_unowned_storage():
    source = np.full((2, 3, 3), 0.5, np.float32)
    alias = source.view()
    assumed = ImageOutput(assume_normalized=True).enforce(source)
    assert assumed is source
    result = ImageOutput().enforce(assumed)
    assert not np.shares_memory(result, assumed)
    alias[:] = 0.25
    assert np.all(result == 0.5)


def test_registered_owner_must_still_be_readonly_and_normalized():
    first = ImageOutput().enforce(np.full((2, 3, 3), 0.5, np.float32))
    first.flags.writeable = True
    assert native_buffers.normalized_readonly(first) is None
    first[0, 0] = -0.0
    first.flags.writeable = False
    assert native_buffers.normalized_readonly(first) is None
    result = ImageOutput().enforce(first)
    assert not np.shares_memory(result, first)
    assert result.tobytes() == normalize(first).tobytes()


def test_registry_is_weak_and_eviction_only_disables_borrowing(monkeypatch):
    monkeypatch.setattr(native_buffers, "_NORMALIZED_OWNER_LIMIT", 2)
    arrays = [
        ImageOutput().enforce(np.full((2, 3, 3), x / 4, np.float32)) for x in range(4)
    ]
    assert native_buffers.normalized_readonly(arrays[0]) is None
    assert native_buffers.normalized_readonly(arrays[-1]) is arrays[-1]
    reference = weakref.ref(arrays[-1])
    identity = id(arrays[-1])
    arrays.pop()
    gc.collect()
    assert reference() is None
    # The next locked registry call forgets the expired entry.
    assert native_buffers.normalized_readonly(arrays[-1]) is arrays[-1]
    assert identity not in native_buffers._normalized_owners  # lifetime assertion.


def test_concurrent_output_registration_and_borrowing():
    def run(index):
        source = np.full((3, 4, 3), index / 64, np.float32)
        first = ImageOutput().enforce(source)
        second = ImageOutput().enforce(first)
        assert np.shares_memory(first, second)
        assert first.tobytes() == second.tobytes()
        assert not np.shares_memory(source, first)
        return weakref.ref(first)

    with ThreadPoolExecutor(max_workers=8) as pool:
        references = list(pool.map(run, range(32)))
    gc.collect()
    assert all(reference() is None for reference in references)


def test_owner_callback_under_held_lock_does_not_deadlock(monkeypatch):
    # A private lock: if the callback blocks, only this test's daemon thread is stuck.
    # The stuck worker keeps image alive, so its callback never meets that lock again.
    lock = Lock()
    monkeypatch.setattr(native_buffers, "_normalized_owner_lock", lock)
    image = ImageOutput().enforce(np.full((2, 3, 3), 0.5, np.float32))

    def expire_while_locked():
        (reference,) = weakref.getweakrefs(image)
        with lock:
            reference.__callback__(reference)

    worker = Thread(target=expire_while_locked, daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), "the weakref callback blocked on the registry lock"
    assert native_buffers.normalized_readonly(image) is None


@pytest.mark.parametrize("locked_call", ["lookup", "register"])
def test_expired_owner_is_forgotten(locked_call, monkeypatch):
    owners = OrderedDict()
    monkeypatch.setattr(native_buffers, "_normalized_owners", owners)
    survivor = ImageOutput().enforce(np.full((2, 3, 3), 0.25, np.float32))
    image = ImageOutput().enforce(np.full((2, 3, 3), 0.5, np.float32))
    # Allocated while image lives, so it cannot reuse the expired entry's id.
    fresh = np.full((2, 3, 3), 0.75, np.float32)
    fresh.flags.writeable = False
    assert list(owners) == [id(survivor), id(image)]
    probe = weakref.ref(image)
    del image
    assert probe() is None
    if locked_call == "lookup":
        assert native_buffers.normalized_readonly(fresh) is None
        assert list(owners) == [id(survivor)]
    else:
        native_buffers.register_normalized_copy(fresh)
        assert list(owners) == [id(survivor), id(fresh)]
    assert native_buffers.normalized_readonly(survivor) is survivor


def test_reused_id_survives_a_stale_callback(monkeypatch):
    owners = OrderedDict()
    monkeypatch.setattr(native_buffers, "_normalized_owners", owners)
    image = ImageOutput().enforce(np.full((2, 3, 3), 0.5, np.float32))
    key = id(image)
    stale = owners[key]
    newer = weakref.ref(image)
    owners[key] = newer
    stale.__callback__(stale)
    assert native_buffers.normalized_readonly(image) is image
    assert owners[key] is newer
