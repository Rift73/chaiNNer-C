"""TensorRT engines and sessions that outlive a chain run.

An entry is shared by every node that uses its key and holds one reference per
node id. A node holds at most one entry per cache: using another key releases
its previous one. `release_node` (the `/clear-cache/individual` hook) drops a
node's references; an entry left without any is closed at once when idle, else
when its last user returns. One lock guards the tables; an entry's own lock
makes its use exclusive, since a session owns one stream and one context. The
wait for that lock is not the waiting node's time (`util.held`).

This module must not import TensorRT: the server calls `release_node` whether
or not the TensorRT package is installed.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from threading import Lock, local
from typing import Any, Generic, TypeVar
from weakref import WeakSet, WeakValueDictionary

from api import NodeId
from util import held

from .model import TensorRTEngine

T = TypeVar("T")

# Every cache `release_node` clears; the session cache joins when its module loads.
_CACHES: WeakSet[SharedCache[Any]] = WeakSet()
_PREVIEWS: WeakValueDictionary[NodeId, Preview] = WeakValueDictionary()


class _PreviewLocal(local):
    current: Preview | None = None


_preview = _PreviewLocal()


class Preview:
    """Cache ownership of one preview body, including a body still running after Stop."""

    def __init__(self, node_id: NodeId) -> None:
        self.node_id = node_id
        self.revoked = False
        previous = _PREVIEWS.get(node_id)
        if previous is not None:
            previous.revoked = True
        _PREVIEWS[node_id] = self

    def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        previous, _preview.current = _preview.current, self
        try:
            return fn(*args, **kwargs)
        finally:
            _preview.current = previous
            if self.revoked:
                for shared in list(_CACHES):
                    shared.release_node(self.node_id, self)


class _Entry(Generic[T]):
    def __init__(self) -> None:
        self.lock = Lock()
        self.value: T | None = None
        self.refs: set[NodeId] = set()
        self.users = 0


class SharedCache(Generic[T]):
    def __init__(self, close: Callable[[T], None]) -> None:
        self._close = close
        self._lock = Lock()
        self._entries: dict[Hashable, _Entry[T]] = {}
        self._held: dict[NodeId, tuple[Hashable, Preview | None]] = {}
        _CACHES.add(self)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _detach(self, key: Hashable) -> T | None:
        """Remove the entry of `key` if it has no references and no users (lock held)."""
        entry = self._entries.get(key)
        if entry is None or entry.refs or entry.users:
            return None
        del self._entries[key]
        return entry.value

    def _drop(self, node_id: NodeId) -> T | None:
        """Drop the reference of `node_id` (lock held); returns a value to close."""
        registration = self._held.pop(node_id, None)
        if registration is None:
            return None
        key, _ = registration
        self._entries[key].refs.discard(node_id)
        return self._detach(key)

    def _close_value(self, value: T | None) -> None:
        if value is not None:
            self._close(value)

    def release_node(self, node_id: NodeId, preview: Preview | None = None) -> None:
        """Release the reference of `node_id`; an unknown id is a no-op."""
        current = _preview.current
        if preview is None and current is not None and current.revoked:
            preview = current
        with self._lock:
            registration = self._held.get(node_id)
            if preview is not None and (
                registration is None or registration[1] is not preview
            ):
                return
            doomed = self._drop(node_id)
        self._close_value(doomed)

    @contextmanager
    def use(
        self, key: Hashable, node_id: NodeId, create: Callable[[], T]
    ) -> Iterator[T]:
        """Reference the entry of `key` for `node_id` and use its value exclusively.

        The value is created on first use. An exception from the body closes the
        value (its state is unknown); the next use creates it again.
        """
        preview = _preview.current
        if preview is not None and preview.node_id != node_id:
            preview = None
        with self._lock:
            retain = preview is None or not preview.revoked
            doomed = None
            if retain and self._held.get(node_id, (key, None))[0] != key:
                doomed = self._drop(node_id)
            entry = self._entries.get(key)
            if entry is None:
                entry = _Entry()
                self._entries[key] = entry
            if retain:
                entry.refs.add(node_id)
                self._held[node_id] = (key, preview)
            entry.users += 1
        try:
            self._close_value(doomed)
            with held(entry.lock):
                if entry.value is None:
                    entry.value = create()
                try:
                    yield entry.value
                except BaseException:
                    value, entry.value = entry.value, None
                    self._close_value(value)
                    raise
        finally:
            with self._lock:
                entry.users -= 1
                doomed = self._detach(key)
            self._close_value(doomed)


# Load Engine's deserialized engines; a session using one keeps it alive (see
# `DeserializedEngine`), so closing an entry frees the engine after its sessions.
engines: SharedCache[TensorRTEngine] = SharedCache(TensorRTEngine.close)


def release_node(node_id: NodeId) -> None:
    """Release a node's references in every cache (Load Engine, Upscale Image)."""
    preview = _PREVIEWS.pop(node_id, None)
    if preview is not None:
        preview.revoked = True
    for shared in list(_CACHES):
        shared.release_node(node_id)
