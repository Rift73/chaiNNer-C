"""TensorRT engines and sessions that outlive a chain run.

An entry is shared by every node that uses its key and holds one reference per
node id. A node holds at most one entry per cache: using another key releases
its previous one. `release_node` (the `/clear-cache/individual` hook) drops a
node's references; an entry left without any is closed at once when idle, else
when its last user returns. A cache built on another (sessions on Load Engine's
engines) keys its entries `(value, ...)` by the other's values: closing one of
those also releases every entry built on it, since no run can pass that value
again (its node makes a new one). One lock guards the tables; an entry's own lock
makes its use exclusive, for values that serve one user at a time. The wait for that
lock is not the waiting node's time (`util.held`).

This module must not import TensorRT: the server calls `release_node` whether
or not the TensorRT package is installed.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from threading import Condition, Lock, local
from typing import Any, Generic, TypeVar
from weakref import WeakSet, WeakValueDictionary

from sanic.log import logger

from api import NodeId
from util import held, waiting

from .model import TensorRTEngine

T = TypeVar("T")

# Every cache `release_node` clears; the session cache joins when its module loads.
_CACHES: WeakSet[SharedCache[Any]] = WeakSet()
_PREVIEWS: WeakValueDictionary[NodeId, Preview] = WeakValueDictionary()


class _PreviewLocal(local):
    current: Preview | None = None


_preview = _PreviewLocal()


def in_preview() -> bool:
    """Whether this thread runs the body of a node's preview."""
    return _preview.current is not None


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
        # A shared use: the users of the value, and that one of them gave it up.
        self.sharers = 0
        self.given_up = False
        self.closed = Condition(self.lock)


class SharedCache(Generic[T]):
    def __init__(
        self, close: Callable[[T], None], built_on: SharedCache[Any] | None = None
    ) -> None:
        """`built_on`: the cache whose values start this cache's keys, `(value, ...)`."""
        self._close = close
        self._lock = Lock()
        self._entries: dict[Hashable, _Entry[T]] = {}
        self._held: dict[NodeId, tuple[Hashable, Preview | None]] = {}
        self.built_on = built_on
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
        if value is None:
            return
        try:
            for shared in list(_CACHES):
                if shared.built_on is self:
                    shared.release_built_on(value)
        finally:
            self._close(value)

    def release_built_on(self, value: object) -> None:
        """Drop every reference to the entries keyed `(value, ...)`; each is closed at
        once when idle, else when its last user returns. Every one is closed even after
        an error; the first error is raised at the end."""
        with self._lock:
            for node_id, (key, _) in list(self._held.items()):
                if isinstance(key, tuple) and key[0] is value:
                    del self._held[node_id]
                    self._entries[key].refs.discard(node_id)
            doomed = [
                self._detach(key)
                for key in list(self._entries)
                if isinstance(key, tuple) and key[0] is value
            ]
        errors: list[Exception] = []
        for item in doomed:
            try:
                self._close_value(item)
            except Exception as e:
                errors.append(e)
        if errors:
            raise errors[0]

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
        self,
        key: Hashable,
        node_id: NodeId,
        create: Callable[[], T],
        *,
        exclusive: bool = True,
    ) -> Iterator[T]:
        """Reference the entry of `key` for `node_id` and use its value exclusively.

        The value is created on first use. An exception from the body closes the
        value (its state is unknown); the next use creates it again.

        exclusive=False, for a value that takes turns itself: users run together. One's
        exception gives the value up: the users already in it go on, the last of them
        closes it, and a new use waits for that to create the next value. So one value
        lives at a time, and a new one finds the memory of the last freed.
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
            if exclusive:
                with held(entry.lock):
                    if entry.value is None:
                        entry.value = create()
                    try:
                        yield entry.value
                    except BaseException:
                        value, entry.value = entry.value, None
                        self._close_value(value)
                        raise
            else:
                with self._shared_value(entry, create) as value:
                    yield value
        finally:
            with self._lock:
                entry.users -= 1
                doomed = self._detach(key)
            self._close_value(doomed)

    @contextmanager
    def _shared_value(self, entry: _Entry[T], create: Callable[[], T]) -> Iterator[T]:
        with held(entry.lock):
            if entry.given_up:
                with waiting():
                    while entry.given_up:
                        entry.closed.wait()
            if entry.value is None:
                entry.value = create()
            value = entry.value
            entry.sharers += 1
        try:
            yield value
        except BaseException:
            with entry.lock:
                entry.given_up = True
            raise
        finally:
            with entry.lock:
                entry.sharers -= 1
                last = entry.given_up and not entry.sharers
            if last:
                self._close_given_up(entry, value)

    def _close_given_up(self, entry: _Entry[T], value: T) -> None:
        """Close a value given up, for its last user, whose image may have succeeded:
        an error here is logged with its traceback, and the item keeps its own result."""
        try:
            self._close_value(value)
        except Exception:
            logger.exception("Closing a TensorRT value given up after an error failed")
        finally:
            with entry.lock:
                entry.value = None
                entry.given_up = False
                entry.closed.notify_all()


# Load Engine's deserialized engines. Closing one releases the sessions built on it
# (each keeps it alive, see `DeserializedEngine`), so the engine is freed after them.
engines: SharedCache[TensorRTEngine] = SharedCache(TensorRTEngine.close)


def release_node(node_id: NodeId) -> None:
    """Release a node's references in every cache (Load Engine, Upscale Image)."""
    preview = _PREVIEWS.pop(node_id, None)
    if preview is not None:
        preview.revoked = True
    for shared in list(_CACHES):
        shared.release_node(node_id)
