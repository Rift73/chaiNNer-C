"""Bounded image-parameter plans; never retain source images or output buffers."""

from __future__ import annotations

import ctypes as ct
from collections import OrderedDict
from collections.abc import Callable, Hashable
from functools import lru_cache
from threading import Lock
from typing import Generic, TypeVar

from .native import lib

T = TypeVar("T")


class PreparedCache(Generic[T]):
    """Immutable plans with bounded retained payload and entry count.

    Construction runs outside the lock. Concurrent misses may build private
    plans; only one is retained. Eviction drops a reference, so an active caller
    keeps its plan alive. Oversized plans work normally without being retained.
    """

    def __init__(self, max_bytes: int, max_entries: int):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._items: OrderedDict[Hashable, tuple[T, int]] = OrderedDict()
        self._bytes = 0
        self._lock = Lock()

    def get(self, key: Hashable, build: Callable[[], tuple[T, int]]) -> T:
        with self._lock:
            found = self._items.get(key)
            if found is not None:
                self._items.move_to_end(key)
                return found[0]
        value, size = build()
        if size > self.max_bytes or self.max_entries == 0:
            return value
        with self._lock:
            found = self._items.get(key)
            if found is not None:
                self._items.move_to_end(key)
                return found[0]
            while self._items and (
                self._bytes + size > self.max_bytes
                or len(self._items) >= self.max_entries
            ):
                _, (_, removed) = self._items.popitem(last=False)
                self._bytes -= removed
            self._items[key] = (value, size)
            self._bytes += size
        return value

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def retained(self) -> tuple[int, int]:
        with self._lock:
            return len(self._items), self._bytes


@lru_cache(maxsize=1)
def _controls():
    function = lib().cn_image_fp_state
    function.argtypes = []
    function.restype = ct.c_uint64
    return function


def float_state() -> int:
    """Rounding/FTZ/DAZ settings, excluding arithmetic sticky status flags."""
    return _controls()()
