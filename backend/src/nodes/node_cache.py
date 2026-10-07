from __future__ import annotations

import functools
import hashlib
import os
import tempfile
import time
from concurrent.futures import Future
from enum import Enum
from threading import RLock, get_ident
from typing import Iterable, NewType

import numpy as np
from sanic.log import logger

from api import RunFn

CACHE_MAX_BYTES = int(os.environ.get("CACHE_MAX_BYTES", str(1024**3)))  # default 1 GiB
CACHE_REGISTRY: list[NodeOutputCache] = []
_CACHE_LOCK = RLock()


class CachedNumpyArray:
    def __init__(self, arr: np.ndarray):
        self.file = tempfile.TemporaryFile()
        self.file.write(arr.tobytes())

        self.shape = arr.shape
        self.dtype = arr.dtype

    def value(self) -> np.ndarray:
        self.file.seek(0)
        return np.frombuffer(self.file.read(), dtype=self.dtype).reshape(self.shape)


CacheKey = NewType("CacheKey", tuple)


class NodeOutputCache:
    def __init__(self):
        self._data: dict[CacheKey, list] = {}
        self._bytes: dict[CacheKey, int] = {}
        self._access_time: dict[CacheKey, float] = {}

        with _CACHE_LOCK:
            CACHE_REGISTRY.append(self)

    @staticmethod
    def args_to_key(args: Iterable[object]) -> CacheKey:
        key = []
        for arg in args:
            if isinstance(arg, (int, float, bool, str, bytes)):
                key.append(arg)
            elif arg is None:
                key.append(None)
            elif isinstance(arg, Enum):
                key.append(arg.value)
            elif isinstance(arg, np.ndarray):
                key.append(tuple(arg.shape))
                key.append(arg.dtype.str)
                key.append(hashlib.sha256(arg.tobytes()).digest())
            elif hasattr(arg, "cache_key_func"):
                key.append(arg.__class__.__name__)
                key.append(arg.cache_key_func())  # type: ignore
            else:
                raise RuntimeError(f"Unexpected argument type {arg.__class__.__name__}")
        return CacheKey(tuple(key))

    @staticmethod
    def _estimate_bytes(output: list[object]) -> int:
        size = 0
        for out in output:
            if isinstance(out, np.ndarray):
                size += out.nbytes
            else:
                # any other type but numpy arrays is probably negligible, but here's an overestimate to handle
                # pathological cases where someone has a pipeline with a million math nodes
                size += 1024  # 1 KiB
        return size

    def empty(self) -> bool:
        with _CACHE_LOCK:
            return len(self._data) == 0

    def oldest(self) -> tuple[CacheKey, float]:
        with _CACHE_LOCK:
            return min(self._access_time.items(), key=lambda x: x[1])

    def size(self):
        with _CACHE_LOCK:
            return sum(self._bytes.values())

    @staticmethod
    def _enforce_limits():
        with _CACHE_LOCK:
            while True:
                total_bytes = sum([cache.size() for cache in CACHE_REGISTRY])
                logger.debug(
                    f"Cache size: {total_bytes} ({100 * total_bytes / CACHE_MAX_BYTES:0.1f}% of limit)"
                )
                if total_bytes <= CACHE_MAX_BYTES:
                    return
                logger.debug("Dropping oldest cache key")

                oldest_keys = [
                    (cache, cache.oldest())
                    for cache in CACHE_REGISTRY
                    if not cache.empty()
                ]

                cache, (key, _) = min(oldest_keys, key=lambda x: x[1][1])
                cache.drop(key)

    @staticmethod
    def _write_arrays_to_disk(output: list) -> list:
        return [
            CachedNumpyArray(item) if isinstance(item, np.ndarray) else item
            for item in output
        ]

    @staticmethod
    def _read_arrays_from_disk(output: list) -> list:
        return [
            item.value() if isinstance(item, CachedNumpyArray) else item
            for item in output
        ]

    @staticmethod
    def _output_to_list(output: object) -> list[object]:
        if isinstance(output, list):
            return output
        elif isinstance(output, tuple):
            return list(output)
        else:
            return [output]

    @staticmethod
    def _list_to_output(output: list[object]):
        if len(output) == 1:
            return output[0]
        return output

    @staticmethod
    def restore(stored: list) -> object:
        """The node output a stored snapshot holds; its arrays are read back fresh."""
        return NodeOutputCache._list_to_output(
            NodeOutputCache._read_arrays_from_disk(stored)
        )

    def get(self, args: Iterable[object]) -> object | None:
        return self.get_by_key(self.args_to_key(args))

    def put(self, args: Iterable[object], output: object):
        self.put_by_key(self.args_to_key(args), output)

    def drop(self, key: CacheKey):
        with _CACHE_LOCK:
            del self._data[key]
            del self._bytes[key]
            del self._access_time[key]

    def get_by_key(self, key: CacheKey) -> object | None:
        with _CACHE_LOCK:
            if key in self._data:
                logger.debug("Cache hit")
                self._access_time[key] = time.time()
                return self.restore(self._data[key])
            logger.debug("Cache miss")
            return None

    def put_by_key(self, key: CacheKey, output: object) -> list:
        values = self._output_to_list(output)
        # Each new snapshot owns separate temporary files, so serialization need
        # not hold the registry lock or block another node's computation.
        stored = self._write_arrays_to_disk(values)
        size = self._estimate_bytes(values)
        with _CACHE_LOCK:
            self._data[key] = stored
            self._bytes[key] = size
            self._access_time[key] = time.time()
            self._enforce_limits()
        # Waiters retain this snapshot even if the memory limit evicts its entry.
        return stored


def cached(run: RunFn):
    cache = NodeOutputCache()
    pending: dict[CacheKey, tuple[Future[list], int]] = {}

    @functools.wraps(run)
    def _run(*args: object):
        # Registered cached nodes do not mutate their inputs. Reuse the key
        # instead of copying and hashing every image again on a cache miss.
        key = cache.args_to_key(args)
        with _CACHE_LOCK:
            out = cache.get_by_key(key)
            if out is not None:
                return out
            active = pending.get(key)
            if active is None:
                future: Future[list] = Future()
                pending[key] = (future, get_ident())
                owner = True
            else:
                future, owner_thread = active
                if owner_thread == get_ident():
                    raise RuntimeError(
                        "Recursive evaluation of the same cached node inputs"
                    )
                owner = False

        if not owner:
            stored = future.result()
            # CachedNumpyArray uses seek/read on a shared temporary file. Keep
            # these reads atomic; each waiter receives its own immutable array.
            with _CACHE_LOCK:
                return cache.restore(stored)

        try:
            output = run(*args)
            stored = cache.put_by_key(key, output)
        except BaseException as error:
            with _CACHE_LOCK:
                future.set_exception(error)
                pending.pop(key)
            raise
        else:
            with _CACHE_LOCK:
                future.set_result(stored)
                pending.pop(key)
            return output

    return _run
