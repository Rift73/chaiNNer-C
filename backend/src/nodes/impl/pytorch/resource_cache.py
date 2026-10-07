"""One exclusively leased model bundle per node context, for one chain only.

No global model lifetime, file-content hashing, device transfer or model logic
is hidden here. Callers choose keys, construction, reset and retention policy.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, TypeVar
from weakref import WeakKeyDictionary

from sanic.log import logger

from api import NodeContext

T = TypeVar("T")


def file_identity(path: Path) -> tuple:
    """Cheap identity/version check; missing files still reach the loader."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (str(path.absolute()), None)
    return (
        str(path.absolute()),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


class _Entry:
    def __init__(self):
        self.lock = RLock()
        self.key: Hashable | None = None
        self.value: Any = None
        self.retain = False
        self.active = False

    def clear(self) -> None:
        with self.lock:
            self.value = None
            self.key = None
            self.retain = False


_ENTRIES: WeakKeyDictionary[NodeContext, _Entry] = WeakKeyDictionary()
_ENTRIES_LOCK = RLock()


@contextmanager
def lease_resource(
    context: NodeContext,
    key: Callable[[], Hashable],
    create: Callable[[], tuple[T, bool]],
    *,
    reuse: bool,
    reset: Callable[[T], None] | None = None,
) -> Iterator[T]:
    with _ENTRIES_LOCK:
        entry = _ENTRIES.get(context)
        if entry is None:
            entry = _Entry()
            _ENTRIES[context] = entry
    # Register on every invocation: the executor may rerun with the same context
    # after draining its cleanup set. The bound method compares/hash-deduplicates.
    context.add_cleanup(entry.clear, after="chain")
    with entry.lock:
        if entry.active:
            # Reentrant callbacks must not reset a helper currently in use.
            value, _ = create()
            try:
                yield value
            except BaseException:
                if reset is not None:
                    try:
                        reset(value)
                    except BaseException:
                        logger.exception(
                            "Reentrant model reset failed during error cleanup"
                        )
                raise
            else:
                if reset is not None:
                    reset(value)
            return
        entry.active = True
        try:
            current_key = key()
            if not reuse or entry.key != current_key:
                entry.clear()
            if entry.value is None:
                value, retainable = create()
                entry.value = value
                # Do not label a resource with a version that changed while it
                # was loading. A first download can therefore require one cold
                # reload; steady-state calls then reuse the stable bundle.
                entry.key = key()
                entry.retain = reuse and retainable and entry.key == current_key
            value = entry.value
            try:
                yield value
            except BaseException:
                entry.clear()
                if reset is not None:
                    try:
                        reset(value)
                    except BaseException:
                        logger.exception(
                            "Model resource reset failed during error cleanup"
                        )
                raise
            else:
                if reset is not None:
                    try:
                        reset(value)
                    except BaseException:
                        entry.clear()
                        raise
        finally:
            if not entry.retain:
                entry.clear()
            entry.active = False


def allow_model_reuse(
    device_type: str, budget_limit: int, force_cache_wipe: bool
) -> bool:
    # Retaining models between nodes must not defeat an explicit RAM/VRAM
    # budget. CUDA cache-wipe mode also opts out of resident GPU model reuse.
    return budget_limit == 0 and (device_type == "cpu" or not force_cache_wipe)
