"""Load Models' read-ahead: the next element is read on one background thread while
the current one is mapped and its item used."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

from api import Generator

from . import numpy_pool

I = TypeVar("I")
L = TypeVar("L")
R = TypeVar("R")


def from_list_read_ahead(
    l: list[L], read: Callable[[L], R], map_fn: Callable[[L, R, int], I]
) -> Generator[I]:
    """Like `Generator.from_list` with `map_fn(x, read(x), i)`, where the next element's
    `read` runs on one background thread while the current element is mapped and its
    item is used. For a read that takes no GPU memory (unlike the executor's read-ahead
    window, which runs whole items ahead). A failed read fails its own element's item.
    """

    def supplier():
        # The reader runs node code, so it installs the NumPy pool handler.
        reader = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="read-ahead",
            initializer=numpy_pool.install,
        )
        try:
            ahead: Future[R] | None = None
            for i, x in enumerate(l):
                current = ahead if ahead is not None else reader.submit(read, x)
                ahead = reader.submit(read, l[i + 1]) if i + 1 < len(l) else None
                try:
                    yield map_fn(x, current.result(), i)
                except Exception as e:
                    yield e
        finally:
            # A stopped iteration does not wait for the read in progress.
            reader.shutdown(wait=False, cancel_futures=True)

    return Generator(supplier, len(l))
