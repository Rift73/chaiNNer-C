from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Iterable, TypeVar

I = TypeVar("I")
L = TypeVar("L")
R = TypeVar("R")


@dataclass
class Generator(Generic[I]):
    supplier: Callable[[], Iterable[I | Exception]]
    expected_length: int
    fail_fast: bool = True
    metadata: object | None = None
    # Files the iterator reads per item (read-ahead safety); None = unknown.
    source_paths: tuple[Path, ...] | None = None

    def with_fail_fast(self, fail_fast: bool):
        self.fail_fast = fail_fast
        return self

    def with_metadata(self, metadata: object):
        self.metadata = metadata
        return self

    @staticmethod
    def from_iter(
        supplier: Callable[[], Iterable[I | Exception]], expected_length: int
    ) -> Generator[I]:
        return Generator(supplier, expected_length)

    @staticmethod
    def from_list(l: list[L], map_fn: Callable[[L, int], I]) -> Generator[I]:
        """
        Creates a new generator from a list that is mapped using the given
        function. The iterable will be equivalent to `map(map_fn, l)`.
        """

        def supplier():
            for i, x in enumerate(l):
                try:
                    yield map_fn(x, i)
                except Exception as e:
                    yield e

        return Generator(supplier, len(l))

    @staticmethod
    def from_list_read_ahead(
        l: list[L], read: Callable[[L], R], map_fn: Callable[[L, R, int], I]
    ) -> Generator[I]:
        """
        Like `from_list` with `map_fn(x, read(x), i)`, where the next element's `read`
        runs on one background thread while the current element is mapped and its
        item is used. For a read that takes no GPU memory (unlike the executor's
        read-ahead window, which runs whole items ahead). A failed read fails its own
        element's item.
        """

        def supplier():
            reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="read-ahead")
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

    @staticmethod
    def from_range(count: int, map_fn: Callable[[int], I]) -> Generator[I]:
        """
        Creates a new generator the given number of items where each item is
        lazily evaluated. The iterable will be equivalent to `map(map_fn, range(count))`.
        """
        assert count >= 0

        def supplier():
            for i in range(count):
                try:
                    yield map_fn(i)
                except Exception as e:
                    yield e

        return Generator(supplier, count)


N = TypeVar("N")


@dataclass
class Collector(Generic[N, R]):
    on_iterate: Callable[[N], None]
    on_complete: Callable[[], R]
