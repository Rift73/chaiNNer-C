from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Iterable, TypeVar

I = TypeVar("I")
L = TypeVar("L")


@dataclass
class Generator(Generic[I]):
    supplier: Callable[[], Iterable[I | Exception]]
    expected_length: int
    fail_fast: bool = True
    metadata: object | None = None
    # Files the iterator reads per item (read-ahead safety); None = unknown.
    source_paths: tuple[Path, ...] | None = None
    # The number of the first item in its source (Load Video's Range: In); the loop's
    # iteration views count from it.
    index_base: int = 0
    # The item's number when the generator runs exactly one chosen item (Load Video's
    # "Run this frame"), else None; nodes of the loop read it as NodeContext.chosen_item.
    chosen_item: int | None = None
    # Context frames: given the wanted (before, after), a fresh iterable over the items
    # index_base - b' .. index_base + expected_length + a' - 1 and the b', a' it yields,
    # clipped to the source. The loop calls it, instead of the supplier, when one of its
    # nodes declares a temporal context (NodeData.temporal_context).
    context: Callable[[int, int], tuple[Iterable[I | Exception], int, int]] | None = (
        None
    )
    # Exact lengths may be compared before combining generators. A video reader
    # with only a count estimate (or a Range upper bound) must instead reach EOF.
    length_is_exact: bool = True
    # The run's warning when the items end early, given how many came, or None for
    # none: Load Video's names its file and also compares with the frame count in the
    # file's header. Without it, an exact length warns at any shortfall.
    shortfall: Callable[[int], str | None] | None = None

    def shortfall_warning(self, name: str, read: int) -> str | None:
        """The warning the executors send once, at the end of the generator's items,
        when `read` items came (none when an item's deferred error explains the end):
        shortfall's, else, when `read` falls short of an exact length, one naming the
        node (`name`)."""
        if self.shortfall is not None:
            return self.shortfall(read)
        if self.length_is_exact and read < self.expected_length:
            return (
                f"{name} ended after {read} of {self.expected_length} items; the items"
                " read were processed."
            )
        return None

    def with_fail_fast(self, fail_fast: bool):
        self.fail_fast = fail_fast
        return self

    def with_metadata(self, metadata: object):
        self.metadata = metadata
        return self

    def with_range(
        self,
        index_base: int,
        chosen_item: int | None = None,
        context: Callable[[int, int], tuple[Iterable[I | Exception], int, int]]
        | None = None,
    ):
        """The generator reads items index_base .. index_base + expected_length - 1 of
        its source: their numbers, the chosen item of a one-item run, and the supplier
        of context frames around them (see the fields)."""
        self.index_base = index_base
        self.chosen_item = chosen_item
        self.context = context
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
R = TypeVar("R")


@dataclass
class Collector(Generic[N, R]):
    on_iterate: Callable[[N], None]
    on_complete: Callable[[], R]
