from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable, TypeVar

T = TypeVar("T")


class _Waited(threading.local):
    """The seconds this thread's calls spent in `waiting`."""

    seconds = 0.0


_waited = _Waited()


@contextmanager
def waiting() -> Iterator[None]:
    """A wait for other work, such as another item's turn with a shared session or a
    look-ahead at a later item: `timed_supplier` leaves it out of the node's time, as
    the executor leaves out its lazy inputs' evaluation."""
    start = time.perf_counter()
    try:
        yield
    finally:
        _waited.seconds += time.perf_counter() - start


def add_waiting(seconds: float) -> None:
    """Counts `seconds` that another thread spent `waiting` for this thread's call
    (on_device_thread) as this thread's."""
    _waited.seconds += seconds


def waited_seconds() -> float:
    """The seconds this thread has spent `waiting` so far: the run engine reads it
    around a node's body, as `timed_supplier` does."""
    return _waited.seconds


@contextmanager
def held(lock: threading.Lock) -> Iterator[None]:
    """Hold `lock`; the wait for it is `waiting`."""
    with waiting():
        lock.acquire()
    try:
        yield
    finally:
        lock.release()


def timed_supplier(supplier: Callable[[], T]) -> Callable[[], tuple[T, float]]:
    """The supplier's result and its time, less the time it spent `waiting`."""

    def wrapper():
        waited = _waited.seconds
        start = time.perf_counter()
        result = supplier()
        duration = time.perf_counter() - start
        return result, duration - (_waited.seconds - waited)

    return wrapper


T = TypeVar("T")


def combine_sets(set_list: list[set[T]]) -> list[set[T]]:
    """
    Combines sets in a list which have at least one intersecting value

    Example:
        in: [{1, 2}, {1, 4}, {3}, {3, 5}, {6}]
        out: [{1, 2, 4}, {3, 5}, {6}]
    Note:
        This code was written by ChatGPT. I tried to make my own algorithm for this, as well as
        find resources to help online, and was unsuccessful. From all my testing, this implementation
        seems to be both correct and performant. However, if you are familiar with this problem
        and you know a better way to do this, please submit a PR with a modification.
    """
    sets = [set(x) for x in set_list]
    combined = True
    while combined:
        combined = False
        new_sets = []
        # Process each set in the input list
        while sets:
            current = sets.pop()
            merged = False
            # Compare the current set with the remaining sets
            for i, s in enumerate(sets):
                if current & s:  # Check for intersection
                    sets[i] = current | s  # Union of sets with common elements
                    merged = True
                    combined = True  # Indicates that a merge occurred
                    break
            if not merged:
                new_sets.append(current)  # No merge, add current set to new_sets
        sets = new_sets  # Update sets with the remaining sets
    return sets
