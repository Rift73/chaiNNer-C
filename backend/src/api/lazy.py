from __future__ import annotations

import time
from asyncio import AbstractEventLoop, Task, get_running_loop
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from threading import Condition, get_ident
from typing import Any, Generic, TypeVar
from weakref import finalize

from nodes.impl import native_profile

T = TypeVar("T")


class _Result(Generic[T]):
    """Either an okay value of T or an error value."""

    def __init__(self, value: T | None, error: BaseException | None):
        self.value = value
        self.error = error

    def result(self) -> T:
        """Returns the value if it is okay, otherwise raises the error."""
        if self.error is not None:
            raise self.error
        return self.value  # type: ignore

    @property
    def is_ok(self) -> bool:
        """Returns True if the result is okay, otherwise False."""
        return self.error is None

    @staticmethod
    def ok(value: T) -> _Result[T]:
        return _Result(value, None)

    @staticmethod
    def err(error: BaseException) -> _Result[T]:
        return _Result(None, error)


def _to_result(fn: Callable[[], T]) -> Callable[[], _Result[T]]:
    def wrapper() -> _Result[T]:
        try:
            return _Result.ok(fn())
        except BaseException as e:
            # Cancellation and interrupts must wake every waiter too. They are
            # re-raised by result(), never converted into successful values.
            return _Result.err(e)

    return wrapper


class Lazy(Generic[T]):
    def __init__(self, factory: Callable[[], T]):
        self._factory = _to_result(factory)
        self._value: _Result[T] | None = None
        self._evaluating = False
        self._eval_time = 0
        self._condition = Condition()
        self._owner_thread: int | None = None
        self._event_loop: AbstractEventLoop | None = None

    @staticmethod
    def ready(value: T) -> Lazy[T]:
        lazy = Lazy(lambda: value)
        lazy._value = _Result.ok(value)
        return lazy

    @staticmethod
    def from_coroutine(
        coroutine: Coroutine[Any, Any, T], loop: AbstractEventLoop
    ) -> Lazy[T]:
        def supplier() -> T:
            if not loop.is_running():
                cleanup.detach()
                coroutine.close()
                raise RuntimeError(
                    "Cannot evaluate Lazy on an event loop that is not running"
                )
            completion: Future[_Result[T]] = Future()
            # asyncio only tracks Tasks weakly. Keep the submitted Task alive
            # in this waiting frame, even if its awaited Future is otherwise
            # owned only by the coroutine itself. The done callback deliberately
            # does not capture this holder, avoiding a Task/callback self-cycle.
            submitted: list[Task[T]] = []

            def complete(task: Task[T]) -> None:
                try:
                    result = _Result.ok(task.result())
                except BaseException as error:
                    # Preserve asyncio cancellation (including cancellation
                    # before the first step) across the concurrent Future.
                    result = _Result.err(error)
                completion.set_result(result)

            def submit() -> None:
                try:
                    task = loop.create_task(coroutine)
                except BaseException as error:
                    coroutine.close()
                    completion.set_result(_Result.err(error))
                else:
                    submitted.append(task)
                    task.add_done_callback(complete)

            try:
                loop.call_soon_threadsafe(submit)
            except BaseException:
                cleanup.detach()
                coroutine.close()
                raise
            # Submission transfers coroutine lifetime to its Task. In
            # particular, an interrupted synchronous reader must not close a
            # coroutine that is still owned by the event loop.
            cleanup.detach()
            try:
                if native_profile.enabled():
                    # From this blocking wait, nested in its reader's timed call;
                    # the submission above is outside it, so it is a lower bound
                    # of the hop and what the coroutine awaits (near zero if the
                    # loop finished the coroutine before the wait began).
                    outcome = native_profile.measure("lazy.resolve", completion.result)
                    return outcome.result()
                return completion.result().result()
            finally:
                submitted.clear()

        lazy = Lazy(supplier)
        lazy._event_loop = loop
        # Skipped lazy inputs must not run their coroutine. Close an unused
        # coroutine when its Lazy owner is discarded instead of leaking it.
        cleanup = finalize(lazy, coroutine.close)
        return lazy

    @property
    def has_value(self) -> bool:
        """Returns True if the value has been computed, otherwise False."""
        with self._condition:
            return self._value is not None and self._value.is_ok

    @property
    def has_error(self) -> bool:
        """Returns True if the value has been computed and it errored instead, otherwise False."""
        with self._condition:
            return self._value is not None and not self._value.is_ok

    @property
    def evaluation_time(self) -> float:
        """The time in seconds that it took to evaluate the value. If the value is not computed, returns 0."""
        with self._condition:
            return self._eval_time

    @property
    def value(self) -> T:
        with self._condition:
            if self._value is None and self._event_loop is not None:
                try:
                    running_loop = get_running_loop()
                except RuntimeError:
                    running_loop = None
                if running_loop is self._event_loop:
                    raise RuntimeError(
                        "Cannot synchronously evaluate Lazy from its event loop thread"
                    )

            while self._evaluating:
                if self._owner_thread == get_ident():
                    raise RuntimeError("Recursive Lazy evaluation")
                self._condition.wait()

            if self._value is not None:
                return self._value.result()
            self._evaluating = True
            self._owner_thread = get_ident()

        # Do not hold the condition while user code runs: independent Lazy
        # values may execute concurrently, and readers can inspect our state.
        result: _Result[T] | None = None
        start = time.time()
        try:
            result = self._factory()
        finally:
            with self._condition:
                if result is not None:
                    self._value = result
                    self._eval_time = time.time() - start
                self._evaluating = False
                self._owner_thread = None
                self._condition.notify_all()
        assert result is not None
        return result.result()
