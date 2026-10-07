"""Resource cleanup for the retained Python executor and native node owners."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from typing import TypeVar

from sanic.log import logger

T = TypeVar("T")


def run_cleanup_functions(
    functions: set[Callable[[], None]], *, preserve_error: bool = False
) -> None:
    # Remove ownership before invoking callbacks: reentry cannot run a callback
    # twice, and callbacks can safely register cleanup for a later phase.
    pending = tuple(functions)
    functions.clear()
    first_interrupt: BaseException | None = None
    for function in pending:
        try:
            function()
        except BaseException as error:
            logger.error("Error running cleanup function: %s", error)
            if not isinstance(error, Exception) and first_interrupt is None:
                first_interrupt = error
    if first_interrupt is not None and not preserve_error:
        raise first_interrupt


def run_cleanup_groups(
    groups: Iterable[set[Callable[[], None]]], *, preserve_error: bool = False
) -> None:
    first_interrupt: BaseException | None = None
    for functions in groups:
        try:
            run_cleanup_functions(functions, preserve_error=preserve_error)
        except BaseException as error:
            if first_interrupt is None:
                first_interrupt = error
    if first_interrupt is not None:
        raise first_interrupt


async def await_owned_node(future: asyncio.Future[T]) -> T:
    # Cancelling an asyncio wrapper cannot stop its running CPU worker. Wait for
    # that same operation to finish before releasing resources it can still use.
    # Further cancellation requests never restart work or create a second worker.
    # asyncio.wait never cancels what it waits for. asyncio.shield is not used: from
    # Python 3.13 on, it logs a failure that ends after its outer future was
    # cancelled, a second report whose record holds the error's traceback.
    try:
        # A finished future returns without a loop turn, as shield's fast path did.
        if not future.done():
            await asyncio.wait((future,))
        return future.result()
    except asyncio.CancelledError:
        while not future.done():
            try:
                await asyncio.wait((future,))
            except asyncio.CancelledError:
                continue
        if not future.cancelled():
            error = future.exception()
            if error is not None:
                logger.error("Node failed while cancellation was pending: %s", error)
            # Like future: its traceback holds this frame.
            del error
        raise
    finally:
        # A failed future holds its error, whose traceback holds this frame.
        del future
