"""Bounded, owned scheduling for the retained node executor."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterator
from typing import Literal, Protocol, TypeVar, cast

from api import NodeId
from chain.chain import Chain
from chain.input import EdgeInput, InputMap

from .execution_cleanup import await_owned_node


class CacheView(Protocol):
    def has(self, node_id: NodeId) -> bool: ...


class DependencyExecutor(Protocol):
    @property
    def chain(self) -> Chain: ...

    @property
    def inputs(self) -> InputMap: ...

    @property
    def node_cache(self) -> CacheView: ...


T = TypeVar("T")


def advance_iterator(iterator: Iterator[T]) -> tuple[bool, T | None]:
    # StopIteration cannot cross an asyncio Future boundary.
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


async def owned_gather(coroutines: list[Awaitable[T]]) -> list[T]:
    """Drain every started branch before cleanup, including on cancellation.

    A lone branch is awaited directly, like an input that is not gathered: it has
    no sibling to drain or to order errors against. Draining its own node stays
    the branch's job (NodeFlights and await_owned_node).
    """
    if len(coroutines) == 1:
        return [await coroutines[0]]

    async def capture(
        coroutine: Awaitable[T],
    ) -> tuple[Literal[True], T] | tuple[Literal[False], BaseException]:
        try:
            return True, await coroutine
        except BaseException as error:
            return False, error

    tasks = [asyncio.create_task(capture(coroutine)) for coroutine in coroutines]
    pending = asyncio.gather(*tasks)
    results = await await_owned_node(pending)
    # Stable input order determines which error is reported, not worker timing.
    for result in results:
        if result[0] is False:
            try:
                raise result[1]
            finally:
                # Like await_owned_node's future: the error's traceback holds this
                # frame, so the frame drops every local that holds the error.
                del tasks, pending, results, result
    return [cast(T, value) for _, value in results]


def pure_cpu_dependency(
    executor: DependencyExecutor, node_id: NodeId, visiting: set[NodeId] | None = None
) -> bool:
    """Parallelize only eager, side-effect-free CPU subgraphs with ready roots."""
    if executor.node_cache.has(node_id):
        return True
    node = executor.chain.nodes[node_id]
    data = node.data
    if (
        data.kind != "regularNode"
        or data.side_effects
        or data.node_context
        or not data.schema_id.startswith(("chainner:image:", "chainner:utility:"))
        or any(item.lazy for item in data.inputs)
    ):
        return False
    visiting = set() if visiting is None else visiting
    if node_id in visiting:
        return False
    visiting.add(node_id)
    try:
        return all(
            not isinstance(value, EdgeInput)
            or pure_cpu_dependency(executor, value.id, visiting)
            for value in executor.inputs.get(node_id)
        )
    finally:
        visiting.remove(node_id)


class NodeFlights:
    """Share only in-flight work; completed values belong to the counted cache."""

    def __init__(self):
        self.pending: dict[str, asyncio.Future] = {}

    async def run(self, key: str, run: Callable[[], Awaitable[T]]) -> T:
        task = self.pending.get(key)
        if task is None:
            task = asyncio.ensure_future(run())
            self.pending[key] = task
        try:
            return await await_owned_node(task)
        finally:
            if task.done() and self.pending.get(key) is task:
                del self.pending[key]
            # A failed task holds its error, whose traceback holds this frame.
            del task


class LatestBroadcasts:
    """One active computation and one latest pending output per node."""

    def __init__(self):
        self.pending: OrderedDict[str, Callable[[], Awaitable[None]]] = OrderedDict()
        self.task: asyncio.Task[None] | None = None
        self.error: BaseException | None = None

    def submit(self, key: str, send: Callable[[], Awaitable[None]]) -> None:
        if self.error is not None:
            raise self.error
        # Replacing a queued preview releases the superseded frame immediately.
        self.pending[key] = send
        if self.task is None:
            self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            while self.pending:
                _, send = self.pending.popitem(last=False)
                await send()
        except BaseException as error:
            self.error = error
            self.pending.clear()
        finally:
            self.task = None

    async def flush(self) -> None:
        while self.task is not None:
            await await_owned_node(self.task)
        if self.error is not None:
            error, self.error = self.error, None
            raise error
